//! BF16 Gemma RMSNorm with FP32 scale and accumulation, without cast buffers.
//! Packed Q/K views may have gaps between tokens; the result is contiguous.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/gemma_norm_ptx.rs"));
}

pub(crate) fn forward(x: &Tensor, scale: &Tensor, epsilon: f32) -> Result<Tensor> {
    apply(x, scale, epsilon, false)
}

/// Match Candle's FP32 sum/mean, square root and division arithmetic.
pub(crate) fn forward_reference(x: &Tensor, scale: &Tensor, epsilon: f32) -> Result<Tensor> {
    apply(x, scale, epsilon, true)
}

fn apply(x: &Tensor, scale: &Tensor, epsilon: f32, reference: bool) -> Result<Tensor> {
    let width = x.dim(candle::D::Minus1)?;
    if x.dtype() != DType::BF16
        || scale.dtype() != DType::F32
        || !x.device().is_cuda()
        || !scale.device().same_device(x.device())
        || !supported_layout(x.layout())
        || !scale.is_contiguous()
        || scale.dims() != [width]
        || width == 0
        || width > 8192
        || x.elem_count() / width > i32::MAX as usize
    {
        candle::bail!("Gemma RMSNorm requires row-contiguous CUDA BF16 activations and FP32 scale");
    }
    if x.elem_count() == 0 {
        return Ok(x.clone());
    }
    x.apply_op2_no_bwd(scale, &Norm { epsilon, reference })
}
// Packed Q/K views have contiguous heads within each token and a gap between tokens.
fn supported_layout(layout: &Layout) -> bool {
    if layout.is_contiguous() {
        return true;
    }
    let dims = layout.dims();
    let strides = layout.stride();
    (dims.len() == 2 && strides[1] == 1)
        || (dims.len() == 3 && strides[2] == 1 && strides[1] == dims[2])
}
struct Norm {
    epsilon: f32,
    reference: bool,
}
impl CustomOp2 for Norm {
    fn name(&self) -> &'static str {
        "gemma-rms-norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma RMSNorm requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        scale: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = x.device();
        let n = xl.shape().elem_count();
        let width = *xl.dims().last().unwrap() as u32;
        let heads = if xl.dims().len() == 3 {
            xl.dims()[1]
        } else {
            1
        } as u32;
        let token_stride = if xl.is_contiguous() {
            width as usize * heads as usize
        } else {
            xl.stride()[0]
        } as u64;
        let x = x.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let scale = scale
            .as_cuda_slice::<f32>()?
            .slice(sl.start_offset()..sl.start_offset() + width as usize);
        // The grid writes every output element exactly once.
        let mut out = unsafe { device.alloc::<half::bf16>(n)? };
        let rows = (n / width as usize) as u32;
        let warp = self.reference && width == 256 && rows >= 4096;
        let compact = self.reference && width > 256 && rows >= 2048;
        let kernel = device.get_or_load_custom_func(
            if warp {
                "gemma_rms_norm_warp_bf16"
            } else if compact {
                "gemma_rms_norm_compact_bf16"
            } else if self.reference {
                "gemma_rms_norm_reference_bf16"
            } else {
                "gemma_rms_norm_bf16"
            },
            "tei-gemma-rms-norm",
            ptx::GEMMA_RMS_NORM,
        )?;
        let mut launch = kernel.builder();
        launch
            .arg(&x)
            .arg(&scale)
            .arg(&mut out)
            .arg(&width)
            .arg(&heads)
            .arg(&token_stride)
            .arg(&self.epsilon);
        if warp {
            launch.arg(&rows);
        }
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (if warp { rows.div_ceil(8) } else { rows }, 1, 1),
                block_dim: (
                    if warp || compact {
                        256
                    } else if self.reference {
                        width.min(1024).next_power_of_two()
                    } else if width <= 128 {
                        32
                    } else {
                        256
                    },
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xl.shape().clone(),
        ))
    }
}

/// Share the RMS reduction for expert input and routing, preserving each
/// branch's original BF16 rounding and the router's root-size/learned scaling.
pub(crate) fn paired_moe(
    x: &Tensor,
    expert_scale: &Tensor,
    router_scale: &Tensor,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let (rows, width) = x.dims2()?;
    if width == 0
        || width > 8192
        || rows > i32::MAX as usize
        || x.dtype() != DType::BF16
        || expert_scale.dtype() != DType::F32
        || router_scale.dtype() != DType::BF16
        || !x.device().is_cuda()
        || !supported_layout(x.layout())
        || [expert_scale, router_scale].iter().any(|t| {
            !t.device().same_device(x.device()) || !t.is_contiguous() || t.dims() != [width]
        })
    {
        candle::bail!(
            "Paired Gemma RMS requires CUDA BF16 rows, FP32 expert and BF16 router scales"
        );
    }
    x.elem_count()
        .checked_mul(2)
        .ok_or_else(|| candle::Error::Msg("Paired RMS size overflow".into()))?;
    if rows == 0 {
        return Ok((x.clone(), x.clone()));
    }
    let out = x.apply_op3_no_bwd(expert_scale, router_scale, &PairedMoe { epsilon })?;
    Ok((out.get(0)?, out.get(1)?))
}
struct PairedMoe {
    epsilon: f32,
}
impl candle::CustomOp3 for PairedMoe {
    fn name(&self) -> &'static str {
        "gemma-paired-moe-rms"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Paired Gemma RMS requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        expert: &CudaStorage,
        el: &Layout,
        router: &CudaStorage,
        rl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = xl.shape().dims2()?;
        let device = x.device();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let es = expert
            .as_cuda_slice::<f32>()?
            .slice(el.start_offset()..el.start_offset() + width);
        let rs = router
            .as_cuda_slice::<half::bf16>()?
            .slice(rl.start_offset()..rl.start_offset() + width);
        let mut out = unsafe { device.alloc::<half::bf16>(2 * rows * width)? };
        let compact = width > 256 && rows >= 2048;
        let kernel = device.get_or_load_custom_func(
            if compact {
                "gemma_rms_norm_pair_compact_bf16"
            } else {
                "gemma_rms_norm_pair_bf16"
            },
            "tei-gemma-rms-norm",
            ptx::GEMMA_RMS_NORM,
        )?;
        let width32 = width as u32;
        let stride = xl.stride()[0] as u64;
        let root = half::bf16::from_f64((width as f64).sqrt().recip());
        let mut launch = kernel.builder();
        launch
            .arg(&xs)
            .arg(&es)
            .arg(&rs)
            .arg(&mut out)
            .arg(&width32)
            .arg(&stride)
            .arg(&self.epsilon)
            .arg(&root);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (
                    width32
                        .min(if compact { 256 } else { 1024 })
                        .next_power_of_two(),
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            (2, rows, width).into(),
        ))
    }
}

/// Gemma4's NeoX rotation, writing packed tokens without a transpose copy.
pub(crate) fn rotary_reference(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, tokens, width) = x.dims4()?;
    if batch != 1
        || heads == 0
        || width == 0
        || width % 2 != 0
        || cos.dims() != [1, 1, tokens, width]
        || sin.dims() != cos.dims()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || !x.device().is_cuda()
        || [x, cos, sin]
            .iter()
            .any(|t| t.dtype() != DType::BF16 || !t.device().same_device(x.device()))
    {
        candle::bail!(
            "Gemma rotary requires CUDA BF16 [1,H,T,D] and contiguous [1,1,T,D] frequencies"
        );
    }
    u32::try_from(x.elem_count()).map_err(candle::Error::wrap)?;
    if tokens == 0 {
        return x.transpose(1, 2)?.squeeze(0)?.contiguous();
    }
    x.apply_op3_no_bwd(cos, sin, &Rotary)
}
struct Rotary;
impl candle::CustomOp3 for Rotary {
    fn name(&self) -> &'static str {
        "gemma-precise-rotary"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma precise rotary requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (_, heads, tokens, width) = xl.shape().dims4()?;
        let device = x.device();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let cs = cos
            .as_cuda_slice::<half::bf16>()?
            .slice(cl.start_offset()..);
        let ss = sin
            .as_cuda_slice::<half::bf16>()?
            .slice(sl.start_offset()..);
        let count = u32::try_from(xl.shape().elem_count()).map_err(candle::Error::wrap)?;
        let heads32 = heads as u32;
        let width32 = width as u32;
        let head_stride = xl.stride()[1] as u64;
        let token_stride = xl.stride()[2] as u64;
        let col_stride = xl.stride()[3] as u64;
        let mut output = unsafe { device.alloc::<half::bf16>(count as usize)? };
        let kernel = device.get_or_load_custom_func(
            "gemma_rope_reference_bf16",
            "tei-gemma-precise-rope",
            ptx::GEMMA_RMS_NORM,
        )?;
        let mut launch = kernel.builder();
        launch
            .arg(&xs)
            .arg(&cs)
            .arg(&ss)
            .arg(&mut output)
            .arg(&count)
            .arg(&heads32)
            .arg(&width32)
            .arg(&head_stride)
            .arg(&token_stride)
            .arg(&col_stride);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (tokens, heads, width).into(),
        ))
    }
}

/// Preserve the normalization's BF16 store before adding the residual.
pub(crate) fn residual_reference(
    x: &Tensor,
    scale: &Tensor,
    residual: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let (rows, width) = x.dims2()?;
    if width == 0
        || width > 8192
        || rows > i32::MAX as usize
        || x.dtype() != DType::BF16
        || scale.dtype() != DType::F32
        || residual.dtype() != DType::BF16
        || !x.device().is_cuda()
        || x.dims() != residual.dims()
        || !supported_layout(x.layout())
        || !supported_layout(residual.layout())
        || !scale.is_contiguous()
        || scale.dims() != [width]
        || ![scale, residual]
            .iter()
            .all(|t| t.device().same_device(x.device()))
    {
        candle::bail!("Residual Gemma RMS requires matching CUDA BF16 rows and FP32 scale");
    }
    if rows == 0 {
        return Ok(x.clone());
    }
    x.apply_op3_no_bwd(scale, residual, &ResidualNorm { epsilon })
}
struct ResidualNorm {
    epsilon: f32,
}
impl candle::CustomOp3 for ResidualNorm {
    fn name(&self) -> &'static str {
        "gemma-residual-rms"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Residual Gemma RMS requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        scale: &CudaStorage,
        sl: &Layout,
        residual: &CudaStorage,
        rl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = xl.shape().dims2()?;
        let device = x.device();
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let ss = scale
            .as_cuda_slice::<f32>()?
            .slice(sl.start_offset()..sl.start_offset() + width);
        let rs = residual
            .as_cuda_slice::<half::bf16>()?
            .slice(rl.start_offset()..);
        let mut out = unsafe { device.alloc::<half::bf16>(rows * width)? };
        let compact = width > 256 && rows >= 2048;
        let kernel = device.get_or_load_custom_func(
            if compact {
                "gemma_rms_norm_residual_compact_bf16"
            } else {
                "gemma_rms_norm_residual_bf16"
            },
            "tei-gemma-rms-norm",
            ptx::GEMMA_RMS_NORM,
        )?;
        let width32 = width as u32;
        let stride = xl.stride()[0] as u64;
        let residual_stride = rl.stride()[0] as u64;
        let mut launch = kernel.builder();
        launch
            .arg(&xs)
            .arg(&ss)
            .arg(&rs)
            .arg(&mut out)
            .arg(&width32)
            .arg(&stride)
            .arg(&residual_stride)
            .arg(&self.epsilon);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (
                    if compact {
                        256
                    } else {
                        width32.min(1024).next_power_of_two()
                    },
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xl.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, D};
    #[test]
    #[ignore = "requires CUDA"]
    fn residual_matches_separate_candle_operations_bitwise() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for rows in [1, 17, 1152, 2047, 2048, 4507] {
            for width in [72, 2816] {
                let backing = Tensor::randn(0f32, 2f32, (rows + 2, width + 8), &device)?
                    .to_dtype(DType::BF16)?;
                let x = backing.narrow(0, 1, rows)?.narrow(1, 3, width)?;
                let residual = Tensor::randn(0f32, 2f32, (rows + 2, width + 16), &device)?
                    .to_dtype(DType::BF16)?
                    .narrow(0, 1, rows)?
                    .narrow(1, 5, width)?;
                let scale = Tensor::randn(1f32, 0.2f32, width + 2, &device)?.narrow(0, 1, width)?;
                let expected = (&residual + forward_reference(&x, &scale, 1e-6)?)?;
                let actual = residual_reference(&x, &scale, &residual, 1e-6)?;
                let bits = |t: &Tensor| -> Result<Vec<u16>> {
                    Ok(t.flatten_all()?
                        .to_vec1::<half::bf16>()?
                        .into_iter()
                        .map(|v| v.to_bits())
                        .collect())
                };
                assert_eq!(
                    bits(&actual)?,
                    bits(&expected)?,
                    "rows={rows},width={width}"
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn paired_moe_matches_separate_candle_operations_bitwise() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for rows in [1, 17, 1152, 2049] {
            for width in [72, 2816] {
                let packed = Tensor::randn(0f32, 2f32, (rows + 2, width + 8), &device)?
                    .to_dtype(DType::BF16)?;
                let x = packed.narrow(0, 1, rows)?.narrow(1, 3, width)?;
                let expert = Tensor::randn(1f32, 0.2f32, width, &device)?;
                let router = Tensor::randn(1f32, 0.2f32, width, &device)?.to_dtype(DType::BF16)?;
                let ones = Tensor::ones(width, DType::F32, &device)?;
                for epsilon in [1e-6, 1e-5] {
                    let expected_expert = forward_reference(&x, &expert, epsilon)?;
                    let expected_router = (forward_reference(&x, &ones, epsilon)?
                        * (width as f64).sqrt().recip())?
                    .broadcast_mul(&router)?;
                    let (a, b) = paired_moe(&x, &expert, &router, epsilon)?;
                    let bits = |t: &Tensor| -> Result<Vec<u16>> {
                        Ok(t.flatten_all()?
                            .to_vec1::<half::bf16>()?
                            .into_iter()
                            .map(|v| v.to_bits())
                            .collect())
                    };
                    assert_eq!(
                        bits(&a)?,
                        bits(&expected_expert)?,
                        "expert rows={rows},width={width}"
                    );
                    assert_eq!(
                        bits(&b)?,
                        bits(&expected_router)?,
                        "router rows={rows},width={width}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn rotary_matches_composed_bitwise() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for tokens in [1, 17, 1152, 4507] {
            for (heads, width) in [(8, 256), (16, 512)] {
                let backing =
                    Tensor::randn(0f32, 2f32, (1, tokens + 2, heads + 2, width), &device)?
                        .to_dtype(DType::BF16)?;
                let x = backing
                    .narrow(1, 1, tokens)?
                    .narrow(2, 1, heads)?
                    .transpose(1, 2)?;
                let freq = Tensor::randn(0f32, 1f32, (tokens + 2, width), &device)?;
                let cos = freq
                    .cos()?
                    .to_dtype(DType::BF16)?
                    .narrow(0, 1, tokens)?
                    .reshape((1, 1, tokens, width))?;
                let sin = freq
                    .sin()?
                    .to_dtype(DType::BF16)?
                    .narrow(0, 1, tokens)?
                    .reshape((1, 1, tokens, width))?;
                let first = x.narrow(D::Minus1, 0, width / 2)?;
                let second = x.narrow(D::Minus1, width / 2, width / 2)?;
                let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
                let expected = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?
                    .transpose(1, 2)?
                    .squeeze(0)?
                    .contiguous()?;
                let actual = rotary_reference(&x, &cos, &sin)?;
                assert!(
                    actual
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(expected
                            .flatten_all()?
                            .to_vec1::<half::bf16>()?
                            .iter()
                            .map(|v| v.to_bits())),
                    "rotary rounding/layout differs"
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn rotary_handles_empty_and_rejects_invalid_frequencies() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let empty = Tensor::zeros((1, 2, 0, 256), DType::BF16, &device)?;
        let frequencies = Tensor::zeros((1, 1, 0, 256), DType::BF16, &device)?;
        assert_eq!(
            rotary_reference(&empty, &frequencies, &frequencies)?.dims(),
            &[0, 2, 256]
        );
        let x = Tensor::zeros((1, 2, 3, 256), DType::BF16, &device)?;
        let wrong = Tensor::zeros((1, 1, 3, 128), DType::BF16, &device)?;
        assert!(rotary_reference(&x, &wrong, &wrong).is_err());
        let strided = Tensor::zeros((1, 1, 3, 512), DType::BF16, &device)?.narrow(3, 0, 256)?;
        assert!(rotary_reference(&x, &strided, &strided).is_err());
        let cpu = Tensor::zeros((1, 1, 3, 256), DType::BF16, &Device::Cpu)?;
        assert!(rotary_reference(&x, &cpu, &cpu).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn strided_matches_contiguous() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for tokens in [1, 7, 511] {
            for (heads, width) in [(1, 32), (3, 256), (4, 768)] {
                let packed = Tensor::randn(0f32, 2f32, (tokens, (heads + 2) * width), &device)?
                    .to_dtype(DType::BF16)?;
                let scale = Tensor::randn(1f32, 0.2f32, width, &device)?;
                let view = packed
                    .reshape((tokens, heads + 2, width))?
                    .narrow(1, 1, heads)?;
                let expected = forward(&view.contiguous()?, &scale, 1e-6)?;
                let actual = forward(&view, &scale, 1e-6)?;
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<half::bf16>()?,
                    expected.flatten_all()?.to_vec1::<half::bf16>()?
                );
                let view2 = packed.narrow(1, width, width)?;
                assert_eq!(
                    forward(&view2, &scale, 1e-6)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?,
                    forward(&view2.contiguous()?, &scale, 1e-6)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn reference_math_matches_unfused_bitwise() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for tokens in [1, 7, 511] {
            for width in [128, 256, 512, 1024, 2816] {
                let packed = Tensor::randn(0f32, 2f32, (tokens, 5, width), &device)?
                    .to_dtype(DType::BF16)?;
                let x = packed.narrow(1, 1, 3)?;
                for weighted in [false, true] {
                    let scale = if weighted {
                        Tensor::randn(1f32, 0.2f32, width, &device)?
                    } else {
                        Tensor::ones(width, DType::F32, &device)?
                    };
                    for epsilon in [1e-6, 1e-5] {
                        let xf = x.to_dtype(DType::F32)?;
                        let expected = xf
                            .broadcast_div(
                                &(xf.sqr()?.mean_keepdim(D::Minus1)? + epsilon)?.sqrt()?,
                            )?
                            .broadcast_mul(&scale)?
                            .to_dtype(DType::BF16)?;
                        let actual = forward_reference(&x, &scale, epsilon as f32)?;
                        assert_eq!(actual.flatten_all()?.to_vec1::<half::bf16>()?,
                            expected.flatten_all()?.to_vec1::<half::bf16>()?,
                            "tokens={tokens}, width={width}, weighted={weighted}, epsilon={epsilon}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA; validates and times the fused Gemma normalization"]
    fn reference_and_timing() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for width in [32, 256, 768, 1024] {
            for rows in [1, 128, 2048] {
                let x = Tensor::randn(0f32, 2f32, (rows, width), &device)?.to_dtype(DType::BF16)?;
                let scale = (Tensor::randn(0f32, 0.2f32, width, &device)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?
                    + 1.)?;
                let old = || {
                    candle_layer_norm::rms_norm(&x.to_dtype(DType::F32)?, &scale, None, 1e-6)?
                        .to_dtype(DType::BF16)
                };
                let actual = forward(&x, &scale, 1e-6)?.to_dtype(DType::F32)?;
                let expected = old()?.to_dtype(DType::F32)?;
                let error = (&actual - &expected)?
                    .abs()?
                    .broadcast_div(&(&expected.abs()? + 0.01)?)?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 0.008,
                    "rows={rows}, width={width}, relative error={error}"
                );
                // Also compare against independent unfused FP32 normalization.
                let xf = x.to_dtype(DType::F32)?;
                let reference = xf
                    .broadcast_div(&(xf.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?
                    .broadcast_mul(&scale)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?;
                let error = (&actual - reference)?
                    .abs()?
                    .broadcast_div(&(&actual.abs()? + 0.01)?)?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(error < 0.008, "independent relative error={error}");
                let mut times = Vec::new();
                for fused in [false, true] {
                    for _ in 0..10 {
                        let _ = if fused {
                            forward(&x, &scale, 1e-6)?
                        } else {
                            old()?
                        };
                    }
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..200 {
                        let _ = if fused {
                            forward(&x, &scale, 1e-6)?
                        } else {
                            old()?
                        };
                    }
                    device.synchronize()?;
                    times.push(start.elapsed().as_secs_f64() * 1e6 / 200.);
                }
                println!(
                    "rows={rows} width={width} old_us={:.2} fused_us={:.2} speedup={:.2}",
                    times[0],
                    times[1],
                    times[0] / times[1]
                );
            }
        }
        Ok(())
    }
}
