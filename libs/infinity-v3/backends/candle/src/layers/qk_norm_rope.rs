//! Fused Qwen3 head normalization and NeoX RoPE, with existing half rounding.
use super::RMSNorm;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use candle::cuda_backend::CudaDType;
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Device, Layout, Result, Shape, Storage, Tensor,
};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/qk_ptx.rs"));
}

pub(crate) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    qn: &RMSNorm,
    kn: &RMSNorm,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    if q.rank() != 3 || k.rank() != 3 {
        return Ok(None);
    }
    let (tokens, qheads, width) = q.dims3()?;
    let (ktokens, kheads, kwidth) = k.dims3()?;
    if width != 128
        || kwidth != 128
        || tokens != ktokens
        || tokens == 0
        || qheads == 0
        || kheads == 0
        || !matches!(q.device(), Device::Cuda(_))
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || qn.epsilon.to_bits() != kn.epsilon.to_bits()
        || tokens
            .checked_mul(qheads + kheads)
            .is_none_or(|n| n > i32::MAX as usize / 128)
        || qn.weight.dims() != [128]
        || kn.weight.dims() != [128]
        || cos.dims() != [tokens, 64]
        || sin.dims() != [tokens, 64]
    {
        return Ok(None);
    }
    // Q/K can be views into one token-major QKV projection.
    for tensor in [q, k] {
        let stride = tensor.stride();
        if stride[1] != 128
            || stride[2] != 1
            || !stride[0].is_multiple_of(8)
            || stride[0] > i32::MAX as usize
        {
            return Ok(None);
        }
    }
    for tensor in [q, k, &qn.weight, &kn.weight, cos, sin] {
        if tensor.dtype() != q.dtype()
            || !tensor.device().same_device(q.device())
            || !tensor.layout().start_offset().is_multiple_of(8)
        {
            return Ok(None);
        }
    }
    if [&qn.weight, &kn.weight, cos, sin]
        .iter()
        .any(|t| !t.is_contiguous())
    {
        return Ok(None);
    }
    let packed = q.apply_op2_no_bwd(
        k,
        &Fused {
            qw: qn.weight.clone(),
            kw: kn.weight.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            epsilon: qn.epsilon,
        },
    )?;
    let nq = q.elem_count();
    Ok(Some((
        packed.narrow(0, 0, nq)?.reshape(q.shape())?,
        packed.narrow(0, nq, k.elem_count())?.reshape(k.shape())?,
    )))
}

struct Fused {
    qw: Tensor,
    kw: Tensor,
    cos: Tensor,
    sin: Tensor,
    epsilon: f32,
}
impl Fused {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        name: &str,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, qheads, _) = ql.shape().dims3()?;
        let (_, kheads, _) = kl.shape().dims3()?;
        let device = q.device();
        let q = q.as_cuda_slice::<T>()?.slice(
            ql.start_offset()..ql.start_offset() + (tokens - 1) * ql.stride()[0] + qheads * 128,
        );
        let k = k.as_cuda_slice::<T>()?.slice(
            kl.start_offset()
                ..kl.start_offset() + (tokens - 1) * kl.stride()[0] + kl.dims()[1] * 128,
        );
        let (qw, qwl) = self.qw.storage_and_layout();
        let (kw, kwl) = self.kw.storage_and_layout();
        let (cos, cl) = self.cos.storage_and_layout();
        let (sin, sl) = self.sin.storage_and_layout();
        let (Storage::Cuda(qw), Storage::Cuda(kw), Storage::Cuda(cos), Storage::Cuda(sin)) =
            (&*qw, &*kw, &*cos, &*sin)
        else {
            candle::bail!("Q/K fusion requires CUDA tensors")
        };
        let qw = qw
            .as_cuda_slice::<T>()?
            .slice(qwl.start_offset()..qwl.start_offset() + 128);
        let kw = kw
            .as_cuda_slice::<T>()?
            .slice(kwl.start_offset()..kwl.start_offset() + 128);
        let cos = cos
            .as_cuda_slice::<T>()?
            .slice(cl.start_offset()..cl.start_offset() + tokens * 64);
        let sin = sin
            .as_cuda_slice::<T>()?
            .slice(sl.start_offset()..sl.start_offset() + tokens * 64);
        let rows = tokens * (qheads + kheads);
        // Every head writes 128 disjoint output elements.
        let mut output = unsafe { device.alloc::<T>(rows * 128)? };
        let function =
            device.get_or_load_custom_func(name, "tei-qk-norm-rope", ptx::QK_NORM_ROPE)?;
        let mut builder = function.builder();
        let qstride = ql.stride()[0] as i32;
        let kstride = kl.stride()[0] as i32;
        let tokens = tokens as i32;
        let qheads = qheads as i32;
        let kheads = kheads as i32;
        builder
            .arg(&q)
            .arg(&k)
            .arg(&qw)
            .arg(&kw)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&tokens)
            .arg(&qheads)
            .arg(&kheads)
            .arg(&qstride)
            .arg(&kstride)
            .arg(&self.epsilon);
        // Shape/dtype/alignment checks above bound every read/write. The launch builder
        // tracks all tensor buffers on their device stream, including the norm weights.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(8).min(4096) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (rows * 128,).into(),
        ))
    }
}
impl CustomOp2 for Fused {
    fn name(&self) -> &'static str {
        "qk-norm-rope"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Q/K fusion requires CUDA")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.launch::<half::f16>(q, ql, k, kl, "qk_norm_rope_f16"),
            DType::BF16 => self.launch::<half::bf16>(q, ql, k, kl, "qk_norm_rope_bf16"),
            dtype => candle::bail!("unsupported Q/K fusion dtype {dtype:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn tensor(shape: &[usize], seed: u32, dtype: DType, device: &Device) -> Result<Tensor> {
        let values: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| {
                let mut x = (i as u32).wrapping_add(seed);
                x ^= x >> 16;
                x = x.wrapping_mul(0x7feb352d);
                x ^= x >> 15;
                x = x.wrapping_mul(0x846ca68b);
                x ^= x >> 16;
                ((x % 10001) as i32 - 5000) as f32 / 1000.
            })
            .collect();
        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    #[test]
    fn fused_qk_matches_separate_norm_and_rope() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for tokens in [1, 7, 257] {
                let q = tensor(&[tokens, 3, 128], 91, dtype, &device)?;
                let k = tensor(&[tokens, 1, 128], 31, dtype, &device)?;
                let cos = tensor(&[tokens, 64], 81, dtype, &device)?;
                let sin = tensor(&[tokens, 64], 51, dtype, &device)?;
                let qweight = tensor(&[128], 73, dtype, &device)?;
                let kweight = tensor(&[128], 111, dtype, &device)?;
                for (epsilon, strided) in [(1e-6, false), (1e-5, false), (1e-6, true), (1e-5, true)]
                {
                    let (q, k) = if strided {
                        let qkv = Tensor::cat(&[&q, &k, &k], 1)?;
                        (qkv.narrow(1, 0, 3)?, qkv.narrow(1, 3, 1)?)
                    } else {
                        (q.clone(), k.clone())
                    };
                    let qnorm = RMSNorm::load(
                        VarBuilder::from_tensors(
                            HashMap::from([("weight".into(), qweight.clone())]),
                            dtype,
                            &device,
                        ),
                        128,
                        epsilon,
                    )?;
                    let knorm = RMSNorm::load(
                        VarBuilder::from_tensors(
                            HashMap::from([("weight".into(), kweight.clone())]),
                            dtype,
                            &device,
                        ),
                        128,
                        epsilon,
                    )?;
                    let (expected_q, _) = qnorm.forward(&q.contiguous()?, None)?;
                    let (expected_k, _) = knorm.forward(&k.contiguous()?, None)?;
                    candle_rotary::apply_rotary_inplace(
                        &expected_q,
                        &expected_k,
                        &cos,
                        &sin,
                        true,
                    )?;
                    let (actual_q, actual_k) = try_forward(&q, &k, &qnorm, &knorm, &cos, &sin)?
                        .expect("supported shape must fuse");
                    for (actual, expected) in [(actual_q, expected_q), (actual_k, expected_k)] {
                        let actual = actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let expected = expected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "dtype={dtype:?}, tokens={tokens}, epsilon={epsilon}, index={i}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn try_unfold_qkv(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ids: &Tensor,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let (compact, qheads, width) = q.dims3()?;
    let (ktokens, kheads, kwidth) = k.dims3()?;
    if !q.device().is_cuda()
        || compact == 0
        || qheads == 0
        || kheads == 0
        || width == 0
        || ids.rank() != 1
        || ids.elem_count() == 0
        || ktokens != compact
        || v.dims() != k.dims()
        || width != kwidth
        || !q.is_contiguous()
        || !k.is_contiguous()
        || v.stride()[2] != 1
        || v.stride()[1] != width
        || !v.stride()[0].is_multiple_of(8)
        || !width.is_multiple_of(8)
        || ids.dtype() != DType::U32
        || !ids.is_contiguous()
        || !matches!(q.dtype(), DType::BF16 | DType::F16)
        || [k, v]
            .iter()
            .any(|t| t.dtype() != q.dtype() || !t.device().same_device(q.device()))
        || !ids.device().same_device(q.device())
        || [q, k, v]
            .iter()
            .any(|t| !t.layout().start_offset().is_multiple_of(8))
    {
        return Ok(None);
    }
    let tokens = ids.elem_count();
    let packed = q.apply_op3_no_bwd(k, v, &UnfoldQkv { ids: ids.clone() })?;
    let nq = tokens * qheads * width;
    let nk = tokens * kheads * width;
    Ok(Some((
        packed.narrow(0, 0, nq)?.reshape((tokens, qheads, width))?,
        packed.narrow(0, nq, nk)?.reshape((tokens, kheads, width))?,
        packed
            .narrow(0, nq + nk, nk)?
            .reshape((tokens, kheads, width))?,
    )))
}
struct UnfoldQkv {
    ids: Tensor,
}
impl UnfoldQkv {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = q.device();
        let q = q
            .as_cuda_slice::<T>()?
            .slice(ql.start_offset()..ql.start_offset() + ql.shape().elem_count());
        let k = k
            .as_cuda_slice::<T>()?
            .slice(kl.start_offset()..kl.start_offset() + kl.shape().elem_count());
        let v = v.as_cuda_slice::<T>()?.slice(
            vl.start_offset()
                ..vl.start_offset()
                    + (vl.dims()[0] - 1) * vl.stride()[0]
                    + vl.dims()[1] * vl.dims()[2],
        );
        let (storage, layout) = self.ids.storage_and_layout();
        let Storage::Cuda(ids) = &*storage else {
            candle::bail!("Expected CUDA indices")
        };
        let ids = ids
            .as_cuda_slice::<u32>()?
            .slice(layout.start_offset()..layout.start_offset() + self.ids.elem_count());
        let tokens = u32::try_from(self.ids.elem_count()).map_err(candle::Error::wrap)?;
        let qw = u32::try_from(ql.dims()[1] * ql.dims()[2] / 8).map_err(candle::Error::wrap)?;
        let kw = u32::try_from(kl.dims()[1] * kl.dims()[2] / 8).map_err(candle::Error::wrap)?;
        let stride = (vl.stride()[0] / 8) as u64;
        let size = tokens as usize * (qw as usize + 2 * kw as usize) * 8;
        let mut output = unsafe { device.alloc::<T>(size)? };
        let function = device.get_or_load_custom_func(
            "qkv_unfold_u16",
            "tei-qk-norm-rope",
            ptx::QK_NORM_ROPE,
        )?;
        let mut builder = function.builder();
        builder
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&ids)
            .arg(&mut output)
            .arg(&tokens)
            .arg(&qw)
            .arg(&kw)
            .arg(&stride);
        // The layout checks guarantee aligned 16-byte accesses. IDs are the same
        // validated internal Radix map used by the existing gather path.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((size / 8).div_ceil(256).min(4096) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (size,).into(),
        ))
    }
}
impl candle::CustomOp3 for UnfoldQkv {
    fn name(&self) -> &'static str {
        "qkv-unfold"
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
        candle::bail!("QKV unfold requires CUDA")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::BF16 => self.launch::<half::bf16>(q, ql, k, kl, v, vl),
            DType::F16 => self.launch::<half::f16>(q, ql, k, kl, v, vl),
            _ => candle::bail!("Unsupported QKV unfold dtype"),
        }
    }
}

#[cfg(test)]
mod unfold_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn fused_unfold_preserves_bits_for_strided_values() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16] {
            for tokens in [1usize, 7, 511] {
                let values: Vec<f32> = (0..tokens * 4096)
                    .map(|i| ((i % 127) as f32 - 63.) / 16.)
                    .collect();
                let packed =
                    Tensor::from_vec(values, (tokens, 32, 128), &device)?.to_dtype(dtype)?;
                let q = packed.narrow(1, 0, 16)?.contiguous()?;
                let k = packed.narrow(1, 16, 8)?.contiguous()?;
                let v = packed.narrow(1, 24, 8)?;
                // Reverse, duplicate, and select both boundaries; include a nonzero ID offset.
                let ids = Tensor::new(
                    vec![0u32, (tokens - 1) as u32, 0, (tokens / 2) as u32, 0],
                    &device,
                )?
                .narrow(0, 1, 4)?;
                let (uq, uk, uv) = try_unfold_qkv(&q, &k, &v, &ids)?.unwrap();
                for (actual, input) in [(uq, q), (uk, k), (uv, v)] {
                    let expected = crate::layers::index_select(&input, &ids, 0)?;
                    assert_eq!(
                        actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        expected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                    );
                }
            }
        }
        Ok(())
    }
}
