mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::ptr;

fn layer_norm_internal_type(dtype: DType) -> Result<u32> {
    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    Ok(internal_type)
}

pub struct LayerNorm {
    pub epsilon: f32,
    pub is_rms_norm: bool,
    pub gamma: Tensor,
    pub beta: Option<Tensor>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

impl LayerNorm {
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: Option<&candle::CudaStorage>,
        r_l: Option<&Layout>,
        return_residual: bool,
        round_residual: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // Assume all tensors are on the same device and take device of x
        let dev = x.device();
        let stream = dev.cuda_stream();

        // Get internal layer norm type id for the given dtype
        let layer_norm_type = layer_norm_internal_type(x.dtype())?;

        // Make sure that gamma is a CUDA tensor and get the underlying storage
        let (g, g_l) = self.gamma.storage_and_layout();
        let g = match &*g {
            Storage::Cuda(g) => g,
            _ => candle::bail!("gamma must be a cuda tensor"),
        };

        // Get cuda slices for all tensors
        let x = x.as_cuda_slice::<T>()?;
        let g = g.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let x = x.slice(x_l.start_offset()..);
        let g = g.slice(g_l.start_offset()..);

        // Input matrix layout
        let rows = x_l.dims()[0];
        let cols = x_l.dims()[1];

        if !(cols.is_multiple_of(8) && cols <= 8192) {
            candle::bail!("hidden size must be % 8 and <= 8192")
        }

        let x_stride = x_l.stride();
        let g_stride = g_l.stride();

        let x_rank = x_stride.len();
        let g_rank = g_stride.len();

        if x_rank != 2 {
            candle::bail!("layer-norm expects input tensors of rank 2. Found: {x_rank}")
        }
        if x_stride[x_rank - 1] != 1 {
            candle::bail!("the last dim of x must be contiguous {x_stride:?}")
        }
        if g_stride[g_rank - 1] != 1 {
            candle::bail!("the last dim of g must be contiguous {g_stride:?}")
        }

        // Round cols to match with the correct kernel
        let cols_rounded = if cols <= 1536 {
            round_multiple(cols, 256)
        } else if cols <= 3072 {
            round_multiple(cols, 512)
        } else {
            round_multiple(cols, 1024)
        };

        let is_rms_norm = if self.is_rms_norm { 1 } else { 0 };

        // A second output is needed only when returning a residual sum.
        let return_residual = return_residual && r.is_some();
        let out_rows = if return_residual { rows * 2 } else { rows };
        let out_shape = Shape::from((out_rows, cols));

        let mut out = unsafe { dev.alloc::<T>(out_shape.elem_count()) }?;

        // If beta is et, get ids device pointer
        let beta_storage = self.beta.as_ref().map(Tensor::storage_and_layout);
        let b_view;
        let r_view;
        let mut guards = Vec::new();
        let b_ptr = if let Some((b, b_l)) = &beta_storage {
            // Make sure that beta is a CUDA tensor and get the underlying storage
            let b = match &**b {
                Storage::Cuda(b) => b,
                _ => candle::bail!("gamma must be a cuda tensor"),
            };

            let b = b.as_cuda_slice::<T>()?;
            b_view = b.slice(b_l.start_offset()..);
            let b = &b_view;

            let b_stride = b_l.stride();
            let b_rank = b_stride.len();

            if b_stride[b_rank - 1] != 1 {
                candle::bail!("the last dim of b must be contiguous {b_stride:?}")
            }
            ({
                let (ptr, guard) = b.device_ptr(&stream);
                guards.push(guard);
                ptr
            }) as *const core::ffi::c_void
        } else {
            ptr::null()
        };

        // If residual is set, get its device pointer
        let r_ptr = if let (Some(r), Some(r_l)) = (r, r_l) {
            // Check shape
            let expected_shape = x_l.shape().dims2()?;
            if r_l.shape().dims2()? != expected_shape {
                candle::bail!("shape mismatch x {:?} and r {:?}", x_l.shape(), r_l.shape());
            }

            let r = r.as_cuda_slice::<T>()?;
            r_view = r.slice(r_l.start_offset()..);
            let r = &r_view;

            let r_stride = r_l.stride();
            let r_rank = r_stride.len();

            if r_rank != 2 {
                candle::bail!("layer-norm expects input tensors of rank 2. Found: {r_rank}")
            }

            if r_stride[r_rank - 1] != 1 {
                candle::bail!("the last dim of r must be contiguous {r_stride:?}")
            }
            ({
                let (ptr, guard) = r.device_ptr(&stream);
                guards.push(guard);
                ptr
            }) as *const std::ffi::c_void
        } else {
            ptr::null()
        };

        // Get cuda device pointers from cuda slices
        let x_ptr = ({
            let (ptr, guard) = x.device_ptr(&stream);
            guards.push(guard);
            ptr
        }) as *const core::ffi::c_void;
        let g_ptr = ({
            let (ptr, guard) = g.device_ptr(&stream);
            guards.push(guard);
            ptr
        }) as *const core::ffi::c_void;
        let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
        guards.push(out_guard);
        let dst_ptr = out_ptr as *const core::ffi::c_void;
        let dst_add_ptr = if return_residual {
            (out_ptr as usize + rows * cols * std::mem::size_of::<T>()) as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };
        // Inference does not consume saved means or inverse standard deviations.
        // The kernel still computes these values internally for normalization.
        let mu_ptr = std::ptr::null();
        let rsigma_ptr = std::ptr::null();

        let multi_processors_count = dev
            .cuda_stream()
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap();

        unsafe {
            // Launch Kernel
            ffi::run_ln(
                x_ptr,
                r_ptr,
                g_ptr,
                b_ptr,
                dst_add_ptr,
                dst_ptr,
                mu_ptr,
                rsigma_ptr,
                self.epsilon,
                cols_rounded as u32,
                rows as u32,
                cols as u32,
                multi_processors_count,
                layer_norm_type,
                layer_norm_type,
                layer_norm_type,
                layer_norm_type,
                2,
                is_rms_norm,
                i32::from(round_residual),
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        }

        drop(guards);
        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "fused-layer-norm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, None, None, false, false),
            DType::BF16 => self.fwd::<bf16>(x, x_l, None, None, false, false),
            DType::F32 => self.fwd::<f32>(x, x_l, None, None, false, false),
            dt => {
                candle::bail!("fused-layer-norm is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

impl candle::CustomOp2 for LayerNorm {
    fn name(&self) -> &'static str {
        "fused-layer-norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: &candle::CudaStorage,
        r_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, Some(r), Some(r_l), true, false),
            DType::BF16 => self.fwd::<bf16>(x, x_l, Some(r), Some(r_l), true, false),
            DType::F32 => self.fwd::<f32>(x, x_l, Some(r), Some(r_l), true, false),
            dt => {
                candle::bail!("fused-layer-norm is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

// Keep the existing residual-returning operation available for pre-norm models.
struct NormalizedResidual {
    norm: LayerNorm,
    return_residual: bool,
    round_residual: bool,
}

impl candle::CustomOp2 for NormalizedResidual {
    fn name(&self) -> &'static str {
        "fused-layer-norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: &candle::CudaStorage,
        r_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.norm.fwd::<f16>(
                x,
                x_l,
                Some(r),
                Some(r_l),
                self.return_residual,
                self.round_residual,
            ),
            DType::BF16 => self.norm.fwd::<bf16>(
                x,
                x_l,
                Some(r),
                Some(r_l),
                self.return_residual,
                self.round_residual,
            ),
            DType::F32 => self.norm.fwd::<f32>(
                x,
                x_l,
                Some(r),
                Some(r_l),
                self.return_residual,
                self.round_residual,
            ),
            dt => {
                candle::bail!("fused-layer-norm is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

/// Normalize `x + res` without materializing the unused residual sum.
/// Uses the same FP32 residual-sum statistics as `fused_add_layer_norm`.
pub fn layer_norm_with_residual(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<Tensor> {
    let op = NormalizedResidual {
        norm: LayerNorm {
            epsilon,
            gamma: gamma.clone(),
            beta: beta.cloned(),
            is_rms_norm: false,
        },
        return_residual: false,
        round_residual: false,
    };
    x.apply_op2_no_bwd(res, &op)
}

/// Layer Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensor has the same dimensions as `x`
pub fn layer_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
    };
    let results = x.apply_op1_no_bwd(&op)?;
    let rows = x.dims()[0];
    results.narrow(0, 0, rows)
}

/// Fused Add Layer Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `res` - Residual tensor of rank 2. Will be added to `x` before normalization. Must have
///   the same shape as `x`.
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensors have the same dimensions as `x`
/// First tensor is the result of the normalization, second is the result of the residual add
pub fn fused_add_layer_norm(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
    };
    let results = x.apply_op2_no_bwd(res, &op)?;
    let rows = x.dims()[0];
    Ok((results.narrow(0, 0, rows)?, results.narrow(0, rows, rows)?))
}

/// Layer RMS Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensor has the same dimensions as `x`
pub fn rms_norm(x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>, epsilon: f32) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: true,
    };
    let results = x.apply_op1_no_bwd(&op)?;
    let rows = x.dims()[0];
    results.narrow(0, 0, rows)
}

/// Fused Add RMS Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `res` - Residual tensor of rank 2. Will be added to `x` before normalization. Must have
///   the same shape as `x`.
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensors have the same dimensions as `x`
/// First tensor is the result of the normalization, second is the result of the residual add
pub fn fused_add_rms_norm(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: true,
    };
    let results = x.apply_op2_no_bwd(res, &op)?;
    let rows = x.dims()[0];
    Ok((results.narrow(0, 0, rows)?, results.narrow(0, rows, rows)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn layer_norm_truth(
        x: &Tensor,
        gamma: &Tensor,
        beta: Option<&Tensor>,
        epsilon: f64,
        rms: bool,
    ) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let (_seq_len, hidden_size) = x.shape().dims2()?;
        let x = x.to_dtype(internal_dtype)?;

        let x = if !rms {
            let mean_x = (x.sum_keepdim(1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };

        let norm_x = (x.sqr()?.sum_keepdim(1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + epsilon)?.sqrt()?)?;

        let mut x = x_normed.to_dtype(x_dtype)?.broadcast_mul(gamma)?;
        if let Some(beta) = beta {
            x = x.broadcast_add(beta)?;
        }
        Ok(x)
    }

    fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
            .collect();
        Ok(t)
    }

    #[test]
    fn bf16_fused_rms_norm_matches_reference() -> Result<()> {
        let device = Device::Cuda(candle::CudaDevice::new_with_stream(0)?);
        let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.) / 32.).collect();
        let x = Tensor::from_vec(values, (4, 32), &device)?.to_dtype(DType::BF16)?;
        let residual = Tensor::full(0.25f32, (4, 32), &device)?.to_dtype(DType::BF16)?;
        let gamma = Tensor::ones(32, DType::BF16, &device)?;
        let (actual, added) = fused_add_rms_norm(&x, &residual, &gamma, None, 1e-5)?;
        let expected_added = (&x + &residual)?;
        let expected = layer_norm_truth(&expected_added, &gamma, None, 1e-5, true)?;
        assert_eq!(
            added.flatten_all()?.to_vec1::<bf16>()?,
            expected_added.flatten_all()?.to_vec1::<bf16>()?
        );
        let error = actual
            .to_dtype(DType::F32)?
            .sub(&expected.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error <= 0.016, "BF16 RMSNorm error: {error}");
        Ok(())
    }

    #[test]
    fn normalization_only_matches_residual_returning_operation() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for rows in [1, 7, 128] {
                for cols in [8, 768, 1024, 4096] {
                    let values: Vec<f32> = (0..rows * cols)
                        .map(|i| ((i * 37 % 1001) as f32 - 500.) / 128.)
                        .collect();
                    let x = Tensor::from_vec(values, (rows, cols), &device)?.to_dtype(dtype)?;
                    let residual =
                        Tensor::full(0.125f32, (rows, cols), &device)?.to_dtype(dtype)?;
                    let gamma = Tensor::full(0.75f32, cols, &device)?.to_dtype(dtype)?;
                    let bias = Tensor::full(-0.1f32, cols, &device)?.to_dtype(dtype)?;
                    for beta in [None, Some(&bias)] {
                        let (expected, _) =
                            fused_add_layer_norm(&x, &residual, &gamma, beta, 1e-5)?;
                        let actual = layer_norm_with_residual(&x, &residual, &gamma, beta, 1e-5)?;
                        assert_eq!(actual.shape(), x.shape());
                        let bits = |t: Tensor| -> Result<Vec<u32>> {
                            Ok(t.to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect())
                        };
                        assert_eq!(bits(actual)?, bits(expected)?, "{dtype:?} {rows}x{cols}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = layer_norm(&x, &g, Some(&b), 1e-12)?;
        let truth = layer_norm_truth(&x, &g, Some(&b), 1e-12, false)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_layer_norm_no_bias() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = layer_norm(&x, &g, None, 1e-12)?;
        let truth = layer_norm_truth(&x, &g, None, 1e-12, false)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = rms_norm(&x, &g, Some(&b), 1e-12)?;
        let truth = layer_norm_truth(&x, &g, Some(&b), 1e-12, true)?;
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_no_bias() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = rms_norm(&x, &g, None, 1e-12)?;
        let truth = layer_norm_truth(&x, &g, None, 1e-12, true)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_layer_norm_add() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let r = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let (res, res_add) = fused_add_layer_norm(&x, &r, &g, Some(&b), 1e-12)?;
        let truth_add = (x + r)?;
        let truth = layer_norm_truth(&truth_add, &g, Some(&b), 1e-12, false)?;
        assert_eq!(to_vec2_round(res_add, 3)?, to_vec2_round(truth_add, 3)?);
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_add() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let r = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let (res, res_add) = fused_add_rms_norm(&x, &r, &g, Some(&b), 1e-12)?;
        let truth_add = (x + r)?;
        let truth = layer_norm_truth(&truth_add, &g, Some(&b), 1e-12, true)?;
        assert_eq!(to_vec2_round(res_add, 3)?, to_vec2_round(truth_add, 3)?);
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }
}

/// Normalize a rounded residual sum without allocating a saved residual output.
pub fn layer_norm_with_rounded_residual(
    x: &Tensor,
    residual: &Tensor,
    gamma: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let op = NormalizedResidual {
        norm: LayerNorm {
            epsilon,
            gamma: gamma.clone(),
            beta: None,
            is_rms_norm: false,
        },
        return_residual: false,
        round_residual: true,
    };
    x.apply_op2_no_bwd(residual, &op)
}

/// Normalize a model-dtype rounded residual sum and return that sum alongside it.
pub fn fused_add_layer_norm_rounded(
    x: &Tensor,
    residual: &Tensor,
    gamma: &Tensor,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let op = NormalizedResidual {
        norm: LayerNorm {
            epsilon,
            gamma: gamma.clone(),
            beta: None,
            is_rms_norm: false,
        },
        return_residual: true,
        round_residual: true,
    };
    let rows = x.dim(0)?;
    let out = x.apply_op2_no_bwd(residual, &op)?;
    Ok((out.narrow(0, 0, rows)?, out.narrow(0, rows, rows)?))
}

#[cfg(test)]
mod rounded_residual_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn rounded_residual_matches_separate_add_and_norm() -> Result<()> {
        let device = candle::Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for width in [768, 1024] {
                for rows in [1, 13] {
                    let x: Vec<f32> = (0..rows * width)
                        .map(|i| ((i * 17 % 997) as f32 - 498.) / 137.)
                        .collect();
                    let r: Vec<f32> = (0..rows * width)
                        .map(|i| ((i * 43 % 991) as f32 - 495.) / 311.)
                        .collect();
                    let x = Tensor::from_vec(x, (rows, width), &device)?.to_dtype(dtype)?;
                    let r = Tensor::from_vec(r, (rows, width), &device)?.to_dtype(dtype)?;
                    let gamma = Tensor::from_vec(
                        (0..width)
                            .map(|i| 0.5 + (i % 19) as f32 / 16.)
                            .collect::<Vec<_>>(),
                        width,
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let expected_sum = (&x + &r)?;
                    let expected = layer_norm(&expected_sum, &gamma, None, 1e-5)?;
                    let (actual, sum) = fused_add_layer_norm_rounded(&x, &r, &gamma, 1e-5)?;
                    let norm_only = layer_norm_with_rounded_residual(&x, &r, &gamma, 1e-5)?;
                    let allocated = {
                        let (storage, _) = norm_only.storage_and_layout();
                        let Storage::Cuda(storage) = &*storage else {
                            candle::bail!("expected CUDA storage")
                        };
                        match dtype {
                            DType::F16 => storage.as_cuda_slice::<f16>()?.len(),
                            DType::BF16 => storage.as_cuda_slice::<bf16>()?.len(),
                            _ => unreachable!(),
                        }
                    };
                    assert_eq!(allocated, rows * width);
                    for (a, b) in [
                        (actual, expected.clone()),
                        (norm_only, expected),
                        (sum, expected_sum),
                    ] {
                        assert_eq!(
                            a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                            b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                            "{dtype:?} rows={rows} width={width}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
