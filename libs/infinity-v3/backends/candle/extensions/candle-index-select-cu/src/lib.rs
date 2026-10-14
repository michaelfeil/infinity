// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright (c) 2025 Michael Feil
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Authors explaination: Provide a copy of the first two lines in each redistributed version.

mod ffi;

pub use candle;
use candle::backend::BackendStorage;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};

#[cfg(feature = "cuda-12")]
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;

fn index_select_internal_type(dtype: DType) -> Result<u32> {
    let code = match dtype {
        DType::F16 | DType::BF16 => 0,
        DType::F32 => 1,
        dt => candle::bail!("candle-index-select only supports f16, bf16 and f32 (got {dt:?})"),
    };
    Ok(code)
}

/// Fast CUDA index_select along dim 0 for N-D inputs with contiguous rows (rank >= 2).
/// We flatten all trailing dimensions into a single `cols = dims[1..].product()`.
pub struct IndexSelect {
    pub dim: usize,
}

/// Validate input layouts and return (rows, cols_flat, index_count, out_shape).
///
/// Requirements for fast path:
/// - x: rank >= 2, contiguous trailing dimensions, dim == 0
/// - ids: rank 1, contiguous
fn validate_inputs(
    x_l: &Layout,
    ids_l: &Layout,
    dim: usize,
) -> Result<(usize, usize, usize, Shape)> {
    let x_dims = x_l.dims();
    let x_stride = x_l.stride();
    let ids_stride = ids_l.stride();
    let x_rank = x_dims.len();
    let ids_rank = ids_stride.len();

    if x_rank < 2 {
        candle::bail!("candle-index-select expects input tensors of rank >= 2. Found: {x_rank}");
    }
    if ids_rank != 1 {
        candle::bail!("candle-index-select expects index tensor of rank 1. Found: {ids_rank}");
    }
    if dim != 0 {
        candle::bail!(
            "candle-index-select only supports dim == 0 for now (got {})",
            dim
        );
    }

    // Indices must be contiguous 1D
    if ids_stride[0] != 1 {
        candle::bail!("indices tensor must be contiguous for candle-index-select ({ids_stride:?})");
    }

    // QKV views may have gaps between rows, but each selected row must be contiguous.
    let mut expected = 1usize;
    for (&d, &s) in x_dims[1..].iter().rev().zip(x_stride[1..].iter().rev()) {
        if d != 1 && s != expected {
            candle::bail!(
                "x rows must be contiguous for candle-index-select (dims={x_dims:?}, stride={x_stride:?})"
            );
        }
        expected *= d;
    }

    let rows = x_dims[0];
    let cols: usize = x_dims[1..].iter().product();
    let index_count = ids_l.dims()[0];

    // Output shape: replace dim 0 with index_count, keep trailing dims.
    let mut out_dims = x_dims.to_vec();
    out_dims[0] = index_count;
    let out_shape = Shape::from(out_dims);

    Ok((rows, cols, index_count, out_shape))
}

// =============================================================================
// cudarc 0.16+ API (candle 0.9+)
// =============================================================================
#[cfg(feature = "cuda-12")]
mod cuda_impl {
    use super::*;
    use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

    #[allow(clippy::too_many_arguments)]
    pub fn fwd_impl<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
        rows: usize,
        cols: usize,
        index_count: usize,
        out_shape: Shape,
        dtype_code: u32,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x.device();

        let x = x.as_cuda_slice::<T>()?;
        let ids = ids.as_cuda_slice::<u32>()?;

        let x = x.slice(x_l.start_offset()..);
        let ids = ids.slice(ids_l.start_offset()..);

        let mut out = unsafe { dev.alloc::<T>(out_shape.elem_count()) }?;

        let stream = dev.cuda_stream();
        let stream_ref = stream.as_ref();

        let multi_processors_count = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .expect("failed to query CUDA multiprocessor count");

        {
            let (x_devptr, _x_sync) = x.device_ptr(stream_ref);
            let x_ptr = x_devptr as usize as *const core::ffi::c_void;

            let (ids_devptr, _ids_sync) = ids.device_ptr(stream_ref);
            let ids_ptr = ids_devptr as usize as *const u32;

            let (out_devptr, _out_sync) = out.device_ptr_mut(stream_ref);
            let dst_ptr = out_devptr as usize as *mut core::ffi::c_void;

            unsafe {
                ffi::run_index_select(
                    x_ptr,
                    ids_ptr,
                    dst_ptr,
                    rows as u32,
                    cols as u32,
                    x_l.stride()[0] as u64,
                    index_count as u32,
                    multi_processors_count,
                    dtype_code,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl IndexSelect {
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, cols, index_count, out_shape) = validate_inputs(x_l, ids_l, self.dim)?;
        let dtype_code = index_select_internal_type(x.dtype())?;
        cuda_impl::fwd_impl::<T>(
            x,
            x_l,
            ids,
            ids_l,
            rows,
            cols,
            index_count,
            out_shape,
            dtype_code,
        )
    }
}

impl candle::CustomOp2 for IndexSelect {
    fn name(&self) -> &'static str {
        "candle-index-select"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for candle-index-select (CUDA only)")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, ids, ids_l),
            DType::BF16 => self.fwd::<bf16>(x, x_l, ids, ids_l),
            DType::F32 => self.fwd::<f32>(x, x_l, ids, ids_l),
            dt => candle::bail!("candle-index-select only supports f16, bf16 and f32 (got {dt:?})"),
        }
    }
}

/// Public API:
/// Fast path + fallback to `Tensor::index_select`.
///
/// * Fast path:
///   - Device: CUDA
///   - x: rank >= 2, contiguous trailing dimensions
///   - indices: rank-1, contiguous, DType::U32
///   - dim == 0
///   - dtype(x) ∈ {F16, BF16, F32}
///
/// * Fallback:
///   - everything else → `x.index_select(indices, dim)`
pub fn index_select(x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
    use candle::{DType, Device};

    let devices_match = match (x.device(), indices.device()) {
        (Device::Cpu, Device::Cpu) => true,
        (Device::Cuda(a), Device::Cuda(b)) => a.id() == b.id(),
        _ => false,
    };
    if !devices_match {
        return x.index_select(indices, dim);
    }

    let device = x.device();

    if !matches!(device, Device::Cuda(_)) {
        return x.contiguous()?.index_select(indices, dim);
    }

    if !matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32) {
        return x.index_select(indices, dim);
    }

    if indices.dtype() != DType::U32 {
        return x.index_select(indices, dim);
    }

    let x_shape = x.shape();
    let ids_shape = indices.shape();

    // Fast path: dim == 0, x rank >= 2, indices rank == 1
    if dim != 0 {
        return x.index_select(indices, dim);
    }
    if x_shape.rank() < 2 || ids_shape.rank() != 1 {
        return x.index_select(indices, dim);
    }

    let (x_sto, x_l) = x.storage_and_layout();
    let (ids_sto, ids_l) = indices.storage_and_layout();

    match (&*x_sto, &*ids_sto) {
        (Storage::Cuda(_), Storage::Cuda(_)) => {
            let xs = x_l.stride();
            let is = ids_l.stride();
            // We’ll let validate_inputs enforce contiguous rows, but we still
            // require the last dims to be contiguous here as a quick filter.
            if xs[xs.len() - 1] != 1 || is[is.len() - 1] != 1 {
                return x.index_select(indices, dim);
            }
        }
        _ => return x.index_select(indices, dim),
    }

    let op = IndexSelect { dim };
    x.apply_op2_no_bwd(indices, &op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|row| row.iter().map(|v| (v * b).round() / b).collect())
            .collect();
        Ok(t)
    }

    #[test]
    fn test_bf16_bitwise_copy_with_offsets_and_duplicate_rows() -> Result<()> {
        let device = Device::Cuda(candle::CudaDevice::new_with_stream(0)?);
        // Include signed zero, subnormal, finite and NaN bit patterns. A gather
        // must copy BF16 bits, never convert through FP16 arithmetic.
        for cols in [7, 8, 1024] {
            let values: Vec<bf16> = (0..6 * cols)
                .map(|i| bf16::from_bits([0x8000, 0x0001, 0x3f80, 0x7fc1, 0xbf80][i % 5]))
                .collect();
            let x = Tensor::from_vec(values.clone(), (6, cols), &device)?.narrow(0, 1, 4)?;
            let ids = Tensor::new(&[99u32, 3, 0, 3, 2], &device)?.narrow(0, 1, 4)?;
            let actual = index_select(&x, &ids, 0)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let expected: Vec<_> = [4usize, 1, 4, 3]
                .into_iter()
                .flat_map(|row| {
                    values[row * cols..(row + 1) * cols]
                        .iter()
                        .map(|x| x.to_bits())
                })
                .collect();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_strided_rows_match_contiguous() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let ids = Tensor::new(&[3u32, 0, 3, 2], &device)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for width in [7, 8, 128] {
                // Odd row strides and nonzero offsets also exercise scalar fallback.
                for gap in [1, 4] {
                    let packed = Tensor::arange(0f32, (6 * (3 * width + gap)) as f32, &device)?
                        .reshape((6, 3 * width + gap))?
                        .to_dtype(dtype)?;
                    let view = packed.narrow(0, 1, 4)?.narrow(1, width, width)?;
                    for view in [view.clone(), view.unsqueeze(1)?] {
                        let actual = index_select(&view, &ids, 0)?;
                        let expected = view.contiguous()?.index_select(&ids, 0)?;
                        assert_eq!(actual.dims(), expected.dims());
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
        }
        Ok(())
    }

    #[test]
    fn test_index_select_matches_candle_f32() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = 128;
        let cols = 1024;

        let x = Tensor::randn(0., 1., (rows, cols), &device)?.to_dtype(DType::F32)?;
        let raw_idx: Vec<u32> = (0..rows as u32).flat_map(|i| [i, i]).collect();
        let index_count = raw_idx.len();
        let indices = Tensor::from_vec(raw_idx, index_count, &device)?.to_dtype(DType::U32)?;

        let y_fast = index_select(&x, &indices, 0)?;
        let y_ref = x.index_select(&indices, 0)?;

        assert_eq!(to_vec2_round(&y_fast, 3)?, to_vec2_round(&y_ref, 3)?);
        Ok(())
    }

    #[test]
    fn test_index_select_matches_candle_f16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = 128;
        let cols = 1024;

        let x = Tensor::randn(0., 1., (rows, cols), &device)?.to_dtype(DType::F16)?;
        let raw_idx: Vec<u32> = (0..rows as u32).collect();
        let indices = Tensor::from_vec(raw_idx, rows, &device)?.to_dtype(DType::U32)?;

        let y_fast = index_select(&x, &indices, 0)?;
        let y_ref = x
            .to_dtype(DType::F32)?
            .index_select(&indices, 0)?
            .to_dtype(DType::F16)?;

        assert_eq!(
            to_vec2_round(&y_fast.to_dtype(DType::F32)?, 3)?,
            to_vec2_round(&y_ref.to_dtype(DType::F32)?, 3)?
        );
        Ok(())
    }
}
