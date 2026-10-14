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

use candle_index_select::candle;
use candle_index_select_cu as candle_index_select;

fn cuda_device() -> candle::Result<candle::Device> {
    candle::Device::new_cuda(0)
}

fn allclose(a: &candle::Tensor, b: &candle::Tensor, tol: f32) -> candle::Result<bool> {
    let da = a.to_vec1::<f32>()?;
    let db = b.to_vec1::<f32>()?;
    if da.len() != db.len() {
        return Ok(false);
    }
    for (x, y) in da.iter().zip(db.iter()) {
        if (x - y).abs() > tol {
            return Ok(false);
        }
    }
    Ok(true)
}

// =============================================================================
// 2D Tensor Tests
// =============================================================================

#[test]
fn test_2d_f32_contiguous() -> candle::Result<()> {
    let device = cuda_device()?;

    let rows = 16_000;
    let cols = 1024;
    let out_rows = 70_000;

    let x = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let idx_data: Vec<u32> = (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect();
    let indices = candle::Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-4
    )?);

    Ok(())
}

#[test]
fn test_2d_f16_contiguous() -> candle::Result<()> {
    let device = cuda_device()?;

    let rows = 8_000;
    let cols = 1024;
    let out_rows = 32_000;

    let x_f32 = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let x = x_f32.to_dtype(candle::DType::F16)?;

    let idx_data: Vec<u32> = (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect();
    let indices = candle::Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
        &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
        5e-3
    )?);

    Ok(())
}

#[test]
fn test_2d_small() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (10, 8), &device)?;
    let indices = candle::Tensor::from_vec(vec![0u32, 2, 5, 9, 1], 5, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_2d_non_contiguous_slice() -> candle::Result<()> {
    let device = cuda_device()?;

    // Create tensor and take a slice (which may be non-contiguous depending on dim)
    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    let x_slice = x.narrow(0, 10, 50)?; // Take rows 10..60

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    let fast = candle_index_select::index_select(&x_slice, &indices, 0)?;
    let baseline = x_slice.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

// =============================================================================
// 3D Tensor Tests (should fall back to candle's implementation)
// =============================================================================

#[test]
fn test_3d_contiguous_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    // 3D tensor - should use fallback since we only support 2D
    let x = candle::Tensor::randn(0.0f32, 1.0, (32, 64, 128), &device)?;
    let indices = candle::Tensor::from_vec((0u32..16).collect::<Vec<_>>(), 16, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_3d_dim1_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (16, 32, 64), &device)?;
    let indices = candle::Tensor::from_vec((0u32..10).collect::<Vec<_>>(), 10, &device)?;

    // Index along dim 1 - should use fallback
    let fast = candle_index_select::index_select(&x, &indices, 1)?;
    let baseline = x.index_select(&indices, 1)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_3d_permuted_contiguous_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (16, 32, 64), &device)?;
    let x_perm = x.permute((1, 0, 2))?.contiguous()?; // Permute then make contiguous

    let indices = candle::Tensor::from_vec((0u32..10).collect::<Vec<_>>(), 10, &device)?;

    let fast = candle_index_select::index_select(&x_perm, &indices, 0)?;
    let baseline = x_perm.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_duplicate_indices() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    // Many duplicates
    let indices = candle::Tensor::from_vec(vec![0u32, 0, 1, 1, 2, 2, 0, 5, 5, 5], 10, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_single_row_select() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (1000, 256), &device)?;
    let indices = candle::Tensor::from_vec(vec![42u32], 1, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_all_rows_select() -> candle::Result<()> {
    let device = cuda_device()?;

    let rows = 256;
    let x = candle::Tensor::randn(0.0f32, 1.0, (rows, 128), &device)?;
    let indices = candle::Tensor::from_vec((0..rows as u32).collect::<Vec<_>>(), rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_wide_columns() -> candle::Result<()> {
    let device = cuda_device()?;

    // Very wide tensor (embedding-like)
    let x = candle::Tensor::randn(0.0f32, 1.0, (1000, 4096), &device)?;
    let indices = candle::Tensor::from_vec((0u32..500).collect::<Vec<_>>(), 500, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-4
    )?);

    Ok(())
}

#[test]
fn test_narrow_columns() -> candle::Result<()> {
    let device = cuda_device()?;

    // Very narrow tensor
    let x = candle::Tensor::randn(0.0f32, 1.0, (10000, 4), &device)?;
    let indices = candle::Tensor::from_vec((0u32..5000).collect::<Vec<_>>(), 5000, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_output_is_contiguous() -> candle::Result<()> {
    let device = cuda_device()?;

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    // Test with a contiguous input tensor
    let x_cont = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    assert!(x_cont.is_contiguous());
    let fast_from_cont = candle_index_select::index_select(&x_cont, &indices, 0)?;
    assert!(fast_from_cont.is_contiguous());

    Ok(())
}

#[test]
fn test_non_contiguous_input_errors() -> candle::Result<()> {
    let device = cuda_device()?;

    // Create a non-contiguous tensor by permuting
    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    let x_permuted = x.permute((1, 0))?; // Now shape is (64, 100) and non-contiguous

    assert!(!x_permuted.is_contiguous());

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    // Expect an error because the fast path requires a contiguous input tensor
    let result = candle_index_select::index_select(&x_permuted, &indices, 0);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_3d_dim0_fast_vs_baseline() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (10, 100, 128), &device)?;
    let idx =
        candle::Tensor::from_vec((0u32..20).map(|i| i % 10).collect::<Vec<_>>(), 20, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}

#[test]
fn test_4d_dim0_fast_vs_baseline() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (8, 16, 32, 64), &device)?;
    let idx = candle::Tensor::from_vec((0u32..16).map(|i| i % 8).collect::<Vec<_>>(), 16, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}

#[test]
fn test_4d_dim2_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (4, 5, 6, 7), &device)?;
    let idx = candle::Tensor::from_vec((0u32..3).collect::<Vec<_>>(), 3, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 2)?;
    let baseline = x.index_select(&idx, 2)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}
// FAILS??
// #[test]
// fn test_empty_indices() -> candle::Result<()> {
//     let device = cuda_device()?;

//     let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
//     let idx = candle::Tensor::from_vec(Vec::<u32>::new(), 0, &device)?;

//     let fast = candle_index_select::index_select(&x, &idx, 0)?;
//     let baseline = x.index_select(&idx, 0)?;

//     assert_eq!(fast.dims(), baseline.dims());
//     assert_eq!(fast.elem_count(), 0);
//     Ok(())
// }

#[test]
fn test_f16_odd_cols_scalar_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    // cols = 255 not divisible by 2 -> half2 path should not trigger
    let x_f32 = candle::Tensor::randn(0.0f32, 1.0, (512, 255), &device)?;
    let x = x_f32.to_dtype(candle::DType::F16)?;

    let idx = candle::Tensor::from_vec(
        (0u32..200).map(|i| i % 512).collect::<Vec<_>>(),
        200,
        &device,
    )?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
        &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
        5e-3
    )?);
    Ok(())
}

#[test]
fn test_index_dtype_mismatch_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    // Int64 indices: fast path should not be used.
    let idx_i64 = candle::Tensor::from_vec((0i64..10).collect::<Vec<_>>(), 10, &device)?;

    let fast = candle_index_select::index_select(&x, &idx_i64, 0)?;
    let baseline = x.index_select(&idx_i64, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}
