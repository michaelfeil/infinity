// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared, allocation-free matrix-operation descriptors.
//!
//! Preparation checks geometry and converts dimensions once. Binding only
//! supplies addresses; it does not enqueue work, retain owners, or synchronize.

use crate::tensor::{Element, Layout, ScalarFormat, TensorError, TensorMut, TensorRef};
use cuda_core::DeviceBuffer;
use std::{fmt, marker::PhantomData, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathError {
    ShapeMismatch,
    EmptyMatrix,
    UnsupportedLayout,
    UnsupportedFormat,
    DimensionOverflow,
    ContextMismatch,
    Tensor(TensorError),
}
impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Self::Tensor(error) = self {
            return error.fmt(f);
        }
        f.write_str(match self {
            Self::ShapeMismatch => "linear input, weight and output shapes disagree",
            Self::EmptyMatrix => "linear matrices must have nonzero dimensions",
            Self::UnsupportedLayout => "linear requires row-major matrices with unit inner stride",
            Self::UnsupportedFormat => "linear format has no implemented cuBLAS method",
            Self::DimensionOverflow => "linear dimensions exceed the cuBLAS integer range",
            Self::ContextMismatch => "linear buffers belong to different CUDA contexts",
            Self::Tensor(_) => "invalid linear tensor storage",
        })
    }
}
impl std::error::Error for MathError {}

/// Arguments for Y = X W^T, expressed in cuBLAS's column-major view.
/// Plain data for the ABI boundary; these addresses do not convey ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmArguments {
    pub x: u64,
    pub weight: u64,
    pub output: u64,
    pub rows: i32,
    pub input: i32,
    pub width: i32,
    pub x_stride: i32,
    pub weight_stride: i32,
    pub output_stride: i32,
    pub cuda_type: i32,
}

/// One concrete method, selected before dispatch, with FP32 accumulation.
/// Does not claim support for FP8/FP4 merely because a storage format exists.
#[derive(Debug)]
pub struct LinearPlan<E: Element> {
    x: Layout<2>,
    weight: Layout<2>,
    output: Layout<2>,
    args: GemmArguments,
    _element: PhantomData<E>,
}
impl<E: Element> LinearPlan<E> {
    pub fn contiguous(rows: usize, input: usize, output: usize) -> Result<Self, MathError> {
        let layout = |shape| Layout::contiguous(shape).map_err(|_| MathError::DimensionOverflow);
        Self::new(
            layout([rows, input])?,
            layout([output, input])?,
            layout([rows, output])?,
        )
    }

    pub fn new(x: Layout<2>, weight: Layout<2>, output: Layout<2>) -> Result<Self, MathError> {
        let [rows, input] = *x.shape();
        let [width, weight_input] = *weight.shape();
        if output.shape() != &[rows, width] || input != weight_input {
            return Err(MathError::ShapeMismatch);
        }
        if rows == 0 || input == 0 || width == 0 {
            return Err(MathError::EmptyMatrix);
        }
        let leading = |layout: &Layout<2>| {
            let [stride, inner] = *layout.strides();
            let [_, columns] = *layout.shape();
            if inner != 1 || stride < columns {
                return Err(MathError::UnsupportedLayout);
            }
            i32::try_from(stride).map_err(|_| MathError::DimensionOverflow)
        };
        let dim = |n| i32::try_from(n).map_err(|_| MathError::DimensionOverflow);
        let cuda_type = match E::FORMAT {
            ScalarFormat::F32 => 0,
            ScalarFormat::F16 => 2,
            ScalarFormat::Bf16 => 14,
            _ => return Err(MathError::UnsupportedFormat),
        };
        Ok(Self {
            args: GemmArguments {
                x: 0,
                weight: 0,
                output: 0,
                rows: dim(rows)?,
                input: dim(input)?,
                width: dim(width)?,
                x_stride: leading(&x)?,
                weight_stride: leading(&weight)?,
                output_stride: leading(&output)?,
                cuda_type,
            },
            x,
            weight,
            output,
            _element: PhantomData,
        })
    }

    #[inline]
    pub fn input_layout(&self) -> Layout<2> {
        self.x
    }
    #[inline]
    pub fn weight_layout(&self) -> Layout<2> {
        self.weight
    }
    #[inline]
    pub fn output_layout(&self) -> Layout<2> {
        self.output
    }

    /// CPU-only address binding, with no allocation, lookup, or driver call.
    /// The executor must validate storage and retain owners through completion.
    #[inline]
    pub fn arguments(&self, x: u64, weight: u64, output: u64) -> GemmArguments {
        GemmArguments {
            x,
            weight,
            output,
            ..self.args
        }
    }

    /// Check borrowed storage without allocation, refcount changes or CUDA
    /// calls. The result is ABI data, not a lifetime guarantee: enqueue is
    /// unsafe unless the executor retains owners through stream completion.
    #[inline]
    pub fn bind(
        &self,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<GemmArguments, MathError> {
        if !Arc::ptr_eq(x.context(), weight.context())
            || !Arc::ptr_eq(x.context(), output.context())
        {
            return Err(MathError::ContextMismatch);
        }
        let x = TensorRef::<E, 2>::new(x, self.x).map_err(MathError::Tensor)?;
        let w = TensorRef::<E, 2>::new(weight, self.weight).map_err(MathError::Tensor)?;
        let y = TensorMut::<E, 2>::new(output, self.output).map_err(MathError::Tensor)?;
        Ok(self.arguments(x.device_ptr(), w.device_ptr(), y.device_ptr()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Bf16, F16, F32};

    #[test]
    fn rejects_inconsistent_or_unrepresentable_geometry() {
        let x = Layout::contiguous([3, 5]).unwrap();
        let w = Layout::contiguous([7, 6]).unwrap();
        let y = Layout::contiguous([3, 7]).unwrap();
        assert!(matches!(
            LinearPlan::<Bf16>::new(x, w, y),
            Err(MathError::ShapeMismatch)
        ));
        assert!(matches!(
            LinearPlan::<Bf16>::contiguous(0, 5, 7),
            Err(MathError::EmptyMatrix)
        ));
        assert!(matches!(
            LinearPlan::<Bf16>::contiguous(i32::MAX as usize + 1, 1, 1),
            Err(MathError::DimensionOverflow)
        ));
    }

    #[test]
    fn padded_rows_reach_the_correct_blas_leading_dimensions() {
        let p = LinearPlan::<Bf16>::new(
            Layout::strided([3, 5], [8, 1]).unwrap(),
            Layout::strided([7, 5], [16, 1]).unwrap(),
            Layout::strided([3, 7], [12, 1]).unwrap(),
        )
        .unwrap();
        let a = p.arguments(100, 200, 300);
        assert_eq!((a.rows, a.input, a.width), (3, 5, 7));
        assert_eq!((a.x_stride, a.weight_stride, a.output_stride), (8, 16, 12));
        assert_eq!((a.x, a.weight, a.output), (100, 200, 300));
        assert_eq!(a.cuda_type, 14);
    }

    #[test]
    fn rejects_column_major_input_instead_of_reinterpreting_it() {
        let x = Layout::strided([3, 5], [1, 3]).unwrap();
        let w = Layout::contiguous([7, 5]).unwrap();
        let y = Layout::contiguous([3, 7]).unwrap();
        assert!(matches!(
            LinearPlan::<Bf16>::new(x, w, y),
            Err(MathError::UnsupportedLayout)
        ));
    }

    #[test]
    fn implemented_storage_formats_select_distinct_blas_types() {
        assert_eq!(
            LinearPlan::<F16>::contiguous(2, 3, 4)
                .unwrap()
                .arguments(0, 0, 0)
                .cuda_type,
            2
        );
        assert_eq!(
            LinearPlan::<F32>::contiguous(2, 3, 4)
                .unwrap()
                .arguments(0, 0, 0)
                .cuda_type,
            0
        );
    }
}
