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

/// Cache a logical row count only after every prepared plan has refreshed.
/// A failed refresh may leave earlier plans at the new shape, so invalidate
/// first: retrying either shape must visit every plan again.
#[inline]
pub(crate) fn refresh_plan_rows<E>(
    cached: &mut Option<usize>,
    rows: usize,
    update: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    if *cached == Some(rows) {
        return Ok(());
    }
    *cached = None;
    update()?;
    *cached = Some(rows);
    Ok(())
}

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

    /// Refresh the logical row count for a plan constructed through
    /// [`Self::contiguous`]. The weight and inner dimensions are immutable, so
    /// this only recomputes the checked input/output layouts and GEMM scalars.
    /// It performs no allocation, heuristic, lookup, or driver call.
    pub fn set_rows(&mut self, rows: usize) -> Result<(), MathError> {
        let [_, input] = *self.x.shape();
        let [width, weight_input] = *self.weight.shape();
        if rows == 0 || input == 0 || width == 0 {
            return Err(MathError::EmptyMatrix);
        }
        if input != weight_input {
            return Err(MathError::ShapeMismatch);
        }
        let [x_stride, x_inner] = *self.x.strides();
        let [output_stride, output_inner] = *self.output.strides();
        if x_inner != 1 || output_inner != 1 || x_stride != input || output_stride != width {
            return Err(MathError::UnsupportedLayout);
        }
        let dim = |n| i32::try_from(n).map_err(|_| MathError::DimensionOverflow);
        let x = Layout::contiguous([rows, input]).map_err(|_| MathError::DimensionOverflow)?;
        let output = Layout::contiguous([rows, width]).map_err(|_| MathError::DimensionOverflow)?;
        self.args.rows = dim(rows)?;
        self.args.x_stride = dim(input)?;
        self.args.output_stride = dim(width)?;
        self.x = x;
        self.output = output;
        Ok(())
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
    fn failed_plan_refresh_retries_all_plans_for_failed_or_previous_shape() {
        // Model individual descriptor rollback: the failing plan keeps its old
        // shape, earlier plans have changed, and later plans were not visited.
        let update = |plans: &mut [usize; 4],
                      calls: &mut [usize; 4],
                      target: usize,
                      fail_at: Option<usize>| {
            for (index, plan) in plans.iter_mut().enumerate() {
                calls[index] += 1;
                if fail_at == Some(index) {
                    return Err(MathError::UnsupportedLayout);
                }
                *plan = target;
            }
            Ok(())
        };
        for recovery_shape in [129usize, 32] {
            let mut cached = Some(32);
            let mut plans = [32; 4];
            let mut calls = [0; 4];
            let failed = refresh_plan_rows(&mut cached, 129, || {
                update(&mut plans, &mut calls, 129, Some(2))
            });
            assert!(matches!(failed, Err(MathError::UnsupportedLayout)));
            assert_eq!(plans, [129, 129, 32, 32]);
            assert_eq!(
                cached, None,
                "neither shape is valid after a partial update"
            );

            refresh_plan_rows(&mut cached, recovery_shape, || {
                update(&mut plans, &mut calls, recovery_shape, None)
            })
            .unwrap();
            assert_eq!(plans, [recovery_shape; 4]);
            assert_eq!(calls, [2, 2, 2, 1], "every plan must be revisited");
            assert_eq!(cached, Some(recovery_shape));

            let previous_calls = calls;
            refresh_plan_rows(&mut cached, recovery_shape, || {
                update(&mut plans, &mut calls, recovery_shape, None)
            })
            .unwrap();
            assert_eq!(
                calls, previous_calls,
                "unchanged valid shapes remain cached"
            );
        }
    }

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
    fn contiguous_plans_refresh_rows_without_changing_weight_geometry() {
        let mut plan = LinearPlan::<Bf16>::contiguous(128, 5, 7).unwrap();
        let weight = plan.weight_layout();
        plan.set_rows(3).unwrap();
        assert_eq!(plan.input_layout().shape(), &[3, 5]);
        assert_eq!(plan.output_layout().shape(), &[3, 7]);
        assert_eq!(plan.arguments(1, 2, 3).rows, 3);
        assert_eq!(plan.arguments(1, 2, 3).x_stride, 5);
        assert_eq!(plan.arguments(1, 2, 3).output_stride, 7);
        assert_eq!(plan.weight_layout(), weight);
        assert!(matches!(plan.set_rows(0), Err(MathError::EmptyMatrix)));
        assert_eq!(plan.input_layout().shape(), &[3, 5]);
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
