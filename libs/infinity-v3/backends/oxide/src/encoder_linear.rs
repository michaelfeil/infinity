// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Opaque prepared encoder projections with biased or plain dispatch.
//!
//! A plan retains one cold-boxed owner. Row refreshes overwrite the inline
//! optional [`LinearPlan`] inside that box and never allocate per shape.

use crate::{
    cublas::Blas,
    lt::{LtPlan, LtProvider, ShapeError},
    math::LinearPlan,
    upload::UploadElement,
};
use cuda_core::DeviceBuffer;
use infinity_v3_backend_core::BackendError;

pub(crate) enum ProjectionPlan<E: UploadElement> {
    Biased(Box<LtPlan<E>>),
    Plain(Box<PlainPlan<E>>),
}

pub(crate) struct PlainPlan<E: UploadElement> {
    pub(crate) input: usize,
    pub(crate) output: usize,
    pub(crate) plan: Option<LinearPlan<E>>,
}

impl<E: UploadElement> PlainPlan<E> {
    pub(crate) fn new(input: usize, output: usize) -> Result<Self, BackendError> {
        LinearPlan::<E>::contiguous(1, input, output).map_err(crate::infer)?;
        Ok(Self {
            input,
            output,
            plan: None,
        })
    }

    /// Prepare the exact logical row count with checked pure-math geometry.
    /// A rejected refresh leaves the previous plan untouched.
    pub(crate) fn set_rows(&mut self, rows: usize) -> Result<(), ShapeError> {
        let plan = LinearPlan::<E>::contiguous(rows, self.input, self.output)
            .map_err(|error| ShapeError::Rejected(crate::infer(error)))?;
        self.plan = Some(plan);
        Ok(())
    }
}

impl<E: UploadElement> ProjectionPlan<E> {
    pub(crate) fn biased(
        provider: &LtProvider,
        input: usize,
        output: usize,
    ) -> Result<Self, BackendError> {
        provider
            .prepare_native_small_bias_shape::<E>(input, output)
            .map(|plan| Self::Biased(Box::new(plan)))
    }

    pub(crate) fn plain(input: usize, output: usize) -> Result<Self, BackendError> {
        PlainPlan::new(input, output).map(|plan| Self::Plain(Box::new(plan)))
    }

    /// Refresh rows transactionally; callers keep the model-row cache outside
    /// and invalidate it through the existing math refresh helper.
    pub(crate) fn set_rows(
        &mut self,
        provider: &LtProvider,
        rows: usize,
    ) -> Result<(), ShapeError> {
        match self {
            Self::Biased(plan) => provider.set_logical_rows(plan, rows),
            Self::Plain(plan) => plan.set_rows(rows),
        }
    }

    /// Enqueue the prepared projection without allocating or rebinding context.
    ///
    /// # Safety
    /// The caller must bind the provider's context on this thread and retain
    /// disjoint input, weight, bias, and output allocations, without conflicting
    /// access, until the stream drains. Borrows alone do not cover deferred use.
    pub(crate) unsafe fn enqueue(
        &self,
        provider: &LtProvider,
        blas: Option<&Blas>,
        input: &DeviceBuffer<E>,
        weight: &DeviceBuffer<E>,
        bias: Option<&DeviceBuffer<E>>,
        output: &mut DeviceBuffer<E>,
    ) -> Result<(), BackendError> {
        match self {
            Self::Biased(plan) => {
                let bias =
                    bias.ok_or_else(|| crate::infer("biased projection requires a bias buffer"))?;
                // SAFETY: the caller owns completion and retains every buffer;
                // this path keeps the prepared fused bias epilogue unchanged.
                unsafe { provider.enqueue(plan, input, weight, bias, output) }
            }
            Self::Plain(plan) => {
                if bias.is_some() {
                    return Err(crate::infer(
                        "plain projection does not accept a bias buffer",
                    ));
                }
                let blas =
                    blas.ok_or_else(|| crate::infer("plain projection requires a cuBLAS handle"))?;
                let plan = plan
                    .plan
                    .as_ref()
                    .ok_or_else(|| crate::infer("plain projection rows are not prepared"))?;
                // SAFETY: the caller owns completion and retains every buffer;
                // plain dispatch uses the ordinary no-epilogue GEMM method.
                unsafe { blas.linear(plan, input, weight, output) }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn logical_rows(&self) -> i32 {
        match self {
            Self::Biased(plan) => plan.logical_rows(),
            Self::Plain(plan) => plan
                .plan
                .as_ref()
                .map_or(0, |plan| plan.arguments(0, 0, 0).rows),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Bf16, Layout};

    #[test]
    fn plain_rows_follow_exact_geometry_and_retain_the_box() {
        let mut projection = ProjectionPlan::<Bf16>::plain(5, 7).unwrap();
        let before = match &projection {
            ProjectionPlan::Plain(plan) => plan.as_ref() as *const PlainPlan<Bf16>,
            ProjectionPlan::Biased(_) => panic!("expected a plain plan"),
        };
        assert_eq!(projection.logical_rows(), 0);
        let ProjectionPlan::Plain(plan) = &mut projection else {
            panic!("expected a plain plan");
        };
        for rows in [3usize, 11, 3] {
            plan.set_rows(rows).unwrap();
            let args = plan.plan.as_ref().unwrap().arguments(0, 0, 0);
            assert_eq!(
                (args.rows, args.x_stride, args.output_stride),
                (rows as i32, 5, 7)
            );
        }
        let after = match &projection {
            ProjectionPlan::Plain(plan) => plan.as_ref() as *const PlainPlan<Bf16>,
            ProjectionPlan::Biased(_) => panic!("expected a plain plan"),
        };
        assert_eq!(before, after, "row refresh must reuse the cold box");
        assert_eq!(projection.logical_rows(), 3);
    }

    #[test]
    fn rejected_geometry_keeps_valid_rows_and_padding_stays_metadata() {
        let mut plan = PlainPlan::<Bf16>::new(5, 7).unwrap();
        plan.set_rows(4).unwrap();
        let valid = plan.plan.as_ref().unwrap().arguments(0, 0, 0);
        for rows in [0usize, usize::try_from(i32::MAX).unwrap() + 1] {
            assert!(matches!(plan.set_rows(rows), Err(ShapeError::Rejected(_))));
            let current = plan.plan.as_ref().unwrap().arguments(0, 0, 0);
            assert_eq!(current.rows, valid.rows);
            assert_eq!((current.input, current.width), (valid.input, valid.width));
        }

        let padded = LinearPlan::<Bf16>::new(
            Layout::strided([3, 5], [8, 1]).unwrap(),
            Layout::strided([7, 5], [16, 1]).unwrap(),
            Layout::strided([3, 7], [12, 1]).unwrap(),
        )
        .unwrap();
        let projection = ProjectionPlan::Plain(Box::new(PlainPlan {
            input: 5,
            output: 7,
            plan: Some(padded),
        }));
        assert_eq!(projection.logical_rows(), 3);
    }
}
