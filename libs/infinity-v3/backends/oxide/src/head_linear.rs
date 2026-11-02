// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared native head projection policy. Missing bias selects matmul rather
//! than a synthetic zero-bias Lt epilogue; biased paths retain their old plan.
use crate::{
    biased_linear::{BiasedLinear, PreparedBiasedLinear},
    cublas::Blas,
    infer,
    lt::{LtProvider, ShapeError},
    math::{GemmArguments, LinearPlan},
    tensor::Bf16,
};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

pub(crate) enum HeadLinearPlan {
    Biased(Box<BiasedLinear>),
    Unbiased { input: usize, output: usize },
}
pub(crate) enum PreparedHeadLinear {
    Biased(PreparedBiasedLinear),
    Unbiased {
        arguments: GemmArguments,
        stream: usize,
    },
}
impl HeadLinearPlan {
    #[cfg(test)]
    pub(crate) fn native_lt_queries(&self, provider: &LtProvider) -> usize {
        match self {
            Self::Biased(plan) => plan.native_lt_queries(provider),
            Self::Unbiased { .. } => 0,
        }
    }
    #[cfg(test)]
    pub(crate) fn native_lt_cache_len(&self, provider: &LtProvider) -> usize {
        match self {
            Self::Biased(plan) => plan.native_lt_cache_len(provider),
            Self::Unbiased { .. } => 0,
        }
    }
    pub(crate) fn new(
        provider: &LtProvider,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<Self, ShapeError> {
        if has_bias {
            Ok(Self::Biased(Box::new(BiasedLinear::new(
                provider, input, output,
            )?)))
        } else {
            LinearPlan::<Bf16>::contiguous(1, input, output)
                .map_err(|e| ShapeError::Fatal(infer(e)))?;
            Ok(Self::Unbiased { input, output })
        }
    }
    pub(crate) fn preflight_rows(&self, rows: usize) -> Result<(), ShapeError> {
        match self {
            Self::Biased(plan) => plan.preflight_rows(rows),
            Self::Unbiased { input, output } => {
                LinearPlan::<Bf16>::contiguous(rows, *input, *output)
                    .map(|_| ())
                    .map_err(|e| ShapeError::Rejected(infer(e)))
            }
        }
    }
    /// # Safety
    /// All buffers share the retained ordered stream and survive completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn prepare(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        rows: usize,
        stream: &Stream,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<PreparedHeadLinear, ShapeError> {
        match self {
            Self::Biased(plan) => {
                unsafe { plan.prepare(provider, blas, rows, stream, x, weight, bias, output) }
                    .map(PreparedHeadLinear::Biased)
            }
            Self::Unbiased {
                input,
                output: width,
            } => {
                if !std::ptr::eq(blas.stream(), stream) || !Arc::ptr_eq(x.context(), blas.context())
                {
                    return Err(ShapeError::Fatal(infer(
                        "biasless head stream/context mismatch",
                    )));
                }
                let plan = LinearPlan::<Bf16>::contiguous(rows, *input, *width)
                    .map_err(|e| ShapeError::Rejected(infer(e)))?;
                let arguments = plan
                    .bind(x, weight, output)
                    .map_err(|e| ShapeError::Rejected(infer(e)))?;
                Ok(PreparedHeadLinear::Unbiased {
                    arguments,
                    stream: stream.cu_stream() as usize,
                })
            }
        }
    }
    /// # Safety
    /// The caller retains every preflight owner through this stream completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch(
        &mut self,
        prepared: &PreparedHeadLinear,
        provider: &LtProvider,
        blas: &Blas,
        seed: &Function,
        stream: &Stream,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<(), BackendError> {
        match (self, prepared) {
            (Self::Biased(plan), PreparedHeadLinear::Biased(prepared)) => unsafe {
                plan.launch(
                    prepared, provider, blas, seed, stream, x, weight, bias, output,
                )
            },
            (
                Self::Unbiased {
                    input,
                    output: width,
                },
                PreparedHeadLinear::Unbiased {
                    arguments,
                    stream: prepared_stream,
                },
            ) => {
                if !std::ptr::eq(blas.stream(), stream)
                    || *prepared_stream != stream.cu_stream() as usize
                    || !Arc::ptr_eq(x.context(), blas.context())
                {
                    return Err(infer(
                        "biasless head stream/context changed after preparation",
                    ));
                }
                let rows = usize::try_from(arguments.rows).map_err(infer)?;
                let plan = LinearPlan::<Bf16>::contiguous(rows, *input, *width).map_err(infer)?;
                let current = plan.bind(x, weight, output).map_err(infer)?;
                if current != *arguments {
                    return Err(infer(
                        "biasless head pointers/shape changed after preparation",
                    ));
                }
                unsafe { blas.linear_native_default_bound(arguments) }
            }
            _ => Err(infer(
                "head projection bias policy changed after preparation",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn biasless_projection_accepts_actual_dynamic_rows_and_rejects_empty_or_overflow() {
        let plan = HeadLinearPlan::Unbiased {
            input: 1024,
            output: 9,
        };
        for rows in [1, 7, 129, 2048] {
            assert!(plan.preflight_rows(rows).is_ok());
        }
        assert!(plan.preflight_rows(0).is_err());
        assert!(plan.preflight_rows(i32::MAX as usize + 1).is_err());
    }
}
