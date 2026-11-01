// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Shared prepared biased-linear dispatch for trained task heads.
//!
//! The policy is selected once per request from the actual dimensions: the
//! native-default cuBLASLt bias epilogue serves `M > 0 && K > 1 && N > 1`,
//! while every other legal geometry uses the native `GemmEx` beta=1 method
//! over a bias-preseeded output. Both methods accumulate in FP32 and round to
//! BF16 once. Cold preparation may run the native heuristic / `AlgoCheck`;
//! cache hits and launches perform no metadata query, allocation, or context
//! query.

use crate::{
    arg, blocks,
    cublas::Blas,
    infer, launch,
    lt::{
        LtPlan, LtProvider, NativeDefaultPending, NativeDefaultPrepared, ShapeError,
        NATIVE_DEFAULT_SEED_ROWS,
    },
    math::{GemmArguments, LinearPlan},
    product,
    tensor::Bf16,
    u32_size,
};
use cuda_core::{DeviceBuffer, Function, Stream};
use std::{mem::size_of, sync::Arc};

pub(crate) struct BiasedLinear {
    input: usize,
    output: usize,
    lt: Option<LtPlan<Bf16>>,
    plain: LinearPlan<Bf16>,
    pending: NativeDefaultPending,
}

pub(crate) enum PreparedBiasedLinear {
    NativeLt {
        prepared: NativeDefaultPrepared,
        boundary: Boundary,
    },
    GemmEx {
        arguments: GemmArguments,
        boundary: Boundary,
        bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Boundary {
    stream: usize,
    x: u64,
    weight: u64,
    bias: u64,
    output: u64,
}

impl Boundary {
    fn current(
        stream: &Stream,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &DeviceBuffer<Bf16>,
    ) -> Self {
        Self {
            stream: stream.cu_stream() as usize,
            x: x.cu_deviceptr(),
            weight: weight.cu_deviceptr(),
            bias: bias.cu_deviceptr(),
            output: output.cu_deviceptr(),
        }
    }
}

/// Current-buffer identity is a handle/owner invariant. A caller that swaps a
/// stale bias, output or stream after preparation must fail closed instead of
/// seeding one allocation and projecting another.
fn require_boundary(expected: Boundary, current: Boundary) -> Result<(), ShapeError> {
    if expected != current {
        return Err(ShapeError::Fatal(infer(
            "classifier launch buffers or stream differ from prepared metadata",
        )));
    }
    Ok(())
}

fn require_contexts(
    buffers: [&Arc<cuda_core::Context>; 4],
    owner: &Arc<cuda_core::Context>,
) -> Result<(), ShapeError> {
    if buffers
        .into_iter()
        .any(|context| !Arc::ptr_eq(context, owner))
    {
        return Err(ShapeError::Fatal(infer(
            "classifier projection buffers belong to a different CUDA context",
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceRange {
    start: u64,
    end: u64,
}

fn checked_range(pointer: u64, elements: usize) -> Result<DeviceRange, ShapeError> {
    let bytes = elements
        .checked_mul(size_of::<Bf16>())
        .ok_or_else(|| ShapeError::Rejected(infer("classifier byte extent overflows usize")))?;
    let bytes = u64::try_from(bytes).map_err(|_| {
        ShapeError::Rejected(infer(
            "classifier byte extent exceeds the device address range",
        ))
    })?;
    let end = pointer.checked_add(bytes).ok_or_else(|| {
        ShapeError::Rejected(infer("classifier device address range overflows u64"))
    })?;
    Ok(DeviceRange {
        start: pointer,
        end,
    })
}

fn overlaps(first: DeviceRange, second: DeviceRange) -> bool {
    first.start < second.end && second.start < first.end
}

fn require_capacity(name: &'static str, actual: usize, required: usize) -> Result<(), ShapeError> {
    if actual < required {
        return Err(ShapeError::Rejected(infer(format!(
            "classifier {name} buffer has {actual} elements but needs at least {required}"
        ))));
    }
    Ok(())
}

/// Pure logical range for the GemmEx bias preseed. Geometry and capacity are
/// caller rejections; current-buffer mismatches at launch become fatal.
fn seed_range(rows: usize, width: usize) -> Result<(u32, u32, usize), ShapeError> {
    if rows == 0 || width == 0 {
        return Err(ShapeError::Rejected(infer(
            "classifier seed requires nonzero rows and width",
        )));
    }
    let elements = product(rows, width).map_err(ShapeError::Rejected)?;
    let bytes = elements.checked_mul(size_of::<Bf16>()).ok_or_else(|| {
        ShapeError::Rejected(infer("classifier seed byte extent overflows usize"))
    })?;
    Ok((
        u32_size(rows).map_err(ShapeError::Rejected)?,
        u32_size(width).map_err(ShapeError::Rejected)?,
        bytes,
    ))
}

/// Host-only storage admission for one projection. Capacity and geometry errors
/// are recoverable rejections; cross-context ownership is fatal.
fn preflight_storage(
    rows: usize,
    input: usize,
    output: usize,
    x: &DeviceBuffer<Bf16>,
    weight: &DeviceBuffer<Bf16>,
    bias: &DeviceBuffer<Bf16>,
    output_buffer: &DeviceBuffer<Bf16>,
) -> Result<(), ShapeError> {
    if rows == 0 || input == 0 || output == 0 {
        return Err(ShapeError::Rejected(infer(
            "classifier projection requires nonzero rows and dimensions",
        )));
    }
    let x_elements = product(rows, input).map_err(ShapeError::Rejected)?;
    let weight_elements = product(output, input).map_err(ShapeError::Rejected)?;
    let output_elements = product(rows, output).map_err(ShapeError::Rejected)?;
    u32_size(x_elements).map_err(ShapeError::Rejected)?;
    u32_size(weight_elements).map_err(ShapeError::Rejected)?;
    u32_size(output_elements).map_err(ShapeError::Rejected)?;
    require_capacity("input", x.len(), x_elements)?;
    require_capacity("weight", weight.len(), weight_elements)?;
    require_capacity("bias", bias.len(), output)?;
    require_capacity("output", output_buffer.len(), output_elements)?;
    require_contexts(
        [
            x.context(),
            weight.context(),
            bias.context(),
            output_buffer.context(),
        ],
        x.context(),
    )?;
    let x_range = checked_range(x.cu_deviceptr(), x.len())?;
    let weight_range = checked_range(weight.cu_deviceptr(), weight.len())?;
    let bias_range = checked_range(bias.cu_deviceptr(), bias.len())?;
    let output_range = checked_range(output_buffer.cu_deviceptr(), output_buffer.len())?;
    for (name, other) in [
        ("input", x_range),
        ("weight", weight_range),
        ("bias", bias_range),
    ] {
        if overlaps(output_range, other) {
            return Err(ShapeError::Rejected(infer(format!(
                "classifier output aliases {name} storage"
            ))));
        }
    }
    Ok(())
}

/// The only source-faithful selection rule for the prepared head methods.
pub(crate) fn selects_native_lt(rows: usize, input: usize, output: usize) -> bool {
    rows > 0 && input > 1 && output > 1
}

impl BiasedLinear {
    pub(crate) fn new(
        provider: &LtProvider,
        input: usize,
        output: usize,
    ) -> Result<Self, ShapeError> {
        let lt = selects_native_lt(2, input, output)
            .then(|| {
                provider
                    .prepare_native_default_bias_shape::<Bf16>(
                        input,
                        output,
                        NATIVE_DEFAULT_SEED_ROWS,
                    )
                    .map_err(ShapeError::Fatal)
            })
            .transpose()?;
        let plain = LinearPlan::<Bf16>::contiguous(NATIVE_DEFAULT_SEED_ROWS, input, output)
            .map_err(|error| ShapeError::Rejected(infer(error)))?;
        Ok(Self {
            input,
            output,
            lt,
            plain,
            pending: NativeDefaultPending::default(),
        })
    }

    pub(crate) fn preflight_rows(&self, rows: usize) -> Result<(), ShapeError> {
        if rows == 0 {
            return Err(ShapeError::Rejected(infer(
                "classifier projection rows must be nonzero",
            )));
        }
        product(rows, self.input).map_err(ShapeError::Rejected)?;
        product(rows, self.output).map_err(ShapeError::Rejected)?;
        product(self.output, self.input).map_err(ShapeError::Rejected)?;
        u32_size(product(rows, self.input).map_err(ShapeError::Rejected)?)
            .map_err(ShapeError::Rejected)?;
        u32_size(product(rows, self.output).map_err(ShapeError::Rejected)?)
            .map_err(ShapeError::Rejected)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn native_lt_queries(&self, provider: &LtProvider) -> usize {
        self.lt
            .as_ref()
            .map_or(0, |plan| provider.native_default_queries(plan))
    }

    #[cfg(test)]
    pub(crate) fn native_lt_cache_len(&self, provider: &LtProvider) -> usize {
        self.lt
            .as_ref()
            .map_or(0, |plan| provider.native_default_cache_len(plan))
    }

    fn end_prepare(&mut self) {
        self.pending.clear();
    }

    /// Bind every actual pointer and select the prepared method before the
    /// request body is submitted. A cold shape may run the native heuristic and
    /// `AlgoCheck`; prepared cache hits do not. This performs no allocation.
    ///
    /// # Safety
    /// The provider/engine CUDA context must already be bound on this thread
    /// and remain current through the ordered launch.
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
    ) -> Result<PreparedBiasedLinear, ShapeError> {
        self.preflight_rows(rows)?;
        let result = (|| {
            preflight_storage(rows, self.input, self.output, x, weight, bias, output)?;
            let boundary = Boundary::current(stream, x, weight, bias, output);
            if selects_native_lt(rows, self.input, self.output) {
                if !std::ptr::eq(provider.stream(), stream) {
                    return Err(ShapeError::Fatal(infer(
                        "classifier stream is not the cuBLASLt provider's creation stream",
                    )));
                }
                require_contexts(
                    [
                        x.context(),
                        weight.context(),
                        bias.context(),
                        output.context(),
                    ],
                    provider.context(),
                )?;
                let plan = self.lt.as_mut().ok_or_else(|| {
                    ShapeError::Fatal(infer("missing native-default classifier plan"))
                })?;
                // SAFETY: the caller's bound-context contract covers this
                // pointer-only preparation.
                let prepared = unsafe {
                    provider.prepare_native_default_bound(
                        plan,
                        &mut self.pending,
                        rows,
                        x,
                        weight,
                        bias,
                        output,
                    )?
                };
                Ok(PreparedBiasedLinear::NativeLt { prepared, boundary })
            } else {
                if !std::ptr::eq(blas.stream(), stream) {
                    return Err(ShapeError::Fatal(infer(
                        "classifier stream is not the plain cuBLAS creation stream",
                    )));
                }
                require_contexts(
                    [
                        x.context(),
                        weight.context(),
                        bias.context(),
                        output.context(),
                    ],
                    blas.context(),
                )?;
                self.plain
                    .set_rows(rows)
                    .map_err(|error| ShapeError::Rejected(infer(error)))?;
                // Every capacity/geometry check above is already complete, so a
                // remaining handle/binding failure is an owner invariant.
                let arguments = blas
                    .preflight_beta_one(&self.plain, x, weight, output)
                    .map_err(|error| ShapeError::Fatal(infer(error)))?;
                let (_, _, bytes) = seed_range(rows, self.output)?;
                Ok(PreparedBiasedLinear::GemmEx {
                    arguments,
                    boundary,
                    bytes,
                })
            }
        })();
        self.end_prepare();
        result
    }

    /// Launch the preselected method after revalidating the exact current
    /// buffers, stream, context and seed range. For the GemmEx path the
    /// bias-seed kernel initializes every logical `M*N` element before the
    /// FP32 beta=1 GEMM.
    ///
    /// # Safety
    /// The caller retains every buffer and the kernel/module handle through
    /// stream completion and passes the same buffers used during preparation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch(
        &mut self,
        prepared: &PreparedBiasedLinear,
        provider: &LtProvider,
        blas: &Blas,
        seed: &Function,
        stream: &Stream,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<(), crate::BackendError> {
        let current = Boundary::current(stream, x, weight, bias, output);
        match prepared {
            PreparedBiasedLinear::NativeLt { prepared, boundary } => {
                require_boundary(*boundary, current).map_err(ShapeError::into_backend)?;
                if !std::ptr::eq(provider.stream(), stream) {
                    return Err(infer(
                        "classifier stream is not the cuBLASLt provider's creation stream",
                    ));
                }
                require_contexts(
                    [
                        x.context(),
                        weight.context(),
                        bias.context(),
                        output.context(),
                    ],
                    provider.context(),
                )
                .map_err(ShapeError::into_backend)?;
                let lt = self
                    .lt
                    .as_mut()
                    .ok_or_else(|| infer("missing native-default classifier plan"))?;
                // SAFETY: the caller's retained-buffer contract covers the
                // preselected invocation, whose bias identity was revalidated.
                unsafe { provider.launch_native_default_bound(lt, prepared, x, weight, output) }
                    .map_err(ShapeError::into_backend)
            }
            PreparedBiasedLinear::GemmEx {
                arguments,
                boundary,
                bytes,
            } => {
                require_boundary(*boundary, current).map_err(ShapeError::into_backend)?;
                if !std::ptr::eq(blas.stream(), stream) {
                    return Err(infer(
                        "classifier stream is not the plain cuBLAS creation stream",
                    ));
                }
                require_contexts(
                    [
                        x.context(),
                        weight.context(),
                        bias.context(),
                        output.context(),
                    ],
                    blas.context(),
                )
                .map_err(ShapeError::into_backend)?;
                if !Arc::ptr_eq(seed.context(), blas.context()) {
                    return Err(infer(
                        "classifier bias-seed kernel belongs to a different CUDA context",
                    ));
                }
                let binding = self
                    .plain
                    .bind(x, weight, output)
                    .map_err(|error| infer(format!("classifier binding changed: {error}")))?;
                if binding != *arguments {
                    return Err(infer("classifier GEMM arguments changed after preparation"));
                }
                let (mut rows, mut width, current_bytes) = seed_range(
                    u32_size(arguments.rows as usize)? as usize,
                    u32_size(arguments.width as usize)? as usize,
                )
                .map_err(ShapeError::into_backend)?;
                if current_bytes != *bytes || bias.len() < arguments.width as usize {
                    return Err(infer(
                        "classifier bias-seed range changed after preparation",
                    ));
                }
                let mut bias_pointer = bias.cu_deviceptr();
                let mut output_pointer = output.cu_deviceptr();
                launch(
                    seed,
                    stream,
                    blocks(product(rows as usize, width as usize)?)?,
                    0,
                    &mut [
                        arg(&mut bias_pointer),
                        arg(&mut output_pointer),
                        arg(&mut rows),
                        arg(&mut width),
                    ],
                )?;
                // SAFETY: the current binding equals the prepared arguments,
                // and the seed above is ordered before the beta=1 GEMM.
                unsafe { blas.enqueue_native_n1(arguments) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_boundary_mismatches_are_fatal_and_seed_geometry_is_rejected() {
        let expected = Boundary {
            stream: 1,
            x: 10,
            weight: 20,
            bias: 30,
            output: 40,
        };
        for current in [
            Boundary {
                stream: 2,
                ..expected
            },
            Boundary { x: 11, ..expected },
            Boundary {
                weight: 21,
                ..expected
            },
            Boundary {
                bias: 31,
                ..expected
            },
            Boundary {
                output: 41,
                ..expected
            },
        ] {
            assert!(matches!(
                require_boundary(expected, current),
                Err(ShapeError::Fatal(_))
            ));
        }
        assert!(require_boundary(expected, expected).is_ok());
        assert!(matches!(seed_range(0, 1), Err(ShapeError::Rejected(_))));
        assert!(matches!(seed_range(1, 0), Err(ShapeError::Rejected(_))));
        assert_eq!(seed_range(3, 5).unwrap(), (3, 5, 3 * 5 * 2));
        assert!(seed_range(usize::MAX, 2).is_err());
    }
}
