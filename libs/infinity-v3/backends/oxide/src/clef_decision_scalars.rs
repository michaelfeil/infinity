// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! One cold BF16-boundary scalar preparation launch for Clef.

use crate::{arg, clef_decision_head::ClefKernels, infer, launch, tensor::Bf16};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

/// Upload the three already-BF16 source scalars, run one cold scalar kernel and
/// return the widened F32 results in Native order. This is the only scalar
/// readback; request enqueue never reads or queries them again.
pub(crate) fn prepare(
    stream: &Arc<Stream>,
    kernels: &ClefKernels,
    scalars: [Bf16; 3],
) -> Result<[f32; 3], BackendError> {
    let input = DeviceBuffer::from_host(stream, &scalars).map_err(infer)?;
    let output = DeviceBuffer::<f32>::zeroed(stream, 3).map_err(infer)?;
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut count = 3u32;
    launch(
        &kernels.scalar_prep,
        stream,
        (1, 1, 1),
        0,
        &mut [arg(&mut input_ptr), arg(&mut output_ptr), arg(&mut count)],
    )?;
    let host: Vec<f32> = output.to_host_vec(stream).map_err(infer)?;
    let values: [f32; 3] = host
        .try_into()
        .map_err(|_| infer("Clef cold scalar output count mismatch"))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(infer(
            "Clef cold scalar preparation produced invalid values",
        ));
    }
    Ok(values)
}
