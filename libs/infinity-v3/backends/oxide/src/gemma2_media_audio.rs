// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Device Gemma4 audio tower: cold BF16 weights, per-request scratch, and the
//! Native masked subsampling -> conformer layers -> output projection.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::cublas::Blas;
use crate::gemma2_media::{
    AudioActivation, AudioConfig, AudioHost, MediaLinear, AUDIO_FEATURE_SIZE,
};
use crate::gemma2_media_exec::{
    audio_norm_rows, expand_bf16_to_f32, media_linear, unit_norm_heads, upload_bf16,
    upload_norm_source, MediaKernels,
};
use crate::math::LinearPlan;
use crate::tensor::{Bf16, F32};
use crate::upload::WeightUploader;
use crate::{arg, blocks, infer, launch, launch_threads, product, start, u32_size};
use cuda_core::{DeviceBuffer, DeviceCopy, Stream};
use infinity_v3_backend_core::media::{AudioDescriptor, MediaFamily};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const AUDIO_CONV_KERNEL: usize = 3;

struct LinearRuntime {
    weight: DeviceBuffer<Bf16>,
    bias: Option<DeviceBuffer<Bf16>>,
    input_clip: Option<[f32; 2]>,
    output_clip: Option<[f32; 2]>,
    shape: [usize; 2],
}

struct AudioMlpRuntime {
    pre_norm: DeviceBuffer<f32>,
    first: LinearRuntime,
    second: LinearRuntime,
    post_norm: DeviceBuffer<f32>,
}

struct AudioLayerRuntime {
    feed_forward1: AudioMlpRuntime,
    feed_forward2: AudioMlpRuntime,
    norm_pre_attn: DeviceBuffer<f32>,
    norm_post_attn: DeviceBuffer<f32>,
    norm_out: DeviceBuffer<f32>,
    q: LinearRuntime,
    k: LinearRuntime,
    v: LinearRuntime,
    relative_k: LinearRuntime,
    post: LinearRuntime,
    per_dim_scale: DeviceBuffer<Bf16>,
    lconv_pre_norm: DeviceBuffer<f32>,
    linear_start: LinearRuntime,
    depthwise_conv1d: DeviceBuffer<Bf16>,
    conv_norm: DeviceBuffer<f32>,
    linear_end: LinearRuntime,
}

pub(crate) struct AudioRuntime {
    pub(crate) config: AudioConfig,
    subsample0_conv: DeviceBuffer<Bf16>,
    subsample0_norm: DeviceBuffer<Bf16>,
    subsample1_conv: DeviceBuffer<Bf16>,
    subsample1_norm: DeviceBuffer<Bf16>,
    input_projection: LinearRuntime,
    output_projection: LinearRuntime,
    embed_projection: LinearRuntime,
    layers: Vec<AudioLayerRuntime>,
}

pub(crate) struct AudioScratch {
    pub(crate) frames: usize,
    features_f32: DeviceBuffer<f32>,
    masked_f32: DeviceBuffer<f32>,
    valid: DeviceBuffer<u8>,
    valid_block1: DeviceBuffer<u8>,
    valid_frames: DeviceBuffer<u8>,
    conv0_columns: DeviceBuffer<f32>,
    conv0_filter: DeviceBuffer<f32>,
    conv_rows_f32: DeviceBuffer<f32>,
    conv0_bf16: DeviceBuffer<Bf16>,
    conv0_transposed: DeviceBuffer<Bf16>,
    conv1_columns: DeviceBuffer<f32>,
    conv1_filter: DeviceBuffer<f32>,
    conv1_bf16: DeviceBuffer<Bf16>,
    conv1_btcw: DeviceBuffer<Bf16>,
    states: DeviceBuffer<Bf16>,
    next: DeviceBuffer<Bf16>,
    normed: DeviceBuffer<Bf16>,
    clamped: DeviceBuffer<Bf16>,
    gate: DeviceBuffer<Bf16>,
    mlp: DeviceBuffer<Bf16>,
    mlp_norm: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    attn: DeviceBuffer<Bf16>,
    post: DeviceBuffer<Bf16>,
    qk_scaled: DeviceBuffer<f32>,
    relative: DeviceBuffer<Bf16>,
    signal: DeviceBuffer<Bf16>,
    gate_start: DeviceBuffer<Bf16>,
    glu: DeviceBuffer<Bf16>,
    depthwise: DeviceBuffer<Bf16>,
    end: DeviceBuffer<Bf16>,
    projected: DeviceBuffer<Bf16>,
    projected_norm: DeviceBuffer<Bf16>,
    output: DeviceBuffer<Bf16>,
    selected: Option<DeviceBuffer<Bf16>>,
    selected_indices: Option<DeviceBuffer<u32>>,
}

struct AudioGeometry {
    frames: usize,
    t1: usize,
    t2: usize,
    w0: usize,
    w1: usize,
    hidden: usize,
    intermediate: usize,
    projected_width: usize,
    output_dims: usize,
    output_width: usize,
}

fn check_linear(
    linear: &MediaLinear,
    name: &str,
    shape: [usize; 2],
    bias: bool,
    clipped: bool,
) -> Result<(), BackendError> {
    if linear.shape != shape
        || linear.weight.shape() != shape.as_slice()
        || linear.bias.is_some() != bias
        || linear.input_clip.is_some() != clipped
        || linear.output_clip.is_some() != clipped
    {
        return Err(start(format!(
            "audio projection {name} disagrees with the Native contract"
        )));
    }
    Ok(())
}

fn upload_linear(
    uploader: &mut WeightUploader,
    linear: &MediaLinear,
    name: &str,
) -> Result<LinearRuntime, BackendError> {
    let weight = upload_bf16(uploader, &linear.weight, &format!("{name}.weight"))?;
    let bias = match &linear.bias {
        Some(bias) => Some(upload_bf16(uploader, bias, &format!("{name}.bias"))?),
        None => None,
    };
    Ok(LinearRuntime {
        weight,
        bias,
        input_clip: linear.input_clip,
        output_clip: linear.output_clip,
        shape: linear.shape,
    })
}

fn ceil_div(value: usize, divisor: usize) -> Result<usize, BackendError> {
    if divisor == 0 {
        return Err(infer("audio subsample stride is zero"));
    }
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| infer("audio subsample width overflow"))
}

fn validate_audio(
    config: &AudioConfig,
    descriptor: &AudioDescriptor,
    feature_values: usize,
    mask_values: usize,
) -> Result<AudioGeometry, BackendError> {
    if descriptor.family != MediaFamily::EmbeddingGemma2 {
        return Err(infer("audio descriptor is not an EmbeddingGemma2 span"));
    }
    if descriptor.feature_size != AUDIO_FEATURE_SIZE {
        return Err(infer("audio descriptor feature size disagrees with 128"));
    }
    if config.activation != AudioActivation::Silu {
        return Err(infer("Gemma4 audio activation must be silu"));
    }
    let frames = descriptor.frames;
    if frames == 0 {
        return Err(infer("audio span has no feature frames"));
    }
    if mask_values != frames {
        return Err(infer(format!(
            "audio mask has {mask_values} rows, expected {frames}"
        )));
    }
    let expected_features = frames
        .checked_mul(AUDIO_FEATURE_SIZE)
        .ok_or_else(|| infer("audio feature length overflow"))?;
    if feature_values != expected_features {
        return Err(infer(format!(
            "audio features have {feature_values} values, expected {expected_features}"
        )));
    }
    if descriptor.token_count == 0 {
        return Err(infer("audio span selects no valid frames"));
    }
    if descriptor.subsampled_frames.len() != descriptor.token_count
        || descriptor.source_frames.len() != descriptor.token_count
    {
        return Err(infer(
            "audio descriptor frame selection disagrees with its token count",
        ));
    }
    let stride0 = config.ss_strides[0];
    let stride1 = config.ss_strides[1];
    let t1 = ceil_div(frames, stride0)?;
    let t2 = ceil_div(t1, stride1)?;
    let w0 = ceil_div(AUDIO_FEATURE_SIZE, stride0)?;
    let w1 = ceil_div(w0, stride1)?;
    let projected_width = config.projected_width()?;
    let expected_projected = config.ss_channels[1]
        .checked_mul(w1)
        .ok_or_else(|| infer("audio projected width overflow"))?;
    if projected_width != expected_projected {
        return Err(infer("audio projected width disagrees with the subsampler"));
    }
    if descriptor.token_count > t2 {
        return Err(infer(
            "audio token count exceeds the subsampled frame count",
        ));
    }
    for &frame in &descriptor.subsampled_frames {
        if frame >= t2 {
            return Err(infer("audio selected frame exceeds the subsampled length"));
        }
    }
    for &frame in &descriptor.source_frames {
        if frame >= frames {
            return Err(infer("audio source frame exceeds the feature length"));
        }
    }
    Ok(AudioGeometry {
        frames,
        t1,
        t2,
        w0,
        w1,
        hidden: config.hidden_size,
        intermediate: config.intermediate_size,
        projected_width,
        output_dims: config.output_proj_dims,
        output_width: config.output_width,
    })
}

fn scratch_typed<T: DeviceCopy>(
    stream: &Arc<Stream>,
    elements: usize,
    name: &str,
) -> Result<DeviceBuffer<T>, BackendError> {
    if elements == 0 {
        return Err(infer(format!("empty audio scratch buffer {name}")));
    }
    // SAFETY: every scratch buffer is fully written before any later kernel
    // reads it; uninitialized contents never reach the host.
    unsafe { DeviceBuffer::uninitialized_async(stream, elements) }
        .map_err(|error| start(format!("audio scratch {name}: {error}")))
}

fn relative_signal(config: &AudioConfig) -> Result<Vec<Bf16>, BackendError> {
    let span = config.relative_span;
    let channels = config.hidden_size;
    let half = channels
        .checked_div(2)
        .ok_or_else(|| infer("audio signal width overflow"))?;
    if span == 0 || half == 0 {
        return Err(infer("audio relative signal has an empty dimension"));
    }
    let denominator = half.saturating_sub(1).max(1);
    let factor = ((-(10_000.0f64).ln()) / denominator as f64) as f32;
    let mut inverse = Vec::new();
    inverse
        .try_reserve_exact(half)
        .map_err(|_| infer("audio inverse frequency allocation failed"))?;
    for index in 0..half {
        inverse.push((index as f32 * factor).exp());
    }
    let total = span
        .checked_mul(channels)
        .ok_or_else(|| infer("audio signal length overflow"))?;
    let mut signal = Vec::new();
    signal
        .try_reserve_exact(total)
        .map_err(|_| infer("audio signal allocation failed"))?;
    for row in 0..span {
        let timing = (span - 1 - row) as f32;
        for &inverse in &inverse {
            signal.push(Bf16::from_bits(
                half::bf16::from_f32((timing * inverse).sin()).to_bits(),
            ));
        }
        for &inverse in &inverse {
            signal.push(Bf16::from_bits(
                half::bf16::from_f32((timing * inverse).cos()).to_bits(),
            ));
        }
    }
    if signal.len() != total {
        return Err(infer("audio relative signal length drifted"));
    }
    Ok(signal)
}

fn subsample_stage(mask: &[u8], stride: usize) -> Result<Vec<u8>, BackendError> {
    if mask.is_empty() || stride == 0 {
        return Err(infer("audio validity stage has an empty dimension"));
    }
    let length = ceil_div(mask.len(), stride)?;
    let mut out = Vec::new();
    out.try_reserve_exact(length)
        .map_err(|_| infer("audio validity allocation failed"))?;
    let last = mask.len() - 1;
    for frame in 0..length {
        let source = frame
            .checked_mul(stride)
            .ok_or_else(|| infer("audio validity index overflow"))?
            .min(last);
        out.push(mask[source]);
    }
    Ok(out)
}

fn project(
    kernels: &MediaKernels,
    blas: &Blas,
    stream: &Stream,
    linear: &LinearRuntime,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    input_width: usize,
    clamp: &mut DeviceBuffer<Bf16>,
) -> Result<(), BackendError> {
    if linear.shape[1] != input_width {
        return Err(infer(
            "audio projection input width disagrees with its weight",
        ));
    }
    media_linear(
        kernels,
        blas,
        stream,
        input,
        output,
        &linear.weight,
        linear.bias.as_ref(),
        rows,
        input_width,
        linear.shape[0],
        linear.input_clip,
        linear.output_clip,
        clamp,
    )
}

fn mask_into(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    valid: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    count: usize,
) -> Result<(), BackendError> {
    if input.len() < count || valid.len() < count || output.len() < count {
        return Err(infer("audio mask buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut valid = valid.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.mask,
        stream,
        blocks(count as usize)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut valid),
            arg(&mut out),
            arg(&mut count),
        ],
    )
}

fn mask_in_place(
    kernels: &MediaKernels,
    stream: &Stream,
    buffer: &mut DeviceBuffer<f32>,
    valid: &DeviceBuffer<u8>,
    count: usize,
) -> Result<(), BackendError> {
    if buffer.len() < count || valid.len() < count {
        return Err(infer("audio block mask buffers are too small"));
    }
    let mut input = buffer.cu_deviceptr();
    let mut valid = valid.cu_deviceptr();
    let mut out = buffer.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.mask,
        stream,
        blocks(count as usize)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut valid),
            arg(&mut out),
            arg(&mut count),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn im2col(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    channels: usize,
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
) -> Result<(), BackendError> {
    let rows = product(out_h, out_w)?;
    let columns = channels
        .checked_mul(AUDIO_CONV_KERNEL)
        .and_then(|value| value.checked_mul(AUDIO_CONV_KERNEL))
        .ok_or_else(|| infer("audio im2col column overflow"))?;
    let elements = product(rows, columns)?;
    if input.len() < product(channels, product(height, width)?)? || output.len() < elements {
        return Err(infer("audio im2col buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut channels = u32_size(channels)?;
    let mut height = u32_size(height)?;
    let mut width = u32_size(width)?;
    let mut kernel = u32_size(AUDIO_CONV_KERNEL)?;
    let mut stride = 2u32;
    let mut pad = 1u32;
    let mut out_h = u32_size(out_h)?;
    let mut out_w = u32_size(out_w)?;
    launch(
        &kernels.im2col,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut channels),
            arg(&mut height),
            arg(&mut width),
            arg(&mut kernel),
            arg(&mut kernel),
            arg(&mut stride),
            arg(&mut pad),
            arg(&mut pad),
            arg(&mut out_h),
            arg(&mut out_w),
        ],
    )
}

fn f32_gemm(
    blas: &Blas,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    input_width: usize,
    output_width: usize,
) -> Result<(), BackendError> {
    let plan = LinearPlan::<F32>::contiguous(rows, input_width, output_width).map_err(infer)?;
    // SAFETY: the caller retains every allocation and the engine binds the
    // context; the plan validates all three buffer extents.
    unsafe { blas.linear(&plan, input, weight, output) }.map_err(infer)
}

fn cast_f32_to_bf16(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<Bf16>,
    count: usize,
) -> Result<(), BackendError> {
    if input.len() < count || output.len() < count {
        return Err(infer("audio F32->BF16 cast buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.f32_to_bf16,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut out), arg(&mut count)],
    )
}

fn widen_bf16(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
    count: usize,
) -> Result<(), BackendError> {
    if input.len() < count || output.len() < count {
        return Err(infer("audio BF16->F32 widening buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut selected = 0u64;
    let mut count = u32_size(count)?;
    let mut width = count;
    launch(
        &kernels.bf16_to_f32,
        stream,
        blocks(count as usize)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut selected),
            arg(&mut count),
            arg(&mut width),
        ],
    )
}

fn channel_norm_relu(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let elements = product(rows, width)?;
    if input.len() < elements || output.len() < elements || scale.len() < width {
        return Err(infer("audio channel norm buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut scale = scale.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    let mut eps = eps;
    launch_threads(
        &kernels.channel_norm_relu,
        stream,
        (rows, 1, 1),
        256,
        0,
        &mut [
            arg(&mut input),
            arg(&mut scale),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    )
}

fn transpose_btwc_to_bctw(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let elements = product(rows, width)?;
    if input.len() < elements || output.len() < elements {
        return Err(infer("audio transpose buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.transpose_btwc_to_bctw,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

fn conv_rows_to_btcw(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    time: usize,
    spatial: usize,
    channels: usize,
) -> Result<(), BackendError> {
    let elements = product(time, product(spatial, channels)?)?;
    if input.len() < elements || output.len() < elements {
        return Err(infer("audio convolution reshape buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut time = u32_size(time)?;
    let mut spatial = u32_size(spatial)?;
    let mut channels = u32_size(channels)?;
    launch(
        &kernels.conv_rows_to_btcw,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut time),
            arg(&mut spatial),
            arg(&mut channels),
        ],
    )
}

fn residual_scale_add(
    kernels: &MediaKernels,
    stream: &Stream,
    destination: &mut DeviceBuffer<Bf16>,
    source: &DeviceBuffer<Bf16>,
    count: usize,
    scale: f32,
) -> Result<(), BackendError> {
    if destination.len() < count || source.len() < count {
        return Err(infer("audio residual scale buffers are too small"));
    }
    let mut destination = destination.cu_deviceptr();
    let mut source = source.cu_deviceptr();
    let mut count = u32_size(count)?;
    let mut scale = scale;
    launch(
        &kernels.residual_scale_add,
        stream,
        blocks(count as usize)?,
        0,
        &mut [
            arg(&mut destination),
            arg(&mut source),
            arg(&mut count),
            arg(&mut scale),
        ],
    )
}

fn add_bf16(
    kernels: &MediaKernels,
    stream: &Stream,
    destination: &mut DeviceBuffer<Bf16>,
    source: &DeviceBuffer<Bf16>,
    count: usize,
) -> Result<(), BackendError> {
    if destination.len() < count || source.len() < count {
        return Err(infer("audio residual add buffers are too small"));
    }
    let mut destination = destination.cu_deviceptr();
    let mut source = source.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.add,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut destination), arg(&mut source), arg(&mut count)],
    )
}

fn silu_in_place(
    kernels: &MediaKernels,
    stream: &Stream,
    buffer: &mut DeviceBuffer<Bf16>,
    count: usize,
) -> Result<(), BackendError> {
    if buffer.len() < count {
        return Err(infer("audio SiLU buffer is too small"));
    }
    let mut input = buffer.cu_deviceptr();
    let mut out = buffer.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.silu,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut out), arg(&mut count)],
    )
}

fn glu(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let elements = product(rows, width)?;
    let input_elements = product(rows, product(width, 2)?)?;
    if input.len() < input_elements || output.len() < elements {
        return Err(infer("audio GLU buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.glu,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

fn depthwise_conv1d(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    weight: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    tokens: usize,
    channels: usize,
    kernel: usize,
    clip: f32,
) -> Result<(), BackendError> {
    let elements = product(tokens, channels)?;
    let weights = product(channels, kernel)?;
    if input.len() < elements || output.len() < elements || weight.len() < weights {
        return Err(infer("audio depthwise convolution buffers are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut weight = weight.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut tokens = u32_size(tokens)?;
    let mut channels = u32_size(channels)?;
    let mut kernel = u32_size(kernel)?;
    let mut clip = clip;
    launch(
        &kernels.depthwise_conv1d,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut weight),
            arg(&mut out),
            arg(&mut tokens),
            arg(&mut channels),
            arg(&mut kernel),
            arg(&mut clip),
        ],
    )
}

fn scaled_qk(
    kernels: &MediaKernels,
    stream: &Stream,
    q: &DeviceBuffer<Bf16>,
    k: &DeviceBuffer<Bf16>,
    per_dim_scale: &DeviceBuffer<Bf16>,
    qk_scaled: &mut DeviceBuffer<f32>,
    tokens: usize,
    heads: usize,
    dim: usize,
) -> Result<(), BackendError> {
    let elements = product(tokens, product(heads, dim)?)?;
    let scaled = elements
        .checked_mul(2)
        .ok_or_else(|| infer("audio scaled QK length overflow"))?;
    if q.len() < elements
        || k.len() < elements
        || per_dim_scale.len() < dim
        || qk_scaled.len() < scaled
    {
        return Err(infer("audio scaled QK buffers are too small"));
    }
    let mut q = q.cu_deviceptr();
    let mut k = k.cu_deviceptr();
    let mut per_dim_scale = per_dim_scale.cu_deviceptr();
    let mut q_out = qk_scaled.cu_deviceptr();
    let offset = u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| infer("audio scaled QK byte overflow"))?,
    )
    .map_err(|_| infer("audio scaled QK offset exceeds u64"))?;
    let mut k_out = q_out
        .checked_add(offset)
        .ok_or_else(|| infer("audio scaled K offset overflow"))?;
    let mut tokens = u32_size(tokens)?;
    let mut heads = u32_size(heads)?;
    let mut dim = u32_size(dim)?;
    launch(
        &kernels.audio_scaled_qk,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut q),
            arg(&mut k),
            arg(&mut per_dim_scale),
            arg(&mut q_out),
            arg(&mut k_out),
            arg(&mut tokens),
            arg(&mut heads),
            arg(&mut dim),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn audio_attention(
    kernels: &MediaKernels,
    stream: &Stream,
    qk_scaled: &DeviceBuffer<f32>,
    v: &DeviceBuffer<Bf16>,
    relative: &DeviceBuffer<Bf16>,
    valid: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<Bf16>,
    tokens: usize,
    heads: usize,
    dim: usize,
    chunk: usize,
    past: usize,
    future: usize,
    span: usize,
    cap: f32,
    invalid: f32,
) -> Result<(), BackendError> {
    let elements = product(tokens, product(heads, dim)?)?;
    let scaled = elements
        .checked_mul(2)
        .ok_or_else(|| infer("audio attention QK length overflow"))?;
    let relative_elements = product(span, product(heads, dim)?)?;
    let chunks = ceil_div(tokens, chunk)?;
    let work = product(chunks, product(chunk, heads)?)?;
    if qk_scaled.len() < scaled
        || v.len() < elements
        || relative.len() < relative_elements
        || valid.len() < tokens
        || output.len() < elements
    {
        return Err(infer("audio attention buffers are too small"));
    }
    let mut q = qk_scaled.cu_deviceptr();
    let offset = u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| infer("audio attention byte overflow"))?,
    )
    .map_err(|_| infer("audio attention offset exceeds u64"))?;
    let mut k = q
        .checked_add(offset)
        .ok_or_else(|| infer("audio attention K offset overflow"))?;
    let mut v = v.cu_deviceptr();
    let mut relative = relative.cu_deviceptr();
    let mut valid = valid.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut tokens = u32_size(tokens)?;
    let mut heads = u32_size(heads)?;
    let mut dim = u32_size(dim)?;
    let mut chunk = u32_size(chunk)?;
    let mut past = u32_size(past)?;
    let mut future = u32_size(future)?;
    let mut span = u32_size(span)?;
    let mut cap = cap;
    let mut invalid = invalid;
    launch(
        &kernels.audio_attention,
        stream,
        blocks(work)?,
        0,
        &mut [
            arg(&mut q),
            arg(&mut k),
            arg(&mut v),
            arg(&mut relative),
            arg(&mut valid),
            arg(&mut out),
            arg(&mut tokens),
            arg(&mut heads),
            arg(&mut dim),
            arg(&mut chunk),
            arg(&mut past),
            arg(&mut future),
            arg(&mut span),
            arg(&mut cap),
            arg(&mut invalid),
        ],
    )
}

fn select_rows(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    selected: &DeviceBuffer<u32>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let elements = product(rows, width)?;
    if output.len() < elements || selected.len() < rows {
        return Err(infer("audio selected rows are too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut selected = selected.cu_deviceptr();
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.select_rows,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut selected),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

impl AudioScratch {
    pub(crate) fn device_bytes(&self) -> usize {
        let mut total = 0usize;
        macro_rules! add {
            ($field:ident) => {
                total = total.saturating_add(self.$field.num_bytes());
            };
        }
        add!(features_f32);
        add!(masked_f32);
        add!(valid);
        add!(valid_block1);
        add!(valid_frames);
        add!(conv0_columns);
        add!(conv0_filter);
        add!(conv_rows_f32);
        add!(conv0_bf16);
        add!(conv0_transposed);
        add!(conv1_columns);
        add!(conv1_filter);
        add!(conv1_bf16);
        add!(conv1_btcw);
        add!(states);
        add!(next);
        add!(normed);
        add!(clamped);
        add!(gate);
        add!(mlp);
        add!(mlp_norm);
        add!(q);
        add!(k);
        add!(v);
        add!(attn);
        add!(post);
        add!(qk_scaled);
        add!(relative);
        add!(signal);
        add!(gate_start);
        add!(glu);
        add!(depthwise);
        add!(end);
        add!(projected);
        add!(projected_norm);
        add!(output);
        if let Some(selected) = self.selected.as_ref() {
            total = total.saturating_add(selected.num_bytes());
        }
        if let Some(indices) = self.selected_indices.as_ref() {
            total = total.saturating_add(indices.num_bytes());
        }
        total
    }
}

impl AudioRuntime {
    pub(crate) fn load(
        uploader: &mut WeightUploader,
        stream: &Arc<Stream>,
        host: &AudioHost,
        config: &AudioConfig,
    ) -> Result<Self, BackendError> {
        if config.activation != AudioActivation::Silu {
            return Err(start("Gemma4 audio activation must be silu"));
        }
        if host.layers.len() != config.layers {
            return Err(start("audio host layer count disagrees with the config"));
        }
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let projected = config.projected_width()?;
        let conv0 = [
            config.ss_channels[0],
            1,
            AUDIO_CONV_KERNEL,
            AUDIO_CONV_KERNEL,
        ];
        let conv1 = [
            config.ss_channels[1],
            config.ss_channels[0],
            AUDIO_CONV_KERNEL,
            AUDIO_CONV_KERNEL,
        ];
        check_linear(
            &host.input_projection,
            "subsample_conv_projection.input_proj_linear",
            [h, projected],
            false,
            false,
        )?;
        check_linear(
            &host.output_projection,
            "audio_tower.output_proj",
            [config.output_proj_dims, h],
            true,
            false,
        )?;
        check_linear(
            &host.embed_projection,
            "embed_audio.embedding_projection",
            [config.output_width, config.output_proj_dims],
            false,
            false,
        )?;
        if host.subsample0_conv.shape() != conv0.as_slice()
            || host.subsample1_conv.shape() != conv1.as_slice()
            || host.subsample0_norm.shape() != [config.ss_channels[0]].as_slice()
            || host.subsample1_norm.shape() != [config.ss_channels[1]].as_slice()
        {
            return Err(start("audio subsampler weights disagree with the config"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers)
            .map_err(|_| start("audio layer allocation failed"))?;
        for (index, layer) in host.layers.iter().enumerate() {
            let name = format!("audio_tower.layers.{index}");
            let mut mlp = |mlp: &crate::gemma2_media::AudioMlpHost,
                           name: &str|
             -> Result<AudioMlpRuntime, BackendError> {
                check_linear(
                    &mlp.first,
                    &format!("{name}.ffw_layer_1"),
                    [i, h],
                    false,
                    config.use_clipped_linears,
                )?;
                check_linear(
                    &mlp.second,
                    &format!("{name}.ffw_layer_2"),
                    [h, i],
                    false,
                    config.use_clipped_linears,
                )?;
                Ok(AudioMlpRuntime {
                    pre_norm: upload_norm_source(stream, &mlp.pre_norm, h)?,
                    first: upload_linear(uploader, &mlp.first, &format!("{name}.ffw_layer_1"))?,
                    second: upload_linear(uploader, &mlp.second, &format!("{name}.ffw_layer_2"))?,
                    post_norm: upload_norm_source(stream, &mlp.post_norm, h)?,
                })
            };
            check_linear(
                &layer.q,
                &format!("{name}.self_attn.q_proj"),
                [h, h],
                false,
                config.use_clipped_linears,
            )?;
            check_linear(
                &layer.k,
                &format!("{name}.self_attn.k_proj"),
                [h, h],
                false,
                config.use_clipped_linears,
            )?;
            check_linear(
                &layer.v,
                &format!("{name}.self_attn.v_proj"),
                [h, h],
                false,
                config.use_clipped_linears,
            )?;
            check_linear(
                &layer.relative_k,
                &format!("{name}.self_attn.relative_k_proj"),
                [h, h],
                false,
                false,
            )?;
            check_linear(
                &layer.post,
                &format!("{name}.self_attn.post"),
                [h, h],
                false,
                config.use_clipped_linears,
            )?;
            check_linear(
                &layer.linear_start,
                &format!("{name}.lconv1d.linear_start"),
                [2 * h, h],
                false,
                config.use_clipped_linears,
            )?;
            check_linear(
                &layer.linear_end,
                &format!("{name}.lconv1d.linear_end"),
                [h, h],
                false,
                config.use_clipped_linears,
            )?;
            if layer.per_dim_scale.shape() != [config.head_dim].as_slice()
                || layer.depthwise_conv1d.shape() != [h, 1, config.conv_kernel].as_slice()
            {
                return Err(start(format!(
                    "audio layer {index} scaling tensors disagree with the config"
                )));
            }
            layers.push(AudioLayerRuntime {
                feed_forward1: mlp(&layer.feed_forward1, &format!("{name}.feed_forward1"))?,
                feed_forward2: mlp(&layer.feed_forward2, &format!("{name}.feed_forward2"))?,
                norm_pre_attn: upload_norm_source(stream, &layer.norm_pre_attn, h)?,
                norm_post_attn: upload_norm_source(stream, &layer.norm_post_attn, h)?,
                norm_out: upload_norm_source(stream, &layer.norm_out, h)?,
                q: upload_linear(uploader, &layer.q, &format!("{name}.self_attn.q_proj"))?,
                k: upload_linear(uploader, &layer.k, &format!("{name}.self_attn.k_proj"))?,
                v: upload_linear(uploader, &layer.v, &format!("{name}.self_attn.v_proj"))?,
                relative_k: upload_linear(
                    uploader,
                    &layer.relative_k,
                    &format!("{name}.self_attn.relative_k_proj"),
                )?,
                post: upload_linear(uploader, &layer.post, &format!("{name}.self_attn.post"))?,
                per_dim_scale: upload_bf16(
                    uploader,
                    &layer.per_dim_scale,
                    &format!("{name}.self_attn.per_dim_scale"),
                )?,
                lconv_pre_norm: upload_norm_source(stream, &layer.lconv_pre_norm, h)?,
                linear_start: upload_linear(
                    uploader,
                    &layer.linear_start,
                    &format!("{name}.lconv1d.linear_start"),
                )?,
                depthwise_conv1d: upload_bf16(
                    uploader,
                    &layer.depthwise_conv1d,
                    &format!("{name}.lconv1d.depthwise_conv1d.weight"),
                )?,
                conv_norm: upload_norm_source(stream, &layer.conv_norm, h)?,
                linear_end: upload_linear(
                    uploader,
                    &layer.linear_end,
                    &format!("{name}.lconv1d.linear_end"),
                )?,
            });
        }
        Ok(Self {
            config: config.clone(),
            subsample0_conv: upload_bf16(
                uploader,
                &host.subsample0_conv,
                "audio_tower.subsample_conv_projection.layer0.conv.weight",
            )?,
            subsample0_norm: upload_bf16(
                uploader,
                &host.subsample0_norm,
                "audio_tower.subsample_conv_projection.layer0.norm.weight",
            )?,
            subsample1_conv: upload_bf16(
                uploader,
                &host.subsample1_conv,
                "audio_tower.subsample_conv_projection.layer1.conv.weight",
            )?,
            subsample1_norm: upload_bf16(
                uploader,
                &host.subsample1_norm,
                "audio_tower.subsample_conv_projection.layer1.norm.weight",
            )?,
            input_projection: upload_linear(
                uploader,
                &host.input_projection,
                "audio_tower.subsample_conv_projection.input_proj_linear",
            )?,
            output_projection: upload_linear(
                uploader,
                &host.output_projection,
                "audio_tower.output_proj",
            )?,
            embed_projection: upload_linear(
                uploader,
                &host.embed_projection,
                "embed_audio.embedding_projection",
            )?,
            layers,
        })
    }

    pub(crate) fn scratch(
        &self,
        stream: &Arc<Stream>,
        frames: usize,
    ) -> Result<AudioScratch, BackendError> {
        if frames == 0 {
            return Err(infer("empty audio scratch frames"));
        }
        let config = &self.config;
        let stride0 = config.ss_strides[0];
        let stride1 = config.ss_strides[1];
        let t1 = ceil_div(frames, stride0)?;
        let t2 = ceil_div(t1, stride1)?;
        let w0 = ceil_div(AUDIO_FEATURE_SIZE, stride0)?;
        let w1 = ceil_div(w0, stride1)?;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let projected_width = config.projected_width()?;
        let output_dims = config.output_proj_dims;
        let output_width = config.output_width;
        let span = config.relative_span;
        let rows0 = product(t1, w0)?;
        let rows1 = product(t2, w1)?;
        let conv0 = product(rows0, config.ss_channels[0])?;
        let conv1 = product(rows1, config.ss_channels[1])?;
        let conv1_width = config.ss_channels[0]
            .checked_mul(9)
            .ok_or_else(|| infer("audio conv1 input width overflow"))?;
        let conv0_columns = product(
            rows0,
            config.ss_channels[0]
                .checked_mul(9)
                .ok_or_else(|| infer("audio conv0 columns overflow"))?,
        )?;
        let conv1_columns = product(rows1, conv1_width)?;
        let token_hidden = product(t2, hidden)?;
        let token_gate = product(t2, 2 * hidden)?;
        let token_intermediate = product(t2, intermediate)?;
        let token_projected = product(t2, output_dims)?;
        let token_output = product(t2, output_width)?;
        let qk = product(token_hidden, 2)?;
        let signal = product(span, hidden)?;
        Ok(AudioScratch {
            frames,
            features_f32: scratch_typed(stream, product(frames, AUDIO_FEATURE_SIZE)?, "features")?,
            masked_f32: scratch_typed(stream, product(frames, AUDIO_FEATURE_SIZE)?, "masked")?,
            valid: scratch_typed(stream, product(frames, AUDIO_FEATURE_SIZE)?, "valid")?,
            valid_block1: scratch_typed(
                stream,
                product(config.ss_channels[0], conv0)?,
                "valid_block1",
            )?,
            valid_frames: scratch_typed(stream, t2, "valid_frames")?,
            conv0_columns: scratch_typed(stream, conv0_columns, "conv0_columns")?,
            conv0_filter: scratch_typed(
                stream,
                product(config.ss_channels[0], AUDIO_CONV_KERNEL * AUDIO_CONV_KERNEL)?,
                "conv0_filter",
            )?,
            conv_rows_f32: scratch_typed(stream, conv0.max(conv1), "conv_rows")?,
            conv0_bf16: scratch_typed(stream, conv0, "conv0_bf16")?,
            conv0_transposed: scratch_typed(stream, conv0, "conv0_transposed")?,
            conv1_columns: scratch_typed(stream, conv1_columns, "conv1_columns")?,
            conv1_filter: scratch_typed(
                stream,
                product(
                    config.ss_channels[1],
                    config.ss_channels[0]
                        .checked_mul(9)
                        .ok_or_else(|| infer("audio conv1 filter overflow"))?,
                )?,
                "conv1_filter",
            )?,
            conv1_bf16: scratch_typed(stream, conv1, "conv1_bf16")?,
            conv1_btcw: scratch_typed(stream, product(t2, projected_width)?, "conv1_btcw")?,
            states: scratch_typed(stream, token_hidden, "states")?,
            next: scratch_typed(stream, token_hidden, "next")?,
            normed: scratch_typed(stream, token_hidden, "normed")?,
            clamped: scratch_typed(stream, token_intermediate, "clamped")?,
            gate: scratch_typed(stream, token_intermediate, "gate")?,
            mlp: scratch_typed(stream, token_hidden, "mlp")?,
            mlp_norm: scratch_typed(stream, token_hidden, "mlp_norm")?,
            q: scratch_typed(stream, token_hidden, "q")?,
            k: scratch_typed(stream, token_hidden, "k")?,
            v: scratch_typed(stream, token_hidden, "v")?,
            attn: scratch_typed(stream, token_hidden, "attn")?,
            post: scratch_typed(stream, token_hidden, "post")?,
            qk_scaled: scratch_typed(stream, qk, "qk_scaled")?,
            relative: scratch_typed(stream, signal, "relative")?,
            signal: scratch_typed(stream, signal, "signal")?,
            gate_start: scratch_typed(stream, token_gate, "gate_start")?,
            glu: scratch_typed(stream, token_hidden, "glu")?,
            depthwise: scratch_typed(stream, token_hidden, "depthwise")?,
            end: scratch_typed(stream, token_hidden, "end")?,
            projected: scratch_typed(stream, token_projected, "projected")?,
            projected_norm: scratch_typed(stream, token_projected, "projected_norm")?,
            output: scratch_typed(stream, token_output, "output")?,
            selected: None,
            selected_indices: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        kernels: &MediaKernels,
        blas: &Blas,
        stream: &Arc<Stream>,
        scratch: &mut AudioScratch,
        descriptor: &AudioDescriptor,
        features: &[f32],
        mask: &[u8],
    ) -> Result<DeviceBuffer<Bf16>, BackendError> {
        let geometry = validate_audio(&self.config, descriptor, features.len(), mask.len())?;
        if geometry.frames > scratch.frames {
            return Err(infer("audio scratch is smaller than the requested span"));
        }
        let config = &self.config;
        let frames = geometry.frames;
        let t1 = geometry.t1;
        let t2 = geometry.t2;
        let hidden = geometry.hidden;
        let intermediate = geometry.intermediate;
        let projected_width = geometry.projected_width;
        let output_dims = geometry.output_dims;
        let output_width = geometry.output_width;
        let head_dim = config.head_dim;
        let heads = config.heads;
        let eps = config.rms_norm_eps;
        let rows0 = product(t1, geometry.w0)?;
        let rows1 = product(t2, geometry.w1)?;
        let conv0 = product(rows0, config.ss_channels[0])?;
        let conv1 = product(rows1, config.ss_channels[1])?;
        let token_hidden = product(t2, hidden)?;
        let token_intermediate = product(t2, intermediate)?;

        // Native states = features.to(BF16); the convolution then widens the
        // masked values back to F32.
        let mut host_features = Vec::new();
        host_features
            .try_reserve_exact(features.len())
            .map_err(|_| infer("audio feature staging allocation failed"))?;
        for &value in features {
            host_features.push(half::bf16::from_f32(value).to_f32());
        }
        scratch.features_f32 = DeviceBuffer::from_host(stream, &host_features)
            .map_err(|error| start(format!("audio feature upload: {error}")))?;

        let mut host_valid = Vec::new();
        host_valid
            .try_reserve_exact(product(frames, AUDIO_FEATURE_SIZE)?)
            .map_err(|_| infer("audio validity allocation failed"))?;
        for &value in mask {
            for _ in 0..AUDIO_FEATURE_SIZE {
                host_valid.push(value);
            }
        }
        scratch.valid = DeviceBuffer::from_host(stream, &host_valid)
            .map_err(|error| start(format!("audio validity upload: {error}")))?;
        mask_into(
            kernels,
            stream,
            &scratch.features_f32,
            &scratch.valid,
            &mut scratch.masked_f32,
            product(frames, AUDIO_FEATURE_SIZE)?,
        )?;

        expand_bf16_to_f32(
            kernels,
            stream,
            &self.subsample0_conv,
            &mut scratch.conv0_filter,
        )?;
        im2col(
            kernels,
            stream,
            &scratch.masked_f32,
            &mut scratch.conv0_columns,
            1,
            frames,
            AUDIO_FEATURE_SIZE,
            t1,
            geometry.w0,
        )?;
        f32_gemm(
            blas,
            &scratch.conv0_columns,
            &scratch.conv0_filter,
            &mut scratch.conv_rows_f32,
            rows0,
            9,
            config.ss_channels[0],
        )?;
        cast_f32_to_bf16(
            kernels,
            stream,
            &scratch.conv_rows_f32,
            &mut scratch.conv0_bf16,
            conv0,
        )?;
        widen_bf16(
            kernels,
            stream,
            &scratch.conv0_bf16,
            &mut scratch.conv_rows_f32,
            conv0,
        )?;
        channel_norm_relu(
            kernels,
            stream,
            &scratch.conv_rows_f32,
            &self.subsample0_norm,
            &mut scratch.conv0_bf16,
            rows0,
            config.ss_channels[0],
            eps,
        )?;
        transpose_btwc_to_bctw(
            kernels,
            stream,
            &scratch.conv0_bf16,
            &mut scratch.conv0_transposed,
            rows0,
            config.ss_channels[0],
        )?;

        let valid0 = subsample_stage(mask, config.ss_strides[0])?;
        if valid0.len() != t1 {
            return Err(infer(
                "audio first-stage validity length disagrees with the subsampler",
            ));
        }
        let mut host_block1 = Vec::new();
        host_block1
            .try_reserve_exact(conv0)
            .map_err(|_| infer("audio block validity allocation failed"))?;
        for _ in 0..config.ss_channels[0] {
            for &value in &valid0 {
                for _ in 0..geometry.w0 {
                    host_block1.push(value);
                }
            }
        }
        scratch.valid_block1 = DeviceBuffer::from_host(stream, &host_block1)
            .map_err(|error| start(format!("audio block validity upload: {error}")))?;
        widen_bf16(
            kernels,
            stream,
            &scratch.conv0_transposed,
            &mut scratch.conv_rows_f32,
            conv0,
        )?;
        mask_in_place(
            kernels,
            stream,
            &mut scratch.conv_rows_f32,
            &scratch.valid_block1,
            conv0,
        )?;
        im2col(
            kernels,
            stream,
            &scratch.conv_rows_f32,
            &mut scratch.conv1_columns,
            config.ss_channels[0],
            t1,
            geometry.w0,
            t2,
            geometry.w1,
        )?;
        expand_bf16_to_f32(
            kernels,
            stream,
            &self.subsample1_conv,
            &mut scratch.conv1_filter,
        )?;
        f32_gemm(
            blas,
            &scratch.conv1_columns,
            &scratch.conv1_filter,
            &mut scratch.conv_rows_f32,
            rows1,
            config.ss_channels[0]
                .checked_mul(9)
                .ok_or_else(|| infer("audio conv1 input width overflow"))?,
            config.ss_channels[1],
        )?;
        cast_f32_to_bf16(
            kernels,
            stream,
            &scratch.conv_rows_f32,
            &mut scratch.conv1_bf16,
            conv1,
        )?;
        widen_bf16(
            kernels,
            stream,
            &scratch.conv1_bf16,
            &mut scratch.conv_rows_f32,
            conv1,
        )?;
        channel_norm_relu(
            kernels,
            stream,
            &scratch.conv_rows_f32,
            &self.subsample1_norm,
            &mut scratch.conv1_bf16,
            rows1,
            config.ss_channels[1],
            eps,
        )?;
        conv_rows_to_btcw(
            kernels,
            stream,
            &scratch.conv1_bf16,
            &mut scratch.conv1_btcw,
            t2,
            geometry.w1,
            config.ss_channels[1],
        )?;
        project(
            kernels,
            blas,
            stream,
            &self.input_projection,
            &scratch.conv1_btcw,
            &mut scratch.states,
            t2,
            projected_width,
            &mut scratch.clamped,
        )?;

        let signal = relative_signal(config)?;
        if signal.len() != product(config.relative_span, hidden)? {
            return Err(infer(
                "audio relative signal shape disagrees with the config",
            ));
        }
        scratch.signal = DeviceBuffer::from_host(stream, &signal)
            .map_err(|error| start(format!("audio signal upload: {error}")))?;

        let mut valid_frames = vec![0u8; t2];
        for &frame in &descriptor.subsampled_frames {
            let slot = valid_frames
                .get_mut(frame)
                .ok_or_else(|| infer("audio selected frame is outside scratch"))?;
            *slot = 1;
        }
        scratch.valid_frames = DeviceBuffer::from_host(stream, &valid_frames)
            .map_err(|error| start(format!("audio frame validity upload: {error}")))?;

        for layer in &self.layers {
            audio_norm_rows(
                kernels,
                stream,
                &scratch.states,
                &layer.feed_forward1.pre_norm,
                &mut scratch.normed,
                t2,
                hidden,
                eps,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.feed_forward1.first,
                &scratch.normed,
                &mut scratch.gate,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            silu_in_place(kernels, stream, &mut scratch.gate, token_intermediate)?;
            project(
                kernels,
                blas,
                stream,
                &layer.feed_forward1.second,
                &scratch.gate,
                &mut scratch.mlp,
                t2,
                intermediate,
                &mut scratch.clamped,
            )?;
            audio_norm_rows(
                kernels,
                stream,
                &scratch.mlp,
                &layer.feed_forward1.post_norm,
                &mut scratch.mlp_norm,
                t2,
                hidden,
                eps,
            )?;
            residual_scale_add(
                kernels,
                stream,
                &mut scratch.states,
                &scratch.mlp_norm,
                token_hidden,
                config.residual_weight,
            )?;

            audio_norm_rows(
                kernels,
                stream,
                &scratch.states,
                &layer.norm_pre_attn,
                &mut scratch.normed,
                t2,
                hidden,
                eps,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.q,
                &scratch.normed,
                &mut scratch.q,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.k,
                &scratch.normed,
                &mut scratch.k,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.v,
                &scratch.normed,
                &mut scratch.v,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            scaled_qk(
                kernels,
                stream,
                &scratch.q,
                &scratch.k,
                &layer.per_dim_scale,
                &mut scratch.qk_scaled,
                t2,
                heads,
                head_dim,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.relative_k,
                &scratch.signal,
                &mut scratch.relative,
                config.relative_span,
                hidden,
                &mut scratch.clamped,
            )?;
            audio_attention(
                kernels,
                stream,
                &scratch.qk_scaled,
                &scratch.v,
                &scratch.relative,
                &scratch.valid_frames,
                &mut scratch.attn,
                t2,
                heads,
                head_dim,
                config.chunk,
                config.past,
                config.context_right,
                config.relative_span,
                config.logit_cap,
                config.invalid_logits,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.post,
                &scratch.attn,
                &mut scratch.post,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            audio_norm_rows(
                kernels,
                stream,
                &scratch.post,
                &layer.norm_post_attn,
                &mut scratch.mlp_norm,
                t2,
                hidden,
                eps,
            )?;
            add_bf16(
                kernels,
                stream,
                &mut scratch.states,
                &scratch.mlp_norm,
                token_hidden,
            )?;

            audio_norm_rows(
                kernels,
                stream,
                &scratch.states,
                &layer.lconv_pre_norm,
                &mut scratch.normed,
                t2,
                hidden,
                eps,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.linear_start,
                &scratch.normed,
                &mut scratch.gate_start,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            glu(
                kernels,
                stream,
                &scratch.gate_start,
                &mut scratch.glu,
                t2,
                hidden,
            )?;
            depthwise_conv1d(
                kernels,
                stream,
                &scratch.glu,
                &layer.depthwise_conv1d,
                &mut scratch.depthwise,
                t2,
                hidden,
                config.conv_kernel,
                config.gradient_clipping,
            )?;
            audio_norm_rows(
                kernels,
                stream,
                &scratch.depthwise,
                &layer.conv_norm,
                &mut scratch.normed,
                t2,
                hidden,
                eps,
            )?;
            silu_in_place(kernels, stream, &mut scratch.normed, token_hidden)?;
            project(
                kernels,
                blas,
                stream,
                &layer.linear_end,
                &scratch.normed,
                &mut scratch.end,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            add_bf16(
                kernels,
                stream,
                &mut scratch.states,
                &scratch.end,
                token_hidden,
            )?;

            audio_norm_rows(
                kernels,
                stream,
                &scratch.states,
                &layer.feed_forward2.pre_norm,
                &mut scratch.normed,
                t2,
                hidden,
                eps,
            )?;
            project(
                kernels,
                blas,
                stream,
                &layer.feed_forward2.first,
                &scratch.normed,
                &mut scratch.gate,
                t2,
                hidden,
                &mut scratch.clamped,
            )?;
            silu_in_place(kernels, stream, &mut scratch.gate, token_intermediate)?;
            project(
                kernels,
                blas,
                stream,
                &layer.feed_forward2.second,
                &scratch.gate,
                &mut scratch.mlp,
                t2,
                intermediate,
                &mut scratch.clamped,
            )?;
            audio_norm_rows(
                kernels,
                stream,
                &scratch.mlp,
                &layer.feed_forward2.post_norm,
                &mut scratch.mlp_norm,
                t2,
                hidden,
                eps,
            )?;
            residual_scale_add(
                kernels,
                stream,
                &mut scratch.states,
                &scratch.mlp_norm,
                token_hidden,
                config.residual_weight,
            )?;
            audio_norm_rows(
                kernels,
                stream,
                &scratch.states,
                &layer.norm_out,
                &mut scratch.next,
                t2,
                hidden,
                eps,
            )?;
            std::mem::swap(&mut scratch.states, &mut scratch.next);
        }

        project(
            kernels,
            blas,
            stream,
            &self.output_projection,
            &scratch.states,
            &mut scratch.projected,
            t2,
            hidden,
            &mut scratch.clamped,
        )?;
        unit_norm_heads(
            kernels,
            stream,
            &scratch.projected,
            &mut scratch.projected_norm,
            t2,
            output_dims,
            1,
            output_dims,
            eps,
        )?;
        project(
            kernels,
            blas,
            stream,
            &self.embed_projection,
            &scratch.projected_norm,
            &mut scratch.output,
            t2,
            output_dims,
            &mut scratch.clamped,
        )?;

        let mut indices = Vec::new();
        indices
            .try_reserve_exact(descriptor.token_count)
            .map_err(|_| infer("audio selected frame allocation failed"))?;
        for &frame in &descriptor.subsampled_frames {
            indices.push(u32::try_from(frame).map_err(|_| infer("audio frame exceeds u32"))?);
        }
        scratch.selected_indices = Some(
            DeviceBuffer::from_host(stream, &indices)
                .map_err(|error| start(format!("audio frame index upload: {error}")))?,
        );
        let selected_len = product(descriptor.token_count, output_width)?;
        let mut selected = match scratch.selected.take() {
            Some(buffer) if buffer.len() == selected_len => buffer,
            _ => {
                // SAFETY: select_rows writes every selected row before it is copied.
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, selected_len) }
                    .map_err(start)?
            }
        };
        let selected_indices = scratch
            .selected_indices
            .as_ref()
            .ok_or_else(|| infer("audio selected frame indices disappeared"))?;
        select_rows(
            kernels,
            stream,
            &scratch.output,
            &mut selected,
            selected_indices,
            descriptor.token_count,
            output_width,
        )?;
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::media::{AudioLayoutPolicy, PackedSequences};
    use infinity_v3_backend_core::AudioFeatures;

    fn audio_config() -> AudioConfig {
        AudioConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
            layers: 12,
            heads: 8,
            head_dim: 128,
            chunk: 12,
            context_left: 13,
            context_right: 0,
            past: 12,
            relative_span: 13,
            residual_weight: 0.5,
            rms_norm_eps: 1e-6,
            logit_cap: 50.0,
            invalid_logits: -1e9,
            gradient_clipping: 1e10,
            conv_kernel: 5,
            ss_channels: [128, 32],
            ss_strides: [2, 2],
            output_proj_dims: 1536,
            use_clipped_linears: true,
            activation: AudioActivation::Silu,
            output_width: 512,
        }
    }

    fn descriptor(frames: usize) -> AudioDescriptor {
        let values = vec![0.0f32; frames * AUDIO_FEATURE_SIZE];
        let mask = vec![1u8; frames];
        let features = AudioFeatures {
            values,
            mask,
            feature_size: AUDIO_FEATURE_SIZE,
            subsample_strides: [2, 2],
        };
        let policy = AudioLayoutPolicy::for_embedding_gemma2(128, [2, 2]).expect("policy");
        let tokens = features.token_count();
        let bounds = [0u32, tokens as u32];
        let sequences = PackedSequences::new(&bounds).expect("sequences");
        AudioDescriptor::new(
            MediaFamily::EmbeddingGemma2,
            &features,
            0,
            &sequences,
            &policy,
        )
        .expect("descriptor")
    }

    #[test]
    fn audio_subsampling_geometry_is_native() {
        let config = audio_config();
        let descriptor = descriptor(100);
        let geometry = validate_audio(
            &config,
            &descriptor,
            descriptor.frames * AUDIO_FEATURE_SIZE,
            descriptor.frames,
        )
        .expect("geometry");
        assert_eq!(geometry.t1, 50);
        assert_eq!(geometry.t2, 25);
        assert_eq!(geometry.w0, 64);
        assert_eq!(geometry.w1, 32);
        assert_eq!(config.subsampled_width().expect("width"), 32);
        assert_eq!(config.projected_width().expect("projected"), 1024);
        assert_eq!(descriptor.token_count, 25);
    }

    #[test]
    fn audio_relative_signal_matches_the_native_shape() {
        let config = audio_config();
        assert_eq!(config.relative_span, 13);
        let signal = relative_signal(&config).expect("signal");
        assert_eq!(signal.len(), 13 * 1024);
        let first = half::bf16::from_bits(signal[0].to_bits()).to_f32();
        let expected = half::bf16::from_f32(12.0f32.sin()).to_f32();
        assert_eq!(first, expected);
        let first_cosine = half::bf16::from_bits(signal[512].to_bits()).to_f32();
        let expected_cosine = half::bf16::from_f32(12.0f32.cos()).to_f32();
        assert_eq!(first_cosine, expected_cosine);
    }

    #[test]
    fn audio_geometry_rejects_wrong_feature_and_mask_lengths() {
        let config = audio_config();
        let descriptor = descriptor(100);
        assert!(validate_audio(
            &config,
            &descriptor,
            descriptor.frames * AUDIO_FEATURE_SIZE - 1,
            descriptor.frames,
        )
        .is_err());
        assert!(validate_audio(
            &config,
            &descriptor,
            descriptor.frames * AUDIO_FEATURE_SIZE,
            descriptor.frames - 1,
        )
        .is_err());
    }
}
