// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold media executor: owned tower weights, per-request scratch, and the
//! packed-row handoff into the text executor's token state.
//!
//! The towers run before the text layers, exactly like the Native `prepare`
//! phase. This module owns every allocation it creates; `PreparedMedia` must be
//! retained until the text stream has drained. Tower math is device-only; cold
//! media payload/fixture uploads use the existing synchronized `from_host`
//! staging path. No CPU math fallback or device query occurs here.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::cublas::Blas;
use crate::gemma2_media::{MediaHost, MediaRequest, MediaSpanKind};
use crate::gemma2_media_prepared::upload_norm;
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::{arg, blocks, infer, launch, launch_threads, product, start, u32_size};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) const MEDIA_KERNEL_EXPORTS: [&str; 24] = [
    "infinity_gemma2_media_im2col_f32",
    "infinity_gemma2_media_mask_f32",
    "infinity_gemma2_media_channel_norm_relu_bf16",
    "infinity_gemma2_media_transpose_btwc_to_bctw_bf16",
    "infinity_gemma2_media_conv_rows_to_btcw_bf16",
    "infinity_gemma2_media_center_pixels_bf16",
    "infinity_gemma2_media_patch_add_position_bf16",
    "infinity_gemma2_media_axial_tables_bf16",
    "infinity_gemma2_media_axial_rope_bf16",
    "infinity_gemma2_media_attention_bf16",
    "infinity_gemma2_media_gated_gelu_split_bf16",
    "infinity_gemma2_media_vision_pool_bf16",
    "infinity_gemma2_media_clamp_bf16",
    "infinity_gemma2_media_residual_scale_add_bf16",
    "infinity_gemma2_media_add_bf16",
    "infinity_gemma2_media_bias_add_bf16",
    "infinity_gemma2_media_audio_scaled_qk_f32",
    "infinity_gemma2_media_audio_attention_bf16",
    "infinity_gemma2_media_glu_bf16",
    "infinity_gemma2_media_depthwise_conv1d_causal_bf16",
    "infinity_gemma2_media_silu_bf16",
    "infinity_gemma2_media_select_rows_bf16",
    "infinity_gemma2_media_scatter_rows_bf16",
    "infinity_gemma2_media_audio_norm_bf16_f32scale",
];

pub(crate) struct MediaKernels {
    pub im2col: Function,
    pub mask: Function,
    pub channel_norm_relu: Function,
    pub transpose_btwc_to_bctw: Function,
    pub conv_rows_to_btcw: Function,
    pub center_pixels: Function,
    pub patch_add_position: Function,
    pub axial_tables: Function,
    pub axial_rope: Function,
    pub attention: Function,
    pub gated_gelu_split: Function,
    pub vision_pool: Function,
    pub clamp: Function,
    pub residual_scale_add: Function,
    pub add: Function,
    pub bias_add: Function,
    pub audio_scaled_qk: Function,
    pub audio_attention: Function,
    pub glu: Function,
    pub depthwise_conv1d: Function,
    pub silu: Function,
    pub select_rows: Function,
    pub scatter_rows: Function,
    pub audio_norm: Function,
    pub norm: Function,
    pub unit_norm: Function,
    pub f32_to_bf16: Function,
    pub bf16_to_f32: Function,
}

impl MediaKernels {
    pub(crate) fn load(
        context: &Arc<Context>,
        module_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        let path = match module_path {
            Some(path) => path.to_path_buf(),
            None => std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(PathBuf::from)
                .ok_or_else(|| start("set INFINITY_OXIDE_KERNEL_MODULE for media kernels"))?,
        };
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| start(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        let function = |name: &str| module.load_function(name).map_err(start);
        Ok(Self {
            im2col: function(MEDIA_KERNEL_EXPORTS[0])?,
            mask: function(MEDIA_KERNEL_EXPORTS[1])?,
            channel_norm_relu: function(MEDIA_KERNEL_EXPORTS[2])?,
            transpose_btwc_to_bctw: function(MEDIA_KERNEL_EXPORTS[3])?,
            conv_rows_to_btcw: function(MEDIA_KERNEL_EXPORTS[4])?,
            center_pixels: function(MEDIA_KERNEL_EXPORTS[5])?,
            patch_add_position: function(MEDIA_KERNEL_EXPORTS[6])?,
            axial_tables: function(MEDIA_KERNEL_EXPORTS[7])?,
            axial_rope: function(MEDIA_KERNEL_EXPORTS[8])?,
            attention: function(MEDIA_KERNEL_EXPORTS[9])?,
            gated_gelu_split: function(MEDIA_KERNEL_EXPORTS[10])?,
            vision_pool: function(MEDIA_KERNEL_EXPORTS[11])?,
            clamp: function(MEDIA_KERNEL_EXPORTS[12])?,
            residual_scale_add: function(MEDIA_KERNEL_EXPORTS[13])?,
            add: function(MEDIA_KERNEL_EXPORTS[14])?,
            bias_add: function(MEDIA_KERNEL_EXPORTS[15])?,
            audio_scaled_qk: function(MEDIA_KERNEL_EXPORTS[16])?,
            audio_attention: function(MEDIA_KERNEL_EXPORTS[17])?,
            glu: function(MEDIA_KERNEL_EXPORTS[18])?,
            depthwise_conv1d: function(MEDIA_KERNEL_EXPORTS[19])?,
            silu: function(MEDIA_KERNEL_EXPORTS[20])?,
            select_rows: function(MEDIA_KERNEL_EXPORTS[21])?,
            scatter_rows: function(MEDIA_KERNEL_EXPORTS[22])?,
            audio_norm: function(MEDIA_KERNEL_EXPORTS[23])?,
            norm: function("infinity_gemma2_norm_bf16_f32scale")?,
            unit_norm: function("infinity_gemma2_unit_norm_bf16")?,
            f32_to_bf16: function("infinity_f32_to_bf16")?,
            bf16_to_f32: function("infinity_bf16_to_f32")?,
        })
    }
}

pub(crate) fn upload_bf16(
    uploader: &mut WeightUploader,
    source: &crate::model::WeightSource,
    name: &str,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(format!("{name} must be BF16")));
    }
    uploader.upload(&[source])
}

pub(crate) struct MediaRuntime {
    pub(crate) kernels: MediaKernels,
    pub(crate) text_width: usize,
    pub(crate) vision: Option<crate::gemma2_media_vision::VisionRuntime>,
    pub(crate) audio: Option<crate::gemma2_media_audio::AudioRuntime>,
}

pub(crate) struct PreparedMedia {
    pub(crate) rows: DeviceBuffer<Bf16>,
    pub(crate) indices: DeviceBuffer<u32>,
    pub(crate) count: usize,
    _request: MediaRequest,
    _vision_scratch: Option<crate::gemma2_media_vision::VisionScratch>,
    _audio_scratch: Option<crate::gemma2_media_audio::AudioScratch>,
}

impl PreparedMedia {
    pub(crate) fn device_bytes(&self) -> Result<usize, BackendError> {
        let mut total = self
            .rows
            .num_bytes()
            .checked_add(self.indices.num_bytes())
            .ok_or_else(|| start("media prepared byte count overflow"))?;
        if let Some(scratch) = self._vision_scratch.as_ref() {
            total = total
                .checked_add(scratch.device_bytes())
                .ok_or_else(|| start("vision scratch byte count overflow"))?;
        }
        if let Some(scratch) = self._audio_scratch.as_ref() {
            total = total
                .checked_add(scratch.device_bytes())
                .ok_or_else(|| start("audio scratch byte count overflow"))?;
        }
        Ok(total)
    }
}

impl MediaRuntime {
    pub(crate) fn load(
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        host: &MediaHost,
        text_width: usize,
        module_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        let kernels = MediaKernels::load(context, module_path)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let vision = match (host.vision.as_ref(), host.vision_config.as_ref()) {
            (Some(host), Some(config)) => Some(crate::gemma2_media_vision::VisionRuntime::load(
                &mut uploader,
                stream,
                host,
                config,
            )?),
            _ => None,
        };
        let audio = match (host.audio.as_ref(), host.audio_config.as_ref()) {
            (Some(host), Some(config)) => Some(crate::gemma2_media_audio::AudioRuntime::load(
                &mut uploader,
                stream,
                host,
                config,
            )?),
            _ => None,
        };
        uploader.finish()?;
        Ok(Self {
            kernels,
            text_width,
            vision,
            audio,
        })
    }

    pub(crate) fn prepare(
        &self,
        blas: &Blas,
        stream: &Arc<Stream>,
        request: MediaRequest,
    ) -> Result<PreparedMedia, BackendError> {
        let rows = product(request.rows, self.text_width)?;
        // SAFETY: every row is written by the tower copy before the text gather reads it.
        let projected =
            unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, rows) }.map_err(start)?;
        let indices = DeviceBuffer::from_host(stream, &request.row_indices).map_err(start)?;
        let mut owners = MediaPrepareOwners {
            stream: Arc::clone(stream),
            projected: Some(projected),
            vision: None,
            audio: None,
            armed: true,
        };
        let mut offset = 0usize;
        let max_image_rows = request
            .spans
            .iter()
            .filter_map(|span| match &span.kind {
                MediaSpanKind::Image { descriptor, .. } => Some(descriptor.patch_rows),
                _ => None,
            })
            .max();
        let max_audio_frames = request
            .spans
            .iter()
            .filter_map(|span| match &span.kind {
                MediaSpanKind::Audio { descriptor, .. } => Some(descriptor.frames),
                _ => None,
            })
            .max();
        if max_image_rows.is_some() && self.vision.is_none() {
            return Err(infer("image spans require the vision runtime"));
        }
        if max_audio_frames.is_some() && self.audio.is_none() {
            return Err(infer("audio spans require the audio runtime"));
        }
        if let (Some(vision), Some(rows)) = (&self.vision, max_image_rows) {
            owners.vision = Some(vision.scratch(stream, rows)?);
        }
        if let (Some(audio), Some(frames)) = (&self.audio, max_audio_frames) {
            owners.audio = Some(audio.scratch(stream, frames)?);
        }
        // Spans are globally token-sorted. Walk that same order so an
        // interleaved audio/image request writes rows to the matching indices.
        for span in &request.spans {
            match &span.kind {
                MediaSpanKind::Image { descriptor, pixels } => {
                    let vision = self
                        .vision
                        .as_ref()
                        .ok_or_else(|| infer("vision runtime disappeared"))?;
                    let scratch = owners
                        .vision
                        .as_mut()
                        .ok_or_else(|| infer("vision scratch disappeared"))?;
                    let image =
                        vision.run(&self.kernels, blas, stream, scratch, descriptor, pixels)?;
                    let destination = owners
                        .projected
                        .as_mut()
                        .ok_or_else(|| infer("media row arena disappeared"))?;
                    copy_rows(stream, destination, offset, &image, self.text_width)?;
                    offset = offset
                        .checked_add(descriptor.token_rows)
                        .ok_or_else(|| infer("media row offset overflow"))?;
                }
                MediaSpanKind::Audio {
                    descriptor,
                    features,
                    mask,
                } => {
                    let audio = self
                        .audio
                        .as_ref()
                        .ok_or_else(|| infer("audio runtime disappeared"))?;
                    let scratch = owners
                        .audio
                        .as_mut()
                        .ok_or_else(|| infer("audio scratch disappeared"))?;
                    let selected = audio.run(
                        &self.kernels,
                        blas,
                        stream,
                        scratch,
                        descriptor,
                        features,
                        mask,
                    )?;
                    let destination = owners
                        .projected
                        .as_mut()
                        .ok_or_else(|| infer("media row arena disappeared"))?;
                    copy_rows(stream, destination, offset, &selected, self.text_width)?;
                    offset = offset
                        .checked_add(descriptor.token_count)
                        .ok_or_else(|| infer("media row offset overflow"))?;
                }
            }
        }
        if offset != request.rows {
            return Err(infer("media towers produced an unexpected row count"));
        }
        let rows = owners
            .projected
            .take()
            .ok_or_else(|| infer("media row arena disappeared"))?;
        let vision_scratch = owners.vision.take();
        let audio_scratch = owners.audio.take();
        owners.armed = false;
        Ok(PreparedMedia {
            rows,
            indices,
            count: request.rows,
            _request: request,
            _vision_scratch: vision_scratch,
            _audio_scratch: audio_scratch,
        })
    }

    /// Copy every prepared row into the already gathered BF16 token state.
    pub(crate) fn scatter(
        &self,
        stream: &Stream,
        state: &mut DeviceBuffer<Bf16>,
        prepared: &PreparedMedia,
    ) -> Result<(), BackendError> {
        if prepared.count == 0 || self.text_width == 0 {
            return Err(infer("empty media scatter"));
        }
        let mut state_ptr = state.cu_deviceptr();
        let mut rows = prepared.rows.cu_deviceptr();
        let mut indices = prepared.indices.cu_deviceptr();
        let mut count = u32_size(prepared.count)?;
        let mut width = u32_size(self.text_width)?;
        launch(
            &self.kernels.scatter_rows,
            stream,
            blocks(product(prepared.count, self.text_width)?)?,
            0,
            &mut [
                arg(&mut state_ptr),
                arg(&mut rows),
                arg(&mut indices),
                arg(&mut count),
                arg(&mut width),
            ],
        )
    }
}

struct MediaPrepareOwners {
    stream: Arc<Stream>,
    projected: Option<DeviceBuffer<Bf16>>,
    vision: Option<crate::gemma2_media_vision::VisionScratch>,
    audio: Option<crate::gemma2_media_audio::AudioScratch>,
    armed: bool,
}

impl Drop for MediaPrepareOwners {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // A failed tower launch may still have work on the stream. Drop only
        // after a proven drain; otherwise retain the physical owners.
        if self.stream.synchronize().is_ok() {
            return;
        }
        if let Some(buffer) = self.projected.take() {
            std::mem::forget(buffer);
        }
        if let Some(scratch) = self.vision.take() {
            std::mem::forget(scratch);
        }
        if let Some(scratch) = self.audio.take() {
            std::mem::forget(scratch);
        }
    }
}

pub(crate) fn row_byte_offset(row: usize, width: usize) -> Result<usize, BackendError> {
    row.checked_mul(width)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<Bf16>()))
        .ok_or_else(|| infer("media copy offset overflow"))
}

fn copy_rows(
    stream: &Stream,
    destination: &mut DeviceBuffer<Bf16>,
    row_offset: usize,
    source: &DeviceBuffer<Bf16>,
    width: usize,
) -> Result<(), BackendError> {
    let offset = row_byte_offset(row_offset, width)?;
    let bytes = source.num_bytes();
    let destination_ptr = destination
        .cu_deviceptr()
        .checked_add(u64::try_from(offset).map_err(infer)?)
        .ok_or_else(|| infer("media copy destination overflow"))?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > destination.num_bytes())
    {
        return Err(infer("media copy exceeds the prepared row arena"));
    }
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            destination_ptr,
            source.cu_deviceptr(),
            bytes,
            stream.cu_stream(),
        )
    }
    .map_err(infer)
}

/// Shared lossless F32 conv-filter expansion used by both subsampling blocks.
pub(crate) fn expand_bf16_to_f32(
    kernels: &MediaKernels,
    stream: &Stream,
    source: &DeviceBuffer<Bf16>,
    destination: &mut DeviceBuffer<f32>,
) -> Result<(), BackendError> {
    if source.len() != destination.len() {
        return Err(infer("conv filter expansion length mismatch"));
    }
    let mut input = source.cu_deviceptr();
    let mut out = destination.cu_deviceptr();
    let mut count = u32_size(source.len())?;
    let mut width = count;
    let mut selected = 0u64;
    launch(
        &kernels.bf16_to_f32,
        stream,
        blocks(source.len())?,
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

/// One checked BF16 projection through the plain cuBLAS handle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn media_linear(
    kernels: &MediaKernels,
    blas: &Blas,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    weight: &DeviceBuffer<Bf16>,
    bias: Option<&DeviceBuffer<Bf16>>,
    rows: usize,
    input_width: usize,
    output_width: usize,
    input_clip: Option<[f32; 2]>,
    output_clip: Option<[f32; 2]>,
    clamp_buffer: &mut DeviceBuffer<Bf16>,
) -> Result<(), BackendError> {
    let mut source = input;
    if let Some([minimum, maximum]) = input_clip {
        if clamp_buffer.len() < input.len() {
            return Err(infer("media clip buffer is too small"));
        }
        clamp_bf16_in_place(kernels, stream, input, clamp_buffer, minimum, maximum)?;
        source = clamp_buffer;
    }
    let plan = crate::math::LinearPlan::<Bf16>::contiguous(rows, input_width, output_width)
        .map_err(infer)?;
    unsafe { blas.linear(&plan, source, weight, output) }.map_err(infer)?;
    if let Some(bias) = bias {
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(output_width)?;
        let mut bias = bias.cu_deviceptr();
        let mut data = output.cu_deviceptr();
        launch(
            &kernels.bias_add,
            stream,
            blocks(product(rows as usize, width as usize)?)?,
            0,
            &mut [
                arg(&mut rows),
                arg(&mut width),
                arg(&mut bias),
                arg(&mut data),
            ],
        )?;
    }
    if let Some([minimum, maximum]) = output_clip {
        clamp_bf16_buffer(kernels, stream, output, minimum, maximum)?;
    }
    Ok(())
}

pub(crate) fn clamp_bf16_buffer(
    kernels: &MediaKernels,
    stream: &Stream,
    buffer: &mut DeviceBuffer<Bf16>,
    minimum: f32,
    maximum: f32,
) -> Result<(), BackendError> {
    let mut input = buffer.cu_deviceptr();
    let mut out = buffer.cu_deviceptr();
    let mut count = u32_size(buffer.len())?;
    let mut min = minimum;
    let mut max = maximum;
    launch(
        &kernels.clamp,
        stream,
        blocks(buffer.len())?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut count),
            arg(&mut min),
            arg(&mut max),
        ],
    )
}

pub(crate) fn clamp_bf16_in_place(
    kernels: &MediaKernels,
    stream: &Stream,
    source: &DeviceBuffer<Bf16>,
    destination: &mut DeviceBuffer<Bf16>,
    minimum: f32,
    maximum: f32,
) -> Result<(), BackendError> {
    if destination.len() < source.len() {
        return Err(infer("media clamp destination is too small"));
    }
    let mut input = source.cu_deviceptr();
    let mut out = destination.cu_deviceptr();
    let mut count = u32_size(source.len())?;
    let mut min = minimum;
    let mut max = maximum;
    launch(
        &kernels.clamp,
        stream,
        blocks(source.len())?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut count),
            arg(&mut min),
            arg(&mut max),
        ],
    )
}

pub(crate) fn norm_rows(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    scale: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut scale = scale.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(width)?;
    let mut heads = 1u32;
    let mut stride = u64::try_from(width).map_err(infer)?;
    let mut eps = eps;
    launch_threads(
        &kernels.norm,
        stream,
        (rows_u32, 1, 1),
        width_u32.clamp(32, 512),
        0,
        &mut [
            arg(&mut input),
            arg(&mut scale),
            arg(&mut out),
            arg(&mut rows_u32),
            arg(&mut width_u32),
            arg(&mut heads),
            arg(&mut stride),
            arg(&mut eps),
        ],
    )
}

pub(crate) fn audio_norm_rows(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    scale: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut scale = scale.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(width)?;
    let mut eps = eps;
    // Native `threads = next_power_of_two(width) capped at 1024`.
    let threads = width.next_power_of_two().clamp(32, 1024);
    launch_threads(
        &kernels.audio_norm,
        stream,
        (rows_u32, 1, 1),
        u32::try_from(threads).map_err(infer)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut scale),
            arg(&mut out),
            arg(&mut rows_u32),
            arg(&mut width_u32),
            arg(&mut eps),
        ],
    )
}

pub(crate) fn norm_heads(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    scale: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    heads: usize,
    token_stride: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut scale = scale.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(width)?;
    let mut heads_u32 = u32_size(heads)?;
    let mut stride = u64::try_from(token_stride).map_err(infer)?;
    let mut eps = eps;
    launch_threads(
        &kernels.norm,
        stream,
        (rows_u32, 1, 1),
        width_u32.clamp(32, 512),
        0,
        &mut [
            arg(&mut input),
            arg(&mut scale),
            arg(&mut out),
            arg(&mut rows_u32),
            arg(&mut width_u32),
            arg(&mut heads_u32),
            arg(&mut stride),
            arg(&mut eps),
        ],
    )
}

pub(crate) fn unit_norm_heads(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    heads: usize,
    token_stride: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(width)?;
    let mut heads_u32 = u32_size(heads)?;
    let mut stride = u64::try_from(token_stride).map_err(infer)?;
    let mut eps = eps;
    launch_threads(
        &kernels.unit_norm,
        stream,
        (rows_u32, 1, 1),
        width_u32.clamp(32, 512),
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut rows_u32),
            arg(&mut width_u32),
            arg(&mut heads_u32),
            arg(&mut stride),
            arg(&mut eps),
        ],
    )
}

pub(crate) fn upload_norm_source(
    stream: &Arc<Stream>,
    source: &crate::model::WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    upload_norm(stream, source, width)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_row_copy_offsets_use_the_full_text_width() {
        assert_eq!(row_byte_offset(0, 512).unwrap(), 0);
        assert_eq!(row_byte_offset(1, 512).unwrap(), 1024);
        assert_eq!(row_byte_offset(266, 512).unwrap(), 266 * 1024);
        assert_eq!(row_byte_offset(369, 512).unwrap(), 369 * 1024);
        assert!(row_byte_offset(usize::MAX, 512).is_err());
    }

    /// Root GPU gate: the actual image and audio towers run on the media kernel
    /// bundle, their projected rows are scattered into a gathered text state,
    /// and every owned device buffer survives the stream.
    #[test]
    #[ignore = "Root GPU: actual EmbeddingGemma2 image+audio towers plus the media kernel bundle"]
    fn actual_image_and_audio_towers_scatter_projected_rows(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::tensor::Bf16;
        use infinity_v3_backend_core::{AudioFeatures, Batch, ImagePatches, MultimodalEncoding};
        use std::path::PathBuf;

        let model = std::env::var_os("INFINITY_OXIDE_GEMMA2_MODEL")
            .map(PathBuf::from)
            .ok_or("set INFINITY_OXIDE_GEMMA2_MODEL")?;
        let module = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
                .ok_or("set INFINITY_OXIDE_KERNEL_MODULE to the media bundle")?,
        );
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
        let text_hidden = raw["text_config"]["hidden_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("missing text_config.hidden_size")?;
        let config = crate::gemma2_media::MediaConfig::from_config_value(&raw, text_hidden)?;
        let tokens = config.tokens.ok_or("config has no media token ids")?;
        let host = MediaHost::load(&model, &config)?.ok_or("model has no media towers")?;
        let context = Context::new(0).map_err(|error| error.to_string())?;
        let stream = context.new_stream().map_err(|error| error.to_string())?;
        let runtime = MediaRuntime::load(&context, &stream, &host, text_hidden, Some(&module))?;
        let blas = Blas::new(context.clone(), stream.clone())?;

        let dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedding_gemma2_media");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("cats_coco.json"))?)?;
        let pixels = std::fs::read(dir.join("cats_coco.patches.f32"))?
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let image = Arc::new(ImagePatches {
            pixels,
            grid_thw: [
                meta["grid_thw"][0].as_u64().unwrap() as usize,
                meta["grid_thw"][1].as_u64().unwrap() as usize,
                meta["grid_thw"][2].as_u64().unwrap() as usize,
            ],
            patch_dim: meta["patch_dim"].as_u64().unwrap() as usize,
            merge_size: meta["merge_size"].as_u64().unwrap() as usize,
        });
        let features = std::fs::read(dir.join("cat_meow.features.f32"))?
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        let mask = std::fs::read(dir.join("cat_meow.mask.u8"))?;
        let audio = Arc::new(AudioFeatures {
            values: features,
            mask,
            feature_size: 128,
            subsample_strides: [2, 2],
        });

        let mut input_ids = vec![2u32];
        input_ids.push(tokens.begin_image);
        let image_start = input_ids.len();
        input_ids.extend(std::iter::repeat_n(tokens.image, image.token_count()));
        input_ids.push(tokens.end_image);
        input_ids.push(tokens.begin_audio);
        let audio_start = input_ids.len();
        input_ids.extend(std::iter::repeat_n(tokens.audio, audio.token_count()));
        input_ids.push(tokens.end_audio);
        let length = input_ids.len() as u32;
        let position_ids: Vec<u32> = (0..length).collect();
        let encoding = Arc::new(MultimodalEncoding {
            images: vec![(image_start, Arc::clone(&image))],
            audios: vec![(audio_start, Arc::clone(&audio))],
            position_ids: std::array::from_fn(|_| position_ids.clone()),
            memory: None,
        });
        let batch = Batch {
            multimodal: vec![Some(encoding)],
            input_ids,
            token_type_ids: vec![0; length as usize],
            position_ids,
            cumulative_seq_lengths: vec![0, length],
            max_length: length,
            pooled_indices: vec![0],
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        };
        let request = MediaRequest::plan(&batch, &config, text_hidden, Some(tokens))?
            .ok_or("media plan produced no request")?;
        let expected = image.token_count() + audio.token_count();
        assert_eq!(request.rows, expected);
        let prepared = runtime.prepare(&blas, &stream, request)?;
        assert_eq!(prepared.count, expected);
        let mut state = DeviceBuffer::<Bf16>::from_host(
            &stream,
            &vec![Bf16::default(); length as usize * text_hidden],
        )?;
        runtime.scatter(&stream, &mut state, &prepared)?;
        stream.synchronize().map_err(|error| error.to_string())?;
        Ok(())
    }
}
