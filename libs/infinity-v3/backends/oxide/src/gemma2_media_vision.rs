// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Device Gemma4 vision tower: cold BF16 weights, per-request scratch, and the
//! Native patch -> axial attention -> pool -> projection composition.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::cublas::Blas;
use crate::gemma2_media::{MediaLinear, VisionActivation, VisionConfig, VisionHost};
use crate::gemma2_media_exec::{
    media_linear, norm_heads, norm_rows, unit_norm_heads, upload_bf16, upload_norm_source,
    MediaKernels,
};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::{arg, blocks, infer, launch, product, start, u32_size};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::media::{ImageDescriptor, MediaFamily};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

struct VisionLayer {
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    pre_feedforward_norm: DeviceBuffer<f32>,
    post_feedforward_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    o: DeviceBuffer<Bf16>,
    gate: DeviceBuffer<Bf16>,
    up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
}

pub(crate) struct VisionRuntime {
    pub(crate) config: VisionConfig,
    patch_projection: DeviceBuffer<Bf16>,
    position_table: DeviceBuffer<Bf16>,
    embed_projection: DeviceBuffer<Bf16>,
    rope_inverse: DeviceBuffer<f32>,
    layers: Vec<VisionLayer>,
}

pub(crate) struct VisionScratch {
    pub(crate) rows: usize,
    centered: DeviceBuffer<Bf16>,
    state: DeviceBuffer<Bf16>,
    next: DeviceBuffer<Bf16>,
    normalized: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    attn: DeviceBuffer<Bf16>,
    o: DeviceBuffer<Bf16>,
    gate: DeviceBuffer<Bf16>,
    up: DeviceBuffer<Bf16>,
    act: DeviceBuffer<Bf16>,
    mlp: DeviceBuffer<Bf16>,
    pooled: DeviceBuffer<Bf16>,
    pooled_norm: DeviceBuffer<Bf16>,
    projected: Option<DeviceBuffer<Bf16>>,
    cosine: DeviceBuffer<Bf16>,
    sine: DeviceBuffer<Bf16>,
}

struct VisionGeometry {
    patch_rows: usize,
    token_rows: usize,
    height: usize,
    width: usize,
}

fn check_linear(linear: &MediaLinear, name: &str, shape: [usize; 2]) -> Result<(), BackendError> {
    if linear.shape != shape
        || linear.weight.shape() != shape.as_slice()
        || linear.bias.is_some()
        || linear.input_clip.is_some()
        || linear.output_clip.is_some()
    {
        return Err(start(format!(
            "vision projection {name} disagrees with the Native biasless contract"
        )));
    }
    Ok(())
}

fn validate_geometry(
    config: &VisionConfig,
    descriptor: &ImageDescriptor,
    pixel_length: usize,
) -> Result<VisionGeometry, BackendError> {
    if descriptor.family != MediaFamily::EmbeddingGemma2 {
        return Err(start("vision descriptor is not an EmbeddingGemma2 image"));
    }
    let [frames, height, width] = descriptor.grid_thw;
    if frames != 1 || height == 0 || width == 0 {
        return Err(start("Gemma4 images must be a single nonempty frame"));
    }
    if descriptor.merge_size != config.pooling {
        return Err(start("vision descriptor merge size disagrees with pooling"));
    }
    if descriptor.patch_dim != config.patch_dim() {
        return Err(start(
            "vision descriptor patch width disagrees with the config",
        ));
    }
    let patch_rows = height
        .checked_mul(width)
        .ok_or_else(|| start("vision patch row overflow"))?;
    if descriptor.patch_rows != patch_rows {
        return Err(start(
            "vision descriptor grid disagrees with its patch rows",
        ));
    }
    if height % config.pooling != 0 || width % config.pooling != 0 {
        return Err(start(
            "vision image grid is not divisible by the pooling kernel",
        ));
    }
    if height > config.position_table_rows || width > config.position_table_rows {
        return Err(start("vision image grid exceeds the position table"));
    }
    let token_rows = (height / config.pooling)
        .checked_mul(width / config.pooling)
        .ok_or_else(|| start("vision token row overflow"))?;
    if token_rows == 0 || descriptor.token_rows != token_rows {
        return Err(start("vision descriptor token rows disagree with the grid"));
    }
    let expected = patch_rows
        .checked_mul(descriptor.patch_dim)
        .ok_or_else(|| start("vision pixel length overflow"))?;
    if pixel_length != expected {
        return Err(start(format!(
            "vision pixels have {pixel_length} values, expected {expected}"
        )));
    }
    Ok(VisionGeometry {
        patch_rows,
        token_rows,
        height,
        width,
    })
}

fn scratch_buffer(
    stream: &Arc<Stream>,
    elements: usize,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    if elements == 0 {
        return Err(infer("empty vision scratch buffer"));
    }
    // SAFETY: every scratch buffer is fully written by a kernel or copy before
    // any later kernel reads it; no uninitialized value reaches the host.
    unsafe { DeviceBuffer::uninitialized_async(stream, elements) }.map_err(start)
}

impl VisionScratch {
    pub(crate) fn device_bytes(&self) -> usize {
        let mut total = 0usize;
        macro_rules! add {
            ($field:ident) => {
                total = total.saturating_add(self.$field.num_bytes());
            };
        }
        add!(centered);
        add!(state);
        add!(next);
        add!(normalized);
        add!(q);
        add!(k);
        add!(v);
        add!(attn);
        add!(o);
        add!(gate);
        add!(up);
        add!(act);
        add!(mlp);
        add!(pooled);
        add!(pooled_norm);
        add!(cosine);
        add!(sine);
        if let Some(projected) = self.projected.as_ref() {
            total = total.saturating_add(projected.num_bytes());
        }
        total
    }
}

impl VisionRuntime {
    pub(crate) fn load(
        uploader: &mut WeightUploader,
        stream: &Arc<Stream>,
        host: &VisionHost,
        config: &VisionConfig,
    ) -> Result<Self, BackendError> {
        if config.standardize {
            return Err(start(
                "Gemma4 vision standardize is unsupported without std_bias/std_scale",
            ));
        }
        if config.use_clipped_linears {
            return Err(start("Gemma4 vision clipped linears are unsupported"));
        }
        if config.activation != VisionActivation::GeluTanh {
            return Err(start("Gemma4 vision activation must be gelu_pytorch_tanh"));
        }
        if host.layers.len() != config.layers {
            return Err(start("vision host layer count disagrees with the config"));
        }
        let h = config.hidden_size;
        let d = config.head_dim;
        let i = config.intermediate_size;
        let kv = d
            .checked_mul(config.kv_heads)
            .ok_or_else(|| start("vision key/value width overflow"))?;
        let output_width = config.output_width;
        if output_width == 0 {
            return Err(start("vision output width disagrees with the hidden size"));
        }
        check_linear(
            &host.patch_projection,
            "patch_embedder.input_proj",
            [h, config.patch_dim()],
        )?;
        check_linear(
            &host.embed_projection,
            "embed_vision.embedding_projection",
            [output_width, h],
        )?;
        if host.position_table.shape() != [2, config.position_table_rows, h].as_slice() {
            return Err(start(
                "vision position table shape disagrees with the config",
            ));
        }
        let patch_projection = upload_bf16(
            uploader,
            &host.patch_projection.weight,
            "vision_tower.patch_embedder.input_proj.weight",
        )?;
        let position_table = upload_bf16(
            uploader,
            &host.position_table,
            "vision_tower.patch_embedder.position_embedding_table",
        )?;
        let embed_projection = upload_bf16(
            uploader,
            &host.embed_projection.weight,
            "embed_vision.embedding_projection.weight",
        )?;
        let inverse = config.rope_inverse();
        if inverse.is_empty() || inverse.iter().any(|value| !value.is_finite()) {
            return Err(start("vision inverse frequencies are invalid"));
        }
        let rope_inverse = DeviceBuffer::from_host(stream, &inverse).map_err(start)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers)
            .map_err(|_| start("vision layer allocation failed"))?;
        for (index, layer) in host.layers.iter().enumerate() {
            let name = format!("vision_tower.encoder.layers.{index}");
            check_linear(&layer.q, &format!("{name}.self_attn.q_proj"), [h, h])?;
            check_linear(&layer.k, &format!("{name}.self_attn.k_proj"), [kv, h])?;
            check_linear(&layer.v, &format!("{name}.self_attn.v_proj"), [kv, h])?;
            check_linear(&layer.o, &format!("{name}.self_attn.o_proj"), [h, h])?;
            check_linear(&layer.gate, &format!("{name}.mlp.gate_proj"), [i, h])?;
            check_linear(&layer.up, &format!("{name}.mlp.up_proj"), [i, h])?;
            check_linear(&layer.down, &format!("{name}.mlp.down_proj"), [h, i])?;
            layers.push(VisionLayer {
                input_norm: upload_norm_source(stream, &layer.input_norm, h)?,
                post_attention_norm: upload_norm_source(stream, &layer.post_attention_norm, h)?,
                pre_feedforward_norm: upload_norm_source(stream, &layer.pre_feedforward_norm, h)?,
                post_feedforward_norm: upload_norm_source(stream, &layer.post_feedforward_norm, h)?,
                q_norm: upload_norm_source(stream, &layer.q_norm, d)?,
                k_norm: upload_norm_source(stream, &layer.k_norm, d)?,
                q: upload_bf16(
                    uploader,
                    &layer.q.weight,
                    &format!("{name}.self_attn.q_proj.weight"),
                )?,
                k: upload_bf16(
                    uploader,
                    &layer.k.weight,
                    &format!("{name}.self_attn.k_proj.weight"),
                )?,
                v: upload_bf16(
                    uploader,
                    &layer.v.weight,
                    &format!("{name}.self_attn.v_proj.weight"),
                )?,
                o: upload_bf16(
                    uploader,
                    &layer.o.weight,
                    &format!("{name}.self_attn.o_proj.weight"),
                )?,
                gate: upload_bf16(
                    uploader,
                    &layer.gate.weight,
                    &format!("{name}.mlp.gate_proj.weight"),
                )?,
                up: upload_bf16(
                    uploader,
                    &layer.up.weight,
                    &format!("{name}.mlp.up_proj.weight"),
                )?,
                down: upload_bf16(
                    uploader,
                    &layer.down.weight,
                    &format!("{name}.mlp.down_proj.weight"),
                )?,
            });
        }
        Ok(Self {
            config: config.clone(),
            patch_projection,
            position_table,
            embed_projection,
            rope_inverse,
            layers,
        })
    }

    pub(crate) fn scratch(
        &self,
        stream: &Arc<Stream>,
        rows: usize,
    ) -> Result<VisionScratch, BackendError> {
        if rows == 0 {
            return Err(infer("empty vision scratch rows"));
        }
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        let p = self.config.pooling;
        let pool_area = p
            .checked_mul(p)
            .ok_or_else(|| infer("vision pooling area overflow"))?;
        if !rows.is_multiple_of(pool_area) {
            return Err(infer(
                "vision patch rows are not divisible by the pooling area",
            ));
        }
        let token_rows = rows / pool_area;
        let hidden = product(rows, h)?;
        let inter = product(rows, i)?;
        let token_hidden = product(token_rows, h)?;
        let projected = product(token_rows, self.config.output_width)?;
        let pairs = self
            .config
            .head_dim
            .checked_div(4)
            .ok_or_else(|| infer("zero vision rotary pairs"))?;
        let angles = product(rows, pairs)?
            .checked_mul(2)
            .ok_or_else(|| infer("vision angle table overflow"))?;
        Ok(VisionScratch {
            rows,
            centered: scratch_buffer(stream, hidden)?,
            state: scratch_buffer(stream, hidden)?,
            next: scratch_buffer(stream, hidden)?,
            normalized: scratch_buffer(stream, hidden)?,
            q: scratch_buffer(stream, hidden)?,
            k: scratch_buffer(stream, hidden)?,
            v: scratch_buffer(stream, hidden)?,
            attn: scratch_buffer(stream, hidden)?,
            o: scratch_buffer(stream, hidden)?,
            gate: scratch_buffer(stream, inter)?,
            up: scratch_buffer(stream, inter)?,
            act: scratch_buffer(stream, inter)?,
            mlp: scratch_buffer(stream, hidden)?,
            pooled: scratch_buffer(stream, token_hidden)?,
            pooled_norm: scratch_buffer(stream, token_hidden)?,
            projected: Some(scratch_buffer(stream, projected)?),
            cosine: scratch_buffer(stream, angles)?,
            sine: scratch_buffer(stream, angles)?,
        })
    }

    pub(crate) fn run(
        &self,
        kernels: &MediaKernels,
        blas: &Blas,
        stream: &Arc<Stream>,
        scratch: &mut VisionScratch,
        descriptor: &ImageDescriptor,
        pixels: &[f32],
    ) -> Result<DeviceBuffer<Bf16>, BackendError> {
        let geometry = validate_geometry(&self.config, descriptor, pixels.len())?;
        let rows = geometry.patch_rows;
        let token_rows = geometry.token_rows;
        if rows > scratch.rows {
            return Err(infer("vision scratch is smaller than the requested image"));
        }
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        let d = self.config.head_dim;
        let heads = self.config.heads;
        let eps = self.config.rms_norm_eps;
        let hidden = product(rows, h)?;
        let projected_len = product(token_rows, self.config.output_width)?;
        let pairs = d / 4;
        let angles = product(rows, pairs)?
            .checked_mul(2)
            .ok_or_else(|| infer("vision angle table overflow"))?;

        // Cold media path: the pixel upload may synchronize before any tower work.
        let pixels = DeviceBuffer::from_host(stream, pixels).map_err(start)?;
        let mut columns = Vec::new();
        let mut grid_rows = Vec::new();
        columns
            .try_reserve_exact(rows)
            .map_err(|_| infer("vision column allocation failed"))?;
        grid_rows
            .try_reserve_exact(rows)
            .map_err(|_| infer("vision row allocation failed"))?;
        for index in 0..rows {
            columns.push(
                u32::try_from(index % geometry.width)
                    .map_err(|_| infer("vision column exceeds u32"))?,
            );
            grid_rows.push(
                u32::try_from(index / geometry.width)
                    .map_err(|_| infer("vision row exceeds u32"))?,
            );
        }
        let columns = DeviceBuffer::from_host(stream, &columns).map_err(start)?;
        let grid_rows = DeviceBuffer::from_host(stream, &grid_rows).map_err(start)?;
        let cu_seqlens =
            DeviceBuffer::from_host(stream, &[0u32, u32_size(rows)?]).map_err(start)?;

        center_pixels(kernels, stream, &pixels, &mut scratch.centered, hidden)?;
        media_linear(
            kernels,
            blas,
            stream,
            &scratch.centered,
            &mut scratch.state,
            &self.patch_projection,
            None,
            rows,
            self.config.patch_dim(),
            h,
            None,
            None,
            &mut scratch.mlp,
        )?;
        patch_add_position(
            kernels,
            stream,
            &scratch.state,
            &columns,
            &grid_rows,
            &self.position_table,
            &mut scratch.next,
            rows,
            h,
            self.config.position_table_rows,
        )?;
        std::mem::swap(&mut scratch.state, &mut scratch.next);
        axial_tables(
            kernels,
            stream,
            &columns,
            &grid_rows,
            &self.rope_inverse,
            &mut scratch.cosine,
            &mut scratch.sine,
            rows,
            pairs,
            angles,
        )?;

        for layer in &self.layers {
            norm_rows(
                kernels,
                stream,
                &scratch.state,
                &layer.input_norm,
                &mut scratch.normalized,
                rows,
                h,
                eps,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.normalized,
                &mut scratch.q,
                &layer.q,
                None,
                rows,
                h,
                h,
                None,
                None,
                &mut scratch.centered,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.normalized,
                &mut scratch.k,
                &layer.k,
                None,
                rows,
                h,
                d.checked_mul(self.config.kv_heads)
                    .ok_or_else(|| infer("vision key width overflow"))?,
                None,
                None,
                &mut scratch.centered,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.normalized,
                &mut scratch.v,
                &layer.v,
                None,
                rows,
                h,
                d.checked_mul(self.config.kv_heads)
                    .ok_or_else(|| infer("vision value width overflow"))?,
                None,
                None,
                &mut scratch.centered,
            )?;

            std::mem::swap(&mut scratch.q, &mut scratch.o);
            norm_heads(
                kernels,
                stream,
                &scratch.o,
                &layer.q_norm,
                &mut scratch.q,
                product(rows, heads)?,
                d,
                heads,
                h,
                eps,
            )?;
            std::mem::swap(&mut scratch.k, &mut scratch.attn);
            norm_heads(
                kernels,
                stream,
                &scratch.attn,
                &layer.k_norm,
                &mut scratch.k,
                product(rows, heads)?,
                d,
                heads,
                h,
                eps,
            )?;
            std::mem::swap(&mut scratch.v, &mut scratch.normalized);
            unit_norm_heads(
                kernels,
                stream,
                &scratch.normalized,
                &mut scratch.v,
                product(rows, heads)?,
                d,
                heads,
                h,
                eps,
            )?;
            axial_rope(
                kernels,
                stream,
                &mut scratch.q,
                &scratch.cosine,
                &scratch.sine,
                rows,
                heads,
                d,
            )?;
            axial_rope(
                kernels,
                stream,
                &mut scratch.k,
                &scratch.cosine,
                &scratch.sine,
                rows,
                heads,
                d,
            )?;
            attention(
                kernels,
                stream,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                &mut scratch.attn,
                &cu_seqlens,
                rows,
                heads,
                self.config.kv_heads,
                d,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.attn,
                &mut scratch.o,
                &layer.o,
                None,
                rows,
                d.checked_mul(self.config.kv_heads)
                    .ok_or_else(|| infer("vision attention width overflow"))?,
                h,
                None,
                None,
                &mut scratch.centered,
            )?;
            norm_rows(
                kernels,
                stream,
                &scratch.o,
                &layer.post_attention_norm,
                &mut scratch.normalized,
                rows,
                h,
                eps,
            )?;
            add_bf16(
                kernels,
                stream,
                &mut scratch.state,
                &scratch.normalized,
                hidden,
            )?;
            norm_rows(
                kernels,
                stream,
                &scratch.state,
                &layer.pre_feedforward_norm,
                &mut scratch.normalized,
                rows,
                h,
                eps,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.normalized,
                &mut scratch.gate,
                &layer.gate,
                None,
                rows,
                h,
                i,
                None,
                None,
                &mut scratch.centered,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.normalized,
                &mut scratch.up,
                &layer.up,
                None,
                rows,
                h,
                i,
                None,
                None,
                &mut scratch.centered,
            )?;
            gated_gelu_split(
                kernels,
                stream,
                &scratch.gate,
                &scratch.up,
                &mut scratch.act,
                rows,
                i,
            )?;
            media_linear(
                kernels,
                blas,
                stream,
                &scratch.act,
                &mut scratch.mlp,
                &layer.down,
                None,
                rows,
                i,
                h,
                None,
                None,
                &mut scratch.centered,
            )?;
            norm_rows(
                kernels,
                stream,
                &scratch.mlp,
                &layer.post_feedforward_norm,
                &mut scratch.normalized,
                rows,
                h,
                eps,
            )?;
            add_bf16(
                kernels,
                stream,
                &mut scratch.state,
                &scratch.normalized,
                hidden,
            )?;
        }

        vision_pool(
            kernels,
            stream,
            &scratch.state,
            &mut scratch.pooled,
            geometry.height,
            geometry.width,
            self.config.pooling,
            h,
        )?;
        unit_norm_heads(
            kernels,
            stream,
            &scratch.pooled,
            &mut scratch.pooled_norm,
            token_rows,
            h,
            1,
            h,
            eps,
        )?;
        let mut projected = match scratch.projected.take() {
            Some(buffer) if buffer.len() == projected_len => buffer,
            _ => scratch_buffer(stream, projected_len)?,
        };
        media_linear(
            kernels,
            blas,
            stream,
            &scratch.pooled_norm,
            &mut projected,
            &self.embed_projection,
            None,
            token_rows,
            h,
            self.config.output_width,
            None,
            None,
            &mut scratch.centered,
        )?;
        Ok(projected)
    }
}

fn center_pixels(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<Bf16>,
    count: usize,
) -> Result<(), BackendError> {
    if output.len() < count {
        return Err(infer("vision centered buffer is too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.center_pixels,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut out), arg(&mut count)],
    )
}

#[allow(clippy::too_many_arguments)]
fn patch_add_position(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    columns: &DeviceBuffer<u32>,
    rows: &DeviceBuffer<u32>,
    table: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    count: usize,
    width: usize,
    positions: usize,
) -> Result<(), BackendError> {
    if output.len() < product(count, width)? {
        return Err(infer("vision position output buffer is too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut columns = columns.cu_deviceptr();
    let mut rows = rows.cu_deviceptr();
    let mut table = table.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut count = u32_size(count)?;
    let mut width = u32_size(width)?;
    let mut positions = u32_size(positions)?;
    launch(
        &kernels.patch_add_position,
        stream,
        blocks(product(count as usize, width as usize)?)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut columns),
            arg(&mut rows),
            arg(&mut table),
            arg(&mut out),
            arg(&mut count),
            arg(&mut width),
            arg(&mut positions),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn axial_tables(
    kernels: &MediaKernels,
    stream: &Stream,
    columns: &DeviceBuffer<u32>,
    rows: &DeviceBuffer<u32>,
    inverse: &DeviceBuffer<f32>,
    cosine: &mut DeviceBuffer<Bf16>,
    sine: &mut DeviceBuffer<Bf16>,
    count: usize,
    pairs: usize,
    angles: usize,
) -> Result<(), BackendError> {
    if cosine.len() < angles || sine.len() < angles {
        return Err(infer("vision axial table buffers are too small"));
    }
    let mut columns = columns.cu_deviceptr();
    let mut rows = rows.cu_deviceptr();
    let mut inverse = inverse.cu_deviceptr();
    let mut cosine = cosine.cu_deviceptr();
    let mut sine = sine.cu_deviceptr();
    let mut count = u32_size(count)?;
    let mut pairs = u32_size(pairs)?;
    launch(
        &kernels.axial_tables,
        stream,
        blocks(product(count as usize, pairs as usize)?)?,
        0,
        &mut [
            arg(&mut columns),
            arg(&mut rows),
            arg(&mut inverse),
            arg(&mut cosine),
            arg(&mut sine),
            arg(&mut count),
            arg(&mut pairs),
        ],
    )
}

fn axial_rope(
    kernels: &MediaKernels,
    stream: &Stream,
    qk: &mut DeviceBuffer<Bf16>,
    cosine: &DeviceBuffer<Bf16>,
    sine: &DeviceBuffer<Bf16>,
    tokens: usize,
    heads: usize,
    dim: usize,
) -> Result<(), BackendError> {
    let mut qk = qk.cu_deviceptr();
    let mut cosine = cosine.cu_deviceptr();
    let mut sine = sine.cu_deviceptr();
    let mut tokens = u32_size(tokens)?;
    let mut heads = u32_size(heads)?;
    let mut dim = u32_size(dim)?;
    launch(
        &kernels.axial_rope,
        stream,
        blocks(product(
            product(tokens as usize, heads as usize)?,
            dim as usize / 2,
        )?)?,
        0,
        &mut [
            arg(&mut qk),
            arg(&mut cosine),
            arg(&mut sine),
            arg(&mut tokens),
            arg(&mut heads),
            arg(&mut dim),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn attention(
    kernels: &MediaKernels,
    stream: &Stream,
    q: &DeviceBuffer<Bf16>,
    k: &DeviceBuffer<Bf16>,
    v: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    cu_seqlens: &DeviceBuffer<u32>,
    total: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<(), BackendError> {
    if output.len() < product(product(total, heads)?, dim)? {
        return Err(infer("vision attention output buffer is too small"));
    }
    let mut q = q.cu_deviceptr();
    let mut k = k.cu_deviceptr();
    let mut v = v.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut cu_seqlens = cu_seqlens.cu_deviceptr();
    let mut total = u32_size(total)?;
    let mut batch = 1u32;
    let mut heads = u32_size(heads)?;
    let mut kv_heads = u32_size(kv_heads)?;
    let mut dim = u32_size(dim)?;
    let mut max_q = total;
    let mut scale = 1.0f32;
    launch(
        &kernels.attention,
        stream,
        blocks(product(total as usize, heads as usize)?)?,
        0,
        &mut [
            arg(&mut q),
            arg(&mut k),
            arg(&mut v),
            arg(&mut out),
            arg(&mut cu_seqlens),
            arg(&mut total),
            arg(&mut batch),
            arg(&mut heads),
            arg(&mut kv_heads),
            arg(&mut dim),
            arg(&mut max_q),
            arg(&mut scale),
        ],
    )
}

fn gated_gelu_split(
    kernels: &MediaKernels,
    stream: &Stream,
    gate: &DeviceBuffer<Bf16>,
    up: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let count = product(rows, width)?;
    if output.len() < count {
        return Err(infer("vision gated GELU output buffer is too small"));
    }
    let mut gate = gate.cu_deviceptr();
    let mut up = up.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.gated_gelu_split,
        stream,
        blocks(count)?,
        0,
        &mut [
            arg(&mut gate),
            arg(&mut up),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_pool(
    kernels: &MediaKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    height: usize,
    width: usize,
    pool: usize,
    hidden: usize,
) -> Result<(), BackendError> {
    let token_rows = product(height / pool, width / pool)?;
    let count = product(token_rows, hidden)?;
    if output.len() < count {
        return Err(infer("vision pooled buffer is too small"));
    }
    let mut input = input.cu_deviceptr();
    let mut out = output.cu_deviceptr();
    let mut height = u32_size(height)?;
    let mut width = u32_size(width)?;
    let mut pool = u32_size(pool)?;
    let mut hidden = u32_size(hidden)?;
    launch(
        &kernels.vision_pool,
        stream,
        blocks(count)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut out),
            arg(&mut height),
            arg(&mut width),
            arg(&mut pool),
            arg(&mut hidden),
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
        return Err(infer("vision residual add buffer is too small"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::media::{ImageDescriptor, MediaFamily, PackedSequences};
    use infinity_v3_backend_core::ImagePatches;

    fn config() -> VisionConfig {
        VisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            layers: 16,
            heads: 12,
            kv_heads: 12,
            head_dim: 64,
            patch_size: 16,
            pooling: 3,
            max_positions: 10240,
            position_table_rows: 10240,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            activation: VisionActivation::GeluTanh,
            standardize: false,
            use_clipped_linears: false,
            output_width: 512,
        }
    }

    fn descriptor(grid: [usize; 3]) -> ImageDescriptor {
        let [_, height, width] = grid;
        let patches = ImagePatches {
            pixels: vec![0.0; height * width * 768],
            grid_thw: grid,
            patch_dim: 768,
            merge_size: 3,
        };
        let tokens = (height / 3) * (width / 3);
        let cumulative = [0u32, u32::try_from(tokens).unwrap()];
        let sequences = PackedSequences::new(&cumulative).unwrap();
        ImageDescriptor::new(MediaFamily::EmbeddingGemma2, &patches, 0, &sequences).unwrap()
    }

    #[test]
    fn config_geometry_matches_the_released_tower() {
        let config = config();
        assert_eq!(config.patch_dim(), 768);
        assert_eq!(config.token_rows([1, 48, 48]).unwrap(), 256);
        assert_eq!(config.rope_inverse().len(), 16);
        assert_eq!(config.rope_inverse()[0], 1.0);
    }

    #[test]
    fn descriptor_maps_a_single_image_to_pooled_rows() {
        let config = config();
        let descriptor = descriptor([1, 48, 48]);
        assert_eq!(descriptor.patch_rows, 2304);
        assert_eq!(descriptor.token_rows, 256);
        assert_eq!(descriptor.token_count, 256);
        let geometry = validate_geometry(&config, &descriptor, 2304 * 768).unwrap();
        assert_eq!(geometry.patch_rows, 2304);
        assert_eq!(geometry.token_rows, 256);
        assert_eq!(geometry.height, 48);
        assert_eq!(geometry.width, 48);
    }

    #[test]
    fn invalid_grid_and_pixel_lengths_are_rejected() {
        let config = config();
        let descriptor = descriptor([1, 48, 48]);
        assert!(validate_geometry(&config, &descriptor, 2304 * 768 - 1).is_err());
        assert!(validate_geometry(&config, &descriptor, 2304 * 768 + 1).is_err());
        let mut bad_grid = descriptor.clone();
        bad_grid.grid_thw = [1, 47, 48];
        assert!(validate_geometry(&config, &bad_grid, 2304 * 768).is_err());
        let mut multi_frame = descriptor.clone();
        multi_frame.grid_thw = [2, 48, 48];
        assert!(validate_geometry(&config, &multi_frame, 2304 * 768).is_err());
    }
}
