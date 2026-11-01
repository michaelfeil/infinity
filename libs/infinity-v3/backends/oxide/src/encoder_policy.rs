// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared biasless prenorm policy sharing encoder arenas and primitive math.
//!
//! The retained stream is `Work::x`; `temp` holds only the latest branch.
//! `attention` is normalized-source scratch until QKV has consumed it, then
//! becomes FA4 output. Every reuse follows the same ordered CUDA stream.
#[cfg(test)]
use crate::native_rope_tables::inverse_frequencies;
use crate::native_rope_tables::Tables;
use crate::{
    arg, blocks,
    cublas::Blas,
    fa4, infer, launch, launch_threads,
    lt::ShapeError,
    math,
    modern_encoder_model::{ModernConfig, ModernWeights},
    packed_attention_layout, product, start,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
    PackedAttention, Work,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmbeddingInputs {
    WordsAndTypes,
    WordsOnly,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttentionLayer {
    Global,
    Local,
}
/// Inclusive native local radius; a centered interior row sees129 positions.
pub(crate) fn attention_schedule(
    layers: usize,
    every: usize,
) -> Result<Vec<AttentionLayer>, BackendError> {
    if layers == 0 || every == 0 {
        return Err(start("Modern attention schedule must be positive"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(layers).map_err(start)?;
    for layer in 0..layers {
        result.push(if layer % every == 0 {
            AttentionLayer::Global
        } else {
            AttentionLayer::Local
        });
    }
    Ok(result)
}

struct Kernels {
    table_init: Arc<Function>,
    word_norm: Function,
    prenorm: Function,
    rope: Function,
    gate: Function,
    pool: Function,
    mean: Function,
    raw_cast: Function,
}
impl Kernels {
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| start("Modern requires INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = Path::new(&path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&path).map_err(start)?;
        let module = context.load_module_from_image(&bytes).map_err(start)?;
        Ok(Self {
            table_init: Arc::new(
                module
                    .load_function("infinity_base_rope_tables_64_bf16")
                    .map_err(start)?,
            ),
            word_norm: module
                .load_function("infinity_embedding_word_norm_768_candle")
                .map_err(start)?,
            prenorm: module
                .load_function("infinity_rounded_residual_norm_768_candle")
                .map_err(start)?,
            rope: module
                .load_function("infinity_rope_neox_hfma")
                .map_err(start)?,
            gate: module
                .load_function("infinity_gated_gelu_up_first")
                .map_err(start)?,
            pool: module.load_function("infinity_pool").map_err(start)?,
            mean: module.load_function("infinity_mean_pool").map_err(start)?,
            raw_cast: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
        })
    }
}

struct Layer {
    qkv: DeviceBuffer<Bf16>,
    attention: DeviceBuffer<Bf16>,
    up_gate: DeviceBuffer<Bf16>,
    output: DeviceBuffer<Bf16>,
    attention_gamma: Option<DeviceBuffer<Bf16>>,
    mlp_gamma: DeviceBuffer<Bf16>,
    attention_kind: AttentionLayer,
}
struct Weights {
    word: DeviceBuffer<Bf16>,
    embedding_gamma: DeviceBuffer<Bf16>,
    final_gamma: DeviceBuffer<Bf16>,
    layers: Vec<Layer>,
}
/// Retain every completed upload if a later upload and fallback drain fail.
/// No allocation is extracted from this owner until the stream has drained.
struct PendingUploads {
    stream: Arc<Stream>,
    buffers: Option<Vec<DeviceBuffer<Bf16>>>,
}
impl PendingUploads {
    fn new(stream: Arc<Stream>, count: usize) -> Result<Self, BackendError> {
        let mut buffers = Vec::new();
        buffers.try_reserve_exact(count).map_err(start)?;
        Ok(Self {
            stream,
            buffers: Some(buffers),
        })
    }
    fn push(
        &mut self,
        uploader: &mut WeightUploader,
        sources: &[&crate::model::WeightSource],
    ) -> Result<(), BackendError> {
        let buffers = self
            .buffers
            .as_mut()
            .ok_or_else(|| start("Missing Modern upload owner"))?;
        buffers.try_reserve(1).map_err(start)?;
        buffers.push(uploader.upload(sources)?);
        Ok(())
    }
    fn finish(
        mut self,
        uploader: &WeightUploader,
    ) -> Result<Vec<DeviceBuffer<Bf16>>, BackendError> {
        uploader.finish()?;
        self.buffers
            .take()
            .ok_or_else(|| start("Missing Modern uploaded weights"))
    }
}
impl Drop for PendingUploads {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            if let Some(buffers) = self.buffers.take() {
                std::mem::forget(buffers);
            }
        }
    }
}
impl Weights {
    fn upload(
        host: ModernWeights,
        schedule: Vec<AttentionLayer>,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        let count = host
            .layers
            .len()
            .checked_mul(6)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| start("Modern weight count overflow"))?;
        let mut pending = PendingUploads::new(stream.clone(), count)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        pending.push(&mut uploader, &[&host.word])?;
        pending.push(&mut uploader, &[&host.embedding_gamma])?;
        pending.push(&mut uploader, &[&host.final_gamma])?;
        for layer in &host.layers {
            pending.push(&mut uploader, &[&layer.qkv])?;
            pending.push(&mut uploader, &[&layer.attention])?;
            pending.push(&mut uploader, &[&layer.up_gate[0], &layer.up_gate[1]])?;
            pending.push(&mut uploader, &[&layer.output])?;
            if let Some(gamma) = &layer.attention_gamma {
                pending.push(&mut uploader, &[gamma])?;
            }
            pending.push(&mut uploader, &[&layer.mlp_gamma])?;
        }
        let uploaded = pending.finish(&uploader)?;
        let mut buffers = uploaded.into_iter();
        let next = |buffers: &mut std::vec::IntoIter<DeviceBuffer<Bf16>>| {
            buffers
                .next()
                .ok_or_else(|| start("Modern uploaded weight layout mismatch"))
        };
        let word = next(&mut buffers)?;
        let embedding_gamma = next(&mut buffers)?;
        let final_gamma = next(&mut buffers)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(host.layers.len()).map_err(start)?;
        if schedule.len() != host.layers.len() {
            return Err(start("Modern layer schedule mismatch"));
        }
        for (host, attention_kind) in host.layers.iter().zip(schedule) {
            layers.push(Layer {
                qkv: next(&mut buffers)?,
                attention: next(&mut buffers)?,
                up_gate: next(&mut buffers)?,
                output: next(&mut buffers)?,
                attention_gamma: if host.attention_gamma.is_some() {
                    Some(next(&mut buffers)?)
                } else {
                    None
                },
                mlp_gamma: next(&mut buffers)?,
                attention_kind,
            });
        }
        if buffers.next().is_some() {
            return Err(start("Unexpected Modern uploaded weights"));
        }
        Ok(Self {
            word,
            embedding_gamma,
            final_gamma,
            layers,
        })
    }
}
struct Plans {
    qkv: math::LinearPlan<Bf16>,
    attention: math::LinearPlan<Bf16>,
    up_gate: math::LinearPlan<Bf16>,
    output: math::LinearPlan<Bf16>,
}
impl Plans {
    fn new(rows: usize, c: &ModernConfig) -> Result<Self, BackendError> {
        let g = &c.geometry;
        let gate = product(g.intermediate_size, 2)?;
        Ok(Self {
            qkv: math::LinearPlan::contiguous(rows, g.hidden_size, product(g.hidden_size, 3)?)
                .map_err(infer)?,
            attention: math::LinearPlan::contiguous(rows, g.hidden_size, g.hidden_size)
                .map_err(infer)?,
            up_gate: math::LinearPlan::contiguous(rows, g.hidden_size, gate).map_err(infer)?,
            output: math::LinearPlan::contiguous(rows, g.intermediate_size, g.hidden_size)
                .map_err(infer)?,
        })
    }
}

pub(crate) struct ModernEncoder {
    pub(crate) config: ModernConfig,
    kernels: Kernels,
    weights: Weights,
    global: Tables,
    local: Tables,
    global_attention: fa4::Fa4,
    local_attention: fa4::Fa4,
    layout: PackedAttention,
    rows: Option<usize>,
    plans: Option<Plans>,
}
impl ModernEncoder {
    pub(crate) fn new(
        host: ModernWeights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(start)?;
        let config = host.config.clone();
        let mode = std::env::var(infinity_v3_config::infinity_env!("OXIDE_ATTENTION"))
            .unwrap_or_else(|_| "auto".into());
        if !matches!(mode.as_str(), "auto" | "fa4") {
            return Err(start("Modern requires FA4 global/local attention"));
        }
        let root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
            .ok_or_else(|| start("Modern requires INFINITY_FA4_ROOT"))?;
        let spec = fa4::AttentionSpec {
            heads: 12,
            kv_heads: 12,
            head_dim: 64,
            causal: false,
            window: fa4::Window::Global,
            dtype: fa4::DType::Bf16,
            scale: fa4::Scale::RsqrtHeadDim,
        };
        let global_attention = fa4::Fa4::new(context.clone(), Path::new(&root), spec)?;
        let local_radius = i32::try_from(config.local_radius).map_err(start)?;
        let local_attention = fa4::Fa4::new(
            context.clone(),
            Path::new(&root),
            fa4::AttentionSpec {
                window: fa4::Window::Local {
                    left: local_radius,
                    right: local_radius,
                },
                ..spec
            },
        )?;
        let layout = packed_attention_layout(&config.geometry)?
            .ok_or_else(|| start("Modern requires aligned packed attention rows"))?;
        let kernels = Kernels::load(context)?;
        let global = Tables::new(
            config.global_theta,
            config.geometry.max_position_embeddings,
            stream,
            kernels.table_init.clone(),
        )?;
        let local = Tables::new(
            config.local_theta,
            config.geometry.max_position_embeddings,
            stream,
            kernels.table_init.clone(),
        )?;
        let schedule = attention_schedule(config.geometry.num_hidden_layers, config.global_every)?;
        let weights = Weights::upload(host, schedule, stream)?;
        Ok(Self {
            config,
            kernels,
            weights,
            global,
            local,
            global_attention,
            local_attention,
            layout,
            rows: None,
            plans: None,
        })
    }
    pub(crate) fn preflight(&mut self, rows: usize) -> Result<(), ShapeError> {
        if self.rows != Some(rows) {
            let plans = Plans::new(rows, &self.config).map_err(ShapeError::Rejected)?;
            self.plans = Some(plans);
            self.rows = Some(rows);
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        stream: &Stream,
        mut input: u64,
        mut residual: u64,
        gamma: &DeviceBuffer<Bf16>,
        mut normalized: u64,
        mut summed: u64,
        mut rows: u32,
    ) -> Result<(), BackendError> {
        let mut gamma = gamma.cu_deviceptr();
        let mut eps = self.config.geometry.layer_norm_eps as f32;
        launch_threads(
            &self.kernels.prenorm,
            stream,
            (rows.div_ceil(4), 1, 1),
            128,
            0,
            &mut [
                arg(&mut input),
                arg(&mut residual),
                arg(&mut gamma),
                arg(&mut normalized),
                arg(&mut summed),
                arg(&mut rows),
                arg(&mut eps),
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        blas: &Blas,
        stream: &Stream,
        w: &mut Work<Bf16>,
        b: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let g = &self.config.geometry;
        let plans = self
            .plans
            .as_ref()
            .ok_or_else(|| infer("Modern projection plans are not prepared"))?;
        let mut rows = u32_size(tokens)?;
        let mut width = u32_size(g.hidden_size)?;
        let base = w.input.cu_deviceptr();
        let mut ids = base;
        let positions = base
            .checked_add(u64::try_from(product(tokens, 4)?).map_err(infer)?)
            .ok_or_else(|| infer("Modern position address overflow"))?;
        let boundaries = base
            .checked_add(u64::try_from(product(tokens, 12)?).map_err(infer)?)
            .ok_or_else(|| infer("Modern boundary address overflow"))?;
        let selected = boundaries
            .checked_add(u64::try_from(product(b.cumulative_seq_lengths.len(), 4)?).map_err(infer)?)
            .ok_or_else(|| infer("Modern selection address overflow"))?;
        let mut word = self.weights.word.cu_deviceptr();
        let mut gamma = self.weights.embedding_gamma.cu_deviceptr();
        let mut residual = w.x.cu_deviceptr();
        let mut eps = g.layer_norm_eps as f32;
        launch_threads(
            &self.kernels.word_norm,
            stream,
            (rows.div_ceil(4), 1, 1),
            128,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut word),
                arg(&mut gamma),
                arg(&mut residual),
                arg(&mut rows),
                arg(&mut eps),
            ],
        )?;
        for layer in &self.weights.layers {
            if let Some(gamma) = &layer.attention_gamma {
                self.norm(
                    stream,
                    w.temp.cu_deviceptr(),
                    residual,
                    gamma,
                    w.attention.cu_deviceptr(),
                    residual,
                    rows,
                )?;
            }
            // SAFETY: plans bind disjoint checked buffers retained by Engine until
            // stream completion. Every reuse below follows its earlier consumers.
            unsafe {
                blas.linear(
                    &plans.qkv,
                    if layer.attention_gamma.is_some() {
                        &w.attention
                    } else {
                        &w.x
                    },
                    &layer.qkv,
                    &mut w.qkv,
                )?;
            }
            let (tables, provider) = match layer.attention_kind {
                AttentionLayer::Global => (&self.global, &self.global_attention),
                AttentionLayer::Local => (&self.local, &self.local_attention),
            };
            let mut q = w.qkv.cu_deviceptr();
            let mut k = q
                .checked_add(self.layout.k_offset)
                .ok_or_else(|| infer("Modern K address overflow"))?;
            let mut cosine = tables.cosine.cu_deviceptr();
            let mut sine = tables.sine.cu_deviceptr();
            let mut pos = positions;
            let mut heads = 12u32;
            let mut kv_heads = 12u32;
            let mut head_dim = 64u32;
            let mut rotary_dim = 64u32;
            let mut q_stride = u64::try_from(product(g.hidden_size, 3)?).map_err(infer)?;
            let mut k_stride = q_stride;
            let mut table_stride = tables.stride;
            launch(
                &self.kernels.rope,
                stream,
                blocks(product(product(tokens, 24)?, 32)?)?,
                0,
                &mut [
                    arg(&mut q),
                    arg(&mut k),
                    arg(&mut cosine),
                    arg(&mut sine),
                    arg(&mut pos),
                    arg(&mut rows),
                    arg(&mut heads),
                    arg(&mut kv_heads),
                    arg(&mut head_dim),
                    arg(&mut rotary_dim),
                    arg(&mut q_stride),
                    arg(&mut k_stride),
                    arg(&mut table_stride),
                ],
            )?;
            let attention = fa4::AttentionBatch {
                q,
                k,
                v: q.checked_add(self.layout.v_offset)
                    .ok_or_else(|| infer("Modern V address overflow"))?,
                out: w.attention.cu_deviceptr(),
                q_offsets: boundaries,
                k_offsets: boundaries,
                q_tokens: tokens,
                k_tokens: tokens,
                batch: b
                    .cumulative_seq_lengths
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| infer("Missing Modern sequence boundaries"))?,
                max_q: max_length,
                strides: self.layout.strides,
            };
            // SAFETY: the caller bound context once. QKV/attention, tables and
            // providers remain owned; ordered-stream consumption precedes reuse.
            unsafe {
                provider.forward_batch_bound(stream, &attention)?;
            }
            // SAFETY: retained disjoint matrix buffers, same ordered stream.
            unsafe {
                blas.linear(
                    &plans.attention,
                    &w.attention,
                    &layer.attention,
                    &mut w.temp,
                )?;
            }
            self.norm(
                stream,
                w.temp.cu_deviceptr(),
                residual,
                &layer.mlp_gamma,
                w.attention.cu_deviceptr(),
                residual,
                rows,
            )?;
            let scratch = (*w.gated)
                .as_mut()
                .ok_or_else(|| infer("Missing Modern packed gate scratch"))?;
            // SAFETY: checked no-bias XW^T, complete retained source/destination.
            unsafe {
                blas.linear(&plans.up_gate, &w.attention, &layer.up_gate, scratch)?;
            }
            let mut input = scratch.cu_deviceptr();
            let mut out = w.intermediate.cu_deviceptr();
            let mut intermediate = u32_size(g.intermediate_size)?;
            launch(
                &self.kernels.gate,
                stream,
                blocks(product(tokens, g.intermediate_size)?)?,
                0,
                &mut [
                    arg(&mut input),
                    arg(&mut out),
                    arg(&mut rows),
                    arg(&mut intermediate),
                ],
            )?;
            // SAFETY: the branch is written after its previous norm consumer.
            unsafe {
                blas.linear(&plans.output, &w.intermediate, &layer.output, &mut w.temp)?;
            }
        }
        self.norm(
            stream,
            w.temp.cu_deviceptr(),
            residual,
            &self.weights.final_gamma,
            w.attention.cu_deviceptr(),
            0,
            rows,
        )?;
        let mut source = w.attention.cu_deviceptr();
        let mut bounds = boundaries;
        let mut indices = selected;
        let mut out = w.pooled.cu_deviceptr();
        let mut count = u32_size(b.pooled_indices.len())?;
        let mut mode = match pool {
            Pool::Cls => 0u32,
            Pool::Mean => 1,
            Pool::LastToken => 2,
            Pool::Splade => return Err(infer("Modern SPLADE unsupported")),
        };
        if count > 0 {
            if mode == 1 {
                launch(
                    &self.kernels.mean,
                    stream,
                    (count, u32_size(g.hidden_size.div_ceil(32))?, 1),
                    0,
                    &mut [
                        arg(&mut source),
                        arg(&mut bounds),
                        arg(&mut indices),
                        arg(&mut out),
                        arg(&mut count),
                        arg(&mut width),
                    ],
                )?;
            } else {
                launch(
                    &self.kernels.pool,
                    stream,
                    blocks(product(count as usize, g.hidden_size)?)?,
                    0,
                    &mut [
                        arg(&mut source),
                        arg(&mut bounds),
                        arg(&mut indices),
                        arg(&mut out),
                        arg(&mut count),
                        arg(&mut width),
                        arg(&mut mode),
                    ],
                )?;
            }
        }
        if !b.raw_indices.is_empty() {
            crate::enqueue_raw_output(
                &self.kernels.raw_cast,
                stream,
                w.attention.cu_deviceptr(),
                w.pooled.cu_deviceptr(),
                selected,
                b,
                g.hidden_size,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modern_inverse_frequencies_preserve_native_f32_even_dimensions() {
        let local = inverse_frequencies(10000.0).unwrap();
        let global = inverse_frequencies(160000.0).unwrap();
        assert_eq!(local.len(), 32);
        assert_eq!(global.len(), 32);
        assert_eq!(local[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(local[16].to_bits(), 0.01f32.to_bits());
        assert_eq!(global[16].to_bits(), 0.0025f32.to_bits());
        assert!(local.windows(2).all(|pair| pair[1] < pair[0]));
        assert!(inverse_frequencies(f64::NAN).is_err());
        assert!(inverse_frequencies(f64::INFINITY).is_err());
        assert!(inverse_frequencies(1.0).is_err());
    }
    #[test]
    fn modern_schedule_starts_global_and_preserves_period() {
        let schedule = attention_schedule(22, 3).unwrap();
        let globals = schedule
            .iter()
            .enumerate()
            .filter_map(|(i, k)| (*k == AttentionLayer::Global).then_some(i))
            .collect::<Vec<_>>();
        assert_eq!(globals, vec![0, 3, 6, 9, 12, 15, 18, 21]);
        assert_eq!(
            schedule
                .iter()
                .filter(|&&k| k == AttentionLayer::Local)
                .count(),
            14
        );
        assert_eq!(
            attention_schedule(2, 1).unwrap(),
            vec![AttentionLayer::Global; 2]
        );
        assert!(attention_schedule(2, 0).is_err());
        assert!(attention_schedule(0, 3).is_err());
    }
}

#[cfg(test)]
#[path = "../tests/support/modern_other_controls.rs"]
mod primitive_other;
#[cfg(test)]
#[path = "../tests/support/modern_schema.rs"]
mod primitive_schema;

#[cfg(test)]
#[path = "../tests/support/modern_norm_control.rs"]
mod primitive_norm;
