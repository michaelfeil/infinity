// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Graph-free BF16 Gemma3 text executor.
//!
//! This is the distinct Gemma3 math loop over the shared decoder workspace,
//! cuBLAS plans, weight uploader, host I/O lease and raw-output bridge. It
//! implements the exact first BF16 scope only: hidden 640, intermediate 2048,
//! four query heads, one key/value head, head dimension 256, 18 layers, a
//! 262144-entry vocabulary and causal local/global attention.

use super::decoder::{selected_readout, DecoderGeometry, DecoderWork, SelectedReadout};
use super::{arg, blocks, fa4, infer, launch, product, start, u32_size, LtProvider, WorkKey};
use crate::cublas::Blas;
use crate::gemma3_model::{Gemma3Config, Gemma3Weights};
use crate::gemma3_prepared::{
    upload_offset_norm, Gemma3KernelSet, Gemma3PreparedMath, GEMMA3_KERNEL_EXPORTS,
};
use crate::gemma3_tables::Tables;
use crate::lt::ShapeError;
use crate::math::{refresh_plan_rows, LinearPlan};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Two full-position cosine/sine tables plus inverse frequencies. The 12B/27B
/// checkpoints use 128K positions, so the dual-table budget is a quarter GiB.
const ROPE_TABLE_BYTE_LIMIT: usize = 512 * 1024 * 1024;
const PLAIN_SEED_ROWS: usize = 128;

/// The native bridge uses its packed BF16 Candle mean only while the selected
/// span count fits `u16`; larger selections retain the generic F32 fallback.
const NATIVE_MEAN_MAX_SELECTIONS: usize = u16::MAX as usize;

/// Mean-only launch plan for the packed BF16 native reduction, or `None` when
/// the generic F32 pool must run. Grid=(count,ceil(hidden/8),1), block256.
fn native_mean_plan(mode: u32, selections: usize, hidden: usize) -> Option<(u32, u32, u32)> {
    if mode != 1 || selections == 0 || selections > NATIVE_MEAN_MAX_SELECTIONS {
        return None;
    }
    Some((
        u32::try_from(selections).ok()?,
        u32::try_from(hidden.div_ceil(8)).ok()?,
        1,
    ))
}

struct Gemma3Kernels {
    set: Gemma3KernelSet,
    gather_scaled: Function,
    selected_rows: Function,
    pool: Function,
    native_mean: Function,
    bf16_to_f32: Function,
    rope_tables: Arc<Function>,
}

impl Gemma3Kernels {
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).map_err(|e| start(format!("{}: {e}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        let set = Gemma3KernelSet::from_functions(
            module
                .load_function(GEMMA3_KERNEL_EXPORTS[0])
                .map_err(start)?,
            module
                .load_function(GEMMA3_KERNEL_EXPORTS[1])
                .map_err(start)?,
            module
                .load_function(GEMMA3_KERNEL_EXPORTS[2])
                .map_err(start)?,
            module
                .load_function("infinity_gemma2_gated_gelu_bf16")
                .map_err(start)?,
        );
        let rope_tables = Arc::new(
            module
                .load_function("infinity_gemma2_rope_tables_bf16")
                .map_err(start)?,
        );
        Ok(Self {
            set,
            gather_scaled: module
                .load_function("infinity_embedding_gather_scaled")
                .map_err(start)?,
            selected_rows: module
                .load_function("infinity_selected_rows")
                .map_err(start)?,
            pool: module.load_function("infinity_pool").map_err(start)?,
            native_mean: module
                .load_function("infinity_mean_pool_bf16_native")
                .map_err(start)?,
            bf16_to_f32: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
            rope_tables,
        })
    }
}

struct Gemma3LayerBuffers {
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    pre_feedforward_norm: DeviceBuffer<f32>,
    post_feedforward_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    qkv: DeviceBuffer<Bf16>,
    attention: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
}

struct Gemma3WeightBuffers {
    embed: DeviceBuffer<Bf16>,
    final_norm: DeviceBuffer<f32>,
    layers: Vec<Gemma3LayerBuffers>,
}

fn dense_mlp(
    layer: &crate::decoder_model::DecoderLayerWeights,
    index: usize,
) -> Result<
    (
        &crate::decoder_model::DecoderLinear,
        &crate::decoder_model::DecoderLinear,
        &crate::decoder_model::DecoderLinear,
    ),
    BackendError,
> {
    match &layer.mlp {
        crate::decoder_model::DecoderMlpWeights::Dense { gate, up, down } => Ok((gate, up, down)),
        crate::decoder_model::DecoderMlpWeights::Routed(_) => Err(start(format!(
            "gemma3 layer {index} has routed MLP weights; dense Gemma3 text only"
        ))),
    }
}

impl Gemma3WeightBuffers {
    fn validate(host: &Gemma3Weights) -> Result<(), BackendError> {
        for (index, layer) in host.layers.iter().enumerate() {
            if layer.common.q_norm.is_none() || layer.common.k_norm.is_none() {
                return Err(start(format!(
                    "gemma3 layer {index} requires a complete q_norm/k_norm pair"
                )));
            }
            let dense_mlp = dense_mlp(&layer.common, index)?;
            for (name, bias) in [
                ("q", &layer.common.q.bias),
                ("k", &layer.common.k.bias),
                ("v", &layer.common.v.bias),
                ("o", &layer.common.o.bias),
                ("gate", &dense_mlp.0.bias),
                ("up", &dense_mlp.1.bias),
                ("down", &dense_mlp.2.bias),
            ] {
                if bias.is_some() {
                    return Err(start(format!(
                        "gemma3 layer {index} has unsupported {name} projection bias"
                    )));
                }
            }
        }
        Ok(())
    }

    fn upload(
        uploader: &mut WeightUploader,
        stream: &Arc<Stream>,
        host: &Gemma3Weights,
        config: &Gemma3Config,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let embed = uploader.upload(&[&host.embed])?;
        let final_norm = upload_offset_norm(stream, &host.norm, hidden)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(host.layers.len()).map_err(start)?;
        for layer in &host.layers {
            let q_norm = layer
                .common
                .q_norm
                .as_ref()
                .ok_or_else(|| start("gemma3 missing q_norm"))?;
            let k_norm = layer
                .common
                .k_norm
                .as_ref()
                .ok_or_else(|| start("gemma3 missing k_norm"))?;
            let (gate, up, down) = dense_mlp(&layer.common, layers.len())?;
            layers.push(Gemma3LayerBuffers {
                input_norm: upload_offset_norm(stream, &layer.common.input_norm, hidden)?,
                post_attention_norm: upload_offset_norm(
                    stream,
                    &layer.common.post_attention_norm,
                    hidden,
                )?,
                pre_feedforward_norm: upload_offset_norm(
                    stream,
                    &layer.pre_feedforward_norm,
                    hidden,
                )?,
                post_feedforward_norm: upload_offset_norm(
                    stream,
                    &layer.post_feedforward_norm,
                    hidden,
                )?,
                q_norm: upload_offset_norm(stream, q_norm, head_dim)?,
                k_norm: upload_offset_norm(stream, k_norm, head_dim)?,
                qkv: uploader.upload(&[
                    &layer.common.q.weight,
                    &layer.common.k.weight,
                    &layer.common.v.weight,
                ])?,
                attention: uploader.upload(&[&layer.common.o.weight])?,
                gate_up: uploader.upload(&[&gate.weight, &up.weight])?,
                down: uploader.upload(&[&down.weight])?,
            });
        }
        Ok(Self {
            embed,
            final_norm,
            layers,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PlanRows {
    full: Option<usize>,
    readout: Option<usize>,
}

struct Gemma3FullPlans {
    qkv: LinearPlan<Bf16>,
    attention: LinearPlan<Bf16>,
    gate_up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
}

struct Gemma3ReadoutPlans {
    attention: LinearPlan<Bf16>,
    gate_up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
}

struct Gemma3Plans {
    rows: PlanRows,
    full: Gemma3FullPlans,
    readout: Gemma3ReadoutPlans,
}

impl Gemma3Plans {
    fn new(config: &Gemma3Config, geometry: &DecoderGeometry) -> Result<Self, BackendError> {
        let gate_up_width = config
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("gemma3 gate/up width overflow"))?;
        let full = Gemma3FullPlans {
            qkv: LinearPlan::contiguous(PLAIN_SEED_ROWS, geometry.hidden, geometry.qkv_width)
                .map_err(infer)?,
            attention: LinearPlan::contiguous(PLAIN_SEED_ROWS, geometry.q_width, geometry.hidden)
                .map_err(infer)?,
            gate_up: LinearPlan::contiguous(PLAIN_SEED_ROWS, geometry.hidden, gate_up_width)
                .map_err(infer)?,
            down: LinearPlan::contiguous(
                PLAIN_SEED_ROWS,
                config.intermediate_size,
                geometry.hidden,
            )
            .map_err(infer)?,
        };
        let readout = Gemma3ReadoutPlans {
            attention: LinearPlan::contiguous(PLAIN_SEED_ROWS, geometry.q_width, geometry.hidden)
                .map_err(infer)?,
            gate_up: LinearPlan::contiguous(PLAIN_SEED_ROWS, geometry.hidden, gate_up_width)
                .map_err(infer)?,
            down: LinearPlan::contiguous(
                PLAIN_SEED_ROWS,
                config.intermediate_size,
                geometry.hidden,
            )
            .map_err(infer)?,
        };
        if full.gate_up.output_layout().shape()[1] > geometry.projection_width
            || full.down.input_layout().shape()[1] > geometry.activation_width
        {
            return Err(start(
                "gemma3 plan widths disagree with the shared decoder geometry",
            ));
        }
        Ok(Self {
            rows: PlanRows::default(),
            full,
            readout,
        })
    }

    fn set_full_rows(&mut self, rows: usize) -> Result<(), ShapeError> {
        refresh_plan_rows(&mut self.rows.full, rows, || {
            let rejected = |error| ShapeError::Rejected(infer(error));
            self.full.qkv.set_rows(rows).map_err(rejected)?;
            self.full.attention.set_rows(rows).map_err(rejected)?;
            self.full.gate_up.set_rows(rows).map_err(rejected)?;
            self.full.down.set_rows(rows).map_err(rejected)?;
            Ok(())
        })
    }

    fn set_readout_rows(&mut self, rows: usize) -> Result<(), ShapeError> {
        refresh_plan_rows(&mut self.rows.readout, rows, || {
            let rejected = |error| ShapeError::Rejected(infer(error));
            self.readout.attention.set_rows(rows).map_err(rejected)?;
            self.readout.gate_up.set_rows(rows).map_err(rejected)?;
            self.readout.down.set_rows(rows).map_err(rejected)?;
            Ok(())
        })
    }
}

fn admit_scope(config: &Gemma3Config) -> Result<(), BackendError> {
    if config.num_hidden_layers == 0 || config.vocab_size == 0 {
        return Err(start("Gemma3 layer and vocabulary counts must be positive"));
    }
    let mut has_local = false;
    let mut has_global = false;
    for index in 0..config.num_hidden_layers {
        if config.layer_policy(index)?.local {
            has_local = true;
        } else {
            has_global = true;
        }
    }
    let mut required = Vec::new();
    if has_local {
        required.push((config.rope_local_base_freq, 1.0));
    }
    if has_global {
        required.push((config.rope_theta, config.global_rope_factor));
    }
    for (theta, factor) in required {
        let bytes = crate::gemma3_tables::required_bytes(
            theta,
            factor,
            config.max_position_embeddings,
            config.head_dim,
        )?;
        if bytes > ROPE_TABLE_BYTE_LIMIT / 2 {
            return Err(start(
                "Gemma3 dual rotary tables exceed the model byte budget",
            ));
        }
    }
    Ok(())
}

/// Validate the complete first executor policy before any CUDA query/upload.
pub(crate) fn validate_admission(config: &Gemma3Config) -> Result<(), BackendError> {
    Gemma3PreparedMath::admit(config)?;
    admit_scope(config)
}

pub(crate) struct Gemma3Model {
    pub(crate) config: Gemma3Config,
    geometry: DecoderGeometry,
    math: Gemma3PreparedMath,
    weights: Gemma3WeightBuffers,
    plans: Gemma3Plans,
    kernels: Gemma3Kernels,
    global_fa4: Option<fa4::Fa4>,
    local_fa4: Option<fa4::Fa4>,
    global_rope: Option<Tables>,
    local_rope: Option<Tables>,
    local_layers: Vec<bool>,
    _context: Arc<Context>,
    _stream: Arc<Stream>,
}

impl Gemma3Model {
    pub(crate) fn new(
        host: Gemma3Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        _lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        let config = host.config.clone();
        let math = Gemma3PreparedMath::admit(&config)?;
        admit_scope(&config)?;
        let geometry = DecoderGeometry::from_dimensions(
            config.hidden_size,
            config.intermediate_size,
            config.q_width(),
            config.kv_width(),
            config.head_dim,
        )?;
        if geometry.hidden != config.hidden_size
            || geometry.q_width != config.q_width()
            || geometry.kv_width != config.kv_width()
            || geometry.qkv_width != math.qk_stride()
            || geometry.projection_width
                < config
                    .intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| start("gemma3 gate/up width overflow"))?
            || geometry.activation_width < config.intermediate_size
        {
            return Err(start(
                "gemma3 geometry disagrees with the prepared kernel widths",
            ));
        }
        if host.layers.len() != config.num_hidden_layers {
            return Err(start(format!(
                "gemma3 checkpoint has {} layers, config declares {}",
                host.layers.len(),
                config.num_hidden_layers
            )));
        }
        Gemma3WeightBuffers::validate(&host)?;

        let mut local_layers = Vec::new();
        local_layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        let mut first_local = None;
        let mut first_global = None;
        for index in 0..config.num_hidden_layers {
            let policy = config.layer_policy(index)?;
            local_layers.push(policy.local);
            if policy.local {
                if first_local.is_none() {
                    first_local = Some(policy);
                }
            } else if first_global.is_none() {
                first_global = Some(policy);
            }
        }
        let local_policy = first_local;
        let global_policy = first_global;
        if local_policy.is_none() && global_policy.is_none() {
            return Err(start("Gemma3 model has no attention layers"));
        }

        let kernels = Gemma3Kernels::load(context)?;
        let fa4_root = if local_policy.is_some() || global_policy.is_some() {
            Some(
                std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
                    .ok_or_else(|| start("Gemma3 attention requires INFINITY_FA4_ROOT"))?,
            )
        } else {
            None
        };
        let fa4_root = fa4_root.as_deref().map(Path::new);
        let global_fa4 = if let Some(policy) = global_policy {
            let spec = fa4::AttentionSpec {
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                causal: config.is_causal,
                window: fa4::Window::Global,
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::Explicit(policy.attention_scale),
            };
            if !fa4::Fa4::supports(&spec) {
                return Err(start(format!(
                    "Gemma3 global attention H{} KV{} d{} causal={} has no qualified FA4 export",
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.is_causal
                )));
            }
            let root =
                fa4_root.ok_or_else(|| start("Gemma3 attention requires INFINITY_FA4_ROOT"))?;
            Some(fa4::Fa4::new(context.clone(), root, spec)?)
        } else {
            None
        };
        let local_fa4 = if let Some(policy) = local_policy {
            let spec = fa4::AttentionSpec {
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                causal: config.is_causal,
                window: fa4::Window::Local {
                    left: policy.window_left,
                    right: policy.window_right,
                },
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::Explicit(policy.attention_scale),
            };
            if !fa4::Fa4::supports(&spec) {
                return Err(start(format!(
                    "Gemma3 local attention H{} KV{} d{} window {}/{} has no qualified FA4 export",
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    policy.window_left,
                    policy.window_right
                )));
            }
            let root =
                fa4_root.ok_or_else(|| start("Gemma3 attention requires INFINITY_FA4_ROOT"))?;
            Some(fa4::Fa4::new(context.clone(), root, spec)?)
        } else {
            None
        };

        let global_rope = if let Some(policy) = global_policy {
            Some(Tables::new(
                policy.theta,
                policy.rope_factor,
                config.max_position_embeddings,
                config.head_dim,
                ROPE_TABLE_BYTE_LIMIT / 2,
                stream,
                kernels.rope_tables.clone(),
            )?)
        } else {
            None
        };
        let local_rope = if let Some(policy) = local_policy {
            Some(Tables::new(
                policy.theta,
                policy.rope_factor,
                config.max_position_embeddings,
                config.head_dim,
                ROPE_TABLE_BYTE_LIMIT / 2,
                stream,
                kernels.rope_tables.clone(),
            )?)
        } else {
            None
        };
        let table_elements = product(config.max_position_embeddings, config.head_dim / 2)?;
        let table_stride = config.head_dim / 2;
        for table in [global_rope.as_ref(), local_rope.as_ref()]
            .into_iter()
            .flatten()
        {
            if usize::try_from(table.stride).map_err(start)? != table_stride
                || table.cosine.len() != table_elements
                || table.sine.len() != table_elements
            {
                return Err(start("Gemma3 rotary table geometry disagrees with config"));
            }
        }

        let mut uploader = WeightUploader::new(stream.clone())?;
        let weights = Gemma3WeightBuffers::upload(&mut uploader, stream, &host, &config)?;
        uploader.finish()?;
        let plans = Gemma3Plans::new(&config, &geometry)?;

        Ok(Self {
            config,
            geometry,
            math,
            weights,
            plans,
            kernels,
            global_fa4,
            local_fa4,
            global_rope,
            local_rope,
            local_layers,
            _context: context.clone(),
            _stream: stream.clone(),
        })
    }

    pub(crate) fn workspace_bytes(&self, key: WorkKey) -> Result<usize, BackendError> {
        DecoderWork::allocation_bytes_geometry(key, &self.geometry)
    }

    pub(crate) fn workspace(
        &self,
        stream: Arc<Stream>,
        key: WorkKey,
    ) -> Result<DecoderWork, BackendError> {
        DecoderWork::new_geometry(stream, &self.geometry, key)
    }

    pub(crate) fn preflight(
        &mut self,
        _provider: &LtProvider,
        batch: &Batch,
        pool: &Pool,
    ) -> Result<(), ShapeError> {
        if matches!(pool, Pool::Splade) {
            return Err(ShapeError::Rejected(infer("Oxide SPLADE unsupported")));
        }
        self.plans.set_full_rows(batch.input_ids.len())?;
        if batch.raw_indices.is_empty()
            && selected_readout(pool, batch.pooled_indices.len()).is_some()
        {
            let sequences = batch
                .cumulative_seq_lengths
                .len()
                .checked_sub(1)
                .filter(|count| *count > 0)
                .ok_or_else(|| ShapeError::Rejected(infer("Missing Gemma3 sequence boundaries")))?;
            self.plans.set_readout_rows(sequences)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        _lt: &LtProvider,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut DecoderWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let blas = plain_blas
            .ok_or_else(|| infer("Gemma3 biasless projections require the plain cuBLAS handle"))?;
        let pool_mode = match pool {
            Pool::Cls => 0u32,
            Pool::Mean => 1,
            Pool::LastToken => 2,
            Pool::Splade => return Err(infer("Oxide SPLADE unsupported")),
        };
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Missing Gemma3 sequence boundaries"))?;
        if tokens == 0 || sequences == 0 {
            return Err(infer("Gemma3 request requires tokens and sequences"));
        }
        let selections = batch.pooled_indices.len();
        work.check(
            tokens,
            sequences,
            super::output::rows(batch)?,
            &self.geometry,
        )?;
        let readout = if batch.raw_indices.is_empty() {
            selected_readout(pool, selections)
        } else {
            None
        };
        let readout_mode = readout.map(SelectedReadout::kernel_mode);

        let c = &self.config;
        let g = self.geometry;
        let h = g.hidden;
        let rows = u32_size(tokens)?;
        let mut width = u32_size(h)?;
        let selections_u32 = u32_size(selections)?;
        let mut readout_count = u32_size(sequences)?;
        let mut q_width_u32 = u32_size(g.q_width)?;
        let embedding_grid = blocks(product(tokens, h)?)?;
        let gather_grid = blocks(product(
            sequences,
            g.q_width
                .checked_add(h)
                .ok_or_else(|| infer("Gemma3 selected row width overflow"))?,
        )?)?;
        let cast_grid = blocks(product(selections, h)?)?;
        let strides = g.attention_strides(c.head_dim)?;
        let q_offset = u64::try_from(product(g.q_width, 2)?).map_err(infer)?;
        let k_offset = u64::try_from(product(g.kv_width, 2)?).map_err(infer)?;

        let base = work.input.cu_deviceptr();
        let id_bytes = u64::try_from(product(tokens, 4)?).map_err(infer)?;
        let type_bytes = u64::try_from(product(tokens, 12)?).map_err(infer)?;
        let positions = base
            .checked_add(id_bytes)
            .ok_or_else(|| infer("Gemma3 position address overflow"))?;
        let boundaries = base
            .checked_add(type_bytes)
            .ok_or_else(|| infer("Gemma3 boundary address overflow"))?;
        let selected = boundaries
            .checked_add(
                u64::try_from(product(batch.cumulative_seq_lengths.len(), 4)?).map_err(infer)?,
            )
            .ok_or_else(|| infer("Gemma3 selection address overflow"))?;
        let raw_selected = selected
            .checked_add(u64::try_from(product(selections, 4)?).map_err(infer)?)
            .ok_or_else(|| infer("Gemma3 raw selection address overflow"))?;

        let residual = work.residual.cu_deviceptr();
        let residual_next = work.residual_next.cu_deviceptr();
        let branch = work.branch.cu_deviceptr();
        let normalized = work.normalized.cu_deviceptr();
        let projection = work.projection.cu_deviceptr();
        let activation = work.activation.cu_deviceptr();
        let q = projection;
        let key = projection
            .checked_add(q_offset)
            .ok_or_else(|| infer("Gemma3 Q address overflow"))?;
        let v = key
            .checked_add(k_offset)
            .ok_or_else(|| infer("Gemma3 V address overflow"))?;
        let mut state = residual;
        let mut next = residual_next;

        let mut gather_ids = base;
        let mut gather_table = self.weights.embed.cu_deviceptr();
        let mut gather_out = state;
        let mut gather_tokens = rows;
        let mut gather_width = width;
        let mut gather_scale = self.math.embedding_scale();
        launch(
            &self.kernels.gather_scaled,
            stream,
            embedding_grid,
            0,
            &mut [
                arg(&mut gather_ids),
                arg(&mut gather_table),
                arg(&mut gather_out),
                arg(&mut gather_tokens),
                arg(&mut gather_width),
                arg(&mut gather_scale),
            ],
        )?;

        for (index, layer) in self.weights.layers.iter().enumerate() {
            let local = *self
                .local_layers
                .get(index)
                .ok_or_else(|| infer("Gemma3 local/global layer metadata mismatch"))?;
            let table = if local {
                self.local_rope.as_ref()
            } else {
                self.global_rope.as_ref()
            }
            .ok_or_else(|| infer("Gemma3 layer requires a missing local/global rotary table"))?;
            let cos = table.cosine.cu_deviceptr();
            let sin = table.sine.cu_deviceptr();
            let table_stride = usize::try_from(table.stride).map_err(infer)?;

            // SAFETY: every buffer belongs to the retained shared workspace,
            // the layer scale was uploaded once at load, and the engine bound
            // this context before the ordered request body.
            unsafe {
                self.kernels.set.launch_norm(
                    &self.math,
                    stream,
                    state,
                    layer.input_norm.cu_deviceptr(),
                    normalized,
                    tokens,
                    h,
                )?;
                blas.linear(
                    &self.plans.full.qkv,
                    &work.normalized,
                    &layer.qkv,
                    &mut work.projection,
                )?;
            }
            // SAFETY: Q/K remain inside the checked packed QKV slab and the
            // loaded RoPE table covers every staged position.
            unsafe {
                self.kernels.set.launch_qk(
                    &self.math,
                    stream,
                    q,
                    key,
                    layer.q_norm.cu_deviceptr(),
                    layer.k_norm.cu_deviceptr(),
                    cos,
                    sin,
                    positions,
                    tokens,
                    g.qkv_width,
                    g.qkv_width,
                    table_stride,
                )?;
            }
            let attention = fa4::AttentionBatch {
                q,
                k: key,
                v,
                out: activation,
                q_offsets: boundaries,
                k_offsets: boundaries,
                q_tokens: tokens,
                k_tokens: tokens,
                batch: sequences,
                max_q: max_length,
                strides,
            };
            // SAFETY: the selected prebound handle was qualified at load; all
            // packed Q/K/V, positions and boundaries remain retained through the
            // ordered stream and the bound context is current.
            let fa4 = if local {
                self.local_fa4.as_ref()
            } else {
                self.global_fa4.as_ref()
            }
            .ok_or_else(|| infer("Gemma3 layer requires a missing local/global FA4 provider"))?;
            unsafe {
                fa4.forward_batch_bound(stream, &attention)?;
            }

            if readout_mode.is_some() && index + 1 == self.weights.layers.len() {
                let mode =
                    readout_mode.ok_or_else(|| infer("Missing Gemma3 selected readout mode"))?;
                let mut gather_first = activation;
                let mut gather_second = state;
                let mut gather_bounds = boundaries;
                let mut gather_selected = 0u64;
                let mut gather_first_out = projection;
                let mut gather_second_out = next;
                let mut gather_mode = mode;
                launch(
                    &self.kernels.selected_rows,
                    stream,
                    gather_grid,
                    0,
                    &mut [
                        arg(&mut gather_first),
                        arg(&mut gather_second),
                        arg(&mut gather_bounds),
                        arg(&mut gather_selected),
                        arg(&mut gather_first_out),
                        arg(&mut gather_second_out),
                        arg(&mut readout_count),
                        arg(&mut q_width_u32),
                        arg(&mut width),
                        arg(&mut gather_mode),
                    ],
                )?;
                // SAFETY: O/MLP run on the canonical-B rows just gathered; the
                // separate source, destination and residual spans remain inside
                // the checked workspace arena.
                unsafe {
                    blas.linear(
                        &self.plans.readout.attention,
                        &work.projection,
                        &layer.attention,
                        &mut work.branch,
                    )?;
                    self.kernels.set.launch_residual_norm(
                        &self.math,
                        stream,
                        branch,
                        next,
                        layer.post_attention_norm.cu_deviceptr(),
                        state,
                        sequences,
                        h,
                    )?;
                    self.kernels.set.launch_norm(
                        &self.math,
                        stream,
                        state,
                        layer.pre_feedforward_norm.cu_deviceptr(),
                        normalized,
                        sequences,
                        h,
                    )?;
                    blas.linear(
                        &self.plans.readout.gate_up,
                        &work.normalized,
                        &layer.gate_up,
                        &mut work.projection,
                    )?;
                    self.kernels.set.launch_gate(
                        &self.math,
                        stream,
                        projection,
                        activation,
                        sequences,
                        c.intermediate_size,
                    )?;
                    blas.linear(
                        &self.plans.readout.down,
                        &work.activation,
                        &layer.down,
                        &mut work.branch,
                    )?;
                    self.kernels.set.launch_residual_norm(
                        &self.math,
                        stream,
                        branch,
                        state,
                        layer.post_feedforward_norm.cu_deviceptr(),
                        next,
                        sequences,
                        h,
                    )?;
                    self.kernels.set.launch_norm(
                        &self.math,
                        stream,
                        next,
                        self.weights.final_norm.cu_deviceptr(),
                        normalized,
                        sequences,
                        h,
                    )?;
                }
                let mut cast_input = normalized;
                let mut cast_output = work.pooled()?.cu_deviceptr();
                let mut cast_selected = selected;
                let mut cast_count = selections_u32;
                let mut cast_width = width;
                launch(
                    &self.kernels.bf16_to_f32,
                    stream,
                    cast_grid,
                    0,
                    &mut [
                        arg(&mut cast_input),
                        arg(&mut cast_output),
                        arg(&mut cast_selected),
                        arg(&mut cast_count),
                        arg(&mut cast_width),
                    ],
                )?;
                return Ok(());
            }

            // SAFETY: O reads the FA4 output and writes the separate branch
            // slab; the post-attention residual norm consumes that branch and
            // the current state into the alternate state buffer.
            unsafe {
                blas.linear(
                    &self.plans.full.attention,
                    &work.activation,
                    &layer.attention,
                    &mut work.branch,
                )?;
                self.kernels.set.launch_residual_norm(
                    &self.math,
                    stream,
                    branch,
                    state,
                    layer.post_attention_norm.cu_deviceptr(),
                    next,
                    tokens,
                    h,
                )?;
            }
            std::mem::swap(&mut state, &mut next);
            // SAFETY: the feed-forward pre-norm reads the just-written state;
            // packed gate/up, BF16 gated GELU and down remain in the checked
            // projection/activation/branch slabs.
            unsafe {
                self.kernels.set.launch_norm(
                    &self.math,
                    stream,
                    state,
                    layer.pre_feedforward_norm.cu_deviceptr(),
                    normalized,
                    tokens,
                    h,
                )?;
                blas.linear(
                    &self.plans.full.gate_up,
                    &work.normalized,
                    &layer.gate_up,
                    &mut work.projection,
                )?;
                self.kernels.set.launch_gate(
                    &self.math,
                    stream,
                    projection,
                    activation,
                    tokens,
                    c.intermediate_size,
                )?;
                blas.linear(
                    &self.plans.full.down,
                    &work.activation,
                    &layer.down,
                    &mut work.branch,
                )?;
                self.kernels.set.launch_residual_norm(
                    &self.math,
                    stream,
                    branch,
                    state,
                    layer.post_feedforward_norm.cu_deviceptr(),
                    next,
                    tokens,
                    h,
                )?;
            }
            std::mem::swap(&mut state, &mut next);
        }

        // SAFETY: the final offset norm reads the current state into the
        // separate normalized slab before pooling or the shared raw bridge.
        unsafe {
            self.kernels.set.launch_norm(
                &self.math,
                stream,
                state,
                self.weights.final_norm.cu_deviceptr(),
                normalized,
                tokens,
                h,
            )?;
        }
        if selections > 0 {
            let mut pool_input = normalized;
            let mut pool_bounds = boundaries;
            let mut pool_indices = selected;
            let mut pool_out = work.pooled()?.cu_deviceptr();
            let mut pool_count = selections_u32;
            let mut pool_width = width;
            if let Some(grid) = native_mean_plan(pool_mode, selections, h) {
                // SAFETY: the native reduction reads the same validated BF16
                // normalized slab and boundary/selection arrays as the generic
                // pool and writes a disjoint F32 row for every selected span.
                launch(
                    &self.kernels.native_mean,
                    stream,
                    grid,
                    0,
                    &mut [
                        arg(&mut pool_input),
                        arg(&mut pool_bounds),
                        arg(&mut pool_indices),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                    ],
                )?;
            } else {
                let mut mode = pool_mode;
                launch(
                    &self.kernels.pool,
                    stream,
                    cast_grid,
                    0,
                    &mut [
                        arg(&mut pool_input),
                        arg(&mut pool_bounds),
                        arg(&mut pool_indices),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                        arg(&mut mode),
                    ],
                )?;
            }
        }
        if !batch.raw_indices.is_empty() {
            super::enqueue_raw_output(
                &self.kernels.bf16_to_f32,
                stream,
                normalized,
                work.pooled()?.cu_deviceptr(),
                raw_selected,
                batch,
                h,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 640,
            intermediate_size: 2048,
            num_hidden_layers: 18,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 256,
            vocab_size: 262_144,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 512,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
            is_causal: true,
            global_rope_factor: 1.0,
        }
    }

    #[test]
    fn native_mean_plan_uses_the_u16_count_bound_and_eight_feature_tiles() {
        assert_eq!(native_mean_plan(1, 1, 640), Some((1, 80, 1)));
        assert_eq!(native_mean_plan(1, 65_535, 640), Some((65_535, 80, 1)));
        assert_eq!(native_mean_plan(1, 65_536, 640), None);
        assert_eq!(native_mean_plan(1, 0, 640), None);
        assert_eq!(native_mean_plan(0, 3, 640), None);
        assert_eq!(native_mean_plan(2, 3, 640), None);
        assert_eq!(native_mean_plan(1, 2, 8), Some((2, 1, 1)));
        assert_eq!(native_mean_plan(1, 2, 641), Some((2, 81, 1)));
    }

    #[test]
    #[ignore = "CPU-only actual trained Gemma3 metadata/scales; set INFINITY_OXIDE_TEST_GEMMA3_MODEL"]
    fn actual_trained_headers_and_offset_scales_before_cuda() {
        let path =
            std::path::PathBuf::from(std::env::var_os("INFINITY_OXIDE_TEST_GEMMA3_MODEL").unwrap());
        let host = Gemma3Weights::load(&path).unwrap();
        validate_admission(&host.config).unwrap();
        Gemma3WeightBuffers::validate(&host).unwrap();
        let h = host.config.hidden_size;
        let d = host.config.head_dim;
        let layers = host.layers.len();
        assert_eq!(host.config.q_width(), host.config.num_attention_heads * d);
        assert_eq!(host.config.kv_width(), host.config.num_key_value_heads * d);
        let mut sources = 1;
        let mut elements = crate::gemma3_prepared::offset_norm_values(&host.norm, h)
            .unwrap()
            .len();
        for layer in &host.layers {
            for source in [
                &layer.common.input_norm,
                &layer.common.post_attention_norm,
                &layer.pre_feedforward_norm,
                &layer.post_feedforward_norm,
            ] {
                elements += crate::gemma3_prepared::offset_norm_values(source, h)
                    .unwrap()
                    .len();
                sources += 1;
            }
            for source in [
                layer.common.q_norm.as_ref().unwrap(),
                layer.common.k_norm.as_ref().unwrap(),
            ] {
                elements += crate::gemma3_prepared::offset_norm_values(source, d)
                    .unwrap()
                    .len();
                sources += 1;
            }
        }
        assert_eq!(sources, 1 + layers * 6);
        assert_eq!(elements, h + layers * (4 * h + 2 * d));
        let g = DecoderGeometry::from_dimensions(
            h,
            host.config.intermediate_size,
            host.config.q_width(),
            host.config.kv_width(),
            d,
        )
        .unwrap();
        let mut plans = Gemma3Plans::new(&host.config, &g).unwrap();
        for rows in [1, 7, 128, 129, 7] {
            plans.set_full_rows(rows).unwrap();
        }
        for rows in [1, 3, 8, 3] {
            plans.set_readout_rows(rows).unwrap();
        }
        assert!(DecoderWork::allocation_bytes_geometry((128, 8, 1024), &g).unwrap() > 0);
        println!("GEMMA3_ACTUAL_CPU:{sources} finite target-BF16 offset norm sources/{elements}scales; matrices remain lazy; no CUDA");
    }

    #[test]
    fn prepared_geometry_uses_independent_query_and_kv_widths() {
        let c = config();
        let g = DecoderGeometry::from_dimensions(
            c.hidden_size,
            c.intermediate_size,
            c.q_width(),
            c.kv_width(),
            c.head_dim,
        )
        .expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (640, 1024, 256, 1536)
        );
        assert_eq!(g.projection_width, 4096);
        assert_eq!(g.activation_width, 2048);
        assert_eq!(g.silu_groups_per_row, 256);
        assert_eq!(
            g.attention_strides(c.head_dim).expect("strides"),
            [1536, 256, 1, 1536, 256, 1, 1536, 256, 1, 1024, 256, 1]
        );
    }

    fn config_12b() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            vocab_size: 262_208,
            max_position_embeddings: 131_072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
            is_causal: true,
            global_rope_factor: 8.0,
        }
    }

    fn config_27b() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 5376,
            intermediate_size: 21504,
            num_hidden_layers: 62,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 128,
            vocab_size: 262_208,
            max_position_embeddings: 131_072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 168.0,
            is_causal: true,
            global_rope_factor: 8.0,
        }
    }

    #[test]
    fn prepared_admission_covers_all_three_target_geometries() {
        for c in [config(), config_12b(), config_27b()] {
            let math = Gemma3PreparedMath::admit(&c).expect("prepared");
            assert_eq!(
                math.embedding_scale().to_bits(),
                half::bf16::from_f32((c.hidden_size as f32).sqrt())
                    .to_f32()
                    .to_bits()
            );
            assert!(admit_scope(&c).is_ok());
        }

        let all_global = config();
        let mut all_global = Gemma3Config {
            sliding_window_pattern: 1,
            ..all_global
        };
        assert!(!all_global.layer_policy(0).unwrap().local);
        assert!(!all_global.layer_policy(17).unwrap().local);
        assert!(Gemma3Config {
            sliding_window_pattern: 0,
            ..all_global.clone()
        }
        .layer_policy(0)
        .is_err());
        all_global.head_dim = 258;
        assert!(Gemma3PreparedMath::admit(&all_global).is_err());
        all_global.head_dim = 255;
        assert!(Gemma3PreparedMath::admit(&all_global).is_err());
        all_global.head_dim = 128;
        assert!(Gemma3PreparedMath::admit(&all_global).is_ok());
    }
}
