// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! ModernBERT + Laya typed-decision checkpoint and packed readout.
//!
//! The native contract is `LayaHead` in `decision_models.cpp`:
//!
//! * `rl_agent_config.json` selects the Laya head and supplies
//!   `head_layers`, `max_len` and `head_max_len`;
//! * the ModernBERT encoder lives under `encoder/` and shares the checkpoint
//!   shards with head tensors, so the head is not an independent model;
//! * each request carries a sequence-relative question type (0..2) and one or
//!   more sequence-relative marker rows;
//! * padded marker rows enter the learned option scorer only after every token
//!   has passed through the attention + ReLU feed-forward refinement layers;
//! * the learned action head consumes each question's first refined token and
//!   top-probability/margin/entropy/count features.
//!
//! Native `bridge.cpp::weight_dtype` casts Laya head tensors to the selected
//! model dtype (only Vela encoder-nonencoder, D3 and Pplx override it); it does
//! not make the Laya head intrinsically FP32. The current Oxide `ModernEncoder`
//! has exactly one prepared runtime: BF16 `Work` storage, BF16 weight sources
//! through `WeightSource::read_bf16`, and BF16 FA4. This module therefore
//! accepts `--dtype bfloat16` only and rejects every other dtype explicitly
//! rather than silently running a float16 request through BF16 math. FP32 is
//! used only where Native itself widens: option logits, action features and the
//! final action probability. There is no CPU or generic OptionHead fallback.
//!
//! The official checkpoint geometry is ModernBERT hidden 1024, 16 heads,
//! intermediate 2624, 28 layers and max positions 8192. The loader admits both
//! that published root `config.json` layout and the legacy
//! `encoder/config.json` + `encoder.` tensor-prefix layout. Modern H768 remains
//! unchanged. Laya attention uses the cold prepared global noncausal FA4 MHA
//! profile; the generic SIMT attention kernel is not part of this route.

use crate::{
    arg,
    biased_linear::PreparedBiasedLinear,
    blocks,
    cublas::Blas,
    decision_head::{DecisionHeadError, PackedDecisionBatch},
    decision_refinement::{
        activation, enqueue_block, launch_projection, layer_norm, layer_norm_aten, prepare_block,
        prepare_projection, AffineNorm, BiasedProjection, RefinementBlock, RefinementBuffers,
        RefinementKernels, RefinementPlan, RefinementPrepared,
    },
    fa4, infer, launch, launch_threads,
    lt::LtProvider,
    model::{path_present, read_json, Checkpoint, WeightSource},
    product,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, DecisionInput};
use safetensors::Dtype;
use serde::Deserialize;
use std::{
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const LAYA_CONFIG: &str = "rl_agent_config.json";
/// The only prepared dtype implemented by the current ModernEncoder route.
pub(crate) const LAYA_RUNTIME_DTYPE: &str = "bfloat16";
pub(crate) const ENCODER_DIR: &str = "encoder";
const MAX_HEAD_LAYERS: usize = 16;
const SCORER_HIDDEN: usize = 256;
const ACTION_CLASSES: usize = 2;
static NEXT_HEAD_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Deserialize)]
struct RawLayaConfig {
    head_layers: Option<usize>,
    max_len: Option<usize>,
    head_max_len: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct LayaConfig {
    pub(crate) head_layers: usize,
    /// Retained for callers that also need the published sidecar bounds; the
    /// packed marker validator uses the actual sequence length, as Native does.
    #[allow(dead_code)]
    pub(crate) max_len: usize,
    #[allow(dead_code)]
    pub(crate) head_max_len: usize,
}

impl LayaConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawLayaConfig = read_json(&path.join(LAYA_CONFIG))?;
        let head_layers = raw
            .head_layers
            .ok_or_else(|| infer("Laya rl_agent_config.json is missing head_layers"))?;
        let max_len = raw
            .max_len
            .ok_or_else(|| infer("Laya rl_agent_config.json is missing max_len"))?;
        let head_max_len = raw
            .head_max_len
            .ok_or_else(|| infer("Laya rl_agent_config.json is missing head_max_len"))?;
        if head_layers > MAX_HEAD_LAYERS {
            return Err(infer(format!(
                "Laya head_layers {head_layers} exceeds the supported maximum {MAX_HEAD_LAYERS}"
            )));
        }
        if max_len == 0 || head_max_len == 0 {
            return Err(infer("Laya max_len and head_max_len must be positive"));
        }
        Ok(Self {
            head_layers,
            max_len,
            head_max_len,
        })
    }
}

pub(crate) fn is_laya_checkpoint(path: &Path) -> Result<bool, BackendError> {
    path_present(&path.join(LAYA_CONFIG))
}

/// Reject a dtype the current ModernEncoder does not implement. This is a
/// genuine capability boundary, not a checkpoint preference: float16 Laya
/// needs an F16 Modern route and is never downgraded to BF16.
fn validate_runtime_dtype(dtype: &str) -> Result<(), BackendError> {
    if dtype == LAYA_RUNTIME_DTYPE {
        Ok(())
    } else {
        Err(infer(format!(
            "Oxide Laya dtype {dtype} is unsupported: the ModernEncoder route implements only {LAYA_RUNTIME_DTYPE} prepared storage, FA4 and weight uploads; float16 Laya requires a separate F16 Modern route"
        )))
    }
}

/// Prefer the Native `encoder/config.json` layout, but admit the published
/// root `config.json` metadata when that is the actual checkpoint layout.
pub(crate) fn laya_encoder_config(path: &Path) -> Result<std::path::PathBuf, BackendError> {
    let nested = path.join(ENCODER_DIR);
    if path_present(&nested.join("config.json"))? {
        Ok(nested)
    } else if path_present(&path.join("config.json"))? {
        Ok(path.to_path_buf())
    } else {
        Err(infer(
            "Laya checkpoint is missing encoder/config.json and config.json",
        ))
    }
}

/// Validate modern encoder sidecar placement before any CUDA context exists.
pub(crate) fn validate_laya_checkpoint(
    path: &Path,
    dtype: &str,
) -> Result<LayaConfig, BackendError> {
    if !is_laya_checkpoint(path)? {
        return Err(infer("Laya checkpoint requires rl_agent_config.json"));
    }
    validate_runtime_dtype(dtype)?;
    let config = LayaConfig::load(path)?;
    let encoder = laya_encoder_config(path)?;
    let _ = crate::modern_encoder_model::ModernConfig::load(&encoder)?;
    Ok(config)
}

struct LayaKernels {
    layer_norm: Function,
    plain_layer_norm: Option<Function>,
    add: Function,
    relu: Function,
    gelu_exact: Function,
    type_add: Function,
    marker_gather: Function,
    cast_rows: Function,
    features: Function,
    action_input: Function,
    bias_seed: Function,
    softmax2: Function,
}

impl LayaKernels {
    fn load(context: &Arc<Context>, hidden: usize) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                infer("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(infer)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| infer(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(infer)?;
        let load = |name: &str| module.load_function(name).map_err(infer);
        Ok(Self {
            layer_norm: load("infinity_layer_norm")?,
            plain_layer_norm: if hidden == 1024 {
                Some(load("infinity_plain_layer_norm_bf16_aten")?)
            } else {
                None
            },
            add: load("infinity_laya_add_bf16")?,
            relu: load("infinity_encoder_activation_relu_bf16")?,
            gelu_exact: load("infinity_laya_gelu_exact_bias_bf16")?,
            type_add: load("infinity_laya_type_add_bf16")?,
            marker_gather: load("infinity_embedding_gather")?,
            cast_rows: load("infinity_bf16_to_f32")?,
            features: load("infinity_laya_action_features_f32")?,
            action_input: load("infinity_laya_action_input_bf16")?,
            bias_seed: load("infinity_classifier_bias_bf16")?,
            softmax2: load("infinity_laya_softmax2_f32")?,
        })
    }
}

struct LayaWeights {
    types: DeviceBuffer<Bf16>,
    layers: Vec<RefinementBlock>,
    scorer_norm: AffineNorm,
    scorer1: BiasedProjection,
    scorer3: BiasedProjection,
    action1: BiasedProjection,
    action2: BiasedProjection,
}

fn source(
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<WeightSource, BackendError> {
    checkpoint.source(name, shape)
}

fn upload_one(
    uploader: &mut WeightUploader,
    source: &WeightSource,
    name: &str,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    if !matches!(source.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        return Err(infer(format!(
            "Laya head tensor {name} has unsupported dtype {:?}",
            source.dtype()
        )));
    }
    uploader.upload(&[source])
}

fn upload_linear(
    uploader: &mut WeightUploader,
    checkpoint: &Checkpoint,
    provider: &LtProvider,
    name: &str,
    input: usize,
    output: usize,
) -> Result<BiasedProjection, BackendError> {
    let weight = upload_one(
        uploader,
        &source(checkpoint, &format!("{name}.weight"), &[output, input])?,
        &format!("{name}.weight"),
    )?;
    let bias = upload_one(
        uploader,
        &source(checkpoint, &format!("{name}.bias"), &[output])?,
        &format!("{name}.bias"),
    )?;
    BiasedProjection::new(provider, input, output, weight, bias)
}

fn upload_norm(
    uploader: &mut WeightUploader,
    checkpoint: &Checkpoint,
    name: &str,
    width: usize,
) -> Result<AffineNorm, BackendError> {
    Ok(AffineNorm {
        gamma: upload_one(
            uploader,
            &source(checkpoint, &format!("{name}.weight"), &[width])?,
            &format!("{name}.weight"),
        )?,
        beta: upload_one(
            uploader,
            &source(checkpoint, &format!("{name}.bias"), &[width])?,
            &format!("{name}.bias"),
        )?,
    })
}

pub(crate) fn output_elements(requests: &[DecisionInput]) -> Result<usize, BackendError> {
    let mut logits = 0usize;
    for request in requests {
        let count = match request {
            DecisionInput::Warmup => 1,
            DecisionInput::Laya {
                question_type,
                markers,
            } if *question_type < 3 && !markers.is_empty() => markers.len(),
            _ => {
                return Err(infer(
                    "Laya requires question_type in 0..3 and nonempty markers",
                ))
            }
        };
        logits = logits
            .checked_add(count)
            .ok_or_else(|| infer("Laya output count overflow"))?;
    }
    logits
        .checked_add(requests.len())
        .ok_or_else(|| infer("Laya action/output count overflow"))
}

/// Row starts for the post-refinement action gather. These are the actual
/// cumulative sequence starts, never the backbone's pre-refinement pooled
/// selection.
pub(crate) fn action_row_starts(
    cumulative: &[u32],
    questions: usize,
) -> Result<Vec<u32>, BackendError> {
    if questions == 0
        || cumulative.len() != questions.saturating_add(1)
        || cumulative.first().copied() != Some(0)
    {
        return Err(infer(
            "Laya action row starts do not match the request count",
        ));
    }
    for pair in cumulative.windows(2) {
        if pair[1] <= pair[0] {
            return Err(infer("Laya action row starts are not strictly increasing"));
        }
    }
    Ok(cumulative[..questions].to_vec())
}

pub(crate) fn result_bytes(output_elements: usize) -> Result<usize, BackendError> {
    output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| infer("Laya result byte size overflow"))
}

/// Required Laya head tensor names and exact geometry. This is metadata-only;
/// no tensor payload is read.
pub(crate) fn validate_head_sources(
    checkpoint: &Checkpoint,
    config: &LayaConfig,
    hidden: usize,
) -> Result<(), BackendError> {
    let mut tensors = vec![
        ("type_emb.weight".to_string(), vec![3, hidden]),
        ("scorer.0.weight".to_string(), vec![hidden]),
        ("scorer.0.bias".to_string(), vec![hidden]),
        ("scorer.1.weight".to_string(), vec![hidden, hidden]),
        ("scorer.1.bias".to_string(), vec![hidden]),
        ("scorer.3.weight".to_string(), vec![1, hidden]),
        ("scorer.3.bias".to_string(), vec![1]),
        (
            "act_head.0.weight".to_string(),
            vec![SCORER_HIDDEN, hidden + 4],
        ),
        ("act_head.0.bias".to_string(), vec![SCORER_HIDDEN]),
        (
            "act_head.2.weight".to_string(),
            vec![ACTION_CLASSES, SCORER_HIDDEN],
        ),
        ("act_head.2.bias".to_string(), vec![ACTION_CLASSES]),
    ];
    for index in 0..config.head_layers {
        let prefix = format!("head.layers.{index}");
        for (name, shape) in [
            (format!("{prefix}.norm1.weight"), vec![hidden]),
            (format!("{prefix}.norm1.bias"), vec![hidden]),
            (format!("{prefix}.norm2.weight"), vec![hidden]),
            (format!("{prefix}.norm2.bias"), vec![hidden]),
            (
                format!("{prefix}.self_attn.in_proj_weight"),
                vec![3 * hidden, hidden],
            ),
            (format!("{prefix}.self_attn.in_proj_bias"), vec![3 * hidden]),
            (
                format!("{prefix}.self_attn.out_proj.weight"),
                vec![hidden, hidden],
            ),
            (format!("{prefix}.self_attn.out_proj.bias"), vec![hidden]),
            (format!("{prefix}.linear1.weight"), vec![4 * hidden, hidden]),
            (format!("{prefix}.linear1.bias"), vec![4 * hidden]),
            (format!("{prefix}.linear2.weight"), vec![hidden, 4 * hidden]),
            (format!("{prefix}.linear2.bias"), vec![hidden]),
        ] {
            tensors.push((name, shape));
        }
    }
    for (name, shape) in tensors {
        let _ = checkpoint.source(&name, &shape)?;
    }
    Ok(())
}

/// Pure packed request metadata. Markers stay sequence-relative on input and
/// become packed hidden-row ids only here; offsets partition the scorer result
/// back into independent requests.
#[derive(Debug)]
pub(crate) struct LayaRequestPlan {
    pub(crate) tokens: usize,
    pub(crate) token_types: Vec<u32>,
    pub(crate) marker_rows: Vec<u32>,
    pub(crate) marker_offsets: Vec<u32>,
    pub(crate) counts: Vec<usize>,
    pub(crate) total_markers: usize,
    pub(crate) max_length: usize,
}

pub(crate) fn plan_requests(
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
) -> Result<LayaRequestPlan, BackendError> {
    if requests.len() != batch.len() {
        return Err(infer(
            "Laya request count does not match the packed decision batch",
        ));
    }
    let tokens = batch.tokens();
    let mut token_types = Vec::new();
    token_types.try_reserve_exact(tokens).map_err(infer)?;
    token_types.resize(tokens, 0u32);
    let mut marker_rows = Vec::new();
    let mut counts = Vec::new();
    counts.try_reserve_exact(requests.len()).map_err(infer)?;
    let mut row_offsets = vec![0usize];
    row_offsets
        .try_reserve_exact(requests.len())
        .map_err(infer)?;
    let mut max_length = 0usize;
    for (index, request) in requests.iter().enumerate() {
        let (start, end) = batch
            .sequence(index)
            .map_err(|error| infer(error.to_string()))?;
        let length = end
            .checked_sub(start)
            .ok_or_else(|| infer("Laya sequence length overflow"))?;
        max_length = max_length.max(length);
        let (kind, markers) = match request {
            DecisionInput::Warmup => (0u32, vec![0usize]),
            DecisionInput::Laya {
                question_type,
                markers,
            } => {
                if *question_type >= 3 || markers.is_empty() {
                    return Err(infer(
                        "Laya requires question_type in 0..3 and nonempty markers",
                    ));
                }
                if *question_type > u32::MAX as usize {
                    return Err(infer("Laya question type exceeds u32"));
                }
                (*question_type as u32, markers.clone())
            }
            _ => {
                return Err(infer(
                    "Only Laya and Warmup requests reach the Laya decision route",
                ))
            }
        };
        token_types
            .get_mut(start..end)
            .ok_or_else(|| infer("Laya type range exceeds packed tokens"))?
            .fill(kind);
        for marker in markers {
            if marker >= length {
                return Err(infer(format!(
                    "Laya marker {marker} is outside sequence length {length}"
                )));
            }
            let row = start
                .checked_add(marker)
                .ok_or_else(|| infer("Laya marker row overflow"))?;
            marker_rows.push(u32::try_from(row).map_err(|_| infer("Laya marker row exceeds u32"))?);
        }
        let previous = row_offsets.last().copied().unwrap_or(0);
        let count = marker_rows
            .len()
            .checked_sub(previous)
            .ok_or_else(|| infer("Laya marker count underflow"))?;
        counts.push(count);
        row_offsets.push(marker_rows.len());
    }
    let marker_offsets = row_offsets
        .iter()
        .map(|&value| u32::try_from(value).map_err(|_| infer("Laya marker offset exceeds u32")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LayaRequestPlan {
        tokens,
        token_types,
        marker_rows,
        marker_offsets,
        counts,
        total_markers: row_offsets.last().copied().unwrap_or(0),
        max_length,
    })
}

impl LayaRequestPlan {
    #[cfg(test)]
    pub(crate) fn metadata_words(&self) -> Result<usize, BackendError> {
        self.tokens
            .checked_add(self.marker_rows.len())
            .and_then(|words| words.checked_add(self.marker_offsets.len()))
            .ok_or_else(|| infer("Laya metadata tail size overflow"))
    }

    /// Pack the one common Laya metadata tail: token types, packed marker rows,
    /// then packed marker offsets. The standard packed input precedes it.
    #[cfg(test)]
    pub(crate) fn write_metadata_tail(&self, out: &mut [u32]) -> Result<(), BackendError> {
        let words = self.metadata_words()?;
        if out.len() < words {
            return Err(infer("Laya metadata tail exceeds the host input buffer"));
        }
        let out = &mut out[..words];
        let (types, rest) = out.split_at_mut(self.tokens);
        types.copy_from_slice(&self.token_types);
        let (rows, offsets) = rest.split_at_mut(self.marker_rows.len());
        rows.copy_from_slice(&self.marker_rows);
        offsets.copy_from_slice(&self.marker_offsets);
        Ok(())
    }
}

/// Cached-capacity Laya request scratch. Every buffer is owned by the existing
/// `Work` arena, charged before allocation and released only after a proven
/// completion fence. `Work` itself supplies the residual/attention/QKV buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayaScratchLayout {
    pub(crate) tokens: usize,
    pub(crate) hidden: usize,
    pub(crate) ffn: usize,
    pub(crate) markers: usize,
    pub(crate) requests: usize,
    pub(crate) metadata_words: usize,
}

impl LayaScratchLayout {
    pub(crate) fn validated(self) -> Result<Self, BackendError> {
        if self.tokens == 0
            || self.hidden == 0
            || self.ffn == 0
            || self.markers == 0
            || self.requests == 0
        {
            return Err(infer(
                "Laya scratch requires tokens, markers, requests and positive widths",
            ));
        }
        let _ = self.bytes()?;
        Ok(self)
    }

    #[cfg(test)]
    pub(crate) fn from_plan(
        plan: &LayaRequestPlan,
        hidden: usize,
        ffn: usize,
    ) -> Result<Self, BackendError> {
        if plan.tokens == 0 || hidden == 0 || ffn == 0 || plan.total_markers == 0 {
            return Err(infer(
                "Laya scratch requires tokens, markers and positive widths",
            ));
        }
        Ok(Self {
            tokens: plan.tokens,
            hidden,
            ffn,
            markers: plan.total_markers,
            requests: plan.counts.len(),
            metadata_words: plan.metadata_words()?,
        })
    }

    pub(crate) fn fits(&self, request: Self) -> bool {
        self.hidden == request.hidden
            && self.ffn == request.ffn
            && self.tokens >= request.tokens
            && self.markers >= request.markers
            && self.requests >= request.requests
            && self.metadata_words >= request.metadata_words
    }

    pub(crate) fn bytes(&self) -> Result<usize, BackendError> {
        let bf16 = |elements: usize| {
            elements
                .checked_mul(std::mem::size_of::<Bf16>())
                .ok_or_else(|| infer("Laya BF16 scratch byte size overflow"))
        };
        let f32 = |elements: usize| {
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Laya F32 scratch byte size overflow"))
        };
        let hidden = product(self.tokens, self.hidden)?;
        let ffn = product(self.tokens, self.ffn)?;
        let markers = product(self.markers, self.hidden)?;
        let parts = [
            bf16(hidden)?,
            bf16(hidden)?,
            bf16(ffn)?,
            bf16(profile_marker_elements(markers, self.markers)?)?,
            f32(self.markers)?,
            f32(product(self.requests, 4)?)?,
            bf16(product(self.requests, self.hidden + 4)?)?,
            bf16(product(self.requests, SCORER_HIDDEN)?)?,
            bf16(product(self.requests, ACTION_CLASSES)?)?,
            f32(product(self.requests, ACTION_CLASSES)?)?,
            f32(self.requests)?,
        ];
        let total = parts
            .into_iter()
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| infer("Laya scratch byte size overflow"))?;
        Ok(total)
    }
}

fn profile_marker_elements(markers: usize, marker_rows: usize) -> Result<usize, BackendError> {
    markers
        .checked_mul(3)
        .and_then(|elements| elements.checked_add(marker_rows))
        .ok_or_else(|| infer("Laya marker scratch element count overflow"))
}

pub(crate) struct LayaScratch {
    pub(crate) norm: DeviceBuffer<Bf16>,
    pub(crate) branch: DeviceBuffer<Bf16>,
    pub(crate) ffn: DeviceBuffer<Bf16>,
    pub(crate) marker_h: DeviceBuffer<Bf16>,
    pub(crate) marker_norm: DeviceBuffer<Bf16>,
    pub(crate) marker_proj: DeviceBuffer<Bf16>,
    pub(crate) marker_logits_bf16: DeviceBuffer<Bf16>,
    pub(crate) logits_f32: DeviceBuffer<f32>,
    pub(crate) features_f32: DeviceBuffer<f32>,
    pub(crate) action_input: DeviceBuffer<Bf16>,
    pub(crate) action_hidden: DeviceBuffer<Bf16>,
    pub(crate) action_logits_bf16: DeviceBuffer<Bf16>,
    pub(crate) action_logits_f32: DeviceBuffer<f32>,
    pub(crate) actions_f32: DeviceBuffer<f32>,
    layout: LayaScratchLayout,
}

impl LayaScratch {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        layout: LayaScratchLayout,
    ) -> Result<Self, BackendError> {
        let bf16 = |elements: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
            // SAFETY: every logical element is written before it is read.
            unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(infer)
        };
        let f32 = |elements: usize| -> Result<DeviceBuffer<f32>, BackendError> {
            // SAFETY: every logical element is written before readback.
            unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, elements) }.map_err(infer)
        };
        let hidden = product(layout.tokens, layout.hidden)?;
        let ffn = product(layout.tokens, layout.ffn)?;
        let markers = product(layout.markers, layout.hidden)?;
        Ok(Self {
            norm: bf16(hidden)?,
            branch: bf16(hidden)?,
            ffn: bf16(ffn)?,
            marker_h: bf16(markers)?,
            marker_norm: bf16(markers)?,
            marker_proj: bf16(markers)?,
            marker_logits_bf16: bf16(layout.markers)?,
            logits_f32: f32(layout.markers)?,
            features_f32: f32(product(layout.requests, 4)?)?,
            action_input: bf16(product(layout.requests, layout.hidden + 4)?)?,
            action_hidden: bf16(product(layout.requests, SCORER_HIDDEN)?)?,
            action_logits_bf16: bf16(product(layout.requests, ACTION_CLASSES)?)?,
            action_logits_f32: f32(product(layout.requests, ACTION_CLASSES)?)?,
            actions_f32: f32(layout.requests)?,
            layout,
        })
    }
    pub(crate) fn fits(&self, request: LayaScratchLayout) -> bool {
        self.layout.fits(request)
    }
}

/// Load the ModernBERT backbone from a Laya checkpoint without changing the
/// shared modern loader's ordinary namespace policy. Head tensors share the
/// shards and may use unprefixed `scorer.*`, `act_head.*` and `head.*` names.
pub(crate) fn load_laya_modern(
    path: &Path,
    config_path: &Path,
) -> Result<crate::modern_encoder_model::ModernWeights, BackendError> {
    use crate::modern_encoder_model::{ModernConfig, ModernLayerWeights, ModernWeights};
    let config = ModernConfig::load(config_path)?;
    crate::model::validate_modules(path)?;
    let checkpoint = Checkpoint::open(path)?;
    let root = if checkpoint.contains("encoder.embeddings.tok_embeddings.weight") {
        "encoder."
    } else if checkpoint.contains("model.embeddings.tok_embeddings.weight") {
        "model."
    } else if checkpoint.contains("embeddings.tok_embeddings.weight") {
        ""
    } else {
        return Err(infer(
            "Laya checkpoint has no ModernBERT token embedding root",
        ));
    };
    let hidden = config.geometry.hidden_size;
    let intermediate = config.geometry.intermediate_size;
    let vocab = config.geometry.vocab_size;
    let layers = config.geometry.num_hidden_layers;
    let word = source(
        &checkpoint,
        &format!("{root}embeddings.tok_embeddings.weight"),
        &[vocab, hidden],
    )?;
    let embedding_gamma = source(
        &checkpoint,
        &format!("{root}embeddings.norm.weight"),
        &[hidden],
    )?;
    let final_gamma = source(&checkpoint, &format!("{root}final_norm.weight"), &[hidden])?;
    let mut result_layers = Vec::new();
    result_layers.try_reserve_exact(layers).map_err(infer)?;
    for index in 0..layers {
        let prefix = format!("{root}layers.{index}");
        let qkv = source(
            &checkpoint,
            &format!("{prefix}.attn.Wqkv.weight"),
            &[3 * hidden, hidden],
        )?;
        let attention = source(
            &checkpoint,
            &format!("{prefix}.attn.Wo.weight"),
            &[hidden, hidden],
        )?;
        let wi = source(
            &checkpoint,
            &format!("{prefix}.mlp.Wi.weight"),
            &[2 * intermediate, hidden],
        )?;
        let up_gate = crate::modern_encoder_model::canonical_up_gate(&wi, hidden, intermediate)?;
        let output = source(
            &checkpoint,
            &format!("{prefix}.mlp.Wo.weight"),
            &[hidden, intermediate],
        )?;
        let mlp_gamma = source(&checkpoint, &format!("{prefix}.mlp_norm.weight"), &[hidden])?;
        let attention_gamma = if index == 0 {
            None
        } else {
            Some(source(
                &checkpoint,
                &format!("{prefix}.attn_norm.weight"),
                &[hidden],
            )?)
        };
        result_layers.push(ModernLayerWeights {
            qkv,
            attention,
            up_gate,
            output,
            attention_gamma,
            mlp_gamma,
        });
    }
    Ok(ModernWeights {
        config,
        word,
        embedding_gamma,
        final_gamma,
        layers: result_layers,
        classifier: None,
    })
}

/// Cold Laya head metadata, tensors and shared kernel handles.
pub(crate) struct LayaHead {
    id: u64,
    hidden: usize,
    ffn: usize,
    heads: usize,
    kernels: LayaKernels,
    attention: fa4::Fa4,
    stream: Arc<Stream>,
    weights: Option<LayaWeights>,
}

impl LayaHead {
    pub(crate) fn load(
        path: &Path,
        config: LayaConfig,
        hidden: usize,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        provider: &LtProvider,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(context, stream.context()) {
            return Err(infer("Laya stream belongs to another CUDA context"));
        }
        if !matches!(hidden, 768 | 1024) || !hidden.is_multiple_of(64) {
            return Err(infer(format!(
                "Laya requires ModernBERT hidden_size 768 or 1024, found {hidden}"
            )));
        }
        context.bind_to_thread().map_err(infer)?;
        let checkpoint = Checkpoint::open(path)?;
        validate_head_sources(&checkpoint, &config, hidden)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let types = upload_one(
            &mut uploader,
            &source(&checkpoint, "type_emb.weight", &[3, hidden])?,
            "type_emb.weight",
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.head_layers)
            .map_err(infer)?;
        for index in 0..config.head_layers {
            let prefix = format!("head.layers.{index}");
            layers.push(RefinementBlock {
                norm1: upload_norm(
                    &mut uploader,
                    &checkpoint,
                    &format!("{prefix}.norm1"),
                    hidden,
                )?,
                norm2: upload_norm(
                    &mut uploader,
                    &checkpoint,
                    &format!("{prefix}.norm2"),
                    hidden,
                )?,
                qkv: {
                    let weight = upload_one(
                        &mut uploader,
                        &source(
                            &checkpoint,
                            &format!("{prefix}.self_attn.in_proj_weight"),
                            &[3 * hidden, hidden],
                        )?,
                        &format!("{prefix}.self_attn.in_proj_weight"),
                    )?;
                    let bias = upload_one(
                        &mut uploader,
                        &source(
                            &checkpoint,
                            &format!("{prefix}.self_attn.in_proj_bias"),
                            &[3 * hidden],
                        )?,
                        &format!("{prefix}.self_attn.in_proj_bias"),
                    )?;
                    BiasedProjection::new(provider, hidden, 3 * hidden, weight, bias)?
                },
                out: upload_linear(
                    &mut uploader,
                    &checkpoint,
                    provider,
                    &format!("{prefix}.self_attn.out_proj"),
                    hidden,
                    hidden,
                )?,
                ffn1: upload_linear(
                    &mut uploader,
                    &checkpoint,
                    provider,
                    &format!("{prefix}.linear1"),
                    hidden,
                    4 * hidden,
                )?,
                ffn2: upload_linear(
                    &mut uploader,
                    &checkpoint,
                    provider,
                    &format!("{prefix}.linear2"),
                    4 * hidden,
                    hidden,
                )?,
            });
        }
        let scorer_norm = upload_norm(&mut uploader, &checkpoint, "scorer.0", hidden)?;
        let scorer1 = upload_linear(
            &mut uploader,
            &checkpoint,
            provider,
            "scorer.1",
            hidden,
            hidden,
        )?;
        let scorer3 = upload_linear(&mut uploader, &checkpoint, provider, "scorer.3", hidden, 1)?;
        let action1 = upload_linear(
            &mut uploader,
            &checkpoint,
            provider,
            "act_head.0",
            hidden + 4,
            SCORER_HIDDEN,
        )?;
        let action2 = upload_linear(
            &mut uploader,
            &checkpoint,
            provider,
            "act_head.2",
            SCORER_HIDDEN,
            ACTION_CLASSES,
        )?;
        uploader.finish()?;
        let kernels = LayaKernels::load(context, hidden)?;
        let heads = (hidden / 64).max(1);
        let head_dim = hidden / heads;
        let root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
            .ok_or_else(|| infer("Laya requires INFINITY_FA4_ROOT for prepared FA4 attention"))?;
        let attention = fa4::Fa4::new(
            context.clone(),
            Path::new(&root),
            fa4::AttentionSpec {
                heads,
                kv_heads: heads,
                head_dim,
                causal: false,
                window: fa4::Window::Global,
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::RsqrtHeadDim,
            },
        )?;
        Ok(Self {
            id: NEXT_HEAD_ID.fetch_add(1, Ordering::Relaxed),
            hidden,
            ffn: 4 * hidden,
            heads,
            kernels,
            attention,
            stream: stream.clone(),
            weights: Some(LayaWeights {
                types,
                layers,
                scorer_norm,
                scorer1,
                scorer3,
                action1,
                action2,
            }),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn output_count(&self, request: &DecisionInput) -> Result<usize, DecisionHeadError> {
        match request {
            DecisionInput::Warmup => Ok(1),
            DecisionInput::Laya {
                question_type,
                markers,
            } => {
                if *question_type >= 3 || markers.is_empty() {
                    return Err(DecisionHeadError::UnsupportedRequest(
                        "Laya requires question_type in 0..3 and at least one marker",
                    ));
                }
                Ok(markers.len())
            }
            _ => Err(DecisionHeadError::UnsupportedRequest(
                "Only Laya and Warmup requests reach the Laya head",
            )),
        }
    }

    /// Build the CPU-only request plan and its prepared projection descriptors.
    /// No CUDA allocation, upload, query or synchronization happens here.
    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> Result<LayaPrepared, BackendError> {
        if requests.len() != batch.len() {
            return Err(infer(
                "Laya request count does not match the packed decision batch",
            ));
        }
        let plan = plan_requests(batch, requests)?;
        let plans = LayaPlans {
            refinement: RefinementPlan::new(plan.tokens, self.hidden, self.ffn)?,
        };
        Ok(LayaPrepared {
            head_id: self.id,
            tokens: plan.tokens,
            markers: plan.total_markers,
            requests: requests.len(),
            max_length: plan.max_length,
            counts: plan.counts,
            token_types: plan.token_types,
            marker_rows: plan.marker_rows,
            marker_offsets: plan.marker_offsets,
            plans,
        })
    }

    pub(crate) fn scratch_layout(
        &self,
        prepared: &LayaPrepared,
    ) -> Result<LayaScratchLayout, BackendError> {
        LayaScratchLayout {
            tokens: prepared.tokens,
            hidden: self.hidden,
            ffn: self.ffn,
            markers: prepared.markers,
            requests: prepared.requests,
            metadata_words: prepared.metadata_words()?,
        }
        .validated()
    }

    /// Bind every biased Laya projection for this request. This runs before the
    /// lease begins and selects the Lt/GemmEx policy without allocating.
    pub(crate) unsafe fn prepare_biased(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        work: &mut crate::Work<Bf16>,
        prepared: &LayaPrepared,
    ) -> Result<LayaBiasedPrepared, BackendError> {
        let weights = self
            .weights
            .as_mut()
            .ok_or_else(|| infer("Laya head weights were released"))?;
        let crate::Work {
            x,
            temp,
            qkv,
            attention,
            laya,
            ..
        } = work;
        let scratch = laya
            .as_mut()
            .ok_or_else(|| infer("Laya Work scratch is not attached"))?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(weights.layers.len())
            .map_err(infer)?;
        for layer in &mut weights.layers {
            let mut buffers = RefinementBuffers {
                state: x,
                next: temp,
                norm: &mut scratch.norm,
                qkv,
                attention,
                branch: &mut scratch.branch,
                ffn: &mut scratch.ffn,
            };
            layers.push(unsafe {
                prepare_block(
                    provider,
                    blas,
                    stream,
                    layer,
                    &prepared.plans.refinement,
                    &mut buffers,
                )?
            });
        }
        Ok(LayaBiasedPrepared {
            layers,
            scorer1: unsafe {
                prepare_projection(
                    &mut weights.scorer1,
                    provider,
                    blas,
                    prepared.markers,
                    stream,
                    &scratch.marker_norm,
                    &mut scratch.marker_proj,
                )?
            },
            scorer3: unsafe {
                prepare_projection(
                    &mut weights.scorer3,
                    provider,
                    blas,
                    prepared.markers,
                    stream,
                    &scratch.marker_proj,
                    &mut scratch.marker_logits_bf16,
                )?
            },
            action1: unsafe {
                prepare_projection(
                    &mut weights.action1,
                    provider,
                    blas,
                    prepared.requests,
                    stream,
                    &scratch.action_input,
                    &mut scratch.action_hidden,
                )?
            },
            action2: unsafe {
                prepare_projection(
                    &mut weights.action2,
                    provider,
                    blas,
                    prepared.requests,
                    stream,
                    &scratch.action_hidden,
                    &mut scratch.action_logits_bf16,
                )?
            },
        })
    }

    /// Enqueue the complete Laya refinement/readout body into the cached Work
    /// arena. `metadata` names the common host-input tail (token types, packed
    /// marker rows, packed offsets) already uploaded once. Output is
    /// `[packed option logits][one action probability per request]`.
    ///
    /// # Safety
    /// The engine bound this context on the current thread. Work, its Laya
    /// scratch, prepared biased selections, the uploaded input/metadata and the
    /// pinned output stay owned until the ordered stream drains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &mut self,
        work: &mut crate::Work<Bf16>,
        prepared: &LayaPrepared,
        biased: &mut LayaBiasedPrepared,
        metadata: LayaDeviceMetadata,
        cumulative: u64,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        host_out: *mut f32,
    ) -> Result<(), BackendError> {
        let required_scratch = self.scratch_layout(prepared)?;
        if prepared.head_id != self.id
            || prepared.tokens != work.tokens
            || prepared.markers == 0
            || prepared.requests == 0
            || work
                .laya()
                .is_none_or(|scratch| !scratch.fits(required_scratch))
            || biased.layers.len()
                != self
                    .weights
                    .as_ref()
                    .map_or(0, |weights| weights.layers.len())
        {
            return Err(infer(
                "Laya prepared metadata does not match the cached Work arena",
            ));
        }
        let weights = self
            .weights
            .as_mut()
            .ok_or_else(|| infer("Laya head weights were released"))?;
        let crate::Work {
            x,
            temp,
            qkv,
            attention,
            pooled,
            laya,
            ..
        } = work;
        let scratch = laya
            .as_mut()
            .ok_or_else(|| infer("Laya Work scratch is not attached"))?;
        let mut tokens = u32_size(prepared.tokens)?;
        let mut hidden_width = u32_size(self.hidden)?;
        let mut source = attention.cu_deviceptr();
        let mut types = metadata.token_types;
        let mut type_emb = weights.types.cu_deviceptr();
        let mut out = x.cu_deviceptr();
        launch(
            &self.kernels.type_add,
            stream,
            blocks(product(prepared.tokens, self.hidden)?)?,
            0,
            &mut [
                arg(&mut source),
                arg(&mut types),
                arg(&mut type_emb),
                arg(&mut out),
                arg(&mut tokens),
                arg(&mut hidden_width),
            ],
        )?;
        let kernels = RefinementKernels {
            layer_norm: &self.kernels.layer_norm,
            layer_norm_aten: self.kernels.plain_layer_norm.as_ref(),
            add: &self.kernels.add,
            attention: &self.attention,
            relu: &self.kernels.relu,
        };
        let seed = &self.kernels.bias_seed;
        for (index, layer) in weights.layers.iter_mut().enumerate() {
            let mut buffers = RefinementBuffers {
                state: x,
                next: temp,
                norm: &mut scratch.norm,
                qkv,
                attention,
                branch: &mut scratch.branch,
                ffn: &mut scratch.ffn,
            };
            let block_prepared = biased
                .layers
                .get(index)
                .ok_or_else(|| infer("Laya prepared refinement block is missing"))?;
            unsafe {
                enqueue_block(
                    provider,
                    blas,
                    seed,
                    stream,
                    &kernels,
                    layer,
                    &prepared.plans.refinement,
                    block_prepared,
                    &mut buffers,
                    cumulative,
                    prepared.requests + 1,
                    self.heads,
                    prepared.max_length,
                )?;
            }
        }
        // Native `forward_packed` selects each sequence's first row from the
        // fully refined stream, not from the backbone's pre-refinement CLS
        // pool. Gather and widen those refined rows into the existing
        // Work.pooled buffer; `cumulative[0..B]` are the row starts.
        let mut refined = x.cu_deviceptr();
        let mut action_pooled = pooled.cu_deviceptr();
        let mut starts = cumulative;
        let mut questions = u32_size(prepared.requests)?;
        launch(
            &self.kernels.cast_rows,
            stream,
            blocks(product(prepared.requests, self.hidden)?)?,
            0,
            &mut [
                arg(&mut refined),
                arg(&mut action_pooled),
                arg(&mut starts),
                arg(&mut questions),
                arg(&mut hidden_width),
            ],
        )?;
        let mut ids = metadata.marker_rows;
        let mut table = x.cu_deviceptr();
        let mut marker_h = scratch.marker_h.cu_deviceptr();
        let mut markers = u32_size(prepared.markers)?;
        let mut width = hidden_width;
        launch(
            &self.kernels.marker_gather,
            stream,
            blocks(product(prepared.markers, self.hidden)?)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut marker_h),
                arg(&mut markers),
                arg(&mut width),
            ],
        )?;
        if let Some(kernel) = self.kernels.plain_layer_norm.as_ref() {
            layer_norm_aten(
                kernel,
                stream,
                &scratch.marker_h,
                &weights.scorer_norm,
                &scratch.marker_norm,
                markers,
                hidden_width,
            )?;
        } else {
            layer_norm(
                &self.kernels.layer_norm,
                stream,
                &scratch.marker_h,
                &weights.scorer_norm,
                &scratch.marker_norm,
                markers,
                hidden_width,
            )?;
        }
        unsafe {
            launch_projection(
                &mut weights.scorer1,
                &biased.scorer1,
                provider,
                blas,
                seed,
                stream,
                &scratch.marker_norm,
                &mut scratch.marker_proj,
            )?;
            activation(
                &self.kernels.gelu_exact,
                stream,
                &scratch.marker_proj,
                &scratch.marker_proj,
                markers,
                hidden_width,
            )?;
            launch_projection(
                &mut weights.scorer3,
                &biased.scorer3,
                provider,
                blas,
                seed,
                stream,
                &scratch.marker_proj,
                &mut scratch.marker_logits_bf16,
            )?;
            let mut input = scratch.marker_logits_bf16.cu_deviceptr();
            let mut logits = scratch.logits_f32.cu_deviceptr();
            let mut selected = 0u64;
            let mut rows = markers;
            let mut one = 1u32;
            launch(
                &self.kernels.cast_rows,
                stream,
                blocks(prepared.markers)?,
                0,
                &mut [
                    arg(&mut input),
                    arg(&mut logits),
                    arg(&mut selected),
                    arg(&mut rows),
                    arg(&mut one),
                ],
            )?;
            let mut counts = metadata.marker_offsets;
            let mut features = scratch.features_f32.cu_deviceptr();
            let mut questions = u32_size(prepared.requests)?;
            launch_threads(
                &self.kernels.features,
                stream,
                (questions, 1, 1),
                128,
                0,
                &mut [
                    arg(&mut logits),
                    arg(&mut counts),
                    arg(&mut features),
                    arg(&mut questions),
                ],
            )?;
            let mut action_input = scratch.action_input.cu_deviceptr();
            let mut action_width = hidden_width;
            launch(
                &self.kernels.action_input,
                stream,
                blocks(product(prepared.requests, self.hidden + 4)?)?,
                0,
                &mut [
                    arg(&mut action_pooled),
                    arg(&mut features),
                    arg(&mut action_input),
                    arg(&mut questions),
                    arg(&mut action_width),
                ],
            )?;
            launch_projection(
                &mut weights.action1,
                &biased.action1,
                provider,
                blas,
                seed,
                stream,
                &scratch.action_input,
                &mut scratch.action_hidden,
            )?;
            activation(
                &self.kernels.gelu_exact,
                stream,
                &scratch.action_hidden,
                &scratch.action_hidden,
                questions,
                u32_size(SCORER_HIDDEN)?,
            )?;
            launch_projection(
                &mut weights.action2,
                &biased.action2,
                provider,
                blas,
                seed,
                stream,
                &scratch.action_hidden,
                &mut scratch.action_logits_bf16,
            )?;
            let mut action_input = scratch.action_logits_bf16.cu_deviceptr();
            let mut action_logits = scratch.action_logits_f32.cu_deviceptr();
            let mut selected = 0u64;
            let mut rows = questions;
            let mut two = u32_size(ACTION_CLASSES)?;
            launch(
                &self.kernels.cast_rows,
                stream,
                blocks(product(prepared.requests, ACTION_CLASSES)?)?,
                0,
                &mut [
                    arg(&mut action_input),
                    arg(&mut action_logits),
                    arg(&mut selected),
                    arg(&mut rows),
                    arg(&mut two),
                ],
            )?;
            let mut actions = scratch.actions_f32.cu_deviceptr();
            launch(
                &self.kernels.softmax2,
                stream,
                blocks(prepared.requests)?,
                0,
                &mut [
                    arg(&mut action_logits),
                    arg(&mut actions),
                    arg(&mut questions),
                ],
            )?;
            let bytes = prepared
                .markers
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Laya logit readback size overflow"))?;
            cuda_core::memory::memcpy_dtoh_async(
                host_out,
                scratch.logits_f32.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )
            .map_err(infer)?;
            let action_bytes = prepared
                .requests
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Laya action readback size overflow"))?;
            cuda_core::memory::memcpy_dtoh_async(
                host_out.add(prepared.markers),
                scratch.actions_f32.cu_deviceptr(),
                action_bytes,
                stream.cu_stream(),
            )
            .map_err(infer)?;
        }
        Ok(())
    }
}

/// Cold-biased selections for all Laya learned projections.
pub(crate) struct LayaBiasedPrepared {
    layers: Vec<RefinementPrepared>,
    scorer1: PreparedBiasedLinear,
    scorer3: PreparedBiasedLinear,
    action1: PreparedBiasedLinear,
    action2: PreparedBiasedLinear,
}

impl Drop for LayaHead {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            self.weights.take();
        } else if let Some(weights) = self.weights.take() {
            std::mem::forget(weights);
        }
    }
}

struct LayaPlans {
    refinement: RefinementPlan,
}

/// Device addresses of the one uploaded Laya metadata tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayaDeviceMetadata {
    pub(crate) token_types: u64,
    pub(crate) marker_rows: u64,
    pub(crate) marker_offsets: u64,
}

/// CPU-only request plan retained by the deferred readout. The actual device
/// metadata tail and all numeric scratch live in the cached `Work` owner.
pub(crate) struct LayaPrepared {
    head_id: u64,
    tokens: usize,
    markers: usize,
    requests: usize,
    max_length: usize,
    counts: Vec<usize>,
    token_types: Vec<u32>,
    marker_rows: Vec<u32>,
    marker_offsets: Vec<u32>,
    plans: LayaPlans,
}

impl LayaPrepared {
    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn output_elements(&self) -> usize {
        self.markers + self.requests
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn metadata_words(&self) -> Result<usize, BackendError> {
        self.token_types
            .len()
            .checked_add(self.marker_rows.len())
            .and_then(|words| words.checked_add(self.marker_offsets.len()))
            .ok_or_else(|| infer("Laya metadata tail size overflow"))
    }

    pub(crate) fn device_metadata(&self, trailer: u64) -> Result<LayaDeviceMetadata, BackendError> {
        let types = u64::try_from(self.token_types.len())
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| infer("Laya token-type metadata extent overflow"))?;
        let rows = u64::try_from(self.marker_rows.len())
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| infer("Laya marker metadata extent overflow"))?;
        let token_types = trailer;
        let marker_rows = token_types
            .checked_add(types)
            .ok_or_else(|| infer("Laya marker row address overflow"))?;
        let marker_offsets = marker_rows
            .checked_add(rows)
            .ok_or_else(|| infer("Laya marker offset address overflow"))?;
        Ok(LayaDeviceMetadata {
            token_types,
            marker_rows,
            marker_offsets,
        })
    }

    pub(crate) fn write_metadata_tail(&self, out: &mut [u32]) -> Result<(), BackendError> {
        let words = self.metadata_words()?;
        if out.len() < words {
            return Err(infer("Laya metadata tail exceeds the host input buffer"));
        }
        let (types, rest) = out.split_at_mut(self.token_types.len());
        types.copy_from_slice(&self.token_types);
        let (rows, offsets) = rest.split_at_mut(self.marker_rows.len());
        rows.copy_from_slice(&self.marker_rows);
        offsets.copy_from_slice(&self.marker_offsets);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use serde_json::{json, Value};
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    const HIDDEN: usize = 768;
    const OFFSETS: [u32; 3] = [0, 3, 5];
    const ACTUAL_ENCODER_CONFIG: &str =
        include_str!("../tests/data/laya_typed_decisions_encoder_config.json");
    const ACTUAL_RL_CONFIG: &str =
        include_str!("../tests/data/laya_typed_decisions_rl_agent_config.json");
    const ACTUAL_PROVENANCE: &str =
        include_str!("../tests/data/laya_typed_decisions_PROVENANCE.json");
    const A3_EXPORTS: &str = include_str!("../device/encoder_activation_exports.txt");

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(config: Value, head_layers: usize, type_shape: &[usize]) -> Self {
            let modern = json!({
                "model_type": "modernbert",
                "architectures": ["ModernBertModel"],
                "hidden_size": HIDDEN,
                "intermediate_size": 16,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "max_position_embeddings": 32,
                "vocab_size": 32,
                "norm_eps": 1e-5,
                "hidden_activation": "gelu",
                "attention_bias": false,
                "mlp_bias": false,
                "norm_bias": false,
                "global_attn_every_n_layers": 3,
                "local_attention": 128,
                "global_rope_theta": 160000.0,
                "local_rope_theta": 10000.0,
                "position_embedding_type": "absolute"
            });
            Self::with_modern(config, modern, head_layers, type_shape, true)
        }

        fn with_modern(
            config: Value,
            modern: Value,
            head_layers: usize,
            type_shape: &[usize],
            nested: bool,
        ) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-laya-head-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            if nested {
                fs::create_dir_all(path.join(ENCODER_DIR)).unwrap();
            } else {
                fs::create_dir_all(&path).unwrap();
            }
            fs::write(path.join(LAYA_CONFIG), serde_json::to_vec(&config).unwrap()).unwrap();
            let config_path = if nested {
                path.join(ENCODER_DIR).join("config.json")
            } else {
                path.join("config.json")
            };
            fs::write(&config_path, serde_json::to_vec(&modern).unwrap()).unwrap();
            let hidden = modern["hidden_size"].as_u64().unwrap() as usize;
            let mut tensors = vec![
                ("type_emb.weight".to_string(), type_shape.to_vec()),
                ("scorer.0.weight".to_string(), vec![hidden]),
                ("scorer.0.bias".to_string(), vec![hidden]),
                ("scorer.1.weight".to_string(), vec![hidden, hidden]),
                ("scorer.1.bias".to_string(), vec![hidden]),
                ("scorer.3.weight".to_string(), vec![1, hidden]),
                ("scorer.3.bias".to_string(), vec![1]),
                (
                    "act_head.0.weight".to_string(),
                    vec![SCORER_HIDDEN, hidden + 4],
                ),
                ("act_head.0.bias".to_string(), vec![SCORER_HIDDEN]),
                (
                    "act_head.2.weight".to_string(),
                    vec![ACTION_CLASSES, SCORER_HIDDEN],
                ),
                ("act_head.2.bias".to_string(), vec![ACTION_CLASSES]),
            ];
            for index in 0..head_layers {
                let prefix = format!("head.layers.{index}");
                for (name, shape) in [
                    (format!("{prefix}.norm1.weight"), vec![hidden]),
                    (format!("{prefix}.norm1.bias"), vec![hidden]),
                    (format!("{prefix}.norm2.weight"), vec![hidden]),
                    (format!("{prefix}.norm2.bias"), vec![hidden]),
                    (
                        format!("{prefix}.self_attn.in_proj_weight"),
                        vec![3 * hidden, hidden],
                    ),
                    (format!("{prefix}.self_attn.in_proj_bias"), vec![3 * hidden]),
                    (
                        format!("{prefix}.self_attn.out_proj.weight"),
                        vec![hidden, hidden],
                    ),
                    (format!("{prefix}.self_attn.out_proj.bias"), vec![hidden]),
                    (format!("{prefix}.linear1.weight"), vec![4 * hidden, hidden]),
                    (format!("{prefix}.linear1.bias"), vec![4 * hidden]),
                    (format!("{prefix}.linear2.weight"), vec![hidden, 4 * hidden]),
                    (format!("{prefix}.linear2.bias"), vec![hidden]),
                ] {
                    tensors.push((name, shape));
                }
            }
            let data = tensors
                .iter()
                .map(|(_, shape)| vec![0u8; shape.iter().product::<usize>() * 4])
                .collect::<Vec<_>>();
            let views = tensors
                .iter()
                .zip(&data)
                .map(|((name, shape), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, &path.join("model.safetensors")).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn raw_config() -> Value {
        json!({"encoder": "synthetic", "head_layers": 0, "max_len": 32, "head_max_len": 32})
    }

    fn laya(question_type: usize, markers: &[usize]) -> DecisionInput {
        DecisionInput::Laya {
            question_type,
            markers: markers.to_vec(),
        }
    }

    #[test]
    fn plans_sequence_relative_markers_and_token_types() {
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [laya(2, &[2, 0, 2]), DecisionInput::Warmup];
        let plan = plan_requests(&batch, &requests).unwrap();
        assert_eq!(plan.tokens, 5);
        assert_eq!(plan.token_types, vec![2, 2, 2, 0, 0]);
        assert_eq!(plan.marker_rows, vec![2, 0, 2, 3]);
        assert_eq!(plan.marker_offsets, vec![0, 3, 4]);
        assert_eq!(plan.counts, vec![3, 1]);
        assert_eq!(plan.total_markers, 4);
        assert_eq!(plan.max_length, 3);
        for (index, &count) in plan.counts.iter().enumerate() {
            let (start, end) = batch.sequence(index).unwrap();
            let begin = plan.marker_offsets[index] as usize;
            let end_marker = plan.marker_offsets[index + 1] as usize;
            assert_eq!(end_marker - begin, count);
            for &row in &plan.marker_rows[begin..end_marker] {
                let row = row as usize;
                assert!(row >= start && row < end, "marker leaked across sequences");
            }
        }
    }

    #[test]
    fn rejects_invalid_markers_types_and_empty_metadata() {
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        for request in [
            laya(0, &[]),
            laya(3, &[0]),
            laya(0, &[3]),
            laya(0, &[usize::MAX]),
        ] {
            assert!(plan_requests(&batch, &[request, DecisionInput::Warmup]).is_err());
        }
        assert!(plan_requests(&batch, &[DecisionInput::Warmup]).is_err());
        assert!(plan_requests(
            &batch,
            &[
                laya(0, &[0]),
                DecisionInput::OptionTokens { token_ids: vec![1] }
            ]
        )
        .is_err());
    }

    #[test]
    fn rejects_empty_or_degenerate_packed_batches() {
        assert!(PackedDecisionBatch::new(&[]).is_err());
        assert!(PackedDecisionBatch::new(&[1, 3]).is_err());
        assert!(PackedDecisionBatch::new(&[0, 3, 3]).is_err());
        assert!(PackedDecisionBatch::new(&[0]).is_err());
        let batch = PackedDecisionBatch::new(&[0, 1]).unwrap();
        assert!(plan_requests(&batch, &[laya(0, &[0]), DecisionInput::Warmup]).is_err());
    }

    #[test]
    fn output_and_result_budget_cover_logits_and_actions() {
        let requests = [laya(0, &[0, 2]), laya(1, &[1]), DecisionInput::Warmup];
        assert_eq!(output_elements(&requests).unwrap(), 4 + 3);
        assert_eq!(result_bytes(7).unwrap(), 28);
        assert!(result_bytes(usize::MAX).is_err());
        assert!(output_elements(&[laya(0, &[])]).is_err());
        assert!(output_elements(&[DecisionInput::OptionTokens { token_ids: vec![1] }]).is_err());
    }

    #[test]
    fn validates_sidecar_and_head_geometry_without_reading_payloads() {
        let fixture = Fixture::new(raw_config(), 0, &[3, HIDDEN]);
        let loaded = LayaConfig::load(fixture.path()).unwrap();
        assert_eq!(loaded.head_layers, 0);
        assert!(validate_laya_checkpoint(fixture.path(), "bfloat16").is_ok());
        let unsupported = validate_laya_checkpoint(fixture.path(), "float16")
            .expect_err("float16 must be rejected");
        assert!(unsupported.to_string().contains("float16"));
        assert!(unsupported.to_string().contains("bfloat16"));
        let checkpoint = Checkpoint::open(fixture.path()).unwrap();
        assert!(validate_head_sources(&checkpoint, &loaded, HIDDEN).is_ok());

        let wrong = Fixture::new(raw_config(), 0, &[3, HIDDEN - 1]);
        let wrong_config = LayaConfig::load(wrong.path()).unwrap();
        let checkpoint = Checkpoint::open(wrong.path()).unwrap();
        assert!(validate_head_sources(&checkpoint, &wrong_config, HIDDEN).is_err());

        let mut invalid = raw_config();
        invalid["head_layers"] = json!(MAX_HEAD_LAYERS + 1);
        let fixture = Fixture::new(invalid, 0, &[3, HIDDEN]);
        assert!(LayaConfig::load(fixture.path()).is_err());

        let mut zero = raw_config();
        zero["max_len"] = json!(0);
        let fixture = Fixture::new(zero, 0, &[3, HIDDEN]);
        assert!(LayaConfig::load(fixture.path()).is_err());
    }

    #[test]
    fn validates_one_refinement_layer_geometry() {
        let fixture = Fixture::new(raw_config(), 1, &[3, HIDDEN]);
        let config = LayaConfig::load(fixture.path()).unwrap();
        let checkpoint = Checkpoint::open(fixture.path()).unwrap();
        assert!(validate_head_sources(&checkpoint, &config, HIDDEN).is_ok());
    }

    #[test]
    fn admits_the_actual_official_modern_geometry_and_root_config_layout() {
        let actual: Value = serde_json::from_str(ACTUAL_ENCODER_CONFIG).unwrap();
        assert_eq!(actual["hidden_size"], 1024);
        assert_eq!(actual["num_attention_heads"], 16);
        assert_eq!(actual["intermediate_size"], 2624);
        assert_eq!(actual["num_hidden_layers"], 28);
        assert_eq!(actual["max_position_embeddings"], 8192);
        let fixture = Fixture::with_modern(raw_config(), actual, 0, &[3, 1024], false);
        let loaded = validate_laya_checkpoint(fixture.path(), "bfloat16").unwrap();
        assert_eq!(loaded.head_layers, 0);
        let config_path = laya_encoder_config(fixture.path()).unwrap();
        assert_eq!(config_path, fixture.path());
        let modern = crate::modern_encoder_model::ModernConfig::load(&config_path).unwrap();
        assert_eq!(modern.geometry.hidden_size, 1024);
        assert_eq!(modern.geometry.num_attention_heads, 16);
        assert_eq!(modern.geometry.intermediate_size, 2624);
        assert_eq!(modern.geometry.num_hidden_layers, 28);
        assert_eq!(modern.geometry.max_position_embeddings, 8192);
        assert_eq!(modern.local_radius, 64);
        assert_eq!(modern.global_every, 3);
        let checkpoint = Checkpoint::open(fixture.path()).unwrap();
        assert!(validate_head_sources(&checkpoint, &loaded, 1024).is_ok());
    }

    #[test]
    fn packs_one_common_metadata_tail_and_projects_its_capacity() {
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [laya(2, &[2, 0, 2]), DecisionInput::Warmup];
        let plan = plan_requests(&batch, &requests).unwrap();
        let words = plan.metadata_words().unwrap();
        assert_eq!(words, 5 + 4 + 3);
        let mut tail = vec![u32::MAX; words + 2];
        plan.write_metadata_tail(&mut tail).unwrap();
        assert_eq!(&tail[..5], &[2, 2, 2, 0, 0]);
        assert_eq!(&tail[5..9], &[2, 0, 2, 3]);
        assert_eq!(&tail[9..12], &[0, 3, 4]);
        assert_eq!(&tail[12..], &[u32::MAX, u32::MAX]);
        assert!(plan
            .write_metadata_tail(&mut tail[..words.saturating_sub(1)])
            .is_err());
    }

    #[test]
    fn cached_scratch_capacity_accepts_smaller_and_rejects_larger_requests() {
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [laya(2, &[2, 0, 2]), DecisionInput::Warmup];
        let plan = plan_requests(&batch, &requests).unwrap();
        let layout = LayaScratchLayout::from_plan(&plan, 1024, 4096).unwrap();
        let cached = layout.validated().unwrap();
        assert!(cached.fits(layout));
        let mut smaller = layout;
        smaller.tokens = 1;
        smaller.markers = 1;
        smaller.requests = 1;
        smaller.metadata_words = 1;
        assert!(cached.fits(smaller));

        let mut more_markers = layout;
        more_markers.markers += 1;
        assert!(!cached.fits(more_markers));
        let mut more_tokens = layout;
        more_tokens.tokens += 1;
        assert!(!cached.fits(more_tokens));
        let mut more_metadata = layout;
        more_metadata.metadata_words += 1;
        assert!(!cached.fits(more_metadata));
        assert!(layout.bytes().unwrap() > 0);
    }

    #[test]
    fn actual_geometry_laya_scratch_is_charged_in_the_work_projection() {
        let actual: Value = serde_json::from_str(ACTUAL_ENCODER_CONFIG).unwrap();
        let fixture = Fixture::with_modern(raw_config(), actual, 0, &[3, 1024], false);
        let modern = crate::modern_encoder_model::ModernConfig::load(
            &laya_encoder_config(fixture.path()).unwrap(),
        )
        .unwrap();
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [laya(2, &[2, 0, 2]), DecisionInput::Warmup];
        let plan = plan_requests(&batch, &requests).unwrap();
        let layout = LayaScratchLayout::from_plan(&plan, 1024, 4096).unwrap();
        let raw = infinity_v3_backend_core::Batch {
            multimodal: vec![],
            input_ids: vec![1; 5],
            token_type_ids: vec![0; 5],
            position_ids: vec![0, 1, 2, 0, 1],
            cumulative_seq_lengths: vec![0, 3, 5],
            max_length: 3,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let key = crate::work_key(&raw).unwrap();
        let input = crate::staging::input_capacity(key).unwrap();
        let base = crate::Work::<Bf16>::allocation_bytes(key, &modern.geometry).unwrap();
        let projected =
            crate::Work::<Bf16>::allocation_bytes_with_laya(key, &modern.geometry, &layout)
                .unwrap();
        let tail_bytes = layout.metadata_words * std::mem::size_of::<u32>();
        assert_eq!(projected - base, layout.bytes().unwrap() + tail_bytes);
        assert!(input > 0);
    }

    #[test]
    fn admits_authoritative_laya_sidecars_from_owned_provenance() {
        use sha2::{Digest, Sha256};

        let provenance: Value = serde_json::from_str(ACTUAL_PROVENANCE).unwrap();
        assert_eq!(
            provenance["revision"],
            "2741340228fc0c982ae9dd63b34dcd5a9e7ce3ce"
        );
        assert_eq!(provenance["weights_downloaded"], false);
        for (contents, name) in [
            (ACTUAL_RL_CONFIG.as_bytes(), "rl_agent_config.json"),
            (ACTUAL_ENCODER_CONFIG.as_bytes(), "encoder/config.json"),
        ] {
            let entry = provenance["files"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["file"] == name)
                .unwrap();
            assert_eq!(
                entry["sha256"],
                format!("{:x}", Sha256::digest(contents)),
                "owned {name} copy differs from provenance"
            );
            assert_eq!(entry["bytes"], contents.len());
        }

        let rl: Value = serde_json::from_str(ACTUAL_RL_CONFIG).unwrap();
        let encoder: Value = serde_json::from_str(ACTUAL_ENCODER_CONFIG).unwrap();
        assert_eq!(rl["head_layers"], 2);
        assert_eq!(rl["max_len"], 1024);
        assert_eq!(rl["head_max_len"], 256);
        assert_eq!(encoder["hidden_size"], 1024);
        assert_eq!(encoder["num_attention_heads"], 16);
        assert_eq!(encoder["num_hidden_layers"], 28);

        let fixture = Fixture::with_modern(rl, encoder, 2, &[3, 1024], true);
        let loaded = validate_laya_checkpoint(fixture.path(), "bfloat16").unwrap();
        assert_eq!(loaded.head_layers, 2);
        assert_eq!(loaded.max_len, 1024);
        assert_eq!(loaded.head_max_len, 256);
        let config_path = laya_encoder_config(fixture.path()).unwrap();
        assert_eq!(config_path, fixture.path().join(ENCODER_DIR));
        let modern = crate::modern_encoder_model::ModernConfig::load(&config_path).unwrap();
        assert_eq!(modern.geometry.hidden_size, 1024);
        assert_eq!(modern.geometry.num_attention_heads, 16);
        assert_eq!(modern.geometry.num_hidden_layers, 28);
        let checkpoint = Checkpoint::open(fixture.path()).unwrap();
        assert!(validate_head_sources(&checkpoint, &loaded, 1024).is_ok());
    }

    #[test]
    fn action_gather_plan_uses_refined_state_row_starts() {
        let starts = action_row_starts(&OFFSETS, 2).unwrap();
        assert_eq!(starts, vec![0, 3]);
        assert_ne!(
            starts,
            vec![0, 1],
            "action rows are token starts, not pooled slots"
        );
        assert!(action_row_starts(&OFFSETS, 1).is_err());
        assert!(action_row_starts(&[0, 2, 2], 2).is_err());
        assert!(action_row_starts(&[1, 2, 3], 2).is_err());
        assert!(action_row_starts(&[0], 0).is_err());
    }

    #[test]
    fn shared_a3_exports_cover_laya_h1024_and_plain_aten_norm() {
        let exports = A3_EXPORTS.lines().collect::<Vec<_>>();
        assert_eq!(exports.len(), 8);
        for name in [
            "infinity_embedding_word_norm_1024_candle",
            "infinity_rounded_residual_norm_1024_candle",
            "infinity_plain_layer_norm_bf16_aten",
            "infinity_gte_layer_norm_1024_candle",
        ] {
            assert!(exports.contains(&name), "missing shared A3 export {name}");
        }
    }
}
