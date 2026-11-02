// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

//! UltraBERT GPU execution for the ordinary Oxide factory.
//!
//! This module reuses the existing oxide runtime pieces: `Blas` cuBLAS
//! projections, `LtProvider` fused RMSNorm, the ABI7 FA4 provider, the packed
//! word-gather/RoPE/SiLU/pool kernels, `BaseRopeTables`, `WeightUploader` and
//! the decoder workspace/staging contracts. It adds only the UltraBERT-specific
//! composition: two immutable RoPE bases, per-layer symmetric windows, a
//! bidirectional layer loop, CLS pooling and the trained classifier readout.
//!
//! The factory builds this executor after checkpoint and FA4 capability checks.
//! No CUDA graphs, semantic padding or host-side projection math exists here.

use super::{
    arg, blocks, fa4, infer, launch, launch_threads, product, u32_size, HostIoLease, LtPlan,
    LtProvider, WorkKey,
};
use crate::{
    cublas::Blas,
    decoder::{
        BiasSupport, DecoderGeometry, DecoderKernels, DecoderMath, DecoderPlans, DecoderWork,
        ReadoutBias,
    },
    decoder_model::{DecoderConfig, DecoderFamily, DecoderRope},
    math::LinearPlan,
    output::{self, OutputGeometry},
    rotary::BaseRopeTables,
    staging::{HostIo, SlotPool},
    tensor::Bf16,
    ultrabert_qualification::{
        UltraBertConfig, UltraBertError, UltraBertFa4Requirement, UltraBertTask, UltraBertWeights,
    },
    ultrabert_qualification::{UltraBertDecisionKind, UltraBertDecisionWeights},
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, DecisionInput, Pool};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    mem::ManuallyDrop,
    path::Path,
    sync::{atomic::Ordering, Arc},
};

const ROPE_TABLE_BYTE_LIMIT: usize = 128 * 1024 * 1024;

const KERNEL_EXPORTS: [&str; 8] = [
    "infinity_embedding_gather",
    "infinity_rms_norm",
    "infinity_add_rms_norm",
    "infinity_rope_neox",
    "infinity_silu_mul_bf16_vector8_native",
    "infinity_pool",
    "infinity_selected_rows",
    "infinity_bf16_to_f32",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UltraBertEmbeddingPool {
    Cls,
    Mean,
    LastToken,
}

impl TryFrom<Pool> for UltraBertEmbeddingPool {
    type Error = UltraBertError;
    fn try_from(value: Pool) -> Result<Self, Self::Error> {
        match value {
            Pool::Cls => Ok(Self::Cls),
            Pool::Mean => Ok(Self::Mean),
            Pool::LastToken => Ok(Self::LastToken),
            Pool::Splade => Err(UltraBertError::UnsupportedPooling {
                mode: "splade".into(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UltraBertExecutionMode {
    Embedding(UltraBertEmbeddingPool),
    DecisionHidden,
    SequenceClassification,
    TokenClassification,
}

impl UltraBertExecutionMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding(_) => "embedding",
            Self::DecisionHidden => "decision_hidden",
            Self::SequenceClassification => "sequence_classification",
            Self::TokenClassification => "token_classification",
        }
    }
    const fn requires_classifier(self) -> bool {
        matches!(
            self,
            Self::SequenceClassification | Self::TokenClassification
        )
    }
}

#[derive(Clone, Debug)]
pub enum UltraBertOutput {
    Embeddings {
        pooled: Vec<Vec<f32>>,
        raw: Vec<Vec<Vec<f32>>>,
    },
    Logits {
        rows: usize,
        width: usize,
        values: Vec<f32>,
    },
}

/// CPU-only typed preparation requirements consumed by `prepare_gpu`.
#[derive(Clone, Debug, Serialize)]
pub struct UltraBertGpuRequirements {
    pub task: UltraBertTask,
    pub runtime_dtype: &'static str,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub local_attention_layers: usize,
    pub global_attention_layers: usize,
    pub rope_bases: Vec<f64>,
    pub fa4_requirements: Vec<UltraBertFa4Requirement>,
    pub missing_profiles: Vec<String>,
    pub classifier_labels: Option<usize>,
    pub kernel_exports: Vec<&'static str>,
}

fn setup(error: impl std::fmt::Display) -> UltraBertError {
    UltraBertError::ExecutionSetup {
        message: error.to_string(),
    }
}

pub(crate) fn requirements(
    config: &UltraBertConfig,
    fa4_requirements: &[UltraBertFa4Requirement],
) -> UltraBertGpuRequirements {
    let local_attention_layers = config
        .attention_windows
        .iter()
        .filter(|window| window.is_some())
        .count();
    let missing_profiles = fa4_requirements
        .iter()
        .filter(|requirement| !requirement.present)
        .map(|requirement| {
            format!(
                "{} {} (heads={} kv_heads={} d={} window={})",
                requirement.scope,
                requirement
                    .matched_profile
                    .as_deref()
                    .unwrap_or(requirement.window.as_str()),
                requirement.heads,
                requirement.kv_heads,
                requirement.head_dim,
                requirement.window
            )
        })
        .collect();
    let (rope_bases, _) = config.rope_bases();
    let mut kernel_exports = KERNEL_EXPORTS.to_vec();
    if config.vision.is_some() {
        kernel_exports.push("infinity_layer_norm");
        kernel_exports.push("infinity_encoder_activation_gelu_tanh_bf16");
        kernel_exports.push("infinity_ultravbert_pixel_shuffle_bf16");
    }
    if let Some(decision) = &config.decision {
        kernel_exports.push(match decision.kind {
            crate::ultrabert_qualification::UltraBertDecisionKind::Text => {
                "infinity_vela_encoder_head_f32"
            }
            crate::ultrabert_qualification::UltraBertDecisionKind::Vision => {
                "infinity_vela_vision_head_f32"
            }
        });
    }
    UltraBertGpuRequirements {
        task: config.task,
        runtime_dtype: "bfloat16",
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size: config.vocab_size,
        max_position_embeddings: config.max_position_embeddings,
        local_attention_layers,
        global_attention_layers: config.num_hidden_layers - local_attention_layers,
        rope_bases,
        fa4_requirements: fa4_requirements.to_vec(),
        missing_profiles,
        classifier_labels: config.num_labels,
        kernel_exports,
    }
}

pub(crate) fn validate_requirements(
    requirements: &UltraBertGpuRequirements,
) -> Result<(), UltraBertError> {
    if let Some(requirement) = requirements
        .fa4_requirements
        .iter()
        .find(|requirement| !requirement.present)
    {
        return Err(UltraBertError::MissingFa4Profile {
            scope: requirement.scope.clone(),
            heads: requirement.heads,
            kv_heads: requirement.kv_heads,
            head_dim: requirement.head_dim,
            window: requirement.window.clone(),
        });
    }
    Ok(())
}

fn decoder_work_config(config: &UltraBertConfig) -> DecoderConfig {
    DecoderConfig {
        family: DecoderFamily::Qwen2,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size: config.vocab_size,
        max_position_embeddings: config.max_position_embeddings,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta[0],
        is_causal: false,
        attention_window: None,
        rope: DecoderRope::FullNeoX,
        rope_scale: None,
        routed: None,
    }
}

struct UltraBertLinear {
    weight: DeviceBuffer<Bf16>,
}

struct UltraBertLayer {
    input_norm: DeviceBuffer<Bf16>,
    post_attention_norm: DeviceBuffer<Bf16>,
    qkv: UltraBertLinear,
    attention: UltraBertLinear,
    gate_up: UltraBertLinear,
    down: UltraBertLinear,
}

struct UltraBertWeightBuffers {
    embed: DeviceBuffer<Bf16>,
    norm: DeviceBuffer<Bf16>,
    layers: Vec<UltraBertLayer>,
    classifier: Option<(DeviceBuffer<Bf16>, DeviceBuffer<Bf16>)>,
    vision: Option<VisionWeightBuffers>,
}

impl UltraBertWeightBuffers {
    fn upload(
        uploader: &mut WeightUploader,
        host: &UltraBertWeights,
        config: &UltraBertConfig,
    ) -> Result<Self, UltraBertError> {
        let mut layers = Vec::with_capacity(host.layers.len());
        for layer in &host.layers {
            layers.push(UltraBertLayer {
                input_norm: uploader.upload(&[&layer.input_norm]).map_err(setup)?,
                post_attention_norm: uploader
                    .upload(&[&layer.post_attention_norm])
                    .map_err(setup)?,
                qkv: UltraBertLinear {
                    weight: uploader
                        .upload(&[&layer.q, &layer.k, &layer.v])
                        .map_err(setup)?,
                },
                attention: UltraBertLinear {
                    weight: uploader.upload(&[&layer.o]).map_err(setup)?,
                },
                gate_up: UltraBertLinear {
                    weight: uploader.upload(&[&layer.gate, &layer.up]).map_err(setup)?,
                },
                down: UltraBertLinear {
                    weight: uploader.upload(&[&layer.down]).map_err(setup)?,
                },
            });
        }
        let classifier = match (&host.classifier_weight, &host.classifier_bias) {
            (Some(weight), Some(bias)) => Some((
                uploader.upload(&[weight]).map_err(setup)?,
                uploader.upload(&[bias]).map_err(setup)?,
            )),
            (None, None) => None,
            _ => {
                return Err(setup(
                    "UltraBERT classifier weight and bias must be present together",
                ))
            }
        };
        if config.task != UltraBertTask::Embedding && classifier.is_none() {
            return Err(setup(
                "UltraBERT classification checkpoint has no classifier",
            ));
        }
        let vision = match (&config.vision, &host.vision) {
            (Some(_), Some(host)) => Some(upload_vision_weights(uploader, host)?),
            (None, None) => None,
            _ => return Err(setup("UltraVBERT vision config and weights disagree")),
        };
        Ok(Self {
            embed: uploader.upload(&[&host.embed]).map_err(setup)?,
            norm: uploader.upload(&[&host.norm]).map_err(setup)?,
            layers,
            classifier,
            vision,
        })
    }
}

struct VisionLinear {
    weight: DeviceBuffer<Bf16>,
    bias: DeviceBuffer<Bf16>,
}

struct VisionLayer {
    norm1_weight: DeviceBuffer<Bf16>,
    norm1_bias: DeviceBuffer<Bf16>,
    q: VisionLinear,
    k: VisionLinear,
    v: VisionLinear,
    o: VisionLinear,
    norm2_weight: DeviceBuffer<Bf16>,
    norm2_bias: DeviceBuffer<Bf16>,
    fc1: VisionLinear,
    fc2: VisionLinear,
}

struct VisionWeightBuffers {
    patch: VisionLinear,
    position: DeviceBuffer<Bf16>,
    layers: Vec<VisionLayer>,
    post_norm_weight: DeviceBuffer<Bf16>,
    post_norm_bias: DeviceBuffer<Bf16>,
    connector: DeviceBuffer<Bf16>,
    connector_norm: DeviceBuffer<Bf16>,
    identity: DeviceBuffer<Bf16>,
}

fn upload_vision_linear(
    uploader: &mut WeightUploader,
    source: &crate::ultrabert_qualification::UltraBertLinearSource,
) -> Result<VisionLinear, UltraBertError> {
    let weight = uploader.upload(&[&source.weight]).map_err(setup)?;
    let bias = source
        .bias
        .as_ref()
        .ok_or_else(|| setup("UltraVBERT vision linear bias is missing"))
        .and_then(|bias| uploader.upload(&[bias]).map_err(setup))?;
    Ok(VisionLinear { weight, bias })
}

fn upload_vision_weights(
    uploader: &mut WeightUploader,
    host: &crate::ultrabert_qualification::UltraBertVisionWeights,
) -> Result<VisionWeightBuffers, UltraBertError> {
    let mut layers = Vec::with_capacity(host.layers.len());
    for layer in &host.layers {
        layers.push(VisionLayer {
            norm1_weight: uploader.upload(&[&layer.norm1.weight]).map_err(setup)?,
            norm1_bias: uploader
                .upload(&[layer
                    .norm1
                    .bias
                    .as_ref()
                    .ok_or_else(|| setup("UltraVBERT vision norm1 bias is missing"))?])
                .map_err(setup)?,
            q: upload_vision_linear(uploader, &layer.q)?,
            k: upload_vision_linear(uploader, &layer.k)?,
            v: upload_vision_linear(uploader, &layer.v)?,
            o: upload_vision_linear(uploader, &layer.o)?,
            norm2_weight: uploader.upload(&[&layer.norm2.weight]).map_err(setup)?,
            norm2_bias: uploader
                .upload(&[layer
                    .norm2
                    .bias
                    .as_ref()
                    .ok_or_else(|| setup("UltraVBERT vision norm2 bias is missing"))?])
                .map_err(setup)?,
            fc1: upload_vision_linear(uploader, &layer.fc1)?,
            fc2: upload_vision_linear(uploader, &layer.fc2)?,
        });
    }
    let mut identity = vec![Bf16::from_bits(0); 768 * 768];
    for index in 0..768 {
        identity[index * 768 + index] = Bf16::from_bits(crate::model::to_bf16(1.0).map_err(setup)?);
    }
    Ok(VisionWeightBuffers {
        patch: VisionLinear {
            weight: uploader.upload(&[&host.patch.weight]).map_err(setup)?,
            bias: uploader
                .upload(&[host
                    .patch
                    .bias
                    .as_ref()
                    .ok_or_else(|| setup("UltraVBERT patch bias is missing"))?])
                .map_err(setup)?,
        },
        position: uploader.upload(&[&host.position]).map_err(setup)?,
        layers,
        post_norm_weight: uploader.upload(&[&host.post_norm.weight]).map_err(setup)?,
        post_norm_bias: uploader
            .upload(&[host
                .post_norm
                .bias
                .as_ref()
                .ok_or_else(|| setup("UltraVBERT post norm bias is missing"))?])
            .map_err(setup)?,
        connector: uploader.upload(&[&host.connector.weight]).map_err(setup)?,
        connector_norm: uploader.upload(&[&host.connector_norm]).map_err(setup)?,
        identity: DeviceBuffer::from_host(uploader_stream(uploader), &identity).map_err(setup)?,
    })
}

fn uploader_stream(uploader: &WeightUploader) -> &Arc<Stream> {
    uploader.stream()
}

struct RopeTable {
    cosine: DeviceBuffer<f32>,
    sine: DeviceBuffer<f32>,
}

fn upload_rope_tables(
    config: &UltraBertConfig,
    stream: &Arc<Stream>,
) -> Result<Vec<RopeTable>, UltraBertError> {
    let (bases, _) = config.rope_bases();
    let mut tables = Vec::with_capacity(bases.len());
    for base in bases {
        let table = BaseRopeTables::new(
            config.head_dim,
            base,
            config.max_position_embeddings,
            ROPE_TABLE_BYTE_LIMIT,
        )
        .map_err(setup)?;
        tables.push(RopeTable {
            cosine: DeviceBuffer::from_host(stream, table.cosine()).map_err(setup)?,
            sine: DeviceBuffer::from_host(stream, table.sine()).map_err(setup)?,
        });
    }
    Ok(tables)
}

struct VisionPlans {
    patch: LtPlan<Bf16>,
    q: LtPlan<Bf16>,
    k: LtPlan<Bf16>,
    v: LtPlan<Bf16>,
    o: LtPlan<Bf16>,
    fc1: LtPlan<Bf16>,
    fc2: LtPlan<Bf16>,
    connector: LinearPlan<Bf16>,
    identity: LinearPlan<Bf16>,
}

impl VisionPlans {
    fn new(lt: &LtProvider) -> Result<Self, UltraBertError> {
        let plan = |input, output| lt.prepare_bias_shape::<Bf16>(input, output).map_err(setup);
        let mut plans = Self {
            patch: plan(768, 768)?,
            q: plan(768, 768)?,
            k: plan(768, 768)?,
            v: plan(768, 768)?,
            o: plan(768, 768)?,
            fc1: plan(768, 3072)?,
            fc2: plan(3072, 768)?,
            connector: LinearPlan::contiguous(64, 12288, 768).map_err(setup)?,
            identity: LinearPlan::contiguous(1024, 768, 768).map_err(setup)?,
        };
        for plan in [
            &mut plans.patch,
            &mut plans.q,
            &mut plans.k,
            &mut plans.v,
            &mut plans.o,
            &mut plans.fc1,
            &mut plans.fc2,
        ] {
            lt.set_logical_rows(plan, 1024)
                .map_err(|error| setup(error.into_backend()))?;
        }
        Ok(plans)
    }
}

struct VisionScratch {
    patch_out: DeviceBuffer<Bf16>,
    x: DeviceBuffer<Bf16>,
    norm: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    attention: DeviceBuffer<Bf16>,
    branch: DeviceBuffer<Bf16>,
    mlp: DeviceBuffer<Bf16>,
    shuffled: DeviceBuffer<Bf16>,
    connector: DeviceBuffer<Bf16>,
    boundaries: DeviceBuffer<u32>,
}

impl VisionScratch {
    fn new(stream: &Arc<Stream>) -> Result<Self, UltraBertError> {
        let alloc = |elements: usize| {
            // SAFETY: every element is written by the ordered vision pass
            // before the first read, and the allocation is owned by the
            // executor for its whole lifetime.
            unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(setup)
        };
        Ok(Self {
            patch_out: alloc(1024 * 768)?,
            x: alloc(1024 * 768)?,
            norm: alloc(1024 * 768)?,
            q: alloc(1024 * 768)?,
            k: alloc(1024 * 768)?,
            v: alloc(1024 * 768)?,
            attention: alloc(1024 * 768)?,
            branch: alloc(1024 * 768)?,
            mlp: alloc(1024 * 3072)?,
            shuffled: alloc(64 * 12288)?,
            connector: alloc(64 * 768)?,
            boundaries: DeviceBuffer::from_host(stream, &[0u32, 1024]).map_err(setup)?,
        })
    }
}

struct VisionRuntime {
    plans: VisionPlans,
    scratch: VisionScratch,
    fa4: fa4::Fa4,
}

struct DecisionRuntime {
    kernel: Function,
    weights: DeviceBuffer<f32>,
    kind: UltraBertDecisionKind,
    projection: usize,
}

pub(crate) struct UltraBertDecisionPlan {
    kind: UltraBertDecisionKind,
    pub(crate) counts: Vec<usize>,
    markers: Vec<u32>,
    offsets: Vec<u32>,
    pub(crate) max_options: usize,
}

impl UltraBertDecisionPlan {
    pub(crate) fn new(
        config: &UltraBertConfig,
        batch: &Batch,
        requests: &[DecisionInput],
    ) -> Result<Self, UltraBertError> {
        let decision = config
            .decision
            .ok_or_else(|| setup("UltraBERT checkpoint has no decision head"))?;
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| setup("UltraBERT decision batch has no sequences"))?;
        if requests.len() != sequences {
            return Err(setup(
                "UltraBERT decision request count must match packed sequences",
            ));
        }
        let mut counts = Vec::with_capacity(sequences);
        let mut markers = Vec::new();
        let mut offsets = Vec::with_capacity(sequences + 1);
        offsets.push(0);
        for (sequence, request) in requests.iter().enumerate() {
            let start = *batch
                .cumulative_seq_lengths
                .get(sequence)
                .ok_or_else(|| setup("UltraBERT decision boundary is missing"))?
                as usize;
            let end = *batch
                .cumulative_seq_lengths
                .get(sequence + 1)
                .ok_or_else(|| setup("UltraBERT decision boundary is missing"))?
                as usize;
            let length = end
                .checked_sub(start)
                .ok_or_else(|| setup("UltraBERT decision boundary underflow"))?;
            let mut push = |value: usize| -> Result<(), UltraBertError> {
                markers.push(
                    u32::try_from(value)
                        .map_err(|_| setup("UltraBERT decision marker exceeds u32"))?,
                );
                Ok(())
            };
            match (decision.kind, request) {
                (_, DecisionInput::Warmup) if decision.kind == UltraBertDecisionKind::Text => {
                    push(0)?;
                    push(0)?;
                    push(length)?;
                    push(0)?;
                    counts.push(1);
                }
                (
                    _,
                    DecisionInput::Laya {
                        markers: request_markers,
                        ..
                    },
                ) if decision.kind == UltraBertDecisionKind::Text => {
                    if request_markers.len() < 4 {
                        return Err(setup(
                            "Vela encoder requires question, state interval and option markers",
                        ));
                    }
                    let question = request_markers[0];
                    let state_start = request_markers[1];
                    let state_end = request_markers[2];
                    if question >= length || state_start > state_end || state_end > length {
                        return Err(setup("Invalid Vela state interval or question marker"));
                    }
                    for marker in request_markers {
                        if *marker >= length {
                            return Err(setup("Invalid Vela option marker"));
                        }
                        push(*marker)?;
                    }
                    counts.push(request_markers.len() - 3);
                }
                (_, DecisionInput::Clef { fields })
                    if decision.kind == UltraBertDecisionKind::Vision =>
                {
                    if fields.len() != 1 || fields[0].options.is_empty() {
                        return Err(setup("Vela vision requires one field of option spans"));
                    }
                    for (span_start, span_end) in &fields[0].options {
                        if span_start >= span_end || *span_end > length {
                            return Err(setup("Invalid Vela vision option span"));
                        }
                        push(*span_start)?;
                        push(*span_end)?;
                    }
                    counts.push(fields[0].options.len());
                }
                (UltraBertDecisionKind::Vision, DecisionInput::Warmup) => {
                    push(0)?;
                    push(length)?;
                    counts.push(1);
                }
                _ => {
                    return Err(setup(
                        "UltraBERT decision protocol does not match its trained head",
                    ))
                }
            }
            offsets.push(
                u32::try_from(markers.len())
                    .map_err(|_| setup("UltraBERT decision marker table exceeds u32"))?,
            );
        }
        let max_options = counts.iter().copied().max().unwrap_or(0);
        if max_options == 0 {
            return Err(setup("UltraBERT decision request has no options"));
        }
        Ok(Self {
            kind: decision.kind,
            counts,
            markers,
            offsets,
            max_options,
        })
    }

    pub(crate) fn logit_elements(&self) -> Result<usize, UltraBertError> {
        self.counts.iter().try_fold(0usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| setup("UltraBERT decision logit count overflow"))
        })
    }
}

fn upload_f32_sources(
    stream: &Arc<Stream>,
    sources: &[&crate::model::WeightSource],
) -> Result<DeviceBuffer<f32>, UltraBertError> {
    let count = sources.iter().try_fold(0usize, |total, source| {
        total
            .checked_add(source.len())
            .ok_or_else(|| setup("UltraBERT decision weight size overflow"))
    })?;
    let mut host = Vec::new();
    host.try_reserve_exact(count)
        .map_err(|_| setup("UltraBERT decision weight allocation failed"))?;
    host.resize(count, 0.0f32);
    let mut offset = 0usize;
    for source in sources {
        let end = offset
            .checked_add(source.len())
            .ok_or_else(|| setup("UltraBERT decision weight offset overflow"))?;
        source.read_f32(0, &mut host[offset..end]).map_err(setup)?;
        offset = end;
    }
    DeviceBuffer::from_host(stream, &host).map_err(setup)
}

struct ClassifierBuffers {
    weight: DeviceBuffer<Bf16>,
    bias: DeviceBuffer<Bf16>,
    labels: usize,
}

pub struct UltraBertExecutor {
    config: UltraBertConfig,
    work_config: DecoderConfig,
    geometry: DecoderGeometry,
    weights: UltraBertWeightBuffers,
    plans: DecoderPlans,
    kernels: DecoderKernels,
    fa4_global: fa4::Fa4,
    fa4_local: BTreeMap<usize, fa4::Fa4>,
    rope: Vec<RopeTable>,
    layer_rope_indices: Vec<usize>,
    classifier: Option<ClassifierBuffers>,
    classifier_plan: Option<LtPlan<Bf16>>,
    vision: Option<VisionRuntime>,
    decision: Option<DecisionRuntime>,
    requirements: UltraBertGpuRequirements,
    lt: LtProvider,
    blas: Blas,
    io: Arc<SlotPool<HostIo>>,
    stream: Arc<Stream>,
    workspace: Option<UltraBertWork>,
}

impl std::fmt::Debug for UltraBertExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UltraBertExecutor")
            .field("task", &self.config.task)
            .field("layers", &self.config.num_hidden_layers)
            .field("local_layers", &self.layer_rope_indices.len())
            .finish_non_exhaustive()
    }
}

impl UltraBertExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: &UltraBertConfig,
        host: &UltraBertWeights,
        requirements: &UltraBertGpuRequirements,
        context: Arc<Context>,
        stream: Arc<Stream>,
        lt: LtProvider,
        blas: Blas,
        fa4_root: &Path,
    ) -> Result<Self, UltraBertError> {
        validate_requirements(requirements)?;
        let work_config = decoder_work_config(config);
        let geometry = DecoderGeometry::new(&work_config).map_err(setup)?;
        let mut kernels =
            DecoderKernels::load(&context, DecoderMath::Qwen2FullNeox).map_err(setup)?;
        let plans = DecoderPlans::new(
            &lt,
            &work_config,
            BiasSupport {
                qkv: false,
                attention: false,
                gate_up: false,
                down: false,
                requires_plain: true,
            },
            ReadoutBias::default(),
        )
        .map_err(setup)?;
        let mut uploader = WeightUploader::new(stream.clone()).map_err(setup)?;
        let mut weights = UltraBertWeightBuffers::upload(&mut uploader, host, config)?;
        uploader.finish().map_err(setup)?;
        let rope = upload_rope_tables(config, &stream)?;
        let (_, layer_rope_indices) = config.rope_bases();
        let global_spec = fa4::AttentionSpec {
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            causal: false,
            window: fa4::Window::Global,
            dtype: fa4::DType::Bf16,
            scale: fa4::Scale::RsqrtHeadDim,
        };
        let fa4_global = fa4::Fa4::new(context.clone(), fa4_root, global_spec).map_err(setup)?;
        let mut fa4_local = BTreeMap::new();
        for window in config.attention_windows.iter().flatten().copied() {
            if fa4_local.contains_key(&window) {
                continue;
            }
            let local_spec = fa4::AttentionSpec {
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                causal: false,
                window: fa4::Window::Local {
                    left: i32::try_from(window).map_err(setup)?,
                    right: i32::try_from(window).map_err(setup)?,
                },
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::RsqrtHeadDim,
            };
            fa4_local.insert(
                window,
                fa4::Fa4::new(context.clone(), fa4_root, local_spec).map_err(setup)?,
            );
        }
        let (classifier, classifier_plan) = if config.task.requires_classifier() {
            let labels = config
                .num_labels
                .ok_or_else(|| setup("UltraBERT classification labels are missing"))?;
            let (weight, bias) = weights
                .classifier
                .take()
                .ok_or_else(|| setup("UltraBERT classifier weights are missing"))?;
            let plan = lt
                .prepare_bias_shape::<Bf16>(config.hidden_size, labels)
                .map_err(setup)?;
            (
                Some(ClassifierBuffers {
                    weight,
                    bias,
                    labels,
                }),
                Some(plan),
            )
        } else {
            (None, None)
        };
        let vision = if config.vision.is_some() {
            let spec = fa4::AttentionSpec {
                heads: 12,
                kv_heads: 12,
                head_dim: 64,
                causal: false,
                window: fa4::Window::Global,
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::RsqrtHeadDim,
            };
            Some(VisionRuntime {
                plans: VisionPlans::new(&lt)?,
                scratch: VisionScratch::new(&stream)?,
                fa4: fa4::Fa4::new(context.clone(), fa4_root, spec).map_err(setup)?,
            })
        } else {
            None
        };
        let decision =
            if let Some(decision) = &host.decision {
                Some(match decision {
                    UltraBertDecisionWeights::Text(weights) => {
                        let kernel = kernels.vela_encoder_head.take().ok_or_else(|| {
                            setup("missing infinity_vela_encoder_head_f32 export")
                        })?;
                        let sources =
                            [
                                &weights.norm.weight,
                                weights
                                    .norm
                                    .bias
                                    .as_ref()
                                    .ok_or_else(|| setup("Vela encoder norm bias is missing"))?,
                                &weights.option.weight,
                                weights
                                    .option
                                    .bias
                                    .as_ref()
                                    .ok_or_else(|| setup("Vela encoder option bias is missing"))?,
                                &weights.question.weight,
                                weights.question.bias.as_ref().ok_or_else(|| {
                                    setup("Vela encoder question bias is missing")
                                })?,
                                &weights.state.weight,
                                weights
                                    .state
                                    .bias
                                    .as_ref()
                                    .ok_or_else(|| setup("Vela encoder state bias is missing"))?,
                                &weights.first.weight,
                                weights
                                    .first
                                    .bias
                                    .as_ref()
                                    .ok_or_else(|| setup("Vela encoder first bias is missing"))?,
                                &weights.last.weight,
                                weights
                                    .last
                                    .bias
                                    .as_ref()
                                    .ok_or_else(|| setup("Vela encoder last bias is missing"))?,
                                &weights.log_tau,
                            ];
                        DecisionRuntime {
                            kernel,
                            weights: upload_f32_sources(&stream, &sources)?,
                            kind: UltraBertDecisionKind::Text,
                            projection: config.decision.as_ref().map_or(0, |d| d.projection),
                        }
                    }
                    UltraBertDecisionWeights::Vision(weights) => {
                        let kernel = kernels
                            .vela_vision_head
                            .take()
                            .ok_or_else(|| setup("missing infinity_vela_vision_head_f32 export"))?;
                        let sources = [
                            &weights.first.weight,
                            weights
                                .first
                                .bias
                                .as_ref()
                                .ok_or_else(|| setup("Vela vision first bias is missing"))?,
                            &weights.norm.weight,
                            weights
                                .norm
                                .bias
                                .as_ref()
                                .ok_or_else(|| setup("Vela vision norm bias is missing"))?,
                            &weights.last.weight,
                            weights
                                .last
                                .bias
                                .as_ref()
                                .ok_or_else(|| setup("Vela vision last bias is missing"))?,
                        ];
                        DecisionRuntime {
                            kernel,
                            weights: upload_f32_sources(&stream, &sources)?,
                            kind: UltraBertDecisionKind::Vision,
                            projection: 0,
                        }
                    }
                })
            } else {
                None
            };
        let io = HostIo::pool(&context).map_err(setup)?;
        Ok(Self {
            config: config.clone(),
            work_config,
            geometry,
            weights,
            plans,
            kernels,
            fa4_global,
            fa4_local,
            rope,
            layer_rope_indices,
            classifier,
            classifier_plan,
            vision,
            decision,
            requirements: requirements.clone(),
            lt,
            blas,
            io,
            stream,
            workspace: None,
        })
    }

    pub fn requirements(&self) -> UltraBertGpuRequirements {
        self.requirements.clone()
    }

    pub(crate) fn config(&self) -> &UltraBertConfig {
        &self.config
    }

    pub fn forward(
        &mut self,
        batch: &Batch,
        mode: UltraBertExecutionMode,
    ) -> Result<UltraBertOutput, UltraBertError> {
        self.forward_impl(batch, mode, None)
    }

    pub(crate) fn forward_decision(
        &mut self,
        batch: &Batch,
        plan: &UltraBertDecisionPlan,
    ) -> Result<Vec<Vec<f32>>, UltraBertError> {
        let runtime = self
            .decision
            .as_ref()
            .ok_or_else(|| setup("UltraBERT decision runtime is missing"))?;
        if plan.kind != runtime.kind {
            return Err(setup("UltraBERT decision plan and trained head disagree"));
        }
        let output =
            self.forward_impl(batch, UltraBertExecutionMode::DecisionHidden, Some(plan))?;
        let UltraBertOutput::Logits { width, values, .. } = output else {
            return Err(setup("UltraBERT decision returned non-logit output"));
        };
        let mut results = Vec::with_capacity(plan.counts.len());
        for (sequence, count) in plan.counts.iter().enumerate() {
            let start = sequence
                .checked_mul(width)
                .ok_or_else(|| setup("UltraBERT decision output offset overflow"))?;
            let end = start
                .checked_add(*count)
                .ok_or_else(|| setup("UltraBERT decision output offset overflow"))?;
            results.push(
                values
                    .get(start..end)
                    .ok_or_else(|| setup("UltraBERT decision output range is short"))?
                    .to_vec(),
            );
        }
        Ok(results)
    }

    fn forward_impl(
        &mut self,
        batch: &Batch,
        mode: UltraBertExecutionMode,
        decision: Option<&UltraBertDecisionPlan>,
    ) -> Result<UltraBertOutput, UltraBertError> {
        self.validate_mode(mode)?;
        validate_batch(&self.config, batch)?;
        validate_selection(batch, mode)?;
        let tokens = batch.input_ids.len();
        let max_length = max_sequence_length(batch)?;
        let key = super::work_key(batch).map_err(setup)?;
        let (rows, labels) = match mode {
            UltraBertExecutionMode::DecisionHidden if decision.is_some() => {
                let plan = decision.ok_or_else(|| setup("UltraBERT decision plan is missing"))?;
                (batch.len(), plan.max_options)
            }
            UltraBertExecutionMode::Embedding(_) | UltraBertExecutionMode::DecisionHidden => (0, 0),
            UltraBertExecutionMode::SequenceClassification => {
                (batch.len(), self.config.num_labels.unwrap_or(0))
            }
            UltraBertExecutionMode::TokenClassification => {
                (tokens, self.config.num_labels.unwrap_or(0))
            }
        };
        let output_width = if mode.requires_classifier() {
            labels
        } else {
            self.config.hidden_size
        };
        let output_geometry =
            OutputGeometry::new(output_width, self.config.hidden_size).map_err(setup)?;
        let mut output_elements = output_geometry.host_elements(batch).map_err(setup)?;
        if mode.requires_classifier() || decision.is_some() {
            output_elements = rows
                .checked_mul(labels)
                .ok_or_else(|| setup("UltraBERT classifier output element overflow"))?;
        }
        let mut lease = HostIoLease::standalone(
            self.io.acquire().map_err(setup)?,
            self.io.clone(),
            self.stream.clone(),
        );
        lease.prepare(key, output_elements, false).map_err(setup)?;
        let mut work = match self.workspace.take() {
            Some(work) if work.fits(key, rows, labels, decision) => work,
            Some(work) => {
                drop(work);
                UltraBertWork::new(
                    self.stream.clone(),
                    &self.work_config,
                    key,
                    mode,
                    decision,
                    rows,
                    labels,
                )?
            }
            None => UltraBertWork::new(
                self.stream.clone(),
                &self.work_config,
                key,
                mode,
                decision,
                rows,
                labels,
            )?,
        };
        work.update_decision(decision)?;
        let (elements, _) = work
            .stage(batch, output_geometry, &mut lease)
            .map_err(setup)?;
        let ticket = lease.begin_tracked().map_err(setup)?;
        work.mark_last_use(ticket);
        work.upload(lease.slot()?.input.as_slice(), elements)
            .map_err(setup)?;
        self.preflight(&work, batch, mode, rows)?;
        self.enqueue(&mut work, batch, mode, decision, max_length, tokens)?;
        let output = match mode {
            UltraBertExecutionMode::DecisionHidden if decision.is_some() => {
                work.finish_logits(rows, labels, &mut lease)
                    .map_err(setup)?;
                lease.wait().map_err(setup)?;
                let values = lease
                    .output(
                        rows.checked_mul(labels)
                            .ok_or_else(|| setup("UltraBERT decision output element overflow"))?,
                    )
                    .map_err(setup)?
                    .to_vec();
                UltraBertOutput::Logits {
                    rows,
                    width: labels,
                    values,
                }
            }
            UltraBertExecutionMode::Embedding(_) | UltraBertExecutionMode::DecisionHidden => {
                work.finish_embeddings(
                    &self.lt,
                    &self.blas,
                    batch.pooled_indices.len(),
                    output::raw_rows(batch).map_err(setup)?,
                    self.config.hidden_size,
                    &mut lease,
                )
                .map_err(setup)?;
                lease.wait().map_err(setup)?;
                let host = lease.output(output_elements).map_err(setup)?;
                pack_embeddings(batch, host, self.config.hidden_size).map_err(setup)?
            }
            UltraBertExecutionMode::SequenceClassification
            | UltraBertExecutionMode::TokenClassification => {
                work.finish_logits(rows, labels, &mut lease)
                    .map_err(setup)?;
                lease.wait().map_err(setup)?;
                let values = lease
                    .output(
                        rows.checked_mul(labels)
                            .ok_or_else(|| setup("UltraBERT classifier output element overflow"))?,
                    )
                    .map_err(setup)?
                    .to_vec();
                UltraBertOutput::Logits {
                    rows,
                    width: labels,
                    values,
                }
            }
        };
        self.workspace = Some(work);
        Ok(output)
    }

    fn validate_mode(&self, mode: UltraBertExecutionMode) -> Result<(), UltraBertError> {
        let matches = matches!(
            (self.config.task, mode),
            (
                UltraBertTask::Embedding,
                UltraBertExecutionMode::Embedding(_)
            ) | (
                UltraBertTask::Decision,
                UltraBertExecutionMode::DecisionHidden
            ) | (
                UltraBertTask::SequenceClassification,
                UltraBertExecutionMode::SequenceClassification
            ) | (
                UltraBertTask::TokenClassification,
                UltraBertExecutionMode::TokenClassification
            )
        );
        if !matches {
            return Err(UltraBertError::ExecutionModeMismatch {
                checkpoint: self.config.task,
                requested: mode.as_str(),
            });
        }
        Ok(())
    }

    fn preflight(
        &mut self,
        work: &UltraBertWork,
        batch: &Batch,
        mode: UltraBertExecutionMode,
        rows: usize,
    ) -> Result<(), UltraBertError> {
        let tokens = batch.input_ids.len();
        self.plans
            .set_rows(&self.lt, tokens)
            .map_err(|error| setup(error.into_backend()))?;
        if mode.requires_classifier() {
            let plan = self
                .classifier_plan
                .as_mut()
                .ok_or_else(|| setup("UltraBERT classifier plan is missing"))?;
            self.lt
                .set_logical_rows(plan, rows)
                .map_err(|error| setup(error.into_backend()))?;
        }
        work.check(
            tokens,
            batch.len(),
            output::rows(batch).map_err(setup)?,
            &self.geometry,
            rows,
            self.config.num_labels.unwrap_or(0),
        )?;
        Ok(())
    }

    fn enqueue(
        &mut self,
        work: &mut UltraBertWork,
        batch: &Batch,
        mode: UltraBertExecutionMode,
        decision: Option<&UltraBertDecisionPlan>,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), UltraBertError> {
        let c = &self.config;
        let g = self.geometry;
        let sequences = batch.len();
        let h = g.hidden;
        let rows = u32_size(tokens).map_err(setup)?;
        let width = u32_size(h).map_err(setup)?;
        let eps = c.rms_norm_eps as f32;
        let full_silu_grid =
            blocks(product(tokens, g.silu_groups_per_row).map_err(setup)?).map_err(setup)?;
        let decision_pointers = if decision.is_some() {
            Some(work.decision_pointers()?)
        } else {
            None
        };
        let (backbone, logits) = work.split_mut();
        let pooled_ptr = backbone.pooled_deviceptr().map_err(setup)?;
        let buffers = backbone.buffers_mut();
        let base = buffers.input.cu_deviceptr();
        let id_bytes = u64::try_from(product(tokens, 4).map_err(setup)?).map_err(setup)?;
        let type_bytes = u64::try_from(product(tokens, 12).map_err(setup)?).map_err(setup)?;
        let mut ids = base;
        let positions = base
            .checked_add(id_bytes)
            .ok_or_else(|| setup("UltraBERT position address overflow"))?;
        let boundaries = base
            .checked_add(type_bytes)
            .ok_or_else(|| setup("UltraBERT boundary address overflow"))?;
        let selected = boundaries
            .checked_add(
                u64::try_from(product(batch.cumulative_seq_lengths.len(), 4).map_err(setup)?)
                    .map_err(setup)?,
            )
            .ok_or_else(|| setup("UltraBERT selection address overflow"))?;
        let residual = buffers.residual.cu_deviceptr();
        let residual_next = buffers.residual_next.cu_deviceptr();
        let branch = buffers.branch.cu_deviceptr();
        let normalized = buffers.normalized.cu_deviceptr();
        let projection = buffers.projection.cu_deviceptr();
        let activation = buffers.activation.cu_deviceptr();
        let k = &self.kernels;
        let mut gather_out = residual;
        let mut table = self.weights.embed.cu_deviceptr();
        let mut gather_tokens = rows;
        let mut gather_width = width;
        launch(
            &k.gather,
            &self.stream,
            blocks(product(tokens, h).map_err(setup)?).map_err(setup)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut gather_out),
                arg(&mut gather_tokens),
                arg(&mut gather_width),
            ],
        )
        .map_err(setup)?;
        Self::enqueue_vision(
            &mut self.vision,
            &self.weights.vision,
            c,
            k,
            &self.lt,
            &self.blas,
            &self.stream,
            batch,
            residual,
        )?;
        let mut norm_input = residual;
        let mut norm_weight = self
            .weights
            .layers
            .first()
            .ok_or_else(|| setup("UltraBERT checkpoint has no layers"))?
            .input_norm
            .cu_deviceptr();
        let mut norm_output = normalized;
        let mut rms_rows = rows;
        let mut rms_width = width;
        let mut rms_eps = eps;
        launch_threads(
            &k.rms_norm,
            &self.stream,
            (rms_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
            k.norm_geometry.threads,
            0,
            &mut [
                arg(&mut norm_input),
                arg(&mut norm_weight),
                arg(&mut norm_output),
                arg(&mut rms_rows),
                arg(&mut rms_width),
                arg(&mut rms_eps),
            ],
        )
        .map_err(setup)?;
        let strides = g.attention_strides(c.head_dim).map_err(setup)?;
        let q_offset = u64::try_from(product(g.q_width, 2).map_err(setup)?).map_err(setup)?;
        let k_offset = u64::try_from(product(g.kv_width, 2).map_err(setup)?).map_err(setup)?;
        let q_heads = u32_size(c.num_attention_heads).map_err(setup)?;
        let kv_heads = u32_size(c.num_key_value_heads).map_err(setup)?;
        let head_dim = u32_size(c.head_dim).map_err(setup)?;
        let table_stride = u32_size(c.head_dim / 2).map_err(setup)?;
        let qkv_stride = u64::try_from(g.qkv_width).map_err(setup)?;
        let mut residual_cur = residual;
        let mut residual_next_ptr = residual_next;
        let pairs = product(tokens, c.num_attention_heads + c.num_key_value_heads)
            .and_then(|value| {
                value.checked_mul(c.head_dim / 2).ok_or_else(|| {
                    BackendError::Start("UltraBERT rotary pair count overflow".into())
                })
            })
            .map_err(setup)?;
        let mut position_ptr = positions;
        let mut rope_tokens = rows;
        let mut rope_q_heads = q_heads;
        let mut rope_kv_heads = kv_heads;
        let mut rope_head_dim = head_dim;
        let mut rope_rotary_dim = head_dim;
        let mut q_stride = qkv_stride;
        let mut k_stride = qkv_stride;
        let mut rope_table_stride = table_stride;
        for (index, layer) in self.weights.layers.iter().enumerate() {
            let rope = &self.rope[self.layer_rope_indices[index]];
            let mut cos = rope.cosine.cu_deviceptr();
            let mut sin = rope.sine.cu_deviceptr();
            unsafe {
                self.blas.linear(
                    &self.plans.plain.qkv,
                    &*buffers.normalized,
                    &layer.qkv.weight,
                    buffers.projection,
                )?;
            }
            let mut q = projection;
            let mut key = projection
                .checked_add(q_offset)
                .ok_or_else(|| setup("UltraBERT Q address overflow"))?;
            let v = key
                .checked_add(k_offset)
                .ok_or_else(|| setup("UltraBERT V address overflow"))?;
            launch(
                &k.rope_neox,
                &self.stream,
                blocks(pairs).map_err(setup)?,
                0,
                &mut [
                    arg(&mut q),
                    arg(&mut key),
                    arg(&mut cos),
                    arg(&mut sin),
                    arg(&mut position_ptr),
                    arg(&mut rope_tokens),
                    arg(&mut rope_q_heads),
                    arg(&mut rope_kv_heads),
                    arg(&mut rope_head_dim),
                    arg(&mut rope_rotary_dim),
                    arg(&mut q_stride),
                    arg(&mut k_stride),
                    arg(&mut rope_table_stride),
                ],
            )
            .map_err(setup)?;
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
            let provider = if let Some(window) = c.attention_windows[index] {
                self.fa4_local
                    .get(&window)
                    .ok_or_else(|| setup("UltraBERT local FA4 provider is missing"))?
            } else {
                &self.fa4_global
            };
            unsafe {
                provider.forward_batch_bound(&self.stream, &attention)?;
            }
            unsafe {
                self.blas.linear(
                    &self.plans.plain.attention,
                    &*buffers.activation,
                    &layer.attention.weight,
                    buffers.branch,
                )?;
            }
            let mut add_input = branch;
            let mut add_residual = residual_cur;
            let mut add_weight = layer.post_attention_norm.cu_deviceptr();
            let mut add_output = normalized;
            let mut add_summed = residual_next_ptr;
            let mut add_rows = rows;
            let mut add_width = width;
            let mut add_eps = eps;
            launch_threads(
                &k.add_rms_norm,
                &self.stream,
                (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                k.norm_geometry.threads,
                0,
                &mut [
                    arg(&mut add_input),
                    arg(&mut add_residual),
                    arg(&mut add_weight),
                    arg(&mut add_output),
                    arg(&mut add_summed),
                    arg(&mut add_rows),
                    arg(&mut add_width),
                    arg(&mut add_eps),
                ],
            )
            .map_err(setup)?;
            std::mem::swap(&mut residual_cur, &mut residual_next_ptr);
            unsafe {
                self.blas.linear(
                    &self.plans.plain.gate_up,
                    &*buffers.normalized,
                    &layer.gate_up.weight,
                    buffers.projection,
                )?;
            }
            let mut silu_input = projection;
            let mut silu_output = activation;
            let mut silu_rows = rows;
            let mut silu_width = u32_size(c.intermediate_size).map_err(setup)?;
            launch(
                &k.silu_mul,
                &self.stream,
                full_silu_grid,
                0,
                &mut [
                    arg(&mut silu_input),
                    arg(&mut silu_output),
                    arg(&mut silu_rows),
                    arg(&mut silu_width),
                ],
            )
            .map_err(setup)?;
            unsafe {
                self.blas.linear(
                    &self.plans.plain.down,
                    &*buffers.activation,
                    &layer.down.weight,
                    buffers.branch,
                )?;
            }
            if let Some(next) = self.weights.layers.get(index + 1) {
                let mut add_input = branch;
                let mut add_residual = residual_cur;
                let mut add_weight = next.input_norm.cu_deviceptr();
                let mut add_output = normalized;
                let mut add_summed = residual_next_ptr;
                let mut add_rows = rows;
                let mut add_width = width;
                let mut add_eps = eps;
                launch_threads(
                    &k.add_rms_norm,
                    &self.stream,
                    (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                    k.norm_geometry.threads,
                    0,
                    &mut [
                        arg(&mut add_input),
                        arg(&mut add_residual),
                        arg(&mut add_weight),
                        arg(&mut add_output),
                        arg(&mut add_summed),
                        arg(&mut add_rows),
                        arg(&mut add_width),
                        arg(&mut add_eps),
                    ],
                )
                .map_err(setup)?;
                std::mem::swap(&mut residual_cur, &mut residual_next_ptr);
            }
        }
        let mut add_input = branch;
        let mut add_residual = residual_cur;
        let mut add_weight = self.weights.norm.cu_deviceptr();
        let mut add_output = normalized;
        let mut add_summed = residual_next_ptr;
        let mut add_rows = rows;
        let mut add_width = width;
        let mut add_eps = eps;
        launch_threads(
            &k.add_rms_norm,
            &self.stream,
            (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
            k.norm_geometry.threads,
            0,
            &mut [
                arg(&mut add_input),
                arg(&mut add_residual),
                arg(&mut add_weight),
                arg(&mut add_output),
                arg(&mut add_summed),
                arg(&mut add_rows),
                arg(&mut add_width),
                arg(&mut add_eps),
            ],
        )
        .map_err(setup)?;
        match mode {
            UltraBertExecutionMode::Embedding(pool) => {
                let mut pool_input = normalized;
                let mut pool_bounds = boundaries;
                let mut pool_indices = selected;
                let mut pool_out = pooled_ptr;
                let mut pool_count = u32_size(batch.pooled_indices.len()).map_err(setup)?;
                let mut pool_width = width;
                let mut mode = match pool {
                    UltraBertEmbeddingPool::Cls => 0u32,
                    UltraBertEmbeddingPool::Mean => 1,
                    UltraBertEmbeddingPool::LastToken => 2,
                };
                if !batch.pooled_indices.is_empty() {
                    launch(
                        &k.pool,
                        &self.stream,
                        blocks(product(batch.pooled_indices.len(), h).map_err(setup)?)
                            .map_err(setup)?,
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
                    )
                    .map_err(setup)?;
                }
                if !batch.raw_indices.is_empty() {
                    super::enqueue_raw_output(
                        &k.bf16_to_f32,
                        &self.stream,
                        normalized,
                        pooled_ptr,
                        selected,
                        batch,
                        h,
                    )
                    .map_err(setup)?;
                }
            }
            UltraBertExecutionMode::DecisionHidden => {
                if let Some(plan) = decision {
                    let runtime = self
                        .decision
                        .as_ref()
                        .ok_or_else(|| setup("UltraBERT decision runtime is missing"))?;
                    let (markers, offsets, counts) = decision_pointers
                        .ok_or_else(|| setup("UltraBERT decision metadata is missing"))?;
                    let logits = logits
                        .ok_or_else(|| setup("UltraBERT decision logits arena is missing"))?;
                    let mut hidden = normalized;
                    let mut boundary = boundaries;
                    let mut marker = markers;
                    let mut offset = offsets;
                    let mut weights = runtime.weights.cu_deviceptr();
                    let mut out = logits.f32.cu_deviceptr();
                    let mut count = counts;
                    let mut hidden_width = u32_size(h).map_err(setup)?;
                    let mut projection = u32_size(runtime.projection).map_err(setup)?;
                    let mut max_options = u32_size(plan.max_options).map_err(setup)?;
                    launch_threads(
                        &runtime.kernel,
                        &self.stream,
                        (
                            u32::try_from(batch.len())
                                .map_err(|_| setup("UltraBERT decision batch exceeds u32"))?,
                            1,
                            1,
                        ),
                        256,
                        0,
                        &mut [
                            arg(&mut hidden),
                            arg(&mut boundary),
                            arg(&mut marker),
                            arg(&mut offset),
                            arg(&mut weights),
                            arg(&mut out),
                            arg(&mut count),
                            arg(&mut hidden_width),
                            arg(&mut projection),
                            arg(&mut max_options),
                        ],
                    )
                    .map_err(setup)?;
                } else if !batch.raw_indices.is_empty() {
                    super::enqueue_raw_output(
                        &k.bf16_to_f32,
                        &self.stream,
                        normalized,
                        pooled_ptr,
                        selected,
                        batch,
                        h,
                    )
                    .map_err(setup)?;
                }
            }
            UltraBertExecutionMode::SequenceClassification
            | UltraBertExecutionMode::TokenClassification => {
                let classifier = self
                    .classifier
                    .as_ref()
                    .ok_or_else(|| setup("UltraBERT classifier buffers are missing"))?;
                let plan = self
                    .classifier_plan
                    .as_ref()
                    .ok_or_else(|| setup("UltraBERT classifier plan is missing"))?;
                let rows = if matches!(mode, UltraBertExecutionMode::SequenceClassification) {
                    batch.len()
                } else {
                    tokens
                };
                if matches!(mode, UltraBertExecutionMode::SequenceClassification) {
                    let mut gather_first = normalized;
                    let mut gather_second = normalized;
                    let mut gather_bounds = boundaries;
                    let mut gather_selected = 0u64;
                    let mut gather_first_out = branch;
                    let mut gather_second_out = branch;
                    let mut gather_count = u32_size(batch.len()).map_err(setup)?;
                    let mut gather_first_width = 0u32;
                    let mut gather_second_width = width;
                    let mut gather_mode = 0u32;
                    launch(
                        &k.selected_rows,
                        &self.stream,
                        blocks(product(batch.len(), h).map_err(setup)?).map_err(setup)?,
                        0,
                        &mut [
                            arg(&mut gather_first),
                            arg(&mut gather_second),
                            arg(&mut gather_bounds),
                            arg(&mut gather_selected),
                            arg(&mut gather_first_out),
                            arg(&mut gather_second_out),
                            arg(&mut gather_count),
                            arg(&mut gather_first_width),
                            arg(&mut gather_second_width),
                            arg(&mut gather_mode),
                        ],
                    )
                    .map_err(setup)?;
                }
                let logits = logits.ok_or_else(|| setup("UltraBERT classifier work is missing"))?;
                let input_buffer: &DeviceBuffer<Bf16> =
                    if matches!(mode, UltraBertExecutionMode::SequenceClassification) {
                        &*buffers.branch
                    } else {
                        &*buffers.normalized
                    };
                unsafe {
                    self.lt.enqueue(
                        plan,
                        input_buffer,
                        &classifier.weight,
                        &classifier.bias,
                        &mut logits.bf16,
                    )?;
                }
                let mut cast_input = logits.bf16.cu_deviceptr();
                let mut cast_output = logits.f32.cu_deviceptr();
                let mut cast_selected = 0u64;
                let mut cast_count = u32_size(rows).map_err(setup)?;
                let mut cast_width = u32_size(classifier.labels).map_err(setup)?;
                launch(
                    &k.bf16_to_f32,
                    &self.stream,
                    blocks(product(rows, classifier.labels).map_err(setup)?).map_err(setup)?,
                    0,
                    &mut [
                        arg(&mut cast_input),
                        arg(&mut cast_output),
                        arg(&mut cast_selected),
                        arg(&mut cast_count),
                        arg(&mut cast_width),
                    ],
                )
                .map_err(setup)?;
            }
        }
        Ok(())
    }
}

impl UltraBertExecutor {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_vision(
        vision: &mut Option<VisionRuntime>,
        plane_weights: &Option<VisionWeightBuffers>,
        config: &UltraBertConfig,
        kernels: &DecoderKernels,
        lt: &LtProvider,
        blas: &Blas,
        stream: &Arc<Stream>,
        batch: &Batch,
        residual: u64,
    ) -> Result<(), UltraBertError> {
        if !batch.multimodal.iter().any(Option::is_some) {
            return Ok(());
        }
        let Some(vision) = vision.as_mut() else {
            return Err(setup("UltraVBERT vision runtime is missing"));
        };
        let Some(weights) = plane_weights.as_ref() else {
            return Err(setup("UltraVBERT vision weights are missing"));
        };
        let text_eps = config.rms_norm_eps as f32;
        let Some(config) = config.vision else {
            return Err(setup("UltraVBERT config is missing"));
        };
        let layer_norm = kernels
            .layer_norm
            .as_ref()
            .ok_or_else(|| setup("UltraVBERT requires the generic BF16 layer-norm export"))?;
        let gelu = kernels
            .gelu_tanh
            .as_ref()
            .ok_or_else(|| setup("UltraVBERT requires the BF16 tanh-GELU export"))?;
        let pixel_shuffle = kernels
            .pixel_shuffle
            .as_ref()
            .ok_or_else(|| setup("UltraVBERT requires the pixel-shuffle export"))?;
        let VisionRuntime {
            plans,
            scratch,
            fa4,
        } = vision;
        let strides = [768i64, 64, 1, 768, 64, 1, 768, 64, 1, 768, 64, 1];
        let width = 768u32;
        let rows = 1024u32;
        let mlp_width = 3072u32;
        let norm_eps = config.layer_norm_eps as f32;
        for (sequence, media) in batch.multimodal.iter().enumerate() {
            let Some(media) = media else {
                continue;
            };
            let sequence_start = *batch
                .cumulative_seq_lengths
                .get(sequence)
                .ok_or_else(|| setup("UltraVBERT sequence boundary is missing"))?
                as usize;
            for (token_start, image) in &media.images {
                let mut pixels = Vec::with_capacity(1024 * 768);
                for &value in &image.pixels {
                    let bits = crate::model::to_bf16(value).map_err(setup)?;
                    pixels.push(Bf16::from_bits(bits));
                }
                let pixels = DeviceBuffer::from_host(stream, &pixels).map_err(setup)?;
                // SAFETY: context is bound by the engine, every buffer is
                // owned by this executor or request closure, and all work is
                // ordered on the same stream.
                unsafe {
                    lt.enqueue(
                        &plans.patch,
                        &pixels,
                        &weights.patch.weight,
                        &weights.patch.bias,
                        &mut scratch.patch_out,
                    )
                    .map_err(setup)?;
                }
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        scratch.x.cu_deviceptr(),
                        weights.position.cu_deviceptr(),
                        1024 * 768 * 2,
                        stream.cu_stream(),
                    )
                    .map_err(setup)?;
                }
                unsafe {
                    blas.linear_accumulate(
                        &plans.identity,
                        &scratch.patch_out,
                        &weights.identity,
                        &mut scratch.x,
                    )
                    .map_err(setup)?;
                }
                for layer in &weights.layers {
                    let mut input = scratch.x.cu_deviceptr();
                    let mut zero = 0u64;
                    let mut gamma = layer.norm1_weight.cu_deviceptr();
                    let mut beta = layer.norm1_bias.cu_deviceptr();
                    let mut norm_out = scratch.norm.cu_deviceptr();
                    let mut norm_rows = rows;
                    let mut norm_width = width;
                    let mut norm_eps_arg = norm_eps;
                    launch_threads(
                        layer_norm,
                        stream,
                        (rows, 1, 1),
                        256,
                        0,
                        &mut [
                            arg(&mut input),
                            arg(&mut zero),
                            arg(&mut zero),
                            arg(&mut gamma),
                            arg(&mut beta),
                            arg(&mut norm_out),
                            arg(&mut norm_rows),
                            arg(&mut norm_width),
                            arg(&mut norm_eps_arg),
                        ],
                    )
                    .map_err(setup)?;
                    unsafe {
                        lt.enqueue(
                            &plans.q,
                            &scratch.norm,
                            &layer.q.weight,
                            &layer.q.bias,
                            &mut scratch.q,
                        )
                        .map_err(setup)?;
                        lt.enqueue(
                            &plans.k,
                            &scratch.norm,
                            &layer.k.weight,
                            &layer.k.bias,
                            &mut scratch.k,
                        )
                        .map_err(setup)?;
                        lt.enqueue(
                            &plans.v,
                            &scratch.norm,
                            &layer.v.weight,
                            &layer.v.bias,
                            &mut scratch.v,
                        )
                        .map_err(setup)?;
                    }
                    let attention = fa4::AttentionBatch {
                        q: scratch.q.cu_deviceptr(),
                        k: scratch.k.cu_deviceptr(),
                        v: scratch.v.cu_deviceptr(),
                        out: scratch.attention.cu_deviceptr(),
                        q_offsets: scratch.boundaries.cu_deviceptr(),
                        k_offsets: scratch.boundaries.cu_deviceptr(),
                        q_tokens: 1024,
                        k_tokens: 1024,
                        batch: 1,
                        max_q: 1024,
                        strides,
                    };
                    unsafe { fa4.forward_batch_bound(stream, &attention).map_err(setup)? };
                    unsafe {
                        lt.enqueue(
                            &plans.o,
                            &scratch.attention,
                            &layer.o.weight,
                            &layer.o.bias,
                            &mut scratch.branch,
                        )
                        .map_err(setup)?;
                        blas.linear_accumulate(
                            &plans.identity,
                            &scratch.branch,
                            &weights.identity,
                            &mut scratch.x,
                        )
                        .map_err(setup)?;
                    }
                    let mut input = scratch.x.cu_deviceptr();
                    let mut zero = 0u64;
                    let mut gamma = layer.norm2_weight.cu_deviceptr();
                    let mut beta = layer.norm2_bias.cu_deviceptr();
                    let mut norm_out = scratch.norm.cu_deviceptr();
                    let mut norm_rows = rows;
                    let mut norm_width = width;
                    let mut norm_eps_arg = norm_eps;
                    launch_threads(
                        layer_norm,
                        stream,
                        (rows, 1, 1),
                        256,
                        0,
                        &mut [
                            arg(&mut input),
                            arg(&mut zero),
                            arg(&mut zero),
                            arg(&mut gamma),
                            arg(&mut beta),
                            arg(&mut norm_out),
                            arg(&mut norm_rows),
                            arg(&mut norm_width),
                            arg(&mut norm_eps_arg),
                        ],
                    )
                    .map_err(setup)?;
                    unsafe {
                        lt.enqueue(
                            &plans.fc1,
                            &scratch.norm,
                            &layer.fc1.weight,
                            &layer.fc1.bias,
                            &mut scratch.mlp,
                        )
                        .map_err(setup)?;
                    }
                    let mut gelu_input = scratch.mlp.cu_deviceptr();
                    let mut gelu_bias = 0u64;
                    let mut gelu_out = scratch.mlp.cu_deviceptr();
                    let mut gelu_rows = rows;
                    let mut gelu_width = mlp_width;
                    let mut gelu_input_stride = mlp_width;
                    let mut gelu_output_stride = mlp_width;
                    launch(
                        gelu,
                        stream,
                        blocks(1024 * 3072).map_err(setup)?,
                        0,
                        &mut [
                            arg(&mut gelu_input),
                            arg(&mut gelu_bias),
                            arg(&mut gelu_out),
                            arg(&mut gelu_rows),
                            arg(&mut gelu_width),
                            arg(&mut gelu_input_stride),
                            arg(&mut gelu_output_stride),
                        ],
                    )
                    .map_err(setup)?;
                    unsafe {
                        lt.enqueue(
                            &plans.fc2,
                            &scratch.mlp,
                            &layer.fc2.weight,
                            &layer.fc2.bias,
                            &mut scratch.branch,
                        )
                        .map_err(setup)?;
                        blas.linear_accumulate(
                            &plans.identity,
                            &scratch.branch,
                            &weights.identity,
                            &mut scratch.x,
                        )
                        .map_err(setup)?;
                    }
                }
                let mut input = scratch.x.cu_deviceptr();
                let mut zero = 0u64;
                let mut gamma = weights.post_norm_weight.cu_deviceptr();
                let mut beta = weights.post_norm_bias.cu_deviceptr();
                let mut norm_out = scratch.norm.cu_deviceptr();
                let mut norm_rows = rows;
                let mut norm_width = width;
                let mut norm_eps_arg = norm_eps;
                launch_threads(
                    layer_norm,
                    stream,
                    (rows, 1, 1),
                    256,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut zero),
                        arg(&mut zero),
                        arg(&mut gamma),
                        arg(&mut beta),
                        arg(&mut norm_out),
                        arg(&mut norm_rows),
                        arg(&mut norm_width),
                        arg(&mut norm_eps_arg),
                    ],
                )
                .map_err(setup)?;
                let mut source = scratch.norm.cu_deviceptr();
                let mut destination = scratch.shuffled.cu_deviceptr();
                launch(
                    pixel_shuffle,
                    stream,
                    blocks(64 * 12288).map_err(setup)?,
                    0,
                    &mut [arg(&mut source), arg(&mut destination)],
                )
                .map_err(setup)?;
                unsafe {
                    blas.linear(
                        &plans.connector,
                        &scratch.shuffled,
                        &weights.connector,
                        &mut scratch.connector,
                    )
                    .map_err(setup)?;
                }
                let mut norm_input = scratch.connector.cu_deviceptr();
                let mut norm_weight = weights.connector_norm.cu_deviceptr();
                let mut norm_output = scratch.connector.cu_deviceptr();
                let mut norm_rows = 64u32;
                let mut norm_width = width;
                let mut norm_eps_arg = text_eps;
                launch_threads(
                    &kernels.rms_norm,
                    stream,
                    (64, 1, 1),
                    256,
                    0,
                    &mut [
                        arg(&mut norm_input),
                        arg(&mut norm_weight),
                        arg(&mut norm_output),
                        arg(&mut norm_rows),
                        arg(&mut norm_width),
                        arg(&mut norm_eps_arg),
                    ],
                )
                .map_err(setup)?;
                let token = sequence_start
                    .checked_add(*token_start)
                    .ok_or_else(|| setup("UltraVBERT image token offset overflow"))?;
                let bytes = u64::try_from(token)
                    .ok()
                    .and_then(|token| token.checked_mul(768 * 2))
                    .ok_or_else(|| setup("UltraVBERT image token address overflow"))?;
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        residual
                            .checked_add(bytes)
                            .ok_or_else(|| setup("UltraVBERT residual address overflow"))?,
                        scratch.connector.cu_deviceptr(),
                        64 * 768 * 2,
                        stream.cu_stream(),
                    )
                    .map_err(setup)?;
                }
            }
        }
        Ok(())
    }
}

fn max_sequence_length(batch: &Batch) -> Result<usize, UltraBertError> {
    let mut maximum = 0usize;
    for pair in batch.cumulative_seq_lengths.windows(2) {
        let length = pair[1]
            .checked_sub(pair[0])
            .ok_or_else(|| setup("UltraBERT sequence length underflow"))?
            as usize;
        maximum = maximum.max(length);
    }
    Ok(maximum)
}

pub(crate) fn validate_selection(
    batch: &Batch,
    mode: UltraBertExecutionMode,
) -> Result<(), UltraBertError> {
    match mode {
        UltraBertExecutionMode::Embedding(_) => {
            if batch.pooled_indices.is_empty() && batch.raw_indices.is_empty() {
                return Err(setup(
                    "UltraBERT embedding request has no pooled or raw selections",
                ));
            }
            for &index in batch.pooled_indices.iter().chain(&batch.raw_indices) {
                if index as usize >= batch.len() {
                    return Err(setup("UltraBERT selection index exceeds batch"));
                }
            }
        }
        UltraBertExecutionMode::DecisionHidden => {
            if !batch.pooled_indices.is_empty()
                || batch.raw_indices.len() != batch.len()
                || batch.raw_indices.iter().copied().ne(0..batch.len() as u32)
            {
                return Err(setup(
                    "UltraBERT decision requires canonical all-sequence hidden rows",
                ));
            }
        }
        UltraBertExecutionMode::SequenceClassification => {
            if !batch.raw_indices.is_empty()
                || batch.pooled_indices.len() != batch.len()
                || batch
                    .pooled_indices
                    .iter()
                    .copied()
                    .ne(0..batch.len() as u32)
            {
                return Err(setup(
                    "UltraBERT sequence classification requires canonical pooled indices and no raw indices",
                ));
            }
        }
        UltraBertExecutionMode::TokenClassification => {
            if !batch.pooled_indices.is_empty()
                || batch.raw_indices.len() != batch.len()
                || batch.raw_indices.iter().copied().ne(0..batch.len() as u32)
            {
                return Err(setup(
                    "UltraBERT token classification requires canonical raw indices and no pooled indices",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_batch(
    config: &UltraBertConfig,
    batch: &Batch,
) -> Result<(), UltraBertError> {
    if batch.input_ids.is_empty() {
        return Err(UltraBertError::BatchEmpty);
    }
    if batch.compact_input_ids.is_some()
        || batch.compact_position_ids.is_some()
        || batch.scatter_unfold.is_some()
        || batch.fold_gather.is_some()
    {
        return Err(setup(
            "UltraBERT supports only packed media/text; compact and folded batches are rejected",
        ));
    }
    if !batch.multimodal.is_empty() {
        if batch.multimodal.len() != batch.cumulative_seq_lengths.len().saturating_sub(1) {
            return Err(setup(
                "UltraVBERT media entries must match packed sequences",
            ));
        }
        for (sequence, media) in batch.multimodal.iter().enumerate() {
            let Some(media) = media else {
                continue;
            };
            if config.vision.is_none() {
                return Err(setup("UltraBERT text checkpoints do not accept media"));
            }
            if !media.audios.is_empty() || media.images.is_empty() {
                return Err(setup("UltraVBERT accepts images only, 1-3 per request"));
            }
            let start = *batch
                .cumulative_seq_lengths
                .get(sequence)
                .ok_or_else(|| setup("UltraVBERT sequence boundary is missing"))?
                as usize;
            let end = *batch
                .cumulative_seq_lengths
                .get(sequence + 1)
                .ok_or_else(|| setup("UltraVBERT sequence boundary is missing"))?
                as usize;
            let length = end
                .checked_sub(start)
                .ok_or_else(|| setup("UltraVBERT sequence boundary underflow"))?;
            for (token_start, image) in &media.images {
                if image.grid_thw != [1, 32, 32]
                    || image.patch_dim != 768
                    || image.merge_size != 4
                    || image.pixels.len() != 1024 * 768
                {
                    return Err(setup("UltraVBERT expects fixed 512px SigLIP tiles"));
                }
                let token_end = token_start
                    .checked_add(image.token_count())
                    .ok_or_else(|| setup("UltraVBERT image token range overflow"))?;
                if token_end > length {
                    return Err(setup("UltraVBERT image token range exceeds its sequence"));
                }
            }
        }
    }
    let total = u32::try_from(batch.input_ids.len())
        .map_err(|_| setup("UltraBERT token count exceeds u32 offsets"))?;
    if batch.position_ids.len() != batch.input_ids.len()
        || batch.token_type_ids.len() != batch.input_ids.len()
        || batch.cumulative_seq_lengths.len() < 2
        || batch.cumulative_seq_lengths.first() != Some(&0)
        || batch
            .cumulative_seq_lengths
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        || batch.cumulative_seq_lengths.last().copied() != Some(total)
    {
        return Err(UltraBertError::BatchOffsets {
            message: "expected zero-based strictly increasing exact offsets".into(),
        });
    }
    let maximum = max_sequence_length(batch)?;
    if maximum > config.max_position_embeddings {
        return Err(UltraBertError::SequenceTooLong {
            length: maximum,
            max: config.max_position_embeddings,
        });
    }
    for &token in &batch.input_ids {
        if token as usize >= config.vocab_size {
            return Err(UltraBertError::TokenOutOfRange {
                token,
                vocab: config.vocab_size,
            });
        }
    }
    Ok(())
}

fn pack_embeddings(
    batch: &Batch,
    host: &[f32],
    hidden: usize,
) -> Result<UltraBertOutput, BackendError> {
    let pooled_rows = batch.pooled_indices.len();
    let mut pooled = Vec::with_capacity(pooled_rows);
    let mut offset = 0usize;
    for _ in 0..pooled_rows {
        let end = offset
            .checked_add(hidden)
            .ok_or_else(|| infer("UltraBERT pooled output offset overflow"))?;
        pooled.push(
            host.get(offset..end)
                .ok_or_else(|| infer("UltraBERT pooled output exceeds host arena"))?
                .to_vec(),
        );
        offset = end;
    }
    let mut raw = Vec::with_capacity(batch.raw_indices.len());
    for &index in &batch.raw_indices {
        let start = *batch
            .cumulative_seq_lengths
            .get(index as usize)
            .ok_or_else(|| infer("UltraBERT raw boundary index is invalid"))?
            as usize;
        let end = *batch
            .cumulative_seq_lengths
            .get(index as usize + 1)
            .ok_or_else(|| infer("UltraBERT raw boundary index is invalid"))?
            as usize;
        let mut rows = Vec::with_capacity(end.saturating_sub(start));
        for _ in start..end {
            let next = offset
                .checked_add(hidden)
                .ok_or_else(|| infer("UltraBERT raw output offset overflow"))?;
            rows.push(
                host.get(offset..next)
                    .ok_or_else(|| infer("UltraBERT raw output exceeds host arena"))?
                    .to_vec(),
            );
            offset = next;
        }
        raw.push(rows);
    }
    Ok(UltraBertOutput::Embeddings { pooled, raw })
}

struct LogitsWork {
    rows: usize,
    width: usize,
    bf16: ManuallyDrop<DeviceBuffer<Bf16>>,
    f32: ManuallyDrop<DeviceBuffer<f32>>,
    bytes: usize,
}

struct DecisionWork {
    markers: ManuallyDrop<DeviceBuffer<u32>>,
    offsets: ManuallyDrop<DeviceBuffer<u32>>,
    counts: ManuallyDrop<DeviceBuffer<u32>>,
    marker_capacity: usize,
    offset_capacity: usize,
    count_capacity: usize,
    bytes: usize,
}

impl DecisionWork {
    fn new(stream: &Arc<Stream>, plan: &UltraBertDecisionPlan) -> Result<Self, UltraBertError> {
        let markers = DeviceBuffer::from_host(stream, &plan.markers).map_err(setup)?;
        let offsets = DeviceBuffer::from_host(stream, &plan.offsets).map_err(setup)?;
        let counts =
            DeviceBuffer::from_host(stream, &vec![0u32; plan.counts.len()]).map_err(setup)?;
        let bytes = (plan.markers.len() + plan.offsets.len() + plan.counts.len())
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| setup("UltraBERT decision metadata size overflow"))?;
        super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(Self {
            marker_capacity: plan.markers.len(),
            offset_capacity: plan.offsets.len(),
            count_capacity: plan.counts.len(),
            markers: ManuallyDrop::new(markers),
            offsets: ManuallyDrop::new(offsets),
            counts: ManuallyDrop::new(counts),
            bytes,
        })
    }

    fn fits(&self, plan: &UltraBertDecisionPlan) -> bool {
        decision_capacity_fits(
            plan,
            self.marker_capacity,
            self.offset_capacity,
            self.count_capacity,
        )
    }

    fn update(
        &mut self,
        stream: &Arc<Stream>,
        plan: &UltraBertDecisionPlan,
    ) -> Result<(), UltraBertError> {
        if !self.fits(plan) {
            return Err(setup("UltraBERT decision metadata exceeds retained arena"));
        }
        self.markers
            .copy_from_host(stream, &plan.markers)
            .map_err(setup)?;
        self.offsets
            .copy_from_host(stream, &plan.offsets)
            .map_err(setup)?;
        self.counts
            .copy_from_host(stream, &vec![0u32; plan.counts.len()])
            .map_err(setup)?;
        Ok(())
    }

    fn pointers(&self) -> (u64, u64, u64) {
        (
            self.markers.cu_deviceptr(),
            self.offsets.cu_deviceptr(),
            self.counts.cu_deviceptr(),
        )
    }
}

fn decision_capacity_fits(
    plan: &UltraBertDecisionPlan,
    marker_capacity: usize,
    offset_capacity: usize,
    count_capacity: usize,
) -> bool {
    plan.markers.len() <= marker_capacity
        && plan.offsets.len() <= offset_capacity
        && plan.counts.len() <= count_capacity
}

struct UltraBertWork {
    stream: Arc<Stream>,
    backbone: DecoderWork,
    logits: Option<LogitsWork>,
    decision: Option<DecisionWork>,
    logits_bytes: usize,
    decision_bytes: usize,
}

impl UltraBertWork {
    fn new(
        stream: Arc<Stream>,
        config: &DecoderConfig,
        key: WorkKey,
        mode: UltraBertExecutionMode,
        decision: Option<&UltraBertDecisionPlan>,
        rows: usize,
        width: usize,
    ) -> Result<Self, UltraBertError> {
        let backbone = DecoderWork::new(stream.clone(), config, key).map_err(setup)?;
        let logits = if mode.requires_classifier() || decision.is_some() {
            if rows == 0 || width == 0 {
                return Err(setup("UltraBERT classifier rows/width must be nonzero"));
            }
            let elements = product(rows, width).map_err(setup)?;
            let bf16 = unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, elements) }
                .map_err(setup)?;
            let f32 = unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, elements) }
                .map_err(setup)?;
            let bytes = product(elements, 8).map_err(setup)?;
            super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
            super::record_live_arena_bytes();
            Some(LogitsWork {
                rows,
                width,
                bf16: ManuallyDrop::new(bf16),
                f32: ManuallyDrop::new(f32),
                bytes,
            })
        } else {
            None
        };
        let logits_bytes = logits.as_ref().map_or(0, |logits| logits.bytes);
        let decision_work = decision
            .map(|plan| DecisionWork::new(&stream, plan))
            .transpose()?;
        let decision_bytes = decision_work.as_ref().map_or(0, |work| work.bytes);
        Ok(Self {
            stream,
            backbone,
            logits,
            decision: decision_work,
            logits_bytes,
            decision_bytes,
        })
    }

    fn fits(
        &self,
        key: WorkKey,
        rows: usize,
        width: usize,
        decision: Option<&UltraBertDecisionPlan>,
    ) -> bool {
        let logits_fit = if width == 0 {
            self.logits.is_none()
        } else {
            self.logits
                .as_ref()
                .is_some_and(|logits| rows <= logits.rows && width == logits.width)
        };
        let decision_fit = match decision {
            Some(plan) => self.decision.as_ref().is_some_and(|work| work.fits(plan)),
            None => self.decision.is_none(),
        };
        self.backbone.fits(key) && logits_fit && decision_fit
    }
    fn check(
        &self,
        tokens: usize,
        sequences: usize,
        selections: usize,
        geometry: &DecoderGeometry,
        rows: usize,
        width: usize,
    ) -> Result<(), UltraBertError> {
        self.backbone
            .check(tokens, sequences, selections, geometry)
            .map_err(setup)?;
        match (&self.logits, width) {
            (None, 0) => {}
            (Some(logits), width) if width != 0 && rows <= logits.rows && width == logits.width => {
            }
            _ => return Err(setup("UltraBERT classifier request exceeds retained arena")),
        }
        Ok(())
    }
    fn stage(
        &self,
        batch: &Batch,
        geometry: OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        self.backbone.stage(batch, geometry, lease)
    }
    fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        self.backbone.upload(host, elements)
    }
    fn mark_last_use(&mut self, ticket: crate::work_retirement::CompletionTicket) {
        self.backbone.mark_last_use(ticket);
    }
    fn split_mut(&mut self) -> (&mut DecoderWork, Option<&mut LogitsWork>) {
        (&mut self.backbone, self.logits.as_mut())
    }
    fn update_decision(
        &mut self,
        plan: Option<&UltraBertDecisionPlan>,
    ) -> Result<(), UltraBertError> {
        match (self.decision.as_mut(), plan) {
            (Some(work), Some(plan)) => work.update(&self.stream, plan),
            (None, None) => Ok(()),
            _ => Err(setup("UltraBERT decision metadata plan mismatch")),
        }
    }
    fn decision_pointers(&self) -> Result<(u64, u64, u64), UltraBertError> {
        self.decision
            .as_ref()
            .map(DecisionWork::pointers)
            .ok_or_else(|| setup("UltraBERT decision metadata is missing"))
    }
    fn finish_embeddings(
        &mut self,
        lt: &LtProvider,
        blas: &Blas,
        pooled_rows: usize,
        raw_rows: usize,
        hidden: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        self.backbone
            .finish(None, lt, Some(blas), pooled_rows, raw_rows, hidden, lease)
    }
    fn finish_logits(
        &self,
        rows: usize,
        width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), UltraBertError> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| setup("UltraBERT classifier work is missing"))?;
        if rows > logits.rows || width != logits.width {
            return Err(setup("UltraBERT classifier request exceeds logits arena"));
        }
        let elements = product(rows, width).map_err(setup)?;
        let host = lease.slot_mut().map_err(setup)?.output.as_mut_slice();
        super::download_output_range(&self.stream, &logits.f32, 0, host, 0, elements)
            .map_err(setup)?;
        lease.record().map_err(setup)?;
        Ok(())
    }
}

impl Drop for UltraBertWork {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            if let Some(mut logits) = self.logits.take() {
                unsafe {
                    ManuallyDrop::drop(&mut logits.bf16);
                    ManuallyDrop::drop(&mut logits.f32);
                }
            }
            if let Some(mut decision) = self.decision.take() {
                unsafe {
                    ManuallyDrop::drop(&mut decision.markers);
                    ManuallyDrop::drop(&mut decision.offsets);
                    ManuallyDrop::drop(&mut decision.counts);
                }
            }
            super::LIVE_ARENA_BYTES
                .fetch_sub(self.logits_bytes + self.decision_bytes, Ordering::Relaxed);
            super::record_live_arena_bytes();
        }
        // On a failed drain the ManuallyDrop logits fields are left in place,
        // so the allocations are retained rather than released unsafely.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_pool_conversion_rejects_splade() {
        for (pool, expected) in [
            (Pool::Cls, UltraBertEmbeddingPool::Cls),
            (Pool::Mean, UltraBertEmbeddingPool::Mean),
            (Pool::LastToken, UltraBertEmbeddingPool::LastToken),
        ] {
            assert_eq!(UltraBertEmbeddingPool::try_from(pool).unwrap(), expected);
        }
        assert_eq!(
            UltraBertEmbeddingPool::try_from(Pool::Splade)
                .unwrap_err()
                .code(),
            "unsupported_pooling"
        );
    }

    fn decision_plan(
        markers: Vec<u32>,
        counts: Vec<usize>,
        max_options: usize,
    ) -> UltraBertDecisionPlan {
        let offsets = (0..=counts.len())
            .map(|index| {
                counts
                    .iter()
                    .take(index)
                    .map(|count| (*count + 3) as u32)
                    .sum::<u32>()
            })
            .collect::<Vec<_>>();
        UltraBertDecisionPlan {
            kind: crate::ultrabert_qualification::UltraBertDecisionKind::Text,
            counts,
            markers,
            offsets,
            max_options,
        }
    }

    #[test]
    fn decision_metadata_capacity_and_logit_budget_are_exact() {
        let small = decision_plan(vec![1, 0, 4, 2, 3], vec![2], 2);
        let large = decision_plan(vec![1, 0, 4, 2, 3, 4], vec![3], 3);
        assert!(decision_capacity_fits(&small, 5, 2, 1));
        assert!(!decision_capacity_fits(&large, 5, 2, 1));
        assert_eq!(small.logit_elements().unwrap(), 2);
        assert_eq!(large.logit_elements().unwrap(), 3);
        assert!(small.logit_elements().unwrap() * std::mem::size_of::<f32>() < 16);
    }

    #[test]
    fn missing_local_profile_is_a_typed_preparation_error() {
        let requirements = UltraBertGpuRequirements {
            task: UltraBertTask::Embedding,
            runtime_dtype: "bfloat16",
            hidden_size: 768,
            intermediate_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 6,
            num_key_value_heads: 6,
            head_dim: 128,
            vocab_size: 49152,
            max_position_embeddings: 32768,
            local_attention_layers: 16,
            global_attention_layers: 8,
            rope_bases: vec![10000.0, 2_560_000.0],
            fa4_requirements: vec![UltraBertFa4Requirement {
                scope: "local".into(),
                heads: 6,
                kv_heads: 6,
                head_dim: 128,
                causal: false,
                window: "local128/128".into(),
                matched_profile: None,
                present: false,
            }],
            missing_profiles: vec!["local local128/128".into()],
            classifier_labels: None,
            kernel_exports: KERNEL_EXPORTS.to_vec(),
        };
        let error = validate_requirements(&requirements).unwrap_err();
        assert_eq!(error.code(), "missing_fa4_profile");
        assert_eq!(error.to_string(), "missing FA4 profile for local attention: heads=6 kv_heads=6 d=128 causal=false window=local128/128");
    }
}
