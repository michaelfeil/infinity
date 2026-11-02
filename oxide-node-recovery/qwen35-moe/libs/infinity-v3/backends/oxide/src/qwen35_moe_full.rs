// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Real Qwen3.5-MoE 40-layer text forward wiring.
//!
//! This module adds the full-model execution plan that the V3 MoE foundation
//! deliberately left out. It reuses the donor dense attention/GDN preparation
//! (`PreparedAttentionWork`), the V3 shared backbone host manifest, the V3
//! routed provider/prepared plan and `DeviceFfnPlan::upload`. No attention or
//! GDN math is duplicated, and no safe raw-pointer API is exposed.
//!
//! The cold path uploads every embedding/norm/attention/routed/shared weight
//! through the bounded `WeightUploader`. The request path prepares typed
//! per-layer attention plans, one provider plan per layer and one shared
//! workspace/scratch set, then submits the whole 40-layer loop on the outer
//! stream. No layer allocates, queries, synchronizes or reads back to the CPU.

use crate::cublas::Blas;
use crate::fa4::Fa4;
use crate::gdn_prepared::{GdnBufferView, GdnPrepared};
use crate::math::LinearPlan;
use crate::qwen35_backbone::{Qwen35BackboneFfn, Qwen35BackboneKind, Qwen35BackboneWeights};
use crate::qwen35_config::Qwen35LayerType;
use crate::qwen35_execute::{
    AttentionGeometry, AttentionLayerWeights, PreparedAttentionWork, PreparedGdn, Qwen35AuxKernels,
    ResidualOp, Slabs,
};
use crate::qwen35_moe_config::Qwen35MoeConfig;
use crate::qwen35_moe_device::{validate_ffn_host, DeviceFfnPlan, Qwen35MoeDeviceFfn};
use crate::qwen35_moe_enqueue::{
    enqueue_routed_ffn, ensure_forward_admitted, RoutedFfnBuffers, RoutedWorkspacePlan,
};
use crate::qwen35_moe_layout::Qwen35MoeAttentionLayout;
use crate::qwen35_moe_provider::{PreparedQwen35MoeRouted, Qwen35MoeProvider};
use crate::qwen35_prepared::Qwen35Kernels;
use crate::qwen35_weights::{upload_backbone_attention, GdnScratch, Qwen35DeviceAttention};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::{arg, blocks, infer, launch, product, u32_size};
use cuda_core::{DeviceBuffer, Function, Module, Stream};
use infinity_v3_backend_core::BackendError;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

const ROUTED_EXPERTS: usize = 256;
const ROUTED_TOP_K: usize = 8;
const SHARED_GATE_KERNEL: &str = "infinity_qwen35_moe_shared_gate_bf16";
const BF16_TO_F32_KERNEL: &str = "infinity_bf16_to_f32";

/// The loaded module does not export the shared-expert gate kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MissingSharedGateKernel;

impl fmt::Display for MissingSharedGateKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Qwen3.5-MoE kernel module lacks the shared-gate BF16 export")
    }
}

impl std::error::Error for MissingSharedGateKernel {}

/// The loaded module does not export the BF16->F32 widening kernel used by the
/// routed provider's router logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MissingBf16ToF32Kernel;

impl fmt::Display for MissingBf16ToF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Qwen3.5-MoE kernel module lacks the BF16->F32 export")
    }
}

impl std::error::Error for MissingBf16ToF32Kernel {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35MoeFullGuardError {
    MissingSharedGateKernel,
    MissingBf16ToF32Kernel,
    MissingProvider,
    MissingRoutedWorkspace,
    MissingLayerPlan(usize),
}

impl fmt::Display for Qwen35MoeFullGuardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSharedGateKernel => {
                formatter.write_str("shared-gate kernel is not loaded")
            }
            Self::MissingBf16ToF32Kernel => {
                formatter.write_str("BF16->F32 widening kernel is not loaded")
            }
            Self::MissingProvider => formatter.write_str("routed provider is not loaded"),
            Self::MissingRoutedWorkspace => {
                formatter.write_str("routed workspace plan is not prepared")
            }
            Self::MissingLayerPlan(index) => {
                write!(formatter, "layer {index} has no prepared routed plan")
            }
        }
    }
}

impl std::error::Error for Qwen35MoeFullGuardError {}

impl From<Qwen35MoeFullGuardError> for BackendError {
    fn from(error: Qwen35MoeFullGuardError) -> Self {
        BackendError::Start(format!("Oxide Qwen3.5-MoE full: {error}"))
    }
}

/// One cold-bound module export set for the MoE full path.
pub(crate) struct Qwen35MoeFullKernels {
    shared_gate: Qwen35MoeSharedGateKernel,
    cast_bf16_to_f32: Function,
}

impl Qwen35MoeFullKernels {
    pub(crate) fn from_module(module: &Arc<Module>) -> Result<Self, Qwen35MoeFullGuardError> {
        let shared_gate = Qwen35MoeSharedGateKernel::from_module(module)
            .map_err(|_| Qwen35MoeFullGuardError::MissingSharedGateKernel)?;
        let cast_bf16_to_f32 = module
            .load_function(BF16_TO_F32_KERNEL)
            .map_err(|_| Qwen35MoeFullGuardError::MissingBf16ToF32Kernel)?;
        Ok(Self {
            shared_gate,
            cast_bf16_to_f32,
        })
    }

    pub(crate) fn shared_gate(&self) -> &Qwen35MoeSharedGateKernel {
        &self.shared_gate
    }

    pub(crate) fn cast_bf16_to_f32(&self) -> &Function {
        &self.cast_bf16_to_f32
    }
}

/// Typed loader for the shared-expert sigmoid gate. A module without the
/// export produces [`MissingSharedGateKernel`] rather than silently falling
/// back to a host implementation.
pub(crate) struct Qwen35MoeSharedGateKernel {
    function: Function,
}

impl Qwen35MoeSharedGateKernel {
    pub(crate) fn from_module(module: &Arc<Module>) -> Result<Self, MissingSharedGateKernel> {
        module
            .load_function(SHARED_GATE_KERNEL)
            .map(|function| Self { function })
            .map_err(|_| MissingSharedGateKernel)
    }

    /// Add the scaled shared expert into the routed output in place.
    ///
    /// # Safety
    /// All three spans belong to the active context, cover `tokens * width`
    /// BF16 elements and stay live until the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_bound(
        &self,
        stream: &Stream,
        routed: u64,
        shared_down: u64,
        gate: u64,
        tokens: u32,
        width: u32,
    ) -> Result<(), BackendError> {
        if tokens == 0 || width == 0 {
            return Ok(());
        }
        let elements = product(
            usize::try_from(tokens).map_err(infer)?,
            usize::try_from(width).map_err(infer)?,
        )?;
        let mut routed = routed;
        let mut shared_down = shared_down;
        let mut gate = gate;
        let mut tokens = tokens;
        let mut width = width;
        launch(
            &self.function,
            stream,
            (u32_size(elements.div_ceil(256))?, 1, 1),
            0,
            &mut [
                arg(&mut routed),
                arg(&mut shared_down),
                arg(&mut gate),
                arg(&mut tokens),
                arg(&mut width),
            ],
        )
    }
}

/// Require every owner needed by a real MoE full enqueue. The helper is pure
/// CPU and is exercised without a CUDA context.
pub(crate) fn ensure_full_enqueue_guard<'a>(
    shared_gate: Option<&'a Qwen35MoeSharedGateKernel>,
    provider: Option<&'a Qwen35MoeProvider>,
    workspace: Option<&'a RoutedWorkspacePlan>,
    prepared: Option<&'a PreparedQwen35MoeRouted>,
) -> Result<
    (
        &'a Qwen35MoeSharedGateKernel,
        &'a Qwen35MoeProvider,
        &'a RoutedWorkspacePlan,
        &'a PreparedQwen35MoeRouted,
    ),
    Qwen35MoeFullGuardError,
> {
    ensure_full_enqueue_guard_parts(
        shared_gate.is_some(),
        provider.is_some(),
        workspace.is_some(),
        prepared.is_some(),
    )?;
    Ok((
        shared_gate.expect("guard checked the shared-gate owner"),
        provider.expect("guard checked the provider owner"),
        workspace.expect("guard checked the workspace owner"),
        prepared.expect("guard checked the prepared plan owner"),
    ))
}

/// Pure part-check used by CPU tests and by the typed enqueue guard. Owner
/// presence is checked in fixed order so the first missing owner is explicit.
pub(crate) fn ensure_full_enqueue_guard_parts(
    shared_gate: bool,
    provider: bool,
    workspace: bool,
    prepared: bool,
) -> Result<(), Qwen35MoeFullGuardError> {
    if !shared_gate {
        return Err(Qwen35MoeFullGuardError::MissingSharedGateKernel);
    }
    if !provider {
        return Err(Qwen35MoeFullGuardError::MissingProvider);
    }
    if !workspace {
        return Err(Qwen35MoeFullGuardError::MissingRoutedWorkspace);
    }
    if !prepared {
        return Err(Qwen35MoeFullGuardError::MissingLayerPlan(0));
    }
    Ok(())
}

/// Host plan proof built from cached headers only. No shard payload is read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35MoeFullHostLayer {
    pub(crate) index: usize,
    pub(crate) attention: Qwen35LayerType,
    pub(crate) routed_gate_up: [usize; 3],
    pub(crate) routed_down: [usize; 3],
    pub(crate) shared_gate_up: [usize; 2],
    pub(crate) shared_down: [usize; 2],
    pub(crate) shared_gate: [usize; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35MoeFullHostPlan {
    layers: Vec<Qwen35MoeFullHostLayer>,
    full_attention_layers: usize,
    linear_attention_layers: usize,
}

impl Qwen35MoeFullHostPlan {
    pub(crate) fn from_backbone(backbone: &Qwen35BackboneWeights) -> Result<Self, BackendError> {
        if backbone.kind != Qwen35BackboneKind::Moe {
            return Err(infer(
                "Qwen3.5-MoE full plan requires a shared MoE backbone",
            ));
        }
        backbone.validate()?;
        if backbone.layers.len() != backbone.geometry.num_hidden_layers
            || backbone.geometry.layer_types.len() != backbone.layers.len()
        {
            return Err(infer("Qwen3.5-MoE full plan layer count mismatch"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(backbone.layers.len())
            .map_err(infer)?;
        let mut full_attention_layers = 0usize;
        let mut linear_attention_layers = 0usize;
        for (index, (layer, kind)) in backbone
            .layers
            .iter()
            .zip(backbone.geometry.layer_types.iter().copied())
            .enumerate()
        {
            match kind {
                Qwen35LayerType::Full => full_attention_layers += 1,
                Qwen35LayerType::Linear => linear_attention_layers += 1,
            }
            let Qwen35BackboneFfn::Routed { routed, shared, .. } = &layer.ffn else {
                return Err(infer(format!(
                    "Qwen3.5-MoE full plan layer {index} has a dense FFN"
                )));
            };
            validate_ffn_host(&layer.ffn)?;
            let intermediate = routed.intermediate;
            let hidden = routed.hidden;
            let experts = routed.experts;
            let shared_intermediate = shared.down.shape()[1];
            layers.push(Qwen35MoeFullHostLayer {
                index,
                attention: kind,
                routed_gate_up: [experts, product(intermediate, 2)?, hidden],
                routed_down: [experts, hidden, intermediate],
                shared_gate_up: [product(shared_intermediate, 2)?, hidden],
                shared_down: [hidden, shared_intermediate],
                shared_gate: [1, hidden],
            });
        }
        if full_attention_layers + linear_attention_layers != layers.len() {
            return Err(infer("Qwen3.5-MoE full plan attention count mismatch"));
        }
        Ok(Self {
            layers,
            full_attention_layers,
            linear_attention_layers,
        })
    }

    pub(crate) fn layers(&self) -> &[Qwen35MoeFullHostLayer] {
        &self.layers
    }

    pub(crate) const fn full_attention_layers(&self) -> usize {
        self.full_attention_layers
    }

    pub(crate) const fn linear_attention_layers(&self) -> usize {
        self.linear_attention_layers
    }
}

/// One cold-uploaded MoE full layer.
pub(crate) struct Qwen35MoeFullDeviceLayer {
    pub(crate) input_norm: DeviceBuffer<Bf16>,
    pub(crate) post_attention_norm: DeviceBuffer<Bf16>,
    pub(crate) attention: Qwen35DeviceAttention,
    pub(crate) ffn: Qwen35MoeDeviceFfn,
}

/// Every immutable device owner for the complete hybrid MoE text backbone.
pub(crate) struct Qwen35MoeFullDeviceWeights {
    pub(crate) config: Qwen35MoeConfig,
    pub(crate) embed: DeviceBuffer<Bf16>,
    pub(crate) final_norm: DeviceBuffer<Bf16>,
    pub(crate) layers: Vec<Qwen35MoeFullDeviceLayer>,
    pub(crate) lm_head: Option<crate::model::WeightSource>,
}

impl Qwen35MoeFullDeviceWeights {
    pub(crate) fn load(path: &Path, stream: &Arc<Stream>) -> Result<Self, BackendError> {
        let config = Qwen35MoeConfig::load(path)?;
        let backbone = Qwen35BackboneWeights::load_moe(path)?;
        Self::upload(&backbone, &config, stream)
    }

    pub(crate) fn upload(
        backbone: &Qwen35BackboneWeights,
        config: &Qwen35MoeConfig,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        if backbone.kind != Qwen35BackboneKind::Moe {
            return Err(infer(
                "Qwen3.5-MoE device upload requires a shared MoE backbone",
            ));
        }
        backbone.validate()?;
        if backbone.layers.len() != config.text.num_hidden_layers
            || config.text.num_hidden_layers != backbone.geometry.num_hidden_layers
            || config.text.hidden_size != backbone.geometry.hidden_size
            || config.text.num_attention_heads != backbone.geometry.num_attention_heads
            || config.text.num_key_value_heads != backbone.geometry.num_key_value_heads
            || config.text.head_dim != backbone.geometry.head_dim
            || config.text.layer_types != backbone.geometry.layer_types
        {
            return Err(infer("Qwen3.5-MoE device upload config/backbone mismatch"));
        }
        let mut uploader = WeightUploader::new(Arc::clone(stream))?;
        let embed = uploader.upload(&[&backbone.embeddings])?;
        let final_norm = uploader.upload(&[&backbone.norm])?;
        let mut scratch = GdnScratch::new_backbone(&backbone.geometry)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(backbone.layers.len())
            .map_err(infer)?;
        for (index, layer) in backbone.layers.iter().enumerate() {
            let input_norm = uploader.upload(&[&layer.input_norm])?;
            let post_attention_norm = uploader.upload(&[&layer.post_attention_norm])?;
            let kind = backbone.geometry.layer_types[index];
            let attention = upload_backbone_attention(
                &mut uploader,
                stream,
                &layer.attention,
                kind,
                &backbone.geometry,
                &mut scratch,
                index,
            )?;
            if !matches!(layer.ffn, Qwen35BackboneFfn::Routed { .. }) {
                return Err(infer(format!(
                    "Qwen3.5-MoE device upload layer {index} has a dense FFN"
                )));
            }
            let ffn = match DeviceFfnPlan::upload(&mut uploader, &layer.ffn)? {
                DeviceFfnPlan::Routed(ffn) => ffn,
                DeviceFfnPlan::Dense { .. } => {
                    return Err(infer(format!(
                        "Qwen3.5-MoE device upload layer {index} produced a dense FFN plan"
                    )));
                }
            };
            layers.push(Qwen35MoeFullDeviceLayer {
                input_norm,
                post_attention_norm,
                attention,
                ffn,
            });
        }
        uploader.finish()?;
        Ok(Self {
            config: config.clone(),
            embed,
            final_norm,
            layers,
            lm_head: backbone.lm_head.clone(),
        })
    }
}

pub(crate) struct Qwen35MoeFullScratch {
    pub(crate) shared_gate_up_out: DeviceBuffer<Bf16>,
    pub(crate) shared_activated: DeviceBuffer<Bf16>,
    pub(crate) shared_down_out: DeviceBuffer<Bf16>,
    pub(crate) shared_gate_logits: DeviceBuffer<Bf16>,
    pub(crate) routed_logits_bf16: DeviceBuffer<Bf16>,
    pub(crate) routed_logits_f32: DeviceBuffer<f32>,
    pub(crate) routed_workspace: DeviceBuffer<u8>,
    pub(crate) ffn_output: DeviceBuffer<Bf16>,
}

struct Qwen35MoeFullLayerPlan {
    provider: PreparedQwen35MoeRouted,
    workspace: RoutedWorkspacePlan,
    shared_gate_up: crate::math::GemmArguments,
    shared_down: crate::math::GemmArguments,
    shared_gate: crate::math::GemmArguments,
    next: ResidualOp,
}

/// Token-id gather precondition. The full enqueue must materialize embeddings
/// into the residual slab before the first RMSNorm.
pub(crate) fn ensure_embedding_input(tokens: usize, ids_len: usize) -> Result<(), BackendError> {
    if tokens == 0 || ids_len < tokens {
        return Err(infer(
            "Qwen3.5-MoE full enqueue requires one token id per residual row",
        ));
    }
    Ok(())
}

/// One typed, allocation-free per-request MoE full plan.
pub(crate) struct Qwen35MoeFullPlan {
    attention: PreparedAttentionWork,
    layers: Vec<Qwen35MoeFullLayerPlan>,
    tokens: usize,
    hidden: usize,
    shared_intermediate: usize,
}

impl Qwen35MoeFullPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        weights: &Qwen35MoeFullDeviceWeights,
        provider: &Qwen35MoeProvider,
        layout: &Qwen35MoeAttentionLayout,
        scratch: &Qwen35MoeFullScratch,
        slabs: &Slabs<'_>,
        tokens: usize,
        positions: u64,
        cumulative: u64,
        sequences: usize,
        max_len: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        let config = &weights.config;
        if config.moe.num_experts != ROUTED_EXPERTS
            || config.moe.num_experts_per_tok != ROUTED_TOP_K
        {
            return Err(infer(
                "Qwen3.5-MoE full plan requires the E256/top8 provider contract",
            ));
        }
        if provider.model().experts != ROUTED_EXPERTS as i32
            || provider.model().top_k != ROUTED_TOP_K as i32
        {
            return Err(infer(
                "Qwen3.5-MoE provider metadata differs from E256/top8",
            ));
        }
        if !Arc::ptr_eq(stream.context(), slabs.residual.context()) {
            return Err(infer("Qwen3.5-MoE full plan stream/slab context mismatch"));
        }
        if weights.layers.len() != config.text.num_hidden_layers {
            return Err(infer("Qwen3.5-MoE full plan device layer count mismatch"));
        }
        let geometry = AttentionGeometry::from_moe_text(&config.text);
        let mut attention_layers = Vec::new();
        attention_layers
            .try_reserve_exact(weights.layers.len())
            .map_err(infer)?;
        for layer in &weights.layers {
            attention_layers.push(AttentionLayerWeights {
                attention: &layer.attention,
                input_norm: &layer.input_norm,
                post_attention_norm: &layer.post_attention_norm,
            });
        }
        let attention = PreparedAttentionWork::new(
            &geometry,
            &attention_layers,
            &weights.embed,
            &weights.final_norm,
            layout,
            slabs,
            tokens,
            positions,
            cumulative,
            sequences,
            max_len,
        )?;
        let hidden = config.text.hidden_size;
        let intermediate = config.moe.moe_intermediate_size;
        let shared_intermediate = config.moe.shared_expert_intermediate_size;
        let two_shared = product(shared_intermediate, 2)?;
        if scratch.routed_workspace.len()
            != provider
                .prepare(tokens, hidden, intermediate, config.moe.norm_topk_prob)?
                .workspace_bytes()
        {
            return Err(infer(
                "Qwen3.5-MoE retained scratch differs from the provider workspace",
            ));
        }
        let norm = attention.norm_address();
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(weights.layers.len())
            .map_err(infer)?;
        let mut workspace_bytes = None;
        for (index, layer) in weights.layers.iter().enumerate() {
            let provider_plan =
                provider.prepare(tokens, hidden, intermediate, config.moe.norm_topk_prob)?;
            let workspace = RoutedWorkspacePlan::from_prepared(&provider_plan)?;
            if workspace.bytes() != scratch.routed_workspace.len() {
                return Err(infer(
                    "Qwen3.5-MoE layer workspace differs from the retained scratch",
                ));
            }
            if let Some(expected) = workspace_bytes {
                if workspace.bytes() != expected {
                    return Err(infer(
                        "Qwen3.5-MoE per-layer workspace sizes are not identical",
                    ));
                }
            } else {
                workspace_bytes = Some(workspace.bytes());
            }
            let shared_gate_up = LinearPlan::<Bf16>::contiguous(tokens, hidden, two_shared)
                .map_err(infer)?
                .arguments(
                    norm,
                    layer.ffn.shared_gate_up.cu_deviceptr(),
                    scratch.shared_gate_up_out.cu_deviceptr(),
                );
            let shared_down = LinearPlan::<Bf16>::contiguous(tokens, shared_intermediate, hidden)
                .map_err(infer)?
                .arguments(
                    scratch.shared_activated.cu_deviceptr(),
                    layer.ffn.shared_down.cu_deviceptr(),
                    scratch.shared_down_out.cu_deviceptr(),
                );
            let shared_gate = LinearPlan::<Bf16>::contiguous(tokens, hidden, 1)
                .map_err(infer)?
                .arguments(
                    norm,
                    layer.ffn.shared_gate_logit.cu_deviceptr(),
                    scratch.shared_gate_logits.cu_deviceptr(),
                );
            let next = ResidualOp::new(
                slabs.residual.cu_deviceptr(),
                scratch.ffn_output.cu_deviceptr(),
                attention.next_gamma(index)?,
                norm,
                tokens,
                hidden,
            )?;
            layers.push(Qwen35MoeFullLayerPlan {
                provider: provider_plan,
                workspace,
                shared_gate_up,
                shared_down,
                shared_gate,
                next,
            });
        }
        Ok(Self {
            attention,
            layers,
            tokens,
            hidden,
            shared_intermediate,
        })
    }

    pub(crate) fn token_count(&self) -> usize {
        self.tokens
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn shared_intermediate(&self) -> usize {
        self.shared_intermediate
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Submit the complete 40-layer hybrid backbone.
    ///
    /// # Safety
    /// The caller must bind the provider context on this thread and keep every
    /// device weight, slab, table, GDN owner, provider plan and stream alive
    /// until completion. On any error the whole stream must be drained or
    /// quarantined. No allocation, query, synchronize or host readback occurs
    /// in this function.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &mut self,
        weights: &Qwen35MoeFullDeviceWeights,
        provider: &Qwen35MoeProvider,
        kernels: &Qwen35MoeFullKernels,
        blas: &Blas,
        math: &Qwen35Kernels,
        aux: &Qwen35AuxKernels,
        fa4: &Fa4,
        input_ids: &DeviceBuffer<u32>,
        gather: &Function,
        stream: &Arc<Stream>,
        slabs: &Slabs<'_>,
        scratch: &mut Qwen35MoeFullScratch,
        gdn: &mut GdnPrepared,
        cumulative: &[u32],
        uploaded_cumulative: GdnBufferView<'_, u32>,
        cosine: u64,
        sine: u64,
        table_stride: u32,
    ) -> Result<(), BackendError> {
        if self.layers.len() != weights.layers.len()
            || self.layers.len() != weights.config.text.num_hidden_layers
        {
            return Err(infer(
                "Qwen3.5-MoE full enqueue layer count differs from retained weights",
            ));
        }
        if !Arc::ptr_eq(stream.context(), slabs.residual.context())
            || !Arc::ptr_eq(stream.context(), blas.context())
            || !Arc::ptr_eq(stream.context(), input_ids.context())
        {
            return Err(infer("Qwen3.5-MoE full enqueue context mismatch"));
        }
        ensure_embedding_input(self.tokens, input_ids.len())?;
        let mut ids = input_ids.cu_deviceptr();
        let mut table = weights.embed.cu_deviceptr();
        let mut residual = slabs.residual.cu_deviceptr();
        let mut rows = u32_size(self.tokens)?;
        let mut width = u32_size(self.hidden)?;
        // SAFETY: the caller retains ids/table/residual through completion and
        // the gather writes exactly the residual prefix normalized below.
        launch(
            gather,
            stream,
            blocks(product(self.tokens, self.hidden)?)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut residual),
                arg(&mut rows),
                arg(&mut width),
            ],
        )?;
        let frame = unsafe {
            self.attention
                .prepare_fresh_frame(gdn, slabs, cumulative, uploaded_cumulative)?
        };
        let prepared_gdn = PreparedGdn::Fresh(&frame);
        // SAFETY: the caller owns the bound context and retained owners.
        unsafe { self.attention.enqueue_first(math, stream)? };
        let hidden = u32_size(self.hidden)?;
        let shared_intermediate = self.shared_intermediate;
        let token_count = self.tokens;
        let tokens = u32_size(token_count)?;
        let Self {
            attention, layers, ..
        } = self;
        for (index, layer_plan) in layers.iter_mut().enumerate() {
            let device_layer = &weights.layers[index];
            let (shared_gate, layer_provider, workspace, prepared) = ensure_full_enqueue_guard(
                Some(kernels.shared_gate()),
                Some(provider),
                Some(&layer_plan.workspace),
                Some(&layer_plan.provider),
            )?;
            if prepared.contract() != layer_provider.contract() {
                return Err(infer(
                    "Qwen3.5-MoE layer provider contract changed after preparation",
                ));
            }
            ensure_forward_admitted(prepared)?;
            // SAFETY: prepared attention plans bind retained weights and slabs.
            unsafe {
                attention.enqueue_attention(
                    index,
                    &device_layer.attention,
                    blas,
                    math,
                    aux,
                    fa4,
                    stream,
                    slabs,
                    gdn,
                    &prepared_gdn,
                    cosine,
                    sine,
                    table_stride,
                )?;
                attention.enqueue_post(index, math, stream)?;
            }

            let buffers = RoutedFfnBuffers {
                input: slabs.activation,
                output: &mut scratch.ffn_output,
                logits_bf16: &mut scratch.routed_logits_bf16,
                logits_f32: &scratch.routed_logits_f32,
                workspace: &mut scratch.routed_workspace,
                stream,
            };
            // SAFETY: the typed scratch, device FFN weights and stream owners
            // are all retained by this plan and the outer Work.
            unsafe {
                enqueue_routed_ffn(
                    prepared,
                    &device_layer.ffn,
                    blas,
                    kernels.cast_bf16_to_f32(),
                    buffers,
                    *workspace,
                )?;
                blas.linear_bound(&layer_plan.shared_gate_up)?;
            }

            let mut silu_input = scratch.shared_gate_up_out.cu_deviceptr();
            let mut silu_output = scratch.shared_activated.cu_deviceptr();
            let mut silu_rows = tokens;
            let mut silu_width = u32_size(shared_intermediate)?;
            let groups = shared_intermediate.div_ceil(8);
            let silu_grid = u32_size(product(token_count, groups)?.div_ceil(256))?;
            launch(
                &aux.silu,
                stream,
                (silu_grid, 1, 1),
                0,
                &mut [
                    arg(&mut silu_input),
                    arg(&mut silu_output),
                    arg(&mut silu_rows),
                    arg(&mut silu_width),
                ],
            )?;

            // SAFETY: shared down/gate weights and the typed scratch belong to
            // this plan and stay live through stream completion.
            unsafe {
                blas.linear_bound(&layer_plan.shared_down)?;
                blas.linear_bound(&layer_plan.shared_gate)?;
                shared_gate.enqueue_bound(
                    stream,
                    scratch.ffn_output.cu_deviceptr(),
                    scratch.shared_down_out.cu_deviceptr(),
                    scratch.shared_gate_logits.cu_deviceptr(),
                    tokens,
                    hidden,
                )?;
                layer_plan
                    .next
                    .enqueue(math, stream, weights.config.text.rms_norm_eps)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn scratch_allocation_bytes_for_shape(
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    two_shared: usize,
    workspace_bytes: usize,
) -> Result<usize, BackendError> {
    let _ = intermediate;
    if workspace_bytes == 0 {
        return Err(infer("Qwen3.5-MoE provider workspace is empty"));
    }
    let bf16 = std::mem::size_of::<Bf16>();
    let f32_bytes = std::mem::size_of::<f32>();
    let mut total = 0usize;
    for (elements, width) in [
        (product(tokens, two_shared)?, bf16),
        (product(tokens, shared_intermediate)?, bf16),
        (product(tokens, hidden)?, bf16),
        (tokens, bf16),
        (product(tokens, ROUTED_EXPERTS)?, bf16),
        (product(tokens, ROUTED_EXPERTS)?, f32_bytes),
        (workspace_bytes, 1),
        (product(tokens, hidden)?, bf16),
    ] {
        total = elements
            .checked_mul(width)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| infer("Qwen3.5-MoE scratch byte count overflow"))?;
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scratch_allocation_bytes(
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    two_shared: usize,
    provider: &Qwen35MoeProvider,
    renormalize: bool,
) -> Result<usize, BackendError> {
    let workspace_bytes = provider
        .prepare(tokens, hidden, intermediate, renormalize)?
        .workspace_bytes();
    scratch_allocation_bytes_for_shape(
        tokens,
        hidden,
        intermediate,
        shared_intermediate,
        two_shared,
        workspace_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn allocate_scratch(
    stream: &Arc<Stream>,
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    two_shared: usize,
    provider: &Qwen35MoeProvider,
    renormalize: bool,
) -> Result<Qwen35MoeFullScratch, BackendError> {
    let workspace_bytes = provider
        .prepare(tokens, hidden, intermediate, renormalize)?
        .workspace_bytes();
    if workspace_bytes == 0 {
        return Err(infer("Qwen3.5-MoE provider workspace is empty"));
    }
    Ok(Qwen35MoeFullScratch {
        shared_gate_up_out: DeviceBuffer::zeroed(stream, product(tokens, two_shared)?)
            .map_err(infer)?,
        shared_activated: DeviceBuffer::zeroed(stream, product(tokens, shared_intermediate)?)
            .map_err(infer)?,
        shared_down_out: DeviceBuffer::zeroed(stream, product(tokens, hidden)?).map_err(infer)?,
        shared_gate_logits: DeviceBuffer::zeroed(stream, tokens).map_err(infer)?,
        routed_logits_bf16: DeviceBuffer::zeroed(stream, product(tokens, ROUTED_EXPERTS)?)
            .map_err(infer)?,
        routed_logits_f32: DeviceBuffer::zeroed(stream, product(tokens, ROUTED_EXPERTS)?)
            .map_err(infer)?,
        routed_workspace: DeviceBuffer::zeroed(stream, workspace_bytes).map_err(infer)?,
        ffn_output: DeviceBuffer::zeroed(stream, product(tokens, hidden)?).map_err(infer)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHED_35B: &str =
        "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b";

    #[test]
    fn full_host_plan_from_cached_35b_headers_has_no_dense_layers() {
        let path = Path::new(CACHED_35B);
        if !path.join("config.json").is_file() {
            if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
                panic!("cached 35B checkpoint required but absent");
            }
            return;
        }
        let backbone = Qwen35BackboneWeights::load_moe(path).unwrap();
        let plan = Qwen35MoeFullHostPlan::from_backbone(&backbone).unwrap();
        assert_eq!(plan.layers().len(), 40);
        assert_eq!(plan.linear_attention_layers(), 30);
        assert_eq!(plan.full_attention_layers(), 10);
        let full = plan
            .layers()
            .iter()
            .filter(|layer| layer.attention == Qwen35LayerType::Full)
            .count();
        let linear = plan
            .layers()
            .iter()
            .filter(|layer| layer.attention == Qwen35LayerType::Linear)
            .count();
        assert_eq!((full, linear), (10, 30));
        for (index, layer) in plan.layers().iter().enumerate() {
            assert_eq!(layer.index, index);
            assert_eq!(layer.routed_gate_up, [256, 1024, 2048]);
            assert_eq!(layer.routed_down, [256, 2048, 512]);
            assert_eq!(layer.shared_gate_up, [1024, 2048]);
            assert_eq!(layer.shared_down, [2048, 512]);
            assert_eq!(layer.shared_gate, [1, 2048]);
        }
    }

    #[test]
    fn enqueue_guard_rejects_missing_shared_gate_kernel_provider_and_workspace() {
        assert_eq!(
            ensure_full_enqueue_guard_parts(false, false, false, false).unwrap_err(),
            Qwen35MoeFullGuardError::MissingSharedGateKernel
        );
        assert_eq!(
            ensure_full_enqueue_guard_parts(true, false, false, false).unwrap_err(),
            Qwen35MoeFullGuardError::MissingProvider
        );
        assert_eq!(
            ensure_full_enqueue_guard_parts(true, true, false, false).unwrap_err(),
            Qwen35MoeFullGuardError::MissingRoutedWorkspace
        );
        assert_eq!(
            ensure_full_enqueue_guard_parts(true, true, true, false).unwrap_err(),
            Qwen35MoeFullGuardError::MissingLayerPlan(0)
        );
        assert!(ensure_full_enqueue_guard_parts(true, true, true, true).is_ok());
        assert!(matches!(
            ensure_full_enqueue_guard(None, None, None, None),
            Err(Qwen35MoeFullGuardError::MissingSharedGateKernel)
        ));
        assert!(format!("{}", MissingSharedGateKernel).contains("shared-gate"));
        assert!(format!("{}", MissingBf16ToF32Kernel).contains("BF16"));
    }

    #[test]
    fn provider_workspace_metadata_is_arch_specific_for_full_plan() {
        let Some(root) = std::env::var_os("QWEN35_MOE_AOT_ROOT") else {
            if std::env::var_os("QWEN35_MOE_REQUIRE_AOT").is_some() {
                panic!("QWEN35_MOE_AOT_ROOT required for provider metadata tests");
            }
            return;
        };
        let root = std::path::PathBuf::from(root);
        let mut sizes = Vec::new();
        for arch in [
            crate::moe_abi::ARCH_SM90,
            crate::moe_abi::ARCH_SM100,
            crate::moe_abi::ARCH_SM120,
        ] {
            let path = root.join(format!("liboxide_moe_sm{arch}.so"));
            assert!(path.is_file(), "missing AOT library {path:?}");
            // SAFETY: trusted reviewed AOT artifact; CPU metadata only.
            let provider = unsafe { Qwen35MoeProvider::load(&path, arch) }.unwrap();
            let prepared = provider.prepare(7, 2048, 512, true).unwrap();
            let workspace = RoutedWorkspacePlan::from_prepared(&prepared).unwrap();
            assert_eq!(workspace.bytes(), prepared.workspace_bytes());
            assert_eq!(
                prepared.forward_admitted(),
                arch == crate::moe_abi::ARCH_SM90
            );
            sizes.push((arch, workspace.bytes()));
        }
        assert_ne!(
            sizes[0].1, sizes[1].1,
            "Hopper and portable workspace ABIs must not be conflated"
        );
    }

    #[test]
    fn dense_prepared_executor_api_remains_present() {
        // Compile-time proof that the donor dense executor is still intact and
        // the MoE module consumes the same attention preparation instead of
        // replacing it.
        let _dense_new = crate::qwen35_execute::PreparedWork::new;
        let _dense_enqueue = crate::qwen35_execute::PreparedWork::enqueue;
        let _attention_new = std::any::type_name::<crate::qwen35_execute::PreparedAttentionWork>();
        let _moe_layout = std::any::type_name::<Qwen35MoeAttentionLayout>();
    }

    #[test]
    fn embedding_gather_precondition_requires_one_id_per_row() {
        assert!(ensure_embedding_input(7, 7).is_ok());
        assert!(ensure_embedding_input(7, 8).is_ok());
        assert!(ensure_embedding_input(7, 6).is_err());
        assert!(ensure_embedding_input(0, 0).is_err());
    }
}
