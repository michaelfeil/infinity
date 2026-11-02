// Explicit numerical and ordering contract of the actual Native Qwen3-VL
// implementation (`qwen3_vl_models.cpp`, `helpers/dense_mlp.cpp`,
// `decoder_models.cpp`).
//
// The contract is named instead of inferred so an Oxide executor cannot
// accidentally call the tower-only path "full forward", or describe the
// multi-vector retrieval profile as the default BF16 profile.

use crate::qwen3_vl_config::{Dtype, Qwen3VlConfig, DEFAULT_VISION_ACT};
use infinity_v3_backend_core::BackendError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionProfile {
    /// Published Qwen3-VL BF16/F16 non-multi-vector path.
    Default16,
    /// ColQwen/retrieval 5-D Conv3D path with equal-length frame grouping and
    /// exact-erf merger GELU.
    MultiVector,
    /// Candle FP16 temporal split, reachable only for `qwen3_5*` checkpoints.
    ExactFp16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionActivation {
    GeluPytorchTanh,
    GeluExact,
    Silu,
}

impl VisionActivation {
    pub const fn label(self) -> &'static str {
        match self {
            Self::GeluPytorchTanh => "tanh-approx GELU (ATen gelu_pytorch_tanh)",
            Self::GeluExact => "exact-erf GELU",
            Self::Silu => "SiLU",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CornerSumOrder {
    /// `corners.sum(0)` in table dtype (default BF16 path).
    DefaultSum,
    /// `(c00 + c10) + (c01 + c11)` (exact FP16 path).
    ExactPair,
    /// `((c00 + c01) + c10) + c11` (multi-vector path).
    MultiVectorLeft,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeNorm {
    /// Final merger normalizes ungrouped patch rows before `view(-1, H*merge^2)`.
    PreGroup,
    /// Deepstack merger groups first and normalizes width `H*merge^2`.
    PostGroup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextRopeSource {
    /// Caller supplies `[3, tokens]` media positions; all three axes are used.
    MultimodalAxes,
    /// Missing media positions: Native expands a single position row to all axes.
    PlainPositions,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepstackInjection {
    /// `index_add` into the text residual after the end of text layer `i`.
    AdditiveAfterTextLayer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionContract {
    pub profile: VisionProfile,
    pub activation: VisionActivation,
    pub merger_activation: VisionActivation,
    pub corner_sum: CornerSumOrder,
    pub final_merger: MergeNorm,
    pub deepstack_merger: MergeNorm,
    pub layer_norm_epsilon: f32,
    pub patch_projection: &'static str,
    pub attention_isolation: &'static str,
    pub rotary_layout: &'static str,
    pub deepstack_capture: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextContract {
    pub rope_source: TextRopeSource,
    pub rope_application: &'static str,
    pub marker_injection: &'static str,
    pub deepstack_injection: DeepstackInjection,
    pub prefix_empty_images: &'static str,
    pub causal: bool,
    pub mrope_axes_formula: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3VlContract {
    pub vision: VisionContract,
    pub text: TextContract,
    pub bf16_stage_order: &'static [&'static str],
}

pub const VISION_STAGES: &[&str] = &[
    "pixels F32 host -> model dtype",
    "patch_embed.proj (5-D flattened linear / temporal split)",
    "pos_embed four-corner interpolation (table dtype)",
    "vision 2-D rotary phases (F32)",
    "blocks.N.norm1 (LayerNorm eps 1e-6)",
    "blocks.N.attn.qkv",
    "blocks.N q/k rotate: F32 -> model dtype",
    "blocks.N frame-isolated packed attention (non-causal)",
    "blocks.N.attn.proj",
    "blocks.N residual",
    "blocks.N.norm2",
    "blocks.N.mlp.linear_fc1 -> GELU -> linear_fc2",
    "blocks.N residual",
    "deepstack_merger_list.i capture (post-group norm) when configured",
    "merger (pre-group norm) final merged rows",
];

pub const TEXT_STAGES: &[&str] = &[
    "embed_tokens",
    "index_copy visual rows into marker spans",
    "per-layer input_layernorm",
    "q/k/v projections",
    "Qwen3 per-head q_norm/k_norm",
    "interleaved mRoPE on q/k (model dtype)",
    "packed causal GQA attention",
    "o_proj + residual",
    "post_attention_layernorm",
    "SwiGLU gate/up/down + residual",
    "deepstack index_add after matching text layer",
    "final norm",
];

fn vision_activation(profile: VisionProfile, act: &str) -> VisionActivation {
    if profile == VisionProfile::ExactFp16 && act != "silu" && act != "gelu" {
        return VisionActivation::GeluExact;
    }
    match act {
        "silu" => VisionActivation::Silu,
        "gelu" => VisionActivation::GeluExact,
        _ => VisionActivation::GeluPytorchTanh,
    }
}

fn merger_activation(profile: VisionProfile) -> VisionActivation {
    match profile {
        VisionProfile::ExactFp16 => VisionActivation::GeluExact,
        VisionProfile::MultiVector => VisionActivation::GeluExact,
        VisionProfile::Default16 => VisionActivation::GeluPytorchTanh,
    }
}

pub fn select(config: &Qwen3VlConfig, multi_vector: bool) -> Result<Qwen3VlContract, BackendError> {
    if config.vision.dtype != Dtype::Bf16 && config.vision.dtype != Dtype::F16 {
        return Err(BackendError::Start(
            "Qwen3-VL contract requires F16 or BF16 model dtype".to_owned(),
        ));
    }
    let profile = config.vision_profile(multi_vector || config.readout.multi_vector);
    let act = if config.vision.hidden_act.is_empty() {
        DEFAULT_VISION_ACT
    } else {
        &config.vision.hidden_act
    };
    Ok(Qwen3VlContract {
        vision: VisionContract {
            profile,
            activation: vision_activation(profile, act),
            merger_activation: merger_activation(profile),
            corner_sum: match profile {
                VisionProfile::Default16 => CornerSumOrder::DefaultSum,
                VisionProfile::ExactFp16 => CornerSumOrder::ExactPair,
                VisionProfile::MultiVector => CornerSumOrder::MultiVectorLeft,
            },
            final_merger: MergeNorm::PreGroup,
            deepstack_merger: MergeNorm::PostGroup,
            layer_norm_epsilon: 1e-6,
            patch_projection: match profile {
                VisionProfile::Default16 => "single BF16 GEMM [rows, patch_dim] x [H, patch_dim]^T + bias",
                VisionProfile::MultiVector => "Conv3D stride=kernel [C,T,16,16] + bias",
                VisionProfile::ExactFp16 => "two temporal-2 GEMMs, rounded sum, then separate bias",
            },
            attention_isolation: "per temporal frame: cu_seqlens advances height*width per frame; non-causal; max_frame=max(height*width)",
            rotary_layout: "split-half 2-D: angles=[row*vision_freq, col*vision_freq]; q/k promoted to F32 then cast to model dtype",
            deepstack_capture: "capture after full block residual in increasing vision layer order",
        },
        text: TextContract {
            rope_source: TextRopeSource::MultimodalAxes,
            rope_application: "interleaved mRoPE on q/k in model dtype; v untouched",
            marker_injection: "replace embedding rows at [token_start, token_start+token_count); not additive",
            deepstack_injection: DeepstackInjection::AdditiveAfterTextLayer,
            prefix_empty_images: "no tower work; empty [0, hidden] visual/deepstack placeholders while text suffix keeps mRoPE",
            causal: true,
            mrope_axes_formula: "axis[j] = 1 if j%3==1 && j<s1*3; 2 if j%3==2 && j<s2*3; else 0",
        },
        bf16_stage_order: VISION_STAGES,
    })
}
