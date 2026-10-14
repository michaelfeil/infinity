#[cfg(feature = "flash-attn")]
use crate::layers::index_select;
use crate::layers::{
    apply_rotary, get_cos_sin, get_inv_freqs, CompactUnfoldTensors, HiddenAct, Linear,
};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::{Batch, DecisionInput, DecisionOutput, ModelType, Pool};
use serde::Deserialize;
use std::collections::HashMap;

fn packed_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, width: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && x.dtype() == DType::BF16 {
        return crate::layers::gemma_rms_norm::rotary_reference(x, cos, sin);
    }
    apply_rotary(x, cos, sin, width)?
        .transpose(1, 2)?
        .squeeze(0)?
        .contiguous()
}

// Each sequence owns a contiguous suffix. FlashAttention aligns that suffix
// with the end of its full K/V sequence using the causal bottom-right mask.
fn compact_query_lengths(batch: &Batch) -> Option<(Vec<u32>, usize)> {
    let gather = batch.fold_gather.as_ref()?;
    if gather.len() >= batch.input_ids.len() {
        return None;
    }
    let mut offsets = vec![0u32];
    let mut cursor = 0;
    let mut max_length = 0;
    for range in batch.cumulative_seq_lengths.windows(2) {
        let start = cursor;
        while cursor < gather.len() && gather[cursor] < range[1] {
            cursor += 1;
        }
        let count = cursor - start;
        if count > (range[1] - range[0]) as usize
            || !gather[start..cursor]
                .iter()
                .copied()
                .eq(range[1] - count as u32..range[1])
        {
            return None;
        }
        offsets.push(cursor as u32);
        max_length = max_length.max(count);
    }
    (cursor == gather.len()).then_some((offsets, max_length))
}

fn default_head_dim() -> usize {
    256
}

fn default_global_head_dim() -> usize {
    512
}

fn default_vocab_size() -> usize {
    262_144
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4RopeParameters {
    pub full_attention: Gemma4RopeLayerParameters,
    pub sliding_attention: Gemma4RopeLayerParameters,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4RopeLayerParameters {
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default, alias = "expert_intermediate_size")]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    pub pad_token_id: u32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub rope_parameters: Gemma4RopeParameters,
    pub sliding_window: usize,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4Config {
    #[serde(default)]
    pub architectures: serde_json::Value,
    #[serde(default)]
    pub auto_map: serde_json::Value,
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(default)]
    pub id2label: HashMap<String, String>,
}

impl Gemma4Config {
    fn publisher_sequence_score(&self) -> bool {
        self.architectures
            .get(0)
            .and_then(serde_json::Value::as_str)
            == Some("Gemma4ForSequenceClassification")
            && self
                .auto_map
                .get("AutoModelForSequenceClassification")
                .and_then(serde_json::Value::as_str)
                == Some("modeling_gemma4_sequence.Gemma4ForSequenceClassification")
    }
}

#[derive(Debug)]
pub(super) struct Gemma4RmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl Gemma4RmsNorm {
    pub(super) fn load(vb: VarBuilder, hidden_size: usize, epsilon: f64) -> Result<Self> {
        Ok(Self {
            // Normalization applies the scale in FP32; convert it once at load time.
            weight: vb.get(hidden_size, "weight")?.to_dtype(DType::F32)?,
            epsilon,
        })
    }

    fn without_weight(vb: &VarBuilder, hidden_size: usize, epsilon: f64) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(hidden_size, DType::F32, vb.device())?,
            epsilon,
        })
    }

    fn forward_residual(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if hidden_states.dtype() == DType::BF16
            && hidden_states.device().is_cuda()
            && hidden_states.rank() == 2
            && hidden_states.dim(1)? <= 8192
        {
            return crate::layers::gemma_rms_norm::residual_reference(
                hidden_states,
                &self.weight,
                residual,
                self.epsilon as f32,
            );
        }
        residual + self.forward(hidden_states)?
    }

    pub(super) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        #[cfg(feature = "cuda")]
        if dtype == DType::BF16
            && hidden_states.device().is_cuda()
            && hidden_states.dim(D::Minus1)? <= 8192
        {
            // Attention carries a singleton batch dimension around packed tokens.
            let squeezed = hidden_states.rank() == 4 && hidden_states.dim(0)? == 1;
            let states = if squeezed {
                hidden_states.squeeze(0)?
            } else {
                hidden_states.clone()
            };
            let states = crate::layers::gemma_rms_norm::forward_reference(
                &states,
                &self.weight,
                self.epsilon as f32,
            )?;
            return if squeezed {
                states.unsqueeze(0)
            } else {
                Ok(states)
            };
        }
        let states = hidden_states.to_dtype(DType::F32)?;
        let variance = states.sqr()?.mean_keepdim(D::Minus1)?;
        let states = states.broadcast_div(&(variance + self.epsilon)?.sqrt()?)?;
        let states = states.broadcast_mul(&self.weight)?;
        states.to_dtype(dtype)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionType {
    Full,
    Sliding,
}

impl AttentionType {
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "full_attention" => Ok(Self::Full),
            "sliding_attention" => Ok(Self::Sliding),
            other => candle::bail!("unsupported Gemma4 attention type `{other}`"),
        }
    }
}

struct Gemma4Attention {
    qkv_proj: Linear,
    k_eq_v: bool,
    o_proj: Linear,
    q_norm: Gemma4RmsNorm,
    k_norm: Option<Gemma4RmsNorm>,
    v_norm: Option<Gemma4RmsNorm>,
    attention_type: AttentionType,
    is_kv_shared: bool,
    store_shared_kv: bool,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    sliding_window: usize,
}

impl Gemma4Attention {
    #[cfg(feature = "flash-attn")]
    #[allow(clippy::too_many_arguments)]
    fn forward_varlen(
        &self,
        states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
        max_length: usize,
        causal: bool,
        compact: &CompactUnfoldTensors,
        shared_kv: &mut SharedKv,
        image_spans: &[(usize, usize, usize)],
        compact_queries: Option<&(Tensor, usize)>,
    ) -> Result<Tensor> {
        use crate::flash_attn::flash_attn_varlen;
        let compact_len = states.dim(0)?;
        let unfold_heads = |x: Tensor, heads: usize| -> Result<Tensor> {
            // The CUDA gather requires strict row-major strides. A singleton
            // head dimension may retain an arbitrary stride even after
            // `contiguous`, so gather flattened rows and restore heads.
            let flat = x.flatten_from(1)?.contiguous()?;
            let expanded = compact.scatter_unfold(&flat)?;
            expanded.reshape((expanded.dim(0)?, heads, self.head_dim))
        };
        let qkv = self.qkv_proj.forward(states)?;
        let projected_heads = qkv.dim(1)? / self.head_dim;
        let qkv = qkv.reshape((1, compact_len, projected_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, self.num_attention_heads)?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let q = packed_rotary(&q, cos, sin, self.head_dim)?;
        let q = if compact_queries.is_some() {
            q
        } else {
            unfold_heads(q, self.num_attention_heads)?
        };

        let (k, v) = if self.is_kv_shared {
            shared_kv
                .get(self.attention_type)
                .cloned()
                .ok_or_else(|| candle::Error::Msg("missing Gemma4 shared KV states".into()))?
        } else {
            let k_unrotated = qkv.narrow(2, self.num_attention_heads, self.num_key_value_heads)?;
            let v_unrotated = if self.k_eq_v {
                k_unrotated.clone()
            } else {
                qkv.narrow(
                    2,
                    self.num_attention_heads + self.num_key_value_heads,
                    self.num_key_value_heads,
                )?
            };
            let k = self
                .k_norm
                .as_ref()
                .unwrap()
                .forward(&k_unrotated)?
                .transpose(1, 2)?;
            let k = packed_rotary(&k, cos, sin, self.head_dim)?;
            let v = self
                .v_norm
                .as_ref()
                .unwrap()
                .forward(&v_unrotated)?
                .squeeze(0)?
                .contiguous()?;
            let k = unfold_heads(k, self.num_key_value_heads)?;
            let v = unfold_heads(v, self.num_key_value_heads)?;
            if self.store_shared_kv {
                shared_kv.set(self.attention_type, k.clone(), v.clone());
            }
            (k, v)
        };
        let (window_left, window_right) = if self.attention_type == AttentionType::Sliding {
            if causal {
                (Some(self.sliding_window.saturating_sub(1)), Some(0))
            } else {
                let half = self.sliding_window / 2;
                (Some(half), Some(half))
            }
        } else {
            (None, None)
        };
        let (cu_queries, max_queries) = compact_queries
            .map(|(cu, len)| (cu, *len))
            .unwrap_or((cu_seqlens, max_length));
        let mut output = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_queries,
            cu_seqlens,
            max_queries,
            max_length,
            1.0,
            causal,
            window_left,
            window_right,
        )?
        .flatten_from(D::Minus2)?;
        // Local layers use AND(sliding-window, OR(causal, same-image)).
        // For image queries, a noncausal call ending at that image's last key
        // adds precisely the within-image future keys; global layers stay causal.
        if self.attention_type == AttentionType::Sliding {
            for &(start, length, sequence_start) in image_spans {
                let key_start = start
                    .saturating_sub(self.sliding_window.saturating_sub(1))
                    .max(sequence_start);
                let key_length = start + length - key_start;
                let cu_q = Tensor::new(&[0u32, length as u32], states.device())?;
                let cu_k = Tensor::new(&[0u32, key_length as u32], states.device())?;
                let image_output = flash_attn_varlen(
                    &q.narrow(0, start, length)?.contiguous()?,
                    &k.narrow(0, key_start, key_length)?.contiguous()?,
                    &v.narrow(0, key_start, key_length)?.contiguous()?,
                    None,
                    &cu_q,
                    &cu_k,
                    length,
                    key_length,
                    1.0,
                    false,
                    Some(self.sliding_window.saturating_sub(1)),
                    None,
                )?
                .flatten_from(D::Minus2)?;
                output = output.slice_scatter0(&image_output, start)?;
            }
        }
        let output = if compact_queries.is_some() {
            output
        } else {
            compact.fold_gather(&output)?
        };
        self.o_proj.forward(&output)
    }

    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        let attention_type = AttentionType::from_name(&config.layer_types[layer_idx])?;
        let is_sliding = attention_type == AttentionType::Sliding;
        let head_dim = if is_sliding {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let use_alternative_attention = config.attention_k_eq_v && !is_sliding;
        let num_key_value_heads = if use_alternative_attention {
            config
                .num_global_key_value_heads
                .unwrap_or(config.num_key_value_heads)
        } else {
            config.num_key_value_heads
        };
        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let is_kv_shared = config.num_kv_shared_layers > 0 && layer_idx >= first_shared;
        let store_shared_kv = !is_kv_shared
            && config.num_kv_shared_layers > 0
            && config.layer_types[..first_shared]
                .iter()
                .rposition(|kind| kind == &config.layer_types[layer_idx])
                == Some(layer_idx);

        let load_linear = |vb: VarBuilder, output: usize, input: usize| -> Result<Linear> {
            let weight = vb.get((output, input), "weight")?;
            let bias = if config.attention_bias {
                Some(vb.get(output, "bias")?)
            } else {
                None
            };
            Ok(Linear::new(weight, bias, None))
        };

        // Shared-KV layers project Q only; K=V layers project Q and K once.
        let mut projections = vec![("q_proj", config.num_attention_heads * head_dim)];
        if !is_kv_shared {
            projections.push(("k_proj", num_key_value_heads * head_dim));
            if !use_alternative_attention {
                projections.push(("v_proj", num_key_value_heads * head_dim));
            }
        }
        let mut weights = Vec::with_capacity(projections.len());
        let mut biases = Vec::with_capacity(projections.len());
        for (name, width) in projections {
            weights.push(vb.pp(name).get((width, config.hidden_size), "weight")?);
            if config.attention_bias {
                biases.push(vb.pp(name).get(width, "bias")?);
            }
        }
        let bias = if config.attention_bias {
            Some(Tensor::cat(&biases, 0)?)
        } else {
            None
        };
        let qkv_proj = Linear::new(Tensor::cat(&weights, 0)?, bias, None);
        let (k_norm, v_norm) = if is_kv_shared {
            (None, None)
        } else {
            (
                Some(Gemma4RmsNorm::load(
                    vb.pp("k_norm"),
                    head_dim,
                    config.rms_norm_eps,
                )?),
                Some(Gemma4RmsNorm::without_weight(
                    &vb,
                    head_dim,
                    config.rms_norm_eps,
                )?),
            )
        };

        Ok(Self {
            qkv_proj,
            k_eq_v: use_alternative_attention,
            o_proj: load_linear(
                vb.pp("o_proj"),
                config.hidden_size,
                config.num_attention_heads * head_dim,
            )?,
            q_norm: Gemma4RmsNorm::load(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?,
            k_norm,
            v_norm,
            attention_type,
            is_kv_shared,
            store_shared_kv,
            head_dim,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads,
            sliding_window: config.sliding_window,
        })
    }
}

#[derive(Default)]
pub(super) struct SharedKv {
    full: Option<(Tensor, Tensor)>,
    sliding: Option<(Tensor, Tensor)>,
}

impl SharedKv {
    fn get(&self, kind: AttentionType) -> Option<&(Tensor, Tensor)> {
        match kind {
            AttentionType::Full => self.full.as_ref(),
            AttentionType::Sliding => self.sliding.as_ref(),
        }
    }

    fn set(&mut self, kind: AttentionType, k: Tensor, v: Tensor) {
        match kind {
            AttentionType::Full => self.full = Some((k, v)),
            AttentionType::Sliding => self.sliding = Some((k, v)),
        }
    }
}

struct Gemma4Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    activation: HiddenAct,
}

impl Gemma4Mlp {
    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        let first_shared = config.num_hidden_layers - config.num_kv_shared_layers;
        let double_wide = config.use_double_wide_mlp
            && config.num_kv_shared_layers > 0
            && layer_idx >= first_shared;
        let intermediate_size = config.intermediate_size * if double_wide { 2 } else { 1 };
        let gate = vb
            .pp("gate_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;
        let up = vb
            .pp("up_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;
        let down = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        Ok(Self {
            gate_up_proj: Linear::new(Tensor::cat(&[&gate, &up], 0)?, None, None),
            down_proj: Linear::new(down, None, None),
            activation: config.hidden_activation.clone(),
        })
    }

    fn forward(&self, states: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(states)?;
        let gated = crate::layers::gated_activation(&gate_up, Some(&self.activation))?;
        self.down_proj.forward(&gated)
    }
}

struct Gemma4PleLayer {
    input_gate: Linear,
    projection: Linear,
    norm: Gemma4RmsNorm,
    activation: HiddenAct,
}

// Non-SM80 builds keep the model type but reject MoE loading before allocating
// these tensors. Its forward implementation is only available with the kernels.
#[cfg_attr(not(gemma4_moe_cuda), allow(dead_code))]
struct Gemma4Moe {
    router_scale: Tensor,
    router_weight: Tensor,
    expert_scale: Tensor,
    gate_up: Tensor,
    down: Tensor,
    dense_norm: Gemma4RmsNorm,
    expert_input_norm: Gemma4RmsNorm,
    expert_output_norm: Gemma4RmsNorm,
}

impl Gemma4Moe {
    fn load(vb: VarBuilder, config: &Gemma4TextConfig) -> Result<Option<Self>> {
        if !config.enable_moe_block {
            return Ok(None);
        }
        if !cfg!(gemma4_moe_cuda)
            || !vb.device().is_cuda()
            || vb.dtype() != DType::BF16
            || config.hidden_size != 2816
            || config.num_experts != Some(128)
            || config.top_k_experts != Some(8)
            || config.moe_intermediate_size != Some(704)
        {
            candle::bail!("Gemma4 MoE requires BF16, an SM80+ CUDA build and the 26B-A4B 128-expert/8-route configuration");
        }
        let h = config.hidden_size;
        let norm = |name: &str| Gemma4RmsNorm::load(vb.pp(name), h, config.rms_norm_eps);
        Ok(Some(Self {
            router_scale: vb.get(h, "router.scale")?,
            router_weight: vb
                .get((128, h), "router.proj.weight")?
                .to_dtype(DType::F32)?,
            expert_scale: vb
                .get(128, "router.per_expert_scale")?
                .to_dtype(DType::F32)?,
            gate_up: vb.get((128, 1408, h), "experts.gate_up_proj")?,
            down: vb.get((128, h, 704), "experts.down_proj")?,
            dense_norm: norm("post_feedforward_layernorm_1")?,
            expert_input_norm: norm("pre_feedforward_layernorm_2")?,
            expert_output_norm: norm("post_feedforward_layernorm_2")?,
        }))
    }
    fn forward(
        &self,
        residual: &Tensor,
        dense: &Tensor,
        compact: Option<&CompactUnfoldTensors>,
    ) -> Result<Tensor> {
        #[cfg(gemma4_moe_cuda)]
        {
            let shape = residual.shape();
            let hidden = residual.dim(D::Minus1)?;
            let residual = residual.reshape((residual.elem_count() / hidden, hidden))?;
            // Both branches normalize this residual with the same epsilon.
            // Keep the router's intermediate BF16 rounding before each scale.
            let (input, routing) = crate::layers::gemma_rms_norm::paired_moe(
                &residual,
                &self.expert_input_norm.weight,
                &self.router_scale,
                self.expert_input_norm.epsilon as f32,
            )?;
            // cuBLAS changes FP32 reduction order with the row count. Tiny router
            // differences propagate through MoE layers into different decisions.
            // Keep this small projection at the logical batch shape; the dense
            // and expert MLPs still operate on shared compact rows.
            let routing = match compact {
                Some(c) => c.scatter_unfold(&routing)?,
                None => routing,
            };
            let logits = routing
                .to_dtype(DType::F32)?
                .matmul(&self.router_weight.t()?)?;
            let logits = match compact {
                Some(c) => c.fold_gather(&logits)?,
                None => logits,
            };
            let expert = crate::layers::gemma4_moe::experts(
                &input,
                &logits,
                &self.expert_scale,
                &self.gate_up,
                &self.down,
            )?;
            self.dense_norm.forward(dense)?
                + self.expert_output_norm.forward(&expert)?.reshape(shape)?
        }
        #[cfg(not(gemma4_moe_cuda))]
        {
            let _ = (residual, dense, compact);
            candle::bail!("Gemma4 MoE requires an SM80+ CUDA build")
        }
    }
}

pub(super) struct Gemma4Layer {
    attention: Gemma4Attention,
    mlp: Gemma4Mlp,
    moe: Option<Gemma4Moe>,
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
    ple: Option<Gemma4PleLayer>,
    layer_scalar: Tensor,
}

impl Gemma4Layer {
    #[cfg(feature = "flash-attn")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_varlen(
        &self,
        states: &Tensor,
        per_layer_input: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
        max_length: usize,
        causal: bool,
        compact: &CompactUnfoldTensors,
        shared_kv: &mut SharedKv,
        image_spans: &[(usize, usize, usize)],
        compact_queries: Option<&(Tensor, usize)>,
    ) -> Result<Tensor> {
        let residual = states;
        let normalized = self.input_layernorm.forward(states)?;
        let attention = self.attention.forward_varlen(
            &normalized,
            cos,
            sin,
            cu_seqlens,
            max_length,
            causal,
            compact,
            shared_kv,
            image_spans,
            compact_queries,
        )?;
        let states = self
            .post_attention_layernorm
            .forward_residual(&attention, residual)?;
        let residual = &states;
        let normalized = self.pre_feedforward_layernorm.forward(&states)?;
        let mlp = self.mlp.forward(&normalized)?;
        let mlp = match &self.moe {
            Some(moe) => moe.forward(residual, &mlp, Some(compact))?,
            None => mlp,
        };
        let mut states = self
            .post_feedforward_layernorm
            .forward_residual(&mlp, residual)?;
        if let (Some(ple), Some(per_layer_input)) = (&self.ple, per_layer_input) {
            let contribution = ple.input_gate.forward(&states)?;
            let contribution = ple.activation.forward(&contribution)?;
            let contribution = (contribution * per_layer_input)?;
            let contribution = ple.projection.forward(&contribution)?;
            let contribution = ple.norm.forward(&contribution)?;
            states = (states + contribution)?;
        }
        states.broadcast_mul(&self.layer_scalar)
    }

    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        Self::load_with_ple(vb.clone(), vb, config, layer_idx)
    }

    pub(super) fn load_with_ple(
        vb: VarBuilder,
        ple_vb: VarBuilder,
        config: &Gemma4TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let norm =
            |name: &str| Gemma4RmsNorm::load(vb.pp(name), config.hidden_size, config.rms_norm_eps);
        let ple = if config.hidden_size_per_layer_input > 0 {
            Some(Gemma4PleLayer {
                input_gate: Linear::new(
                    ple_vb.pp("per_layer_input_gate").get(
                        (config.hidden_size_per_layer_input, config.hidden_size),
                        "weight",
                    )?,
                    None,
                    None,
                ),
                projection: Linear::new(
                    ple_vb.pp("per_layer_projection").get(
                        (config.hidden_size, config.hidden_size_per_layer_input),
                        "weight",
                    )?,
                    None,
                    None,
                ),
                norm: Gemma4RmsNorm::load(
                    ple_vb.pp("post_per_layer_input_norm"),
                    config.hidden_size,
                    config.rms_norm_eps,
                )?,
                activation: config.hidden_activation.clone(),
            })
        } else {
            None
        };
        Ok(Self {
            attention: Gemma4Attention::load(vb.pp("self_attn"), config, layer_idx)?,
            mlp: Gemma4Mlp::load(vb.pp("mlp"), config, layer_idx)?,
            moe: Gemma4Moe::load(vb.clone(), config)?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            ple,
            layer_scalar: vb.get(1, "layer_scalar")?,
        })
    }
}

struct Gemma4Ple {
    embeddings: Embedding,
    model_projection: Linear,
    projection_norm: Gemma4RmsNorm,
    num_layers: usize,
    hidden_size: usize,
    scale: f64,
    projection_scale: f64,
}

impl Gemma4Ple {
    fn load(vb: VarBuilder, config: &Gemma4TextConfig) -> Result<Option<Self>> {
        let hidden_size = config.hidden_size_per_layer_input;
        if hidden_size == 0 {
            return Ok(None);
        }
        let packed_size = config.num_hidden_layers * hidden_size;
        Ok(Some(Self {
            embeddings: Embedding::new(
                vb.get(
                    (config.vocab_size_per_layer_input, packed_size),
                    "embed_tokens_per_layer.weight",
                )?,
                packed_size,
            ),
            model_projection: Linear::new(
                vb.get(
                    (packed_size, config.hidden_size),
                    "per_layer_model_projection.weight",
                )?,
                None,
                None,
            ),
            projection_norm: Gemma4RmsNorm::load(
                vb.pp("per_layer_projection_norm"),
                hidden_size,
                config.rms_norm_eps,
            )?,
            num_layers: config.num_hidden_layers,
            hidden_size,
            scale: (hidden_size as f64).sqrt(),
            projection_scale: (config.hidden_size as f64).sqrt().recip(),
        }))
    }

    fn forward(&self, input_ids: &Tensor, input_embeddings: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let token_inputs = (self.embeddings.forward(input_ids)? * self.scale)?.reshape((
            batch,
            seq_len,
            self.num_layers,
            self.hidden_size,
        ))?;
        let projected = (self.model_projection.forward(input_embeddings)? * self.projection_scale)?
            .reshape((batch, seq_len, self.num_layers, self.hidden_size))?;
        let projected = self.projection_norm.forward(&projected)?;
        (token_inputs + projected)? * std::f64::consts::FRAC_1_SQRT_2
    }
}

enum Gemma4Output {
    Decision(Tensor),
    Embedding(Pool),
    Classifier(Linear),
}

pub struct Gemma4Model {
    #[cfg(feature = "flash-attn")]
    vision: Option<super::gemma4_vision::Gemma4Vision>,
    embeddings: Embedding,
    embedding_scale: f64,
    ple: Option<Gemma4Ple>,
    layers: Vec<Gemma4Layer>,
    norm: Gemma4RmsNorm,
    output: Gemma4Output,
    final_logit_softcapping: Option<f64>,
    publisher_sequence_score: bool,
    local_rope: (Tensor, Tensor),
    full_rope: (Tensor, Tensor),
    device: Device,
}

impl Gemma4Model {
    #[cfg(feature = "flash-attn")]
    fn forward_hidden_varlen(
        &self,
        batch: &Batch,
        causal: bool,
    ) -> Result<(Tensor, CompactUnfoldTensors)> {
        if !causal && batch.compact_input_ids.is_some() {
            candle::bail!("Bidirectional Gemma4 inference cannot fold causal prefixes")
        }
        let (input_ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut embeddings = (self.embeddings.forward(&input_ids)? * self.embedding_scale)?;
        let has_images = batch
            .multimodal
            .iter()
            .flatten()
            .any(|media| !media.images.is_empty());
        if has_images
            && batch.compact_input_ids.is_some()
            && !batch
                .scatter_unfold
                .as_ref()
                .zip(batch.fold_gather.as_ref())
                .is_some_and(|(scatter, fold)| {
                    infinity_v3_backend_core::MultimodalEncoding::allows_radix_fold(
                        &batch.multimodal,
                        &batch.input_ids,
                        &batch.cumulative_seq_lengths,
                        scatter,
                        fold,
                    )
                })
        {
            candle::bail!("Radix image folding requires identical images and image prefixes");
        }
        // Compute PLE from text embeddings before replacing vision slots.

        let per_layer_inputs = match &self.ple {
            Some(ple) => Some(
                ple.forward(&input_ids.unsqueeze(0)?, &embeddings.unsqueeze(0)?)?
                    .squeeze(0)?,
            ),
            None => None,
        };
        let image_spans = if has_images {
            let mut full_embeddings = if batch.compact_input_ids.is_some() {
                let ids = Tensor::new(batch.input_ids.as_slice(), &self.device)?;
                (self.embeddings.forward(&ids)? * self.embedding_scale)?
            } else {
                embeddings.clone()
            };
            let spans = self
                .vision
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("Gemma4 vision weights unavailable".into()))?
                .inject(batch, &mut full_embeddings)?;
            embeddings = compact.fold_gather(&full_embeddings)?;
            spans
        } else {
            vec![]
        };
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch.len() + 1,
            &self.device,
        )?;
        #[cfg(feature = "fa4")]
        let _fa4_batch =
            crate::fa4_native::prepare_batch(&cu_seqlens, &batch.cumulative_seq_lengths)?;
        let compact_queries = if causal && !has_images {
            compact_query_lengths(batch)
                .map(|(cu, len)| Ok::<_, candle::Error>((Tensor::new(cu, &self.device)?, len)))
                .transpose()?
        } else {
            None
        };
        let positions = &compact.position_ids_compact;
        let rope = |cache: &(Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
            Ok((
                index_select(&cache.0, positions, 0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
                index_select(&cache.1, positions, 0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
            ))
        };
        let local = rope(&self.local_rope)?;
        let full = rope(&self.full_rope)?;
        let mut states = embeddings;
        let mut shared_kv = SharedKv::default();
        for (idx, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = match layer.attention.attention_type {
                AttentionType::Full => (&full.0, &full.1),
                AttentionType::Sliding => (&local.0, &local.1),
            };
            let per_layer = match &per_layer_inputs {
                Some(inputs) => Some(inputs.i((.., idx, ..))?),
                None => None,
            };
            states = layer.forward_varlen(
                &states,
                per_layer.as_ref(),
                cos,
                sin,
                &cu_seqlens,
                batch.max_length as usize,
                causal,
                &compact,
                &mut shared_kv,
                &image_spans,
                compact_queries.as_ref(),
            )?;
        }
        Ok((self.norm.forward(&states)?, compact))
    }

    pub fn load(vb: VarBuilder, config: &Gemma4Config, model_type: ModelType) -> Result<Self> {
        if !vb.device().is_cuda() || vb.dtype() != DType::BF16 || !cfg!(feature = "flash-attn") {
            candle::bail!("Gemma4 requires CUDA BF16 with packed FlashAttention v2");
        }
        let text = &config.text_config;
        if text.layer_types.len() != text.num_hidden_layers {
            candle::bail!(
                "Gemma4 layer_types has {} entries, expected {}",
                text.layer_types.len(),
                text.num_hidden_layers
            )
        }

        #[cfg(feature = "flash-attn")]
        let vision = if model_type == ModelType::Decision
            && text.use_bidirectional_attention.as_deref() == Some("vision")
            && text.hidden_size_per_layer_input == 0
        {
            config
                .vision_config
                .as_ref()
                .map(|vision| {
                    super::gemma4_vision::Gemma4Vision::load(vb.clone(), vision, text.hidden_size)
                })
                .transpose()?
        } else {
            None
        };
        let output_vb = vb.clone();

        let vb = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb
        };
        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((text.vocab_size, text.hidden_size), "weight")?,
            text.hidden_size,
        );
        let score = match model_type {
            ModelType::Decision => {
                Gemma4Output::Decision(if config.tie_word_embeddings || text.tie_word_embeddings {
                    embeddings.embeddings().clone()
                } else {
                    output_vb
                        .pp("lm_head")
                        .get((text.vocab_size, text.hidden_size), "weight")?
                })
            }
            ModelType::Embedding(pool) => Gemma4Output::Embedding(pool),
            ModelType::Classifier => {
                // Match the router and HF defaults when label metadata is omitted.
                let num_labels = if config.id2label.is_empty() {
                    config.num_labels.unwrap_or(2)
                } else {
                    config.id2label.len()
                };
                if num_labels == 0 {
                    candle::bail!("Gemma4 classifier config does not define any labels")
                }
                Gemma4Output::Classifier(Linear::new(
                    output_vb
                        .pp("score")
                        .get((num_labels, text.hidden_size), "weight")?,
                    None,
                    None,
                ))
            }
        };

        let ple = Gemma4Ple::load(vb.clone(), text)?;
        let layers = (0..text.num_hidden_layers)
            .map(|idx| Gemma4Layer::load(vb.pp(format!("layers.{idx}")), text, idx))
            .collect::<Result<Vec<_>>>()?;
        let norm = Gemma4RmsNorm::load(vb.pp("norm"), text.hidden_size, text.rms_norm_eps)?;

        let local_inv = get_inv_freqs(
            text.head_dim,
            text.rope_parameters.sliding_attention.rope_theta,
            vb.device(),
            None,
        )?;
        let local_rope = get_cos_sin(text.max_position_embeddings, &local_inv, vb.dtype(), true)?;
        let factor = text
            .rope_parameters
            .full_attention
            .partial_rotary_factor
            .unwrap_or(1.0);
        let rotary_pairs = (factor * text.global_head_dim as f32 / 2.0) as usize;
        let full_inv: Vec<f32> = (0..text.global_head_dim / 2)
            .map(|i| {
                if i < rotary_pairs {
                    1.0 / text
                        .rope_parameters
                        .full_attention
                        .rope_theta
                        .powf((2 * i) as f32 / text.global_head_dim as f32)
                } else {
                    0.0
                }
            })
            .collect();
        let full_inv = Tensor::from_vec(full_inv, (1, text.global_head_dim / 2), vb.device())?;
        let full_rope = get_cos_sin(text.max_position_embeddings, &full_inv, vb.dtype(), true)?;

        Ok(Self {
            #[cfg(feature = "flash-attn")]
            vision,
            embeddings,
            embedding_scale: (text.hidden_size as f64).sqrt(),
            ple,
            layers,
            norm,
            output: score,
            final_logit_softcapping: text.final_logit_softcapping,
            publisher_sequence_score: config.publisher_sequence_score(),
            local_rope,
            full_rope,
            device: vb.device().clone(),
        })
    }

    #[cfg(feature = "flash-attn")]
    fn embed_batch_varlen(
        &self,
        batch: Batch,
        pool: Pool,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (states, compact) = self.forward_hidden_varlen(&batch, false)?;
        let outputs = compact.scatter_unfold(&states)?;
        let pooled = if batch.pooled_indices.is_empty() {
            None
        } else {
            let values = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let start = batch.cumulative_seq_lengths[i as usize] as usize;
                    let end = batch.cumulative_seq_lengths[i as usize + 1] as usize;
                    match pool {
                        Pool::Cls => outputs.i(start)?.unsqueeze(0),
                        Pool::LastToken => outputs.i(end - 1)?.unsqueeze(0),
                        Pool::Mean => {
                            outputs.narrow(0, start, end - start)?.sum_keepdim(0)?
                                / (end - start) as f64
                        }
                        Pool::Splade => candle::bail!("Splade pooling is not supported for Gemma4"),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        let raw = if batch.raw_indices.is_empty() {
            None
        } else {
            let values = batch
                .raw_indices
                .iter()
                .map(|&i| {
                    let start = batch.cumulative_seq_lengths[i as usize] as usize;
                    let end = batch.cumulative_seq_lengths[i as usize + 1] as usize;
                    outputs.narrow(0, start, end - start)
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        Ok((pooled, raw))
    }

    fn embed_batch(&self, batch: Batch, pool: Pool) -> Result<(Option<Tensor>, Option<Tensor>)> {
        #[cfg(feature = "flash-attn")]
        {
            self.embed_batch_varlen(batch, pool)
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            let _ = (batch, pool);
            candle::bail!("Gemma4 requires packed FlashAttention v2")
        }
    }

    fn predict_batch(&self, batch: Batch, score: &Linear) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            let (states, compact) = self.forward_hidden_varlen(&batch, true)?;
            let states = compact.scatter_unfold(&states)?;
            let indices: Vec<u32> = batch
                .cumulative_seq_lengths
                .windows(2)
                .map(|bounds| bounds[1] - 1)
                .collect();
            let indices = Tensor::from_vec(indices, batch.len(), &self.device)?;
            if self.publisher_sequence_score {
                // The declared custom sequence model scores every logical token
                // before BF16 softcapping and selection. Selecting first changes
                // GEMM rounding, including for an unfolded radix batch.
                let mut logits = score.forward(&states)?;
                if let Some(cap) = self.final_logit_softcapping {
                    // Torch scalar operations use FP32 opmath for BF16/F16.
                    // Candle affine otherwise rounds 1/cap before division.
                    // Restore the score dtype after every publisher operation.
                    let dtype = logits.dtype();
                    let opmath = match dtype {
                        DType::BF16 | DType::F16 => DType::F32,
                        _ => dtype,
                    };
                    logits = (logits.to_dtype(opmath)? / cap)?.to_dtype(dtype)?;
                    logits = logits.tanh()?;
                    logits = (logits.to_dtype(opmath)? * cap)?.to_dtype(dtype)?;
                }
                index_select(&logits, &indices, 0)
            } else {
                // An LM softcap alone does not specify a sequence score contract.
                score.forward(&index_select(&states, &indices, 0)?)
            }
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            let _ = (batch, score);
            candle::bail!("Gemma4 requires packed FlashAttention v2")
        }
    }
}

impl Model for Gemma4Model {
    #[cfg(feature = "flash-attn")]
    fn decide(&self, batch: Batch, inputs: Vec<DecisionInput>) -> Result<Vec<DecisionOutput>> {
        let Gemma4Output::Decision(weights) = &self.output else {
            candle::bail!("Gemma4 was not loaded for decisions")
        };
        if inputs.len() != batch.len() {
            candle::bail!("Decision metadata count does not match batch")
        }
        let vocabulary = weights.dim(0)?;
        let tokens = inputs
            .into_iter()
            .map(|input| match input {
                DecisionInput::OptionTokens { token_ids }
                    if !token_ids.is_empty()
                        && token_ids.iter().all(|&id| (id as usize) < vocabulary) =>
                {
                    Ok(token_ids)
                }
                DecisionInput::Warmup => Ok(vec![0]),
                _ => candle::bail!("Gemma4 requires valid option-token metadata"),
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "flash-attn")]
        let last = {
            let (states, compact) = self.forward_hidden_varlen(&batch, true)?;
            let states = compact.scatter_unfold(&states)?;
            let indices = Tensor::from_vec(
                batch
                    .cumulative_seq_lengths
                    .iter()
                    .skip(1)
                    .map(|&end| end - 1)
                    .collect::<Vec<_>>(),
                batch.len(),
                &self.device,
            )?;
            index_select(&states, &indices, 0)?
        };

        let lengths: Vec<usize> = tokens.iter().map(Vec::len).collect();
        let scores = tokens
            .into_iter()
            .enumerate()
            .map(|(i, ids)| {
                let ids = Tensor::new(ids.as_slice(), &self.device)?;
                let selected = weights.index_select(&ids, 0)?;
                let mut logits = last.i(i)?.unsqueeze(0)?.matmul(&selected.t()?)?;
                if let Some(cap) = self.final_logit_softcapping {
                    logits = ((logits / cap)?.tanh()? * cap)?;
                }
                Ok(logits)
            })
            .collect::<Result<Vec<_>>>()?;
        if scores.is_empty() {
            return Ok(Vec::new());
        }
        if scores.len() == 1 {
            return Ok(vec![DecisionOutput {
                memory: None,
                logits: scores[0].to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
                action_probability: 1.0,
            }]);
        }
        // Keep each GEMM's shape and arithmetic, then read all scores once.
        let scores = Tensor::cat(&scores, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut offset = 0;
        Ok(lengths
            .into_iter()
            .map(|len| {
                let logits = scores[offset..offset + len].to_vec();
                offset += len;
                DecisionOutput {
                    memory: None,
                    logits,
                    action_probability: 1.0,
                }
            })
            .collect())
    }

    fn supports_radix_mlp(&self) -> bool {
        self.device.is_cuda() && cfg!(feature = "flash-attn")
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        match &self.output {
            Gemma4Output::Embedding(pool) => self.embed_batch(batch, pool.clone()),
            Gemma4Output::Classifier(_) | Gemma4Output::Decision(_) => {
                candle::bail!("`embed` is not available for a Gemma4 classifier")
            }
        }
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.output {
            Gemma4Output::Classifier(score) => self.predict_batch(batch, score),
            Gemma4Output::Embedding(_) | Gemma4Output::Decision(_) => {
                candle::bail!("`predict` is not available for a Gemma4 embedding model")
            }
        }
    }
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    #[test]
    fn custom_sequence_score_contract_requires_declared_publisher() -> anyhow::Result<()> {
        let mut value = serde_json::json!({
            "text_config": {
                "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 12,
                "intermediate_size": 24, "layer_types": ["full_attention"],
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "max_position_embeddings": 32,
                "pad_token_id": 0, "sliding_window": 8,
                "final_logit_softcapping": 30.,
                "rope_parameters": {"full_attention": {"rope_theta": 10000.0},
                                    "sliding_attention": {"rope_theta": 10000.0}}
            }
        });
        let config: Gemma4Config = serde_json::from_value(value.clone())?;
        assert!(!config.publisher_sequence_score());
        value["architectures"] = serde_json::json!(["Gemma4ForSequenceClassification"]);
        let config: Gemma4Config = serde_json::from_value(value.clone())?;
        assert!(!config.publisher_sequence_score());
        value["auto_map"] = serde_json::json!({"AutoModelForSequenceClassification":
            "modeling_gemma4_sequence.Gemma4ForSequenceClassification"});
        let config: Gemma4Config = serde_json::from_value(value.clone())?;
        assert!(config.publisher_sequence_score());
        value["architectures"] = serde_json::Value::Null;
        let config: Gemma4Config = serde_json::from_value(value.clone())?;
        assert!(!config.publisher_sequence_score());
        value["architectures"] = serde_json::json!(["Gemma4ForSequenceClassification"]);
        value["auto_map"]["AutoModelForSequenceClassification"] =
            "other.Gemma4ForSequenceClassification".into();
        let config: Gemma4Config = serde_json::from_value(value)?;
        assert!(!config.publisher_sequence_score());
        Ok(())
    }

    #[test]
    fn compact_queries_require_contiguous_owned_suffixes() {
        let mut batch = Batch {
            input_ids: vec![0; 12],
            cumulative_seq_lengths: vec![0, 4, 9, 12],
            fold_gather: Some(vec![0, 1, 2, 3, 7, 8]),
            multimodal: vec![],
            token_type_ids: vec![],
            position_ids: vec![],
            max_length: 5,
            pooled_indices: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            tokens: vec![],
            offsets: vec![],
        };
        assert_eq!(compact_query_lengths(&batch), Some((vec![0, 4, 6, 6], 4)));
        batch.fold_gather = Some(vec![0, 1, 2, 3, 6, 8]);
        assert!(compact_query_lengths(&batch).is_none());
        batch.fold_gather = Some(vec![0, 1, 2, 3, 7, 8, 0]);
        assert!(compact_query_lengths(&batch).is_none());
        batch.fold_gather = Some((0..12).collect());
        assert!(compact_query_lengths(&batch).is_none());
    }

    #[test]
    fn fused_projection_preserves_kv_variants() -> anyhow::Result<()> {
        for bias in [false, true] {
            for (kind, shared, k_eq_v) in [
                ("sliding_attention", false, false),
                ("full_attention", false, false),
                ("full_attention", false, true),
                ("full_attention", true, true),
            ] {
                let config: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
                    "attention_bias": bias, "attention_k_eq_v": k_eq_v,
                    "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 12,
                    "head_dim": 4, "global_head_dim": 8, "intermediate_size": 24,
                    "layer_types": [kind, kind], "num_hidden_layers": 2,
                    "num_kv_shared_layers": if shared { 1 } else { 0 },
                    "num_attention_heads": 4, "num_key_value_heads": 2,
                    "num_global_key_value_heads": 1, "max_position_embeddings": 32,
                    "pad_token_id": 0, "sliding_window": 8,
                    "rope_parameters": {"full_attention": {"rope_theta": 10000.0},
                                        "sliding_attention": {"rope_theta": 10000.0}}
                }))?;
                let vars = candle_nn::VarMap::new();
                let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
                Gemma4Attention::load(vb.clone(), &config, 1)?;
                for (name, var) in vars.data().lock().unwrap().iter() {
                    let offset = name.bytes().map(usize::from).sum::<usize>();
                    let data = (0..var.elem_count())
                        .map(|i| ((i + offset) as f32).sin() * 0.1)
                        .collect::<Vec<_>>();
                    var.set(&Tensor::from_vec(data, var.shape(), &Device::Cpu)?)?;
                }
                let model = Gemma4Attention::load(vb.clone(), &config, 1)?;
                assert_eq!(model.is_kv_shared, shared);
                assert_eq!(vb.contains_tensor("k_proj.weight"), !shared);
                assert_eq!(vb.contains_tensor("v_proj.weight"), !shared && !k_eq_v);
                let input = Tensor::arange(0f32, 36f32, &Device::Cpu)?.reshape((3, 12))?;
                let mut separate = Vec::new();
                for name in ["q_proj", "k_proj", "v_proj"] {
                    if !vb.contains_tensor(&format!("{name}.weight")) {
                        continue;
                    }
                    let heads = if name == "q_proj" {
                        model.num_attention_heads
                    } else {
                        model.num_key_value_heads
                    };
                    let width = heads * model.head_dim;
                    let weight = vb.pp(name).get((width, 12), "weight")?;
                    let bias = if bias {
                        Some(vb.pp(name).get(width, "bias")?)
                    } else {
                        None
                    };
                    separate.push(Linear::new(weight, bias, None).forward(&input)?);
                }
                let error = (model.qkv_proj.forward(&input)? - Tensor::cat(&separate, 1)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(error < 1e-5, "Gemma4 projection changed by {error}");
            }
        }
        Ok(())
    }

    #[test]
    fn dense_and_moe_config_fields() -> anyhow::Result<()> {
        let mut value = serde_json::json!({
            "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 1536,
            "intermediate_size": 6144, "layer_types": ["sliding_attention"],
            "max_position_embeddings": 131072, "num_attention_heads": 8,
            "num_hidden_layers": 1, "num_key_value_heads": 2, "pad_token_id": 0,
            "sliding_window": 512,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        });
        for missing in [true, false] {
            if !missing {
                for name in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
                    value[name] = serde_json::Value::Null;
                }
            }
            let config: Gemma4TextConfig = serde_json::from_value(value.clone())?;
            assert!(!config.enable_moe_block);
            assert_eq!(config.num_experts, None);
            assert_eq!(config.top_k_experts, None);
            assert_eq!(config.moe_intermediate_size, None);
            assert!(
                Gemma4Moe::load(VarBuilder::zeros(DType::F32, &Device::Cpu), &config)?.is_none()
            );
        }
        value["enable_moe_block"] = true.into();
        value["num_experts"] = 128.into();
        value["top_k_experts"] = 8.into();
        value["moe_intermediate_size"] = 704.into();
        let config: Gemma4TextConfig = serde_json::from_value(value)?;
        assert!(config.enable_moe_block);
        assert_eq!(config.num_experts, Some(128));
        assert_eq!(config.top_k_experts, Some(8));
        assert_eq!(config.moe_intermediate_size, Some(704));
        let error = Gemma4Moe::load(VarBuilder::zeros(DType::BF16, &Device::Cpu), &config)
            .err()
            .expect("CPU must reject MoE before loading expert weights");
        assert!(error.to_string().contains("SM80+ CUDA"));
        Ok(())
    }
}
