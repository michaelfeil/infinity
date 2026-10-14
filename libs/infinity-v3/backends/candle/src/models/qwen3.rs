use crate::layers::{HiddenAct, Linear};
use candle::Result;
use candle_nn::VarBuilder;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3Config {
    pub attention_bias: bool,
    pub vocab_size: usize,
    pub head_dim: Option<usize>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub eos_token_id: usize,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    #[serde(default)]
    pub use_linear_output_projection: bool,
    #[serde(default)]
    pub linear_output_size: usize,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default = "default_norm_topk")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

fn default_sparse_step() -> usize {
    1
}
fn default_norm_topk() -> bool {
    true
}

impl Qwen3Config {
    pub(crate) fn is_moe_layer(&self, index: usize) -> Result<bool> {
        if self.num_experts == 0 {
            return Ok(false);
        }
        if self.quantization_config.is_some() {
            candle::bail!("Quantized Qwen3-MoE checkpoints are not supported");
        }
        if self.rope_scaling.is_some() {
            candle::bail!("Scaled RoPE is not yet supported for Qwen3-MoE");
        }
        if self.decoder_sparse_step == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.moe_intermediate_size == 0
        {
            candle::bail!("Invalid Qwen3-MoE routing or expert dimensions");
        }
        Ok(!self.mlp_only_layers.contains(&index)
            && (index + 1).is_multiple_of(self.decoder_sparse_step))
    }

    pub(crate) fn load_output_projection(
        &self,
        root: &VarBuilder,
        model: &VarBuilder,
    ) -> Result<Option<Linear>> {
        // Voyage's projection is outside `model` and has no bias. Do not infer
        // a projection from num_labels alone: causal checkpoints may set it too.
        if self.use_bidirectional_attention && root.contains_tensor("linear.weight") {
            let size = self.num_labels.ok_or_else(|| {
                candle::Error::Msg("Qwen3 linear.weight requires num_labels".into())
            })?;
            let weight = root.pp("linear").get((size, self.hidden_size), "weight")?;
            return Ok(Some(Linear::new(weight, None, None)));
        }
        if self.use_linear_output_projection {
            let vb = model.pp("linear_output_projection");
            let weight = vb.get((self.linear_output_size, self.hidden_size), "weight")?;
            let bias = if vb.contains_tensor("bias") {
                Some(vb.get(self.linear_output_size, "bias")?)
            } else {
                None
            };
            return Ok(Some(Linear::new(weight, bias, None)));
        }
        Ok(None)
    }
}
