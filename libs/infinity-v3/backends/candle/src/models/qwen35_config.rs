#![cfg_attr(not(all(feature = "cuda", feature = "flash-attn")), allow(dead_code))]
use candle::Result;
use serde::Deserialize;

/// Saved in decision_config.json, independently of the backbone configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DecisionAttentionMode {
    #[default]
    Causal,
    NoncausalFullAttention,
}

#[derive(Debug, Deserialize)]
pub(super) struct ReadoutConfig {
    // Original v1 checkpoints omit both fields and use causal, last-token inference.
    #[serde(default)]
    pub attention_mode: DecisionAttentionMode,
    #[serde(default)]
    pooling: Option<String>,
}

impl ReadoutConfig {
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes).map_err(candle::Error::wrap)?;
        if config.pooling.as_deref().is_some_and(|pool| pool != "last") {
            candle::bail!("Pplx requires last-token pooling")
        }
        Ok(config)
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Qwen35Config {
    Multimodal {
        text_config: Qwen35TextConfig,
        #[serde(default)]
        vision_config: Option<serde_json::Value>,
        #[serde(default)]
        tie_word_embeddings: bool,
        #[serde(default)]
        use_bidirectional_attention: bool,
        #[serde(default)]
        use_linear_output_projection: bool,
        #[serde(default)]
        quantization_config: Option<serde_json::Value>,
    },
    Text(Qwen35TextConfig),
}
impl Qwen35Config {
    pub fn text(&self) -> &Qwen35TextConfig {
        match self {
            Self::Multimodal { text_config, .. } => text_config,
            Self::Text(c) => c,
        }
    }
    pub fn vision(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Multimodal { vision_config, .. } => vision_config.as_ref(),
            Self::Text(_) => None,
        }
    }
    pub fn tie_word_embeddings(&self) -> bool {
        self.text().tie_word_embeddings
            || matches!(
                self,
                Self::Multimodal {
                    tie_word_embeddings: true,
                    ..
                }
            )
    }
    pub fn validate(&self) -> Result<()> {
        if matches!(
            self,
            Self::Multimodal {
                use_bidirectional_attention: true,
                ..
            } | Self::Multimodal {
                use_linear_output_projection: true,
                ..
            }
        ) {
            candle::bail!(
                "Qwen3.5 supports causal text embeddings without an output projection only"
            )
        }
        if matches!(
            self,
            Self::Multimodal {
                quantization_config: Some(_),
                ..
            }
        ) {
            candle::bail!("Quantized Qwen3.5 checkpoints are unsupported")
        };
        self.text().validate()
    }
}
#[derive(Debug, Clone, Deserialize)]
pub struct Rope {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Option<[usize; 3]>,
    #[serde(default)]
    pub mrope_interleaved: bool,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub rope_parameters: Rope,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    #[serde(default)]
    pub use_linear_output_projection: bool,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    pub hidden_act: String,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}
fn default_norm_topk_prob() -> bool {
    true
}

impl Qwen35TextConfig {
    pub(crate) fn moe_config(&self) -> Result<super::Qwen3Config> {
        serde_json::from_value(serde_json::json!({
            "attention_bias": false,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_parameters.rope_theta,
            "use_sliding_window": false,
            "eos_token_id": 0,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "norm_topk_prob": self.norm_topk_prob
        }))
        .map_err(candle::Error::wrap)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
    pub fn validate(&self) -> Result<()> {
        if self.use_bidirectional_attention || self.use_linear_output_projection {
            candle::bail!(
                "Qwen3.5 supports causal text embeddings without an output projection only"
            )
        }
        if !self.mlp_only_layers.is_empty()
            || self.quantization_config.is_some()
            || self.attention_bias
            || self.hidden_act != "silu"
            || self.rope_parameters.rope_type != "default"
        {
            candle::bail!("Unsupported Qwen3.5 quantization, attention bias, activation or RoPE configuration")
        }
        if self.max_position_embeddings == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.
            || self.num_hidden_layers == 0
            || self.layer_types.len() != self.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|x| x != "linear_attention" && x != "full_attention")
            || self.hidden_size == 0
            || self.vocab_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.head_dim == 0
            || self.rotary_dim() == 0
            || !self.rotary_dim().is_multiple_of(2)
            || self.rotary_dim() > self.head_dim
            || self.linear_key_head_dim != 128
            || self.linear_value_head_dim != 128
            || self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || self.linear_num_value_heads > 128
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || self.linear_conv_kernel_dim == 0
            || self.linear_conv_kernel_dim > 16
            || if self.num_experts == 0 {
                self.intermediate_size == 0 || self.num_experts_per_tok != 0
            } else {
                self.num_experts != 256
                    || self.num_experts_per_tok != 8
                    || self.moe_intermediate_size == 0
                    || self.shared_expert_intermediate_size == 0
            }
        {
            candle::bail!("Unsupported Qwen3.5 architecture dimensions")
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readout_attention_modes() -> Result<()> {
        for bytes in [b"{}".as_slice(), br#"{"attention_mode":"causal"}"#] {
            assert_eq!(
                ReadoutConfig::from_slice(bytes)?.attention_mode,
                DecisionAttentionMode::Causal
            );
        }
        let config = ReadoutConfig::from_slice(
            br#"{"format_version":1,"attention_mode":"noncausal_full_attention","pooling":"last","temperature":1.0087417621345625}"#,
        )?;
        assert_eq!(
            config.attention_mode,
            DecisionAttentionMode::NoncausalFullAttention
        );
        for bytes in [
            br#"{"attention_mode":"bidirectional"}"#.as_slice(),
            br#"{"attention_mode":null}"#,
            br#"{"pooling":"mean"}"#,
        ] {
            assert!(ReadoutConfig::from_slice(bytes).is_err());
        }
        Ok(())
    }
    fn value() -> serde_json::Value {
        serde_json::json!({
            "hidden_size":2048,"vocab_size":248320,"num_hidden_layers":4,
            "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
            "max_position_embeddings":262144,"rms_norm_eps":1e-6,
            "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
            "rope_parameters":{"rope_type":"default","rope_theta":10000000.,"partial_rotary_factor":0.25},
            "num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,
            "linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"hidden_act":"silu"
        })
    }
    #[test]
    fn dense_configs_and_invalid_expert_mix() -> Result<()> {
        for (hidden, intermediate, heads, kv, values, layers) in
            [(1024, 3584, 8, 2, 16, 24), (4096, 12288, 16, 4, 32, 32)]
        {
            let mut v = value();
            for key in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ] {
                v.as_object_mut().unwrap().remove(key);
            }
            v["hidden_size"] = hidden.into();
            v["intermediate_size"] = intermediate.into();
            v["num_attention_heads"] = heads.into();
            v["num_key_value_heads"] = kv.into();
            v["linear_num_value_heads"] = values.into();
            v["num_hidden_layers"] = layers.into();
            v["layer_types"] = serde_json::json!((0..layers)
                .map(|i| if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                })
                .collect::<Vec<_>>());
            let c: Qwen35Config =
                serde_json::from_value(serde_json::json!({"text_config":v.clone()})).unwrap();
            c.validate()?;
            assert_eq!(c.text().num_experts, 0);
            v["num_experts_per_tok"] = 8.into();
            let c: Qwen35Config = serde_json::from_value(v).unwrap();
            assert!(c.validate().is_err());
        }
        Ok(())
    }

    #[test]
    fn wrapped_and_text_configs() -> Result<()> {
        for config in [
            value(),
            serde_json::json!({"text_config":value(),"tie_word_embeddings":true}),
        ] {
            let c: Qwen35Config = serde_json::from_value(config).unwrap();
            c.validate()?;
            assert_eq!(c.text().rotary_dim(), 64);
        }
        Ok(())
    }
    #[test]
    fn reject_unsupported_embedding_modes() {
        for name in [
            "use_bidirectional_attention",
            "use_linear_output_projection",
        ] {
            let mut text = value();
            text[name] = true.into();
            for config in [text.clone(), serde_json::json!({"text_config": text})] {
                let config: Qwen35Config = serde_json::from_value(config).unwrap();
                assert!(config.validate().is_err(), "{name}");
            }
            let mut wrapper = serde_json::json!({"text_config": value()});
            wrapper[name] = true.into();
            let config: Qwen35Config = serde_json::from_value(wrapper).unwrap();
            assert!(config.validate().is_err(), "outer {name}");
        }
    }
    #[test]
    fn preserve_topk_normalization_setting() -> Result<()> {
        let defaults: Qwen35TextConfig = serde_json::from_value(value()).unwrap();
        assert!(defaults.moe_config()?.norm_topk_prob);
        for renormalize in [false, true] {
            let mut config = value();
            config["norm_topk_prob"] = serde_json::json!(renormalize);
            let text: Qwen35TextConfig = serde_json::from_value(config).unwrap();
            assert_eq!(text.moe_config()?.norm_topk_prob, renormalize);
        }
        Ok(())
    }

    #[test]
    fn reject_unsupported_architecture() {
        for (key, value) in [
            ("num_experts", serde_json::json!(128)),
            ("linear_key_head_dim", serde_json::json!(64)),
            ("layer_types", serde_json::json!(["full_attention"])),
            (
                "quantization_config",
                serde_json::json!({"quant_method":"fp8"}),
            ),
        ] {
            let mut v = self::value();
            v[key] = value;
            let c: Qwen35Config = serde_json::from_value(v).unwrap();
            assert!(c.validate().is_err());
        }
        let v =
            serde_json::json!({"text_config":value(),"quantization_config":{"quant_method":"fp8"}});
        let c: Qwen35Config = serde_json::from_value(v).unwrap();
        assert!(c.validate().is_err());
    }
}
