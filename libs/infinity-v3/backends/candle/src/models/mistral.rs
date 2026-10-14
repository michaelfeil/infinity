use crate::layers::{HiddenAct, RopeScaling};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "MistralConfigFields")]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f32,
    pub model_type: Option<String>,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    pub is_causal: Option<bool>,
    pub rope_parameters: Option<MinistralRope>,
}

// HF Ministral3 stores theta inside rope_parameters. Legacy Mistral checkpoints
// retain the root field; reject a missing theta when neither source provides it.
#[derive(Deserialize)]
struct MistralConfigFields {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: Option<usize>,
    num_key_value_heads: usize,
    hidden_act: HiddenAct,
    max_position_embeddings: usize,
    initializer_range: f64,
    rms_norm_eps: f32,
    model_type: Option<String>,
    rope_theta: Option<f32>,
    sliding_window: Option<usize>,
    rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    use_bidirectional_attention: bool,
    is_causal: Option<bool>,
    rope_parameters: Option<MinistralRope>,
}

impl TryFrom<MistralConfigFields> for MistralConfig {
    type Error = &'static str;

    fn try_from(fields: MistralConfigFields) -> Result<Self, Self::Error> {
        let rope_theta = fields
            .rope_theta
            .or_else(|| fields.rope_parameters.as_ref().map(|rope| rope.rope_theta))
            .ok_or("Mistral requires rope_theta or rope_parameters.rope_theta")?;
        Ok(Self {
            vocab_size: fields.vocab_size,
            hidden_size: fields.hidden_size,
            intermediate_size: fields.intermediate_size,
            num_hidden_layers: fields.num_hidden_layers,
            num_attention_heads: fields.num_attention_heads,
            head_dim: fields.head_dim,
            num_key_value_heads: fields.num_key_value_heads,
            hidden_act: fields.hidden_act,
            max_position_embeddings: fields.max_position_embeddings,
            initializer_range: fields.initializer_range,
            rms_norm_eps: fields.rms_norm_eps,
            model_type: fields.model_type,
            rope_theta,
            sliding_window: fields.sliding_window,
            rope_scaling: fields.rope_scaling,
            use_bidirectional_attention: fields.use_bidirectional_attention,
            is_causal: fields.is_causal,
            rope_parameters: fields.rope_parameters,
        })
    }
}

impl MistralConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn bidirectional(&self) -> bool {
        self.is_causal
            .map(|causal| !causal)
            .unwrap_or(self.use_bidirectional_attention)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MinistralRope {
    pub rope_type: String,
    pub rope_theta: f32,
    pub factor: f32,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: usize,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub llama_4_scaling_beta: f32,
}

impl MinistralRope {
    // `apply_yarn_scaling` in Nemotron's checkpoint is vLLM compatibility metadata;
    // Transformers applies YaRN using these parameters regardless of that flag.
    pub fn frequencies(&self, dim: usize) -> candle::Result<(Vec<f32>, f64)> {
        if self.rope_type != "yarn"
            || self.factor < 1.0
            || self.rope_theta <= 1.0
            || self.original_max_position_embeddings == 0
            || self.beta_fast <= 0.0
            || self.beta_slow <= 0.0
            || dim == 0
            || !dim.is_multiple_of(2)
        {
            candle::bail!("Unsupported Ministral RoPE parameters");
        }
        let correction = |rotations: f64| {
            dim as f64
                * (self.original_max_position_embeddings as f64
                    / (rotations * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * (self.rope_theta as f64).ln())
        };
        let low = correction(self.beta_fast).floor().max(0.0) as f32;
        let mut high = correction(self.beta_slow).ceil().min((dim - 1) as f64) as f32;
        if low == high {
            high += 0.001;
        }
        let inv = (0..dim / 2)
            .map(|i| {
                let freq = self.rope_theta.powf((2 * i) as f32 / dim as f32);
                let extrapolation = 1.0 - ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                (1.0 / (self.factor * freq)) * (1.0 - extrapolation) + (1.0 / freq) * extrapolation
            })
            .collect();
        let mscale = |m: f64| 1.0 + 0.1 * m * (self.factor as f64).ln();
        let scale = if self.mscale != 0.0 && self.mscale_all_dim != 0.0 {
            mscale(self.mscale) / mscale(self.mscale_all_dim)
        } else {
            mscale(1.0)
        };
        Ok((inv, scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ministral_nested_theta_config_preserves_legacy_requirement() -> candle::Result<()> {
        let mut value = serde_json::json!({
            "vocab_size":131072,"hidden_size":3072,"intermediate_size":9216,
            "num_hidden_layers":26,"num_attention_heads":32,"num_key_value_heads":8,
            "head_dim":128,"hidden_act":"silu","max_position_embeddings":262144,
            "initializer_range":0.02,"rms_norm_eps":1e-5,"model_type":"ministral3",
            "rope_parameters":{"rope_type":"yarn","rope_theta":1000000.,
                "factor":16.,"beta_fast":32.,"beta_slow":1.,
                "original_max_position_embeddings":16384,"mscale":1.,
                "mscale_all_dim":1.,"llama_4_scaling_beta":0.1}
        });
        let config: MistralConfig =
            serde_json::from_value(value.clone()).map_err(candle::Error::wrap)?;
        assert_eq!(config.rope_theta, 1_000_000.);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.rope_parameters.unwrap().llama_4_scaling_beta, 0.1);

        value.as_object_mut().unwrap().remove("rope_parameters");
        assert!(serde_json::from_value::<MistralConfig>(value.clone()).is_err());
        value["rope_theta"] = 10000.into();
        let legacy: MistralConfig = serde_json::from_value(value).map_err(candle::Error::wrap)?;
        assert_eq!(legacy.rope_theta, 10000.);
        assert!(legacy.rope_parameters.is_none());
        Ok(())
    }

    #[test]
    fn nemotron_yarn_matches_transformers() -> candle::Result<()> {
        let rope = MinistralRope {
            rope_type: "yarn".into(),
            rope_theta: 1_000_000.0,
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 16384,
            mscale: 1.0,
            mscale_all_dim: 1.0,
            llama_4_scaling_beta: 0.1,
        };
        let (freqs, scale) = rope.frequencies(128)?;
        // Independently generated by Transformers' Ministral3RotaryEmbedding.
        for (index, expected) in [
            (0, 1.0),
            (12, 0.07498941570520401),
            (24, 0.004382954444736242),
            (32, 0.0003382352879270911),
            (48, 1.9764233911701012e-6),
            (63, 7.75586102008674e-8),
        ] {
            assert!((freqs[index] as f64 / expected - 1.0).abs() < 1e-6);
        }
        assert_eq!(scale, 1.0);
        Ok(())
    }
}
