use crate::layers::{HiddenAct, LayerNorm, Linear};
use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::deepseek2::{BincountOp, NonZeroOp, TopKLastDimOp, TopKOutput};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NomicConfig {
    pub prenorm: bool,
    pub rotary_emb_fraction: f32,
    pub qkv_proj_bias: bool,
    pub rotary_emb_base: f32,
    pub rotary_emb_interleaved: bool,
    pub mlp_fc1_bias: bool,
    pub mlp_fc2_bias: bool,
    pub rotary_scaling_factor: Option<f32>,
    #[serde(default = "default_max_trained_positions")]
    pub max_trained_positions: usize,

    pub moe_every_n_layers: Option<usize>,
    pub moe_normalize_expert_weights: Option<bool>,
    pub moe_top_k: Option<usize>,
    pub num_experts: Option<usize>,

    pub n_embd: usize,
    pub n_head: usize,
    pub n_inner: usize,
    pub n_layer: usize,
    pub n_positions: usize,

    pub activation_function: HiddenAct,

    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f32,
}

fn default_max_trained_positions() -> usize {
    2048
}

impl NomicConfig {
    // For now, we only support these parameters
    pub fn valid(&self) -> bool {
        !self.prenorm
            && self.rotary_emb_fraction == 1.0
            && !self.rotary_emb_interleaved
            && self.type_vocab_size > 0
    }
}

#[derive(Debug)]
pub struct NomicBertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,

    span: tracing::Span,
}

impl NomicBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("embeddings.word_embeddings")
                    .get((config.vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("embeddings.token_type_embeddings")
                    .get((config.type_vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            layer_norm: LayerNorm::load(vb.pp("emb_ln"), config.n_embd, config.layer_norm_epsilon)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = self
            .layer_norm
            .forward(&input_embeddings, Some(&token_type_embeddings))?;

        Ok(embeddings)
    }
}

pub struct NomicBertGatedMLP {
    fc1: Linear,
    fc2: Linear,

    span: tracing::Span,
}

impl NomicBertGatedMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let intermediate_size = config.n_inner;

        let fc11_weight = vb
            .pp("fc11")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc12_weight = vb
            .pp("fc12")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc1_weight = Tensor::cat(&[fc12_weight, fc11_weight], 0)?;

        let fc1_bias = if config.mlp_fc1_bias {
            let fc11_bias = vb.pp("fc11").get((intermediate_size,), "bias")?;
            let fc12_bias = vb.pp("fc12").get((intermediate_size,), "bias")?;
            Some(Tensor::cat(&[fc12_bias, fc11_bias], 0)?)
        } else {
            None
        };

        let fc1 = Linear::new(
            fc1_weight,
            fc1_bias,
            Some(config.activation_function.clone()),
        );

        let fc2_weight = vb
            .pp("fc2")
            .get((config.n_embd, intermediate_size), "weight")?;
        let fc2_bias = if config.mlp_fc2_bias {
            Some(vb.pp("fc2").get((config.n_embd,), "bias")?)
        } else {
            None
        };
        let fc2 = Linear::new(fc2_weight, fc2_bias, None);

        Ok(Self {
            fc1,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "gated_mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.fc1.forward(hidden_states)?;
        self.fc2.forward(&gate_up_states)
    }
}

pub struct NomicBertMLP {
    fc1: Linear,
    fc2: Linear,

    span: tracing::Span,
}

impl NomicBertMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let intermediate_size = config.n_inner;

        let fc1_weight = vb
            .pp("fc1")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc1_bias = if config.mlp_fc1_bias {
            Some(vb.pp("fc1").get((intermediate_size,), "bias")?)
        } else {
            None
        };
        let fc1 = Linear::new(
            fc1_weight,
            fc1_bias,
            Some(config.activation_function.clone()),
        );

        let fc2_weight = vb
            .pp("fc2")
            .get((config.n_embd, intermediate_size), "weight")?;
        let fc2_bias = if config.mlp_fc2_bias {
            Some(vb.pp("fc2").get((config.n_embd,), "bias")?)
        } else {
            None
        };
        let fc2 = Linear::new(fc2_weight, fc2_bias, None);

        Ok(Self {
            fc1,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let y = self.fc1.forward(hidden_states)?;
        self.fc2.forward(&y)
    }
}

pub struct NomicRouter {
    layer: Linear,
    moe_top_k: usize,

    span: tracing::Span,
}

impl NomicRouter {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let num_experts = config.num_experts.unwrap();
        let moe_top_k = config.moe_top_k.unwrap();

        let layer_weight = vb.pp("layer").get((num_experts, config.n_embd), "weight")?;
        let layer = Linear::new(layer_weight, None, None);

        Ok(Self {
            layer,
            moe_top_k,
            span: tracing::span!(tracing::Level::TRACE, "router"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let weights = hidden_states.reshape(((), hidden_states.dim(D::Minus1)?))?;
        let weights = self.layer.forward(&weights)?.to_dtype(DType::F32)?;
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;

        let TopKOutput { values, indices } = weights.topk(self.moe_top_k)?;

        let values = values.to_dtype(hidden_states.dtype())?;

        Ok((values, indices))
    }
}

pub struct NomicExpertMLP {
    w1: Tensor,
    w2: Tensor,
    activation: HiddenAct,

    span: tracing::Span,
}

impl NomicExpertMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let hidden_size = config.n_embd;
        let ffn_hidden_size = config.n_inner;
        let moe_num_experts = config.num_experts.unwrap();
        let activation = config.activation_function.clone();

        let w1 = vb
            .get((moe_num_experts * ffn_hidden_size, hidden_size), "w1")?
            .reshape((moe_num_experts, ffn_hidden_size, hidden_size))?;
        let w2 = vb
            .get((moe_num_experts * ffn_hidden_size, hidden_size), "w2")?
            .reshape((moe_num_experts, ffn_hidden_size, hidden_size))?;

        Ok(Self {
            w1,
            w2,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "expert_mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, expert_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        let expert_w1 = self.w1.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
        let expert_w2 = self.w2.narrow(0, expert_idx, 1)?.squeeze(0)?;

        let hidden_states = hidden_states.broadcast_matmul(&expert_w1)?;
        let hidden_states = self.activation.forward(&hidden_states)?;

        hidden_states.broadcast_matmul(&expert_w2)
    }
}

pub struct NomicExperts {
    moe_num_experts: usize,
    mlp: NomicExpertMLP,
    bias: Tensor,

    span: tracing::Span,
}

impl NomicExperts {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let moe_num_experts = config.num_experts.unwrap();

        let mlp = NomicExpertMLP::load(vb.pp("mlp"), config)?;

        let bias = vb.get((config.n_embd,), "bias")?;

        Ok(Self {
            moe_num_experts,
            mlp,
            bias,
            span: tracing::span!(tracing::Level::TRACE, "experts"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        top_weights: &Tensor,
        top_experts: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let dims = hidden_states.dims();
        let ndim = dims.len();

        let (bs, seq_len, hidden_size) = match ndim {
            3 => (dims[0], dims[1], dims[2]),
            2 => (1, dims[0], dims[1]),
            _ => unreachable!(),
        };

        let hidden_states = hidden_states.reshape(((), hidden_size))?;

        let mut out = Tensor::zeros_like(&hidden_states)?;

        let counts = top_experts
            .flatten_all()?
            .bincount(self.moe_num_experts as u32)?;

        for (expert_idx, &count) in counts.iter().enumerate().take(self.moe_num_experts) {
            if count == 0u32 {
                continue;
            }

            let idx_top = top_experts.eq(expert_idx as f64)?.nonzero()?.t()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;

            let expert_out = self
                .mlp
                .forward(&hidden_states.index_select(idx, 0)?, expert_idx)?
                .broadcast_mul(
                    &top_weights
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(hidden_states.dtype())?,
                )?;

            out = out.index_add(idx, &expert_out, 0)?;
        }

        if ndim == 3 {
            out = out.reshape((bs, seq_len, hidden_size))?;
        }

        out.broadcast_add(&self.bias)
    }
}

pub struct NomicMoELayer {
    router: NomicRouter,
    experts: NomicExperts,

    span: tracing::Span,
}

impl NomicMoELayer {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let router = NomicRouter::load(vb.pp("router"), config)?;
        let experts = NomicExperts::load(vb.pp("experts"), config)?;

        Ok(Self {
            router,
            experts,
            span: tracing::span!(tracing::Level::TRACE, "moe"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (top_weights, top_experts) = self.router.forward(hidden_states)?;

        self.experts
            .forward(hidden_states, &top_weights, &top_experts)
    }
}

pub enum NomicMLP {
    MoE(NomicMoELayer),
    GatedMLP(NomicBertGatedMLP),
    Mlp(NomicBertMLP),
}

impl NomicMLP {
    pub fn load(vb: VarBuilder, index: usize, config: &NomicConfig) -> Result<Self> {
        let use_moe = matches!(config.moe_every_n_layers, Some(n) if n > 0 && index % n == 1);

        if use_moe {
            Ok(Self::MoE(NomicMoELayer::load(vb, config)?))
        } else if config.activation_function == HiddenAct::Gelu {
            Ok(Self::Mlp(NomicBertMLP::load(vb, config)?))
        } else {
            Ok(Self::GatedMLP(NomicBertGatedMLP::load(vb, config)?))
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE(layer) => layer.forward(hidden_states),
            Self::GatedMLP(layer) => layer.forward(hidden_states),
            Self::Mlp(layer) => layer.forward(hidden_states),
        }
    }
}
