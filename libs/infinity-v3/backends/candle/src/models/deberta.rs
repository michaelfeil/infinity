//! Packed DeBERTa-v2/v3 inference using FA4's disentangled-attention score hook.
use crate::layers::{index_select, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, Module, Result, Tensor};
use candle_flash_attn_v4::{deberta_attn_varlen, RelativeBuckets, Seqlens};
use candle_nn::{Activation, Embedding, VarBuilder};
use infinity_v3_backend_core::{Batch, ModelType, Pool};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DebertaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f64,
    hidden_act: Activation,
    relative_attention: bool,
    position_buckets: i64,
    max_relative_positions: i64,
    position_biased_input: bool,
    share_att_key: bool,
    pos_att_type: serde_json::Value,
    norm_rel_ebd: String,
    embedding_size: Option<usize>,
    attention_head_size: Option<usize>,
    conv_kernel_size: usize,
    conv_groups: usize,
    conv_act: String,
    pooler_hidden_size: Option<usize>,
    pooler_hidden_act: Activation,
    architectures: Vec<String>,
    num_labels: Option<usize>,
    id2label: Option<std::collections::HashMap<String, String>>,
}
impl Default for DebertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 0,
            layer_norm_eps: 1e-7,
            hidden_act: Activation::Gelu,
            relative_attention: false,
            position_buckets: -1,
            max_relative_positions: -1,
            position_biased_input: true,
            share_att_key: false,
            pos_att_type: serde_json::Value::Null,
            norm_rel_ebd: "none".into(),
            embedding_size: None,
            attention_head_size: None,
            conv_kernel_size: 0,
            conv_groups: 1,
            conv_act: "tanh".into(),
            pooler_hidden_size: None,
            pooler_hidden_act: Activation::Gelu,
            architectures: vec![],
            num_labels: None,
            id2label: None,
        }
    }
}
fn linear(vb: VarBuilder, input: usize, output: usize, bias: bool) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?,
        if bias {
            Some(vb.get(output, "bias")?)
        } else {
            None
        },
        None,
    ))
}
fn norm(vb: VarBuilder, c: &DebertaConfig) -> Result<LayerNorm> {
    LayerNorm::load(vb, c.hidden_size, c.layer_norm_eps as f32)
}
fn embedding(vb: VarBuilder, n: usize, d: usize) -> Result<Embedding> {
    Ok(Embedding::new(vb.get((n, d), "weight")?, d))
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    pk: Option<Tensor>,
    pq: Option<Tensor>,
    out: Linear,
    att_norm: LayerNorm,
    up: Linear,
    down: Linear,
    out_norm: LayerNorm,
    heads: usize,
    dim: usize,
    scale: Tensor,
    activation: Activation,
    span: usize,
}
impl Layer {
    fn load(
        vb: VarBuilder,
        c: &DebertaConfig,
        relative: Option<&Tensor>,
        span: usize,
        c2p: bool,
        p2c: bool,
    ) -> Result<Self> {
        let h = c.num_attention_heads;
        let d = c.attention_head_size.unwrap_or(c.hidden_size / h);
        let all = h * d;
        let a = vb.pp("attention.self");
        let q = linear(a.pp("query_proj"), c.hidden_size, all, true)?;
        let k = linear(a.pp("key_proj"), c.hidden_size, all, true)?;
        let v = linear(a.pp("value_proj"), c.hidden_size, all, true)?;
        let project = |is_query: bool| -> Result<Option<Tensor>> {
            let Some(rel) = relative else { return Ok(None) };
            let enabled = if is_query { p2c } else { c2p };
            if !enabled {
                return Ok(None);
            }
            let projected = if c.share_att_key {
                if is_query {
                    q.forward(rel)?
                } else {
                    k.forward(rel)?
                }
            } else {
                linear(
                    a.pp(if is_query {
                        "pos_query_proj"
                    } else {
                        "pos_key_proj"
                    }),
                    c.hidden_size,
                    all,
                    true,
                )?
                .forward(rel)?
            };
            Ok(Some(
                projected
                    .reshape((2 * span, h, d))?
                    .transpose(0, 1)?
                    .transpose(1, 2)?
                    .contiguous()?,
            ))
        };
        let pk = project(false)?;
        let pq = project(true)?;
        Ok(Self {
            pk,
            pq,
            q,
            k,
            v,
            out: linear(vb.pp("attention.output.dense"), all, c.hidden_size, true)?,
            att_norm: norm(vb.pp("attention.output.LayerNorm"), c)?,
            up: linear(
                vb.pp("intermediate.dense"),
                c.hidden_size,
                c.intermediate_size,
                true,
            )?,
            down: linear(
                vb.pp("output.dense"),
                c.intermediate_size,
                c.hidden_size,
                true,
            )?,
            out_norm: norm(vb.pp("output.LayerNorm"), c)?,
            heads: h,
            dim: d,
            // Match HF: pos_att_type controls scaling even if relative_attention
            // is false. Only projection/table construction is gated by that flag.
            scale: Tensor::new(
                &[(((1 + c2p as usize + p2c as usize) * d) as f32).sqrt()],
                vb.device(),
            )?
            .to_dtype(vb.dtype())?,
            activation: c.hidden_act,
            span,
        })
    }
    fn forward(&self, x: &Tensor, lengths: &Seqlens, buckets: &RelativeBuckets) -> Result<Tensor> {
        let t = x.dim(0)?;
        let shape = (t, self.heads, self.dim);
        let q = self.q.forward(x)?.reshape(shape)?;
        let k = self.k.forward(x)?.reshape(shape)?;
        let v = self.v.forward(x)?.reshape(shape)?;
        let rel = |input: &Tensor, weight: &Option<Tensor>| -> Result<Tensor> {
            match weight {
                Some(w) => input
                    .transpose(0, 1)?
                    .contiguous()?
                    .matmul(w)?
                    .broadcast_div(&self.scale),
                None => Tensor::zeros((self.heads, t, 2 * self.span), x.dtype(), x.device()),
            }
        };
        let c2p = rel(&q, &self.pk)?;
        let p2c = rel(&k, &self.pq)?;
        let att = deberta_attn_varlen(
            &q,
            &k.broadcast_div(&self.scale)?,
            &v,
            &c2p,
            &p2c,
            lengths,
            buckets,
        )?
        .flatten_from(1)?;
        // DeBERTa rounds the residual addition before LayerNorm, unlike the
        // optional fused residual APIs used by some other infinity_v3 architectures.
        let x = self
            .att_norm
            .forward(&self.out.forward(&att)?.add(x)?, None)?;
        let mlp = self
            .down
            .forward(&self.activation.forward(&self.up.forward(&x)?)?)?;
        self.out_norm.forward(&mlp.add(&x)?, None)
    }
}
struct Conv {
    layer: candle_nn::Conv1d,
    norm: LayerNorm,
    act: String,
}
impl Conv {
    fn activate(&self, x: &Tensor) -> Result<Tensor> {
        match self.act.as_str() {
            "tanh" => x.tanh(),
            "gelu" => x.gelu_erf(),
            "relu" => x.relu(),
            "silu" => x.silu(),
            _ => candle::bail!("unsupported DeBERTa convolution activation"),
        }
    }
}
pub struct DebertaModel {
    word: Embedding,
    pos: Option<Embedding>,
    types: Option<Embedding>,
    embed_proj: Option<Linear>,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    conv: Option<Conv>,
    pool: Pool,
    head: Option<Linear>,
    pooler: Option<Linear>,
    pooler_act: Activation,
    span: usize,
    log_buckets: bool,
    max_position: usize,
    device: Device,
}
impl DebertaModel {
    pub fn load(vb: VarBuilder, c: &DebertaConfig, model_type: ModelType) -> Result<Self> {
        if !matches!(vb.dtype(), DType::F16 | DType::BF16) || !vb.device().is_cuda() {
            candle::bail!("DeBERTa FA4 requires FP16/BF16 CUDA")
        }
        if c.num_attention_heads == 0
            || c.hidden_size == 0
            || !c.hidden_size.is_multiple_of(c.num_attention_heads)
            || c.attention_head_size
                .unwrap_or(c.hidden_size / c.num_attention_heads)
                != 64
        {
            candle::bail!("DeBERTa FA4 currently requires head dimension 64")
        }
        let flags: Vec<String> = match &c.pos_att_type {
            serde_json::Value::Null => vec![],
            serde_json::Value::String(s) => s.split('|').map(|v| v.trim().to_owned()).collect(),
            serde_json::Value::Array(a) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| candle::Error::Msg("invalid pos_att_type".into()))
                })
                .collect::<Result<_>>()?,
            _ => candle::bail!("invalid pos_att_type"),
        };
        if flags.iter().any(|v| v != "c2p" && v != "p2c") {
            candle::bail!("unsupported DeBERTa position term")
        }
        let max_position = if c.max_relative_positions > 0 {
            c.max_relative_positions as usize
        } else {
            c.max_position_embeddings
        };
        let span = if c.position_buckets > 0 {
            c.position_buckets as usize
        } else {
            max_position
        };
        let base = if vb.contains_tensor("deberta.embeddings.word_embeddings.weight") {
            vb.pp("deberta")
        } else {
            vb.clone()
        };
        let e = base.pp("embeddings");
        let es = c.embedding_size.unwrap_or(c.hidden_size);
        let mut relative = if c.relative_attention {
            Some(
                base.pp("encoder.rel_embeddings")
                    .get((span * 2, c.hidden_size), "weight")?,
            )
        } else {
            None
        };
        for n in c.norm_rel_ebd.split('|').map(str::trim) {
            match n {
                "none" | "" => {}
                "layer_norm" => {
                    if let Some(r) = &relative {
                        relative = Some(norm(base.pp("encoder.LayerNorm"), c)?.forward(r, None)?);
                    }
                }
                _ => candle::bail!("unsupported norm_rel_ebd"),
            }
        }
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                Layer::load(
                    base.pp(format!("encoder.layer.{i}")),
                    c,
                    relative.as_ref(),
                    span,
                    flags.iter().any(|s| s == "c2p"),
                    flags.iter().any(|s| s == "p2c"),
                )
            })
            .collect::<Result<_>>()?;
        let (pool, head, pooler) = match model_type {
            ModelType::Decision => candle::bail!("Typed decisions require a Laya checkpoint"),
            ModelType::Embedding(Pool::Splade) => {
                candle::bail!("DeBERTa SPLADE is not implemented")
            }
            ModelType::Embedding(p) => (p, None, None),
            ModelType::Classifier => {
                let token = c
                    .architectures
                    .iter()
                    .any(|s| s.ends_with("ForTokenClassification"));
                let labels = c
                    .id2label
                    .as_ref()
                    .map(|v| v.len())
                    .or(c.num_labels)
                    .unwrap_or(2);
                let pd = c.pooler_hidden_size.unwrap_or(c.hidden_size);
                if pd != c.hidden_size {
                    candle::bail!("unsupported DeBERTa pooler width")
                }
                (
                    Pool::Cls,
                    Some(linear(vb.pp("classifier"), c.hidden_size, labels, true)?),
                    if token {
                        None
                    } else {
                        Some(linear(vb.pp("pooler.dense"), pd, pd, true)?)
                    },
                )
            }
        };
        let conv = if c.conv_kernel_size > 0 {
            if c.conv_kernel_size.is_multiple_of(2)
                || c.conv_groups == 0
                || !c.hidden_size.is_multiple_of(c.conv_groups)
            {
                candle::bail!("invalid DeBERTa convolution geometry")
            }
            Some(Conv {
                layer: candle_nn::conv1d(
                    c.hidden_size,
                    c.hidden_size,
                    c.conv_kernel_size,
                    candle_nn::Conv1dConfig {
                        padding: (c.conv_kernel_size - 1) / 2,
                        groups: c.conv_groups,
                        ..Default::default()
                    },
                    base.pp("encoder.conv.conv"),
                )?,
                norm: norm(base.pp("encoder.conv.LayerNorm"), c)?,
                act: c.conv_act.clone(),
            })
        } else {
            None
        };
        Ok(Self {
            word: embedding(e.pp("word_embeddings"), c.vocab_size, es)?,
            pos: if c.position_biased_input {
                Some(embedding(
                    e.pp("position_embeddings"),
                    c.max_position_embeddings,
                    es,
                )?)
            } else {
                None
            },
            types: if c.type_vocab_size > 0 {
                Some(embedding(
                    e.pp("token_type_embeddings"),
                    c.type_vocab_size,
                    es,
                )?)
            } else {
                None
            },
            embed_proj: if es != c.hidden_size {
                Some(linear(e.pp("embed_proj"), es, c.hidden_size, false)?)
            } else {
                None
            },
            embed_norm: norm(e.pp("LayerNorm"), c)?,
            layers,
            conv,
            pool,
            head,
            pooler,
            pooler_act: c.pooler_hidden_act,
            span,
            log_buckets: c.position_buckets > 0,
            max_position,
            device: vb.device().clone(),
        })
    }
    fn forward(&self, b: &Batch) -> Result<Tensor> {
        let lengths = Seqlens::new(&b.cumulative_seq_lengths, &self.device)?;
        if b.cumulative_seq_lengths.last().copied() != Some(b.input_ids.len() as u32)
            || b.position_ids.len() != b.input_ids.len()
            || b.token_type_ids.len() != b.input_ids.len()
        {
            candle::bail!("invalid packed DeBERTa batch")
        }
        // The relative-position lookup assumes positions restart at zero for each sequence.
        for w in b.cumulative_seq_lengths.windows(2) {
            if b.position_ids[w[0] as usize..w[1] as usize]
                .iter()
                .enumerate()
                .any(|(i, &p)| i != p as usize)
            {
                candle::bail!("DeBERTa requires contiguous per-sequence positions")
            }
        }
        let max_len = b
            .cumulative_seq_lengths
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap();
        let buckets = RelativeBuckets::new(
            max_len,
            self.span,
            self.log_buckets,
            self.max_position,
            &self.device,
        )?;
        let mut x = self
            .word
            .forward(&Tensor::new(b.input_ids.as_slice(), &self.device)?)?;
        if let Some(p) = &self.pos {
            x = x.add(&p.forward(&Tensor::new(b.position_ids.as_slice(), &self.device)?)?)?;
        }
        if let Some(t) = &self.types {
            x = x.add(&t.forward(&Tensor::new(b.token_type_ids.as_slice(), &self.device)?)?)?;
        }
        if let Some(p) = &self.embed_proj {
            x = p.forward(&x)?;
        }
        x = self.embed_norm.forward(&x, None)?;
        let embedding = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &lengths, &buckets)?;
            if i == 0 {
                if let Some(conv) = &self.conv {
                    // Separate sequence views prevent convolution crossing packed boundaries.
                    let pieces = b
                        .cumulative_seq_lengths
                        .windows(2)
                        .map(|w| {
                            let y = embedding
                                .narrow(0, w[0] as usize, (w[1] - w[0]) as usize)?
                                .transpose(0, 1)?
                                .unsqueeze(0)?
                                .contiguous()?;
                            conv.activate(
                                &conv
                                    .layer
                                    .forward(&y)?
                                    .squeeze(0)?
                                    .transpose(0, 1)?
                                    .contiguous()?,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    x = conv
                        .norm
                        .forward(&x.add(&Tensor::cat(&pieces, 0)?)?, None)?;
                }
            }
        }
        Ok(x)
    }
    fn pool(&self, x: &Tensor, b: &Batch) -> Result<Tensor> {
        if self.pool == Pool::Mean {
            return crate::layers::mean_pool(x, &b.cumulative_seq_lengths, &b.pooled_indices);
        }
        let ids = b
            .pooled_indices
            .iter()
            .map(|&i| {
                let w = b
                    .cumulative_seq_lengths
                    .get(i as usize..i as usize + 2)
                    .ok_or_else(|| candle::Error::Msg("pool index out of range".into()))?;
                Ok(if self.pool == Pool::LastToken {
                    w[1] - 1
                } else {
                    w[0]
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        index_select(x, &Tensor::new(ids.as_slice(), &self.device)?, 0)
    }
    fn raw(&self, x: &Tensor, b: &Batch) -> Result<Tensor> {
        let mut ids = Vec::new();
        for &i in &b.raw_indices {
            let w = b
                .cumulative_seq_lengths
                .get(i as usize..i as usize + 2)
                .ok_or_else(|| candle::Error::Msg("raw index out of range".into()))?;
            ids.extend(w[0]..w[1]);
        }
        index_select(x, &Tensor::new(ids.as_slice(), &self.device)?, 0)
    }
}
impl Model for DebertaModel {
    fn embed(&self, b: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let x = self.forward(&b)?;
        Ok((
            if b.pooled_indices.is_empty() {
                None
            } else {
                Some(self.pool(&x, &b)?)
            },
            if b.raw_indices.is_empty() {
                None
            } else {
                Some(self.raw(&x, &b)?)
            },
        ))
    }
    fn predict(&self, b: Batch) -> Result<Tensor> {
        let head = self
            .head
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("DeBERTa classifier not loaded".into()))?;
        let pooler = self.pooler.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "sequence prediction requires a sequence classifier checkpoint".into(),
            )
        })?;
        head.forward(
            &self
                .pooler_act
                .forward(&pooler.forward(&self.pool(&self.forward(&b)?, &b)?)?)?,
        )
    }
    fn predict_tokens(&self, b: Batch) -> Result<Tensor> {
        let head = self
            .head
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("DeBERTa token classifier not loaded".into()))?;
        if self.pooler.is_some() {
            candle::bail!("token prediction requires a token classifier checkpoint")
        }
        head.forward(&self.raw(&self.forward(&b)?, &b)?)
    }
}
