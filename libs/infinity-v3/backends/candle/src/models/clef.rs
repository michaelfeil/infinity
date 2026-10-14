#![cfg_attr(not(all(feature = "cuda", feature = "flash-attn")), allow(dead_code))]
//! Cloudflare's joint schema head; span metadata refers to the unfolded sequence.
use crate::layers::{LayerNorm, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::ClefField;
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
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
fn norm(vb: VarBuilder, width: usize) -> Result<LayerNorm> {
    LayerNorm::load(vb, width, 1e-5)
}
fn normalize(x: &Tensor) -> Result<Tensor> {
    let denom = x
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?
        .to_dtype(x.dtype())?;
    x.broadcast_div(&denom)
}
fn mean_span(x: &Tensor, (start, end): (usize, usize)) -> Result<Tensor> {
    if start >= end || end > x.dim(0)? {
        candle::bail!("Invalid Clef token span")
    }
    x.narrow(0, start, end - start)?
        .to_dtype(DType::F32)?
        .mean(0)?
        .to_dtype(x.dtype())
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}
impl Attention {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        let weight = vb.get((3 * c.width, c.width), "in_proj_weight")?;
        let bias = vb.get(3 * c.width, "in_proj_bias")?;
        let part = |i| -> Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, i * c.width, c.width)?.contiguous()?,
                Some(bias.narrow(0, i * c.width, c.width)?.contiguous()?),
                None,
            ))
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: linear(vb.pp("out_proj"), c.width, c.width, true)?,
            heads: c.heads,
        })
    }
    fn forward(&self, query: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let (n, width) = query.dims2()?;
        let m = memory.dim(0)?;
        let dim = width / self.heads;
        let project = |l: &Linear, x: &Tensor, len| {
            l.forward(x)?.reshape((len, self.heads, dim))?.contiguous()
        };
        let q = project(&self.q, query, n)?;
        let k = project(&self.k, memory, m)?;
        let v = project(&self.v, memory, m)?;
        #[cfg(feature = "flash-attn")]
        if query.device().is_cuda() {
            let offsets_q = [0u32, n as u32];
            let cuq = Tensor::new(&offsets_q[..], query.device())?;
            let cuk = if query.id() == memory.id() {
                cuq.clone()
            } else {
                Tensor::new(&[0u32, m as u32][..], query.device())?
            };
            #[cfg(feature = "fa4")]
            let _guard = crate::fa4_native::prepare_batch(&cuq, &offsets_q)?;
            let output = crate::flash_attn::flash_attn_varlen(
                &q,
                &k,
                &v,
                None,
                &cuq,
                &cuk,
                n,
                m,
                1.0 / (dim as f32).sqrt(),
                false,
                None,
                None,
            )?;
            return self.out.forward(&output.reshape((n, width))?);
        }
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? / (dim as f64).sqrt())?
            .to_dtype(DType::F32)?;
        let p = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(query.dtype())?;
        self.out.forward(
            &p.matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n, width))?,
        )
    }
}
struct Routing {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: Attention,
    feed_norm: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}
impl Routing {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        Ok(Self {
            query_norm: norm(vb.pp("query_norm"), c.width)?,
            memory_norm: norm(vb.pp("memory_norm"), c.width)?,
            attention: Attention::load(vb.pp("attention"), c)?,
            feed_norm: norm(vb.pp("feedforward_norm"), c.width)?,
            ff1: linear(vb.pp("feedforward.0"), c.width, c.feedforward, true)?,
            ff2: linear(vb.pp("feedforward.3"), c.feedforward, c.width, true)?,
        })
    }
    fn forward(&self, q: &Tensor, m: &Tensor) -> Result<Tensor> {
        let q = (q + self.attention.forward(
            &self.query_norm.forward(q, None)?,
            &self.memory_norm.forward(m, None)?,
        )?)?;
        &q + self.ff2.forward(
            &self
                .ff1
                .forward(&self.feed_norm.forward(&q, None)?)?
                .gelu_erf()?,
        )?
    }
}
struct Decoder {
    norms: Vec<LayerNorm>,
    self_attn: Attention,
    cross_attn: Attention,
    ff1: Linear,
    ff2: Linear,
}
impl Decoder {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        Ok(Self {
            norms: (1..=3)
                .map(|i| norm(vb.pp(format!("norm{i}")), c.width))
                .collect::<Result<_>>()?,
            self_attn: Attention::load(vb.pp("self_attn"), c)?,
            cross_attn: Attention::load(vb.pp("multihead_attn"), c)?,
            ff1: linear(vb.pp("linear1"), c.width, c.feedforward, true)?,
            ff2: linear(vb.pp("linear2"), c.feedforward, c.width, true)?,
        })
    }
    fn forward(&self, x: &Tensor, m: &Tensor) -> Result<Tensor> {
        let n = self.norms[0].forward(x, None)?;
        let x = (x + self.self_attn.forward(&n, &n)?)?;
        let x = (&x
            + self
                .cross_attn
                .forward(&self.norms[1].forward(&x, None)?, m)?)?;
        &x + self.ff2.forward(
            &self
                .ff1
                .forward(&self.norms[2].forward(&x, None)?)?
                .gelu_erf()?,
        )?
    }
}
pub struct JointSchemaHead {
    hidden_norm: LayerNorm,
    projections: Vec<Linear>,
    types: Embedding,
    routing: Vec<Routing>,
    decoder: Vec<Decoder>,
    summary_norm: LayerNorm,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    scorer1: Linear,
    scorer2: Linear,
    prior_scale: f64,
    joint_scale: f64,
    gate: f64,
}
impl JointSchemaHead {
    pub fn load(
        path: &std::path::Path,
        hidden: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let c: Config =
            serde_json::from_slice(&std::fs::read(path.join("joint_head_config.json"))?)
                .map_err(candle::Error::wrap)?;
        if c.hidden_size != hidden
            || c.width == 0
            || c.heads == 0
            || !c.width.is_multiple_of(c.heads)
        {
            candle::bail!("Invalid Clef head dimensions")
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.join("joint_head.safetensors")],
                dtype,
                device,
            )?
        };
        let scale = |name: &str| {
            vb.get((), name)?
                .clamp(f64::NEG_INFINITY, 100f64.ln())?
                .exp()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()
                .map(|v| v as f64)
        };
        let gate = candle_nn::ops::sigmoid(&vb.get((), "residual_gate")?)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()? as f64;
        Ok(Self {
            hidden_norm: norm(vb.pp("hidden_norm"), hidden)?,
            projections: [
                "memory_projection",
                "question_projection",
                "option_question_projection",
                "global_projection",
                "option_context_projection",
                "option_lexical_projection",
            ]
            .into_iter()
            .map(|n| linear(vb.pp(n), hidden, c.width, false))
            .collect::<Result<_>>()?,
            types: Embedding::new(
                vb.pp("type_embedding").get((3, c.width), "weight")?,
                c.width,
            ),
            routing: (0..c.routing_layers)
                .map(|i| Routing::load(vb.pp(format!("evidence_layers.{i}")), &c))
                .collect::<Result<_>>()?,
            decoder: (0..c.layers)
                .map(|i| Decoder::load(vb.pp(format!("layers.{i}")), &c))
                .collect::<Result<_>>()?,
            summary_norm: norm(vb.pp("option_summary_norm"), c.width)?,
            field_norm: norm(vb.pp("field_norm"), c.width)?,
            option_norm: norm(vb.pp("option_norm"), c.width)?,
            scorer1: linear(vb.pp("residual_scorer.0"), 4 * c.width, c.width, true)?,
            scorer2: linear(vb.pp("residual_scorer.3"), c.width, 1, true)?,
            prior_scale: scale("prior_logit_scale")?,
            joint_scale: scale("joint_logit_scale")?,
            gate,
        })
    }
    pub fn forward(
        &self,
        hidden: &Tensor,
        ids: &[u32],
        fields: &[ClefField],
        lexical_weight: &Tensor,
    ) -> Result<Vec<f32>> {
        if fields.is_empty() || fields.iter().any(|f| f.kind > 2 || f.options.is_empty()) {
            candle::bail!("Invalid Clef schema")
        }
        let hidden = self.hidden_norm.forward(hidden, None)?;
        let global = hidden.i(hidden.dim(0)? - 1)?;
        let memory = self.projections[0].forward(&hidden)?;
        let questions = Tensor::stack(
            &fields
                .iter()
                .map(|f| mean_span(&hidden, f.question))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?;
        let mut lexical = Vec::new();
        let mut queries = Vec::new();
        for (i, field) in fields.iter().enumerate() {
            let contexts = Tensor::stack(
                &field
                    .options
                    .iter()
                    .map(|&s| mean_span(&hidden, s))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let words = Tensor::stack(
                &field
                    .options
                    .iter()
                    .map(|&(a, b)| {
                        if a >= b || b > ids.len() {
                            candle::bail!("Invalid Clef option span")
                        };
                        let tokens = Tensor::new(&ids[a..b], hidden.device())?;
                        let rows = lexical_weight.index_select(&tokens, 0)?;
                        mean_span(&rows, (0, b - a))
                    })
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let query = (self.projections[4].forward(&contexts)?
                + self.projections[5].forward(&words)?)?
            .broadcast_add(&self.projections[2].forward(&questions.i(i)?.unsqueeze(0)?)?)?;
            lexical.push(words);
            queries.push(query);
        }
        let mut routed = Tensor::cat(&queries, 0)?;
        for l in &self.routing {
            routed = l.forward(&routed, &memory)?;
        }
        let base = self.projections[1].forward(&questions)?;
        let mut options = Vec::new();
        let mut summaries = Vec::new();
        let mut offset = 0;
        for (i, f) in fields.iter().enumerate() {
            let o = routed.narrow(0, offset, f.options.len())?;
            offset += f.options.len();
            let scores =
                (o.matmul(&base.i(i)?.unsqueeze(1)?)?.squeeze(1)? / (o.dim(1)? as f64).sqrt())?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            summaries.push(o.broadcast_mul(&weights.unsqueeze(1)?)?.sum(0)?);
            options.push(o);
        }
        let types = Tensor::new(
            fields.iter().map(|f| f.kind as u32).collect::<Vec<_>>(),
            hidden.device(),
        )?;
        let mut fields_hidden = (base
            + self
                .summary_norm
                .forward(&Tensor::stack(&summaries, 0)?, None)?)?
        .broadcast_add(&self.projections[3].forward(&global.unsqueeze(0)?)?)?;
        fields_hidden = (fields_hidden + self.types.forward(&types)?)?;
        for l in &self.decoder {
            fields_hidden = l.forward(&fields_hidden, &memory)?;
        }
        let fields_hidden = self.field_norm.forward(&fields_hidden, None)?;
        let mut logits = Vec::new();
        for i in 0..fields.len() {
            let anchor = normalize(&(&questions.i(i)? + &global)?)?;
            let prior = (normalize(&lexical[i])?
                .matmul(&anchor.unsqueeze(1)?)?
                .squeeze(1)?
                * self.prior_scale)?;
            let o = self.option_norm.forward(&options[i], None)?;
            let f = fields_hidden.i(i)?.broadcast_as(o.shape())?.contiguous()?;
            let cosine = (normalize(&f)? * normalize(&o)?)?.sum(D::Minus1)?;
            let features = Tensor::cat(&[&f, &o, &(&f * &o)?, &(&f - &o)?.abs()?], 1)?;
            let residual = self
                .scorer2
                .forward(&self.scorer1.forward(&features)?.gelu_erf()?)?
                .squeeze(1)?;
            logits.push((prior + ((cosine * self.joint_scale)? + residual)? * self.gate)?)
        }
        Tensor::cat(&logits, 0)?.to_dtype(DType::F32)?.to_vec1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Clef head weights and an upstream head fixture"]
    fn head_matches_upstream() -> Result<()> {
        let path = std::env::var("CLEF_CHECKPOINT_DIR").unwrap();
        let fixture = std::env::var("CLEF_HEAD_FIXTURE").unwrap();
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{fixture}.json"))?)
                .map_err(candle::Error::wrap)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[format!("{fixture}.safetensors")],
                DType::F32,
                &Device::Cpu,
            )?
        };
        let config: Config =
            serde_json::from_slice(&std::fs::read(format!("{path}/joint_head_config.json"))?)
                .map_err(candle::Error::wrap)?;
        let hidden = vb.get((32, config.hidden_size), "hidden")?;
        let lexical = vb.get((16, config.hidden_size), "lexical")?;
        let fields = data["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ClefField {
                kind: v["kind"].as_u64().unwrap() as usize,
                question: serde_json::from_value(v["question"].clone()).unwrap(),
                options: serde_json::from_value(v["options"].clone()).unwrap(),
            })
            .collect::<Vec<_>>();
        let ids: Vec<u32> = serde_json::from_value(data["input_ids"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(data["logits"].clone()).unwrap();
        let head = JointSchemaHead::load(
            std::path::Path::new(&path),
            config.hidden_size,
            DType::F32,
            &Device::Cpu,
        )?;
        let actual = head.forward(&hidden, &ids, &fields, &lexical)?;
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_error < 1e-4,
            "head max error {max_error}: {actual:?} vs {expected:?}"
        );
        Ok(())
    }
}
