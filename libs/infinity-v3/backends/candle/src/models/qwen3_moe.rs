use crate::layers::{gated_activation, HiddenAct, Linear};
use crate::models::Qwen3Config;
use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

/// Qwen3 routed MLP. Attention and pooling use the Qwen3 model.
pub(crate) struct Qwen3Moe {
    gate: Linear,
    gate_up: Tensor,
    down: Tensor,
    top_k: usize,
    renormalize: bool,
}

impl Qwen3Moe {
    pub(crate) fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        if config.hidden_size == 0
            || config.num_experts == 0
            || config.num_experts_per_tok == 0
            || config.num_experts_per_tok > config.num_experts
            || config.moe_intermediate_size == 0
        {
            candle::bail!("Invalid Qwen3-MoE dimensions or routing configuration");
        }
        if config.hidden_act != HiddenAct::Silu {
            candle::bail!("Qwen3-MoE requires SiLU expert activation");
        }
        let (experts, hidden, intermediate) = (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        );
        let gate = Linear::new(vb.pp("gate").get((experts, hidden), "weight")?, None, None);
        let evb = vb.pp("experts");
        let (gate_up, down) = if evb.contains_tensor("gate_up_proj") {
            (
                evb.get((experts, 2 * intermediate, hidden), "gate_up_proj")?,
                evb.get((experts, hidden, intermediate), "down_proj")?,
            )
        } else {
            let mut gate_up = Vec::with_capacity(experts);
            let mut down = Vec::with_capacity(experts);
            for index in 0..experts {
                let expert = evb.pp(index.to_string());
                let gate = expert
                    .pp("gate_proj")
                    .get((intermediate, hidden), "weight")?;
                let up = expert.pp("up_proj").get((intermediate, hidden), "weight")?;
                gate_up.push(Tensor::cat(&[gate, up], 0)?);
                down.push(
                    expert
                        .pp("down_proj")
                        .get((hidden, intermediate), "weight")?,
                );
            }
            (Tensor::stack(&gate_up, 0)?, Tensor::stack(&down, 0)?)
        };
        Ok(Self {
            gate,
            gate_up,
            down,
            top_k: config.num_experts_per_tok,
            renormalize: config.norm_topk_prob,
        })
    }

    pub(crate) fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let shape = hidden.shape();
        let width = hidden.dim(candle::D::Minus1)?;
        let flat = hidden
            .reshape((hidden.elem_count() / width, width))?
            .contiguous()?;
        if flat.dim(0)? > u32::MAX as usize {
            candle::bail!("Qwen3-MoE batch exceeds the token index range");
        }
        let logits = self.gate.forward(&flat)?.to_dtype(DType::F32)?;
        #[cfg(gemma4_moe_cuda)]
        if matches!(hidden.device(), candle::Device::Cuda(_))
            && hidden.dtype() == DType::BF16
            && self.top_k == 8
            && matches!(self.gate_up.dim(0)?, 128 | 256)
            && width.is_multiple_of(8)
            && self.down.dim(2)?.is_multiple_of(8)
        {
            return crate::layers::qwen3_moe::experts(
                &flat,
                &logits,
                &self.gate_up,
                &self.down,
                self.renormalize,
            )?
            .reshape(shape);
        }
        // Reference path for CPU/Metal, other dtypes and nonstandard expert counts.
        // CUDA BF16 checkpoints with 128 experts/top-8 use the device-only path above.
        let probabilities = candle_nn::ops::softmax_last_dim(&logits)?.to_vec2::<f32>()?;
        let experts = self.gate_up.dim(0)?;
        let mut tokens = vec![Vec::<u32>::new(); experts];
        let mut weights = vec![Vec::<f32>::new(); experts];
        for (token, row) in probabilities.iter().enumerate() {
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            order.truncate(self.top_k);
            let divisor = if self.renormalize {
                order.iter().map(|&i| row[i]).sum()
            } else {
                1.
            };
            for index in order {
                tokens[index].push(token as u32);
                weights[index].push(row[index] / divisor);
            }
        }
        let mut output = flat.zeros_like()?;
        for index in 0..experts {
            if tokens[index].is_empty() {
                continue;
            }
            let ids = Tensor::new(tokens[index].as_slice(), hidden.device())?;
            let input = flat.index_select(&ids, 0)?;
            let gate_up = Linear::new(self.gate_up.i(index)?, None, None).forward(&input)?;
            let activated = gated_activation(&gate_up, Some(&HiddenAct::Silu))?;
            let projected = Linear::new(self.down.i(index)?, None, None).forward(&activated)?;
            let weight = Tensor::new(weights[index].as_slice(), hidden.device())?
                .to_dtype(hidden.dtype())?
                .unsqueeze(1)?;
            output = output.index_add(&ids, &projected.broadcast_mul(&weight)?, 0)?;
        }
        output.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use std::collections::HashMap;

    fn config() -> Qwen3Config {
        serde_json::from_value(serde_json::json!({
            "attention_bias":false, "vocab_size":8, "head_dim":2,
            "hidden_size":2, "intermediate_size":4, "num_hidden_layers":4,
            "num_attention_heads":1, "num_key_value_heads":1, "hidden_act":"silu",
            "max_position_embeddings":32, "rms_norm_eps":0.000001,
            "rope_theta":10000.0, "use_sliding_window":false, "eos_token_id":7,
            "num_experts":2, "num_experts_per_tok":1, "moe_intermediate_size":2
        }))
        .unwrap()
    }

    #[test]
    fn mixed_dense_sparse_schedule_and_invalid_routing() -> Result<()> {
        let mut cfg = config();
        cfg.decoder_sparse_step = 2;
        cfg.mlp_only_layers = vec![1];
        assert_eq!(
            (0..4)
                .map(|i| cfg.is_moe_layer(i))
                .collect::<Result<Vec<_>>>()?,
            vec![false, false, false, true]
        );
        cfg.decoder_sparse_step = 0;
        assert!(cfg.is_moe_layer(0).is_err());
        cfg.decoder_sparse_step = 1;
        cfg.num_experts_per_tok = 3;
        assert!(cfg.is_moe_layer(0).is_err());
        cfg.num_experts_per_tok = 1;
        cfg.quantization_config = Some(serde_json::json!({"quant_method":"fp8"}));
        assert!(cfg.is_moe_layer(0).is_err());
        cfg.quantization_config = None;
        cfg.rope_scaling = Some(serde_json::json!({"rope_type":"yarn", "factor":4.0}));
        assert!(cfg.is_moe_layer(0).is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_layouts_and_routing_normalization() -> Result<()> {
        let device = Device::Cpu;
        let identity = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &device)?;
        let router = Tensor::from_vec(vec![1f32, 0., -1., 0.], (2, 2), &device)?;
        let input = Tensor::from_vec(vec![1f32, 2., -1., 2.], (1, 2, 2), &device)?;
        for fused in [false, true] {
            for renormalize in [false, true] {
                let mut cfg = config();
                cfg.norm_topk_prob = renormalize;
                let mut tensors = HashMap::new();
                tensors.insert("gate.weight".into(), router.clone());
                let gate_up = Tensor::cat(&[&identity, &identity], 0)?;
                let down_two = identity.affine(2., 0.)?;
                if fused {
                    tensors.insert(
                        "experts.gate_up_proj".into(),
                        Tensor::stack(&[&gate_up, &gate_up], 0)?,
                    );
                    tensors.insert(
                        "experts.down_proj".into(),
                        Tensor::stack(&[&identity, &down_two], 0)?,
                    );
                } else {
                    for expert in 0..2 {
                        for name in ["gate_proj", "up_proj"] {
                            tensors.insert(
                                format!("experts.{expert}.{name}.weight"),
                                identity.clone(),
                            );
                        }
                        tensors.insert(
                            format!("experts.{expert}.down_proj.weight"),
                            if expert == 0 {
                                identity.clone()
                            } else {
                                down_two.clone()
                            },
                        );
                    }
                }
                let model =
                    Qwen3Moe::load(VarBuilder::from_tensors(tensors, DType::F32, &device), &cfg)?;
                let out = model.forward(&input)?;
                assert_eq!(out.dims(), input.dims());
                let values = out.flatten_all()?.to_vec1::<f32>()?;
                for (index, x) in [1f32, 2., -1., 2.].into_iter().enumerate() {
                    let route = if renormalize {
                        1.
                    } else {
                        1. / (1. + (-2f32).exp())
                    };
                    let expert = if index < 2 { 1. } else { 2. };
                    let expected = x / (1. + (-x).exp()) * x * expert * route;
                    assert!(
                        (values[index] - expected).abs() < 0.00001,
                        "{fused}/{renormalize}: {} != {expected}",
                        values[index]
                    );
                }
            }
        }
        Ok(())
    }
}
