// Shared Qwen3.5 backbone typed weights for dense GDN and MoE FFN variants.
//
// The backbone owns embeddings, final norm, per-layer RMSNorm offsets, the
// existing typed full/linear attention weight enum, and one FFN enum. Dense
// loading delegates to the qualified donor `Qwen35Weights::load`, so dense
// attention/GDN binding is not duplicated. MoE loading uses the v3 cold MoE
// manifest and binds the same attention tensors without requiring any dense
// MLP gate/up/down weights.

use crate::model::{start, Checkpoint, WeightSource};
use crate::qwen35_config::{Qwen35Config, Qwen35LayerType};
use crate::qwen35_model::{Qwen35AttentionWeights, Qwen35Weights};
use crate::qwen35_moe_config::Qwen35MoeConfig;
use crate::qwen35_moe_weights::{Qwen35MoeWeights, RoutedLayerWeights, SharedExpertSources};
use infinity_v3_backend_core::BackendError;
use std::path::Path;

const DELTA_HEAD_DIM: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35BackboneKind {
    Dense,
    Moe,
}

#[derive(Clone, Debug)]
pub(crate) struct Qwen35BackboneGeometry {
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) layer_types: Vec<Qwen35LayerType>,
}

impl Qwen35BackboneGeometry {
    pub(crate) fn dense(config: &Qwen35Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
            layer_types: config.layer_types.clone(),
        }
    }

    pub(crate) fn moe(config: &Qwen35MoeConfig) -> Self {
        Self {
            hidden_size: config.text.hidden_size,
            num_hidden_layers: config.text.num_hidden_layers,
            num_attention_heads: config.text.num_attention_heads,
            num_key_value_heads: config.text.num_key_value_heads,
            head_dim: config.text.head_dim,
            vocab_size: config.text.vocab_size,
            linear_num_key_heads: config.text.linear_num_key_heads,
            linear_num_value_heads: config.text.linear_num_value_heads,
            linear_key_head_dim: config.text.linear_key_head_dim,
            linear_value_head_dim: config.text.linear_value_head_dim,
            linear_conv_kernel_dim: config.text.linear_conv_kernel_dim,
            layer_types: config.text.layer_types.clone(),
        }
    }

    fn q_width(&self) -> Result<usize, BackendError> {
        checked(self.num_attention_heads, self.head_dim, "query width")
    }

    fn kv_width(&self) -> Result<usize, BackendError> {
        checked(self.num_key_value_heads, self.head_dim, "key/value width")
    }

    pub(crate) fn linear_widths(&self) -> Result<(usize, usize, usize), BackendError> {
        let key = checked(
            self.linear_num_key_heads,
            self.linear_key_head_dim,
            "linear key width",
        )?;
        let value = checked(
            self.linear_num_value_heads,
            self.linear_value_head_dim,
            "linear value width",
        )?;
        let qkv = checked(key, 2, "linear key pair width")?
            .checked_add(value)
            .ok_or_else(|| start("linear QKV width overflow"))?;
        Ok((key, value, qkv))
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Qwen35BackboneFfn {
    Dense {
        gate: WeightSource,
        up: WeightSource,
        down: WeightSource,
        intermediate_size: usize,
    },
    Routed {
        routed: RoutedLayerWeights,
        shared: SharedExpertSources,
        intermediate_size: usize,
        shared_intermediate_size: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Qwen35BackboneLayer {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) attention: Qwen35AttentionWeights,
    pub(crate) ffn: Qwen35BackboneFfn,
}

#[derive(Clone, Debug)]
pub(crate) struct Qwen35BackboneWeights {
    pub(crate) geometry: Qwen35BackboneGeometry,
    pub(crate) kind: Qwen35BackboneKind,
    pub(crate) embeddings: WeightSource,
    pub(crate) norm: WeightSource,
    pub(crate) layers: Vec<Qwen35BackboneLayer>,
    pub(crate) lm_head: Option<WeightSource>,
    pub(crate) root: String,
}

impl Qwen35BackboneWeights {
    pub(crate) fn load_dense(path: &Path) -> Result<Self, BackendError> {
        let dense = Qwen35Weights::load(path)?;
        let geometry = Qwen35BackboneGeometry::dense(&dense.config);
        let intermediate_size = dense.config.intermediate_size;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(dense.layers.len())
            .map_err(start)?;
        for layer in dense.layers {
            layers.push(Qwen35BackboneLayer {
                input_norm: layer.input_norm,
                post_attention_norm: layer.post_attention_norm,
                attention: layer.attention,
                ffn: Qwen35BackboneFfn::Dense {
                    gate: layer.mlp_gate,
                    up: layer.mlp_up,
                    down: layer.mlp_down,
                    intermediate_size,
                },
            });
        }
        Ok(Self {
            geometry,
            kind: Qwen35BackboneKind::Dense,
            embeddings: dense.embeddings,
            norm: dense.norm,
            layers,
            lm_head: dense.lm_head,
            root: dense.root,
        })
    }

    pub(crate) fn load_moe(path: &Path) -> Result<Self, BackendError> {
        let moe = Qwen35MoeWeights::load(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let geometry = Qwen35BackboneGeometry::moe(&moe.config);
        let intermediate_size = moe.config.moe.moe_intermediate_size;
        let shared_intermediate_size = moe.config.moe.shared_expert_intermediate_size;
        let mut layers = Vec::new();
        layers.try_reserve_exact(moe.layers.len()).map_err(start)?;
        for layer in moe.layers {
            let attention = bind_attention(
                &checkpoint,
                &layer.attention_prefix,
                &geometry,
                geometry.layer_types[layer.index],
            )?;
            layers.push(Qwen35BackboneLayer {
                input_norm: layer.input_norm,
                post_attention_norm: layer.post_attention_norm,
                attention,
                ffn: Qwen35BackboneFfn::Routed {
                    routed: layer.routed,
                    shared: layer.shared,
                    intermediate_size,
                    shared_intermediate_size,
                },
            });
        }
        Ok(Self {
            geometry,
            kind: Qwen35BackboneKind::Moe,
            embeddings: moe.embeddings,
            norm: moe.norm,
            layers,
            lm_head: moe.lm_head,
            root: moe.root,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), BackendError> {
        if self.layers.len() != self.geometry.num_hidden_layers {
            return Err(start("Qwen3.5 backbone layer count mismatch"));
        }
        for (index, (layer, kind)) in self
            .layers
            .iter()
            .zip(self.geometry.layer_types.iter())
            .enumerate()
        {
            match (kind, &layer.attention) {
                (Qwen35LayerType::Full, Qwen35AttentionWeights::Full { .. })
                | (Qwen35LayerType::Linear, Qwen35AttentionWeights::Linear { .. }) => {}
                _ => {
                    return Err(start(format!(
                        "Qwen3.5 backbone layer {index} attention kind mismatch"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn bind_attention(
    checkpoint: &Checkpoint,
    prefix: &str,
    geometry: &Qwen35BackboneGeometry,
    kind: Qwen35LayerType,
) -> Result<Qwen35AttentionWeights, BackendError> {
    let hidden = geometry.hidden_size;
    let q = geometry.q_width()?;
    let kv = geometry.kv_width()?;
    match kind {
        Qwen35LayerType::Full => Ok(Qwen35AttentionWeights::Full {
            query_gate: checkpoint.source(
                &format!("{prefix}self_attn.q_proj.weight"),
                &[checked(q, 2, "query-gate width")?, hidden],
            )?,
            key: checkpoint.source(&format!("{prefix}self_attn.k_proj.weight"), &[kv, hidden])?,
            value: checkpoint.source(&format!("{prefix}self_attn.v_proj.weight"), &[kv, hidden])?,
            output: checkpoint.source(&format!("{prefix}self_attn.o_proj.weight"), &[hidden, q])?,
            q_norm: checkpoint.source(
                &format!("{prefix}self_attn.q_norm.weight"),
                &[geometry.head_dim],
            )?,
            k_norm: checkpoint.source(
                &format!("{prefix}self_attn.k_norm.weight"),
                &[geometry.head_dim],
            )?,
        }),
        Qwen35LayerType::Linear => {
            let (_, value_width, qkv_width) = geometry.linear_widths()?;
            Ok(Qwen35AttentionWeights::Linear {
                qkv: checkpoint.source(
                    &format!("{prefix}linear_attn.in_proj_qkv.weight"),
                    &[qkv_width, hidden],
                )?,
                z: checkpoint.source(
                    &format!("{prefix}linear_attn.in_proj_z.weight"),
                    &[value_width, hidden],
                )?,
                a: checkpoint.source(
                    &format!("{prefix}linear_attn.in_proj_a.weight"),
                    &[geometry.linear_num_value_heads, hidden],
                )?,
                b: checkpoint.source(
                    &format!("{prefix}linear_attn.in_proj_b.weight"),
                    &[geometry.linear_num_value_heads, hidden],
                )?,
                output: checkpoint.source(
                    &format!("{prefix}linear_attn.out_proj.weight"),
                    &[hidden, value_width],
                )?,
                conv: checkpoint.source(
                    &format!("{prefix}linear_attn.conv1d.weight"),
                    &[qkv_width, 1, geometry.linear_conv_kernel_dim],
                )?,
                a_log: checkpoint.source(
                    &format!("{prefix}linear_attn.A_log"),
                    &[geometry.linear_num_value_heads],
                )?,
                dt_bias: checkpoint.source(
                    &format!("{prefix}linear_attn.dt_bias"),
                    &[geometry.linear_num_value_heads],
                )?,
                norm: checkpoint.source(
                    &format!("{prefix}linear_attn.norm.weight"),
                    &[DELTA_HEAD_DIM],
                )?,
            })
        }
    }
}

fn checked(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}
