// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Clef joint-head weights, immutable attention provider and kernel leaf.

use crate::{
    clef_decision::{next_head_id, ClefConfig, CLEF_WEIGHTS},
    clef_decision_weights::{scalar_bf16, upload_tensor_bf16, visit_joint_tensors},
    fa4::{AttentionSpec, DType, Fa4, Scale, Window},
    infer, start,
    tensor::Bf16,
};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use std::{collections::BTreeMap, mem::ManuallyDrop, path::Path, sync::Arc};

/// One affine LayerNorm with Native epsilon `1e-5`.
pub(crate) struct ClefNorm {
    pub(crate) gamma: cuda_core::DeviceBuffer<Bf16>,
    pub(crate) beta: cuda_core::DeviceBuffer<Bf16>,
}

/// A biasless projection (`Linear(w, name, in, out, false)`).
pub(crate) struct ClefLinear {
    pub(crate) weight: cuda_core::DeviceBuffer<Bf16>,
}

/// A biased projection.
pub(crate) struct ClefBiased {
    pub(crate) weight: cuda_core::DeviceBuffer<Bf16>,
    pub(crate) bias: cuda_core::DeviceBuffer<Bf16>,
}

/// Native `Attention` keeps Q/K/V packed in one `[3*width,width]` matrix.
pub(crate) struct ClefAttention {
    pub(crate) qkv: cuda_core::DeviceBuffer<Bf16>,
    pub(crate) bias: cuda_core::DeviceBuffer<Bf16>,
    pub(crate) out: ClefBiased,
}

pub(crate) struct ClefRouting {
    pub(crate) query: ClefNorm,
    pub(crate) memory: ClefNorm,
    pub(crate) feed: ClefNorm,
    pub(crate) attention: ClefAttention,
    pub(crate) ffn1: ClefBiased,
    pub(crate) ffn2: ClefBiased,
}

pub(crate) struct ClefDecoder {
    pub(crate) n1: ClefNorm,
    pub(crate) n2: ClefNorm,
    pub(crate) n3: ClefNorm,
    pub(crate) self_attn: ClefAttention,
    pub(crate) cross: ClefAttention,
    pub(crate) linear1: ClefBiased,
    pub(crate) linear2: ClefBiased,
}

pub(crate) struct ClefWeights {
    pub(crate) hidden_norm: ClefNorm,
    pub(crate) option_summary_norm: ClefNorm,
    pub(crate) field_norm: ClefNorm,
    pub(crate) option_norm: ClefNorm,
    pub(crate) memory_projection: ClefLinear,
    pub(crate) question_projection: ClefLinear,
    pub(crate) option_question_projection: ClefLinear,
    pub(crate) global_projection: ClefLinear,
    pub(crate) option_context_projection: ClefLinear,
    pub(crate) option_lexical_projection: ClefLinear,
    pub(crate) types: cuda_core::DeviceBuffer<Bf16>,
    pub(crate) scorer1: ClefBiased,
    pub(crate) scorer3: ClefBiased,
    pub(crate) routing: Vec<ClefRouting>,
    pub(crate) decoder: Vec<ClefDecoder>,
    pub(crate) prior_scale: f32,
    pub(crate) joint_scale: f32,
    pub(crate) gate: f32,
}

pub(crate) use crate::clef_decision_kernels::ClefKernels;

pub(crate) struct ClefHead {
    id: u64,
    pub(crate) config: ClefConfig,
    pub(crate) context: Arc<Context>,
    pub(crate) stream: Arc<Stream>,
    pub(crate) fa4: Fa4,
    pub(crate) kernels: ClefKernels,
    weights: Option<ManuallyDrop<ClefWeights>>,
}

fn take_map<T>(map: &mut BTreeMap<String, T>, name: &str) -> Result<T, BackendError> {
    map.remove(name)
        .ok_or_else(|| start(format!("Clef joint head is missing {name}")))
}

fn norm(
    map: &mut BTreeMap<String, cuda_core::DeviceBuffer<Bf16>>,
    name: &str,
) -> Result<ClefNorm, BackendError> {
    Ok(ClefNorm {
        gamma: take_map(map, &format!("{name}.weight"))?,
        beta: take_map(map, &format!("{name}.bias"))?,
    })
}

fn linear(
    map: &mut BTreeMap<String, cuda_core::DeviceBuffer<Bf16>>,
    name: &str,
) -> Result<ClefLinear, BackendError> {
    Ok(ClefLinear {
        weight: take_map(map, &format!("{name}.weight"))?,
    })
}

fn biased(
    map: &mut BTreeMap<String, cuda_core::DeviceBuffer<Bf16>>,
    name: &str,
) -> Result<ClefBiased, BackendError> {
    Ok(ClefBiased {
        weight: take_map(map, &format!("{name}.weight"))?,
        bias: take_map(map, &format!("{name}.bias"))?,
    })
}

fn attention(
    map: &mut BTreeMap<String, cuda_core::DeviceBuffer<Bf16>>,
    name: &str,
) -> Result<ClefAttention, BackendError> {
    Ok(ClefAttention {
        qkv: take_map(map, &format!("{name}.in_proj_weight"))?,
        bias: take_map(map, &format!("{name}.in_proj_bias"))?,
        out: biased(map, &format!("{name}.out_proj"))?,
    })
}

impl ClefHead {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn weights(&self) -> &ClefWeights {
        self.weights
            .as_deref()
            .expect("Clef head weights are retained until drop")
    }

    /// Cold-load all joint-head tensors, the shared FA4 provider and the small
    /// oxide kernel entries. All source tensors are cast to BF16 exactly as
    /// `bridge.cpp::weight_dtype` casts them to the BF16 model dtype.
    pub(crate) fn load(
        path: &Path,
        config: ClefConfig,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        fa4_root: &Path,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(context, stream.context()) {
            return Err(start("Clef stream belongs to another CUDA context"));
        }
        context.bind_to_thread().map_err(infer)?;
        let spec = AttentionSpec {
            heads: config.heads,
            kv_heads: config.heads,
            head_dim: config.head_dim(),
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        if !Fa4::supports(&spec) {
            return Err(start(format!(
                "FA4 has no BF16 noncausal global profile for Clef Q{}xKV{} d{}",
                config.heads,
                config.heads,
                config.head_dim()
            )));
        }
        let fa4 = Fa4::new(context.clone(), fa4_root, spec)?;
        let kernels = ClefKernels::load(context)?;
        let mut map = BTreeMap::new();
        let mut scalars = [None; 3];
        visit_joint_tensors(
            &path.join(CLEF_WEIGHTS),
            &config,
            |name, shape, dtype, bytes| {
                match name {
                    "prior_logit_scale" => scalars[0] = Some(scalar_bf16(dtype, shape, bytes)?),
                    "joint_logit_scale" => scalars[1] = Some(scalar_bf16(dtype, shape, bytes)?),
                    "residual_gate" => scalars[2] = Some(scalar_bf16(dtype, shape, bytes)?),
                    _ => {
                        map.insert(
                            name.to_string(),
                            upload_tensor_bf16(stream, dtype, shape, bytes)?,
                        );
                    }
                }
                Ok(())
            },
        )?;
        let hidden_norm = norm(&mut map, "hidden_norm")?;
        let option_summary_norm = norm(&mut map, "option_summary_norm")?;
        let field_norm = norm(&mut map, "field_norm")?;
        let option_norm = norm(&mut map, "option_norm")?;
        let memory_projection = linear(&mut map, "memory_projection")?;
        let question_projection = linear(&mut map, "question_projection")?;
        let option_question_projection = linear(&mut map, "option_question_projection")?;
        let global_projection = linear(&mut map, "global_projection")?;
        let option_context_projection = linear(&mut map, "option_context_projection")?;
        let option_lexical_projection = linear(&mut map, "option_lexical_projection")?;
        let types = take_map(&mut map, "type_embedding.weight")?;
        let scorer1 = biased(&mut map, "residual_scorer.0")?;
        let scorer3 = biased(&mut map, "residual_scorer.3")?;
        let mut routing = Vec::new();
        for index in 0..config.routing_layers {
            let prefix = format!("evidence_layers.{index}");
            routing.push(ClefRouting {
                query: norm(&mut map, &format!("{prefix}.query_norm"))?,
                memory: norm(&mut map, &format!("{prefix}.memory_norm"))?,
                feed: norm(&mut map, &format!("{prefix}.feedforward_norm"))?,
                attention: attention(&mut map, &format!("{prefix}.attention"))?,
                ffn1: biased(&mut map, &format!("{prefix}.feedforward.0"))?,
                ffn2: biased(&mut map, &format!("{prefix}.feedforward.3"))?,
            });
        }
        let mut decoder = Vec::new();
        for index in 0..config.layers {
            let prefix = format!("layers.{index}");
            decoder.push(ClefDecoder {
                n1: norm(&mut map, &format!("{prefix}.norm1"))?,
                n2: norm(&mut map, &format!("{prefix}.norm2"))?,
                n3: norm(&mut map, &format!("{prefix}.norm3"))?,
                self_attn: attention(&mut map, &format!("{prefix}.self_attn"))?,
                cross: attention(&mut map, &format!("{prefix}.multihead_attn"))?,
                linear1: biased(&mut map, &format!("{prefix}.linear1"))?,
                linear2: biased(&mut map, &format!("{prefix}.linear2"))?,
            });
        }
        if !map.is_empty() {
            return Err(start(format!(
                "Clef joint head contains unexpected tensors: {:?}",
                map.keys().take(4).collect::<Vec<_>>()
            )));
        }
        let scalars = [
            scalars[0].ok_or_else(|| start("missing Clef prior_logit_scale"))?,
            scalars[1].ok_or_else(|| start("missing Clef joint_logit_scale"))?,
            scalars[2].ok_or_else(|| start("missing Clef residual_gate"))?,
        ];
        let prepared_scalars = crate::clef_decision_scalars::prepare(stream, &kernels, scalars)?;
        Ok(Self {
            id: next_head_id(),
            config,
            context: context.clone(),
            stream: stream.clone(),
            fa4,
            kernels,
            weights: Some(ManuallyDrop::new(ClefWeights {
                hidden_norm,
                option_summary_norm,
                field_norm,
                option_norm,
                memory_projection,
                question_projection,
                option_question_projection,
                global_projection,
                option_context_projection,
                option_lexical_projection,
                types,
                scorer1,
                scorer3,
                routing,
                decoder,
                prior_scale: prepared_scalars[0],
                joint_scale: prepared_scalars[1],
                gate: prepared_scalars[2],
            })),
        })
    }
}

impl Drop for ClefHead {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            if let Some(mut weights) = self.weights.take() {
                // SAFETY: the ordered stream drained, so every weight owner is idle.
                unsafe { ManuallyDrop::drop(&mut weights) };
            }
        } else if let Some(weights) = self.weights.take() {
            // Completion is uncertain: retain every device owner.
            #[allow(clippy::forget_non_drop)]
            std::mem::forget(weights);
        }
    }
}
