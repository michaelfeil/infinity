// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact `joint_head.safetensors` schema for the Native Clef head.
//!
//! Shapes come from `decision_models.cpp::ClefHead/Routing/FieldDecoder` and
//! the shared `Linear`/`Norm`/`Attention` helpers. `bridge.cpp::weight_dtype`
//! has no Clef override, so the whole joint head is cast to the backbone model
//! dtype; the public Clef checkpoints are BF16. This module validates metadata
//! without assuming that any tensor payload is available.

use crate::{
    clef_decision::{ClefConfig, CLEF_FIELDS},
    infer, start,
    tensor::Bf16,
};
use cuda_core::{DeviceBuffer, Stream};
use half::bf16 as HalfBf16;
use infinity_v3_backend_core::BackendError;
use safetensors::{tensor::SafeTensors, Dtype};
use std::{path::Path, sync::Arc};

const PROJECTIONS: [&str; 6] = [
    "memory_projection",
    "question_projection",
    "option_question_projection",
    "global_projection",
    "option_context_projection",
    "option_lexical_projection",
];

fn linear_shape(input: usize, output: usize, bias: bool) -> Vec<(String, Vec<usize>)> {
    let mut shapes = vec![("weight".to_string(), vec![output, input])];
    if bias {
        shapes.push(("bias".to_string(), vec![output]));
    }
    shapes
}

fn norm_shape(width: usize) -> Vec<(String, Vec<usize>)> {
    vec![
        ("weight".to_string(), vec![width]),
        ("bias".to_string(), vec![width]),
    ]
}

fn attention_shape(width: usize) -> Vec<(String, Vec<usize>)> {
    vec![
        ("in_proj_weight".to_string(), vec![3 * width, width]),
        ("in_proj_bias".to_string(), vec![3 * width]),
        ("out_proj.weight".to_string(), vec![width, width]),
        ("out_proj.bias".to_string(), vec![width]),
    ]
}

fn prefixed(prefix: &str, entries: Vec<(String, Vec<usize>)>) -> Vec<(String, Vec<usize>)> {
    entries
        .into_iter()
        .map(|(name, shape)| (format!("{prefix}.{name}"), shape))
        .collect()
}

/// Complete Native tensor schema for one validated configuration.
pub(crate) fn tensor_shapes(config: &ClefConfig) -> Vec<(String, Vec<usize>)> {
    let (h, w, ff) = (config.hidden, config.width, config.feedforward);
    let mut tensors = Vec::new();
    for name in [
        "hidden_norm",
        "option_summary_norm",
        "field_norm",
        "option_norm",
    ] {
        let width = if name == "hidden_norm" { h } else { w };
        tensors.extend(prefixed(name, norm_shape(width)));
    }
    for name in PROJECTIONS {
        tensors.extend(prefixed(name, linear_shape(h, w, false)));
    }
    tensors.push(("type_embedding.weight".into(), vec![CLEF_FIELDS, w]));
    tensors.extend(prefixed("residual_scorer.0", linear_shape(4 * w, w, true)));
    tensors.extend(prefixed("residual_scorer.3", linear_shape(w, 1, true)));
    for name in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
        tensors.push((name.to_string(), Vec::new()));
    }
    for index in 0..config.routing_layers {
        let prefix = format!("evidence_layers.{index}");
        for name in ["query_norm", "memory_norm", "feedforward_norm"] {
            tensors.extend(prefixed(&format!("{prefix}.{name}"), norm_shape(w)));
        }
        tensors.extend(prefixed(&format!("{prefix}.attention"), attention_shape(w)));
        tensors.extend(prefixed(
            &format!("{prefix}.feedforward.0"),
            linear_shape(w, ff, true),
        ));
        tensors.extend(prefixed(
            &format!("{prefix}.feedforward.3"),
            linear_shape(ff, w, true),
        ));
    }
    for index in 0..config.layers {
        let prefix = format!("layers.{index}");
        for name in ["norm1", "norm2", "norm3"] {
            tensors.extend(prefixed(&format!("{prefix}.{name}"), norm_shape(w)));
        }
        tensors.extend(prefixed(&format!("{prefix}.self_attn"), attention_shape(w)));
        tensors.extend(prefixed(
            &format!("{prefix}.multihead_attn"),
            attention_shape(w),
        ));
        tensors.extend(prefixed(
            &format!("{prefix}.linear1"),
            linear_shape(w, ff, true),
        ));
        tensors.extend(prefixed(
            &format!("{prefix}.linear2"),
            linear_shape(ff, w, true),
        ));
    }
    tensors
}

/// Visit every tensor in the joint shard exactly once, with its declared shape
/// and payload bytes. No global tensor map or full-file tensor cache is kept.
pub(crate) fn visit_joint_tensors(
    path: &Path,
    config: &ClefConfig,
    mut visit: impl FnMut(&str, &[usize], Dtype, &[u8]) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let bytes = std::fs::read(path).map_err(|error| {
        start(format!(
            "Clef joint_head.safetensors read failed at {}: {error}",
            path.display()
        ))
    })?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| start(format!("invalid Clef joint_head.safetensors: {error}")))?;
    for (name, shape) in tensor_shapes(config) {
        let tensor = tensors.tensor(&name).map_err(|error| {
            start(format!(
                "Clef joint head is missing tensor {name} or its header is invalid: {error}"
            ))
        })?;
        if tensor.shape() != shape.as_slice() {
            return Err(start(format!(
                "Clef tensor {name} shape {:?} != native {shape:?}",
                tensor.shape()
            )));
        }
        if !matches!(tensor.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            return Err(start(format!(
                "Clef tensor {name} has unsupported dtype {:?}",
                tensor.dtype()
            )));
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .and_then(|count| count.checked_mul(if tensor.dtype() == Dtype::F32 { 4 } else { 2 }))
            .ok_or_else(|| infer(format!("Clef tensor {name} byte size overflow")))?;
        if expected != tensor.data().len() {
            return Err(start(format!(
                "Clef tensor {name} payload {} != expected {expected}",
                tensor.data().len()
            )));
        }
        visit(&name, &shape, tensor.dtype(), tensor.data())?;
    }
    Ok(())
}

/// Validate every required name, shape, dtype and byte length with no device
/// allocation. The caller still needs the backbone `lm_head.weight`/tied
/// embedding for the lexical prior.
pub(crate) fn validate_joint_sources(path: &Path, config: &ClefConfig) -> Result<(), BackendError> {
    visit_joint_tensors(path, config, |_, _, _, _| Ok(()))
}

fn convert_bf16(dtype: Dtype, bytes: &[u8], out: &mut [Bf16]) -> Result<(), BackendError> {
    match dtype {
        Dtype::BF16 => {
            for (destination, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *destination = Bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
            }
        }
        Dtype::F16 => {
            for (destination, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let value = half::f16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
                *destination = Bf16::from_bits(HalfBf16::from_f32(value.to_f32()).to_bits());
            }
        }
        Dtype::F32 => {
            for (destination, word) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                *destination = Bf16::from_bits(HalfBf16::from_f32(value).to_bits());
            }
        }
        other => return Err(infer(format!("unsupported Clef weight dtype {other:?}"))),
    }
    Ok(())
}

/// Upload one cold tensor as BF16 storage, casting F32/F16 exactly as the
/// Native `to(model.dtype)` weight path does. Cold construction only.
pub(crate) fn upload_tensor_bf16(
    stream: &Arc<Stream>,
    dtype: Dtype,
    shape: &[usize],
    bytes: &[u8],
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| infer("Clef tensor element count overflow"))?;
    let mut host = Vec::new();
    host.try_reserve_exact(count)
        .map_err(|error| infer(format!("Clef tensor host allocation failed: {error}")))?;
    host.resize(count, Bf16::from_bits(0));
    convert_bf16(dtype, bytes, &mut host)?;
    DeviceBuffer::from_host(stream, &host).map_err(infer)
}

/// Read one rank-0 source scalar and apply the Native first boundary:
/// cast the original F32/F16/BF16 payload to the model BF16 dtype. Exp/sigmoid
/// belong to the cold scalar kernel, not this parser.
pub(crate) fn scalar_bf16(
    dtype: Dtype,
    shape: &[usize],
    bytes: &[u8],
) -> Result<Bf16, BackendError> {
    if !shape.is_empty() {
        return Err(start("Clef scalar tensor must have rank 0"));
    }
    let expected = match dtype {
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F32 => 4,
        other => return Err(infer(format!("unsupported Clef scalar dtype {other:?}"))),
    };
    if bytes.len() != expected {
        return Err(start(format!(
            "Clef scalar payload has {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    let mut out = [Bf16::from_bits(0)];
    convert_bf16(dtype, bytes, &mut out)?;
    let value = HalfBf16::from_bits(out[0].to_bits()).to_f32();
    if !value.is_finite() {
        return Err(infer("Clef scalar weight is not finite after BF16 cast"));
    }
    Ok(out[0])
}

#[cfg(test)]
#[path = "clef_decision_weights_tests.rs"]
mod tests;
