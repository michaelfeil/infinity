// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Test-only scalar oracle for the dense Gemma3 text block.
//!
//! This module is compiled only for tests. It is never selected by the Oxide
//! factory and never runs on any request path. Its purpose is to encode the
//! Native `gemma_models.cpp` layer order, offset norms, local/global windows and
//! gated tanh-GELU without CUDA so host fixtures can exercise real math.

use crate::gemma3_model::{Gemma3Config, Gemma3LayerPolicy};

#[derive(Clone, Debug)]
pub(crate) struct ReferenceLayer {
    pub(crate) input_norm: Vec<f32>,
    pub(crate) post_attention_norm: Vec<f32>,
    pub(crate) pre_feedforward_norm: Vec<f32>,
    pub(crate) post_feedforward_norm: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) o: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) down: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct ReferenceModel {
    pub(crate) config: Gemma3Config,
    pub(crate) embed: Vec<f32>,
    pub(crate) norm: Vec<f32>,
    pub(crate) layers: Vec<ReferenceLayer>,
}

pub(crate) fn embedding_scale(config: &Gemma3Config) -> f32 {
    half::bf16::from_f32((config.hidden_size as f32).sqrt()).to_f32()
}

fn rms_norm(x: &[f32], offset_weight: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), offset_weight.len());
    let sum = x.iter().map(|value| value * value).sum::<f32>();
    let inverse = 1.0f32 / (sum / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(offset_weight)
        .map(|(value, scale)| (value * inverse) * (scale + 1.0))
        .collect()
}

fn residual_norm(residual: &[f32], x: &[f32], offset_weight: &[f32], eps: f32) -> Vec<f32> {
    let normalized = rms_norm(x, offset_weight, eps);
    residual
        .iter()
        .zip(normalized)
        .map(|(residual, value)| residual + value)
        .collect()
}

fn linear(x: &[f32], weight: &[f32], out: usize) -> Vec<f32> {
    debug_assert_eq!(weight.len(), out * x.len());
    (0..out)
        .map(|row| {
            weight[row * x.len()..(row + 1) * x.len()]
                .iter()
                .zip(x)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect()
}

fn gelu_tanh(value: f32) -> f32 {
    let cube = (value * value) * value;
    let alpha = value + 0.044715 * cube;
    let tangent = (0.797_884_6 * alpha).tanh();
    0.5 * value * (1.0 + tangent)
}

fn rope_split_half(values: &mut [f32], position: usize, theta: f64, linear: f32) {
    let dim = values.len();
    let half = dim / 2;
    for index in 0..half {
        let exponent = (index * 2) as f32 / dim as f32;
        let inverse = (1.0f32 / (theta as f32).powf(exponent)) / linear;
        let phase = position as f32 * inverse;
        let (sine, cosine) = phase.sin_cos();
        let a = values[index];
        let b = values[index + half];
        values[index] = a * cosine - b * sine;
        values[index + half] = b * cosine + a * sine;
    }
}

/// Native causal/local visibility for one query/key pair.
pub(crate) fn visible(policy: &Gemma3LayerPolicy, causal: bool, query: usize, key: usize) -> bool {
    if key > query && causal {
        return false;
    }
    if !policy.local {
        return true;
    }
    if causal {
        let distance = query.saturating_sub(key);
        distance as i64 <= policy.window_left as i64
    } else {
        let delta = query as i64 - key as i64;
        delta <= policy.window_right as i64 && delta >= -(policy.window_left as i64)
    }
}

impl ReferenceModel {
    pub(crate) fn forward(&self, ids: &[usize], positions: &[usize]) -> Vec<Vec<f32>> {
        let config = &self.config;
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let group = q_heads / kv_heads;
        let scale = embedding_scale(config);
        let mut states = ids
            .iter()
            .map(|id| {
                self.embed[id * hidden..(id + 1) * hidden]
                    .iter()
                    .map(|value| value * scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (index, layer) in self.layers.iter().enumerate() {
            let policy = config.layer_policy(index).expect("layer policy");
            let mut normalized = states
                .iter()
                .map(|state| rms_norm(state, &layer.input_norm, config.rms_norm_eps as f32))
                .collect::<Vec<_>>();
            let mut q = Vec::with_capacity(ids.len());
            let mut k = Vec::with_capacity(ids.len());
            let mut v = Vec::with_capacity(ids.len());
            for token in 0..ids.len() {
                let mut qh = linear(&normalized[token], &layer.q, q_heads * head_dim);
                let mut kh = linear(&normalized[token], &layer.k, kv_heads * head_dim);
                let vh = linear(&normalized[token], &layer.v, kv_heads * head_dim);
                for head in 0..q_heads {
                    let row = &mut qh[head * head_dim..(head + 1) * head_dim];
                    row.copy_from_slice(&rms_norm(row, &layer.q_norm, config.rms_norm_eps as f32));
                    rope_split_half(row, positions[token], policy.theta, policy.rope_factor);
                }
                for head in 0..kv_heads {
                    let row = &mut kh[head * head_dim..(head + 1) * head_dim];
                    row.copy_from_slice(&rms_norm(row, &layer.k_norm, config.rms_norm_eps as f32));
                    rope_split_half(row, positions[token], policy.theta, policy.rope_factor);
                }
                q.push(qh);
                k.push(kh);
                v.push(vh);
            }
            let mut attention = vec![0.0f32; ids.len() * q_heads * head_dim];
            for query in 0..ids.len() {
                for head in 0..q_heads {
                    let kv_head = head / group;
                    let mut logits = Vec::new();
                    let mut keys = Vec::new();
                    for (key, key_rows) in k.iter().enumerate().take(ids.len()) {
                        if !visible(&policy, config.is_causal, query, key) {
                            continue;
                        }
                        let dot = (0..head_dim)
                            .map(|lane| {
                                q[query][head * head_dim + lane]
                                    * key_rows[kv_head * head_dim + lane]
                            })
                            .sum::<f32>();
                        logits.push(dot * policy.attention_scale as f32);
                        keys.push(key);
                    }
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = logits
                        .iter()
                        .map(|value| (value - max).exp())
                        .collect::<Vec<_>>();
                    let total = weights.iter().sum::<f32>();
                    for lane in 0..head_dim {
                        let value = keys
                            .iter()
                            .zip(&weights)
                            .map(|(key, weight)| {
                                weight / total * v[*key][kv_head * head_dim + lane]
                            })
                            .sum::<f32>();
                        attention[query * q_heads * head_dim + head * head_dim + lane] = value;
                    }
                }
            }
            for token in 0..ids.len() {
                let projected = linear(
                    &attention[token * q_heads * head_dim..(token + 1) * q_heads * head_dim],
                    &layer.o,
                    hidden,
                );
                states[token] = residual_norm(
                    &states[token],
                    &projected,
                    &layer.post_attention_norm,
                    config.rms_norm_eps as f32,
                );
                normalized[token] = rms_norm(
                    &states[token],
                    &layer.pre_feedforward_norm,
                    config.rms_norm_eps as f32,
                );
                let gate = linear(&normalized[token], &layer.gate, config.intermediate_size);
                let up = linear(&normalized[token], &layer.up, config.intermediate_size);
                let activated = gate
                    .iter()
                    .zip(&up)
                    .map(|(gate, up)| gelu_tanh(*gate) * up)
                    .collect::<Vec<_>>();
                let down = linear(&activated, &layer.down, hidden);
                states[token] = residual_norm(
                    &states[token],
                    &down,
                    &layer.post_feedforward_norm,
                    config.rms_norm_eps as f32,
                );
            }
        }
        states
            .iter()
            .map(|state| rms_norm(state, &self.norm, config.rms_norm_eps as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            vocab_size: 5,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 100.0,
            sliding_window: 2,
            sliding_window_pattern: 2,
            query_pre_attn_scalar: 2.0,
            is_causal: true,
            global_rope_factor: 1.0,
        }
    }

    fn zeros_model() -> ReferenceModel {
        let config = tiny_config();
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let d = config.head_dim;
        let layer = ReferenceLayer {
            input_norm: vec![0.0; h],
            post_attention_norm: vec![0.0; h],
            pre_feedforward_norm: vec![0.0; h],
            post_feedforward_norm: vec![0.0; h],
            q_norm: vec![0.0; d],
            k_norm: vec![0.0; d],
            q: vec![0.0; q * h],
            k: vec![0.0; kv * h],
            v: vec![0.0; kv * h],
            o: vec![0.0; h * q],
            gate: vec![0.0; i * h],
            up: vec![0.0; i * h],
            down: vec![0.0; h * i],
        };
        ReferenceModel {
            config,
            embed: vec![0.0; 5 * h],
            norm: vec![0.0; h],
            layers: vec![layer.clone(), layer],
        }
    }

    #[test]
    fn offset_norm_matches_known_legacy_values() {
        let actual = rms_norm(&[3.0, 4.0], &[0.0, 0.0], 0.0);
        let root = 12.5f32.sqrt();
        assert!((actual[0] - 3.0 / root).abs() < 1e-6);
        assert!((actual[1] - 4.0 / root).abs() < 1e-6);
        let shifted = rms_norm(&[3.0, 4.0], &[1.0, 0.0], 0.0);
        assert!((shifted[0] - 6.0 / root).abs() < 1e-6);
        assert!((shifted[1] - 4.0 / root).abs() < 1e-6);
    }

    #[test]
    fn embedding_scale_is_native_bf16_rounding() {
        let config = tiny_config();
        assert_eq!(embedding_scale(&config), 2.0);
        let mut big = config;
        big.hidden_size = 3840;
        assert_eq!(
            embedding_scale(&big).to_bits(),
            half::bf16::from_f32(3840f32.sqrt()).to_f32().to_bits()
        );
    }

    #[test]
    fn local_and_global_visibility_follow_native_pattern() {
        let config = tiny_config();
        let local = config.layer_policy(0).unwrap();
        assert!(local.local);
        assert!(visible(&local, true, 1, 0));
        assert!(!visible(&local, true, 4, 0));
        assert!(!visible(&local, true, 0, 1));
        let global = config.layer_policy(1).unwrap();
        assert!(!global.local);
        assert!(visible(&global, true, 4, 0));
        assert!(!visible(&global, true, 0, 4));
    }

    #[test]
    fn nonlinear_reference_is_causal_and_state_stable() {
        let mut model = zeros_model();
        for (position, value) in model.embed.iter_mut().enumerate() {
            *value = (position as f32 - 9.0) * 0.03125;
        }
        for layer in &mut model.layers {
            for (index, weight) in layer
                .q
                .iter_mut()
                .chain(&mut layer.k)
                .chain(&mut layer.v)
                .chain(&mut layer.o)
                .chain(&mut layer.gate)
                .chain(&mut layer.up)
                .chain(&mut layer.down)
                .enumerate()
            {
                *weight = ((index % 11) as f32 - 5.0) * 0.015625;
            }
            for weight in layer
                .input_norm
                .iter_mut()
                .chain(&mut layer.post_attention_norm)
                .chain(&mut layer.pre_feedforward_norm)
                .chain(&mut layer.post_feedforward_norm)
                .chain(&mut layer.q_norm)
                .chain(&mut layer.k_norm)
            {
                *weight = 0.125;
            }
        }
        let prefix = model.forward(&[1, 2, 3], &[0, 1, 2]);
        let full = model.forward(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        for (left, right) in prefix.iter().zip(&full) {
            for (a, b) in left.iter().zip(right) {
                assert!((a - b).abs() < 1e-6);
            }
        }
        let mut changed_future = model.clone();
        changed_future.embed[4 * 4..5 * 4].fill(5.0);
        let changed = changed_future.forward(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        for (left, right) in prefix.iter().zip(&changed) {
            for (a, b) in left.iter().zip(right) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }
}
