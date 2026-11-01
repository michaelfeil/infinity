// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! MPNet family configuration, checkpoint adapter and relative-bias math.
//!
//! MPNet's encoder is a post-norm transformer with absolute learned position
//! embeddings and a learned symmetric 32-bucket relative attention bias. The
//! checkpoint has no token-type embedding table and uses `mpnet.` or bare
//! encoder namespaces. This module keeps those rules in one place. The bucket
//! derivation follows the staged native `Mpnet::position_bias` implementation,
//! which follows Hugging Face Transformers `modeling_mpnet.py`:
//! Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
//! Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//! Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0

use crate::model::{
    elements, required_usize, select_root, BertConfig, BertWeights, Checkpoint, EncoderFamily,
    EncoderRecipe, FeedForwardRecipe, LayerWeights, LinearWeights, NormArithmetic, PositionRecipe,
    RawEncoderConfig, WeightSource,
};
use crate::start;
use infinity_v3_backend_core::BackendError;
use std::collections::BTreeSet;
use std::path::Path;

pub(crate) const BUCKETS: usize = 32;
pub(crate) const POSITION_OFFSET: usize = 2;
#[cfg_attr(not(test), allow(dead_code))]
const MAX_EXACT: u64 = 8;

/// Parse and validate the actual MPNet config fields. Native MPNet always uses
/// 32 symmetric buckets, has no token-type embeddings, and uses exact erf GELU.
pub(crate) fn config(raw: &RawEncoderConfig) -> Result<BertConfig, BackendError> {
    if raw.model_type != "mpnet" {
        return Err(start("MPNet config requires model_type mpnet"));
    }
    let accepted = ["MPNetModel", "MPNetForMaskedLM", "MPNetForPreTraining"];
    if let Some(architectures) = raw.architectures.as_deref() {
        if let Some(architecture) = architectures
            .iter()
            .find(|architecture| !accepted.contains(&architecture.as_str()))
        {
            return Err(start(format!(
                "unsupported MPNet architecture {architecture}"
            )));
        }
    }
    if let Some(value) = raw.type_vocab_size {
        if value != 0 {
            return Err(start(format!(
                "MPNet has no token-type embeddings; type_vocab_size must be 0, got {value}"
            )));
        }
    }
    let buckets = raw.relative_attention_num_buckets.unwrap_or(BUCKETS);
    if buckets != BUCKETS {
        return Err(start(format!(
            "MPNet requires exactly {BUCKETS} relative attention buckets, got {buckets}"
        )));
    }
    let hidden_act = raw.hidden_act.as_deref().unwrap_or("gelu");
    if hidden_act != "gelu" {
        return Err(start(format!(
            "MPNet requires exact gelu activation, got {hidden_act}"
        )));
    }
    let position_embedding_type = raw.position_embedding_type.as_deref().unwrap_or("absolute");
    if position_embedding_type != "absolute" {
        return Err(start(format!(
            "MPNet requires absolute position embeddings, got {position_embedding_type}"
        )));
    }
    if raw.is_decoder == Some(true) || raw.add_cross_attention == Some(true) {
        return Err(start(
            "MPNet encoder does not support decoder or cross-attention modes",
        ));
    }
    let hidden = required_usize(raw.hidden_size, "hidden_size")?;
    let intermediate = required_usize(raw.intermediate_size, "intermediate_size")?;
    let heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
    let layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
    let max_position = required_usize(raw.max_position_embeddings, "max_position_embeddings")?;
    let vocab = required_usize(raw.vocab_size, "vocab_size")?;
    if max_position < POSITION_OFFSET + 1 {
        return Err(start(format!(
            "MPNet max_position_embeddings must leave room for the +{POSITION_OFFSET} lookup offset"
        )));
    }
    // Native MPNet uses plain model-dtype tensor adds before `at::layer_norm`,
    // so word+position and projection+residual both round to BF16 before the
    // statistics kernel. This is the rounded recipe, not the specialized BERT
    // exact-residual path.
    let config = BertConfig {
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_attention_heads: heads,
        num_hidden_layers: layers,
        max_position_embeddings: max_position,
        // MPNet uses the learned absolute position table only; there is no
        // rotary table in the native checkpoint or recipe.
        rotary_table_positions: None,
        architectures: raw.architectures.clone().unwrap_or_default(),
        vocab_size: vocab,
        type_vocab_size: 1,
        configured_type_vocab_size: raw.type_vocab_size,
        layer_norm_eps: raw.layer_norm_eps.unwrap_or(1e-12),
        family: EncoderFamily::Mpnet,
        recipe: EncoderRecipe {
            embedding: NormArithmetic::RoundedAddBeforeStats,
            residual: NormArithmetic::RoundedAddBeforeStats,
            positions: PositionRecipe::LearnedRelative { buckets },
            feed_forward: FeedForwardRecipe::DenseGeluErf,
        },
        hidden_act: hidden_act.into(),
        position_embedding_type: position_embedding_type.into(),
        is_decoder: false,
        add_cross_attention: false,
    };
    config.validate()?;
    validate_recipe(&config)?;
    Ok(config)
}

fn validate_recipe(config: &BertConfig) -> Result<(), BackendError> {
    let PositionRecipe::LearnedRelative { buckets } = config.recipe.positions else {
        return Err(start("invalid MPNet position recipe"));
    };
    if buckets != BUCKETS
        || config.recipe.feed_forward != FeedForwardRecipe::DenseGeluErf
        || config.type_vocab_size != 1
        || config.max_position_embeddings <= POSITION_OFFSET
    {
        return Err(start("invalid MPNet arithmetic recipe"));
    }
    elements(&[BUCKETS, config.num_attention_heads])?;
    elements(&[config.max_position_embeddings, config.hidden_size])?;
    Ok(())
}

/// Number of token positions an MPNet request can address after the native
/// `positions + 2` embedding lookup offset.
pub(crate) fn position_capacity(config: &BertConfig) -> Result<u32, BackendError> {
    if config.family != EncoderFamily::Mpnet || config.max_position_embeddings <= POSITION_OFFSET {
        return Err(start("invalid MPNet position capacity"));
    }
    let capacity = config.max_position_embeddings - POSITION_OFFSET;
    u32::try_from(capacity).map_err(|_| start("MPNet position capacity exceeds CUDA range"))
}

/// Native symmetric 32-bucket relative-position bucket for the signed
/// `relative = k_index - q_index` produced by native `idx.unsqueeze(0) -
/// idx.unsqueeze(1)` at score position `[q, k]`.
///
/// The map itself stays generic over `relative`; the query/key lookup chooses
/// the native `k - q` order. Packed sequences use local 0..length-1 positions.
///
/// Hugging Face computes the large half as
/// `floor(log(d / 8) / log(16) * 8) + 8` in F32. For integer distances this is
/// exactly `8 + floor(2 * log2(d / 8))`, so the comparisons below are not an
/// approximation: they are the same integer bucket boundaries without a
/// device transcendental.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const fn relative_bucket(relative: i64) -> u32 {
    let distance = relative.unsigned_abs();
    let bucket = if distance < MAX_EXACT {
        distance as u32
    } else if distance < 12 {
        8
    } else if distance < 16 {
        9
    } else if distance < 23 {
        10
    } else if distance < 32 {
        11
    } else if distance < 46 {
        12
    } else if distance < 64 {
        13
    } else if distance < 91 {
        14
    } else {
        15
    };
    if relative > 0 {
        bucket + BUCKETS as u32 / 2
    } else {
        bucket
    }
}

/// Round an F32 bias to the model dtype before the F32 score add, matching the
/// native model-dtype relative-bias table.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn round_bias_bf16(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn round_bias_f16(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}

/// Reference for the additive FA4 score hook: F32 QK scaled by 1/sqrt(64),
/// plus one model-dtype bias widened back to F32. The sum is never rounded.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn relative_score(qk: f32, bias: f32) -> f32 {
    qk * 0.125 + bias
}

/// Bias value addressed by `head` and the signed q/k positions.
///
/// Native `position_bias` broadcasts `idx.unsqueeze(0) - idx.unsqueeze(1)`, so
/// entry `[q, k]` holds `k - q`, not `q - k`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn table_bias(
    table: &[f32],
    heads: usize,
    head: usize,
    q_index: i64,
    k_index: i64,
) -> Option<f32> {
    if head >= heads {
        return None;
    }
    let bucket = relative_bucket(k_index - q_index) as usize;
    table
        .get(bucket.checked_mul(heads)?.checked_add(head)?)
        .copied()
}

fn source(
    checkpoint: &Checkpoint,
    root: &str,
    relative: &str,
    shape: &[usize],
    used: &mut BTreeSet<String>,
) -> Result<WeightSource, BackendError> {
    let name = format!("{root}{relative}");
    let weight = checkpoint.source(&name, shape)?;
    used.insert(name);
    Ok(weight)
}

fn dense(
    checkpoint: &Checkpoint,
    root: &str,
    prefix: &str,
    weight_shape: &[usize],
    bias_shape: &[usize],
    used: &mut BTreeSet<String>,
) -> Result<LinearWeights, BackendError> {
    Ok(LinearWeights {
        weight: source(
            checkpoint,
            root,
            &format!("{prefix}.weight"),
            weight_shape,
            used,
        )?,
        bias: source(
            checkpoint,
            root,
            &format!("{prefix}.bias"),
            bias_shape,
            used,
        )?,
    })
}

fn validate_root(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    match root {
        "mpnet." => {
            if let Some(name) = names.iter().find(|name| {
                !name.starts_with("mpnet.")
                    && (name.starts_with("embeddings.")
                        || name.starts_with("encoder.")
                        || name.starts_with("pooler."))
            }) {
                return Err(start(format!(
                    "incoherent MPNet checkpoint namespace: {name} lacks the mpnet. root"
                )));
            }
        }
        "" => {
            if let Some(name) = names.iter().find(|name| name.starts_with("mpnet.")) {
                return Err(start(format!(
                    "incoherent MPNet checkpoint namespace: {name} uses the mpnet. root"
                )));
            }
        }
        _ => {
            return Err(start(format!(
                "unsupported MPNet checkpoint namespace {root}"
            )))
        }
    }
    Ok(())
}

fn reject_unexpected(
    checkpoint: &Checkpoint,
    root: &str,
    used: &BTreeSet<String>,
) -> Result<(), BackendError> {
    let position_ids = format!("{root}embeddings.position_ids");
    let pooler_weight = format!("{root}pooler.dense.weight");
    let pooler_bias = format!("{root}pooler.dense.bias");
    for name in checkpoint.names() {
        if used.contains(name)
            || name == &position_ids
            || name == &pooler_weight
            || name == &pooler_bias
        {
            continue;
        }
        if name.ends_with("embeddings.token_type_embeddings.weight") {
            return Err(start(format!(
                "MPNet checkpoint must not contain token-type embeddings: {name}"
            )));
        }
        return Err(start(format!(
            "unsupported MPNet checkpoint tensor {name}; only position_ids and the pooler may be ignored"
        )));
    }
    Ok(())
}

/// Load the MPNet encoder layout into the shared post-norm encoder weights.
pub(crate) fn load(path: &Path, config: BertConfig) -> Result<BertWeights, BackendError> {
    if config.family != EncoderFamily::Mpnet {
        return Err(start("MPNet checkpoint loader requires the MPNet family"));
    }
    validate_recipe(&config)?;
    let checkpoint = Checkpoint::open(path)?;
    let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
    let root = select_root(config.family, &names)?;
    validate_root(root, &names)?;

    let hidden = config.hidden_size;
    let intermediate_width = config.intermediate_size;
    let mut used = BTreeSet::new();
    let word = source(
        &checkpoint,
        root,
        "embeddings.word_embeddings.weight",
        &[config.vocab_size, hidden],
        &mut used,
    )?;
    let position = source(
        &checkpoint,
        root,
        "embeddings.position_embeddings.weight",
        &[config.max_position_embeddings, hidden],
        &mut used,
    )?;
    let gamma = source(
        &checkpoint,
        root,
        "embeddings.LayerNorm.weight",
        &[hidden],
        &mut used,
    )?;
    let beta = source(
        &checkpoint,
        root,
        "embeddings.LayerNorm.bias",
        &[hidden],
        &mut used,
    )?;
    let relative_bias = source(
        &checkpoint,
        root,
        "encoder.relative_attention_bias.weight",
        &[BUCKETS, config.num_attention_heads],
        &mut used,
    )?;

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.num_hidden_layers)
        .map_err(start)?;
    for layer_index in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layer.{layer_index}");
        let attention = format!("{prefix}.attention");
        let attn = format!("{attention}.attn");
        layers.push(LayerWeights {
            query: dense(
                &checkpoint,
                root,
                &format!("{attn}.q"),
                &[hidden, hidden],
                &[hidden],
                &mut used,
            )?,
            key: dense(
                &checkpoint,
                root,
                &format!("{attn}.k"),
                &[hidden, hidden],
                &[hidden],
                &mut used,
            )?,
            value: dense(
                &checkpoint,
                root,
                &format!("{attn}.v"),
                &[hidden, hidden],
                &[hidden],
                &mut used,
            )?,
            attention: dense(
                &checkpoint,
                root,
                &format!("{attn}.o"),
                &[hidden, hidden],
                &[hidden],
                &mut used,
            )?,
            intermediate: dense(
                &checkpoint,
                root,
                &format!("{prefix}.intermediate.dense"),
                &[intermediate_width, hidden],
                &[intermediate_width],
                &mut used,
            )?,
            output: dense(
                &checkpoint,
                root,
                &format!("{prefix}.output.dense"),
                &[hidden, intermediate_width],
                &[hidden],
                &mut used,
            )?,
            attention_gamma: source(
                &checkpoint,
                root,
                &format!("{attention}.LayerNorm.weight"),
                &[hidden],
                &mut used,
            )?,
            attention_beta: source(
                &checkpoint,
                root,
                &format!("{attention}.LayerNorm.bias"),
                &[hidden],
                &mut used,
            )?,
            output_gamma: source(
                &checkpoint,
                root,
                &format!("{prefix}.output.LayerNorm.weight"),
                &[hidden],
                &mut used,
            )?,
            output_beta: source(
                &checkpoint,
                root,
                &format!("{prefix}.output.LayerNorm.bias"),
                &[hidden],
                &mut used,
            )?,
        });
    }
    reject_unexpected(&checkpoint, root, &used)?;
    // MPNet has no token-type embeddings. The shared embedding kernel reads a
    // single zero row; request validation requires all token-type IDs to be 0.
    let token_type = WeightSource::zeroed(vec![1, hidden])?;
    Ok(BertWeights {
        config,
        word,
        position,
        token_type,
        gamma,
        beta,
        relative_bias: Some(relative_bias),
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BertConfig;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 8;
    const HEADS: usize = 2;
    const MAX_POSITION: usize = 8;
    const VOCAB: usize = 4;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-mpnet-loader-{}-{}-{}",
                module_path!().replace("::", "-"),
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.config(serde_json::json!({}));
            fixture
        }
        fn config(&self, overrides: serde_json::Value) {
            let mut value = serde_json::json!({
                "model_type": "mpnet",
                "architectures": ["MPNetForMaskedLM"],
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "num_attention_heads": HEADS,
                "num_hidden_layers": 1,
                "max_position_embeddings": MAX_POSITION,
                "vocab_size": VOCAB,
                "layer_norm_eps": 1e-5,
                "hidden_act": "gelu",
                "relative_attention_num_buckets": BUCKETS,
                "pad_token_id": 1
            });
            for (key, val) in overrides.as_object().unwrap() {
                value[key] = val.clone();
            }
            std::fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
        fn entries(root: &str) -> Vec<(String, Vec<usize>, String)> {
            let mut entries = vec![
                (
                    format!("{root}embeddings.word_embeddings.weight"),
                    vec![VOCAB, HIDDEN],
                    "word".into(),
                ),
                (
                    format!("{root}embeddings.position_embeddings.weight"),
                    vec![MAX_POSITION, HIDDEN],
                    "position".into(),
                ),
                (
                    format!("{root}embeddings.LayerNorm.weight"),
                    vec![HIDDEN],
                    "embedding_gamma".into(),
                ),
                (
                    format!("{root}embeddings.LayerNorm.bias"),
                    vec![HIDDEN],
                    "embedding_beta".into(),
                ),
                (
                    format!("{root}encoder.relative_attention_bias.weight"),
                    vec![BUCKETS, HEADS],
                    "relative".into(),
                ),
            ];
            for (relative, shape, canonical) in [
                ("attention.attn.q.weight", vec![HIDDEN, HIDDEN], "q_w"),
                ("attention.attn.q.bias", vec![HIDDEN], "q_b"),
                ("attention.attn.k.weight", vec![HIDDEN, HIDDEN], "k_w"),
                ("attention.attn.k.bias", vec![HIDDEN], "k_b"),
                ("attention.attn.v.weight", vec![HIDDEN, HIDDEN], "v_w"),
                ("attention.attn.v.bias", vec![HIDDEN], "v_b"),
                ("attention.attn.o.weight", vec![HIDDEN, HIDDEN], "o_w"),
                ("attention.attn.o.bias", vec![HIDDEN], "o_b"),
                ("attention.LayerNorm.weight", vec![HIDDEN], "attn_gamma"),
                ("attention.LayerNorm.bias", vec![HIDDEN], "attn_beta"),
                (
                    "intermediate.dense.weight",
                    vec![INTERMEDIATE, HIDDEN],
                    "intermediate_w",
                ),
                (
                    "intermediate.dense.bias",
                    vec![INTERMEDIATE],
                    "intermediate_b",
                ),
                (
                    "output.dense.weight",
                    vec![HIDDEN, INTERMEDIATE],
                    "output_w",
                ),
                ("output.dense.bias", vec![HIDDEN], "output_b"),
                ("output.LayerNorm.weight", vec![HIDDEN], "output_gamma"),
                ("output.LayerNorm.bias", vec![HIDDEN], "output_beta"),
            ] {
                entries.push((
                    format!("{root}encoder.layer.0.{relative}"),
                    shape,
                    canonical.into(),
                ));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        }
        fn weights(
            &self,
            root: &str,
            dtype: Dtype,
            omit: &[&str],
            extra: &[(&str, Vec<usize>, Dtype)],
        ) {
            let mut entries = Self::entries(root)
                .into_iter()
                .filter(|(_, _, canonical)| !omit.contains(&canonical.as_str()))
                .collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let payloads = entries
                .iter()
                .map(|(_, shape, canonical)| Self::data(shape, dtype, canonical))
                .chain(
                    extra
                        .iter()
                        .map(|(name, shape, extra_dtype)| Self::data(shape, *extra_dtype, name)),
                )
                .collect::<Vec<_>>();
            let mut views = entries
                .iter()
                .zip(&payloads)
                .map(|((name, shape, _), data)| {
                    (
                        name.clone(),
                        TensorView::new(dtype, shape.clone(), data).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            views.extend(extra.iter().zip(payloads.iter().skip(entries.len())).map(
                |((name, shape, dtype), data)| {
                    (
                        (*name).into(),
                        TensorView::new(*dtype, shape.clone(), data).unwrap(),
                    )
                },
            ));
            std::fs::write(
                self.0.join("model.safetensors"),
                serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
            )
            .unwrap();
        }
        fn data(shape: &[usize], dtype: Dtype, name: &str) -> Vec<u8> {
            let count = shape.iter().product::<usize>();
            let mut bytes = Vec::new();
            for index in 0..count {
                let scalar = if name.ends_with("gamma") || name.contains("LayerNorm.weight") {
                    1.0
                } else if name.ends_with("beta")
                    || name.contains("LayerNorm.bias")
                    || name.ends_with(".bias")
                {
                    (index as f32 + 1.0) / 1024.0
                } else if name.ends_with("relative") {
                    ((index as f32) - 16.0) / 64.0
                } else {
                    ((index % 997) as f32 + 0.5) / 512.0
                };
                match dtype {
                    Dtype::F32 => bytes.extend_from_slice(&scalar.to_le_bytes()),
                    Dtype::F16 => bytes
                        .extend_from_slice(&half::f16::from_f32(scalar).to_bits().to_le_bytes()),
                    Dtype::BF16 => bytes
                        .extend_from_slice(&half::bf16::from_f32(scalar).to_bits().to_le_bytes()),
                    Dtype::I64 => bytes.extend_from_slice(&(index as i64).to_le_bytes()),
                    _ => unreachable!(),
                }
            }
            bytes
        }
        fn error(&self) -> String {
            match BertWeights::load(&self.0) {
                Ok(_) => panic!("invalid MPNet fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read_all(source: &WeightSource) -> Vec<u16> {
        let count = source.shape().iter().product::<usize>();
        let mut bits = vec![0u16; count];
        source.read_bf16(0, &mut bits).unwrap();
        bits
    }

    #[test]
    fn loads_bare_and_prefixed_mpnet_layouts_for_float_precisions() {
        for root in ["", "mpnet."] {
            for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
                let fixture = Fixture::new();
                fixture.weights(
                    root,
                    dtype,
                    &[],
                    &[
                        (
                            &format!("{root}embeddings.position_ids"),
                            vec![1, MAX_POSITION],
                            Dtype::I64,
                        ),
                        (
                            &format!("{root}pooler.dense.weight"),
                            vec![HIDDEN, HIDDEN],
                            dtype,
                        ),
                        (&format!("{root}pooler.dense.bias"), vec![HIDDEN], dtype),
                    ],
                );
                let weights = BertWeights::load(&fixture.0).unwrap();
                assert_eq!(weights.config.family, EncoderFamily::Mpnet);
                assert_eq!(weights.config.type_vocab_size, 1);
                assert_eq!(weights.config.rotary_table_positions, None);
                assert_eq!(
                    weights.config.architectures,
                    vec!["MPNetForMaskedLM".to_string()]
                );
                assert_eq!(
                    weights.config.recipe,
                    EncoderRecipe {
                        embedding: NormArithmetic::RoundedAddBeforeStats,
                        residual: NormArithmetic::RoundedAddBeforeStats,
                        positions: PositionRecipe::LearnedRelative { buckets: BUCKETS },
                        feed_forward: FeedForwardRecipe::DenseGeluErf,
                    }
                );
                assert_eq!(
                    weights.relative_bias.as_ref().unwrap().shape(),
                    &[BUCKETS, HEADS]
                );
                assert_eq!(weights.position.shape(), &[MAX_POSITION, HIDDEN]);
                assert_eq!(weights.layers.len(), 1);
                assert_eq!(weights.layers[0].query.weight.shape(), &[HIDDEN, HIDDEN]);
                assert!(read_all(&weights.token_type)
                    .iter()
                    .all(|bits| *bits == half::bf16::from_f32(0.0).to_bits()));
                assert_eq!(
                    weights.relative_bias.as_ref().unwrap().shape(),
                    &[BUCKETS, HEADS]
                );
            }
        }
    }

    #[test]
    fn rejects_token_types_mixed_roots_and_unknown_tensors() {
        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[(
                "embeddings.token_type_embeddings.weight",
                vec![1, HIDDEN],
                Dtype::F32,
            )],
        );
        assert!(fixture.error().contains("token-type"));

        let fixture = Fixture::new();
        fixture.weights(
            "mpnet.",
            Dtype::F32,
            &[],
            &[(
                "encoder.relative_attention_bias.weight",
                vec![BUCKETS, HEADS],
                Dtype::F32,
            )],
        );
        assert!(fixture.error().contains("lacks the mpnet. root"));

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[("encoder.unknown.weight", vec![HIDDEN], Dtype::F32)],
        );
        assert!(fixture
            .error()
            .contains("unsupported MPNet checkpoint tensor"));
    }

    #[test]
    fn requires_exact_relative_table_and_layer_shapes() {
        let fixture = Fixture::new();
        fixture.weights("", Dtype::F32, &["relative"], &[]);
        assert!(fixture.error().contains("missing weight"));

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &["relative"],
            &[(
                "encoder.relative_attention_bias.weight",
                vec![BUCKETS - 1, HEADS],
                Dtype::F32,
            )],
        );
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &["q_w"],
            &[(
                "encoder.layer.0.attention.attn.q.weight",
                vec![HIDDEN, HIDDEN + 1],
                Dtype::F32,
            )],
        );
        assert!(fixture.error().contains("wrong shape"));
    }

    #[test]
    fn rejects_unsupported_weight_dtype_at_load() {
        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &["relative"],
            &[(
                "encoder.relative_attention_bias.weight",
                vec![BUCKETS, HEADS],
                Dtype::I64,
            )],
        );
        let error = match BertWeights::load(&fixture.0) {
            Ok(_) => panic!("unsupported dtype was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("unsupported weight dtype"), "{error}");
    }

    #[test]
    fn rejects_invalid_mpnet_config_contracts() {
        for (fields, message) in [
            (serde_json::json!({"type_vocab_size": 2}), "no token-type"),
            (
                serde_json::json!({"relative_attention_num_buckets": 16}),
                "exactly 32",
            ),
            (
                serde_json::json!({"hidden_act": "gelu_pytorch_tanh"}),
                "exact gelu",
            ),
            (
                serde_json::json!({"position_embedding_type": "rope"}),
                "absolute position",
            ),
            (serde_json::json!({"is_decoder": true}), "decoder"),
            (
                serde_json::json!({"add_cross_attention": true}),
                "cross-attention",
            ),
            (
                serde_json::json!({"max_position_embeddings": POSITION_OFFSET}),
                "lookup offset",
            ),
            (serde_json::json!({"layer_norm_eps": 0.0}), "layer_norm_eps"),
            (
                serde_json::json!({"num_attention_heads": 0}),
                "invalid or unsupported",
            ),
            (
                serde_json::json!({"architectures": ["BertModel"]}),
                "unsupported MPNet architecture",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.config(fields);
            let error = BertConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn actual_reference_config_has_the_expected_position_capacity() {
        let raw: RawEncoderConfig = serde_json::from_value(serde_json::json!({
            "model_type": "mpnet",
            "architectures": ["MPNetForMaskedLM"],
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "max_position_embeddings": 514,
            "vocab_size": 30527,
            "layer_norm_eps": 1e-5,
            "hidden_act": "gelu",
            "relative_attention_num_buckets": 32
        }))
        .unwrap();
        let config = config(&raw).unwrap();
        assert_eq!(position_capacity(&config).unwrap(), 512);
        assert_eq!(
            config.recipe.positions,
            PositionRecipe::LearnedRelative { buckets: 32 }
        );
        assert_eq!(config.recipe.feed_forward, FeedForwardRecipe::DenseGeluErf);
        assert_eq!(
            config.recipe.embedding,
            NormArithmetic::RoundedAddBeforeStats
        );
        assert_eq!(
            config.recipe.residual,
            NormArithmetic::RoundedAddBeforeStats
        );
        assert_eq!(config.layer_norm_eps, 1e-5);
    }

    fn log_reference(relative: i64) -> u32 {
        let distance = relative.unsigned_abs();
        let float_distance = distance.max(1) as f32;
        let large =
            (((float_distance / 8.0).ln() / (128.0f32 / 8.0).ln() * 8.0) as i64 + 8).min(15) as u32;
        let bucket = if distance < 8 { distance as u32 } else { large };
        if relative > 0 {
            bucket + 16
        } else {
            bucket
        }
    }

    #[test]
    fn integer_bucket_boundaries_match_the_native_f32_log_formula() {
        for relative in -4096..=4096 {
            assert_eq!(
                relative_bucket(relative),
                log_reference(relative),
                "relative {relative}"
            );
        }
        let expected = [
            (-128, 15),
            (-91, 15),
            (-90, 14),
            (-64, 14),
            (-63, 13),
            (-46, 13),
            (-45, 12),
            (-32, 12),
            (-23, 11),
            (-22, 10),
            (-16, 10),
            (-12, 9),
            (-11, 8),
            (-8, 8),
            (-7, 7),
            (-1, 1),
            (0, 0),
            (1, 17),
            (7, 23),
            (8, 24),
            (11, 24),
            (12, 25),
            (16, 26),
            (32, 28),
            (46, 29),
            (64, 30),
            (90, 30),
            (91, 31),
            (128, 31),
            (4096, 31),
        ];
        for (relative, bucket) in expected {
            assert_eq!(relative_bucket(relative), bucket, "relative {relative}");
        }
    }

    #[test]
    fn bias_rounding_and_score_math_keep_fp32_scores() {
        let mut table = vec![0.0f32; BUCKETS * 2];
        let negative_future = round_bias_bf16(-1.0 / 3.0);
        let zero = round_bias_bf16(0.1);
        let positive_past = round_bias_bf16(0.25);
        table[0] = zero;
        table[23 * 2 + 1] = negative_future;
        table[17 * 2] = round_bias_bf16(-0.1);
        table[2 + 1] = positive_past;
        assert_eq!(table[0], half::bf16::from_f32(0.1).to_f32());
        // Native indexes `[q, k]` by k - q. q=0, k=7 is a later key:
        // relative +7 -> bucket 7 + 16 = 23.
        assert_eq!(table_bias(&table, 2, 1, 0, 7), Some(negative_future));
        // q=7, k=0 is an earlier key: relative -7 -> bucket 7.
        assert_eq!(table_bias(&table, 2, 0, 7, 0), Some(0.0));
        // Asymmetric table: q=0, k=1 -> bucket 17 and q=1, k=0 -> bucket 1.
        // Distinct heads and signs would fail a mirrored q - k lookup.
        assert_eq!(table_bias(&table, 2, 0, 0, 1), Some(round_bias_bf16(-0.1)));
        assert_eq!(table_bias(&table, 2, 1, 1, 0), Some(positive_past));
        assert_eq!(table_bias(&table, 2, 2, 0, 0), None);
        assert_eq!(table_bias(&table, 2, 0, 0, 0), Some(zero));
        let qk = 0.75f32;
        let bias = round_bias_bf16(-0.2);
        let score = relative_score(qk, bias);
        assert_eq!(score, qk * 0.125f32 + bias);
        assert_ne!(score, (qk * 0.125f32 + (-0.2f32)));
        assert_eq!(
            relative_score(qk, round_bias_f16(-0.2)),
            qk * 0.125 + half::f16::from_f32(-0.2).to_f32()
        );
    }

    #[test]
    fn validates_the_actual_public_mpnet_config_and_header_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../../public-reference");
        let Ok(config_bytes) = std::fs::read(root.join("actual-mpnet-config.json")) else {
            return;
        };
        let raw: RawEncoderConfig = serde_json::from_slice(&config_bytes).unwrap();
        let config = config(&raw).unwrap();
        assert_eq!(position_capacity(&config).unwrap(), 512);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);

        let Ok(header_bytes) = std::fs::read(root.join("actual-mpnet-header-metadata.json")) else {
            return;
        };
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).unwrap();
        let tensors = header["tensors"].as_object().unwrap();
        let mut expected = vec![
            ("embeddings.position_ids", vec![1, 514], "I64"),
            ("embeddings.LayerNorm.bias", vec![768], "F32"),
            ("embeddings.LayerNorm.weight", vec![768], "F32"),
            (
                "embeddings.position_embeddings.weight",
                vec![514, 768],
                "F32",
            ),
            ("embeddings.word_embeddings.weight", vec![30527, 768], "F32"),
            (
                "encoder.relative_attention_bias.weight",
                vec![32, 12],
                "F32",
            ),
            ("pooler.dense.bias", vec![768], "F32"),
            ("pooler.dense.weight", vec![768, 768], "F32"),
        ];
        for layer in 0..12 {
            for (name, rows, columns) in [
                ("attention.attn.k", 768, 768),
                ("attention.attn.o", 768, 768),
                ("attention.attn.q", 768, 768),
                ("attention.attn.v", 768, 768),
            ] {
                expected.push((
                    Box::leak(format!("encoder.layer.{layer}.{name}.weight").into_boxed_str()),
                    vec![rows, columns],
                    "F32",
                ));
                expected.push((
                    Box::leak(format!("encoder.layer.{layer}.{name}.bias").into_boxed_str()),
                    vec![rows],
                    "F32",
                ));
            }
            expected.push((
                Box::leak(
                    format!("encoder.layer.{layer}.attention.LayerNorm.bias").into_boxed_str(),
                ),
                vec![768],
                "F32",
            ));
            expected.push((
                Box::leak(
                    format!("encoder.layer.{layer}.attention.LayerNorm.weight").into_boxed_str(),
                ),
                vec![768],
                "F32",
            ));
            expected.push((
                Box::leak(
                    format!("encoder.layer.{layer}.intermediate.dense.weight").into_boxed_str(),
                ),
                vec![3072, 768],
                "F32",
            ));
            expected.push((
                Box::leak(
                    format!("encoder.layer.{layer}.intermediate.dense.bias").into_boxed_str(),
                ),
                vec![3072],
                "F32",
            ));
            expected.push((
                Box::leak(format!("encoder.layer.{layer}.output.dense.weight").into_boxed_str()),
                vec![768, 3072],
                "F32",
            ));
            expected.push((
                Box::leak(format!("encoder.layer.{layer}.output.dense.bias").into_boxed_str()),
                vec![768],
                "F32",
            ));
            expected.push((
                Box::leak(format!("encoder.layer.{layer}.output.LayerNorm.bias").into_boxed_str()),
                vec![768],
                "F32",
            ));
            expected.push((
                Box::leak(
                    format!("encoder.layer.{layer}.output.LayerNorm.weight").into_boxed_str(),
                ),
                vec![768],
                "F32",
            ));
        }
        for (name, shape, dtype) in expected {
            let tensor = tensors
                .get(name)
                .unwrap_or_else(|| panic!("actual MPNet header lacks {name}"));
            assert_eq!(tensor["dtype"], dtype, "{name}");
            assert_eq!(tensor["shape"], serde_json::json!(shape), "{name}");
        }
        assert!(
            !tensors
                .keys()
                .any(|name| name.ends_with("embeddings.token_type_embeddings.weight")),
            "actual MPNet checkpoint must not have token-type embeddings"
        );
    }

    #[test]
    fn model_dtype_biases_are_rounded_once_before_addition() {
        // 0.1 is not representable in BF16; adding its exact F32 product would
        // not match the native BF16 table widened back to F32.
        let rounded = round_bias_bf16(0.1);
        assert_ne!(rounded, 0.1);
        assert_eq!(relative_score(4.0, rounded), 0.5 + rounded);
        assert_eq!(relative_score(4.0, 0.1), 0.5 + 0.1);
        assert_ne!(rounded, round_bias_f16(0.1));
    }
}
