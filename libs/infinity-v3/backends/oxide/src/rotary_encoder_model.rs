// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! GTE NewModel checkpoint adapter.
//!
//! GTE stores packed QKV and an up/gate feed-forward projection that the common
//! post-norm BERT engine does not name directly. This loader keeps the existing
//! `BertWeights`/`LayerWeights` layout by creating checked views into those
//! packed tensors. Payloads stay in their original shard and are read in bounded
//! chunks by `WeightSource`; only metadata views are created here.

use crate::model::{
    select_root, BertConfig, BertWeights, Checkpoint, EncoderFamily, LayerWeights, LinearWeights,
    WeightSource,
};
use crate::start;
use infinity_v3_backend_core::BackendError;
use std::collections::BTreeSet;
use std::path::Path;

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

fn split_qkv(
    weight: &WeightSource,
    bias: &WeightSource,
    hidden: usize,
) -> Result<[LinearWeights; 3], BackendError> {
    let matrix = hidden
        .checked_mul(hidden)
        .ok_or_else(|| start("GTE QKV matrix size overflow"))?;
    let second_matrix = matrix
        .checked_mul(2)
        .ok_or_else(|| start("GTE QKV matrix size overflow"))?;
    let second_bias = hidden
        .checked_mul(2)
        .ok_or_else(|| start("GTE QKV bias size overflow"))?;
    Ok([
        LinearWeights {
            weight: weight.slice(0, vec![hidden, hidden])?,
            bias: bias.slice(0, vec![hidden])?,
        },
        LinearWeights {
            weight: weight.slice(matrix, vec![hidden, hidden])?,
            bias: bias.slice(hidden, vec![hidden])?,
        },
        LinearWeights {
            weight: weight.slice(second_matrix, vec![hidden, hidden])?,
            bias: bias.slice(second_bias, vec![hidden])?,
        },
    ])
}

fn validate_root(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    match root {
        "new." => {
            if let Some(name) = names
                .iter()
                .find(|name| name.starts_with("embeddings.") || name.starts_with("encoder.layer."))
            {
                return Err(start(format!(
                    "incoherent GTE checkpoint namespace: {name} lacks the new. root"
                )));
            }
        }
        "" => {
            if let Some(name) = names.iter().find(|name| name.starts_with("new.")) {
                return Err(start(format!(
                    "incoherent GTE checkpoint namespace: {name} uses the new. root"
                )));
            }
        }
        _ => {
            return Err(start(format!(
                "unsupported GTE checkpoint namespace root {root}"
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
    let pooler_weight = format!("{root}pooler.dense.weight");
    let pooler_bias = format!("{root}pooler.dense.bias");
    for name in checkpoint.names() {
        if used.contains(name) || name == &pooler_weight || name == &pooler_bias {
            continue;
        }
        return Err(start(format!(
            "unsupported GTE checkpoint tensor {name}; only the final pooler may be ignored"
        )));
    }
    Ok(())
}

pub(crate) fn load(path: &Path, config: BertConfig) -> Result<BertWeights, BackendError> {
    if config.family != EncoderFamily::Gte {
        return Err(start(
            "rotary encoder checkpoint loader requires the GTE NewModel family",
        ));
    }
    let checkpoint = Checkpoint::open(path)?;
    let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
    let root = select_root(config.family, &names)?;
    validate_root(root, &names)?;

    let hidden = config.hidden_size;
    let intermediate_width = config.intermediate_size;
    let qkv_rows = hidden
        .checked_mul(3)
        .ok_or_else(|| start("GTE QKV row count overflow"))?;
    let gate_rows = intermediate_width
        .checked_mul(2)
        .ok_or_else(|| start("GTE gated feed-forward row count overflow"))?;
    let mut used = BTreeSet::new();

    let word = source(
        &checkpoint,
        root,
        "embeddings.word_embeddings.weight",
        &[config.vocab_size, hidden],
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
    // GTE NewModel is word-only. The common struct still carries position/type
    // sources, so retain single-row zero placeholders until the engine drops
    // those fields; the rotary engine neither uploads nor reads them.
    let position = WeightSource::zeroed(vec![1, hidden])?;
    let token_type = WeightSource::zeroed(vec![1, hidden])?;

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.num_hidden_layers)
        .map_err(start)?;
    for layer_index in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layer.{layer_index}");
        let qkv_weight = source(
            &checkpoint,
            root,
            &format!("{prefix}.attention.qkv_proj.weight"),
            &[qkv_rows, hidden],
            &mut used,
        )?;
        let qkv_bias = source(
            &checkpoint,
            root,
            &format!("{prefix}.attention.qkv_proj.bias"),
            &[qkv_rows],
            &mut used,
        )?;
        let [query, key, value] = split_qkv(&qkv_weight, &qkv_bias, hidden)?;
        let attention = dense(
            &checkpoint,
            root,
            &format!("{prefix}.attention.o_proj"),
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?;
        let intermediate_weight = source(
            &checkpoint,
            root,
            &format!("{prefix}.mlp.up_gate_proj.weight"),
            &[gate_rows, hidden],
            &mut used,
        )?;
        // GTE's packed up/gate projection has no bias. Keep a zero view for the
        // current common struct; the gated engine uses a plain GEMM without it.
        let intermediate = LinearWeights {
            weight: intermediate_weight,
            bias: WeightSource::zeroed(vec![gate_rows])?,
        };
        let output = dense(
            &checkpoint,
            root,
            &format!("{prefix}.mlp.down_proj"),
            &[hidden, intermediate_width],
            &[hidden],
            &mut used,
        )?;
        let attention_gamma = source(
            &checkpoint,
            root,
            &format!("{prefix}.attn_ln.weight"),
            &[hidden],
            &mut used,
        )?;
        let attention_beta = source(
            &checkpoint,
            root,
            &format!("{prefix}.attn_ln.bias"),
            &[hidden],
            &mut used,
        )?;
        let output_gamma = source(
            &checkpoint,
            root,
            &format!("{prefix}.mlp_ln.weight"),
            &[hidden],
            &mut used,
        )?;
        let output_beta = source(
            &checkpoint,
            root,
            &format!("{prefix}.mlp_ln.bias"),
            &[hidden],
            &mut used,
        )?;
        layers.push(LayerWeights {
            query,
            key,
            value,
            attention,
            intermediate,
            output,
            attention_gamma,
            attention_beta,
            output_gamma,
            output_beta,
        });
    }
    reject_unexpected(&checkpoint, root, &used)?;
    Ok(BertWeights {
        relative_bias: None,
        config,
        word,
        position,
        token_type,
        gamma,
        beta,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const HIDDEN: usize = 768;
    const INTERMEDIATE: usize = 3072;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-gte-loader-{}-{}-{}",
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
                "model_type": "new",
                "architectures": ["NewModel"],
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "num_attention_heads": 12,
                "num_hidden_layers": 1,
                "max_position_embeddings": 8,
                "vocab_size": 4,
                "type_vocab_size": 0,
                "hidden_act": "gelu"
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
        fn gte_entries(root: &str) -> Vec<(String, Vec<usize>, String)> {
            let mut entries = vec![
                (
                    format!("{root}embeddings.word_embeddings.weight"),
                    vec![4, HIDDEN],
                    "word".into(),
                ),
                (
                    format!("{root}embeddings.LayerNorm.weight"),
                    vec![HIDDEN],
                    "emb_gamma".into(),
                ),
                (
                    format!("{root}embeddings.LayerNorm.bias"),
                    vec![HIDDEN],
                    "emb_beta".into(),
                ),
            ];
            for (relative, shape, canonical) in [
                (
                    "encoder.layer.0.attention.qkv_proj.weight",
                    vec![3 * HIDDEN, HIDDEN],
                    "qkv_weight",
                ),
                (
                    "encoder.layer.0.attention.qkv_proj.bias",
                    vec![3 * HIDDEN],
                    "qkv_bias",
                ),
                (
                    "encoder.layer.0.attention.o_proj.weight",
                    vec![HIDDEN, HIDDEN],
                    "o_weight",
                ),
                (
                    "encoder.layer.0.attention.o_proj.bias",
                    vec![HIDDEN],
                    "o_bias",
                ),
                ("encoder.layer.0.attn_ln.weight", vec![HIDDEN], "attn_gamma"),
                ("encoder.layer.0.attn_ln.bias", vec![HIDDEN], "attn_beta"),
                (
                    "encoder.layer.0.mlp.up_gate_proj.weight",
                    vec![2 * INTERMEDIATE, HIDDEN],
                    "up_gate",
                ),
                (
                    "encoder.layer.0.mlp.down_proj.weight",
                    vec![HIDDEN, INTERMEDIATE],
                    "down_weight",
                ),
                (
                    "encoder.layer.0.mlp.down_proj.bias",
                    vec![HIDDEN],
                    "down_bias",
                ),
                ("encoder.layer.0.mlp_ln.weight", vec![HIDDEN], "mlp_gamma"),
                ("encoder.layer.0.mlp_ln.bias", vec![HIDDEN], "mlp_beta"),
            ] {
                entries.push((format!("{root}{relative}"), shape, canonical.into()));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        }
        fn weights(&self, root: &str, dtype: Dtype, omit: &[&str], extra: &[(&str, Vec<usize>)]) {
            let mut entries = Self::gte_entries(root)
                .into_iter()
                .filter(|(_, _, canonical)| !omit.contains(&canonical.as_str()))
                .collect::<Vec<_>>();
            for (name, shape) in extra {
                entries.push(((*name).into(), shape.clone(), (*name).into()));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let data = entries
                .iter()
                .map(|(_, shape, canonical)| {
                    let count = shape.iter().product::<usize>();
                    let mut bytes = Vec::new();
                    for index in 0..count {
                        let scalar = test_value(canonical, index);
                        match dtype {
                            Dtype::F32 => bytes.extend_from_slice(&scalar.to_le_bytes()),
                            Dtype::BF16 => bytes.extend_from_slice(
                                &half::bf16::from_f32(scalar).to_bits().to_le_bytes(),
                            ),
                            _ => unreachable!(),
                        }
                    }
                    bytes
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape, _), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(dtype, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            std::fs::write(
                self.0.join("model.safetensors"),
                serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
            )
            .unwrap();
        }
        fn error(&self) -> String {
            match BertWeights::load(&self.0) {
                Ok(_) => panic!("invalid GTE fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_value(name: &str, index: usize) -> f32 {
        if name.ends_with("gamma") {
            1.0
        } else if name.ends_with("beta") || name.ends_with("bias") {
            (index as f32 + 1.0) / 4096.0
        } else {
            ((index % 997) as f32 + 0.5) / 1024.0
        }
    }
    fn bf16(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }
    fn read(source: &WeightSource, offset: usize) -> u16 {
        let mut out = [0u16; 1];
        source.read_bf16(offset, &mut out).unwrap();
        out[0]
    }

    #[test]
    fn loads_both_roots_and_splits_distinct_packed_qkv_views() {
        for root in ["", "new."] {
            let fixture = Fixture::new();
            fixture.weights(root, Dtype::F32, &[], &[]);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.config.family, EncoderFamily::Gte);
            assert_eq!(weights.config.type_vocab_size, 1);
            assert_eq!(weights.position.shape(), &[1, HIDDEN]);
            assert_eq!(weights.token_type.shape(), &[1, HIDDEN]);
            assert_eq!(read(&weights.position, 0), 0);
            assert_eq!(read(&weights.token_type, 0), 0);
            assert_eq!(weights.word.dtype(), Dtype::F32);
            let matrix = HIDDEN * HIDDEN;
            for (part, base) in [
                (&weights.layers[0].query.weight, 0),
                (&weights.layers[0].key.weight, matrix),
                (&weights.layers[0].value.weight, 2 * matrix),
            ] {
                assert_eq!(part.shape(), &[HIDDEN, HIDDEN]);
                for offset in [0usize, 1, HIDDEN, matrix - 1] {
                    assert_eq!(
                        read(part, offset),
                        bf16(test_value("qkv_weight", base + offset))
                    );
                }
            }
            for (part, base) in [
                (&weights.layers[0].query.bias, 0),
                (&weights.layers[0].key.bias, HIDDEN),
                (&weights.layers[0].value.bias, 2 * HIDDEN),
            ] {
                assert_eq!(part.shape(), &[HIDDEN]);
                assert_eq!(read(part, 0), bf16(test_value("qkv_bias", base)));
                assert_eq!(
                    read(part, HIDDEN - 1),
                    bf16(test_value("qkv_bias", base + HIDDEN - 1))
                );
            }
            assert_eq!(
                weights.layers[0].intermediate.weight.shape(),
                &[2 * INTERMEDIATE, HIDDEN]
            );
            assert_eq!(
                weights.layers[0].intermediate.bias.shape(),
                &[2 * INTERMEDIATE]
            );
            assert_eq!(read(&weights.layers[0].intermediate.bias, 0), 0);
            assert_eq!(
                weights.layers[0].output.weight.shape(),
                &[HIDDEN, INTERMEDIATE]
            );
        }
    }

    #[test]
    fn accepts_f32_conversion_word_only_embeddings_and_ignores_pooler() {
        let fixture = Fixture::new();
        fixture.weights(
            "new.",
            Dtype::F32,
            &[],
            &[("new.pooler.dense.weight", vec![HIDDEN, HIDDEN])],
        );
        let weights = BertWeights::load(&fixture.0).unwrap();
        assert_eq!(weights.word.shape(), &[4, HIDDEN]);
        assert_eq!(read(&weights.word, 1), bf16(test_value("word", 1)));
        assert_eq!(read(&weights.gamma, 0), bf16(1.0));
    }

    #[test]
    fn rejects_ambiguous_incoherent_and_unexpected_checkpoint_tensors() {
        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[("new.embeddings.word_embeddings.weight", vec![4, HIDDEN])],
        );
        assert!(fixture.error().contains("ambiguous checkpoint namespace"));

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[("new.embeddings.LayerNorm.weight", vec![HIDDEN])],
        );
        assert!(fixture
            .error()
            .contains("incoherent GTE checkpoint namespace"));

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[("classifier.weight", vec![2, HIDDEN])],
        );
        assert!(fixture.error().contains("only the final pooler"));
    }

    #[test]
    fn rejects_missing_bad_shape_and_gate_bias_tensors() {
        let fixture = Fixture::new();
        fixture.weights("", Dtype::F32, &["down_bias"], &[]);
        assert!(fixture.error().contains("missing weight"));

        let fixture = Fixture::new();
        let mut entries = Fixture::gte_entries("");
        for entry in &mut entries {
            if entry.2 == "o_weight" {
                entry.1 = vec![HIDDEN, HIDDEN - 1];
            }
        }
        fixture.write_entries(entries);
        assert!(fixture.error().contains("wrong shape"));

        let fixture = Fixture::new();
        fixture.weights(
            "",
            Dtype::F32,
            &[],
            &[(
                "encoder.layer.0.mlp.up_gate_proj.bias",
                vec![2 * INTERMEDIATE],
            )],
        );
        assert!(fixture.error().contains("only the final pooler"));
    }

    #[test]
    fn rejects_unsupported_gte_configurations_before_reading_weights() {
        for (overrides, expected) in [
            (serde_json::json!({"hidden_size": 384}), "GTE NewModel"),
            (serde_json::json!({"type_vocab_size": 2}), "type_vocab_size"),
            (
                serde_json::json!({"architectures": ["BertModel"]}),
                "architecture",
            ),
            (serde_json::json!({"pack_qkv": false}), "packed QKV"),
            (
                serde_json::json!({"layer_norm_type": "rms_norm"}),
                "layer_norm",
            ),
            (serde_json::json!({"logn_attention_scale": true}), "logn"),
            (
                serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}}),
                "ntk",
            ),
            (
                serde_json::json!({"rope_scaling": {"type": "ntk", "factor": 2.0,
                    "high_freq_factor": 4.0}}),
                "rope_scaling field",
            ),
            (
                serde_json::json!({"position_embedding_type": "absolute"}),
                "positions",
            ),
            (
                serde_json::json!({"max_position_embeddings": 8193}),
                "GTE NewModel",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = fixture.error();
            assert!(error.contains(expected), "{error}");
        }
    }

    impl Fixture {
        fn write_entries(&self, entries: Vec<(String, Vec<usize>, String)>) {
            let data = entries
                .iter()
                .map(|(_, shape, canonical)| {
                    let count = shape.iter().product::<usize>();
                    let mut bytes = Vec::new();
                    for index in 0..count {
                        bytes.extend_from_slice(&test_value(canonical, index).to_le_bytes());
                    }
                    bytes
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape, _), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            std::fs::write(
                self.0.join("model.safetensors"),
                serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
            )
            .unwrap();
        }
    }
}
