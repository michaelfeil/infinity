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
            if let Some(name) = names
                .iter()
                .find(|name| name.starts_with("new.") && !name.starts_with("new.pooler."))
            {
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
    used: &BTreeSet<String>,
    classifier: bool,
) -> Result<(), BackendError> {
    let pooler_weight = "new.pooler.dense.weight";
    let pooler_bias = "new.pooler.dense.bias";
    let root_pooler_weight = "pooler.dense.weight";
    let root_pooler_bias = "pooler.dense.bias";
    for name in checkpoint.names() {
        if used.contains(name)
            || name == pooler_weight
            || name == pooler_bias
            || name == root_pooler_weight
            || name == root_pooler_bias
        {
            continue;
        }
        if classifier && (name == "classifier.weight" || name == "classifier.bias") {
            continue;
        }
        return Err(start(format!(
            "unsupported GTE checkpoint tensor {name}; only the final pooler and admitted classifier head may be ignored"
        )));
    }
    Ok(())
}

pub(crate) fn load_with_checkpoint(
    path: &Path,
    mut config: BertConfig,
) -> Result<(BertWeights, Checkpoint), BackendError> {
    if config.family != EncoderFamily::Gte {
        return Err(start(
            "rotary encoder checkpoint loader requires the GTE NewModel family",
        ));
    }
    let classifier = config.architectures.iter().any(|architecture| {
        architecture == "NewForSequenceClassification"
            || architecture == "NewForTokenClassification"
    });
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
    // Native consumes a segment table when the checkpoint actually has one.
    // Without it, both declared type-vocabulary hints and incoming type IDs
    // are irrelevant to the word-only ATen embedding normalization.
    let position = WeightSource::zeroed(vec![1, hidden])?;
    let type_name = format!("{root}embeddings.token_type_embeddings.weight");
    let token_type = match checkpoint.shape(&type_name) {
        Some([rows, width]) if (1..=2).contains(rows) && *width == hidden => {
            if config
                .configured_type_vocab_size
                .is_some_and(|configured| configured != 0 && configured != *rows)
            {
                return Err(start(format!("wrong shape for {type_name}: checkpoint rows {rows} disagree with configured type_vocab_size")));
            }
            config.type_vocab_size = *rows;
            config.recipe.embedding = crate::model::NormArithmetic::ExactWordTypeBiased;
            used.insert(type_name.clone());
            checkpoint.source(&type_name, &[*rows, hidden])?
        }
        Some(shape) => {
            return Err(start(format!(
                "wrong shape for {type_name}: expected [1 or 2, {hidden}], got {shape:?}"
            )))
        }
        None => WeightSource::zeroed(vec![1, hidden])?,
    };
    config.validate()?;

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
    let embedding_companion = config.architectures.len() == 2
        && config.architectures.iter().any(|name| name == "NewModel")
        && config
            .architectures
            .iter()
            .any(|name| name == "NewForTokenClassification");
    // The companion's unused root classifier does not change embedding math.
    // Every other unexpected checkpoint group remains rejected.
    reject_unexpected(&checkpoint, &used, classifier || embedding_companion)?;
    Ok((
        BertWeights {
            relative_bias: None,
            config,
            word,
            position,
            token_type,
            gamma,
            beta,
            layers,
        },
        checkpoint,
    ))
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

    fn segment_request(segment: u32) -> infinity_v3_backend_core::Batch {
        infinity_v3_backend_core::Batch {
            multimodal: vec![],
            input_ids: vec![1],
            token_type_ids: vec![segment],
            position_ids: vec![8191],
            cumulative_seq_lengths: vec![0, 1],
            max_length: 1,
            pooled_indices: vec![0],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    #[test]
    fn actual_large_geometry_and_full_checkpoint_headers_load_without_payload_reads() {
        use std::io::Write;
        for with_segment_table in [false, true] {
            let fixture = Fixture::new();
            let config: serde_json::Value =
                serde_json::from_str(include_str!("../tests/fixtures/gte-large-config.json"))
                    .unwrap();
            std::fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let mut tensors = Vec::new();
            for (name, mut shape, _) in Fixture::gte_entries("") {
                for dim in &mut shape {
                    *dim = match *dim {
                        768 => 1024,
                        2304 => 3072,
                        3072 => 4096,
                        6144 => 8192,
                        other => other,
                    };
                }
                if name.contains(".layer.0.") {
                    for layer in 0..24 {
                        tensors.push((
                            name.replace(".layer.0.", &format!(".layer.{layer}.")),
                            shape.clone(),
                        ));
                    }
                } else {
                    if name.ends_with("embeddings.word_embeddings.weight") {
                        shape[0] = 30528;
                    }
                    tensors.push((name, shape));
                }
            }
            if with_segment_table {
                tensors.push((
                    "embeddings.token_type_embeddings.weight".into(),
                    vec![2, 1024],
                ));
            }
            let mut offset = 0u64;
            let mut header = serde_json::Map::new();
            for (name, shape) in tensors {
                let end = offset + shape.iter().product::<usize>() as u64 * 2;
                header.insert(
                    name,
                    serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,end]}),
                );
                offset = end;
            }
            let mut bytes = serde_json::to_vec(&header).unwrap();
            while !bytes.len().is_multiple_of(8) {
                bytes.push(b' ');
            }
            let mut file = std::fs::File::create(fixture.0.join("model.safetensors")).unwrap();
            file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.set_len(8 + bytes.len() as u64 + offset).unwrap();
            drop(file);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.word.shape(), &[30528, 1024]);
            assert_eq!(weights.layers.len(), 24);
            assert_eq!(weights.config.num_attention_heads, 16);
            assert_eq!(weights.config.intermediate_size, 4096);
            assert_eq!(weights.layers[0].query.weight.shape(), &[1024, 1024]);
            assert_eq!(weights.layers[0].intermediate.weight.shape(), &[8192, 1024]);
            assert_eq!(weights.layers[23].output.weight.shape(), &[1024, 4096]);
            assert_eq!(
                weights.config.recipe.embedding,
                if with_segment_table {
                    crate::model::NormArithmetic::ExactWordTypeBiased
                } else {
                    crate::model::NormArithmetic::AtenEmbedding
                }
            );
            assert_eq!(
                weights.config.recipe.residual,
                crate::model::NormArithmetic::ExactBiased
            );
            let mut request = segment_request(1);
            assert!(crate::validate(&request, &weights.config).is_ok());
            request.token_type_ids[0] = 2;
            assert_eq!(
                crate::validate(&request, &weights.config).is_err(),
                with_segment_table
            );
            if with_segment_table {
                assert_eq!(weights.token_type.shape(), &[2, 1024]);
            }
            request.input_ids[0] = 30528;
            assert!(crate::validate(&request, &weights.config).is_err());
        }
    }

    #[test]
    fn actual_multilingual_embedding_companion_loads_the_existing_shared_backbone() {
        use std::io::Write;
        let fixture = Fixture::new();
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gte-multilingual-base-config.json"
        ))
        .unwrap();
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let vocab = config["vocab_size"].as_u64().unwrap() as usize;
        let layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
        let mut tensors = Vec::new();
        for (name, mut shape, _) in Fixture::gte_entries("new.") {
            if name.contains(".layer.0.") {
                for layer in 0..layers {
                    tensors.push((
                        name.replace(".layer.0.", &format!(".layer.{layer}.")),
                        shape.clone(),
                    ));
                }
            } else {
                if name.ends_with("embeddings.word_embeddings.weight") {
                    shape[0] = vocab;
                }
                tensors.push((name, shape));
            }
        }
        tensors.push(("classifier.weight".into(), vec![2, HIDDEN]));
        tensors.push(("classifier.bias".into(), vec![2]));
        tensors.push((
            "new.embeddings.token_type_embeddings.weight".into(),
            vec![1, HIDDEN],
        ));
        let mut offset = 0u64;
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let end = offset + shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                name,
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,end]}),
            );
            offset = end;
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        while !bytes.len().is_multiple_of(8) {
            bytes.push(b' ');
        }
        let mut file = std::fs::File::create(fixture.0.join("model.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + offset).unwrap();
        drop(file);
        let weights = BertWeights::load(&fixture.0).unwrap();
        assert_eq!(weights.layers.len(), 12);
        assert_eq!(weights.word.shape(), &[250048, HIDDEN]);
        assert_eq!(weights.token_type.shape(), &[1, HIDDEN]);
        assert_eq!(
            weights.config.recipe.embedding,
            crate::model::NormArithmetic::ExactWordTypeBiased
        );
        assert!(!crate::classifier_head::is_classifier_config(
            &weights.config
        ));
        // The two markers select an embedding route only; an explicit task
        // request still has no unambiguous classifier and remains rejected.
        assert!(BertConfig::load_classifier(&fixture.0).is_err());
        for architectures in [
            serde_json::json!(["NewForTokenClassification"]),
            serde_json::json!(["NewModel", "NewForTokenClassification", "OtherModel"]),
            serde_json::json!(["NewModel", "NewModel", "NewForTokenClassification"]),
        ] {
            let mut wrong = config.clone();
            wrong["architectures"] = architectures;
            std::fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&wrong).unwrap(),
            )
            .unwrap();
            assert!(BertConfig::load(&fixture.0).is_err());
        }
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
    fn admits_gte_sequence_classifier_head_and_pooler_namespaces() {
        for segment_rows in [0, 2] {
            let fixture = Fixture::new();
            fixture.config(serde_json::json!({
                "architectures": ["NewForSequenceClassification"],
                "type_vocab_size": segment_rows
            }));
            let mut additional = vec![
                ("classifier.weight", vec![1, HIDDEN]),
                ("classifier.bias", vec![1]),
                ("new.pooler.dense.weight", vec![HIDDEN, HIDDEN]),
                ("new.pooler.dense.bias", vec![HIDDEN]),
                ("pooler.dense.weight", vec![HIDDEN, HIDDEN]),
                ("pooler.dense.bias", vec![HIDDEN]),
            ];
            if segment_rows > 0 {
                additional.push((
                    "embeddings.token_type_embeddings.weight",
                    vec![segment_rows, HIDDEN],
                ));
            }
            fixture.weights("", Dtype::F32, &[], &additional);
            let (weights, checkpoint) = BertWeights::load_classifier(&fixture.0).unwrap();
            assert_eq!(
                weights.config.architectures,
                vec!["NewForSequenceClassification".to_owned()]
            );
            let head =
                crate::classifier_head::ClassifierHeadHost::load(&checkpoint, &weights.config)
                    .unwrap();
            assert_eq!(
                head.readout.task,
                crate::classifier_head::ClassifierTask::Sequence
            );
            assert_eq!(head.readout.width, 1);
            assert!(head.first.is_some());
            assert!(crate::classifier_head::validate_admission(&fixture.0, None).is_ok());
        }

        let token = Fixture::new();
        token.config(serde_json::json!({
            "architectures": ["NewForTokenClassification"]
        }));
        token.weights(
            "",
            Dtype::F32,
            &[],
            &[
                ("classifier.weight", vec![1, HIDDEN]),
                ("classifier.bias", vec![1]),
            ],
        );
        let (token_weights, token_checkpoint) = BertWeights::load_classifier(&token.0).unwrap();
        assert!(crate::classifier_head::ClassifierHeadHost::load(
            &token_checkpoint,
            &token_weights.config,
        )
        .is_err());
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
            (serde_json::json!({"type_vocab_size": 3}), "type_vocab_size"),
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
