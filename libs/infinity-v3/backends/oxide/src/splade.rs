// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Host-side loading and bounded geometry checks for a BF16 BERT SPLADE head.

use crate::{
    infer,
    model::{
        BertConfig, Checkpoint, EncoderFamily, FeedForwardRecipe, LinearWeights, NormArithmetic,
        PositionRecipe, WeightSource,
    },
    start,
};
use infinity_v3_backend_core::BackendError;
use std::mem::size_of;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum SpladeDecoderHost {
    TiedWord,
    Explicit(Box<WeightSource>),
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct SpladeHeadHost {
    pub transform: LinearWeights,
    pub gamma: WeightSource,
    pub beta: WeightSource,
    pub decoder: SpladeDecoderHost,
    pub decoder_bias: WeightSource,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct SpladeGeometry {
    pub tokens: usize,
    pub pooled: usize,
    pub raw_rows: usize,
    pub scratch_elements: usize,
    pub scratch_bytes: usize,
    pub logits_elements: usize,
    pub logits_bytes: usize,
    pub output_elements: usize,
    pub output_bytes: usize,
    pub raw_elements: usize,
    pub raw_bytes: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl SpladeHeadHost {
    pub(crate) fn validate_config(config: &BertConfig) -> Result<(), BackendError> {
        if config.family != EncoderFamily::Bert {
            return Err(start("SPLADE head requires the BERT encoder family"));
        }
        if config.hidden_size != 768 {
            return Err(start(format!(
                "SPLADE head requires hidden_size 768, got {}",
                config.hidden_size
            )));
        }
        if config.num_attention_heads != 12 {
            return Err(start(format!(
                "SPLADE head requires 12 attention heads, got {}",
                config.num_attention_heads
            )));
        }
        if config.intermediate_size != 3072 {
            return Err(start(format!(
                "SPLADE head requires intermediate_size 3072, got {}",
                config.intermediate_size
            )));
        }
        if config.num_hidden_layers == 0 {
            return Err(start("SPLADE head requires at least one hidden layer"));
        }
        if config.vocab_size == 0 {
            return Err(start("SPLADE head requires a positive vocabulary size"));
        }
        if config.architectures.len() != 1
            || config.architectures.first().map(String::as_str) != Some("BertForMaskedLM")
        {
            return Err(start(
                "SPLADE head requires exactly one BertForMaskedLM architecture",
            ));
        }
        if config.recipe.embedding != NormArithmetic::ExactBiased
            || config.recipe.residual != NormArithmetic::ExactBiased
        {
            return Err(start(
                "SPLADE head requires exact-biased embedding and residual arithmetic",
            ));
        }
        if config.recipe.positions != PositionRecipe::Absolute {
            return Err(start("SPLADE head requires absolute positions"));
        }
        if config.recipe.feed_forward != FeedForwardRecipe::DenseGelu {
            return Err(start("SPLADE head requires dense GELU feed-forward"));
        }
        let eps = config.layer_norm_eps as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(start(format!(
                "SPLADE head requires a positive finite f32 epsilon, got {}",
                config.layer_norm_eps
            )));
        }

        Ok(())
    }

    pub(crate) fn load(
        checkpoint: &Checkpoint,
        config: &BertConfig,
        word: &WeightSource,
    ) -> Result<Self, BackendError> {
        Self::validate_config(config)?;
        let eps = config.layer_norm_eps as f32;

        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        match word.shape() {
            [rows, cols] if *rows == vocab && *cols == hidden => {}
            shape => {
                return Err(start(format!(
                    "wrong word embedding shape: expected {vocab}x{hidden}, got {shape:?}"
                )))
            }
        }

        let transform = LinearWeights {
            weight: checkpoint
                .source("cls.predictions.transform.dense.weight", &[hidden, hidden])?,
            bias: checkpoint.source("cls.predictions.transform.dense.bias", &[hidden])?,
        };
        let gamma = checkpoint.source("cls.predictions.transform.LayerNorm.weight", &[hidden])?;
        let beta = checkpoint.source("cls.predictions.transform.LayerNorm.bias", &[hidden])?;
        let decoder_bias = checkpoint.source("cls.predictions.bias", &[vocab])?;
        let decoder =
            match checkpoint.optional_source("cls.predictions.decoder.weight", &[vocab, hidden])? {
                Some(source) => SpladeDecoderHost::Explicit(Box::new(source)),
                None => SpladeDecoderHost::TiedWord,
            };
        // Native uses cls.predictions.bias; an unused decoder.bias alias must
        // neither replace it nor change admission when it has another shape.

        Ok(Self {
            transform,
            gamma,
            beta,
            decoder,
            decoder_bias,
            hidden,
            vocab,
            eps,
        })
    }

    pub(crate) fn geometry(
        &self,
        tokens: usize,
        pooled: usize,
        raw_rows: usize,
        max_logits_bytes: usize,
    ) -> Result<SpladeGeometry, BackendError> {
        if tokens == 0 && (pooled != 0 || raw_rows != 0) {
            return Err(infer("SPLADE selections require at least one input token"));
        }
        check_launch(tokens, "token")?;
        check_launch(pooled, "pooled row")?;
        check_launch(raw_rows, "raw row")?;

        let raw_elements = checked_elements(raw_rows, self.hidden, "raw output")?;
        let raw_bytes = checked_bytes(raw_elements, "raw output")?;

        let (
            scratch_elements,
            scratch_bytes,
            logits_elements,
            logits_bytes,
            output_elements,
            output_bytes,
        ) = if pooled == 0 {
            (0, 0, 0, 0, 0, 0)
        } else {
            let scratch_elements = checked_elements(tokens, self.hidden, "SPLADE scratch")?;
            let scratch_bytes = checked_storage_bytes(scratch_elements, "SPLADE scratch")?;
            let logits_elements = checked_elements(tokens, self.vocab, "SPLADE logits")?;
            let logits_bytes = checked_storage_bytes(logits_elements, "SPLADE logits")?;
            if logits_bytes > max_logits_bytes {
                return Err(infer(format!(
                    "SPLADE logits need {logits_bytes} bytes, budget is {max_logits_bytes}"
                )));
            }
            let output_elements = checked_elements(pooled, self.vocab, "SPLADE output")?;
            let output_bytes = checked_bytes(output_elements, "SPLADE output")?;
            (
                scratch_elements,
                scratch_bytes,
                logits_elements,
                logits_bytes,
                output_elements,
                output_bytes,
            )
        };

        Ok(SpladeGeometry {
            tokens,
            pooled,
            raw_rows,
            scratch_elements,
            scratch_bytes,
            logits_elements,
            logits_bytes,
            output_elements,
            output_bytes,
            raw_elements,
            raw_bytes,
        })
    }
}

fn check_launch(value: usize, label: &str) -> Result<(), BackendError> {
    if value > u32::MAX as usize {
        return Err(infer(format!("{label} count exceeds u32 launch range")));
    }
    if value > i32::MAX as usize {
        return Err(infer(format!("{label} count exceeds i32 index range")));
    }
    Ok(())
}

fn checked_elements(rows: usize, cols: usize, label: &str) -> Result<usize, BackendError> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| infer(format!("{label} element count overflow")))?;
    if elements > i32::MAX as usize {
        return Err(infer(format!("{label} exceeds i32 indexing range")));
    }
    Ok(elements)
}

fn checked_bytes(elements: usize, label: &str) -> Result<usize, BackendError> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| infer(format!("{label} byte size overflow")))
}

fn checked_storage_bytes(elements: usize, label: &str) -> Result<usize, BackendError> {
    elements
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| infer(format!("{label} byte size overflow")))
}

/// Explicit max/ReLU SPLADE pipelines only. Ordinary embedding module
/// validation stays unchanged, and a missing optional pipeline does not hide
/// the required masked-LM tensors checked by the head loader.
pub(crate) fn validate_pipeline(path: &std::path::Path) -> Result<(), BackendError> {
    use serde::Deserialize;
    use std::path::{Component, Path};
    if !crate::model::path_present(&path.join("modules.json"))? {
        return Ok(());
    }
    #[derive(Deserialize)]
    struct Module {
        #[serde(rename = "type")]
        kind: String,
        path: Option<String>,
    }
    let modules: Vec<Module> = crate::model::read_json(&path.join("modules.json"))?;
    let [transformer, pooling] = modules.as_slice() else {
        return Err(start(
            "SPLADE requires exactly an MLM Transformer and max/ReLU SpladePooling module",
        ));
    };
    if !matches!(
        transformer.kind.as_str(),
        "sentence_transformers.models.Transformer"
            | "sentence_transformers.sparse_encoder.models.MLMTransformer"
    ) || !matches!(transformer.path.as_deref(), None | Some("" | "."))
        || pooling.kind != "sentence_transformers.sparse_encoder.models.SpladePooling"
    {
        return Err(start(
            "unsupported SPLADE pipeline type/order or Transformer path",
        ));
    }
    let relative = Path::new(
        pooling
            .path
            .as_deref()
            .ok_or_else(|| start("SPLADE pooling module path is missing"))?,
    );
    let mut components = relative.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(start(
            "SPLADE pooling module path must be one relative directory",
        ));
    }
    let root = path.canonicalize().map_err(start)?;
    let directory = root.join(relative).canonicalize().map_err(start)?;
    if !directory.starts_with(&root) {
        return Err(start("SPLADE pooling module escapes checkpoint root"));
    }
    #[derive(Deserialize)]
    struct Pooling {
        pooling_strategy: String,
        activation_function: String,
    }
    let config: Pooling = crate::model::read_json(&directory.join("config.json"))?;
    if config.pooling_strategy != "max" || config.activation_function != "relu" {
        return Err(start("SPLADE requires max pooling and ReLU activation"));
    }
    Ok(())
}

pub(crate) fn validate_admission(
    path: &std::path::Path,
    dense_paths: Option<&[String]>,
) -> Result<(), BackendError> {
    let config = BertConfig::load(path)?;
    SpladeHeadHost::validate_config(&config)?;
    // Exact public metadata for the documented BERT SPLADE rows: H768/L12/
    // Q12/I3072/V30522, P512, two type rows and eps=1e-12. This is a metadata
    // contract, not a checkpoint-name allowlist.
    if config.hidden_size != 768
        || config.intermediate_size != 3072
        || config.num_attention_heads != 12
        || config.num_hidden_layers != 12
        || config.max_position_embeddings != 512
        || config.vocab_size != 30522
        || config.type_vocab_size != 2
        || config.layer_norm_eps != 1e-12
    {
        return Err(start(format!(
            "BF16 BERT SPLADE requires the documented H768/I3072/Q12/L12/V30522/P512 metadata with eps=1e-12; got H{}/I{}/Q{}/L{}/V{}/P{}/eps={}",
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.vocab_size,
            config.max_position_embeddings,
            config.layer_norm_eps
        )));
    }
    if dense_paths.is_some_and(|paths| !paths.is_empty()) {
        return Err(start("SPLADE Dense projection chains are not implemented"));
    }
    validate_pipeline(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    struct ConfigSpec {
        model_type: &'static str,
        architectures: &'static str,
        hidden: usize,
        heads: usize,
        intermediate: usize,
        layers: usize,
        vocab: usize,
        eps: &'static str,
    }

    fn config_spec() -> ConfigSpec {
        ConfigSpec {
            model_type: "bert",
            architectures: "[\"BertForMaskedLM\"]",
            hidden: 768,
            heads: 12,
            intermediate: 3072,
            layers: 1,
            vocab: 7,
            eps: "1e-5",
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxide-splade-{label}-{}-{nanos}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_config(dir: &Path, spec: &ConfigSpec) {
        let ConfigSpec {
            model_type,
            architectures,
            hidden,
            heads,
            intermediate,
            layers,
            vocab,
            eps,
        } = spec;
        let json = format!(
            r#"{{
                "model_type": "{model_type}",
                "architectures": {architectures},
                "hidden_size": {hidden},
                "intermediate_size": {intermediate},
                "num_attention_heads": {heads},
                "num_hidden_layers": {layers},
                "max_position_embeddings": 16,
                "vocab_size": {vocab},
                "type_vocab_size": 2,
                "layer_norm_eps": {eps},
                "hidden_act": "gelu",
                "position_embedding_type": "absolute"
            }}"#
        );
        fs::write(dir.join("config.json"), json).unwrap();
    }

    fn tensor(name: &str, shape: &[usize]) -> (String, Vec<usize>) {
        (name.to_string(), shape.to_vec())
    }

    fn head_tensors(
        hidden: usize,
        vocab: usize,
        decoder_hidden: Option<usize>,
    ) -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            tensor("cls.predictions.transform.dense.weight", &[hidden, hidden]),
            tensor("cls.predictions.transform.dense.bias", &[hidden]),
            tensor("cls.predictions.transform.LayerNorm.weight", &[hidden]),
            tensor("cls.predictions.transform.LayerNorm.bias", &[hidden]),
            tensor("cls.predictions.bias", &[vocab]),
        ];
        if let Some(decoder_hidden) = decoder_hidden {
            tensors.push(tensor(
                "cls.predictions.decoder.weight",
                &[vocab, decoder_hidden],
            ));
        }
        tensors
    }

    fn replace(tensors: &mut [(String, Vec<usize>)], name: &str, shape: &[usize]) {
        for (tensor_name, tensor_shape) in tensors {
            if tensor_name == name {
                *tensor_shape = shape.to_vec();
            }
        }
    }

    fn write_safetensors(dir: &Path, tensors: &[(String, Vec<usize>)]) {
        let mut entries = Vec::new();
        let mut offset = 0usize;
        for (name, shape) in tensors {
            let count = shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .unwrap();
            let end = offset + count * 2;
            entries.push(format!(
                r#""{name}":{{"dtype":"BF16","shape":{shape:?},"data_offsets":[{offset},{end}]}}"#
            ));
            offset = end;
        }
        let mut header = format!("{{{}}}", entries.join(","));
        while header.len() & 7 != 0 {
            header.push(' ');
        }
        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(8 + header.len() + offset, 0);
        fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    fn checkpoint_at(dir: &Path) -> Checkpoint {
        Checkpoint::open(dir).unwrap()
    }

    fn load_at(dir: &Path, hidden: usize, vocab: usize) -> Result<SpladeHeadHost, BackendError> {
        let config = BertConfig::load(dir).unwrap();
        let checkpoint = checkpoint_at(dir);
        let word = WeightSource::zeroed(vec![vocab, hidden]).unwrap();
        SpladeHeadHost::load(&checkpoint, &config, &word)
    }

    fn geometry_head(hidden: usize, vocab: usize) -> SpladeHeadHost {
        SpladeHeadHost {
            transform: LinearWeights {
                weight: WeightSource::zeroed(vec![1, 1]).unwrap(),
                bias: WeightSource::zeroed(vec![1]).unwrap(),
            },
            gamma: WeightSource::zeroed(vec![1]).unwrap(),
            beta: WeightSource::zeroed(vec![1]).unwrap(),
            decoder: SpladeDecoderHost::TiedWord,
            decoder_bias: WeightSource::zeroed(vec![1]).unwrap(),
            hidden,
            vocab,
            eps: 1e-5,
        }
    }

    #[test]
    fn decoder_precedence_ties_only_when_decoder_is_absent() {
        let dir = temp_dir("tied");
        write_config(&dir, &config_spec());
        write_safetensors(&dir, &head_tensors(768, 7, None));
        let tied = load_at(&dir, 768, 7).unwrap();
        assert!(matches!(&tied.decoder, SpladeDecoderHost::TiedWord));
        assert_eq!(tied.transform.weight.shape(), &[768, 768]);
        assert_eq!(tied.gamma.shape(), &[768]);
        assert_eq!(tied.beta.shape(), &[768]);
        assert_eq!(tied.decoder_bias.shape(), &[7]);

        let dir = temp_dir("explicit");
        write_config(&dir, &config_spec());
        write_safetensors(&dir, &head_tensors(768, 7, Some(768)));
        let explicit = load_at(&dir, 768, 7).unwrap();
        match &explicit.decoder {
            SpladeDecoderHost::Explicit(source) => assert_eq!(source.shape(), &[7, 768]),
            SpladeDecoderHost::TiedWord => panic!("explicit decoder was silently tied"),
        }
    }

    #[test]
    fn rejects_malformed_head_tensors() {
        let spec = config_spec();

        let dir = temp_dir("bad-decoder");
        write_config(&dir, &spec);
        write_safetensors(&dir, &head_tensors(768, 7, Some(767)));
        assert!(load_at(&dir, 768, 7).is_err());

        let dir = temp_dir("alias-only");
        write_config(&dir, &spec);
        let mut tensors = head_tensors(768, 7, None);
        tensors.retain(|(name, _)| name != "cls.predictions.bias");
        tensors.push(tensor("cls.predictions.decoder.bias", &[7]));
        write_safetensors(&dir, &tensors);
        assert!(load_at(&dir, 768, 7).is_err());

        let dir = temp_dir("unused-alias");
        write_config(&dir, &spec);
        let mut tensors = head_tensors(768, 7, Some(768));
        tensors.push(tensor("cls.predictions.decoder.bias", &[6]));
        write_safetensors(&dir, &tensors);
        assert!(load_at(&dir, 768, 7).is_ok());

        let dir = temp_dir("bad-affine");
        write_config(&dir, &spec);
        let mut tensors = head_tensors(768, 7, None);
        replace(&mut tensors, "cls.predictions.transform.dense.bias", &[767]);
        write_safetensors(&dir, &tensors);
        assert!(load_at(&dir, 768, 7).is_err());
    }

    #[test]
    fn rejects_wrong_or_non_bert_head_geometry() {
        let dir = temp_dir("wrong-geometry");
        write_config(&dir, &config_spec());
        write_safetensors(&dir, &head_tensors(768, 7, None));
        let config = BertConfig::load(&dir).unwrap();
        let checkpoint = checkpoint_at(&dir);
        let word = WeightSource::zeroed(vec![7, 768]).unwrap();
        let check = |config: &BertConfig| SpladeHeadHost::load(&checkpoint, config, &word).is_err();

        assert!(!check(&config));
        let cases: [fn(&mut BertConfig); 9] = [
            |c| c.hidden_size = 384,
            |c| c.num_attention_heads = 16,
            |c| c.intermediate_size = 1,
            |c| c.num_hidden_layers = 0,
            |c| c.vocab_size = 0,
            |c| c.architectures.push("BertModel".to_string()),
            |c| c.family = EncoderFamily::Roberta,
            |c| c.recipe.embedding = NormArithmetic::RoundedAddBeforeStats,
            |c| c.vocab_size = 8,
        ];
        for mutate in cases {
            let mut wrong = config.clone();
            mutate(&mut wrong);
            assert!(check(&wrong));
        }
    }

    #[test]
    fn validates_epsilon_and_word_shape() {
        let dir = temp_dir("eps");
        write_config(&dir, &config_spec());
        write_safetensors(&dir, &head_tensors(768, 7, None));
        let config = BertConfig::load(&dir).unwrap();
        let checkpoint = checkpoint_at(&dir);
        let word = WeightSource::zeroed(vec![7, 768]).unwrap();
        let head = SpladeHeadHost::load(&checkpoint, &config, &word).unwrap();
        assert!(head.eps > 0.0 && head.eps.is_finite());
        assert_eq!(head.eps, 1e-5f32);
        assert!(load_at(&dir, 768, 6).is_err());

        for eps in [0.0, f64::INFINITY, 1e300] {
            let mut wrong = config.clone();
            wrong.layer_norm_eps = eps;
            assert!(SpladeHeadHost::load(&checkpoint, &wrong, &word).is_err());
        }
    }

    #[test]
    fn geometry_separates_hidden_vocab_and_raw_rows() {
        let head = geometry_head(5, 7);
        let plan = head.geometry(3, 2, 4, 1024).unwrap();
        assert_eq!(plan.tokens, 3);
        assert_eq!(plan.pooled, 2);
        assert_eq!(plan.raw_rows, 4);
        assert_eq!((plan.scratch_elements, plan.scratch_bytes), (15, 30));
        assert_eq!((plan.logits_elements, plan.logits_bytes), (21, 42));
        assert_eq!((plan.output_elements, plan.output_bytes), (14, 56));
        assert_eq!((plan.raw_elements, plan.raw_bytes), (20, 80));
    }

    #[test]
    fn geometry_raw_only_ignores_vocab_and_logits_budget() {
        let head = geometry_head(5, usize::MAX);
        let plan = head.geometry(9, 0, 4, 0).unwrap();
        assert_eq!(plan.scratch_elements, 0);
        assert_eq!(plan.scratch_bytes, 0);
        assert_eq!(plan.logits_elements, 0);
        assert_eq!(plan.logits_bytes, 0);
        assert_eq!(plan.output_elements, 0);
        assert_eq!(plan.output_bytes, 0);
        assert_eq!((plan.raw_elements, plan.raw_bytes), (20, 80));
        assert!(head.geometry(9, 1, 0, usize::MAX).is_err());
    }

    #[test]
    fn geometry_rejects_budget_and_overflow_at_exact_limit() {
        let head = geometry_head(5, 7);
        assert!(head.geometry(10, 2, 0, 140).is_ok());
        assert!(head.geometry(10, 2, 0, 139).is_err());

        let huge = geometry_head(usize::MAX / 2, 7);
        assert!(huge.geometry(4, 1, 0, usize::MAX).is_err());
        assert!(head.geometry((u32::MAX as usize) + 1, 0, 0, 0).is_err());
    }

    #[test]
    fn geometry_zero_selection_is_valid() {
        let head = geometry_head(5, 7);
        let plan = head.geometry(0, 0, 0, 0).unwrap();
        assert_eq!(plan.scratch_elements, 0);
        assert_eq!(plan.logits_elements, 0);
        assert_eq!(plan.output_elements, 0);
        assert_eq!(plan.raw_elements, 0);
        assert_eq!(plan.raw_bytes, 0);

        assert!(head.geometry(0, 1, 0, 0).is_err());
        assert!(head.geometry(0, 0, 1, 0).is_err());
        let raw_only = head.geometry(8, 0, 0, 0).unwrap();
        assert_eq!(raw_only.scratch_bytes, 0);
        assert_eq!(raw_only.logits_bytes, 0);
        assert_eq!(raw_only.raw_bytes, 0);
    }
    fn write_pipeline(dir: &Path, activation: &str, strategy: &str) {
        let pool = dir.join("1_SpladePooling");
        fs::create_dir_all(&pool).unwrap();
        fs::write(pool.join("config.json"), serde_json::to_vec(&serde_json::json!({
            "activation_function":activation,"pooling_strategy":strategy,"word_embedding_dimension":null
        })).unwrap()).unwrap();
        fs::write(dir.join("modules.json"), serde_json::to_vec(&serde_json::json!([
            {"type":"sentence_transformers.sparse_encoder.models.MLMTransformer","path":""},
            {"type":"sentence_transformers.sparse_encoder.models.SpladePooling","path":"1_SpladePooling"}
        ])).unwrap()).unwrap();
    }

    fn write_documented_config(dir: &Path) {
        fs::write(
            dir.join("config.json"),
            br#"{
            "model_type":"bert","architectures":["BertForMaskedLM"],
            "hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,
            "num_hidden_layers":12,"max_position_embeddings":512,"vocab_size":30522,
            "type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"gelu",
            "position_embedding_type":"absolute"
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn splade_admission_requires_the_documented_bert_metadata() {
        let dir = temp_dir("documented");
        write_documented_config(&dir);
        assert!(validate_admission(&dir, None).is_ok());
        assert!(validate_admission(&dir, Some(&["2_Dense".into()])).is_err());
        write_config(&dir, &config_spec());
        assert!(validate_admission(&dir, None).is_err());
    }

    #[test]
    fn splade_pipeline_is_explicit_and_does_not_widen_dense_embedding_modules() {
        let dir = temp_dir("pipeline");
        write_documented_config(&dir);
        assert!(validate_admission(&dir, None).is_ok());
        write_pipeline(&dir, "relu", "max");
        assert!(validate_pipeline(&dir).is_ok());
        assert!(validate_admission(&dir, Some(&[])).is_ok());
        assert!(validate_admission(&dir, Some(&["2_Dense".into()])).is_err());
        assert!(crate::model::validate_modules(&dir).is_err());
        write_pipeline(&dir, "sigmoid", "max");
        assert!(validate_pipeline(&dir).is_err());
        write_pipeline(&dir, "relu", "sum");
        assert!(validate_pipeline(&dir).is_err());
    }

    #[test]
    fn splade_pipeline_rejects_missing_paths_and_extra_or_reordered_modules() {
        let dir = temp_dir("bad-pipeline");
        write_config(&dir, &config_spec());
        write_pipeline(&dir, "relu", "max");
        let normal: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("modules.json")).unwrap()).unwrap();
        for value in [
            serde_json::json!([normal[1].clone(), normal[0].clone()]),
            serde_json::json!([normal[0].clone(),normal[1].clone(),{"type":"sentence_transformers.models.Normalize"}]),
            serde_json::json!([normal[0].clone(),{"type":"sentence_transformers.sparse_encoder.models.SpladePooling"}]),
            serde_json::json!([normal[0].clone(),{"type":"sentence_transformers.sparse_encoder.models.SpladePooling","path":"../outside"}]),
        ] {
            fs::write(
                dir.join("modules.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
            assert!(validate_pipeline(&dir).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn splade_pipeline_does_not_follow_an_escaping_pool_directory() {
        let dir = temp_dir("pool-escape");
        let outside = temp_dir("pool-outside");
        write_pipeline(&outside, "relu", "max");
        fs::write(
            dir.join("modules.json"),
            fs::read(outside.join("modules.json")).unwrap(),
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.join("1_SpladePooling"), dir.join("1_SpladePooling"))
            .unwrap();
        assert!(validate_pipeline(&dir).is_err());
    }
}
