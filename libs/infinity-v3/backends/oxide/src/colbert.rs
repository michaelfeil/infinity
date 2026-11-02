// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Host admission, tokenizer policy and CPU-side selection for BF16 BERT
//! ColBERT token-matrix serving.

use crate::{
    infer,
    model::{
        BertConfig, Checkpoint, EncoderFamily, FeedForwardRecipe, LinearWeights, NormArithmetic,
        PositionRecipe, WeightSource,
    },
    start,
};
use infinity_v3_backend_core::{BackendError, Batch, Embedding, Embeddings};
use std::path::{Component, Path};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColbertTokenizerPolicy {
    pub(crate) query_marker: Option<u32>,
    pub(crate) mask_token: Option<u32>,
    pub(crate) skip_tokens: Vec<u32>,
}

impl ColbertTokenizerPolicy {
    /// Read the same `[unused0]`, `[MASK]` and printable punctuation ids used by
    /// the Native LibTorch loader. A missing tokenizer leaves an empty policy.
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let file = path.join("tokenizer.json");
        if !crate::model::path_present(&file)? {
            return Ok(Self::default());
        }
        let tokenizer: serde_json::Value = crate::model::read_json(&file)?;
        let vocab = tokenizer
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| start("ColBERT tokenizer.json has no model.vocab object"))?;
        let id = |name: &str| -> Result<Option<u32>, BackendError> {
            vocab
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .map(u32::try_from)
                .transpose()
                .map_err(|_| start(format!("ColBERT tokenizer id for {name} exceeds u32")))
        };
        let query_marker = id("[unused0]")?;
        let mask_token = id("[MASK]")?;
        let mut skip_tokens = Vec::new();
        for byte in b'!'..=b'~' {
            if !byte.is_ascii_punctuation() {
                continue;
            }
            if let Some(token) = id(&(byte as char).to_string())? {
                skip_tokens.push(token);
            }
        }
        skip_tokens.sort_unstable();
        skip_tokens.dedup();
        Ok(Self {
            query_marker,
            mask_token,
            skip_tokens,
        })
    }

    pub(crate) fn is_punctuation(&self, token: u32) -> bool {
        self.skip_tokens.binary_search(&token).is_ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColbertGeometry {
    pub(crate) tokens: usize,
    pub(crate) sequences: usize,
    pub(crate) keys: usize,
    pub(crate) projected_elements: usize,
    pub(crate) projected_bytes: usize,
    pub(crate) key_index_bytes: usize,
    pub(crate) key_offset_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ColbertSelection {
    tokens: usize,
    sequences: usize,
    key_indices: Vec<u32>,
    key_offsets: Vec<u32>,
    excluded_keys: bool,
}

fn boundaries(batch: &Batch) -> Result<usize, BackendError> {
    let cumulative = &batch.cumulative_seq_lengths;
    if cumulative.first().copied() != Some(0)
        || cumulative.last().copied() != u32::try_from(batch.input_ids.len()).ok()
    {
        return Err(infer(
            "ColBERT cumulative lengths do not cover the input tokens",
        ));
    }
    if cumulative
        .windows(2)
        .any(|pair| pair.first().zip(pair.get(1)).is_some_and(|(a, b)| b <= a))
    {
        return Err(infer("ColBERT cumulative lengths are not monotonic"));
    }
    cumulative
        .len()
        .checked_sub(1)
        .ok_or_else(|| infer("ColBERT batch has no sequence boundaries"))
}

fn is_query(batch: &Batch, index: usize, policy: &ColbertTokenizerPolicy) -> bool {
    let Some((&start, &end)) = batch
        .cumulative_seq_lengths
        .get(index)
        .zip(batch.cumulative_seq_lengths.get(index + 1))
    else {
        return false;
    };
    let marker = policy.query_marker;
    marker.is_some_and(|marker| {
        end.saturating_sub(start) > 1
            && batch
                .input_ids
                .get(start as usize + 1)
                .is_some_and(|token| *token == marker)
    })
}

impl ColbertSelection {
    pub(crate) fn new(
        batch: &Batch,
        policy: &ColbertTokenizerPolicy,
    ) -> Result<Self, BackendError> {
        let sequences = boundaries(batch)?;
        let tokens = batch.input_ids.len();
        let mut key_indices = Vec::new();
        key_indices
            .try_reserve_exact(tokens)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        let mut key_offsets = Vec::new();
        key_offsets
            .try_reserve_exact(sequences + 1)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        key_offsets.push(0);
        let mut excluded_keys = false;
        for sequence in 0..sequences {
            let start = batch.cumulative_seq_lengths[sequence] as usize;
            let end = batch.cumulative_seq_lengths[sequence + 1] as usize;
            let query = is_query(batch, sequence, policy);
            for token in start..end {
                if query && policy.mask_token == Some(batch.input_ids[token]) {
                    excluded_keys = true;
                    continue;
                }
                key_indices.push(
                    u32::try_from(token)
                        .map_err(|_| infer("ColBERT key index exceeds u32 range"))?,
                );
            }
            key_offsets.push(
                u32::try_from(key_indices.len())
                    .map_err(|_| infer("ColBERT key count exceeds u32 range"))?,
            );
        }
        Ok(Self {
            tokens,
            sequences,
            key_indices,
            key_offsets,
            excluded_keys,
        })
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn sequences(&self) -> usize {
        self.sequences
    }

    pub(crate) fn keys(&self) -> usize {
        self.key_indices.len()
    }

    pub(crate) fn excludes_attention_keys(&self) -> bool {
        self.excluded_keys
    }

    pub(crate) fn key_indices(&self) -> &[u32] {
        &self.key_indices
    }

    pub(crate) fn key_offsets(&self) -> &[u32] {
        &self.key_offsets
    }

    pub(crate) fn geometry(&self, output_width: usize) -> Result<ColbertGeometry, BackendError> {
        let projected_elements = self
            .tokens
            .checked_mul(output_width)
            .ok_or_else(|| infer("ColBERT projected element count overflow"))?;
        if projected_elements > i32::MAX as usize {
            return Err(infer("ColBERT projected output exceeds i32 indexing range"));
        }
        let projected_bytes = projected_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| infer("ColBERT projected byte size overflow"))?;
        let key_index_bytes = self
            .key_indices
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| infer("ColBERT key index byte size overflow"))?;
        let key_offset_bytes = self
            .key_offsets
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| infer("ColBERT key offset byte size overflow"))?;
        Ok(ColbertGeometry {
            tokens: self.tokens,
            sequences: self.sequences,
            keys: self.key_indices.len(),
            projected_elements,
            projected_bytes,
            key_index_bytes,
            key_offset_bytes,
        })
    }
}

pub(crate) struct ColbertHeadHost {
    pub(crate) projection: LinearWeights,
    pub(crate) has_bias: bool,
    pub(crate) hidden: usize,
    pub(crate) output: usize,
    pub(crate) eps: f32,
}

impl ColbertHeadHost {
    pub(crate) fn validate_config(config: &BertConfig) -> Result<(), BackendError> {
        if config.family != EncoderFamily::Bert {
            return Err(start("ColBERT requires the BERT encoder family"));
        }
        if config.hidden_size == 0 {
            return Err(start("ColBERT requires a positive hidden width"));
        }
        if !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        {
            return Err(start(
                "ColBERT hidden width must be divisible by its attention head count",
            ));
        }
        if config.vocab_size == 0 {
            return Err(start("ColBERT requires a positive vocabulary"));
        }
        if config.architectures.len() != 1
            || config.architectures.first().map(String::as_str) != Some("HF_ColBERT")
        {
            return Err(start(
                "ColBERT requires exactly one HF_ColBERT architecture",
            ));
        }
        // Exact public metadata for the two documented BERT ColBERT rows.
        // Geometry is authoritative; checkpoint names are not admission keys.
        let documented = matches!(
            (
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
            ),
            (384, 1536, 12) | (768, 3072, 12)
        ) && config.num_hidden_layers == 12
            && config.max_position_embeddings == 512
            && config.vocab_size == 30522
            && config.type_vocab_size == 2
            && config.layer_norm_eps == 1e-12;
        if !documented {
            return Err(start(format!(
                "ColBERT requires the documented BERT H384/I1536 or H768/I3072 metadata with V30522, L12, P512 and eps=1e-12; got H{}/I{}/Q{}/L{}/V{}/P{}/eps={}",
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_hidden_layers,
                config.vocab_size,
                config.max_position_embeddings,
                config.layer_norm_eps
            )));
        }
        if config.recipe.embedding != NormArithmetic::ExactBiased
            || config.recipe.residual != NormArithmetic::ExactBiased
        {
            return Err(start(
                "ColBERT requires exact-biased embedding and residual arithmetic",
            ));
        }
        if config.recipe.positions != PositionRecipe::Absolute {
            return Err(start("ColBERT requires absolute positions"));
        }
        if config.recipe.feed_forward != FeedForwardRecipe::DenseGelu {
            return Err(start("ColBERT requires dense GELU feed-forward"));
        }
        Ok(())
    }

    pub(crate) fn load(checkpoint: &Checkpoint, config: &BertConfig) -> Result<Self, BackendError> {
        Self::validate_config(config)?;
        let hidden = config.hidden_size;
        let mut projection = None;
        for name in ["embedding_proj_layer.weight", "linear.weight"] {
            let Some(shape) = checkpoint.shape(name) else {
                continue;
            };
            let [output, input] = shape else {
                return Err(start(format!(
                    "ColBERT token projection {name} must be a matrix, got {shape:?}"
                )));
            };
            if *input != hidden || *output == 0 {
                return Err(start(format!(
                    "ColBERT token projection {name} expects [output,{hidden}], got {shape:?}"
                )));
            }
            let weight = checkpoint.source(name, shape)?;
            let bias_name = name.trim_end_matches(".weight").to_owned() + ".bias";
            let source = checkpoint.optional_source(&bias_name, &[*output])?;
            let has_bias = source.is_some();
            let bias = match source {
                Some(source) => source,
                None => WeightSource::zeroed(vec![*output])?,
            };
            projection = Some((LinearWeights { weight, bias }, *output, has_bias));
            break;
        }
        let (projection, output, has_bias) = projection
            .ok_or_else(|| start("ColBERT checkpoint is missing its token projection"))?;
        Ok(Self {
            projection,
            has_bias,
            hidden,
            output,
            // MultiVectorModel's default; ColBERT does not use the ColQwen3
            // model-dtype epsilon variant.
            eps: 1e-12,
        })
    }
}

pub(crate) fn validate_modules(path: &Path) -> Result<(), BackendError> {
    let file = path.join("modules.json");
    if !crate::model::path_present(&file)? {
        return Ok(());
    }
    #[derive(serde::Deserialize)]
    struct Module {
        #[serde(rename = "type")]
        kind: String,
        path: Option<String>,
    }
    let modules: Vec<Module> = crate::model::read_json(&file)?;
    for module in &modules {
        let kind = module.kind.as_str();
        let accepted = matches!(
            kind,
            "sentence_transformers.models.Transformer"
                | "sentence_transformers.models.Pooling"
                | "sentence_transformers.models.Normalize"
                | "sentence_transformers.multi_vector_encoder.modules.MultiVectorMask"
        );
        if !accepted {
            return Err(start(format!(
                "unsupported ColBERT sentence-transformers module {kind}"
            )));
        }
        if let Some(value) = module.path.as_deref() {
            let relative = Path::new(value);
            if !relative.as_os_str().is_empty()
                && relative
                    .components()
                    .any(|part| !matches!(part, Component::Normal(_)))
            {
                return Err(start("ColBERT module path must stay inside the checkpoint"));
            }
        }
    }
    Ok(())
}

pub(crate) fn is_checkpoint(path: &Path) -> Result<bool, BackendError> {
    let value: serde_json::Value = crate::model::read_json(&path.join("config.json"))?;
    Ok(value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .any(|value| value.as_str() == Some("HF_ColBERT"))
        }))
}

pub(crate) fn validate_admission(
    path: &Path,
    dense_paths: Option<&[String]>,
) -> Result<(), BackendError> {
    let config = BertConfig::load(path)?;
    ColbertHeadHost::validate_config(&config)?;
    if dense_paths.is_some_and(|paths| !paths.is_empty()) {
        return Err(start(
            "ColBERT uses its learned token projection, not a Dense module chain",
        ));
    }
    validate_modules(path)
}

pub(crate) fn validate_request(batch: &Batch) -> Result<(), BackendError> {
    if !batch.pooled_indices.is_empty() {
        return Err(infer(
            "ColBERT returns token embeddings; pooling is unsupported",
        ));
    }
    Ok(())
}

/// Pack the already projected/normalized token rows, removing document
/// punctuation rows exactly once while keeping every query token.
pub(crate) fn pack(
    output: &[f32],
    output_width: usize,
    batch: &Batch,
    policy: &ColbertTokenizerPolicy,
) -> Result<Embeddings, BackendError> {
    validate_request(batch)?;
    if output_width == 0 {
        return Err(infer("Invalid ColBERT token output width"));
    }
    let rows = crate::output::raw_rows(batch)?;
    let expected = rows
        .checked_mul(output_width)
        .ok_or_else(|| infer("ColBERT host output size overflow"))?;
    if output.len() != expected {
        return Err(infer("ColBERT host output length mismatch"));
    }
    let mut results = Embeddings::default();
    results
        .try_reserve(batch.raw_indices.len())
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    let mut rest = output;
    for &index in &batch.raw_indices {
        let index = index as usize;
        let start = *batch
            .cumulative_seq_lengths
            .get(index)
            .ok_or_else(|| infer("Invalid ColBERT raw output index"))? as usize;
        let end = *batch
            .cumulative_seq_lengths
            .get(index + 1)
            .ok_or_else(|| infer("Invalid ColBERT raw output boundaries"))?
            as usize;
        let length = end
            .checked_sub(start)
            .filter(|length| *length > 0)
            .ok_or_else(|| infer("Invalid ColBERT raw output boundaries"))?;
        let row_elements = length
            .checked_mul(output_width)
            .ok_or_else(|| infer("ColBERT raw output size overflow"))?;
        let (values, tail) = rest
            .split_at_checked(row_elements)
            .ok_or_else(|| infer("ColBERT raw output buffer is too short"))?;
        let query = is_query(batch, index, policy);
        let input_tokens = batch
            .input_ids
            .get(start..end)
            .ok_or_else(|| infer("Invalid ColBERT raw output token span"))?;
        let mut keep = Vec::new();
        keep.try_reserve_exact(length)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for token in input_tokens {
            keep.push(query || !policy.is_punctuation(*token));
        }
        let kept = keep.iter().filter(|keep| **keep).count();
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(kept)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for (row, keep) in values.chunks_exact(output_width).zip(keep) {
            if keep {
                let mut copied = Vec::new();
                copied
                    .try_reserve_exact(output_width)
                    .map_err(|error| BackendError::Overloaded(error.to_string()))?;
                copied.extend_from_slice(row);
                tokens.push(copied);
            }
        }
        // Duplicate raw selectors follow the native map contract: the later
        // selection replaces the earlier one.
        results.insert(index, Embedding::All(tokens));
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(infer("Unexpected trailing ColBERT token output data"));
    }
    Ok(results)
}

#[cfg(test)]
pub(crate) fn l2_normalize_reference(
    rows: &[f32],
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, BackendError> {
    if width == 0 || !rows.len().is_multiple_of(width) {
        return Err(infer("invalid ColBERT L2 reference shape"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(infer("invalid ColBERT L2 epsilon"));
    }
    let mut out = Vec::with_capacity(rows.len());
    for row in rows.chunks_exact(width) {
        // This CPU reference checks model-dtype boundaries only. It is not an
        // oracle for ATen's GPU reduction order.
        let rounded: Vec<f32> = row
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_f32())
            .collect();
        let norm = rounded
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let divisor = half::bf16::from_f32(norm)
            .to_f32()
            .max(half::bf16::from_f32(eps).to_f32());
        out.extend(
            rounded
                .iter()
                .map(|value| half::bf16::from_f32(value / divisor).to_f32()),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BertConfig;
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxide-colbert-{label}-{}-{nanos}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn config(path: &Path, architectures: &[&str], hidden: usize) {
        let json = serde_json::json!({
            "model_type": "bert",
            "architectures": architectures,
            "hidden_size": hidden,
            "intermediate_size": hidden * 4,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "max_position_embeddings": 512,
            "vocab_size": 30522,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "position_embedding_type": "absolute"
        });
        fs::write(path.join("config.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    }

    fn write_safetensors(dir: &Path, tensors: &[(&str, &[usize])]) {
        let mut entries = Vec::new();
        let mut offset = 0usize;
        for (name, shape) in tensors {
            let count = shape.iter().product::<usize>();
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

    fn batch(lengths: &[usize], ids: &[u32]) -> Batch {
        let mut cumulative = vec![0];
        for &length in lengths {
            cumulative.push(cumulative.last().copied().unwrap() + length as u32);
        }
        Batch {
            multimodal: Vec::new(),
            input_ids: ids.to_vec(),
            token_type_ids: vec![0; ids.len()],
            position_ids: (0..ids.len()).map(|i| (i % 7) as u32).collect(),
            cumulative_seq_lengths: cumulative,
            max_length: *lengths.iter().max().unwrap() as u32,
            pooled_indices: Vec::new(),
            raw_indices: (0..lengths.len() as u32).collect(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn policy() -> ColbertTokenizerPolicy {
        ColbertTokenizerPolicy {
            query_marker: Some(1),
            mask_token: Some(2),
            skip_tokens: vec![3],
        }
    }

    #[test]
    fn selection_excludes_mask_only_from_query_attention_keys() {
        // doc: [CLS] . [SEP]; query: [CLS] [unused0] MASK . [SEP]
        let b = batch(&[3, 5], &[9, 3, 10, 9, 1, 2, 3, 10]);
        let selection = ColbertSelection::new(&b, &policy()).unwrap();
        assert_eq!(selection.key_indices(), &[0, 1, 2, 3, 4, 6, 7]);
        assert_eq!(selection.key_offsets(), &[0, 3, 7]);
        assert!(selection.excludes_attention_keys());
        assert_eq!(selection.tokens(), 8);
        assert_eq!(selection.sequences(), 2);
    }

    #[test]
    fn documents_drop_punctuation_output_rows_but_keep_them_as_keys() {
        let b = batch(&[3, 5], &[9, 3, 10, 9, 1, 2, 3, 10]);
        let selection = ColbertSelection::new(&b, &policy()).unwrap();
        assert_eq!(selection.key_indices(), &[0, 1, 2, 3, 4, 6, 7]);
        let width = 2;
        let mut output = Vec::new();
        for token in 0..8 {
            output.push(token as f32);
            output.push(token as f32 + 0.5);
        }
        let packed = pack(&output, width, &b, &policy()).unwrap();
        let Embedding::All(document) = packed.get(&0).unwrap() else {
            panic!("document must remain a token matrix")
        };
        assert_eq!(document.len(), 2, "document punctuation row removed");
        assert_eq!(document[0][0], 0.0);
        assert_eq!(document[1][0], 2.0);
        let Embedding::All(query) = packed.get(&1).unwrap() else {
            panic!("query must remain a token matrix")
        };
        assert_eq!(query.len(), 5, "query keeps punctuation and MASK output");
        assert_eq!(
            query.iter().map(|row| row[0]).collect::<Vec<_>>(),
            vec![3., 4., 5., 6., 7.]
        );
    }

    #[test]
    fn duplicate_raw_override_and_pooling_rejection_are_typed() {
        let mut b = batch(&[3], &[9, 3, 10]);
        b.raw_indices = vec![0, 0];
        let output: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let packed = pack(&output, 4, &b, &policy()).unwrap();
        let Embedding::All(rows) = packed.get(&0).unwrap() else {
            panic!("expected raw matrix")
        };
        assert_eq!(rows.len(), 2);
        // The later duplicate selection owns the map entry.
        assert_eq!(rows[0][0], 12.0);
        assert_eq!(rows[1][0], 20.0);
        b.pooled_indices = vec![0];
        assert!(validate_request(&b).is_err());
        assert!(pack(&output, 4, &b, &policy()).is_err());
    }

    #[test]
    fn tokenizer_policy_reads_documented_markers_and_punctuation() {
        let dir = temp_dir("tokenizer");
        let tokenizer = serde_json::json!({
            "model": {"vocab": {
                "[unused0]": 1,
                "[MASK]": 2,
                ".": 3,
                ",": 4,
                "word": 5
            }},
            "version": "1.0"
        });
        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        let policy = ColbertTokenizerPolicy::load(&dir).unwrap();
        assert_eq!(policy.query_marker, Some(1));
        assert_eq!(policy.mask_token, Some(2));
        assert_eq!(policy.skip_tokens, vec![3, 4]);
        assert!(policy.is_punctuation(3));
        assert!(!policy.is_punctuation(5));
    }

    fn loader_fixture(
        label: &str,
        tensors: &[(&str, &[usize])],
        architectures: &[&str],
        hidden: usize,
    ) -> (PathBuf, BertConfig, Checkpoint) {
        let dir = temp_dir(label);
        config(&dir, architectures, hidden);
        write_safetensors(&dir, tensors);
        let config = BertConfig::load(&dir).unwrap();
        let checkpoint = Checkpoint::open(&dir).unwrap();
        (dir, config, checkpoint)
    }

    #[test]
    fn learned_projection_prefers_native_name_and_accepts_optional_bias() {
        let (dir, config, checkpoint) = loader_fixture(
            "projection",
            &[("linear.weight", &[4, 384]), ("linear.bias", &[4])],
            &["HF_ColBERT"],
            384,
        );
        let head = ColbertHeadHost::load(&checkpoint, &config).unwrap();
        assert!(head.has_bias);
        assert_eq!((head.hidden, head.output), (384, 4));
        assert_eq!(head.projection.weight.shape(), &[4, 384]);
        assert_eq!(head.projection.bias.shape(), &[4]);
        assert!((head.eps - 1e-12).abs() < f32::EPSILON);

        let (dir2, config2, checkpoint2) = loader_fixture(
            "projection-no-bias",
            &[("linear.weight", &[4, 384])],
            &["HF_ColBERT"],
            384,
        );
        let head2 = ColbertHeadHost::load(&checkpoint2, &config2).unwrap();
        assert!(!head2.has_bias);
        assert_eq!(head2.projection.bias.shape(), &[4]);
        assert_eq!(head2.projection.bias.len(), 4);
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(dir2).unwrap();
    }

    #[test]
    fn malformed_or_missing_projection_and_geometry_are_rejected() {
        let (dir, config, checkpoint) = loader_fixture(
            "bad-projection",
            &[("linear.weight", &[4, 383])],
            &["HF_ColBERT"],
            384,
        );
        assert!(ColbertHeadHost::load(&checkpoint, &config).is_err());
        fs::remove_dir_all(dir).unwrap();

        let (dir, config, checkpoint) = loader_fixture("missing", &[], &["HF_ColBERT"], 384);
        assert!(ColbertHeadHost::load(&checkpoint, &config).is_err());
        fs::remove_dir_all(dir).unwrap();

        let (dir, config, checkpoint) = loader_fixture(
            "wrong-architecture",
            &[("linear.weight", &[4, 384])],
            &["BertModel"],
            384,
        );
        assert!(ColbertHeadHost::load(&checkpoint, &config).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn l2_reference_preserves_model_dtype_boundaries_and_zero_rows() {
        let rows = [3.0f32, 4.0, 0.0, 0.0];
        let normalized = l2_normalize_reference(&rows, 2, 1e-12).unwrap();
        // BF16 quotients, widened for the result API. An F32-only division
        // gives 0.6/0.8 and fails this concrete regression.
        assert_eq!(normalized[0], 0.6015625);
        assert_eq!(normalized[1], 0.80078125);
        assert_eq!(normalized[2], 0.0);
        assert_eq!(normalized[3], 0.0);
        // The BF16 norm itself is materialized before division: sqrt(58)
        // rounds to 7.625. Rounding only the final quotient gives 0.39453125.
        assert_eq!(
            l2_normalize_reference(&[3.0, 7.0], 2, 1e-12).unwrap(),
            [half::bf16::from_bits(0x3ec9).to_f32(), 0.91796875]
        );
        assert!(l2_normalize_reference(&rows, 3, 1e-12).is_err());
        assert!(l2_normalize_reference(&rows, 2, 0.0).is_err());
    }

    #[test]
    fn both_documented_colbert_geometries_are_admitted_by_exact_metadata() {
        for hidden in [384usize, 768] {
            let dir = temp_dir(&format!("geometry-{hidden}"));
            config(&dir, &["HF_ColBERT"], hidden);
            let config = BertConfig::load(&dir).unwrap();
            assert!(ColbertHeadHost::validate_config(&config).is_ok());
            fs::remove_dir_all(dir).unwrap();
        }
        let dir = temp_dir("unsupported-geometry");
        config(&dir, &["HF_ColBERT"], 576);
        let config = BertConfig::load(&dir).unwrap();
        assert!(ColbertHeadHost::validate_config(&config).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn documented_checkpoint_ids_are_present_in_the_public_catalog() {
        let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/supported-models.md");
        let text = fs::read_to_string(docs).unwrap();
        for id in [
            "colbert-ir/colbertv2.0",
            "answerdotai/answerai-colbert-small-v1",
        ] {
            assert!(text.contains(id), "missing documented ColBERT id {id}");
        }
    }
}
