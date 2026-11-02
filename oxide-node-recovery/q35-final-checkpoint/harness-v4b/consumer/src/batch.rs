use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAX_BATCH_JSON: u64 = 32 << 20;
pub const BATCH_FIELDS: [&str; 7] = [
    "input_ids",
    "position_ids",
    "token_type_ids",
    "cumulative_seq_lengths",
    "pooled_indices",
    "raw_indices",
    "max_length",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchJson {
    pub input_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub pooled_indices: Vec<u32>,
    pub raw_indices: Vec<u32>,
    pub max_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchError(pub String);

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BatchError {}

impl BatchError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

#[derive(Clone, Copy)]
enum Field {
    InputIds,
    PositionIds,
    TokenTypeIds,
    CumulativeSeqLengths,
    PooledIndices,
    RawIndices,
    MaxLength,
    Unknown,
}

struct FieldVisitor;

impl<'de> Visitor<'de> for FieldVisitor {
    type Value = Field;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("one of the seven physical batch fields")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(match value {
            "input_ids" => Field::InputIds,
            "position_ids" => Field::PositionIds,
            "token_type_ids" => Field::TokenTypeIds,
            "cumulative_seq_lengths" => Field::CumulativeSeqLengths,
            "pooled_indices" => Field::PooledIndices,
            "raw_indices" => Field::RawIndices,
            "max_length" => Field::MaxLength,
            _ => Field::Unknown,
        })
    }
}

impl<'de> Deserialize<'de> for Field {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(FieldVisitor)
    }
}

struct BatchVisitor;

impl<'de> Visitor<'de> for BatchVisitor {
    type Value = BatchJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an exact seven-field physical batch object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut input_ids: Option<Vec<u32>> = None;
        let mut position_ids: Option<Vec<u32>> = None;
        let mut token_type_ids: Option<Vec<u32>> = None;
        let mut cumulative_seq_lengths: Option<Vec<u32>> = None;
        let mut pooled_indices: Option<Vec<u32>> = None;
        let mut raw_indices: Option<Vec<u32>> = None;
        let mut max_length: Option<u32> = None;

        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::InputIds => {
                    if input_ids.is_some() {
                        return Err(de::Error::duplicate_field("input_ids"));
                    }
                    input_ids = Some(map.next_value()?);
                }
                Field::PositionIds => {
                    if position_ids.is_some() {
                        return Err(de::Error::duplicate_field("position_ids"));
                    }
                    position_ids = Some(map.next_value()?);
                }
                Field::TokenTypeIds => {
                    if token_type_ids.is_some() {
                        return Err(de::Error::duplicate_field("token_type_ids"));
                    }
                    token_type_ids = Some(map.next_value()?);
                }
                Field::CumulativeSeqLengths => {
                    if cumulative_seq_lengths.is_some() {
                        return Err(de::Error::duplicate_field("cumulative_seq_lengths"));
                    }
                    cumulative_seq_lengths = Some(map.next_value()?);
                }
                Field::PooledIndices => {
                    if pooled_indices.is_some() {
                        return Err(de::Error::duplicate_field("pooled_indices"));
                    }
                    pooled_indices = Some(map.next_value()?);
                }
                Field::RawIndices => {
                    if raw_indices.is_some() {
                        return Err(de::Error::duplicate_field("raw_indices"));
                    }
                    raw_indices = Some(map.next_value()?);
                }
                Field::MaxLength => {
                    if max_length.is_some() {
                        return Err(de::Error::duplicate_field("max_length"));
                    }
                    max_length = Some(map.next_value()?);
                }
                Field::Unknown => {
                    return Err(de::Error::unknown_field("unknown", &BATCH_FIELDS));
                }
            }
        }

        Ok(BatchJson {
            input_ids: input_ids.ok_or_else(|| de::Error::missing_field("input_ids"))?,
            position_ids: position_ids.ok_or_else(|| de::Error::missing_field("position_ids"))?,
            token_type_ids: token_type_ids
                .ok_or_else(|| de::Error::missing_field("token_type_ids"))?,
            cumulative_seq_lengths: cumulative_seq_lengths
                .ok_or_else(|| de::Error::missing_field("cumulative_seq_lengths"))?,
            pooled_indices: pooled_indices
                .ok_or_else(|| de::Error::missing_field("pooled_indices"))?,
            raw_indices: raw_indices.ok_or_else(|| de::Error::missing_field("raw_indices"))?,
            max_length: max_length.ok_or_else(|| de::Error::missing_field("max_length"))?,
        })
    }
}

impl<'de> Deserialize<'de> for BatchJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("BatchJson", &BATCH_FIELDS, BatchVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchFacts {
    pub sequence_count: usize,
    pub token_count: usize,
    pub sequence_lengths: Vec<u32>,
    pub max_length: u32,
    pub pooled_count: usize,
    pub raw_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Segment {
    pub kind: String,
    pub index: usize,
    pub shape: Vec<usize>,
    pub offset_floats: usize,
    pub length_floats: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub hidden_width: usize,
    pub pooled_width: usize,
    pub pooled_count: usize,
    pub raw_count: usize,
    pub raw_token_rows: usize,
    pub pooled_float_count: usize,
    pub raw_float_count: usize,
    pub physical_float_count: usize,
    pub physical_bytes: usize,
    pub width_note: String,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpectedKind {
    Pooled,
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpectedEntry {
    pub index: usize,
    pub kind: ExpectedKind,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalFacts {
    pub entry_count: usize,
    pub float_count: usize,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
}

fn checked_product(left: usize, right: usize) -> Result<usize, BatchError> {
    left.checked_mul(right)
        .ok_or_else(|| BatchError::new("batch layout size overflow"))
}

fn checked_add(left: usize, right: usize) -> Result<usize, BatchError> {
    left.checked_add(right)
        .ok_or_else(|| BatchError::new("batch layout offset overflow"))
}

fn positive_width(value: usize, name: &str) -> Result<usize, BatchError> {
    if value == 0 {
        return Err(BatchError::new(format!("{name} must be positive")));
    }
    Ok(value)
}

pub fn sequence_lengths(batch: &BatchJson) -> Result<Vec<u32>, BatchError> {
    let mut lengths = Vec::with_capacity(batch.cumulative_seq_lengths.len().saturating_sub(1));
    for pair in batch.cumulative_seq_lengths.windows(2) {
        lengths.push(pair[1] - pair[0]);
    }
    Ok(lengths)
}

pub fn validate(batch: &BatchJson) -> Result<(), BatchError> {
    if batch.input_ids.len() != batch.position_ids.len()
        || batch.input_ids.len() != batch.token_type_ids.len()
    {
        return Err(BatchError::new(
            "input_ids, position_ids and token_type_ids lengths differ",
        ));
    }
    let cumulative = &batch.cumulative_seq_lengths;
    if cumulative.len() < 2 {
        return Err(BatchError::new(
            "cumulative_seq_lengths must describe at least one sequence",
        ));
    }
    if cumulative[0] != 0 {
        return Err(BatchError::new("cumulative_seq_lengths must start at 0"));
    }
    if cumulative[cumulative.len() - 1] as usize != batch.input_ids.len() {
        return Err(BatchError::new(
            "cumulative_seq_lengths must end at the token count",
        ));
    }
    if cumulative.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(BatchError::new(
            "cumulative_seq_lengths must be strictly increasing",
        ));
    }
    let lengths = sequence_lengths(batch)?;
    let maximum = *lengths
        .iter()
        .max()
        .ok_or_else(|| BatchError::new("batch has no sequences"))?;
    if batch.max_length != maximum {
        return Err(BatchError::new(
            "max_length must equal the actual maximum sequence length",
        ));
    }
    let count = lengths.len();
    for index in &batch.pooled_indices {
        if *index as usize >= count {
            return Err(BatchError::new(format!(
                "pooled_indices selection {index} is outside batch of {count}"
            )));
        }
    }
    for index in &batch.raw_indices {
        if *index as usize >= count {
            return Err(BatchError::new(format!(
                "raw_indices selection {index} is outside batch of {count}"
            )));
        }
    }
    Ok(())
}

pub fn batch_facts(batch: &BatchJson) -> Result<BatchFacts, BatchError> {
    validate(batch)?;
    Ok(BatchFacts {
        sequence_count: batch.cumulative_seq_lengths.len() - 1,
        token_count: batch.input_ids.len(),
        sequence_lengths: sequence_lengths(batch)?,
        max_length: batch.max_length,
        pooled_count: batch.pooled_indices.len(),
        raw_count: batch.raw_indices.len(),
    })
}

pub fn physical_layout(
    batch: &BatchJson,
    hidden_width: usize,
    pooled_width: usize,
) -> Result<Layout, BatchError> {
    validate(batch)?;
    let hidden_width = positive_width(hidden_width, "hidden width")?;
    let pooled_width = positive_width(pooled_width, "pooled width")?;
    let mut segments = Vec::new();
    let mut offset = 0usize;
    for selection in &batch.pooled_indices {
        let length = pooled_width;
        segments.push(Segment {
            kind: "pooled".into(),
            index: *selection as usize,
            shape: vec![pooled_width],
            offset_floats: offset,
            length_floats: length,
        });
        offset = checked_add(offset, length)?;
    }
    let mut raw_rows = 0usize;
    for selection in &batch.raw_indices {
        let index = *selection as usize;
        let start = batch.cumulative_seq_lengths[index] as usize;
        let end = batch.cumulative_seq_lengths[index + 1] as usize;
        let rows = end - start;
        let length = checked_product(rows, hidden_width)?;
        segments.push(Segment {
            kind: "raw".into(),
            index,
            shape: vec![rows, hidden_width],
            offset_floats: offset,
            length_floats: length,
        });
        offset = checked_add(offset, length)?;
        raw_rows = checked_add(raw_rows, rows)?;
    }
    let pooled_float_count = checked_product(batch.pooled_indices.len(), pooled_width)?;
    let raw_float_count = checked_product(raw_rows, hidden_width)?;
    Ok(Layout {
        hidden_width,
        pooled_width,
        pooled_count: batch.pooled_indices.len(),
        raw_count: batch.raw_indices.len(),
        raw_token_rows: raw_rows,
        pooled_float_count,
        raw_float_count,
        physical_float_count: offset,
        physical_bytes: checked_product(offset, 4)?,
        width_note: "pooled segments are D floats; raw token segments are H floats; the pooled width is the final Dense output width".into(),
        segments,
    })
}

pub fn canonical_expected(
    layout: &Layout,
) -> Result<(BTreeMap<usize, ExpectedEntry>, CanonicalFacts), BatchError> {
    let mut expected = BTreeMap::new();
    let mut seen_pooled = BTreeSet::new();
    let mut seen_raw = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    let mut overridden = BTreeSet::new();
    for segment in &layout.segments {
        let kind = if segment.kind == "pooled" {
            if !seen_pooled.insert(segment.index) {
                duplicates.insert(segment.index);
            }
            ExpectedKind::Pooled
        } else {
            if seen_pooled.contains(&segment.index) {
                overridden.insert(segment.index);
            }
            if !seen_raw.insert(segment.index) {
                duplicates.insert(segment.index);
            }
            ExpectedKind::Raw
        };
        expected.insert(
            segment.index,
            ExpectedEntry {
                index: segment.index,
                kind,
                shape: segment.shape.clone(),
            },
        );
    }
    let float_count = expected.values().try_fold(0usize, |total, entry| {
        let size = entry.shape.iter().try_fold(1usize, |product, dimension| {
            checked_product(product, *dimension)
        })?;
        checked_add(total, size)
    })?;
    let entry_count = expected.len();
    Ok((
        expected,
        CanonicalFacts {
            entry_count,
            float_count,
            raw_overrides_pooled: overridden.into_iter().collect(),
            duplicate_selection_indices: duplicates.into_iter().collect(),
        },
    ))
}

pub fn parse_json(data: &[u8]) -> Result<BatchJson, BatchError> {
    let batch: BatchJson = serde_json::from_slice(data)
        .map_err(|error| BatchError::new(format!("invalid batch JSON: {error}")))?;
    validate(&batch)?;
    Ok(batch)
}

pub fn load_batch(path: &Path) -> Result<(String, BatchJson), BatchError> {
    let metadata = fs::metadata(path)
        .map_err(|error| BatchError::new(format!("missing batch JSON {path:?}: {error}")))?;
    if metadata.len() > MAX_BATCH_JSON {
        return Err(BatchError::new(format!(
            "batch JSON exceeds 32 MiB: {path:?}"
        )));
    }
    let data = fs::read(path)
        .map_err(|error| BatchError::new(format!("cannot read batch JSON {path:?}: {error}")))?;
    if data.len() as u64 > MAX_BATCH_JSON {
        return Err(BatchError::new(format!(
            "batch JSON exceeds 32 MiB: {path:?}"
        )));
    }
    let digest = format!("{:x}", Sha256::digest(&data));
    Ok((digest, parse_json(&data)?))
}

pub fn to_backend_batch(batch: &BatchJson) -> infinity_v3_backend_core::Batch {
    infinity_v3_backend_core::Batch {
        multimodal: Vec::new(),
        input_ids: batch.input_ids.clone(),
        token_type_ids: batch.token_type_ids.clone(),
        position_ids: batch.position_ids.clone(),
        cumulative_seq_lengths: batch.cumulative_seq_lengths.clone(),
        max_length: batch.max_length,
        pooled_indices: batch.pooled_indices.clone(),
        raw_indices: batch.raw_indices.clone(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}
