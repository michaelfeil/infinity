use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use infinity_v3_backend_core::{Embedding, Embeddings};

use crate::batch::{BatchError, CanonicalFacts, ExpectedEntry, ExpectedKind};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputEntry {
    pub index: usize,
    pub kind: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputFacts {
    pub entry_count: usize,
    pub float_count: usize,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalOutputs {
    pub outputs: Vec<OutputEntry>,
    pub facts: OutputFacts,
}

fn output_error(message: impl Into<String>) -> BatchError {
    BatchError(message.into())
}

fn check_finite(values: &[f32], label: &str) -> Result<(), BatchError> {
    if let Some(position) = values.iter().position(|value| !value.is_finite()) {
        return Err(output_error(format!(
            "{label} position {position} is non-finite"
        )));
    }
    Ok(())
}

pub fn canonicalize(
    result: &Embeddings,
    expected: &BTreeMap<usize, ExpectedEntry>,
    canonical: &CanonicalFacts,
    pooled_width: usize,
    hidden_width: usize,
) -> Result<CanonicalOutputs, BatchError> {
    let actual_keys: BTreeSet<usize> = result.keys().copied().collect();
    let expected_keys: BTreeSet<usize> = expected.keys().copied().collect();
    if actual_keys != expected_keys {
        let missing: Vec<usize> = expected_keys.difference(&actual_keys).copied().collect();
        let extra: Vec<usize> = actual_keys.difference(&expected_keys).copied().collect();
        return Err(output_error(format!(
            "canonical result key mismatch missing={missing:?} extra={extra:?}"
        )));
    }

    let mut outputs = Vec::with_capacity(expected.len());
    let mut float_count = 0usize;
    for (index, expectation) in expected {
        let embedding = result
            .get(index)
            .ok_or_else(|| output_error(format!("missing canonical index {index}")))?;
        match (expectation.kind, embedding) {
            (ExpectedKind::Pooled, Embedding::Pooled(values)) => {
                if values.len() != pooled_width {
                    return Err(output_error(format!(
                        "pooled index {index} width {} does not match D {pooled_width}",
                        values.len()
                    )));
                }
                if expectation.shape != vec![pooled_width] {
                    return Err(output_error(format!(
                        "pooled index {index} expected shape {:?} is inconsistent",
                        expectation.shape
                    )));
                }
                check_finite(values, &format!("pooled index {index}"))?;
                float_count = float_count
                    .checked_add(values.len())
                    .ok_or_else(|| output_error("canonical float count overflow"))?;
                outputs.push(OutputEntry {
                    index: *index,
                    kind: "pooled".into(),
                    shape: vec![pooled_width],
                    values: values.clone(),
                });
            }
            (ExpectedKind::Raw, Embedding::All(rows)) => {
                if expectation.shape.len() != 2 {
                    return Err(output_error(format!(
                        "raw index {index} expected shape is not [rows, H]"
                    )));
                }
                let expected_rows = expectation.shape[0];
                if rows.len() != expected_rows {
                    return Err(output_error(format!(
                        "raw index {index} has {} rows, expected {expected_rows}",
                        rows.len()
                    )));
                }
                let mut values = Vec::with_capacity(expected_rows.saturating_mul(hidden_width));
                for (row_index, row) in rows.iter().enumerate() {
                    if row.len() != hidden_width {
                        return Err(output_error(format!(
                            "raw index {index} row {row_index} width {} does not match H {hidden_width}",
                            row.len()
                        )));
                    }
                    check_finite(row, &format!("raw index {index} row {row_index}"))?;
                    values.extend_from_slice(row);
                }
                float_count = float_count
                    .checked_add(values.len())
                    .ok_or_else(|| output_error("canonical float count overflow"))?;
                outputs.push(OutputEntry {
                    index: *index,
                    kind: "raw".into(),
                    shape: vec![expected_rows, hidden_width],
                    values,
                });
            }
            (ExpectedKind::Pooled, Embedding::All(_)) => {
                return Err(output_error(format!(
                    "index {index} must be pooled but the backend returned raw rows"
                )));
            }
            (ExpectedKind::Raw, Embedding::Pooled(_)) => {
                return Err(output_error(format!(
                    "index {index} must be raw but the backend returned a pooled vector"
                )));
            }
        }
    }
    Ok(CanonicalOutputs {
        outputs,
        facts: OutputFacts {
            entry_count: expected.len(),
            float_count,
            raw_overrides_pooled: canonical.raw_overrides_pooled.clone(),
            duplicate_selection_indices: canonical.duplicate_selection_indices.clone(),
        },
    })
}
