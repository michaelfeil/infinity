// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Pure packed-request planning for the Clef joint head.
//!
//! The layout below follows Native `ClefHead::score` exactly: request order,
//! field order, then each field's option order. Question and option intervals
//! are request-relative on input and become packed token rows only here.
//! `Warmup` synthesizes the Native field `{kind:0, question:(0,1),
//! options:[(0,1)]}` without special-casing the scoring math.

use crate::{
    clef_decision::{
        infer, request_output_count, CLEF_FIELDS, MAX_CLEF_FIELDS, MAX_CLEF_OPTIONS,
        MAX_CLEF_OPTION_TOKENS,
    },
    decision_head::PackedDecisionBatch,
};
use infinity_v3_backend_core::{BackendError, DecisionInput};

/// Offsets inside the single packed u32 metadata tail.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ClefMetadataLayout {
    pub(crate) question_boundaries: usize,
    pub(crate) question_selected: usize,
    pub(crate) option_boundaries: usize,
    pub(crate) option_selected: usize,
    pub(crate) lexical_boundaries: usize,
    pub(crate) lexical_selected: usize,
    pub(crate) last_rows: usize,
    pub(crate) request_field_offsets: usize,
    pub(crate) request_option_offsets: usize,
    pub(crate) sequence_offsets: usize,
    pub(crate) field_request: usize,
    pub(crate) field_kind: usize,
    pub(crate) field_option_offsets: usize,
    pub(crate) field_of_option: usize,
    pub(crate) option_token_ids: usize,
    pub(crate) total: usize,
}

/// Validated geometry, host counts and the single metadata tail to upload.
#[derive(Debug)]
pub(crate) struct ClefRequestPlan {
    pub(crate) tokens: usize,
    pub(crate) requests: usize,
    pub(crate) fields: usize,
    pub(crate) options: usize,
    pub(crate) option_tokens: usize,
    pub(crate) max_sequence: usize,
    pub(crate) max_options: usize,
    pub(crate) max_fields: usize,
    pub(crate) counts: Vec<usize>,
    pub(crate) layout: ClefMetadataLayout,
    pub(crate) metadata: Vec<u32>,
}

impl ClefRequestPlan {
    pub(crate) fn output_elements(&self) -> Result<usize, BackendError> {
        self.options
            .checked_add(self.requests)
            .ok_or_else(|| infer("Clef output count overflow"))
    }

    pub(crate) fn metadata_len(&self) -> usize {
        self.metadata.len()
    }
}

struct Builder {
    values: Vec<u32>,
    layout: ClefMetadataLayout,
}

impl Builder {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            layout: ClefMetadataLayout::default(),
        }
    }

    fn section(&mut self, values: impl IntoIterator<Item = u32>) -> Result<usize, BackendError> {
        let offset = self.values.len();
        let mut values = values.into_iter();
        self.values
            .try_reserve(values.size_hint().0)
            .map_err(|error| infer(format!("Clef metadata allocation failed: {error}")))?;
        self.values.extend(&mut values);
        Ok(offset)
    }
}

fn interval(
    field: usize,
    option: usize,
    interval: (usize, usize),
    start: usize,
    length: usize,
) -> Result<(usize, usize), BackendError> {
    let (a, b) = interval;
    if a >= b || b > length {
        return Err(infer(format!(
            "Clef field {field} option {option} span ({a},{b}) is outside sequence length {length}"
        )));
    }
    Ok((start + a, start + b))
}

/// Build one packed plan without touching a device or reading model weights.
pub(crate) fn plan_requests(
    batch: &PackedDecisionBatch<'_>,
    ids: &[u32],
    requests: &[DecisionInput],
) -> Result<ClefRequestPlan, BackendError> {
    if requests.len() != batch.len() {
        return Err(infer(
            "Clef request count does not match the packed decision batch",
        ));
    }
    if ids.len() != batch.tokens() {
        return Err(infer(
            "Clef packed token ids do not match the decision batch",
        ));
    }
    let mut builder = Builder::new();
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(requests.len())
        .map_err(|error| infer(format!("Clef count allocation failed: {error}")))?;
    let mut question_boundaries = Vec::new();
    let mut option_boundaries = Vec::new();
    let mut lexical_boundaries = vec![0u32];
    let mut last_rows = Vec::new();
    let mut request_field_offsets = vec![0u32];
    let mut request_option_offsets = vec![0u32];
    let mut sequence_offsets = Vec::new();
    let mut field_request = Vec::new();
    let mut field_kind = Vec::new();
    let mut field_option_offsets = vec![0u32];
    let mut field_of_option = Vec::new();
    let mut option_token_ids = Vec::new();
    let mut fields = 0usize;
    let mut options = 0usize;
    let mut option_tokens = 0usize;
    let mut max_sequence = 0usize;
    let mut max_options = 0usize;
    let mut max_fields = 0usize;
    for (request_index, request) in requests.iter().enumerate() {
        let (start, end) = batch
            .sequence(request_index)
            .map_err(|error| infer(error.to_string()))?;
        let length = end
            .checked_sub(start)
            .ok_or_else(|| infer("Clef sequence length underflow"))?;
        if length == 0 {
            return Err(infer("Clef requires nonempty packed sequences"));
        }
        sequence_offsets.push(start as u32);
        last_rows.push((end - 1) as u32);
        max_sequence = max_sequence.max(length);
        let request_fields = match request {
            DecisionInput::Warmup => vec![(0, (0, 1), vec![(0, 1)])],
            DecisionInput::Clef { fields } if !fields.is_empty() => fields
                .iter()
                .map(|field| {
                    if field.kind >= CLEF_FIELDS {
                        return Err(infer("Clef field kind must be 0..3"));
                    }
                    if field.options.is_empty() {
                        return Err(infer("Clef field options must not be empty"));
                    }
                    Ok((field.kind, field.question, field.options.clone()))
                })
                .collect::<Result<_, BackendError>>()?,
            DecisionInput::Clef { .. } => return Err(infer("Clef requires at least one field")),
            _ => return Err(infer("Only Clef and Warmup requests reach the Clef head")),
        };
        if request_fields.len() > MAX_CLEF_FIELDS {
            return Err(infer(format!(
                "Clef field count {} exceeds {MAX_CLEF_FIELDS}",
                request_fields.len()
            )));
        }
        let request_field_count = request_fields.len();
        let mut request_options = 0usize;
        for (kind, question, options_spans) in request_fields.into_iter() {
            let (qa, qb) = interval(fields, 0, question, start, length)?;
            question_boundaries.extend([qa as u32, qb as u32]);
            field_request.push(request_index as u32);
            field_kind.push(kind as u32);
            for (option_index, span) in options_spans.into_iter().enumerate() {
                let (oa, ob) = interval(fields, option_index, span, start, length)?;
                option_boundaries.extend([oa as u32, ob as u32]);
                field_of_option.push(fields as u32);
                option_token_ids.extend_from_slice(
                    ids.get(oa..ob)
                        .ok_or_else(|| infer("Clef option token range exceeds packed ids"))?,
                );
                option_tokens += ob - oa;
                lexical_boundaries.push(
                    u32::try_from(option_token_ids.len())
                        .map_err(|_| infer("Clef lexical token count exceeds u32"))?,
                );
                options = options
                    .checked_add(1)
                    .ok_or_else(|| infer("Clef option count overflow"))?;
                request_options += 1;
            }
            if request_options > MAX_CLEF_OPTIONS {
                return Err(infer(format!(
                    "Clef option count {request_options} exceeds {MAX_CLEF_OPTIONS}"
                )));
            }
            fields += 1;
            field_option_offsets.push(
                u32::try_from(options)
                    .map_err(|_| infer("Clef field option offset exceeds u32"))?,
            );
        }
        if option_token_ids.len() > MAX_CLEF_OPTION_TOKENS {
            return Err(infer(format!(
                "Clef option token count {} exceeds {MAX_CLEF_OPTION_TOKENS}",
                option_token_ids.len()
            )));
        }
        max_options = max_options.max(request_options);
        max_fields = max_fields.max(request_field_count);
        counts.push(request_output_count(request)?);
        request_field_offsets.push(fields as u32);
        request_option_offsets.push(options as u32);
    }
    sequence_offsets.push(batch.tokens() as u32);
    let identity = |n: usize| -> Result<Vec<u32>, BackendError> {
        (0..n)
            .map(|value| u32::try_from(value).map_err(|_| infer("Clef metadata index exceeds u32")))
            .collect()
    };
    let question_selected = identity(fields)?;
    let option_selected = identity(options)?;
    let lexical_selected = identity(options)?;
    builder.layout.sequence_offsets = builder.section(sequence_offsets)?;
    builder.layout.last_rows = builder.section(last_rows)?;
    builder.layout.request_field_offsets = builder.section(request_field_offsets)?;
    builder.layout.request_option_offsets = builder.section(request_option_offsets)?;
    builder.layout.field_option_offsets = builder.section(field_option_offsets)?;
    builder.layout.field_request = builder.section(field_request)?;
    builder.layout.field_kind = builder.section(field_kind)?;
    builder.layout.field_of_option = builder.section(field_of_option)?;
    builder.layout.question_boundaries = builder.section(question_boundaries)?;
    builder.layout.question_selected = builder.section(question_selected)?;
    builder.layout.option_boundaries = builder.section(option_boundaries)?;
    builder.layout.option_selected = builder.section(option_selected)?;
    builder.layout.lexical_boundaries = builder.section(lexical_boundaries)?;
    builder.layout.lexical_selected = builder.section(lexical_selected)?;
    builder.layout.option_token_ids = builder.section(option_token_ids)?;
    builder.layout.total = builder.values.len();
    Ok(ClefRequestPlan {
        tokens: batch.tokens(),
        requests: requests.len(),
        fields,
        options,
        option_tokens,
        max_sequence,
        max_options,
        max_fields,
        counts,
        layout: builder.layout,
        metadata: builder.values,
    })
}

#[cfg(test)]
#[path = "clef_decision_plan_tests.rs"]
mod tests;
