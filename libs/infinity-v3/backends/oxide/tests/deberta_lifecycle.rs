use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Batch, Inference, ModelType, Predictions};
use std::{env, error::Error, fmt, path::PathBuf, thread};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type FixtureResult<T> = Result<T, FixtureError>;

#[derive(Debug, Clone)]
struct FixtureError(String);

impl FixtureError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for FixtureError {}

#[derive(Clone, Debug)]
struct BatchFixture {
    lengths: Vec<usize>,
    input_ids: Vec<u32>,
    position_ids: Vec<u32>,
    token_type_ids: Vec<u32>,
    cumulative_seq_lengths: Vec<u32>,
    max_length: u32,
    pooled_indices: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(usize, usize)>,
    ignored_position_type: bool,
}

impl BatchFixture {
    fn ragged(lengths: &[usize], seed: u32) -> FixtureResult<Self> {
        if lengths.is_empty() {
            return Err(FixtureError::new("fixture requires at least one sequence"));
        }

        let mut input_ids = Vec::new();
        let mut position_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut cumulative_seq_lengths = vec![0u32];
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut token_count = 0usize;
        let mut byte_cursor = 0usize;
        let mut max_length = 0usize;

        for (sequence, &length) in lengths.iter().enumerate() {
            if length == 0 {
                return Err(FixtureError::new("fixture sequences must be nonzero"));
            }
            if length > 512 {
                return Err(FixtureError::new("fixture sequence exceeds 512 positions"));
            }
            max_length = max_length.max(length);

            for local in 0..length {
                let mixed = u64::from(seed)
                    .wrapping_mul(131)
                    .wrapping_add((sequence as u64).wrapping_mul(4099))
                    .wrapping_add((local as u64).wrapping_mul(37));
                let token_id = 20u32 + (mixed % 20_000) as u32;
                input_ids.push(token_id);
                position_ids.push(local as u32);
                token_type_ids.push((sequence % 2) as u32);

                let text = format!("t{token_id}");
                let end = byte_cursor + text.len();
                tokens.push(text);
                offsets.push((byte_cursor, end));
                byte_cursor = end;
            }

            token_count = token_count
                .checked_add(length)
                .ok_or_else(|| FixtureError::new("fixture token count overflow"))?;
            cumulative_seq_lengths.push(
                u32::try_from(token_count)
                    .map_err(|_| FixtureError::new("fixture token count exceeds u32"))?,
            );
        }

        let mut pooled_indices = Vec::with_capacity(lengths.len());
        for index in 0..lengths.len() {
            pooled_indices.push(
                u32::try_from(index)
                    .map_err(|_| FixtureError::new("fixture sequence index exceeds u32"))?,
            );
        }

        let fixture = Self {
            lengths: lengths.to_vec(),
            input_ids,
            position_ids,
            token_type_ids,
            cumulative_seq_lengths,
            max_length: u32::try_from(max_length)
                .map_err(|_| FixtureError::new("fixture length exceeds u32"))?,
            pooled_indices,
            tokens,
            offsets,
            ignored_position_type: false,
        };
        fixture.validate()?;
        Ok(fixture)
    }

    fn validate(&self) -> FixtureResult<()> {
        let sequences = self.lengths.len();
        if sequences == 0 {
            return Err(FixtureError::new("fixture requires at least one sequence"));
        }

        let total = self.lengths.iter().try_fold(0usize, |total, &length| {
            total
                .checked_add(length)
                .ok_or_else(|| FixtureError::new("fixture token count overflow"))
        })?;

        if self.cumulative_seq_lengths.len() != sequences + 1 {
            return Err(FixtureError::new(
                "fixture cumulative offsets must contain one boundary per sequence",
            ));
        }
        if self.cumulative_seq_lengths.first() != Some(&0)
            || self.cumulative_seq_lengths.last().copied() != u32::try_from(total).ok()
        {
            return Err(FixtureError::new(
                "fixture cumulative offsets must span the complete token count",
            ));
        }
        for pair in self.cumulative_seq_lengths.windows(2) {
            let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
                return Err(FixtureError::new(
                    "fixture cumulative offsets are malformed",
                ));
            };
            if end <= start {
                return Err(FixtureError::new(
                    "fixture cumulative offsets must advance for every sequence",
                ));
            }
        }

        if self.max_length != self.lengths.iter().copied().max().unwrap_or(0) as u32 {
            return Err(FixtureError::new(
                "fixture max_length is not the longest sequence",
            ));
        }
        if self.input_ids.len() != total
            || self.position_ids.len() != total
            || self.token_type_ids.len() != total
            || self.tokens.len() != total
            || self.offsets.len() != total
        {
            return Err(FixtureError::new(
                "fixture input, position, type, token and offset arrays must match the token count",
            ));
        }
        if self
            .input_ids
            .iter()
            .any(|&token_id| !(20..=20_019).contains(&token_id))
        {
            return Err(FixtureError::new(
                "fixture token ids must stay inside 20..=20019",
            ));
        }
        if self.pooled_indices.len() != sequences
            || self
                .pooled_indices
                .iter()
                .any(|&index| index as usize >= sequences)
        {
            return Err(FixtureError::new(
                "fixture selectors must name an existing sequence",
            ));
        }

        let mut cursor = 0usize;
        for (sequence, &length) in self.lengths.iter().enumerate() {
            for local in 0..length {
                let index = cursor + local;
                if self.ignored_position_type {
                    if self.position_ids[index] != u32::MAX
                        || self.token_type_ids[index] != u32::MAX
                    {
                        return Err(FixtureError::new(
                            "ignored position/type fixture must fill every entry with u32::MAX",
                        ));
                    }
                } else {
                    if self.position_ids[index] != local as u32 {
                        return Err(FixtureError::new(format!(
                            "fixture position must reset locally at sequence {sequence}, token {local}",
                        )));
                    }
                    if self.token_type_ids[index] >= 2 {
                        return Err(FixtureError::new(
                            "fixture token type ids must stay inside the admitted type vocabulary",
                        ));
                    }
                }
            }
            cursor += length;
        }

        let mut expected_start = 0usize;
        for &(start, end) in &self.offsets {
            if start != expected_start || end < start {
                return Err(FixtureError::new(
                    "fixture token offsets must be contiguous and ordered",
                ));
            }
            expected_start = end;
        }

        Ok(())
    }

    fn with_duplicate_pooled_selectors(mut self) -> FixtureResult<Self> {
        if self.lengths.len() < 2 {
            return Err(FixtureError::new(
                "duplicate selectors require at least two sequences",
            ));
        }
        self.pooled_indices = vec![0u32; self.lengths.len()];
        self.validate()?;
        Ok(self)
    }

    fn with_ignored_position_and_type(mut self) -> FixtureResult<Self> {
        self.position_ids.fill(u32::MAX);
        self.token_type_ids.fill(u32::MAX);
        self.ignored_position_type = true;
        self.validate()?;
        Ok(self)
    }

    fn to_batch(&self) -> FixtureResult<Batch> {
        self.validate()?;
        Ok(Batch {
            multimodal: Vec::new(),
            input_ids: self.input_ids.clone(),
            token_type_ids: self.token_type_ids.clone(),
            position_ids: self.position_ids.clone(),
            cumulative_seq_lengths: self.cumulative_seq_lengths.clone(),
            max_length: self.max_length,
            pooled_indices: self.pooled_indices.clone(),
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: self.tokens.clone(),
            offsets: self.offsets.clone(),
        })
    }

    fn invalid_input_id_batch(&self) -> FixtureResult<Batch> {
        let mut batch = self.to_batch()?;
        let first = batch
            .input_ids
            .first_mut()
            .ok_or_else(|| FixtureError::new("cannot corrupt an empty token fixture"))?;
        *first = u32::MAX;
        Ok(batch)
    }

    fn malformed_offsets_batch(&self) -> FixtureResult<Batch> {
        let mut batch = self.to_batch()?;
        if batch.cumulative_seq_lengths.len() < 3 {
            return Err(FixtureError::new(
                "malformed ragged offsets require a multi-sequence fixture",
            ));
        }
        batch.cumulative_seq_lengths[1] = batch.cumulative_seq_lengths[0];
        batch.offsets.pop();
        Ok(batch)
    }
}

fn fixture_a() -> FixtureResult<BatchFixture> {
    BatchFixture::ragged(&[3, 7, 2], 41)
}

fn fixture_b() -> FixtureResult<BatchFixture> {
    BatchFixture::ragged(&[9, 5], 73)
}

fn model_path() -> TestResult<PathBuf> {
    env::var_os("INFINITY_DEBERTA_MODEL")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| {
            FixtureError::new(
                "set INFINITY_DEBERTA_MODEL to the admitted DeBERTa sequence classifier",
            )
            .into()
        })
}

fn admitted_width() -> TestResult<usize> {
    let value = env::var("INFINITY_DEBERTA_WIDTH").unwrap_or_else(|_| "1".to_owned());
    let width = value.parse::<usize>()?;
    if width == 0 {
        return Err(FixtureError::new("INFINITY_DEBERTA_WIDTH must be nonzero").into());
    }
    Ok(width)
}

fn open_backend() -> TestResult<OxideBackend> {
    let path = model_path()?;
    OxideBackend::new(&path, "bfloat16", ModelType::Classifier, 0)
        .map_err(|error| FixtureError::new(error.to_string()).into())
}

fn row_bits_from_get(
    predictions: &Predictions,
    sequences: usize,
    width: usize,
) -> FixtureResult<Vec<Vec<u32>>> {
    if width == 0 {
        return Err(FixtureError::new("prediction width must be nonzero"));
    }
    let mut rows = Vec::with_capacity(sequences);
    for index in 0..sequences {
        let row = predictions
            .get(&index)
            .ok_or_else(|| FixtureError::new(format!("missing prediction row {index}")))?;
        if row.len() != width {
            return Err(FixtureError::new(format!(
                "prediction row {index} has width {}, expected {width}",
                row.len(),
            )));
        }
        rows.push(row.iter().map(|value| value.to_bits()).collect());
    }
    Ok(rows)
}

fn row_bits_from_into_iter(
    predictions: Predictions,
    sequences: usize,
    width: usize,
) -> FixtureResult<Vec<Vec<u32>>> {
    if width == 0 {
        return Err(FixtureError::new("prediction width must be nonzero"));
    }
    let mut entries = predictions.into_iter().collect::<Vec<_>>();
    if entries.len() != sequences {
        return Err(FixtureError::new(format!(
            "prediction map has {} rows, expected {sequences}",
            entries.len(),
        )));
    }
    entries.sort_by_key(|(index, _)| *index);

    let mut rows = Vec::with_capacity(sequences);
    for (expected, (index, row)) in entries.into_iter().enumerate() {
        if index != expected {
            return Err(FixtureError::new(format!(
                "prediction map row {index} is not canonical index {expected}",
            )));
        }
        if row.len() != width {
            return Err(FixtureError::new(format!(
                "prediction row {index} has width {}, expected {width}",
                row.len(),
            )));
        }
        rows.push(row.into_iter().map(|value| value.to_bits()).collect());
    }
    Ok(rows)
}

#[test]
fn ragged_fixture_contract_catches_offsets_local_reset_and_ids() -> TestResult {
    let a = BatchFixture::ragged(&[3, 7, 2], 41)?;
    let b = BatchFixture::ragged(&[9, 5], 73)?;
    let batch_a = a.to_batch()?;
    let batch_b = b.to_batch()?;

    assert_eq!(batch_a.cumulative_seq_lengths, vec![0, 3, 10, 12]);
    assert_eq!(batch_b.cumulative_seq_lengths, vec![0, 9, 14]);
    assert_eq!(batch_a.max_length, 7);
    assert_eq!(batch_b.max_length, 9);
    assert_eq!(
        batch_a.position_ids,
        vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 0, 1]
    );
    assert_eq!(
        batch_b.position_ids,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4]
    );
    assert_eq!(
        batch_a.token_type_ids,
        vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    );
    assert!(batch_a
        .input_ids
        .iter()
        .all(|&token_id| (20..=20_019).contains(&token_id)));
    assert_eq!(batch_a.input_ids.len(), 12);
    assert_eq!(batch_a.tokens.len(), batch_a.input_ids.len());
    assert_eq!(batch_a.offsets.len(), batch_a.input_ids.len());

    let duplicated = a.clone().with_duplicate_pooled_selectors()?;
    assert_eq!(
        duplicated.to_batch()?.pooled_indices,
        vec![0, 0, 0],
        "duplicate selectors remain representable for canonical override",
    );

    let ignored = a.clone().with_ignored_position_and_type()?;
    let ignored_batch = ignored.to_batch()?;
    assert_eq!(ignored_batch.position_ids.len(), 12);
    assert_eq!(ignored_batch.token_type_ids.len(), 12);
    assert!(ignored_batch
        .position_ids
        .iter()
        .all(|&position| position == u32::MAX));
    assert!(ignored_batch
        .token_type_ids
        .iter()
        .all(|&token_type| token_type == u32::MAX));

    let mut malformed_offsets = a.clone();
    malformed_offsets.cumulative_seq_lengths = vec![0, 3, 3, 12];
    assert!(
        malformed_offsets.to_batch().is_err(),
        "zero-length ragged offsets must be rejected",
    );

    let mut global_positions = a.clone();
    global_positions.position_ids[3] = 3;
    assert!(
        global_positions.to_batch().is_err(),
        "positions must reset locally at every sequence boundary",
    );

    let mut outside_vocabulary = a.clone();
    outside_vocabulary.input_ids[0] = 20_020;
    assert!(
        outside_vocabulary.to_batch().is_err(),
        "synthetic token ids outside 20..=20019 must be rejected",
    );

    let mut wrong_array_length = a.clone();
    wrong_array_length.token_type_ids.pop();
    assert!(
        wrong_array_length.to_batch().is_err(),
        "physical batches require complete token_type_ids",
    );

    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_actual_backend_ragged_a_b_a_is_bit_deterministic() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    backend.health()?;

    let a = fixture_a()?.to_batch()?;
    let b = fixture_b()?.to_batch()?;
    let first_a = row_bits_from_get(&backend.predict(a.clone())?, 3, width)?;
    let first_b = row_bits_from_get(&backend.predict(b)?, 2, width)?;
    let second_a = row_bits_from_into_iter(backend.predict(a)?, 3, width)?;

    assert_eq!(first_a.len(), 3);
    assert_eq!(first_b.len(), 2);
    assert_eq!(
        first_a, second_a,
        "ragged A/B/A must reproduce exact F32 result bits",
    );
    backend.health()?;
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_deferred_ragged_requests_complete_out_of_order_across_threads() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let a = fixture_a()?.to_batch()?;
    let b = fixture_b()?.to_batch()?;
    let expected_a = row_bits_from_get(&backend.predict(a.clone())?, 3, width)?;
    let expected_b = row_bits_from_get(&backend.predict(b.clone())?, 2, width)?;

    let deferred_a = backend.submit_predict(a)?;
    let deferred_b = backend.submit_predict(b)?;
    assert!(
        matches!(&deferred_a, Inference::Deferred(_)),
        "ragged A must remain held for deferred completion",
    );
    assert!(
        matches!(&deferred_b, Inference::Deferred(_)),
        "ragged B must remain held for deferred completion",
    );

    let completed_b = thread::spawn(move || deferred_b.complete())
        .join()
        .map_err(|_| FixtureError::new("deferred B completion thread panicked"))??;
    let actual_b = row_bits_from_get(&completed_b, 2, width)?;
    let actual_a = row_bits_from_into_iter(deferred_a.complete()?, 3, width)?;

    assert_eq!(actual_b, expected_b, "deferred B must preserve exact bits");
    assert_eq!(
        actual_a, expected_a,
        "deferred A must preserve exact bits after B completes first",
    );
    backend.health()?;
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_dropped_deferred_request_does_not_disturb_other_completion() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let a = fixture_a()?.to_batch()?;
    let b = fixture_b()?.to_batch()?;
    let expected_a = row_bits_from_get(&backend.predict(a.clone())?, 3, width)?;

    let deferred_a = backend.submit_predict(a)?;
    let deferred_b = backend.submit_predict(b)?;
    assert!(matches!(&deferred_a, Inference::Deferred(_)));
    assert!(matches!(&deferred_b, Inference::Deferred(_)));

    let completed_a = thread::spawn(move || deferred_a.complete());
    drop(deferred_b);
    let completed_a = completed_a
        .join()
        .map_err(|_| FixtureError::new("deferred A completion thread panicked"))??;
    let actual_a = row_bits_from_into_iter(completed_a, 3, width)?;
    assert_eq!(
        actual_a, expected_a,
        "dropping one deferred request must not change the other result",
    );
    backend.health()?;
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_invalid_input_and_malformed_offsets_leave_held_result_unchanged() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let fixture = fixture_a()?;
    let expected = row_bits_from_get(&backend.predict(fixture.to_batch()?)?, 3, width)?;

    let held = backend.submit_predict(fixture.to_batch()?)?;
    assert!(
        matches!(&held, Inference::Deferred(_)),
        "held A must be deferred while invalid requests are rejected",
    );

    assert!(
        backend.predict(fixture.invalid_input_id_batch()?).is_err(),
        "invalid token ids must reject before touching the held result",
    );
    assert!(
        backend.predict(fixture.malformed_offsets_batch()?).is_err(),
        "malformed ragged offsets must reject before touching the held result",
    );
    backend.health()?;

    let actual = row_bits_from_into_iter(held.complete()?, 3, width)?;
    assert_eq!(
        actual, expected,
        "held A must retain exact result bits after rejected requests",
    );
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_ignored_position_and_type_values_preserve_result_bits() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let fixture = fixture_a()?;
    let expected = row_bits_from_get(&backend.predict(fixture.to_batch()?)?, 3, width)?;

    let ignored = fixture.with_ignored_position_and_type()?;
    let ignored_batch = ignored.to_batch()?;
    assert_eq!(
        ignored_batch.position_ids.len(),
        ignored_batch.input_ids.len(),
    );
    assert_eq!(
        ignored_batch.token_type_ids.len(),
        ignored_batch.input_ids.len(),
    );
    let actual = row_bits_from_into_iter(backend.predict(ignored_batch)?, 3, width)?;

    assert_eq!(
        actual, expected,
        "u32::MAX position and type entries must be ignored for sequence prediction",
    );
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_changed_shapes_are_bit_deterministic() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let shapes: &[&[usize]] = &[
        &[1],
        &[7],
        &[127],
        &[128],
        &[129],
        &[256],
        &[512],
        &[3, 7, 2],
    ];

    for &shape in shapes {
        let fixture = BatchFixture::ragged(shape, 29)?;
        let first = row_bits_from_get(&backend.predict(fixture.to_batch()?)?, shape.len(), width)?;
        let second =
            row_bits_from_into_iter(backend.predict(fixture.to_batch()?)?, shape.len(), width)?;
        assert_eq!(
            first, second,
            "shape {shape:?} must reproduce exact F32 result bits",
        );
    }
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_duplicate_selectors_are_canonically_overridden_for_sequence_classification() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let canonical = fixture_a()?;
    let expected = row_bits_from_get(&backend.predict(canonical.to_batch()?)?, 3, width)?;

    let duplicated = canonical.with_duplicate_pooled_selectors()?;
    let duplicate_batch = duplicated.to_batch()?;
    assert_eq!(duplicate_batch.pooled_indices, vec![0, 0, 0]);
    let actual = row_bits_from_into_iter(backend.predict(duplicate_batch)?, 3, width)?;

    assert_eq!(
        actual, expected,
        "duplicate selectors must be harmlessly replaced by canonical row indices",
    );
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_native_token_prediction_rejects_before_leases_and_sequence_recovers() -> TestResult {
    let width = admitted_width()?;
    let backend = open_backend()?;
    let fixture = fixture_a()?;
    let expected = row_bits_from_get(&backend.predict(fixture.to_batch()?)?, 3, width)?;

    for attempt in 0..64 {
        assert!(
            backend.predict_tokens(fixture.to_batch()?).is_err(),
            "predict_tokens must reject attempt {attempt} before acquiring host leases",
        );
        assert!(
            backend.submit_predict_tokens(fixture.to_batch()?).is_err(),
            "submit_predict_tokens must reject attempt {attempt} before acquiring host leases",
        );
    }

    let held = backend.submit_predict(fixture.to_batch()?)?;
    assert!(matches!(&held, Inference::Deferred(_)));
    assert!(backend.predict_tokens(fixture.to_batch()?).is_err());
    assert!(backend.submit_predict_tokens(fixture.to_batch()?).is_err());

    let actual = row_bits_from_into_iter(held.complete()?, 3, width)?;
    assert_eq!(
        actual, expected,
        "sequence prediction must remain healthy after token prediction rejection",
    );
    backend.health()?;
    Ok(())
}

#[test]
#[ignore = "Root-only GPU device 0: requires INFINITY_DEBERTA_MODEL"]
fn gpu_held_deferred_result_completes_after_backend_is_dropped() -> TestResult {
    let width = admitted_width()?;
    let fixture = fixture_a()?;
    let expected = {
        let backend = open_backend()?;
        row_bits_from_get(&backend.predict(fixture.to_batch()?)?, 3, width)?
    };

    let deferred = {
        let backend = open_backend()?;
        let deferred = backend.submit_predict(fixture.to_batch()?)?;
        assert!(
            matches!(&deferred, Inference::Deferred(_)),
            "result must remain deferred after backend drop",
        );
        drop(backend);
        deferred
    };
    let actual = row_bits_from_into_iter(deferred.complete()?, 3, width)?;

    assert_eq!(
        actual, expected,
        "deferred result must complete with exact bits after backend drop",
    );
    Ok(())
}
