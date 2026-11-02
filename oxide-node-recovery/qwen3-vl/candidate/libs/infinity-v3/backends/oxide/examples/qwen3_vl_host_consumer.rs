// Unified CPU host consumer for Qwen3-VL.
//
// Modes:
//   manifest        actual config, normal-factory alias, readout head, full inventory
//   factory         Batch.multimodal -> pack_batch for all 8 real fixtures + typed media errors
//   lifecycle       repeated A/B/A growth/drop and recoverable invalid media
//   gpu-forward     actual Qwen3VlBackend forward (Root only; needs CUDA)
//   gpu-quality     actual Backend 3-pool + raw F32 serialization (Root only)
//   gpu-lifecycle   historical two-hold smoke (Root only)
//   gpu-owning-life full ordinary-baseline all-8-fixture owning lifecycle (Root only)
//   gpu-quality-life one cold load for quality serialization plus owning lifecycle (Root only)
//
// The author CPU build runs manifest/factory/lifecycle plus parser/packing/report
// contract tests and never constructs CUDA or executes a model.

use infinity_backend_oxide::qwen3_vl_batch::pack_batch;
use infinity_backend_oxide::qwen3_vl_config::{Dtype, Qwen3VlConfig, ReadoutKind};
use infinity_backend_oxide::qwen3_vl_fixtures::{
    fixture_batch, pool_from_name, read_fixture_file, Fixture,
};
use infinity_backend_oxide::qwen3_vl_inventory::{validate_checkpoint_path, InventoryScope};
use infinity_backend_oxide::qwen3_vl_readout::Qwen3VlReadout;
use infinity_backend_oxide::qwen3_vl_service::Qwen3VlBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ImagePatches, MultimodalEncoding, Pool};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_CHECKPOINTS: &[&str] = &[
    "/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B",
    "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b",
    "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2",
    "/tmp/perplexity-embedding-audit/multivector-support/nvidia--nemotron-colembed-vl-8b-v2",
];

const OWNING_LIFE_HOLD_A: &str = "single-cats";
const OWNING_LIFE_HOLD_B: &str = "varying-shapes";
const OWNING_LIFE_CHANGED: &str = "three-frame-cats";
const OWNING_LIFE_RECOVER: &str = "ragged-three-seq";

fn sha256(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn readout_for(
    root: &Path,
    config: &Qwen3VlConfig,
) -> Result<Option<Qwen3VlReadout>, Box<dyn std::error::Error>> {
    if config.readout.kind == ReadoutKind::Embedding && !config.readout.multi_vector {
        return Ok(None);
    }
    Ok(Some(Qwen3VlReadout::load(
        root,
        &config.readout,
        config.text.hidden_size,
    )?))
}

fn manifest(root: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let config = Qwen3VlConfig::from_path(root)?;
    let alias = infinity_backend_oxide::qwen3_vl_alias(root)?
        .ok_or("normal factory did not detect the Qwen3-VL alias")?;
    let (_, report) = validate_checkpoint_path(root, InventoryScope::Full)?;
    let readout = readout_for(root, &config)?;
    let readout_width = readout.as_ref().map(Qwen3VlReadout::output_width);
    Ok(json!({
        "checkpoint": root,
        "config_sha256": sha256(&root.join("config.json"))?,
        "alias": alias.model_type(),
        "model_type": config.model_type,
        "dtype": match config.text.dtype {
            Dtype::Bf16 => "bfloat16",
            Dtype::F16 => "float16",
            Dtype::F32 => "float32",
        },
        "text_hidden": config.text.hidden_size,
        "vision_hidden": config.vision.hidden_size,
        "vision_depth": config.vision.depth,
        "deepstack": config.vision.deepstack_visual_indexes,
        "readout_kind": format!("{:?}", config.readout.kind),
        "multi_vector": config.readout.multi_vector,
        "norm_eps": config.readout.norm_eps,
        "readout_width": readout_width,
        "tensor_count": report.bound.len(),
        "readout_tensors": report.readout_count,
        "total_bytes": report.total_bytes,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1e3,
    }))
}

fn factory(root: &Path, fixtures: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let config = Qwen3VlConfig::from_path(root)?;
    let _ = readout_for(root, &config)?;
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    let file = read_fixture_file(fixtures)?;
    let mut records = Vec::new();
    for fixture in &file.fixtures {
        let raw = config.readout.multi_vector;
        let batch = fixture_batch(fixture, fixture_root, raw)?;
        let pool = pool_from_name(fixture.pool.as_deref())?;
        let request = pack_batch(&batch, pool, raw)?;
        records.push(json!({
            "fixture": fixture.name,
            "tokens": request.tokens,
            "sequences": request.sequences,
            "images": request.forward.images.len(),
            "merged_rows": request.forward.media_positions.is_some(),
            "raw": raw,
        }));
    }
    // Typed, recoverable invalid media: wrong pixel payload size.
    let invalid = Batch {
        multimodal: vec![Some(Arc::new(MultimodalEncoding {
            images: vec![(
                0,
                Arc::new(ImagePatches {
                    pixels: vec![0.0; 3],
                    grid_thw: [1, 2, 2],
                    patch_dim: 4,
                    merge_size: 2,
                }),
            )],
            audios: vec![],
            position_ids: [vec![0, 0], vec![0, 0], vec![0, 0]],
            memory: None,
        }))],
        input_ids: vec![1, 2],
        token_type_ids: vec![0, 0],
        position_ids: vec![0, 1],
        cumulative_seq_lengths: vec![0, 2],
        max_length: 2,
        pooled_indices: vec![0],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let error =
        pack_batch(&invalid, Pool::LastToken, false).expect_err("invalid media must be rejected");
    // The same consumer path must remain usable after the typed error.
    let valid = fixture_batch(&file.fixtures[0], fixture_root, config.readout.multi_vector)?;
    let _ = pack_batch(&valid, Pool::LastToken, config.readout.multi_vector)?;
    Ok(json!({
        "checkpoint": root,
        "fixtures": records,
        "invalid_media": {
            "kind": format!("{error:?}"),
            "recoverable": true,
            "valid_after_error": true
        }
    }))
}

fn lifecycle(
    checkpoints: &[PathBuf],
    fixtures: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut rounds = Vec::new();
    for round in 0..3 {
        let order: Vec<&PathBuf> = if round % 2 == 0 {
            checkpoints.iter().collect()
        } else {
            checkpoints.iter().rev().collect()
        };
        let mut held = Vec::new();
        for root in order {
            held.push(manifest(root)?);
            held.push(factory(root, fixtures)?);
        }
        // Keep the whole growth set alive, then drop it as one generation.
        rounds.push(json!({
            "round": round,
            "held": held.len(),
            "dropped": true,
        }));
    }
    Ok(json!({
        "checkpoint_count": checkpoints.len(),
        "rounds": rounds,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1e3,
        "invalid_media_recoverable": true
    }))
}

fn gpu_forward(
    checkpoints: &[PathBuf],
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    let file = read_fixture_file(fixtures)?;
    let mut records = Vec::new();
    for root in checkpoints {
        let config = Qwen3VlConfig::from_path(root)?;
        let pool = Pool::LastToken;
        let backend = Qwen3VlBackend::new_with_pool(root, "bfloat16", 0, pool.clone())?;
        for fixture in &file.fixtures {
            let batch = fixture_batch(fixture, fixture_root, config.readout.multi_vector)?;
            let started = Instant::now();
            let result = backend.embed(batch)?;
            let values = result
                .values()
                .map(|embedding| match embedding {
                    Embedding::Pooled(row) => row.len(),
                    Embedding::All(rows) => rows.iter().map(Vec::len).sum(),
                })
                .sum::<usize>();
            records.push(json!({
                "checkpoint": root,
                "fixture": fixture.name,
                "values": values,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1e3,
            }));
        }
    }
    std::fs::create_dir_all(output)?;
    let report = json!({"schema":"qwen3-vl-host-consumer-gpu/v1","records":records});
    std::fs::write(
        output.join("gpu-forward.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn write_rows(
    directory: &Path,
    fixture: &str,
    kind: &str,
    width: usize,
    sequences: usize,
    cumulative: &[u32],
    values: &[f32],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("{fixture}.{kind}.f32"));
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, &bytes)?;
    Ok(json!({
        "fixture": fixture,
        "kind": kind,
        "width": width,
        "sequences": sequences,
        "values": values.len(),
        "cumulative": cumulative,
        "sha256": format!("{:x}", Sha256::digest(&bytes)),
        "path": path,
        "finite": values.iter().all(|value| value.is_finite()),
    }))
}

fn pooled_rows(
    embeddings: &infinity_v3_backend_core::Embeddings,
    sequences: usize,
    width: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut values = Vec::with_capacity(sequences * width);
    for row in 0..sequences {
        match embeddings.get(&row) {
            Some(Embedding::Pooled(vector)) if vector.len() == width => {
                values.extend_from_slice(vector)
            }
            _ => return Err(format!("missing pooled row {row}").into()),
        }
    }
    Ok(values)
}

fn raw_rows(
    embeddings: &infinity_v3_backend_core::Embeddings,
    sequences: usize,
    width: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for row in 0..sequences {
        match embeddings.get(&row) {
            Some(Embedding::All(rows)) => {
                for token in rows {
                    if token.len() != width {
                        return Err(format!("raw row width {} != {width}", token.len()).into());
                    }
                    values.extend_from_slice(token);
                }
            }
            _ => return Err(format!("missing raw row {row}").into()),
        }
    }
    Ok(values)
}

fn f32_sha(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn exact_f32(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn fixture_shape(fixture: &Fixture) -> String {
    let sequences = fixture.cumulative.len().saturating_sub(1);
    let grids = fixture
        .images
        .iter()
        .map(|image| {
            format!(
                "{}x{}x{}",
                image.grid_thw[0], image.grid_thw[1], image.grid_thw[2]
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "sequences={sequences};tokens={};images={};grids=[{grids}]",
        fixture.ids.len(),
        fixture.images.len()
    )
}

fn find_fixture<'a>(
    file: &'a infinity_backend_oxide::qwen3_vl_fixtures::FixtureFile,
    name: &str,
) -> Result<&'a Fixture, Box<dyn std::error::Error>> {
    file.fixtures
        .iter()
        .find(|fixture| fixture.name == name)
        .ok_or_else(|| format!("fixture {name} is missing from the manifest").into())
}

fn embedding_rows(
    embeddings: &infinity_v3_backend_core::Embeddings,
    sequences: usize,
    width: usize,
    raw: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if raw {
        raw_rows(embeddings, sequences, width)
    } else {
        pooled_rows(embeddings, sequences, width)
    }
}

fn ordinary_vector(
    backend: &Qwen3VlBackend,
    fixture: &Fixture,
    fixture_root: &Path,
    raw: bool,
    width: usize,
) -> Result<(Batch, Vec<f32>, String), Box<dyn std::error::Error>> {
    let batch = fixture_batch(fixture, fixture_root, raw)?;
    let sequences = batch.cumulative_seq_lengths.len() - 1;
    let embeddings = backend.embed(batch.clone())?;
    let values = embedding_rows(&embeddings, sequences, width, raw)?;
    let digest = f32_sha(&values);
    Ok((batch, values, digest))
}

struct OwningLifeEvidence {
    checkpoint: PathBuf,
    fixture_count: usize,
    width: usize,
    multi_vector: bool,
    baselines: Vec<serde_json::Value>,
    hold_a: String,
    hold_a_shape: String,
    hold_b: String,
    hold_b_shape: String,
    drop_health: bool,
    complete_b_exact: bool,
    changed: String,
    changed_shape: String,
    changed_exact: bool,
    invalid_media_typed: bool,
    invalid_media_zero_selectors: bool,
    recovered: String,
    recovered_exact: bool,
    health_after_recovery: bool,
}

impl OwningLifeEvidence {
    fn report(&self) -> serde_json::Value {
        json!({
            "schema": "qwen3-vl-host-consumer-gpu-owning-life/v1",
            "checkpoint": &self.checkpoint,
            "route": "Qwen3VlBackend::submit_embed",
            "backend_constructions": 1,
            "holding_contract": "two_held_then_drop_a_then_complete_b_then_next",
            "third_submit_while_two_held": false,
            "fixture_count": self.fixture_count,
            "width": self.width,
            "multi_vector": self.multi_vector,
            "baselines": &self.baselines,
            "held": {
                "a": &self.hold_a,
                "a_shape": &self.hold_a_shape,
                "b": &self.hold_b,
                "b_shape": &self.hold_b_shape,
            },
            "drop_without_completion_fenced": self.drop_health,
            "complete_b_exact": self.complete_b_exact,
            "changed_shape": {
                "fixture": &self.changed,
                "shape": &self.changed_shape,
                "exact": self.changed_exact,
            },
            "invalid_media_typed": self.invalid_media_typed,
            "invalid_media_zero_selectors": self.invalid_media_zero_selectors,
            "recovered": {
                "fixture": &self.recovered,
                "exact": self.recovered_exact,
            },
            "health_after_recovery": self.health_after_recovery,
        })
    }
}

fn gpu_quality(
    checkpoints: &[PathBuf],
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let root = checkpoints.first().ok_or("missing checkpoint")?;
    let config = Qwen3VlConfig::from_path(root)?;
    let backend = Qwen3VlBackend::new_with_pool(root, "bfloat16", 0, Pool::LastToken)?;
    gpu_quality_with_backend(root, &config, &backend, fixtures, output)
}

fn gpu_quality_with_backend(
    root: &Path,
    config: &Qwen3VlConfig,
    backend: &Qwen3VlBackend,
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    let file = read_fixture_file(fixtures)?;
    let width = backend.output_width()?;
    let mut pools = Vec::new();
    if !config.readout.multi_vector {
        for (pool, name) in [
            (Pool::Cls, "pool0"),
            (Pool::Mean, "pool1"),
            (Pool::LastToken, "pool2"),
        ] {
            let directory = output.join(name);
            let mut records = Vec::new();
            for fixture in &file.fixtures {
                let mut batch = fixture_batch(fixture, fixture_root, false)?;
                let sequences = batch.cumulative_seq_lengths.len() - 1;
                batch.pooled_indices = (0..sequences as u32).collect();
                batch.raw_indices.clear();
                let embeddings = backend
                    .submit_with_pool(batch.clone(), pool.clone())?
                    .complete()?;
                let values = pooled_rows(&embeddings, sequences, width)?;
                records.push(write_rows(
                    &directory,
                    &fixture.name,
                    name,
                    width,
                    sequences,
                    &batch.cumulative_seq_lengths,
                    &values,
                )?);
            }
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                directory.join("oxide-consumer.json"),
                serde_json::to_vec_pretty(
                    &json!({"schema":"qwen3-vl-quality-consumer/v1","pool":name,"records":records}),
                )?,
            )?;
            pools.push(name);
        }
    }
    let directory = output.join("raw");
    let mut records = Vec::new();
    for fixture in &file.fixtures {
        let mut batch = fixture_batch(fixture, fixture_root, true)?;
        let sequences = batch.cumulative_seq_lengths.len() - 1;
        batch.pooled_indices.clear();
        batch.raw_indices = (0..sequences as u32).collect();
        let embeddings = backend
            .submit_with_pool(batch.clone(), Pool::LastToken)?
            .complete()?;
        let values = raw_rows(&embeddings, sequences, width)?;
        records.push(write_rows(
            &directory,
            &fixture.name,
            "tokens",
            width,
            sequences,
            &batch.cumulative_seq_lengths,
            &values,
        )?);
    }
    std::fs::create_dir_all(&directory)?;
    std::fs::write(
        directory.join("oxide-consumer.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"qwen3-vl-quality-consumer/v1","pool":"raw","records":records}),
        )?,
    )?;
    let report = json!({
        "schema": "qwen3-vl-host-consumer-gpu-quality/v1",
        "checkpoint": root,
        "multi_vector": config.readout.multi_vector,
        "width": width,
        "pools": pools,
        "raw": true,
    });
    std::fs::write(
        output.join("gpu-quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn gpu_owning_life(
    checkpoints: &[PathBuf],
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let root = checkpoints.first().ok_or("missing checkpoint")?;
    let config = Qwen3VlConfig::from_path(root)?;
    let backend = Qwen3VlBackend::new_with_pool(root, "bfloat16", 0, Pool::LastToken)?;
    gpu_owning_life_with_backend(root, &config, &backend, fixtures, output)
}

fn gpu_owning_life_with_backend(
    root: &Path,
    config: &Qwen3VlConfig,
    backend: &Qwen3VlBackend,
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    let file = read_fixture_file(fixtures)?;
    if file.fixtures.len() != 8 {
        return Err(format!(
            "owning-life requires all 8 real fixtures, found {}",
            file.fixtures.len()
        )
        .into());
    }
    let width = backend.output_width()?;
    let raw = config.readout.multi_vector;
    let mut by_name: std::collections::BTreeMap<String, (Batch, Vec<f32>, String)> =
        std::collections::BTreeMap::new();
    let mut baselines = Vec::new();
    for fixture in &file.fixtures {
        let (batch, values, digest) = ordinary_vector(backend, fixture, fixture_root, raw, width)?;
        let sequences = batch.cumulative_seq_lengths.len() - 1;
        let record = write_rows(
            &output.join("life-baselines"),
            &fixture.name,
            "baseline",
            width,
            sequences,
            &batch.cumulative_seq_lengths,
            &values,
        )?;
        baselines.push(json!({
            "fixture": &fixture.name,
            "shape": fixture_shape(fixture),
            "sequences": sequences,
            "width": width,
            "values": values.len(),
            "sha256": &digest,
            "file": record,
        }));
        by_name.insert(fixture.name.clone(), (batch, values, digest));
    }

    let hold_a = find_fixture(&file, OWNING_LIFE_HOLD_A)?;
    let hold_b = find_fixture(&file, OWNING_LIFE_HOLD_B)?;
    let changed = find_fixture(&file, OWNING_LIFE_CHANGED)?;
    let recover = find_fixture(&file, OWNING_LIFE_RECOVER)?;
    let (batch_a, _, _) = by_name.get(&hold_a.name).ok_or("missing A baseline")?;
    let (batch_b, baseline_b, _) = by_name.get(&hold_b.name).ok_or("missing B baseline")?;
    let (_, baseline_c, _) = by_name
        .get(&changed.name)
        .ok_or("missing changed baseline")?;
    let (_, baseline_d, _) = by_name
        .get(&recover.name)
        .ok_or("missing recovery baseline")?;

    // Exactly two owning requests are held on the actual Backend route. The
    // normal service keeps blocking SlotPool::acquire semantics, so no third
    // submit runs while both leases are live.
    let held_a = backend.submit_embed(batch_a.clone())?;
    let held_b = backend.submit_embed(batch_b.clone())?;
    // Drop A without invoking its closure. The RAII owner fences and returns
    // its HostIo lease before B is completed.
    drop(held_a);
    backend.health()?;
    let drop_health = true;
    let embeddings_b = held_b.complete()?;
    let values_b = embedding_rows(
        &embeddings_b,
        batch_b.cumulative_seq_lengths.len() - 1,
        width,
        raw,
    )?;
    let complete_b_exact = exact_f32(&values_b, baseline_b);
    if !complete_b_exact {
        return Err("completed B differs from its ordinary baseline".into());
    }

    // A different fixture with a different shape must be valid after the
    // workspace/lease generation changed.
    let changed_batch = fixture_batch(changed, fixture_root, raw)?;
    let changed_embeddings = backend.submit_embed(changed_batch.clone())?.complete()?;
    let changed_values = embedding_rows(
        &changed_embeddings,
        changed_batch.cumulative_seq_lengths.len() - 1,
        width,
        raw,
    )?;
    let changed_exact = exact_f32(&changed_values, baseline_c);
    if !changed_exact {
        return Err("changed-shape request differs from its ordinary baseline".into());
    }
    backend.health()?;

    // Invalid media must be rejected by the config-aware preflight even when
    // there are no output selectors, before any lease or ticket.
    let mut malformed = batch_a.clone();
    malformed.pooled_indices.clear();
    malformed.raw_indices.clear();
    malformed.multimodal = vec![Some(Arc::new(MultimodalEncoding {
        images: vec![(
            0,
            Arc::new(ImagePatches {
                pixels: vec![0.0; 3],
                grid_thw: [1, 2, 2],
                patch_dim: 4,
                merge_size: 2,
            }),
        )],
        audios: vec![],
        position_ids: [
            vec![0; malformed.input_ids.len()],
            vec![0; malformed.input_ids.len()],
            vec![0; malformed.input_ids.len()],
        ],
        memory: None,
    }))];
    let invalid = backend.submit_embed(malformed);
    let invalid_media_typed = matches!(
        &invalid,
        Err(infinity_v3_backend_core::BackendError::Start(_))
            | Err(infinity_v3_backend_core::BackendError::Inference(_))
    );
    if let Ok(inference) = invalid {
        inference.complete()?;
    }
    if !invalid_media_typed {
        return Err(
            "invalid media with empty selectors was not rejected with a typed error".into(),
        );
    }
    backend.health()?;

    // A different fixture must produce its ordinary vector after the typed
    // error, proving the two-lease pool recovered.
    let recovered_batch = fixture_batch(recover, fixture_root, raw)?;
    let recovered_embeddings = backend.submit_embed(recovered_batch.clone())?.complete()?;
    let recovered_values = embedding_rows(
        &recovered_embeddings,
        recovered_batch.cumulative_seq_lengths.len() - 1,
        width,
        raw,
    )?;
    let recovered_exact = exact_f32(&recovered_values, baseline_d);
    if !recovered_exact {
        return Err("recovery request differs from its ordinary baseline".into());
    }
    backend.health()?;

    let report = OwningLifeEvidence {
        checkpoint: root.to_path_buf(),
        fixture_count: file.fixtures.len(),
        width,
        multi_vector: raw,
        baselines,
        hold_a: hold_a.name.clone(),
        hold_a_shape: fixture_shape(hold_a),
        hold_b: hold_b.name.clone(),
        hold_b_shape: fixture_shape(hold_b),
        drop_health,
        complete_b_exact,
        changed: changed.name.clone(),
        changed_shape: fixture_shape(changed),
        changed_exact,
        invalid_media_typed,
        invalid_media_zero_selectors: true,
        recovered: recover.name.clone(),
        recovered_exact,
        health_after_recovery: true,
    }
    .report();
    std::fs::create_dir_all(output)?;
    std::fs::write(
        output.join("gpu-owning-life.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn gpu_quality_life(
    checkpoints: &[PathBuf],
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let root = checkpoints.first().ok_or("missing checkpoint")?;
    let config = Qwen3VlConfig::from_path(root)?;
    let backend = Qwen3VlBackend::new_with_pool(root, "bfloat16", 0, Pool::LastToken)?;
    let quality = gpu_quality_with_backend(root, &config, &backend, fixtures, output)?;
    let owning_life = gpu_owning_life_with_backend(root, &config, &backend, fixtures, output)?;
    let report = json!({
        "schema": "qwen3-vl-host-consumer-gpu-quality-life/v1",
        "checkpoint": root,
        "backend_constructions": 1,
        "quality": quality,
        "owning_life": owning_life,
    });
    std::fs::create_dir_all(output)?;
    std::fs::write(
        output.join("gpu-quality-life.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn gpu_lifecycle(
    checkpoints: &[PathBuf],
    fixtures: &Path,
    output: &Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let root = checkpoints.first().ok_or("missing checkpoint")?;
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    let file = read_fixture_file(fixtures)?;
    let config = Qwen3VlConfig::from_path(root)?;
    let batch = fixture_batch(&file.fixtures[0], fixture_root, config.readout.multi_vector)?;
    let backend = Qwen3VlBackend::new_with_pool(root, "bfloat16", 0, Pool::LastToken)?;
    // Two deferred owners held on the actual production path. The normal
    // Backend service keeps blocking SlotPool::acquire semantics: a direct
    // same-thread third submit must first release one of the two leases.
    let first = backend.submit_embed(batch.clone())?;
    let second = backend.submit_embed(batch.clone())?;
    // Drop the first Inference without invoking its closure. The RAII owner
    // Drop must fence/quarantine before freeing the request resources and
    // returning its HostIo slot.
    drop(first);
    backend.health()?;
    let third = backend.submit_embed(batch.clone())?;
    third.complete()?;
    second.complete()?;
    backend.health()?;
    // Malformed media with zero selectors must be rejected even though there
    // is no output selection, before any lease is taken.
    let malformed = {
        let mut b = batch.clone();
        b.pooled_indices.clear();
        b.raw_indices.clear();
        let image = Arc::new(ImagePatches {
            pixels: vec![0.0; 3],
            grid_thw: [1, 2, 2],
            patch_dim: 4,
            merge_size: 2,
        });
        b.multimodal = vec![Some(Arc::new(MultimodalEncoding {
            images: vec![(0, image)],
            audios: vec![],
            position_ids: [
                vec![0; b.input_ids.len()],
                vec![0; b.input_ids.len()],
                vec![0; b.input_ids.len()],
            ],
            memory: None,
        }))];
        b
    };
    let invalid = backend.submit_embed(malformed);
    let invalid_typed = matches!(
        invalid,
        Err(infinity_v3_backend_core::BackendError::Start(_))
            | Err(infinity_v3_backend_core::BackendError::Inference(_))
    );
    if let Ok(inference) = invalid {
        inference.complete()?;
    }
    backend.health()?;
    let report = json!({
        "schema": "qwen3-vl-host-consumer-gpu-lifecycle/v1",
        "checkpoint": root,
        "direct_third_requires_released_lease": true,
        "drop_without_completion_health": true,
        "invalid_media_typed": invalid_typed,
        "health_after_invalid": true
    });
    std::fs::create_dir_all(output)?;
    std::fs::write(
        output.join("gpu-lifecycle.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Cli {
    checkpoints: Vec<PathBuf>,
    fixtures: PathBuf,
    output: PathBuf,
    mode: String,
}

fn parse_args<I>(args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut checkpoints = Vec::new();
    let mut fixtures = PathBuf::from("native-baseline/fixtures/fixtures.json");
    let mut output = PathBuf::from("host-consumer-out");
    let mut mode = "all".to_owned();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--checkpoint" => checkpoints.push(PathBuf::from(
                args.next()
                    .ok_or_else(|| "--checkpoint requires a value".to_owned())?,
            )),
            "--fixtures" => {
                fixtures = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--fixtures requires a value".to_owned())?,
                )
            }
            "--out" => {
                output = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out requires a value".to_owned())?,
                )
            }
            "--mode" => {
                mode = args
                    .next()
                    .ok_or_else(|| "--mode requires a value".to_owned())?
            }
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    if checkpoints.is_empty() {
        checkpoints = DEFAULT_CHECKPOINTS.iter().map(PathBuf::from).collect();
    }
    Ok(Cli {
        checkpoints,
        fixtures,
        output,
        mode,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Cli {
        checkpoints,
        fixtures,
        output,
        mode,
    } = parse_args(std::env::args().skip(1))?;
    std::fs::create_dir_all(&output)?;
    if mode == "gpu-forward" {
        let report = gpu_forward(&checkpoints, &fixtures, &output)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if mode == "gpu-lifecycle" {
        let report = gpu_lifecycle(&checkpoints, &fixtures, &output)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if mode == "gpu-owning-life" {
        let report = gpu_owning_life(&checkpoints, &fixtures, &output)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if mode == "gpu-quality" {
        let report = gpu_quality(&checkpoints, &fixtures, &output)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if mode == "gpu-quality-life" {
        let report = gpu_quality_life(&checkpoints, &fixtures, &output)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let mut report = json!({"schema":"qwen3-vl-host-consumer/v1","mode":mode});
    if mode == "all" || mode == "manifest" {
        let mut records = Vec::new();
        for root in &checkpoints {
            records.push(manifest(root)?);
        }
        report["manifest"] = json!(records);
    }
    if mode == "all" || mode == "factory" {
        let mut records = Vec::new();
        for root in &checkpoints {
            records.push(factory(root, &fixtures)?);
        }
        report["factory"] = json!(records);
    }
    if mode == "all" || mode == "lifecycle" {
        report["lifecycle"] = lifecycle(&checkpoints, &fixtures)?;
    }
    std::fs::write(
        output.join("host-consumer.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_owning_life_and_rejects_missing_values() {
        let cli = parse_args([
            "--mode".to_owned(),
            "gpu-owning-life".to_owned(),
            "--checkpoint".to_owned(),
            "/tmp/checkpoint".to_owned(),
        ])
        .unwrap();
        assert_eq!(cli.mode, "gpu-owning-life");
        assert_eq!(cli.checkpoints, vec![PathBuf::from("/tmp/checkpoint")]);
        assert_eq!(
            cli.fixtures,
            PathBuf::from("native-baseline/fixtures/fixtures.json")
        );
        assert_eq!(cli.output, PathBuf::from("host-consumer-out"));
        assert!(parse_args(["--mode".to_owned()]).is_err());
        assert!(parse_args(["--bogus".to_owned()]).is_err());
    }

    #[test]
    fn real_fixtures_pack_and_empty_selector_media_is_typed() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../native-baseline/fixtures/fixtures.json");
        let fixture_root = manifest.parent().unwrap();
        let file = read_fixture_file(&manifest).unwrap();
        assert_eq!(file.fixtures.len(), 8);
        for fixture in &file.fixtures {
            let batch = fixture_batch(fixture, fixture_root, false).unwrap();
            let request = pack_batch(&batch, Pool::LastToken, false).unwrap();
            assert_eq!(request.sequences, fixture.cumulative.len() - 1);
            assert_eq!(request.tokens, fixture.ids.len());
        }
        let hold_a = find_fixture(&file, OWNING_LIFE_HOLD_A).unwrap();
        let hold_b = find_fixture(&file, OWNING_LIFE_HOLD_B).unwrap();
        let changed = find_fixture(&file, OWNING_LIFE_CHANGED).unwrap();
        let recover = find_fixture(&file, OWNING_LIFE_RECOVER).unwrap();
        assert_ne!(fixture_shape(hold_a), fixture_shape(hold_b));
        assert_ne!(fixture_shape(hold_b), fixture_shape(changed));
        assert_ne!(fixture_shape(changed), fixture_shape(recover));
        assert!(!hold_a.images.is_empty());
        assert!(hold_b.images.len() > 1);
        assert_eq!(changed.images[0].grid_thw[0], 3);

        let mut malformed = fixture_batch(hold_a, fixture_root, false).unwrap();
        malformed.pooled_indices.clear();
        malformed.raw_indices.clear();
        malformed.multimodal = vec![Some(Arc::new(MultimodalEncoding {
            images: vec![(
                0,
                Arc::new(ImagePatches {
                    pixels: vec![0.0; 3],
                    grid_thw: [1, 2, 2],
                    patch_dim: 4,
                    merge_size: 2,
                }),
            )],
            audios: vec![],
            position_ids: [
                vec![0; malformed.input_ids.len()],
                vec![0; malformed.input_ids.len()],
                vec![0; malformed.input_ids.len()],
            ],
            memory: None,
        }))];
        assert!(pack_batch(&malformed, Pool::LastToken, false).is_err());
    }

    #[test]
    fn owning_life_report_shape_has_no_fake_claims() {
        let evidence = OwningLifeEvidence {
            checkpoint: PathBuf::from("/tmp/checkpoint"),
            fixture_count: 8,
            width: 4096,
            multi_vector: false,
            baselines: vec![json!({
                "fixture": "single-cats",
                "shape": "sequences=1",
                "sequences": 1,
                "width": 4096,
                "values": 4096,
                "sha256": "00",
            })],
            hold_a: OWNING_LIFE_HOLD_A.to_owned(),
            hold_a_shape: fixture_shape(
                &read_fixture_file(
                    &Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../../../native-baseline/fixtures/fixtures.json"),
                )
                .unwrap()
                .fixtures[0],
            ),
            hold_b: OWNING_LIFE_HOLD_B.to_owned(),
            hold_b_shape: "sequences=1;tokens=10;images=2;grids=[1x30x40,1x64x64]".to_owned(),
            drop_health: true,
            complete_b_exact: true,
            changed: OWNING_LIFE_CHANGED.to_owned(),
            changed_shape: "sequences=1;tokens=10;images=1;grids=[3x30x40]".to_owned(),
            changed_exact: true,
            invalid_media_typed: true,
            invalid_media_zero_selectors: true,
            recovered: OWNING_LIFE_RECOVER.to_owned(),
            recovered_exact: true,
            health_after_recovery: true,
        };
        let report = evidence.report();
        assert_eq!(
            report["schema"],
            "qwen3-vl-host-consumer-gpu-owning-life/v1"
        );
        assert_eq!(report["backend_constructions"], 1);
        assert_eq!(report["third_submit_while_two_held"], false);
        assert_eq!(report["drop_without_completion_fenced"], true);
        assert_eq!(report["fixture_count"], 8);
        assert_eq!(report["invalid_media_zero_selectors"], true);
        let rendered = serde_json::to_string(&report).unwrap();
        assert!(!rendered.contains("Overloaded"));
        assert!(!rendered.contains("gpu_executed"));
    }
}
