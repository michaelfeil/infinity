// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Qwen3 MoE ordinary serving, request ownership, token-exact plan capacity
//! and deferred lifetime recovery on the actual 61 GB checkpoint.

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const HIDDEN: usize = 2048;
const VOCAB: u32 = 151_936;
const MAX_POSITION: u32 = 40_960;
const EXPERTS: usize = 128;
const TOP_K: usize = 8;
const MOE_INTERMEDIATE: usize = 768;

type OutputBits = BTreeMap<usize, Vec<Vec<u32>>>;

fn fixture(lengths: &[u32], pooled: &[u32], raw: &[u32]) -> Batch {
    let mut cumulative_seq_lengths = vec![0];
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    for &length in lengths {
        assert!(length > 0 && length <= MAX_POSITION);
        for position in 0..length {
            input_ids.push(20 + (input_ids.len() as u32 * 13) % 20_000);
            position_ids.push(position);
        }
        cumulative_seq_lengths.push(input_ids.len() as u32);
    }
    Batch {
        multimodal: Vec::new(),
        token_type_ids: vec![0; input_ids.len()],
        input_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: lengths.iter().copied().max().unwrap(),
        pooled_indices: pooled.to_vec(),
        raw_indices: raw.to_vec(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn bits(result: &Embeddings, batch: &Batch) -> OutputBits {
    let raw: BTreeSet<_> = batch.raw_indices.iter().map(|&i| i as usize).collect();
    let selected: BTreeSet<_> = batch
        .pooled_indices
        .iter()
        .chain(&batch.raw_indices)
        .map(|&i| i as usize)
        .collect();
    assert_eq!(result.len(), selected.len());
    let mut output = BTreeMap::new();
    for (&index, embedding) in result.iter() {
        assert!(selected.contains(&index));
        let rows: Vec<&[f32]> = match embedding {
            Embedding::Pooled(row) => {
                assert!(!raw.contains(&index));
                vec![row]
            }
            Embedding::All(rows) => {
                assert!(raw.contains(&index));
                let length =
                    batch.cumulative_seq_lengths[index + 1] - batch.cumulative_seq_lengths[index];
                assert_eq!(rows.len(), length as usize);
                rows.iter().map(Vec::as_slice).collect()
            }
        };
        let rows = rows
            .into_iter()
            .map(|row| {
                assert_eq!(row.len(), HIDDEN);
                assert!(row.iter().all(|value| value.is_finite()));
                row.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        assert!(output.insert(index, rows).is_none());
    }
    assert_eq!(output.keys().copied().collect::<BTreeSet<_>>(), selected);
    output
}
fn completed(call: Inference<Embeddings>, batch: &Batch) -> OutputBits {
    bits(&call.complete().unwrap(), batch)
}
fn deferred(backend: &OxideBackend, batch: &Batch) -> Inference<Embeddings> {
    let call = backend.submit_embed(batch.clone()).unwrap();
    assert!(matches!(&call, Inference::Deferred(_)));
    call
}
fn reference(backend: &OxideBackend, batch: &Batch) -> OutputBits {
    let first = bits(&backend.embed(batch.clone()).unwrap(), batch);
    assert_eq!(bits(&backend.embed(batch.clone()).unwrap(), batch), first);
    first
}
fn rejected(backend: &OxideBackend, batch: Batch) {
    let error = match backend.submit_embed(batch) {
        Ok(_) => panic!("invalid Qwen3 MoE request was admitted"),
        Err(error) => error,
    };
    assert!(matches!(error, BackendError::Inference(_)), "{error:?}");
    backend.health().unwrap();
}
fn phase(message: impl AsRef<str>) {
    use std::io::Write as _;
    let mut stderr = std::io::stderr().lock();
    writeln!(stderr, "[qwen3-moe-serving-v1] {}", message.as_ref()).unwrap();
    stderr.flush().unwrap();
}
fn run_pool(path: &std::path::Path, pool: Pool) {
    let started = std::time::Instant::now();
    let pool_label = format!("{pool:?}");
    phase(format!("run_pool_begin pool={pool_label}"));
    phase(format!("backend_new_start pool={pool_label}"));
    let backend = OxideBackend::new(path, "bfloat16", ModelType::Embedding(pool), 0).unwrap();
    phase(format!(
        "backend_new_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));
    // A=7, B=129, A=7 again, C=1, D=33. B retains capacity 256 while its native
    // plan is token-exact at 129, so the following A=7 must shrink without a
    // plan/arena token-offset confusion. A has duplicate pooled selections and
    // B has duplicate raw rows overlapping pooled selection 1.
    let a = fixture(&[3, 4], &[1, 0, 1], &[]);
    let b = fixture(&[1, 127, 1], &[2, 0, 1], &[1, 1]);
    let c = fixture(&[1], &[], &[0]);
    let d = fixture(&[33], &[0], &[]);
    phase(format!("baseline_references_start pool={pool_label}"));
    let base_a = reference(&backend, &a);
    let base_b = reference(&backend, &b);
    assert_eq!(reference(&backend, &a), base_a);
    let base_c = reference(&backend, &c);
    let base_d = reference(&backend, &d);
    // The same physical B rows through a full raw route keep index 1 bit-exact.
    let raw_b = fixture(&[1, 127, 1], &[], &[2, 1, 0]);
    assert_eq!(reference(&backend, &raw_b).get(&1), base_b.get(&1));
    // Qwen3 MoE follows native Qwen3: type-id values are staged but ignored.
    let mut changed = b.clone();
    for (index, kind) in changed.token_type_ids.iter_mut().enumerate() {
        *kind = if index % 2 == 0 { u32::MAX } else { 1 };
    }
    assert_eq!(reference(&backend, &changed), base_b);
    phase(format!(
        "baseline_references_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));
    // Two real leases may coexist; B completes first while A is still pending.
    phase(format!("coexisting_deferred_start pool={pool_label}"));
    let held_a = deferred(&backend, &a);
    let held_b = deferred(&backend, &b);
    assert_eq!(completed(held_b, &b), base_b);
    assert_eq!(completed(held_a, &a), base_a);
    phase(format!(
        "coexisting_deferred_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));
    phase(format!("drop_temporary_deferred_start pool={pool_label}"));
    let held_a = deferred(&backend, &a);
    drop(deferred(&backend, &c));
    assert_eq!(reference(&backend, &d), base_d);
    assert_eq!(completed(held_a, &a), base_a);
    assert_eq!(reference(&backend, &c), base_c);
    phase(format!(
        "drop_temporary_deferred_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));
    // Every rejection occurs while B owns a real result; B must stay usable.
    phase(format!("reject_while_held_start pool={pool_label}"));
    let held_b = deferred(&backend, &b);
    let mutators: [fn(&mut Batch); 5] = [
        |b| b.input_ids[0] = VOCAB,
        |b| b.position_ids[0] = MAX_POSITION,
        |b| b.token_type_ids.clear(),
        |b| b.raw_indices = vec![2],
        |b| b.cumulative_seq_lengths[1] = 0,
    ];
    for mutate in mutators {
        let mut invalid = a.clone();
        mutate(&mut invalid);
        rejected(&backend, invalid);
    }
    assert_eq!(completed(held_b, &b), base_b);
    assert_eq!(reference(&backend, &a), base_a);
    phase(format!(
        "reject_while_held_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));

    phase(format!("empty_request_start pool={pool_label}"));
    let empty = fixture(&[3, 4], &[], &[]);
    assert!(backend.embed(empty).unwrap().is_empty());
    backend.health().unwrap();
    phase(format!(
        "empty_request_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));

    // Exactly two pending owners outlive the facade and finish against their
    // own references after the model owner is dropped first.
    phase(format!(
        "drop_owner_with_two_deferred_start pool={pool_label}"
    ));
    let held_a = deferred(&backend, &a);
    let held_b = deferred(&backend, &b);
    drop(backend);
    assert_eq!(completed(held_b, &b), base_b);
    assert_eq!(completed(held_a, &a), base_a);
    phase(format!(
        "run_pool_done pool={pool_label} elapsed_ms={}",
        started.elapsed().as_millis()
    ));
}
#[test]
#[ignore = "Requires CUDA, Qwen3 MoE FA4 canonical26 mode22/sidecar modules, INFINITY_OXIDE_TEST_MOE_MODEL and the pinned 61 GB checkpoint"]
fn qwen3_moe_all_pools_ordinary_serving_positive_lifetime() {
    let started = std::time::Instant::now();
    let path = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_TEST_MOE_MODEL")
            .expect("INFINITY_OXIDE_TEST_MOE_MODEL must name the verified Qwen3 MoE checkpoint"),
    );
    phase(format!("test_begin model={}", path.display()));
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "qwen3_moe");
    assert_eq!(
        config["architectures"],
        serde_json::json!(["Qwen3MoeForCausalLM"])
    );
    assert_eq!(config["hidden_size"], HIDDEN);
    assert_eq!(config["vocab_size"], VOCAB);
    assert_eq!(config["max_position_embeddings"], MAX_POSITION);
    assert_eq!(config["moe_intermediate_size"], MOE_INTERMEDIATE);
    assert_eq!(config["num_experts"], EXPERTS);
    assert_eq!(config["num_experts_per_tok"], TOP_K);
    assert_eq!(config["num_attention_heads"], 32);
    assert_eq!(config["num_key_value_heads"], 4);
    assert_eq!(config["head_dim"], 128);
    phase("pre_driver_rejection_start");
    let error = OxideBackend::new(&path, "float16", ModelType::Embedding(Pool::Cls), 0)
        .err()
        .expect("ordinary serving construction admitted non-BF16 Qwen3 MoE");
    assert!(
        error.to_string().contains("requires the BF16 runtime"),
        "{error:?}"
    );
    phase(format!(
        "pre_driver_rejection_done elapsed_ms={}",
        started.elapsed().as_millis()
    ));
    // One engine at a time: the checkpoint is roughly 61 GB and is dropped here.
    // Default is the complete supported pool set; Root may set
    // INFINITY_OXIDE_TEST_MOE_POOLS=cls,mean,last-token for a narrower run.
    let pools = match std::env::var("INFINITY_OXIDE_TEST_MOE_POOLS") {
        Ok(value) => value
            .split(',')
            .map(|name| match name.trim() {
                "cls" => Pool::Cls,
                "mean" => Pool::Mean,
                "last-token" | "last_token" => Pool::LastToken,
                other => panic!("unknown Qwen3 MoE pool {other:?}"),
            })
            .collect::<Vec<_>>(),
        Err(_) => vec![Pool::Cls, Pool::Mean, Pool::LastToken],
    };
    assert!(!pools.is_empty(), "at least one Qwen3 MoE pool is required");
    for pool in pools {
        let pool_label = format!("{pool:?}");
        phase(format!(
            "mode_begin pool={pool_label} elapsed_ms={}",
            started.elapsed().as_millis()
        ));
        run_pool(&path, pool);
        phase(format!(
            "mode_done pool={pool_label} elapsed_ms={}",
            started.elapsed().as_millis()
        ));
    }
    phase(format!(
        "test_done elapsed_ms={}",
        started.elapsed().as_millis()
    ));
}
#[test]
fn duplicate_mixed_fixture_checks_qwen3_moe_raw_override() {
    let batch = fixture(&[1, 2], &[1, 0, 1], &[1, 1]);
    let mut result = Embeddings::default();
    result.insert(0, Embedding::Pooled(vec![1.0; HIDDEN]));
    result.insert(1, Embedding::All(vec![vec![-0.0; HIDDEN]; 2]));
    let readout = bits(&result, &batch);
    assert_eq!(readout.len(), 2);
    assert_eq!(readout[&0].len(), 1);
    assert_eq!(readout[&1].len(), 2);
    assert!(readout[&1][0]
        .iter()
        .all(|&word| word == (-0.0f32).to_bits()));
}
