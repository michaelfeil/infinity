// End-to-end CPU acceptance tests for the cold Qwen3.5-MoE foundation.
//
// These tests construct tiny legal checkpoints, verify Native manifest
// precedence and offsets, exercise real rejection paths, and optionally bind
// the cached 35B metadata headers without reading expert payloads.

use crate::model::Checkpoint;
use crate::qwen35_moe_config::{Qwen35MoeConfig, Qwen35MoeDtype, Qwen35MoeSpec};
use crate::qwen35_moe_contract::{cuda_shared_gate_bf16, cuda_silu_mul_bf16};
use crate::qwen35_moe_layout::{native_workspace_bytes, Qwen35MoeNativeScratchLayout};
use crate::qwen35_moe_router::RouterPlan;
use crate::qwen35_moe_weights::{ExpertStorageKind, GateUpSources, Qwen35MoeWeights};
use half::bf16;
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const ROOT: &str = "model.language_model.";
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir()
            .join("qwen35-moe-v2-tests")
            .join(format!(
                "{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn tiny_config() -> Value {
    json!({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention"
            ],
            "full_attention_interval": 4,
            "attn_output_gate": true,
            "hidden_act": "silu",
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 6,
            "shared_expert_intermediate_size": 6,
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 1.0
            }
        }
    })
}

#[allow(dead_code)]
fn tiny_config_with(mutate: impl FnOnce(&mut Value)) -> Value {
    let mut value = tiny_config();
    mutate(&mut value);
    value
}

fn write_config(path: &Path, value: &Value) {
    fs::write(path.join("config.json"), serde_json::to_vec(value).unwrap()).unwrap();
}

#[derive(Clone)]
struct Entry {
    name: String,
    shape: Vec<usize>,
    dtype: Dtype,
    values: Vec<f32>,
}

fn entry(name: impl Into<String>, shape: Vec<usize>, dtype: Dtype, base: f32) -> Entry {
    let count = shape.iter().product::<usize>();
    Entry {
        name: name.into(),
        shape,
        dtype,
        values: (0..count).map(|index| base + index as f32).collect(),
    }
}

fn bytes(entry: &Entry) -> Vec<u8> {
    let mut out = Vec::new();
    for value in &entry.values {
        match entry.dtype {
            Dtype::BF16 => out.extend_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes()),
            Dtype::F32 => out.extend_from_slice(&value.to_le_bytes()),
            other => panic!("unsupported test dtype {other:?}"),
        }
    }
    out
}

fn write_shards(path: &Path, entries: Vec<Entry>, shards: usize, duplicate_first: bool) {
    let mut payloads = Vec::new();
    for _ in 0..shards {
        payloads.push(Vec::new());
    }
    let mut index = BTreeMap::new();
    let mut seen = Vec::new();
    for (position, item) in entries.iter().enumerate() {
        let shard = position % shards;
        seen.push((item, shard));
    }
    if duplicate_first {
        let (item, _) = seen[0];
        seen.push((item, (seen[0].1 + 1) % shards));
    }
    for (item, shard) in seen {
        let data = bytes(item);
        payloads[shard].push((item.name.clone(), item.shape.clone(), item.dtype, data));
        index.insert(item.name.clone(), format!("model-{shard}.safetensors"));
    }
    for (index_number, payload) in payloads.iter().enumerate() {
        let views = payload
            .iter()
            .map(|(name, shape, dtype, data)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        fs::write(
            path.join(format!("model-{index_number}.safetensors")),
            serialize(
                views
                    .iter()
                    .map(|(name, view)| (name.clone(), view.clone())),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    }
    fs::write(
        path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": index})).unwrap(),
    )
    .unwrap();
}

fn base_entries(packed: bool, separate_too: bool) -> Vec<Entry> {
    let hidden = 16;
    let vocab = 32;
    let experts = 4;
    let intermediate = 6;
    let shared = 6;
    let layers = 4;
    let mut entries = vec![
        entry(
            format!("{ROOT}embed_tokens.weight"),
            vec![vocab, hidden],
            Dtype::BF16,
            1000.0,
        ),
        entry(
            format!("{ROOT}norm.weight"),
            vec![hidden],
            Dtype::BF16,
            2000.0,
        ),
        entry(
            "lm_head.weight".to_owned(),
            vec![vocab, hidden],
            Dtype::BF16,
            3000.0,
        ),
    ];
    for layer in 0..layers {
        let prefix = format!("{ROOT}layers.{layer}");
        let base = 10_000.0 * (layer as f32 + 1.0);
        entries.push(entry(
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            Dtype::BF16,
            base,
        ));
        entries.push(entry(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
            Dtype::BF16,
            base + 100.0,
        ));
        entries.push(entry(
            format!("{prefix}.mlp.gate.weight"),
            vec![experts, hidden],
            Dtype::BF16,
            base + 200.0,
        ));
        if packed {
            entries.push(entry(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                vec![experts, intermediate * 2, hidden],
                Dtype::BF16,
                base + 300.0,
            ));
            entries.push(entry(
                format!("{prefix}.mlp.experts.down_proj"),
                vec![experts, hidden, intermediate],
                Dtype::BF16,
                base + 400.0,
            ));
        }
        if !packed || separate_too {
            for expert in 0..experts {
                let expert_base = base + 500.0 + 1000.0 * expert as f32;
                entries.push(entry(
                    format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                    vec![intermediate, hidden],
                    Dtype::BF16,
                    expert_base,
                ));
                entries.push(entry(
                    format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                    vec![intermediate, hidden],
                    Dtype::BF16,
                    expert_base + 100.0,
                ));
                entries.push(entry(
                    format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                    vec![hidden, intermediate],
                    Dtype::BF16,
                    expert_base + 200.0,
                ));
            }
        }
        entries.push(entry(
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            vec![shared, hidden],
            Dtype::BF16,
            base + 8000.0,
        ));
        entries.push(entry(
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            vec![shared, hidden],
            Dtype::BF16,
            base + 8200.0,
        ));
        entries.push(entry(
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            vec![hidden, shared],
            Dtype::BF16,
            base + 8400.0,
        ));
        entries.push(entry(
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            vec![1, hidden],
            Dtype::BF16,
            base + 8600.0,
        ));
    }
    entries
}

fn read_first(source: &crate::model::WeightSource, count: usize) -> Vec<u16> {
    let mut out = vec![0u16; count];
    source.read_bf16(0, &mut out).unwrap();
    out
}

fn expected(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

#[test]
fn packed_expert_offsets_and_shared_views_are_exact() {
    let fixture = Fixture::new("packed");
    write_config(&fixture.path, &tiny_config());
    write_shards(&fixture.path, base_entries(true, false), 2, false);
    let weights = Qwen35MoeWeights::load(&fixture.path).unwrap();
    assert_eq!(weights.root, ROOT);
    assert_eq!(weights.layers.len(), 4);
    assert_eq!(
        weights.routed_storage_kind(0).unwrap(),
        ExpertStorageKind::Packed
    );
    let layer = &weights.layers[0];
    assert_eq!(layer.attention_prefix, format!("{ROOT}layers.0."));
    assert_eq!(
        layer.routed.packed_expert_offsets(1).unwrap(),
        Some((2 * 6 * 16, 16 * 6))
    );
    assert_eq!(
        layer
            .routed
            .packed_expert_offsets(4)
            .unwrap_err()
            .to_string(),
        "Could not start backend: Oxide: expert index out of range"
    );
    let expert = layer.routed.expert(1).unwrap();
    let packed = match expert.gate_up {
        GateUpSources::Packed(source) => source,
        _ => panic!("expected packed experts"),
    };
    assert_eq!(packed.shape(), &[12, 16]);
    assert_eq!(
        read_first(&packed, 4),
        expected(&[
            10_000.0 + 300.0 + 192.0,
            10_000.0 + 300.0 + 193.0,
            10_000.0 + 300.0 + 194.0,
            10_000.0 + 300.0 + 195.0
        ])
    );
    assert_eq!(
        read_first(&expert.down, 3),
        expected(&[
            10_000.0 + 400.0 + 96.0,
            10_000.0 + 400.0 + 97.0,
            10_000.0 + 400.0 + 98.0
        ])
    );
    assert_eq!(
        read_first(&layer.shared.gate_logit, 2),
        expected(&[10_000.0 + 8600.0, 10_000.0 + 8601.0])
    );
    assert_eq!(layer.shared.down.shape(), &[16, 6]);
    assert_eq!(weights.lm_head.as_ref().unwrap().shape(), &[32, 16]);
    let plan = weights.layer_binding_plan(1).unwrap();
    assert_eq!(plan.index, 1);
    assert_eq!(plan.attention_prefix, format!("{ROOT}layers.1."));
    assert_eq!(plan.router, format!("{ROOT}layers.1.mlp.gate.weight"));
    assert_eq!(
        plan.packed_gate_up.as_deref(),
        Some(format!("{ROOT}layers.1.mlp.experts.gate_up_proj").as_str())
    );
    assert_eq!(
        plan.packed_down.as_deref(),
        Some(format!("{ROOT}layers.1.mlp.experts.down_proj").as_str())
    );
    assert!(plan.separate_experts.is_empty());
    assert_eq!(
        plan.shared_gate_logit,
        format!("{ROOT}layers.1.mlp.shared_expert_gate.weight")
    );
    assert_eq!(plan.dtype, Dtype::BF16);
    assert!(weights.layer_binding_plan(4).is_err());
}

#[test]
fn packed_tensors_win_over_separate_aliases() {
    let fixture = Fixture::new("alias-precedence");
    write_config(&fixture.path, &tiny_config());
    write_shards(&fixture.path, base_entries(true, true), 1, false);
    let weights = Qwen35MoeWeights::load(&fixture.path).unwrap();
    for layer in &weights.layers {
        assert_eq!(layer.routed.storage_kind(), ExpertStorageKind::Packed);
        match layer.routed.expert(1).unwrap().gate_up {
            GateUpSources::Packed(source) => assert_eq!(source.shape(), &[12, 16]),
            _ => panic!("packed alias must win"),
        }
    }
}

#[test]
fn separate_fallback_is_all_or_nothing_and_keeps_expert_order() {
    let fixture = Fixture::new("separate");
    write_config(&fixture.path, &tiny_config());
    write_shards(&fixture.path, base_entries(false, false), 2, false);
    let weights = Qwen35MoeWeights::load(&fixture.path).unwrap();
    assert_eq!(
        weights.routed_storage_kind(0).unwrap(),
        ExpertStorageKind::Separate
    );
    let expert = weights.layers[0].routed.expert(1).unwrap();
    match expert.gate_up {
        GateUpSources::Separate { gate, up } => {
            assert_eq!(read_first(&gate, 1), expected(&[10_000.0 + 500.0 + 1000.0]));
            assert_eq!(
                read_first(&up, 1),
                expected(&[10_000.0 + 500.0 + 1000.0 + 100.0])
            );
        }
        _ => panic!("expected separate experts"),
    }
    assert!(weights.layers[0]
        .routed
        .packed_expert_offsets(0)
        .unwrap()
        .is_none());
    let plan = weights.layer_binding_plan(0).unwrap();
    assert!(plan.packed_gate_up.is_none());
    assert_eq!(plan.separate_experts.len(), 4);
    assert_eq!(
        plan.separate_experts[3].up,
        format!("{ROOT}layers.0.mlp.experts.3.up_proj.weight")
    );

    let fixture = Fixture::new("separate-missing");
    write_config(&fixture.path, &tiny_config());
    let mut entries = base_entries(false, false);
    entries.retain(|item| !item.name.ends_with("experts.3.up_proj.weight"));
    write_shards(&fixture.path, entries, 1, false);
    assert!(Qwen35MoeWeights::load(&fixture.path).is_err());
}

#[test]
fn pytorch_archives_are_explicitly_rejected() {
    let fixture = Fixture::new("pytorch-reject");
    write_config(&fixture.path, &tiny_config());
    write_shards(&fixture.path, base_entries(true, false), 1, false);
    fs::write(fixture.path.join("pytorch_model.bin"), b"not admitted").unwrap();
    assert!(Qwen35MoeWeights::load(&fixture.path).is_err());
}

#[test]
fn wrong_shapes_dtypes_and_missing_shared_gate_are_rejected() {
    fn wrong_router_shape(entries: &mut Vec<Entry>) {
        entries.retain(|item| !item.name.ends_with("layers.0.mlp.gate.weight"));
        entries.push(entry(
            format!("{ROOT}layers.0.mlp.gate.weight"),
            vec![3, 16],
            Dtype::BF16,
            0.0,
        ));
    }
    fn wrong_packed_shape(entries: &mut Vec<Entry>) {
        entries.retain(|item| !item.name.ends_with("layers.0.mlp.experts.gate_up_proj"));
        entries.push(entry(
            format!("{ROOT}layers.0.mlp.experts.gate_up_proj"),
            vec![4, 12, 15],
            Dtype::BF16,
            0.0,
        ));
    }
    fn wrong_router_dtype(entries: &mut Vec<Entry>) {
        entries.retain(|item| !item.name.ends_with("layers.0.mlp.gate.weight"));
        entries.push(entry(
            format!("{ROOT}layers.0.mlp.gate.weight"),
            vec![4, 16],
            Dtype::F32,
            0.0,
        ));
    }
    fn missing_shared_gate(entries: &mut Vec<Entry>) {
        entries.retain(|item| !item.name.ends_with("shared_expert_gate.weight"));
    }
    fn packed_without_down(entries: &mut Vec<Entry>) {
        entries.retain(|item| !item.name.ends_with("layers.0.mlp.experts.down_proj"));
        // Add complete separate aliases so a silent fallback would succeed and
        // the test would catch it; the packed path must remain a hard error.
        for expert in 0..4 {
            let base = 10_000.0 + 5_000.0 + 1_000.0 * expert as f32;
            entries.push(entry(
                format!("{ROOT}layers.0.mlp.experts.{expert}.gate_proj.weight"),
                vec![6, 16],
                Dtype::BF16,
                base,
            ));
            entries.push(entry(
                format!("{ROOT}layers.0.mlp.experts.{expert}.up_proj.weight"),
                vec![6, 16],
                Dtype::BF16,
                base + 100.0,
            ));
            entries.push(entry(
                format!("{ROOT}layers.0.mlp.experts.{expert}.down_proj.weight"),
                vec![16, 6],
                Dtype::BF16,
                base + 200.0,
            ));
        }
    }
    type Mutation = fn(&mut Vec<Entry>);
    let cases: [(&str, Mutation); 5] = [
        ("router-shape", wrong_router_shape),
        ("packed-shape", wrong_packed_shape),
        ("router-dtype", wrong_router_dtype),
        ("missing-shared-gate", missing_shared_gate),
        ("packed-without-down", packed_without_down),
    ];
    for (label, mutate) in cases {
        let fixture = Fixture::new(label);
        write_config(&fixture.path, &tiny_config());
        let mut entries = base_entries(true, false);
        mutate(&mut entries);
        write_shards(&fixture.path, entries, 1, false);
        assert!(
            Qwen35MoeWeights::load(&fixture.path).is_err(),
            "{label} was admitted"
        );
    }
}

#[test]
fn roots_aliases_and_mtp_tensors_are_scoped_exactly() {
    let fixture = Fixture::new("duplicate-alias");
    write_config(&fixture.path, &tiny_config());
    write_shards(&fixture.path, base_entries(true, false), 2, true);
    assert!(Checkpoint::open(&fixture.path).is_err());
    assert!(Qwen35MoeWeights::load(&fixture.path).is_err());

    let fixture = Fixture::new("mixed-root");
    write_config(&fixture.path, &tiny_config());
    let mut entries = base_entries(true, false);
    entries.push(entry(
        "model.layers.0.input_layernorm.weight".to_owned(),
        vec![16],
        Dtype::BF16,
        0.0,
    ));
    write_shards(&fixture.path, entries, 1, false);
    assert!(Qwen35MoeWeights::load(&fixture.path).is_err());

    let fixture = Fixture::new("mtp-excluded");
    write_config(&fixture.path, &tiny_config());
    let mut entries = base_entries(true, false);
    entries.push(entry(
        "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_owned(),
        vec![6, 16],
        Dtype::BF16,
        0.0,
    ));
    entries.push(entry(
        "model.visual.blocks.0.attn.qkv.weight".to_owned(),
        vec![4, 16],
        Dtype::BF16,
        0.0,
    ));
    write_shards(&fixture.path, entries, 1, false);
    let weights = Qwen35MoeWeights::load(&fixture.path).unwrap();
    assert_eq!(weights.layers.len(), 4);
    assert_eq!(weights.root, ROOT);
}

#[test]
fn tiny_schedule_router_and_workspace_are_consistent() {
    let config = Qwen35MoeConfig::from_json(&tiny_config()).unwrap();
    assert_eq!(config.dtype, Qwen35MoeDtype::Bf16);
    assert_eq!(config.text.full_attention_layers(), 1);
    assert_eq!(config.text.linear_attention_layers(), 3);
    assert!(!config.admission().executor_implemented);
    let schedule = crate::qwen35_moe_schedule::Qwen35MoeSchedule::new(&config).unwrap();
    assert_eq!(schedule.full_attention_layers(), 1);
    assert_eq!(
        schedule.layer(2).unwrap().attention,
        crate::qwen35_config::Qwen35LayerType::Linear
    );
    assert_eq!(
        schedule.layer(3).unwrap().attention,
        crate::qwen35_config::Qwen35LayerType::Full
    );
    let mut native = tiny_config();
    native["text_config"]["num_experts"] = json!(256);
    native["text_config"]["num_experts_per_tok"] = json!(8);
    native["text_config"]["moe_intermediate_size"] = json!(8);
    native["text_config"]["shared_expert_intermediate_size"] = json!(8);
    let native = Qwen35MoeConfig::from_json(&native).unwrap();
    let layout = Qwen35MoeNativeScratchLayout::from_config(&native, 2).unwrap();
    assert_eq!(layout.slots, 16);
    assert_eq!(layout.expert_output_f32.element_bytes(), 4);
    assert_eq!(layout.total_bytes(), 23_808);
    assert_eq!(
        native_workspace_bytes(2, 16, 8, 256).unwrap(),
        layout.total_bytes()
    );
    assert!(Qwen35MoeNativeScratchLayout::from_config(&native, 0).is_err());
    assert_eq!(
        Qwen35MoeConfig::from_json(&tiny_config())
            .unwrap()
            .native_contract(true),
        crate::qwen35_moe_contract::Qwen35MoeNativeContract::DecoderMoeForwardFallback
    );
    assert_eq!(
        native.native_contract(true),
        crate::qwen35_moe_contract::Qwen35MoeNativeContract::RoutedMoeCuda
    );

    let plan = RouterPlan::new(&config.moe, 16).unwrap();
    let route = plan.route(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    assert_eq!(route.ids(), &[0, 1]);
    assert!((route.weights().iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let raw = RouterPlan::new(
        &Qwen35MoeSpec {
            norm_topk_prob: false,
            ..config.moe.clone()
        },
        16,
    )
    .unwrap();
    assert!(raw.route(&[1.0, 1.0, 1.0, 1.0]).unwrap().weights()[0] < 1.0);
    assert_eq!(
        cuda_shared_gate_bf16(half::bf16::from_f32(0.0).to_bits()),
        half::bf16::from_f32(0.5).to_bits()
    );
    assert_eq!(
        cuda_silu_mul_bf16(
            half::bf16::from_f32(0.0).to_bits(),
            half::bf16::from_f32(3.0).to_bits()
        ),
        half::bf16::from_f32(0.0).to_bits()
    );
}

#[test]
fn overflow_and_bad_router_specs_fail_cold() {
    let mut huge = tiny_config();
    huge["text_config"]["hidden_size"] = json!(i32::MAX as i64);
    huge["text_config"]["num_experts"] = json!(4);
    assert!(Qwen35MoeConfig::from_json(&huge).is_err());

    let mut bad_top = tiny_config();
    bad_top["text_config"]["num_experts_per_tok"] = json!(5);
    assert!(Qwen35MoeConfig::from_json(&bad_top).is_err());

    let valid = Qwen35MoeConfig::from_json(&tiny_config()).unwrap();
    let bad_plan = Qwen35MoeSpec {
        num_experts_per_tok: 5,
        ..valid.moe.clone()
    };
    assert!(RouterPlan::new(&bad_plan, 16).is_err());
    let zero_plan = Qwen35MoeSpec {
        num_experts_per_tok: 0,
        ..valid.moe.clone()
    };
    assert!(RouterPlan::new(&zero_plan, 16).is_err());
}

#[test]
fn cached_35b_headers_bind_packed_layers_without_payload_reads() {
    let path =
        Path::new("/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b");
    if !path.join("config.json").is_file() {
        if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
            panic!("cached 35B checkpoint metadata required but absent");
        }
        eprintln!("cached 35B checkpoint metadata absent; skipping header binding");
        return;
    }
    let started = std::time::Instant::now();
    let weights = Qwen35MoeWeights::load(path).unwrap();
    eprintln!(
        "bound {} packed layers from cached metadata headers in {:?}",
        weights.layers.len(),
        started.elapsed()
    );
    assert_eq!(weights.root, ROOT);
    assert_eq!(weights.layers.len(), 40);
    assert_eq!(weights.config.moe.num_experts, 256);
    assert_eq!(weights.config.moe.num_experts_per_tok, 8);
    assert_eq!(weights.config.text.full_attention_layers(), 10);
    assert_eq!(weights.config.text.linear_attention_layers(), 30);
    assert_eq!(weights.config.vision_depth, 27);
    assert_eq!(weights.config.mtp_num_hidden_layers, 1);
    for layer in &weights.layers {
        assert_eq!(layer.routed.storage_kind(), ExpertStorageKind::Packed);
        assert_eq!(layer.shapes.router, [256, 2048]);
        assert_eq!(layer.shapes.routed_gate_up, [256, 1024, 2048]);
        assert_eq!(layer.shapes.routed_down, [256, 2048, 512]);
        assert_eq!(layer.shapes.shared_gate_up, [1024, 2048]);
        assert_eq!(layer.shapes.shared_down, [2048, 512]);
        assert_eq!(layer.shapes.shared_gate, [1, 2048]);
    }
    assert!(
        weights
            .schedule()
            .unwrap()
            .routed_payload_bytes(&weights.config)
            .unwrap()
            > 0
    );
    let admission = weights.config.admission();
    assert!(admission.config_admitted);
    assert!(!admission.executor_implemented);
    assert!(!admission.whole_checkpoint_implemented);
    assert!(admission
        .blockers
        .iter()
        .any(|blocker| blocker.contains("vision")));
    assert!(admission
        .blockers
        .iter()
        .any(|blocker| blocker.contains("MTP")));
}

// ---------------------------------------------------------------------------
// v3 shared backbone / typed FFN / provider tests
// ---------------------------------------------------------------------------

#[test]
fn moe_shared_backbone_binds_attention_without_dense_mlp_and_exact_cases() {
    use crate::qwen35_backbone::{Qwen35BackboneKind, Qwen35BackboneWeights};
    use crate::qwen35_config::Qwen35LayerType;
    use crate::qwen35_model::Qwen35AttentionWeights;
    use crate::qwen35_moe_cases::{CASES, LONG_TOKENS, SHORT_TOKENS};
    use crate::qwen35_moe_device::validate_ffn_host;

    assert_eq!(SHORT_TOKENS, [1, 7, 128]);
    assert_eq!(LONG_TOKENS, 2048);
    assert_eq!(CASES.len(), 5);

    let path = std::path::Path::new(
        "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b",
    );
    if !path.join("config.json").is_file() {
        if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
            panic!("cached 35B checkpoint required but absent");
        }
        return;
    }
    let backbone = Qwen35BackboneWeights::load_moe(path).unwrap();
    assert_eq!(backbone.kind, Qwen35BackboneKind::Moe);
    assert_eq!(backbone.layers.len(), 40);
    assert_eq!(backbone.geometry.layer_types.len(), 40);
    let mut full = 0;
    let mut linear = 0;
    for (layer, kind) in backbone.layers.iter().zip(&backbone.geometry.layer_types) {
        match (kind, &layer.attention) {
            (Qwen35LayerType::Full, Qwen35AttentionWeights::Full { .. }) => full += 1,
            (Qwen35LayerType::Linear, Qwen35AttentionWeights::Linear { .. }) => linear += 1,
            _ => panic!("shared backbone attention kind mismatch"),
        }
        validate_ffn_host(&layer.ffn).unwrap();
    }
    assert_eq!((full, linear), (10, 30));
}

#[test]
fn dense_shared_backbone_reuses_donor_loader_without_moe_ffn() {
    use crate::qwen35_backbone::{Qwen35BackboneFfn, Qwen35BackboneKind, Qwen35BackboneWeights};
    let Some(path) = std::env::var_os("QWEN35_DENSE_BACKBONE_CHECKPOINT") else {
        if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
            panic!("QWEN35_DENSE_BACKBONE_CHECKPOINT required");
        }
        return;
    };
    let backbone = Qwen35BackboneWeights::load_dense(std::path::Path::new(&path)).unwrap();
    assert_eq!(backbone.kind, Qwen35BackboneKind::Dense);
    assert_eq!(backbone.layers.len(), backbone.geometry.num_hidden_layers);
    for layer in &backbone.layers {
        assert!(matches!(layer.ffn, Qwen35BackboneFfn::Dense { .. }));
    }
    backbone.validate().unwrap();
}

#[test]
fn routed_weight_lengths_include_the_two_i_gate_up_factor() {
    use crate::qwen35_moe_enqueue::routed_weight_lengths;
    assert_eq!(
        routed_weight_lengths(2048, 512).unwrap(),
        (256 * 2048, 256 * 1024 * 2048, 256 * 2048 * 512)
    );
    assert!(routed_weight_lengths(usize::MAX, 512).is_err());
}
