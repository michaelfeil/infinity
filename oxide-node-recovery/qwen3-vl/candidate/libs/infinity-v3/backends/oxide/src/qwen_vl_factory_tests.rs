// CPU acceptance for the Qwen3-VL composition/factory layer.
//
// These tests do not create a CUDA context. They cover alias selection, both
// Native rope spellings, packed-batch plumbing, text-only mRoPE positions and
// the trained ColQwen3 Dense token projection.

use crate::qwen3_vl_batch::pack_batch;
use crate::qwen3_vl_config::{Qwen3VlConfig, ReadoutKind, COLQWEN3_TOKEN_WIDTH};
use crate::qwen3_vl_readout::Qwen3VlReadout;
use crate::qwen3_vl_vision::ImageGrid;
use infinity_v3_backend_core::{Batch, ImagePatches, MultimodalEncoding, Pool};
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const BASE_CONFIG: &str = r#"{
  "architectures": ["Qwen3VLForConditionalGeneration"],
  "dtype": "bfloat16",
  "image_token_id": 151655,
  "model_type": "qwen3_vl",
  "text_config": {
    "attention_bias": false,
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24,20,20], "rope_type": "default"},
    "rope_theta": 5000000,
    "use_cache": true,
    "vocab_size": 151936
  },
  "tie_word_embeddings": false,
  "video_token_id": 151656,
  "vision_config": {
    "deepstack_visual_indexes": [8,16,24],
    "depth": 27,
    "dtype": "bfloat16",
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "intermediate_size": 4304,
    "model_type": "qwen3_vl",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652
}"#;

fn config_with(update: impl FnOnce(&mut serde_json::Value)) -> Qwen3VlConfig {
    let mut value: serde_json::Value = serde_json::from_str(BASE_CONFIG).unwrap();
    update(&mut value);
    Qwen3VlConfig::from_value(value).expect("config")
}

fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "infinity-qwen-vl-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn native_aliases_select_the_same_tower_and_real_readout() {
    let plain = config_with(|_| {});
    assert_eq!(plain.alias.model_type(), "qwen3_vl");
    assert_eq!(plain.readout.kind, ReadoutKind::Embedding);
    assert!(!plain.readout.multi_vector);

    let colqwen = config_with(|value| value["model_type"] = json!("colqwen3"));
    assert_eq!(colqwen.readout.kind, ReadoutKind::ColQwen3);
    assert!(colqwen.readout.multi_vector);
    assert_eq!(
        colqwen.readout.token_projection_width,
        Some(COLQWEN3_TOKEN_WIDTH)
    );
    assert!(colqwen.readout.transformers_math);
    assert!(colqwen.readout.normalize);
    assert_eq!(colqwen.vision.hidden_size, plain.vision.hidden_size);

    let nemotron = config_with(|value| value["model_type"] = json!("qwen3_vl_nemotron_embed"));
    assert_eq!(nemotron.readout.kind, ReadoutKind::NemotronColEmbed);
    assert!(nemotron.readout.multi_vector);
    assert!(nemotron.readout.skip_final_norm);
    assert!(nemotron.readout.normalize);
    assert_eq!(nemotron.vision.hidden_size, plain.vision.hidden_size);
}

#[test]
fn qwen35_and_unrelated_model_types_are_not_admitted_by_the_vl_adapter() {
    for model_type in [
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3",
        "mistral3",
        "qwen3_vl_text",
    ] {
        let mut value: serde_json::Value = serde_json::from_str(BASE_CONFIG).unwrap();
        value["model_type"] = json!(model_type);
        assert!(
            Qwen3VlConfig::from_value(value).is_err(),
            "{model_type} must not route through the Qwen3-VL adapter"
        );
    }
}

#[test]
fn rope_parameters_spelling_preserves_native_mrope_and_theta() {
    let config = config_with(|value| {
        let text = value["text_config"].as_object_mut().unwrap();
        let rope = text.remove("rope_scaling").unwrap();
        text.remove("rope_theta");
        let mut rope = rope;
        rope["rope_theta"] = json!(424242);
        text.insert("rope_parameters".into(), rope);
    });
    assert_eq!(config.text.rope_theta, 424242.0);
    assert_eq!(config.text.mrope_section, [24, 20, 20]);
    assert_eq!(config.text.mrope_section_sum().unwrap(), 64);
}

#[test]
fn packed_batch_offsets_images_and_concatenates_media_positions() {
    let image = Arc::new(ImagePatches {
        pixels: vec![0f32; 8 * 4],
        grid_thw: [1, 2, 4],
        patch_dim: 4,
        merge_size: 2,
    });
    let media = MultimodalEncoding {
        images: vec![(1, image)],
        audios: vec![],
        position_ids: [vec![0, 0, 1, 2], vec![0, 0, 0, 1], vec![0, 0, 0, 1]],
        memory: None,
    };
    let batch = Batch {
        multimodal: vec![None, Some(Arc::new(media))],
        input_ids: vec![1, 9, 9, 2, 1, 9, 9, 3],
        token_type_ids: vec![0; 8],
        position_ids: vec![0, 1, 2, 3, 0, 1, 2, 3],
        cumulative_seq_lengths: vec![0, 4, 8],
        max_length: 4,
        pooled_indices: vec![0, 1],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let request = pack_batch(&batch, Pool::LastToken, false).expect("packed batch");
    assert_eq!(request.sequences, 2);
    assert_eq!(request.tokens, 8);
    assert!(request.has_media);
    let forward = &request.forward;
    assert_eq!(forward.images.len(), 1);
    assert_eq!(forward.images[0].token_start, 5);
    assert_eq!(forward.images[0].grid_thw, [1, 2, 4]);
    let positions = forward.media_positions.as_ref().expect("positions");
    assert_eq!(positions[0], [0, 1, 2, 3, 0, 0, 1, 2]);
    assert_eq!(positions[1], [0, 1, 2, 3, 0, 0, 0, 1]);
    assert_eq!(positions[2], [0, 1, 2, 3, 0, 0, 0, 1]);
}

#[test]
fn text_only_explicit_media_positions_are_preserved() {
    let media = MultimodalEncoding {
        images: vec![],
        audios: vec![],
        position_ids: [vec![5, 6, 7], vec![2, 2, 2], vec![9, 9, 9]],
        memory: None,
    };
    let batch = Batch {
        multimodal: vec![Some(Arc::new(media))],
        input_ids: vec![1, 2, 3],
        token_type_ids: vec![0; 3],
        position_ids: vec![0, 1, 2],
        cumulative_seq_lengths: vec![0, 3],
        max_length: 3,
        pooled_indices: vec![0],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let request = pack_batch(&batch, Pool::LastToken, false).expect("text-only media");
    assert!(request.forward.images.is_empty());
    let positions = request.forward.media_positions.as_ref().expect("positions");
    assert_eq!(positions[0], [5, 6, 7]);
    assert_eq!(positions[1], [2, 2, 2]);
    assert_eq!(positions[2], [9, 9, 9]);
}

#[test]
fn multivector_profile_uses_exact_merger_gelu() {
    let config = config_with(|value| value["model_type"] = json!("colqwen3"));
    let contract = crate::qwen3_vl_contract::select(&config, false).expect("contract");
    assert_eq!(
        contract.vision.profile,
        crate::qwen3_vl_contract::VisionProfile::MultiVector
    );
    assert_eq!(
        contract.vision.merger_activation,
        crate::qwen3_vl_contract::VisionActivation::GeluExact
    );
    assert_eq!(
        contract.vision.corner_sum,
        crate::qwen3_vl_contract::CornerSumOrder::MultiVectorLeft
    );
}

#[test]
fn colqwen3_dense_projection_is_loaded_from_modules_and_normalized() {
    let root = temp_dir("colqwen3");
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "colqwen3",
            "architectures": ["ColQwen3"],
            "dtype": "bfloat16"
        }))
        .unwrap(),
    )
    .unwrap();
    // A real ColQwen3 root always has the VLM shards; the projection here is
    // intentionally only in the Dense module so the fallback is exercised.
    let dummy = [0f32].map(|value| value.to_le_bytes()).concat();
    fs::write(
        root.join("model.safetensors"),
        serialize(
            vec![(
                "dummy",
                TensorView::new(Dtype::F32, vec![1], &dummy).unwrap(),
            )],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("modules.json"),
        serde_json::to_vec(&json!([
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Dense","type":"sentence_transformers.models.Dense"}
        ]))
        .unwrap(),
    )
    .unwrap();
    let dense = root.join("1_Dense");
    fs::create_dir_all(&dense).unwrap();
    fs::write(
        dense.join("config.json"),
        serde_json::to_vec(&json!({
            "in_features": 4,
            "out_features": 2,
            "bias": false,
            "activation_function": "torch.nn.modules.linear.Identity"
        }))
        .unwrap(),
    )
    .unwrap();
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let weight_bytes: Vec<u8> = weight
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::F32, vec![2, 4], &weight_bytes).unwrap();
    fs::write(
        dense.join("model.safetensors"),
        serialize(vec![("linear.weight", view)], None).unwrap(),
    )
    .unwrap();

    let config = crate::qwen3_vl_config::ReadoutConfig {
        kind: ReadoutKind::ColQwen3,
        multi_vector: true,
        skip_final_norm: false,
        transformers_math: true,
        token_projection_width: Some(2),
        normalize: true,
        norm_eps: 1.0 / 128.0,
    };
    let readout = Qwen3VlReadout::load(&root, &config, 4).expect("readout");
    assert_eq!(readout.output_width(), 2);
    let rows = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0];
    let output = readout.apply_rows(&rows, 2).expect("projected");
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!(output[1].abs() < 1e-6);
    assert!(output[2].abs() < 1e-6);
    assert!((output[3] - 1.0).abs() < 1e-6);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn image_grid_accepts_video_frames_without_changing_merge_geometry() {
    let grid = ImageGrid::new(3, 4, 8, 2, 1, 24).expect("three-frame video");
    assert_eq!(grid.patch_rows().unwrap(), 96);
    assert_eq!(grid.token_count, 24);
}

#[test]
fn backend_type_is_send_for_the_engine_factory_boundary() {
    fn assert_send<T: Send>() {}
    assert_send::<crate::Qwen3VlBackend>();
}

#[test]
fn rerank_readout_is_explicit_and_never_auto_selected() {
    let config = config_with(|_| {});
    assert_ne!(config.readout.kind, ReadoutKind::Rerank);
    let rerank = crate::qwen3_vl_config::ReadoutConfig::rerank();
    assert_eq!(rerank.kind, ReadoutKind::Rerank);
    assert!(!rerank.multi_vector);
}

#[test]
fn nemotron_rows_are_normalized_without_a_projection() {
    let root = temp_dir("nemotron");
    let config = crate::qwen3_vl_config::ReadoutConfig {
        kind: ReadoutKind::NemotronColEmbed,
        multi_vector: true,
        skip_final_norm: true,
        transformers_math: true,
        token_projection_width: None,
        normalize: true,
        norm_eps: 1.0 / 128.0,
    };
    let readout = Qwen3VlReadout::load(&root, &config, 2).expect("nemotron readout");
    let output = readout
        .apply_rows(&[3.0, 4.0, 0.0, 5.0], 2)
        .expect("normalized rows");
    assert!((output[0] - 0.6).abs() < 1e-6);
    assert!((output[1] - 0.8).abs() < 1e-6);
    assert!(output[2].abs() < 1e-6);
    assert!((output[3] - 1.0).abs() < 1e-6);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn explicit_rerank_head_loads_only_from_a_real_score_weight() {
    let root = temp_dir("rerank");
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({"model_type": "qwen3_vl", "dtype": "bfloat16"})).unwrap(),
    )
    .unwrap();
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let weight_bytes: Vec<u8> = weight
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let bias: Vec<f32> = vec![0.5];
    let bias_bytes: Vec<u8> = bias.iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(
        root.join("model.safetensors"),
        serialize(
            vec![
                (
                    "score.weight",
                    TensorView::new(Dtype::F32, vec![1, 4], &weight_bytes).unwrap(),
                ),
                (
                    "score.bias",
                    TensorView::new(Dtype::F32, vec![1], &bias_bytes).unwrap(),
                ),
            ],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let readout = Qwen3VlReadout::load(&root, &crate::qwen3_vl_config::ReadoutConfig::rerank(), 4)
        .expect("rerank readout");
    let scores = readout.apply_scores(&[2.0, 9.0, 9.0, 9.0], 1).unwrap();
    assert_eq!(scores, vec![2.5]);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn synthetic_fixture_pixels_and_positions_flow_into_the_engine_batch() {
    let root = temp_dir("fixture");
    let pixels: Vec<f32> = (0..8 * 4).map(|index| index as f32).collect();
    let bytes: Vec<u8> = pixels
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(root.join("img.f32"), bytes).unwrap();
    let manifest = json!({
        "fixtures": [{
            "name": "synthetic",
            "ids": [1, 9, 9, 2],
            "types": [0, 0, 0, 0],
            "positions": [0, 1, 2, 3],
            "cumulative": [0, 4],
            "media_positions": [[0, 0, 1, 2], [0, 0, 0, 1], [0, 0, 0, 1]],
            "images": [{
                "grid_thw": [1, 2, 4],
                "merge_size": 2,
                "patch_dim": 4,
                "pixels": "img.f32",
                "token_start": 1,
                "token_count": 2
            }],
            "pool": "last_token"
        }]
    });
    let path = root.join("fixtures.json");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let file = crate::qwen3_vl_fixtures::read_fixture_file(&path).unwrap();
    let batch = crate::qwen3_vl_fixtures::fixture_batch(&file.fixtures[0], &root, false).unwrap();
    assert_eq!(batch.input_ids, vec![1, 9, 9, 2]);
    assert_eq!(batch.multimodal.len(), 1);
    let media = batch.multimodal[0].as_ref().unwrap();
    assert_eq!(media.images.len(), 1);
    assert_eq!(media.images[0].0, 1);
    assert_eq!(media.images[0].1.token_count(), 2);
    assert_eq!(media.images[0].1.pixels, pixels);
    assert_eq!(media.position_ids[0], vec![0, 0, 1, 2]);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn delivered_real_fixture_manifest_parses_when_present() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../native-baseline/fixtures/fixtures.json");
    if !root.is_file() {
        return;
    }
    let file = crate::qwen3_vl_fixtures::read_fixture_file(&root).unwrap();
    assert_eq!(file.fixtures.len(), 8);
    let names: Vec<&str> = file
        .fixtures
        .iter()
        .map(|fixture| fixture.name.as_str())
        .collect();
    assert!(names.contains(&"three-frame-cats"));
    assert!(names.contains(&"ragged-three-seq"));
    assert!(names.contains(&"empty-images"));
}

#[test]
fn all_real_fixtures_pack_into_engine_batches() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../native-baseline/fixtures");
    let manifest = root.join("fixtures.json");
    if !manifest.is_file() {
        return;
    }
    let file = crate::qwen3_vl_fixtures::read_fixture_file(&manifest).unwrap();
    for fixture in &file.fixtures {
        let batch = crate::qwen3_vl_fixtures::fixture_batch(fixture, &root, false)
            .unwrap_or_else(|error| panic!("{} fixture: {error}", fixture.name));
        let request = crate::qwen3_vl_batch::pack_batch(
            &batch,
            crate::qwen3_vl_fixtures::pool_from_name(fixture.pool.as_deref()).unwrap(),
            false,
        )
        .unwrap_or_else(|error| panic!("{} pack: {error}", fixture.name));
        assert_eq!(request.tokens, fixture.ids.len());
        assert_eq!(request.sequences, fixture.cumulative.len() - 1);
        let image_rows: usize = request
            .forward
            .images
            .iter()
            .map(|image| image.merged_rows().unwrap())
            .sum();
        let expected_rows: usize = fixture.images.iter().map(|image| image.token_count).sum();
        assert_eq!(image_rows, expected_rows, "{}", fixture.name);
    }
}

#[test]
fn actual_colqwen3_checkpoint_binds_vlm_prefix_and_embedding_projection() {
    let root = PathBuf::from(
        "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b",
    );
    if !root.join("config.json").is_file() {
        return;
    }
    let config = Qwen3VlConfig::from_path(&root).expect("actual ColQwen3 config");
    assert_eq!(config.alias.model_type(), "colqwen3");
    assert_eq!(config.readout.kind, ReadoutKind::ColQwen3);
    assert_eq!(config.vision.dtype, crate::qwen3_vl_config::Dtype::Bf16);
    assert_eq!(config.text.dtype, crate::qwen3_vl_config::Dtype::Bf16);
    assert_eq!(config.text.hidden_size, 2560);
    assert_eq!(config.vision.hidden_size, 1024);
    assert_eq!(config.vision.depth, 24);
    assert_eq!(config.vision.deepstack_visual_indexes, vec![5, 11, 17]);
    assert_eq!(config.readout.token_projection_width, Some(320));
    assert_eq!(config.readout.norm_eps, 1.0 / 128.0);

    let readout = Qwen3VlReadout::load(&root, &config.readout, config.text.hidden_size)
        .expect("actual embedding_proj_layer");
    assert_eq!(readout.output_width(), 320);

    let metadata =
        crate::checkpoint::CheckpointMetadata::open(&root).expect("actual ColQwen3 headers");
    let report = crate::qwen3_vl_inventory::validate_checkpoint(
        &config,
        &metadata,
        crate::qwen3_vl_inventory::InventoryScope::Full,
    )
    .expect("actual ColQwen3 inventory");
    assert_eq!(report.bound.len(), 715);
    assert_eq!(report.readout_count, 2);
}

#[test]
fn geometry_report_covers_cached_aliases_and_reports_missing_fa4_only() {
    let report = crate::qwen3_vl_geometry::report();
    let tuples = report["tuples"].as_array().unwrap();
    let by_name = |name: &str| {
        tuples
            .iter()
            .find(|tuple| tuple["name"] == name)
            .unwrap_or_else(|| panic!("missing tuple {name}"))
    };
    for name in [
        "Qwen3-VL-8B-Embedding",
        "TomoroAI-ColQwen3-4B",
        "Nemotron-ColEmbed-VL-4B-v2",
        "Nemotron-ColEmbed-VL-8B-v2",
    ] {
        let support = &by_name(name)["support"];
        assert_eq!(support["status"], "ready", "{name}: {support}");
        assert_eq!(support["structural"], true);
        assert_eq!(support["dense_decoder"], true);
        assert_eq!(support["vision_fa4"], true);
        assert_eq!(support["text_fa4"], true);
    }
    assert_eq!(by_name("Qwen3-VL-2B")["support"]["status"], "ready");
    // Q64/KV8 d128 reuses the existing ratio-8 profiles (mode 22 causal,
    // mode 5 noncausal); absolute head counts are dynamic.
    assert_eq!(by_name("Qwen3-VL-32B")["support"]["status"], "ready");
    assert_eq!(
        by_name("Qwen3-VL-30B-A3B")["support"]["status"],
        "missing_decoder"
    );
    let missing = report["central_fa4_owner_packet"]["profiles"]
        .as_array()
        .unwrap();
    assert!(missing.iter().any(|item| {
        item["missing"]["kind"] == "text"
            && item["missing"]["heads"] == 40
            && item["missing"]["kv_heads"] == 8
            && item["missing"]["head_dim"] == 128
            && item["missing"]["causal"] == true
    }));
    assert!(missing.iter().any(|item| {
        item["missing"]["kind"] == "text"
            && item["missing"]["heads"] == 40
            && item["missing"]["kv_heads"] == 8
            && item["missing"]["causal"] == false
    }));
}

#[test]
fn actual_nemotron_colembed_configs_bind_skip_final_norm_without_projection() {
    for root in [
        PathBuf::from("/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2"),
        PathBuf::from("/tmp/perplexity-embedding-audit/multivector-support/nvidia--nemotron-colembed-vl-8b-v2"),
    ] {
        if !root.join("config.json").is_file() {
            continue;
        }
        let config = Qwen3VlConfig::from_path(&root).expect("actual Nemotron config");
        assert_eq!(config.alias.model_type(), "qwen3_vl_nemotron_embed");
        assert_eq!(config.readout.kind, ReadoutKind::NemotronColEmbed);
        assert!(config.readout.skip_final_norm);
        assert_eq!(config.readout.norm_eps, 1.0 / 128.0);
        assert_eq!(config.text.dtype, crate::qwen3_vl_config::Dtype::Bf16);
        assert_eq!(config.vision.dtype, crate::qwen3_vl_config::Dtype::Bf16);
        let readout = Qwen3VlReadout::load(&root, &config.readout, config.text.hidden_size)
            .expect("Nemotron no-projection readout");
        assert_eq!(readout.output_width(), config.text.hidden_size);
        let metadata =
            crate::checkpoint::CheckpointMetadata::open(&root).expect("actual Nemotron headers");
        let report = crate::qwen3_vl_inventory::validate_checkpoint(
            &config,
            &metadata,
            crate::qwen3_vl_inventory::InventoryScope::Full,
        )
        .expect("actual Nemotron inventory");
        assert_eq!(report.readout_count, 0);
    }
}

#[test]
fn fa4_selection_aliases_runtime_heads_by_ratio_and_preserves_mask_family() {
    use crate::fa4::{AttentionSpec, DType, Fa4, Scale, Window};
    let spec = |heads, kv, dim, causal, window| AttentionSpec {
        heads,
        kv_heads: kv,
        head_dim: dim,
        causal,
        window,
        dtype: DType::Bf16,
        scale: Scale::RsqrtHeadDim,
    };
    // Absolute heads are dynamic: BERT's 16/16 d64 global kernel serves 12/12
    // and 24/24; Qwen's ratio-8 d128 kernels serve Q64/KV8.
    assert_eq!(
        Fa4::profile_name(&spec(12, 12, 64, false, Window::Global)),
        Some("bert")
    );
    assert_eq!(
        Fa4::profile_name(&spec(24, 24, 64, false, Window::Global)),
        Some("bert")
    );
    assert_eq!(
        Fa4::profile_name(&spec(64, 8, 128, true, Window::Global)),
        Some("causal_gqa8")
    );
    assert_eq!(
        Fa4::profile_name(&spec(64, 8, 128, false, Window::Global)),
        Some("gqa8")
    );
    // Grouping ratio 5 is genuinely absent; do not borrow a different ratio.
    assert!(!Fa4::supports(&spec(40, 8, 128, true, Window::Global)));
    assert!(!Fa4::supports(&spec(40, 8, 128, false, Window::Global)));
    // Global vision must select the global kernel, never the Modern local one.
    assert_eq!(
        Fa4::profile_name(&spec(12, 12, 64, false, Window::Global)),
        Some("bert")
    );
    assert_eq!(
        Fa4::profile_name(&spec(
            12,
            12,
            64,
            false,
            Window::Local {
                left: 64,
                right: 64
            }
        )),
        Some("modern_local")
    );
    // Right-asymmetric causal local and symmetric noncausal local remain
    // distinct mask families.
    assert!(matches!(
        Fa4::profile_name(&spec(
            32,
            8,
            128,
            true,
            Window::Local { left: 64, right: 0 }
        )),
        Some("causal_local128")
    ));
    assert_eq!(
        Fa4::profile_name(&spec(16, 16, 128, false, Window::Global)),
        Some("mha128")
    );
    assert_eq!(
        Fa4::profile_name(&spec(
            16,
            16,
            128,
            false,
            Window::Local {
                left: 32,
                right: 32
            }
        )),
        Some("local128")
    );
}

#[test]
fn malformed_media_with_zero_selectors_is_still_rejected() {
    let image = Arc::new(ImagePatches {
        pixels: vec![0.0; 3],
        grid_thw: [1, 2, 2],
        patch_dim: 4,
        merge_size: 2,
    });
    let batch = Batch {
        multimodal: vec![Some(Arc::new(MultimodalEncoding {
            images: vec![(0, image)],
            audios: vec![],
            position_ids: [vec![0, 0], vec![0, 0], vec![0, 0]],
            memory: None,
        }))],
        input_ids: vec![1, 2],
        token_type_ids: vec![0, 0],
        position_ids: vec![0, 1],
        cumulative_seq_lengths: vec![0, 2],
        max_length: 2,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let error = pack_batch(&batch, Pool::LastToken, true)
        .expect_err("zero-selector malformed media must be rejected");
    assert!(format!("{error}").contains("payload"), "{error}");
}

#[test]
fn zero_merge_size_and_audio_are_typed_rejections_without_panics() {
    let zero_merge = Arc::new(ImagePatches {
        pixels: vec![0.0; 16],
        grid_thw: [1, 2, 2],
        patch_dim: 4,
        merge_size: 0,
    });
    let batch = |media: Arc<MultimodalEncoding>| Batch {
        multimodal: vec![Some(media)],
        input_ids: vec![1, 2],
        token_type_ids: vec![0, 0],
        position_ids: vec![0, 1],
        cumulative_seq_lengths: vec![0, 2],
        max_length: 2,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let image_error = pack_batch(
        &batch(Arc::new(MultimodalEncoding {
            images: vec![(0, zero_merge)],
            audios: vec![],
            position_ids: [vec![0, 0], vec![0, 0], vec![0, 0]],
            memory: None,
        })),
        Pool::LastToken,
        false,
    )
    .expect_err("zero merge size must be a typed error");
    assert!(
        format!("{image_error}").contains("merge size"),
        "{image_error}"
    );

    let audio_error = pack_batch(
        &batch(Arc::new(MultimodalEncoding {
            images: vec![],
            audios: vec![(
                0,
                Arc::new(infinity_v3_backend_core::AudioFeatures {
                    values: vec![0.0],
                    mask: vec![1],
                    feature_size: 1,
                }),
            )],
            position_ids: [vec![0, 1], vec![0, 1], vec![0, 1]],
            memory: None,
        })),
        Pool::LastToken,
        false,
    )
    .expect_err("audio must be typed-rejected");
    assert!(
        format!("{audio_error}").contains("audio is unsupported"),
        "{audio_error}"
    );
}

#[test]
fn config_inconsistent_zero_selector_media_is_rejected_by_prepare_batch() {
    use crate::qwen3_vl_service::prepare_batch;
    let config = config_with(|_| {});
    let make = |merge_size: usize, patch_dim: usize, pixels: Vec<f32>, grid: [usize; 3]| Batch {
        multimodal: vec![Some(Arc::new(MultimodalEncoding {
            images: vec![(
                0,
                Arc::new(ImagePatches {
                    pixels,
                    grid_thw: grid,
                    patch_dim,
                    merge_size,
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
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let cases = [
        make(2, 0, vec![], [1, 2, 2]),
        make(2, 4, vec![0.0; 16], [1, 2, 2]),
        make(4, 1536, vec![0.0; 16 * 1536], [1, 4, 4]),
    ];
    for batch in cases {
        assert!(
            prepare_batch(&config, &batch, Pool::LastToken, false).is_err(),
            "config-inconsistent zero-selector media must be rejected"
        );
    }
}

#[test]
fn selector_bounds_duplicates_and_mixed_kinds_are_typed_rejected() {
    use crate::qwen3_vl_service::prepare_batch;
    let config = config_with(|_| {});
    let mut batch = Batch {
        multimodal: vec![],
        input_ids: vec![1, 2, 3, 4],
        token_type_ids: vec![0; 4],
        position_ids: vec![0, 1, 0, 1],
        cumulative_seq_lengths: vec![0, 2, 4],
        max_length: 2,
        pooled_indices: vec![0, 1],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    assert!(prepare_batch(&config, &batch, Pool::LastToken, false).is_ok());
    batch.pooled_indices = vec![2];
    assert!(prepare_batch(&config, &batch, Pool::LastToken, false).is_err());
    batch.pooled_indices = vec![0, 0];
    assert!(prepare_batch(&config, &batch, Pool::LastToken, false).is_err());
    batch.pooled_indices = vec![0];
    batch.raw_indices = vec![0];
    assert!(prepare_batch(&config, &batch, Pool::LastToken, false).is_err());
    batch.pooled_indices = vec![];
    batch.raw_indices = vec![1, 1];
    assert!(prepare_batch(&config, &batch, Pool::LastToken, true).is_err());
}

#[test]
fn plain_raw_output_uses_final_hidden_rows_without_a_readout() {
    use crate::qwen3_vl_service::Qwen3VlModel;
    let raw = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (rows, width) = Qwen3VlModel::raw_output_rows(None, &raw, 2, 3, false).unwrap();
    assert_eq!(width, 3);
    assert_eq!(rows, raw);
    assert!(Qwen3VlModel::raw_output_rows(None, &raw, 2, 3, true).is_err());
}
