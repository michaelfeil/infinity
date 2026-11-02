// CPU acceptance tests for the isolated Qwen3.5 vision-language leaf.
//
// These tests exercise observable contracts: root alias admission, real media
// geometry, the actual host vision forward over packed patches, external mRoPE
// positions, and the exact visual-row replacement used before text prefill.

use crate::qwen35_vl_config::{Qwen35VlConfig, TextLayerKind};
use crate::qwen35_vl_media::{
    plan_image, plan_video, smart_resize, MediaGrid, MediaKind, MediaSpan,
};
use crate::qwen35_vl_text::{
    fuse_visual_rows, media_position_ids, mrope_axis_map, mrope_frequencies, prepare_text_fusion,
    TextFusionPlan,
};
use crate::qwen35_vl_vision::{vision_forward, HostImage, HostTensor, VisionHostWeights};
use serde_json::json;
use std::path::Path;

fn dense_root(depth: usize, hidden: usize) -> serde_json::Value {
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 10,
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "text_config": {
            "attn_output_gate": true,
            "dtype": "bfloat16",
            "head_dim": 8,
            "hidden_act": "silu",
            "hidden_size": hidden,
            "intermediate_size": 24,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 0],
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000000.0,
                "rope_type": "default"
            },
            "use_cache": true,
            "vocab_size": 64
        },
        "video_token_id": 11,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": depth,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": hidden,
            "in_channels": 3,
            "intermediate_size": 16,
            "model_type": "qwen3_5",
            "num_heads": 2,
            "num_position_embeddings": 4,
            "out_hidden_size": hidden,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 13,
        "vision_start_token_id": 12
    })
}

fn moe_root() -> serde_json::Value {
    let mut value = dense_root(1, 8);
    value["architectures"] = json!(["Qwen3_5MoeForConditionalGeneration"]);
    value["model_type"] = json!("qwen3_5_moe");
    value["text_config"]["model_type"] = json!("qwen3_5_moe_text");
    value["text_config"]["num_experts"] = json!(32);
    value["text_config"]["num_experts_per_tok"] = json!(4);
    value["text_config"]["moe_intermediate_size"] = json!(16);
    value["text_config"]["shared_expert_intermediate_size"] = json!(16);
    value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("intermediate_size");
    value["vision_config"]["model_type"] = json!("qwen3_5_moe");
    value
}

fn tiny_weights(config: &Qwen35VlConfig) -> VisionHostWeights {
    let vision = &config.vision;
    let hidden = vision.hidden_size;
    let intermediate = vision.intermediate_size;
    let grouped = hidden * vision.merge_area().unwrap();
    let mut weights = VisionHostWeights::default();
    let mut tensor = |name: &str, shape: Vec<usize>, scale: f32| {
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|index| ((index % 11) as f32 - 5.0) * scale)
            .collect();
        weights.insert(name, HostTensor::new(shape, data).unwrap());
    };
    tensor(
        "patch_embed.proj.weight",
        vec![hidden, 3, 2, vision.patch_size, vision.patch_size],
        0.01,
    );
    tensor("patch_embed.proj.bias", vec![hidden], 0.001);
    tensor("pos_embed.weight", vec![4, hidden], 0.01);
    for layer in 0..vision.depth {
        let prefix = format!("blocks.{layer}");
        for norm in ["norm1", "norm2"] {
            tensor(&format!("{prefix}.{norm}.weight"), vec![hidden], 0.01);
            tensor(&format!("{prefix}.{norm}.bias"), vec![hidden], 0.001);
        }
        tensor(
            &format!("{prefix}.attn.qkv.weight"),
            vec![hidden * 3, hidden],
            0.01,
        );
        tensor(&format!("{prefix}.attn.qkv.bias"), vec![hidden * 3], 0.001);
        tensor(
            &format!("{prefix}.attn.proj.weight"),
            vec![hidden, hidden],
            0.01,
        );
        tensor(&format!("{prefix}.attn.proj.bias"), vec![hidden], 0.001);
        tensor(
            &format!("{prefix}.mlp.linear_fc1.weight"),
            vec![intermediate, hidden],
            0.01,
        );
        tensor(
            &format!("{prefix}.mlp.linear_fc1.bias"),
            vec![intermediate],
            0.001,
        );
        tensor(
            &format!("{prefix}.mlp.linear_fc2.weight"),
            vec![hidden, intermediate],
            0.01,
        );
        tensor(
            &format!("{prefix}.mlp.linear_fc2.bias"),
            vec![hidden],
            0.001,
        );
    }
    tensor("merger.norm.weight", vec![hidden], 0.01);
    tensor("merger.norm.bias", vec![hidden], 0.001);
    tensor("merger.linear_fc1.weight", vec![grouped, grouped], 0.01);
    tensor("merger.linear_fc1.bias", vec![grouped], 0.001);
    tensor("merger.linear_fc2.weight", vec![hidden, grouped], 0.01);
    tensor("merger.linear_fc2.bias", vec![hidden], 0.001);
    weights
}

fn tiny_image(config: &Qwen35VlConfig) -> HostImage {
    let grid = MediaGrid::new(1, 2, 2, config.vision.spatial_merge_size).unwrap();
    let patch_dim = config.vision.patch_dim().unwrap();
    let rows = grid.patch_rows().unwrap();
    HostImage {
        name: "tiny".to_owned(),
        pixels: (0..rows * patch_dim)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.03)
            .collect(),
        grid,
        token_start: 0,
    }
}

#[test]
fn root_aliases_admit_dense_and_moe_conditional_generation() {
    let dense = Qwen35VlConfig::from_value(dense_root(1, 8)).unwrap();
    assert_eq!(dense.model_type, "qwen3_5");
    assert_eq!(dense.text.model_type, "qwen3_5_text");
    assert_eq!(dense.vision.model_type, "qwen3_5");
    assert_eq!(dense.text.rotary_dim, 4);
    assert_eq!(dense.text.mrope_section, [1, 1, 0]);
    assert_eq!(dense.vision.deepstack_visual_indexes, Vec::<usize>::new());

    let moe = Qwen35VlConfig::from_value(moe_root()).unwrap();
    assert_eq!(moe.model_type, "qwen3_5_moe");
    assert_eq!(moe.text.model_type, "qwen3_5_moe_text");
    assert_eq!(moe.vision.model_type, "qwen3_5_moe");
    assert!(moe.text.is_moe());
    assert_eq!(moe.text.num_experts, 32);
}

#[test]
fn text_only_architecture_is_not_admitted_as_vl() {
    let mut value = dense_root(1, 8);
    value["architectures"] = json!(["Qwen3_5ForCausalLM"]);
    let error = Qwen35VlConfig::from_value(value).unwrap_err().to_string();
    assert!(error.contains("text-only"), "{error}");
}

#[test]
fn non_square_position_table_is_rejected() {
    let mut value = dense_root(1, 8);
    value["vision_config"]["num_position_embeddings"] = json!(5);
    let error = Qwen35VlConfig::from_value(value).unwrap_err().to_string();
    assert!(error.contains("perfect square"), "{error}");
}

#[test]
fn partial_rotary_must_cover_mrope_sections() {
    let mut value = dense_root(1, 8);
    value["text_config"]["rope_parameters"]["mrope_section"] = json!([1, 1, 1]);
    let error = Qwen35VlConfig::from_value(value).unwrap_err().to_string();
    assert!(error.contains("mRoPE sections"), "{error}");
}

#[test]
fn smart_resize_rounds_to_patch_merge_factor_and_honours_pixel_bounds() {
    assert_eq!(
        smart_resize(100, 100, 16, 2, 4096, 25165824).unwrap(),
        (96, 96)
    );
    assert_eq!(
        smart_resize(70, 100, 16, 2, 4096, 25165824).unwrap(),
        (64, 96)
    );
    let (height, width) = smart_resize(1000, 2000, 16, 2, 65536, 262144).unwrap();
    assert!(height * width <= 262144 || height == 32 || width == 32);
    assert_eq!(height % 32, 0);
    assert_eq!(width % 32, 0);
    assert!(smart_resize(8, 8, 16, 2, 4096, 25165824).is_err());
}

#[test]
fn image_and_video_plans_keep_temporal_patch_pairs() {
    let config = Qwen35VlConfig::from_value(dense_root(1, 8)).unwrap();
    let media = crate::qwen35_vl_config::MediaPreprocessConfig {
        image_shortest_edge: 64,
        image_longest_edge: 16777216,
        video_shortest_edge: 64,
        video_longest_edge: 25165824,
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 2,
        image_mean: vec![0.5, 0.5, 0.5],
        image_std: vec![0.5, 0.5, 0.5],
        image_processor_type: "Qwen2VLImageProcessorFast".to_owned(),
        video_processor_type: "Qwen3VLVideoProcessor".to_owned(),
    };
    let image = plan_image(8, 8, &media, &config.vision).unwrap();
    assert_eq!(image.grid.grid_thw(), [1, 4, 4]);
    assert_eq!(image.grid.merged_tokens().unwrap(), 4);
    let video = plan_video(8, 8, 6, &media, &config.vision).unwrap();
    assert_eq!(video.grid.grid_thw(), [3, 4, 4]);
    assert_eq!(video.grid.merged_tokens().unwrap(), 12);
    assert!(plan_video(8, 8, 5, &media, &config.vision).is_err());
}

#[test]
fn patch_rows_are_merge_group_major() {
    let grid = MediaGrid::new(1, 4, 4, 2).unwrap();
    let coordinates: Vec<_> = (0..16)
        .map(|index| grid.merge_major_coords(index).unwrap())
        .collect();
    assert_eq!(coordinates[0], (0, 0));
    assert_eq!(coordinates[1], (0, 1));
    assert_eq!(coordinates[2], (1, 0));
    assert_eq!(coordinates[3], (1, 1));
    assert_eq!(coordinates[4], (0, 2));
    assert_eq!(coordinates[15], (3, 3));
}

#[test]
fn host_vision_forward_produces_merged_rows() {
    let config = Qwen35VlConfig::from_value(dense_root(2, 8)).unwrap();
    let weights = tiny_weights(&config);
    let image = tiny_image(&config);
    let output = vision_forward(&config, &weights, std::slice::from_ref(&image)).unwrap();
    assert_eq!(output.merged_rows, 1);
    assert_eq!(output.out_hidden, 8);
    assert_eq!(output.merged.len(), 8);
    assert!(output.merged.iter().all(|value| value.is_finite()));
    assert!(output.deepstack.is_empty());
    let repeats = vision_forward(&config, &weights, std::slice::from_ref(&image)).unwrap();
    assert_eq!(output.merged, repeats.merged);
}

#[test]
fn mrope_axis_map_uses_partial_rotary_width() {
    let axes = mrope_axis_map(64, [11, 11, 10]).unwrap();
    assert_eq!(axes.len(), 32);
    assert_eq!(axes[0], 0);
    assert_eq!(axes[1], 1);
    assert_eq!(axes[2], 2);
    assert!(axes.iter().all(|axis| *axis < 3));
    assert!(mrope_axis_map(64, [11, 11, 11]).is_err());
    let frequency = mrope_frequencies(64, 10_000_000.0).unwrap();
    assert_eq!(frequency.len(), 32);
    assert!(frequency[0] == 1.0);
}

#[test]
fn media_positions_match_native_advance() {
    let image = MediaSpan {
        kind: MediaKind::Image,
        token_start: 2,
        token_count: 1,
        grid: MediaGrid::new(1, 2, 2, 2).unwrap(),
    };
    let positions = media_position_ids(5, &[image]).unwrap();
    assert_eq!(positions[0], vec![0, 1, 2, 3, 4]);
    assert_eq!(positions[1], vec![0, 1, 2, 3, 4]);
    assert_eq!(positions[2], vec![0, 1, 2, 3, 4]);

    let video = MediaSpan {
        kind: MediaKind::Video,
        token_start: 0,
        token_count: 2,
        grid: MediaGrid::new(2, 2, 2, 2).unwrap(),
    };
    let positions = media_position_ids(4, &[video]).unwrap();
    assert_eq!(positions[0], vec![0, 1, 2, 3]);
    assert_eq!(positions[1], vec![0, 0, 2, 3]);
    assert_eq!(positions[2], vec![0, 0, 2, 3]);

    // Multi-row/column block used by the full Elephant/Mammoth media contract.
    let image = MediaSpan {
        kind: MediaKind::Image,
        token_start: 2,
        token_count: 4,
        grid: MediaGrid::new(1, 4, 4, 2).unwrap(),
    };
    let positions = media_position_ids(8, &[image]).unwrap();
    assert_eq!(positions[0], vec![0, 1, 2, 2, 2, 2, 4, 5]);
    assert_eq!(positions[1], vec![0, 1, 2, 2, 3, 3, 4, 5]);
    assert_eq!(positions[2], vec![0, 1, 2, 3, 2, 3, 4, 5]);
}

#[test]
fn text_fusion_replaces_only_media_rows_and_splits_gdn_from_mrope() {
    let config = Qwen35VlConfig::from_value(dense_root(1, 8)).unwrap();
    let span = MediaSpan {
        kind: MediaKind::Image,
        token_start: 2,
        token_count: 1,
        grid: MediaGrid::new(1, 2, 2, 2).unwrap(),
    };
    let plan = prepare_text_fusion(&config, 5, &[span]).unwrap();
    assert_eq!(plan.phases.tokens, 5);
    assert_eq!(plan.phases.half_dim, config.text.rotary_dim / 2);
    assert_eq!(plan.mrope_layers(&config), vec![1]);
    assert_eq!(plan.gdn_layers(&config), vec![0]);
    assert!(TextFusionPlan::layer_uses_mrope(TextLayerKind::Full));
    assert!(!TextFusionPlan::layer_uses_mrope(TextLayerKind::Linear));

    let mut hidden = vec![0f32; 5 * 3];
    for (index, value) in hidden.iter_mut().enumerate() {
        *value = index as f32;
    }
    let features = vec![7f32, 8.0, 9.0];
    fuse_visual_rows(&mut hidden, 3, &features, &[span]).unwrap();
    assert_eq!(&hidden[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&hidden[6..9], &features);
    assert_eq!(&hidden[9..], &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn inventory_derives_published_tensor_counts() {
    for (depth, expected) in [(12, 153), (24, 297), (27, 333)] {
        let config = Qwen35VlConfig::from_value(dense_root(depth, 8)).unwrap();
        assert_eq!(config.expected_vision_tensors(), expected);
        assert_eq!(
            crate::qwen35_vl_inventory::expected_vision_tensors(&config)
                .unwrap()
                .len(),
            expected
        );
    }
}

#[test]
#[ignore = "metadata-only admission against a cached original checkpoint"]
fn actual_cached_checkpoint_metadata_matches_inventory() {
    let path = std::env::var_os("INFINITY_OXIDE_TEST_QWEN35VL_CHECKPOINT")
        .expect("actual Qwen3.5-VL checkpoint directory");
    let path = Path::new(&path);
    // Some cached original weight snapshots carry config + shards but omit the
    // two preprocessor JSON files. In that case validate the same configuration
    // and checkpoint metadata without inventing a media policy.
    let config = match Qwen35VlConfig::from_path(path) {
        Ok(config) => config,
        Err(error) if error.to_string().contains("preprocessor_config.json") => {
            Qwen35VlConfig::from_config_path(path).unwrap()
        }
        Err(error) => panic!("config admission failed: {error}"),
    };
    let report = crate::qwen35_vl_inventory::validate_vision_inventory_path(&config, path).unwrap();
    assert!(report.vision_tensors >= 153);
    assert!(report.total_tensors > report.vision_tensors);
}

fn combined_processor() -> serde_json::Value {
    json!({
        "processor_class": "Qwen3VLProcessor",
        "image_processor": {
            "image_processor_type": "Qwen2VLImageProcessor",
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        },
        "video_processor": {
            "video_processor_type": "Qwen3VLVideoProcessor",
            "size": {"shortest_edge": 4096, "longest_edge": 25165824},
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }
    })
}

#[test]
fn combined_processor_config_is_accepted_as_original_media_contract() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "oxide-qwen35vl-config-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("config.json"), dense_root(1, 8).to_string()).unwrap();
    std::fs::write(
        path.join("processor_config.json"),
        combined_processor().to_string(),
    )
    .unwrap();
    let config = Qwen35VlConfig::from_path(&path).unwrap();
    assert_eq!(config.media.image_shortest_edge, 65536);
    assert_eq!(config.media.video_longest_edge, 25165824);
    assert_eq!(config.media.patch_size, 2);
    assert_eq!(config.media.image_processor_type, "Qwen2VLImageProcessor");
    assert_eq!(config.media.video_processor_type, "Qwen3VLVideoProcessor");
    let _ = std::fs::remove_dir_all(path);
}

#[test]
#[ignore = "metadata-only admission against actual full Elephant/Mammoth checkpoints"]
fn actual_full_checkpoints_admit_vision_and_text_geometry() {
    for path in std::env::var("INFINITY_OXIDE_TEST_QWEN35VL_FULL")
        .unwrap_or_default()
        .split(',')
        .filter(|value| !value.is_empty())
    {
        let path = Path::new(path);
        let config = Qwen35VlConfig::from_path(path).unwrap();
        assert_eq!(config.vision.depth, 27);
        assert_eq!(config.vision.hidden_size, 1152);
        assert_eq!(config.vision.num_heads, 16);
        assert_eq!(config.vision.head_dim(), 72);
        assert_eq!(config.text.rotary_dim, 64);
        assert!(matches!(config.text.hidden_size, 4096 | 5120));
        assert_eq!(config.expected_vision_tensors(), 333);
        let report =
            crate::qwen35_vl_inventory::validate_vision_inventory_path(&config, path).unwrap();
        assert_eq!(report.vision_tensors, 333);
        // The text owner admits the same full Elephant/Mammoth geometry, so the
        // owning VL engine can attach without a second text admission gate.
        crate::qwen35_config::Qwen35Config::load(path)
            .unwrap()
            .validate_candidate()
            .unwrap();
    }
}

#[test]
fn dtype_coverage_is_explicit_and_bf16_only_on_device() {
    let coverage = crate::qwen35_vl_engine::dtype_coverage();
    assert_eq!(coverage.reference_recipe, "BF16-rounded CPU reference");
    assert!(coverage.device_recipe.starts_with("BF16 device only"));
    assert!(coverage.f16_patch_stem.contains("not an F16 device"));
    assert!(coverage.f32_decay_parameters.contains("A_log/dt_bias"));
}

#[test]
fn central_fa4_mode_mapping_matches_actual_native_head_geometries() {
    use crate::qwen35_vl_engine::expected_central_mode;
    // Published FA4 profiles bake the ratio, not absolute heads.
    assert_eq!(expected_central_mode(16, 4, 256, true, false), Some(18));
    assert_eq!(expected_central_mode(8, 2, 256, true, false), Some(18));
    assert_eq!(expected_central_mode(16, 4, 256, false, false), Some(6));
    assert_eq!(expected_central_mode(8, 2, 256, false, false), Some(6));
    assert_eq!(expected_central_mode(24, 4, 256, true, false), Some(17));
    assert_eq!(expected_central_mode(24, 4, 256, false, false), Some(16));
    assert_eq!(expected_central_mode(16, 2, 256, true, false), Some(26));
    assert_eq!(expected_central_mode(8, 2, 256, false, false), Some(6));
    assert_eq!(expected_central_mode(12, 12, 64, false, true), Some(0));
    assert_eq!(expected_central_mode(16, 16, 64, false, true), Some(0));
    assert_eq!(expected_central_mode(16, 16, 72, false, true), Some(8));
    // A different ratio/head_dim is not silently mapped.
    assert_eq!(expected_central_mode(12, 12, 32, false, true), None);
    assert_eq!(expected_central_mode(16, 2, 128, true, false), None);
}

#[test]
#[ignore = "metadata-only run plans for every cached Native-accepted checkpoint"]
fn actual_all_checkpoints_have_explicit_normal_factory_run_plans() {
    let roots = std::env::var("INFINITY_OXIDE_TEST_QWEN35VL_ALL").unwrap_or_default();
    for entry in roots.split(',').filter(|value| !value.is_empty()) {
        let path = Path::new(entry);
        let plan = crate::qwen35_vl_engine::Qwen35VlEngine::run_plan(path).unwrap();
        println!("PLAN {} {:?}", plan.path, plan);
        assert!(plan.vision_tensors >= 153);
        assert!(plan
            .projection_score_route
            .contains("no Dense text projection"));
        for mode in &plan.modes {
            if mode.present() {
                if let Some(expected) = mode.expected_mode {
                    assert_eq!(Some(expected), mode.registry_mode, "{}", mode.component);
                }
            }
        }
        assert_eq!(plan.modes.len(), 2, "vision + the actual text mask only");
        if plan.text_hidden == 4096 {
            assert!(plan.normal_factory_ready, "{}: {:?}", plan.path, plan.gaps);
            assert_eq!(plan.modes[0].registry_mode, Some(8));
            assert!(!plan.text_causal);
            assert_eq!(plan.modes[1].registry_mode, Some(6));
        }
        if plan.text_hidden == 5120 {
            assert!(plan.normal_factory_ready, "{}: {:?}", plan.path, plan.gaps);
            assert_eq!(plan.modes[0].registry_mode, Some(8));
            assert!(!plan.text_causal);
            assert_eq!(plan.modes[1].registry_mode, Some(16));
        }
        if plan.family == "dense" && matches!(plan.text_hidden, 1024 | 2048 | 2560) {
            assert!(plan.normal_factory_ready, "{}: {:?}", plan.path, plan.gaps);
            assert_eq!(plan.modes[0].registry_mode, Some(0));
            assert_eq!(plan.modes[1].registry_mode, Some(18));
        }
        if plan.family == "moe" {
            assert!(plan.moe_owner_required);
            assert_eq!(plan.modes[0].registry_mode, Some(8));
            assert_eq!(plan.modes[1].expected_mode, Some(26));
            assert_eq!(plan.modes[1].registry_mode, None);
            assert!(!plan.normal_factory_ready);
        }
    }
}

#[test]
#[ignore = "metadata-only normal-factory route proof for every cached checkpoint"]
fn actual_all_checkpoints_use_normal_public_factory_route() {
    use infinity_v3_backend_core::{ModelType, Pool};
    for entry in std::env::var("INFINITY_OXIDE_TEST_QWEN35VL_ALL")
        .unwrap_or_default()
        .split(',')
        .filter(|value| !value.is_empty())
    {
        let path = Path::new(entry);
        let plan = crate::qwen35_vl_engine::Qwen35VlEngine::run_plan(path).unwrap();
        let (model_type, expected_route) = if plan.text_hidden == 4096 || plan.text_hidden == 5120 {
            (ModelType::Decision, "DynamicBlockDecision")
        } else {
            (ModelType::Embedding(Pool::Mean), "Text(causal=true)")
        };
        let proof =
            crate::qwen35_vl_engine::normal_factory_proof(path, "bfloat16", &model_type).unwrap();
        assert!(proof.is_qwen35_checkpoint, "{}", proof.gaps.join("; "));
        assert!(proof.dtype_accepted);
        assert_eq!(proof.serving_route, expected_route);
        assert_eq!(proof.run_plan_ready, plan.normal_factory_ready);
        if plan.normal_factory_ready {
            assert!(proof.gaps.is_empty());
        } else {
            assert!(!proof.gaps.is_empty());
        }
    }
}

#[test]
fn dropping_uncalled_vl_deferred_owner_runs_the_fence() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let fences = Arc::new(AtomicUsize::new(0));
    let signal = Arc::clone(&fences);
    let inner: infinity_v3_backend_core::Inference<u32> =
        infinity_v3_backend_core::Inference::Deferred(Box::new(move || {
            signal.fetch_add(1, Ordering::SeqCst);
            Ok(7)
        }));
    let owner = crate::qwen35_vl_engine::Qwen35VlInference::new(inner, None, None, None, None);
    drop(owner);
    assert_eq!(fences.load(Ordering::SeqCst), 1);
}

#[test]
fn leaf_media_validation_runs_before_empty_selector_ready() {
    use crate::qwen35_vl_engine::validate_media_requests;
    use crate::Qwen35VlRequest;
    let config = Qwen35VlConfig::from_value(dense_root(1, 8)).unwrap();
    let patch_dim = config.vision.patch_dim().unwrap();
    let grid = MediaGrid::new(1, 2, 2, 2).unwrap();
    let rows = grid.patch_rows().unwrap();
    let valid = Qwen35VlRequest {
        name: "valid".to_owned(),
        kind: MediaKind::Image,
        pixels: vec![0.0; rows * patch_dim],
        grid,
        token_start: 1,
    };
    assert!(validate_media_requests(&config, 5, 262_144, std::slice::from_ref(&valid)).is_ok());
    let wrong_pixels = Qwen35VlRequest {
        pixels: vec![0.0; rows * patch_dim - 1],
        ..valid.clone()
    };
    assert!(
        validate_media_requests(&config, 5, 262_144, std::slice::from_ref(&wrong_pixels)).is_err()
    );
    let outside = Qwen35VlRequest {
        token_start: 5,
        ..valid
    };
    assert!(validate_media_requests(&config, 5, 262_144, std::slice::from_ref(&outside)).is_err());
}

#[cfg(test)]
mod lifecycle_guard_tests {
    use crate::qwen35_vl_engine::{
        FailureSink, Fence, Qwen35VlFailure, Qwen35VlFusion, Qwen35VlInference, Qwen35VlPrepGuard,
    };
    use crate::Inference;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct FakeFence(bool);
    impl Fence for FakeFence {
        fn proven(&self) -> bool {
            self.0
        }
    }

    struct FakeSink(Arc<AtomicUsize>);
    impl FailureSink for FakeSink {
        fn close(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn handle(failed: &Arc<AtomicBool>, closes: &Arc<AtomicUsize>) -> Qwen35VlFailure {
        Qwen35VlFailure::new(Arc::clone(failed), Arc::new(FakeSink(Arc::clone(closes))))
    }

    #[test]
    fn prep_guard_unproven_fence_fails_closed_once() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        drop(Qwen35VlPrepGuard::test_guard(
            handle(&failed, &closes),
            Arc::new(FakeFence(false)),
        ));
        assert!(failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prep_guard_proven_fence_does_not_close() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        drop(Qwen35VlPrepGuard::test_guard(
            handle(&failed, &closes),
            Arc::new(FakeFence(true)),
        ));
        assert!(!failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn fusion_unproven_fence_fails_closed_once() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        drop(Qwen35VlFusion::test_guard(
            handle(&failed, &closes),
            Arc::new(FakeFence(false)),
            false,
            None,
        ));
        assert!(failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn fusion_proven_fence_does_not_close() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        drop(Qwen35VlFusion::test_guard(
            handle(&failed, &closes),
            Arc::new(FakeFence(true)),
            false,
            None,
        ));
        assert!(!failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn owner_unproven_fence_quarantines_and_fails_closed() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        let failure = handle(&failed, &closes);
        let fusion =
            Qwen35VlFusion::test_guard(failure.clone(), Arc::new(FakeFence(false)), false, None);
        let inner: Inference<()> =
            Inference::Deferred(Box::new(|| Err(crate::infer("test failure"))));
        let mut owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            None,
            Some(failure),
            Some(Arc::new(FakeFence(false))),
        );
        assert!(owner.complete().is_err());
        assert!(failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn owner_proven_fence_does_not_fail_closed() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        let failure = handle(&failed, &closes);
        let fusion =
            Qwen35VlFusion::test_guard(failure.clone(), Arc::new(FakeFence(true)), false, None);
        let inner: Inference<()> =
            Inference::Deferred(Box::new(|| Err(crate::infer("test failure"))));
        let mut owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            None,
            Some(failure),
            Some(Arc::new(FakeFence(true))),
        );
        assert!(owner.complete().is_err());
        assert!(!failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn owner_unproven_cancel_retains_fusion_token_and_fails_closed() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        let failure = handle(&failed, &closes);
        let token = Arc::new(());
        let fusion = Qwen35VlFusion::test_guard(
            failure.clone(),
            Arc::new(FakeFence(false)),
            false,
            Some(Arc::clone(&token)),
        );
        let inner: Inference<()> = Inference::Deferred(Box::new(|| Err(crate::infer("cancelled"))));
        let owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            None,
            Some(failure),
            Some(Arc::new(FakeFence(false))),
        );
        drop(owner);
        assert!(failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&token), 2);
    }

    #[test]
    fn owner_proven_pack_error_releases_fusion_token() {
        let failed = Arc::new(AtomicBool::new(false));
        let closes = Arc::new(AtomicUsize::new(0));
        let failure = handle(&failed, &closes);
        let token = Arc::new(());
        let fusion = Qwen35VlFusion::test_guard(
            failure.clone(),
            Arc::new(FakeFence(true)),
            false,
            Some(Arc::clone(&token)),
        );
        let inner: Inference<()> =
            Inference::Deferred(Box::new(|| Err(crate::infer("pack error"))));
        let mut owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            None,
            Some(failure),
            Some(Arc::new(FakeFence(true))),
        );
        assert!(owner.complete().is_err());
        assert!(!failed.load(Ordering::SeqCst));
        assert_eq!(closes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&token), 1);
    }
}
