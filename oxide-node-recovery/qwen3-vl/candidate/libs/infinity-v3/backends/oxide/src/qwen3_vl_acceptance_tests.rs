// CPU acceptance tests for actual Qwen3-VL Oxide coverage.
//
// The cached-checkpoint tests read config/index/safetensors headers only. They
// skip with an explicit message if the reviewed checkpoint is not cached; they
// never download or synthesize weights.

use crate::checkpoint::CheckpointMetadata;
use crate::qwen3_vl_capi::{
    CaseImage, CaseSpec, RootControlPacket, RuntimeProfile, BASE_COMMIT, CUBLASLT_PIN,
    PACKET_SCHEMA, SDK_PREFIX,
};
use crate::qwen3_vl_config::{Dtype, Qwen3VlConfig};
use crate::qwen3_vl_contract::{
    select, CornerSumOrder, MergeNorm, TextRopeSource, VisionActivation, VisionProfile,
};
use crate::qwen3_vl_enqueue::{BlockStage, PreparedVision, VisionSink, BLOCK_STAGES};
use crate::qwen3_vl_inventory::{
    validate_checkpoint, validate_optional_bias_groups, InventoryScope,
};
use crate::qwen3_vl_layout::{fusion_layout, text_arena, vision_arena};
use crate::qwen3_vl_text::{
    deepstack_text_layers, empty_prefix_deepstack, image_position_ids, marker_spans,
    mrope_axis_map, mrope_frequencies, prepare_text_fusion,
};
use crate::qwen3_vl_vision::{
    deepstack_capture_order, frame_groups, frame_plan, image_feature_slices, merge_major_coords,
    position_interpolation, vision_angles, vision_batches, vision_frequency, ImageGrid,
};
use crate::qwen3_vl_weights::plan as weight_plan;
use std::path::Path;

const FIXTURE_CONFIG: &str = r#"{
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

fn fixture_config() -> Qwen3VlConfig {
    Qwen3VlConfig::from_json(FIXTURE_CONFIG.as_bytes()).expect("fixture config")
}

fn grid(frames: usize, height: usize, width: usize, start: usize) -> ImageGrid {
    let count = frames * height * width / 4;
    ImageGrid::new(frames, height, width, 2, start, count).expect("grid")
}

fn cached() -> Option<(Qwen3VlConfig, CheckpointMetadata)> {
    if !crate::cached_checkpoint_present() {
        if std::env::var_os("QWEN3_VL_REQUIRE_CACHED").is_some() {
            panic!(
                "reviewed Qwen3-VL checkpoint is required but absent at {}",
                crate::CACHED_CHECKPOINT
            );
        }
        eprintln!("cached Qwen3-VL checkpoint absent; skipping header binding test");
        return None;
    }
    let root = Path::new(crate::CACHED_CHECKPOINT);
    let config = Qwen3VlConfig::from_path(root).expect("cached config");
    let metadata = CheckpointMetadata::open(root).expect("cached headers");
    Some((config, metadata))
}

fn packet_for(
    config: &Qwen3VlConfig,
    metadata: &CheckpointMetadata,
    scope: InventoryScope,
    profile: RuntimeProfile,
) -> RootControlPacket {
    let report = validate_checkpoint(config, metadata, scope).expect("inventory");
    let plan = weight_plan(&report, metadata).expect("weight plan");
    let cases = vec![CaseSpec {
        name: "fixture-tower".to_owned(),
        scope: "tower_only".to_owned(),
        tokens: 8,
        images: vec![CaseImage {
            frames: 1,
            height: 2,
            width: 4,
            merge_size: 2,
            token_start: 2,
            token_count: 2,
        }],
        expected_merged_rows: 2,
        expected_frames: 1,
        expected_deepstack_rows: 6,
        note: "structural case only; not a quality claim".to_owned(),
    }];
    RootControlPacket::build(
        Path::new(crate::CACHED_CHECKPOINT),
        config,
        metadata,
        &report,
        &plan,
        cases,
        profile,
    )
    .expect("packet")
}

#[test]
fn cached_config_binds_exact_native_geometry() {
    let Some((config, _)) = cached() else {
        return;
    };
    assert_eq!(config.model_type, "qwen3_vl");
    assert_eq!(config.vision.hidden_size, 1152);
    assert_eq!(config.vision.num_heads, 16);
    assert_eq!(config.vision.head_dim(), 72);
    assert_eq!(config.vision.depth, 27);
    assert_eq!(config.vision.intermediate_size, 4304);
    assert_eq!(config.vision.spatial_merge_size, 2);
    assert_eq!(config.vision.patch_size, 16);
    assert_eq!(config.vision.temporal_patch_size, 2);
    assert_eq!(config.vision.num_position_embeddings, 2304);
    assert_eq!(config.vision.out_hidden_size, 4096);
    assert_eq!(config.vision.deepstack_visual_indexes, vec![8, 16, 24]);
    assert_eq!(config.text.hidden_size, 4096);
    assert_eq!(config.text.num_attention_heads, 32);
    assert_eq!(config.text.num_key_value_heads, 8);
    assert_eq!(config.text.head_dim, 128);
    assert_eq!(config.text.num_hidden_layers, 36);
    assert_eq!(config.text.intermediate_size, 12288);
    assert_eq!(config.text.vocab_size, 151936);
    assert_eq!(config.text.rope_theta, 5_000_000.0);
    assert_eq!(config.text.mrope_section, [24, 20, 20]);
    assert_eq!(config.vision.patch_dim().unwrap(), 3 * 2 * 16 * 16);
    assert_eq!(config.vision.dtype, Dtype::Bf16);
    assert_eq!(config.vision.position_side(), 48);
    assert_eq!(
        crate::qwen3_vl_capi::activation_code(&config.text.hidden_act).unwrap(),
        3
    );
}

#[test]
fn cached_inventory_is_complete_and_balanced() {
    let Some((config, metadata)) = cached() else {
        return;
    };
    let full = validate_checkpoint(&config, &metadata, InventoryScope::Full).expect("full");
    assert_eq!(full.vision_count, 333);
    assert_eq!(full.text_count, 398);
    assert_eq!(full.deepstack_count, 18);
    assert_eq!(full.vision_count + full.deepstack_count, 351);
    assert_eq!(full.bound.len(), 749);
    assert_eq!(full.vision_layers.len(), 27);
    assert_eq!(full.text_layers.len(), 36);
    assert_eq!(full.deepstack_layers, vec![8, 16, 24]);
    assert_eq!(full.total_elements, 8_144_793_840);
    assert_eq!(full.total_bytes, 16_289_587_680);
    let tower = validate_checkpoint(&config, &metadata, InventoryScope::TowerOnly).expect("tower");
    assert_eq!(tower.bound.len(), 351);
    assert_eq!(tower.vision_count, 333);
    assert_eq!(tower.text_count, 0);
    assert_eq!(tower.deepstack_count, 18);
    let plan = weight_plan(&full, &metadata).expect("plan");
    plan.verify(&full).expect("verified plan");
    assert_eq!(plan.tensor_count, 749);
    assert_eq!(plan.total_source_bytes, full.total_bytes);
    for shard in &plan.shards {
        for weight in &shard.weights {
            assert_eq!(weight.destination_offset % 256, 0);
            assert_eq!(
                weight.source_length as usize,
                weight.shape.iter().product::<usize>() * 2
            );
        }
    }
    let tower_plan = weight_plan(&tower, &metadata).expect("tower plan");
    assert_eq!(tower_plan.tensor_count, 351);
}

#[test]
fn contract_names_the_default_bf16_stage_order_and_rejects_fake_profiles() {
    let config = fixture_config();
    let contract = select(&config, false).expect("contract");
    assert_eq!(contract.vision.profile, VisionProfile::Default16);
    assert_eq!(
        contract.vision.activation,
        VisionActivation::GeluPytorchTanh
    );
    assert_eq!(
        contract.vision.merger_activation,
        VisionActivation::GeluPytorchTanh
    );
    assert_eq!(contract.vision.corner_sum, CornerSumOrder::DefaultSum);
    assert_eq!(contract.vision.final_merger, MergeNorm::PreGroup);
    assert_eq!(contract.vision.deepstack_merger, MergeNorm::PostGroup);
    assert_eq!(contract.vision.layer_norm_epsilon, 1e-6);
    assert_eq!(contract.text.rope_source, TextRopeSource::MultimodalAxes);
    assert!(contract.text.causal);
    assert!(contract
        .bf16_stage_order
        .iter()
        .any(|stage| stage.contains("frame-isolated")));
    let multi = select(&config, true).expect("multi contract");
    assert_eq!(multi.vision.profile, VisionProfile::MultiVector);
    assert_eq!(multi.vision.merger_activation, VisionActivation::GeluExact);
    assert_eq!(multi.vision.corner_sum, CornerSumOrder::MultiVectorLeft);
}

#[test]
fn deepstack_capture_uses_tower_layer_order_and_name_index() {
    assert_eq!(
        deepstack_capture_order(&[8, 16, 24], 27).unwrap(),
        vec![(8, 0), (16, 1), (24, 2)]
    );
    assert_eq!(
        deepstack_capture_order(&[24, 8, 16], 27).unwrap(),
        vec![(8, 1), (16, 2), (24, 0)]
    );
    assert!(deepstack_capture_order(&[27], 27).is_err());
    assert!(deepstack_capture_order(&[8, 8], 27).is_err());
    assert_eq!(
        crate::qwen3_vl_inventory::deepstack_mapping(&fixture_config()),
        vec![(8, 0), (16, 1), (24, 2)]
    );
}

#[test]
fn vision_geometry_preserves_frames_merge_and_slices() {
    let a = grid(1, 2, 4, 0);
    let b = grid(3, 2, 2, 2);
    assert_eq!(a.patch_rows().unwrap(), 8);
    assert_eq!(a.token_count, 2);
    assert_eq!(b.patch_rows().unwrap(), 12);
    assert_eq!(b.token_count, 3);
    let plan = frame_plan(&[a, b]).unwrap();
    assert_eq!(plan.cu_seqlens, vec![0, 8, 12, 16, 20]);
    assert_eq!(plan.frame_lengths, vec![8, 4, 4, 4]);
    assert_eq!(plan.image_frame_spans, vec![(0, 1), (1, 4)]);
    assert_eq!(plan.merged_counts, vec![2, 3]);
    assert_eq!(plan.max_frame, 8);
    assert_eq!(frame_groups(&plan).len(), 2);
    assert_eq!(image_feature_slices(&[a, b]).unwrap(), vec![(0, 2), (2, 3)]);
    let slices = image_feature_slices(&[a, b]).unwrap();
    assert_eq!(slices.iter().map(|(_, count)| count).sum::<usize>(), 5);
}

#[test]
fn merge_major_coordinates_match_native_recovery() {
    let recovered: Vec<(usize, usize)> = (0..8)
        .map(|index| merge_major_coords(index, 2, 2, 4).unwrap())
        .collect();
    assert_eq!(
        recovered,
        vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3)
        ]
    );
    assert!(merge_major_coords(8, 2, 2, 4).is_err());
    assert!(merge_major_coords(0, 2, 3, 4).is_err());
}

#[test]
fn position_interpolation_matches_four_corner_native_formula() {
    let corners = position_interpolation(2, 2, 4, 2).unwrap();
    assert_eq!(corners.len(), 8);
    assert_eq!(
        corners[0],
        crate::qwen3_vl_vision::PositionCorner {
            row: 0.0,
            column: 0.0,
            r0: 0,
            r1: 0,
            c0: 0,
            c1: 0,
            dr: 0.0,
            dc: 0.0
        }
    );
    assert_eq!(corners[4].row, 0.0);
    assert!((corners[4].column - 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(corners[4].c0, 0);
    assert_eq!(corners[4].c1, 1);
    assert!((corners[4].dc - 2.0 / 3.0).abs() < 1e-6);
    let wide = position_interpolation(3, 2, 4, 2).unwrap();
    assert_eq!(wide[1].column, 1.0 / 3.0 * 2.0);
    assert_eq!(wide[1].c0, 0);
    assert_eq!(wide[1].c1, 1);
}

#[test]
fn vision_rotary_uses_two_dimensional_split_half_frequencies() {
    let frequency = vision_frequency(72).unwrap();
    assert_eq!(frequency.len(), 18);
    assert_eq!(frequency[0], 1.0);
    assert!((frequency[1] - (1.0f64 / 10_000f64.powf(2.0 / 36.0)) as f32).abs() < 1e-6);
    let angles = vision_angles(72, 3.0, 5.0).unwrap();
    assert_eq!(angles.len(), 36);
    assert_eq!(angles[0], 3.0);
    assert_eq!(angles[18], 5.0);
}

#[test]
fn vision_batching_is_bounded_and_order_preserving() {
    let all = vision_batches(&[8, 12, 4], 1152, 4304, u64::MAX).unwrap();
    assert_eq!(all, vec![(0, 3)]);
    let each = vision_batches(&[8, 12, 4], 1152, 4304, 1).unwrap();
    assert_eq!(each, vec![(0, 1), (1, 2), (2, 3)]);
    let mixed = vision_batches(&[2, 2, 9, 1], 1152, 4304, 40_000).unwrap();
    assert_eq!(mixed.first().unwrap().0, 0);
    assert_eq!(mixed.last().unwrap().1, 4);
    let mut covered = vec![false; 4];
    for (start, end) in mixed {
        for slot in covered.iter_mut().take(end).skip(start) {
            assert!(!*slot);
            *slot = true;
        }
    }
    assert!(covered.into_iter().all(|value| value));
    let large = vision_batches(&[4096usize; 8], 1152, 4304, 2 * 1024 * 1024 * 1024).unwrap();
    assert!(large.len() > 1);
    assert_eq!(large.first().unwrap().0, 0);
    assert_eq!(large.last().unwrap().1, 8);
}

#[test]
fn mrope_axes_match_interleaved_sections() {
    let axes = mrope_axis_map(128, [24, 20, 20]).unwrap();
    assert_eq!(axes.len(), 64);
    assert_eq!(axes[0], 0);
    assert_eq!(axes[1], 1);
    assert_eq!(axes[2], 2);
    assert_eq!(axes[4], 1);
    assert_eq!(axes[5], 2);
    assert_eq!(axes[7], 1);
    assert_eq!(axes[8], 2);
    assert_eq!(axes[58], 1);
    assert_eq!(axes[59], 2);
    assert_eq!(axes[60], 0);
    assert_eq!(axes[63], 0);
    assert!(mrope_axis_map(128, [24, 20, 19]).is_err());
    let frequency = mrope_frequencies(128, 5_000_000.0).unwrap();
    assert_eq!(frequency.len(), 64);
    assert_eq!(frequency[0], 1.0);
    let phases = prepare_text_fusion(&fixture_config().text, 3, &[], 0).unwrap();
    assert_eq!(
        phases.positions,
        [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]]
    );
    let (cos, sin) = phases.phases.row(0).unwrap();
    assert_eq!(cos[0], 1.0);
    assert_eq!(sin[0], 0.0);
    assert_eq!(cos.len(), 64);
}

#[test]
fn image_positions_match_frozen_media_foundation_vector() {
    let portrait = ImageGrid::new(1, 6, 4, 2, 2, 6).unwrap();
    let landscape = ImageGrid::new(1, 2, 6, 2, 10, 3).unwrap();
    let axes = image_position_ids(15, &[landscape, portrait]).unwrap();
    assert_eq!(axes[0], vec![0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 10, 11]);
    assert_eq!(axes[1], vec![0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 10, 11]);
    assert_eq!(axes[2], vec![0, 1, 2, 3, 2, 3, 2, 3, 5, 6, 7, 8, 9, 10, 11]);
    assert!(
        image_position_ids(15, &[portrait, ImageGrid::new(1, 2, 6, 2, 7, 3).unwrap()]).is_err()
    );
}

#[test]
fn marker_spans_and_deepstack_stages_are_accounted() {
    let a = grid(1, 2, 4, 2);
    let b = grid(3, 2, 2, 4);
    let spans = marker_spans(&[b, a], 8).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].token_start, 2);
    assert_eq!(spans[0].token_count, 2);
    assert_eq!(spans[1].token_start, 4);
    assert_eq!(spans[1].token_count, 3);
    assert!(marker_spans(&[a, a], 8).is_err());
    assert_eq!(deepstack_text_layers(36, 3).unwrap(), vec![0, 1, 2]);
    assert!(deepstack_text_layers(2, 3).is_err());
    let config = fixture_config();
    let placeholders = empty_prefix_deepstack(&config.text, 3).unwrap();
    assert_eq!(placeholders.len(), 3);
    assert!(placeholders.iter().all(|shape| *shape == [0, 4096]));
    let fusion = prepare_text_fusion(&config.text, 8, &[a], 3).unwrap();
    assert_eq!(fusion.markers.len(), 1);
    assert_eq!(fusion.deepstack_layers, vec![0, 1, 2]);
    assert_eq!(fusion.phases.tokens, 8);
}

#[test]
fn layout_is_aligned_and_scales_with_frames_and_deepstack() {
    let config = fixture_config();
    let image = grid(3, 2, 2, 0);
    let arena = vision_arena(&config, &[image]).unwrap();
    assert_eq!(arena.total_bytes % 256, 0);
    for section in &arena.sections {
        assert_eq!(section.offset % 256, 0);
        assert_eq!(section.bytes, section.elements * section.element_bytes);
    }
    let merged_rows = image.token_count as u64;
    let grouped = (config.vision.hidden_size * config.vision.merge_area().unwrap()) as u64;
    let merger = arena
        .sections
        .iter()
        .find(|section| section.name == "merger_fc1_bf16")
        .unwrap();
    assert_eq!(merger.elements, merged_rows * grouped);
    let text = text_arena(&config, 8).unwrap();
    assert_eq!(text.total_bytes % 256, 0);
    let qkv = text
        .sections
        .iter()
        .find(|section| section.name == "qkv_bf16")
        .unwrap();
    assert_eq!(
        qkv.elements,
        8u64 * (config.text.q_width().unwrap() + 2 * config.text.kv_width().unwrap()) as u64
    );
    let fusion = fusion_layout(&config, 8, &[image, grid(1, 2, 4, 3)], true).unwrap();
    assert!(fusion.vision.is_some());
    assert_eq!(fusion.marker_rows, 5);
    assert_eq!(fusion.deepstack_rows, 15);
    assert_eq!(fusion.text.total_bytes, text.total_bytes);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Call {
    PatchEmbed,
    PositionRotary,
    Block(usize, BlockStage),
    Deepstack(usize, usize),
    Merger,
}

struct OrderSink {
    calls: Vec<Call>,
}

impl VisionSink for OrderSink {
    type Error = ();
    fn patch_embed(&mut self, _batch: &crate::qwen3_vl_enqueue::VisionBatch) -> Result<(), ()> {
        self.calls.push(Call::PatchEmbed);
        Ok(())
    }
    fn position_rotary(&mut self, _batch: &crate::qwen3_vl_enqueue::VisionBatch) -> Result<(), ()> {
        self.calls.push(Call::PositionRotary);
        Ok(())
    }
    fn block(
        &mut self,
        _batch: &crate::qwen3_vl_enqueue::VisionBatch,
        layer: usize,
        stage: BlockStage,
    ) -> Result<(), ()> {
        self.calls.push(Call::Block(layer, stage));
        Ok(())
    }
    fn deepstack(
        &mut self,
        _batch: &crate::qwen3_vl_enqueue::VisionBatch,
        layer: usize,
        name_index: usize,
    ) -> Result<(), ()> {
        self.calls.push(Call::Deepstack(layer, name_index));
        Ok(())
    }
    fn merger(&mut self, _batch: &crate::qwen3_vl_enqueue::VisionBatch) -> Result<(), ()> {
        self.calls.push(Call::Merger);
        Ok(())
    }
}

#[test]
fn prepared_vision_enqueue_is_stage_ordered_and_allocation_free() {
    let config = fixture_config();
    let images = [grid(1, 2, 4, 0), grid(3, 2, 2, 2), grid(1, 2, 2, 5)];
    let prepared = PreparedVision::prepare(&config, &images, 1).unwrap();
    assert_eq!(prepared.batches.len(), 3);
    assert_eq!(prepared.total_patch_rows, 8 + 12 + 4);
    assert_eq!(prepared.total_merged_rows, 2 + 3 + 1);
    let expected_calls = prepared.stage_calls();
    let mut sink = OrderSink {
        calls: Vec::with_capacity(expected_calls),
    };
    let capacity = sink.calls.capacity();
    prepared.enqueue(&mut sink).unwrap();
    assert_eq!(sink.calls.len(), expected_calls);
    assert_eq!(sink.calls.capacity(), capacity);
    assert_eq!(sink.calls[0], Call::PatchEmbed);
    assert_eq!(sink.calls[1], Call::PositionRotary);
    assert_eq!(sink.calls[2], Call::Block(0, BlockStage::Norm1));
    let deepstack_at = 2 + 8 * BLOCK_STAGES.len() + BLOCK_STAGES.len();
    assert_eq!(sink.calls[deepstack_at], Call::Deepstack(8, 0));
    assert_eq!(
        sink.calls[deepstack_at + 1],
        Call::Block(9, BlockStage::Norm1)
    );
    let merger_at = 2 + 27 * BLOCK_STAGES.len() + 3;
    assert_eq!(sink.calls[merger_at], Call::Merger);
    let wrong_merge = ImageGrid::new(1, 4, 4, 4, 0, 1).unwrap();
    assert!(PreparedVision::prepare(&config, &[wrong_merge], u64::MAX).is_err());
}

#[test]
fn control_packet_preserves_every_weight_and_original_config() {
    let Some((config, metadata)) = cached() else {
        return;
    };
    let packet = packet_for(
        &config,
        &metadata,
        InventoryScope::Full,
        RuntimeProfile::NativeProduction,
    );
    packet.verify(&metadata).expect("verified packet");
    assert_eq!(packet.schema, PACKET_SCHEMA);
    assert_eq!(packet.base_commit, BASE_COMMIT);
    assert_eq!(packet.weights.len(), 749);
    assert_eq!(packet.inventory.tensor_count, 749);
    assert_eq!(packet.original_config, config.raw);
    assert_eq!(packet.config_sha256.len(), 64);
    assert_eq!(packet.text_config_sha256.len(), 64);
    assert_eq!(packet.sdk.expected_prefix, SDK_PREFIX);
    assert_eq!(packet.sdk.cublaslt_expected, CUBLASLT_PIN);
    assert!(!packet.sdk.library_loaded);
    assert_eq!(packet.runtime_profile, "native_production");
    assert_eq!(
        packet
            .runtime_options
            .get("_cuda_graphs")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        packet
            .runtime_options
            .get("_media_cuda_graphs")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        packet
            .runtime_options
            .get("_prefix_cache_max_bytes")
            .map(String::as_str),
        Some("0")
    );
    assert!(packet.isolation.cuda_visible_devices.is_empty());
    assert!(packet.isolation.cuda_hidden);
    assert!(!packet.isolation.tensor_payload_read);
    let json = packet.to_json().unwrap();
    let round = RootControlPacket::from_json(json.as_bytes()).unwrap();
    round.verify(&metadata).unwrap();
    let preflight = packet.preflight(&metadata).unwrap();
    assert_eq!(preflight.tensor_count, 749);
    assert!(preflight.config_preserved);
    assert!(!preflight.library_loaded);
    let tower = packet_for(
        &config,
        &metadata,
        InventoryScope::TowerOnly,
        RuntimeProfile::NativeProduction,
    );
    tower.verify(&metadata).unwrap();
    assert_eq!(tower.weights.len(), 351);
    assert_eq!(tower.inventory.text_count, 0);
}

#[test]
fn control_packet_rejects_mutated_config_dropped_weight_and_bad_range() {
    let Some((config, metadata)) = cached() else {
        return;
    };
    let packet = packet_for(
        &config,
        &metadata,
        InventoryScope::Full,
        RuntimeProfile::NativeProduction,
    );

    let mut mutated = packet.clone();
    mutated.original_config["text_config"]["hidden_size"] = serde_json::json!(4097);
    assert!(mutated.verify(&metadata).is_err());

    let mut dropped = packet.clone();
    dropped.weights.pop();
    dropped.inventory.tensor_count -= 1;
    assert!(dropped.verify(&metadata).is_err());

    let mut bad = packet.clone();
    bad.weights[0].source_offset = u64::MAX - 4;
    assert!(bad.verify(&metadata).is_err());

    let mut graph = packet.clone();
    graph
        .runtime_options
        .insert("_media_cuda_graphs".to_owned(), "true".to_owned());
    assert!(graph.verify(&metadata).is_err());

    let mut payload = packet.clone();
    payload.isolation.tensor_payload_read = true;
    assert!(payload.verify(&metadata).is_err());
}

#[test]
fn oxide_low_overhead_profile_is_separate_from_native_production() {
    let Some((config, metadata)) = cached() else {
        return;
    };
    let packet = packet_for(
        &config,
        &metadata,
        InventoryScope::Full,
        RuntimeProfile::OxideLowOverhead,
    );
    packet.verify(&metadata).unwrap();
    assert_eq!(packet.runtime_profile, "oxide_low_overhead");
    assert_eq!(
        packet
            .runtime_options
            .get("_cuda_graphs")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        packet
            .runtime_options
            .get("_media_cuda_graphs")
            .map(String::as_str),
        Some("false")
    );
    let graph = packet_for(
        &config,
        &metadata,
        InventoryScope::Full,
        RuntimeProfile::NativeGraphEvidence,
    );
    graph.verify(&metadata).unwrap();
    assert_eq!(
        graph
            .runtime_options
            .get("_media_cuda_graphs")
            .map(String::as_str),
        Some("true")
    );
}

#[test]
fn optional_bias_groups_match_native_all_or_none_contract() {
    let config = fixture_config();
    let mut present = std::collections::BTreeSet::new();
    validate_optional_bias_groups(&config.text, &present).unwrap();
    present.insert("model.language_model.layers.0.self_attn.q_proj.bias".to_owned());
    assert!(validate_optional_bias_groups(&config.text, &present).is_err());
    present.insert("model.language_model.layers.0.self_attn.k_proj.bias".to_owned());
    present.insert("model.language_model.layers.0.self_attn.v_proj.bias".to_owned());
    validate_optional_bias_groups(&config.text, &present).unwrap();
    present.insert("model.language_model.layers.0.mlp.gate_proj.bias".to_owned());
    assert!(validate_optional_bias_groups(&config.text, &present).is_err());
    present.insert("model.language_model.layers.0.mlp.up_proj.bias".to_owned());
    validate_optional_bias_groups(&config.text, &present).unwrap();
}

#[test]
fn config_validation_rejects_fake_strips_and_bad_geometry() {
    let mut config = fixture_config();
    config.text.mrope_interleaved = false;
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.text.mrope_section = [24, 20, 19];
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.vision.deepstack_visual_indexes = vec![8, 8];
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.vision.depth = 0;
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.model_type = "qwen3_vl_text".to_owned();
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.text.hidden_size = 4095;
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.vision.temporal_patch_size = 1;
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.text.dtype = Dtype::F16;
    assert!(config.validate().is_err());
    let mut config = fixture_config();
    config.vision.num_position_embeddings = 2305;
    assert!(config.validate().is_err());
}
