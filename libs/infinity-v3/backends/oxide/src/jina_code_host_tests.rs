// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU-only policy tests for the JinaCode host integration.

use super::*;
use crate::fa4_alibi::{profile_for_symbol, AlibiProfile};
use crate::model::{select_root, EncoderRecipe};
use std::collections::BTreeSet;

fn request() -> Batch {
    Batch {
        input_ids: vec![1, 2, 3],
        position_ids: vec![0, 0, 1],
        token_type_ids: vec![0, 1, 0],
        cumulative_seq_lengths: vec![0, 1, 3],
        pooled_indices: vec![0, 1],
        max_length: 2,
        multimodal: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}

#[test]
fn jina_code_geometry_owns_the_exact_alibi_post_norm_recipe() {
    let config = BertConfig::jina_code_geometry(768, 3072, 12, 12, 8192, 61056, 2, 1e-12);
    assert_eq!(config.family, EncoderFamily::JinaCode);
    assert_eq!(
        config.recipe,
        EncoderRecipe {
            embedding: NormArithmetic::ExactWordTypeBiased,
            residual: NormArithmetic::ExactBiased,
            positions: PositionRecipe::AlibiModelBias,
            feed_forward: FeedForwardRecipe::GatedGeluUpFirst,
        }
    );

    let mut batch = request();
    assert_eq!(validate(&batch, &config).unwrap(), 2);
    batch.token_type_ids[0] = 2;
    assert!(validate(&batch, &config).is_err());
    batch.token_type_ids[0] = 1;
    batch.position_ids[0] = 8192;
    assert!(validate(&batch, &config).is_err());
    batch.position_ids[0] = 8191;
    batch.input_ids[0] = 61056;
    assert!(validate(&batch, &config).is_err());
    batch.input_ids[0] = 1;
    assert!(validate(&batch, &config).is_ok());
}

#[test]
fn jina_code_root_selection_prefers_bare_then_bert_namespace() {
    let bare = BTreeSet::from(["embeddings.word_embeddings.weight".to_owned()]);
    assert_eq!(select_root(EncoderFamily::JinaCode, &bare).unwrap(), "");
    let prefixed = BTreeSet::from(["bert.embeddings.word_embeddings.weight".to_owned()]);
    assert_eq!(
        select_root(EncoderFamily::JinaCode, &prefixed).unwrap(),
        "bert."
    );
}

#[test]
fn alibi_typed_profiles_keep_eight_and_add_twelve_without_changing_the_abi() {
    let eight = AlibiProfile::for_heads(8).expect("legacy eight-head profile");
    assert_eq!(eight.host_symbol(), b"infinity_fa4_alibi_model_bias_v1\0");
    assert_eq!(
        eight.slopes(),
        [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625].as_slice()
    );

    let twelve = AlibiProfile::TwelveNativeFlash;
    assert!(AlibiProfile::for_heads(12).is_none());
    assert_eq!(
        twelve.host_symbol(),
        b"infinity_fa4_jinacode_flash_alibi_v1\0"
    );
    assert_eq!(twelve.slopes().len(), 12);
    assert_eq!(eight.token_limit(), i32::MAX as usize / 512);
    assert_eq!(twelve.token_limit(), i32::MAX as usize / 768);
    assert!(twelve.token_limit() + 1 < eight.token_limit());
    assert_eq!(&twelve.slopes()[..8], eight.slopes());
    assert!((twelve.slopes()[8] - 0.5f32.sqrt()).abs() < 1e-7);
    assert!((twelve.slopes()[9] - 0.125f32.sqrt()).abs() < 1e-7);
    assert!(AlibiProfile::for_heads(16).is_none());

    assert_eq!(
        profile_for_symbol("fa4_alibi_model_bias_bf16", "bf16"),
        Some(AlibiProfile::Eight)
    );
    assert_eq!(
        profile_for_symbol("fa4_jinacode_flash_alibi_bf16", "bf16"),
        Some(AlibiProfile::TwelveNativeFlash)
    );
    assert_eq!(
        profile_for_symbol("fa4_jinacode_flash_alibi_bf16", "fp16"),
        None
    );
}

#[test]
fn alibi_twelve_profile_rejects_incoherent_bias_metadata() {
    let correct = serde_json::json!({
        "symbol": "fa4_jinacode_flash_alibi_bf16", "dtype": "bf16",
        "kind": "alibi12_native_flash", "score_source": "alibi12_native_score.py",
        "heads": 12, "kv_heads": 12, "d": 64, "causal": false,
        "window": [null, null], "score_abi": 1, "qk_scale": 0.125,
        "external_softmax_scale": 1.0, "bias_rounding": "float32_fma_unscaled_then_pow2_scale",
        "alibi_heads": 12, "alibi_slope_count": 12
    });
    assert!(crate::fa4_alibi::kernel_matches(
        &correct,
        "bf16",
        AlibiProfile::TwelveNativeFlash
    ));
    for (field, value) in [
        ("kind", serde_json::json!("alibi_model_bias")),
        ("heads", serde_json::json!(8)),
        ("alibi_slope_count", serde_json::json!(8)),
        ("bias_rounding", serde_json::json!("float32")),
        ("external_softmax_scale", serde_json::json!(0.125)),
    ] {
        let mut changed = correct.clone();
        changed[field] = value;
        assert!(!crate::fa4_alibi::kernel_matches(
            &changed,
            "bf16",
            AlibiProfile::TwelveNativeFlash
        ));
    }
    assert!(!crate::fa4_alibi::kernel_matches(
        &correct,
        "bf16",
        AlibiProfile::Eight
    ));
}

#[test]
fn full_context_raw_and_mixed_fit_two_host_leases_and_release_large_token_arenas() {
    let config = BertConfig::jina_code_geometry(768, 3072, 12, 12, 8192, 61056, 2, 1e-12);
    let geometry = output::OutputGeometry::new(768, 768).unwrap();
    let build = |lengths: &[u32], pooled: Vec<u32>, raw: Vec<u32>| {
        let mut batch = request();
        let count = lengths.iter().map(|length| *length as usize).sum();
        batch.input_ids = vec![1; count];
        batch.token_type_ids = vec![0; count];
        batch.position_ids = lengths.iter().flat_map(|&length| 0..length).collect();
        batch.cumulative_seq_lengths = vec![0];
        for length in lengths {
            batch
                .cumulative_seq_lengths
                .push(batch.cumulative_seq_lengths.last().unwrap() + length);
        }
        batch.max_length = *lengths.iter().max().unwrap();
        batch.pooled_indices = pooled;
        batch.raw_indices = raw;
        batch
    };
    for (name, batch, key, gpu, pinned) in [
        (
            "pooled8192",
            build(&[8192], vec![0], vec![]),
            (8192, 2, 1),
            264_342_540,
            101_388,
        ),
        (
            "raw8192",
            build(&[8192], vec![], vec![0]),
            (8192, 2, 8192),
            289_538_056,
            25_296_904,
        ),
        (
            "mixed8192plus3",
            build(&[8192, 3], vec![1], vec![0]),
            (16384, 4, 16384),
            579_076_112,
            25_431_056,
        ),
    ] {
        validate(&batch, &config).unwrap();
        assert_eq!(work_key(&batch).unwrap(), key, "{name}");
        assert_eq!(
            Work::<Bf16>::allocation_bytes(key, &config).unwrap(),
            gpu,
            "{name}"
        );
        assert_eq!(
            staging::pinned_bytes(key, geometry.host_elements(&batch).unwrap()).unwrap(),
            pinned,
            "{name}"
        );
        assert!(
            2 * pinned < cache::BYTE_BUDGET,
            "{name}: two host leases must fit"
        );
        assert!(!needs_exclusive_staging(gpu, pinned), "{name}");
        if gpu > cache::BYTE_BUDGET {
            assert!(
                !capacity_fits(key, (32, 2, 1), gpu),
                "{name}: a later small token class releases the large arena"
            );
        }
        eprintln!(
            "{name}: GPU={gpu}B pinned-per-lease={pinned}B two-host-leases={}B",
            2 * pinned
        );
    }
}

#[test]
#[ignore = "CPU-only actual checkpoint admission; requires INFINITY_JINACODE_MODEL"]
fn actual_jina_code_checkpoint_admits_all_258_f16_sources_without_cuda() {
    let path = std::path::PathBuf::from(std::env::var("INFINITY_JINACODE_MODEL").unwrap());
    assert_eq!(
        detects_architecture(&path).unwrap(),
        CheckpointArchitecture::JinaCode
    );
    let weights = crate::jina_code_encoder_model::load(&path).unwrap();
    let geometry = weights.config.geometry();
    assert_eq!(geometry.hidden_size, 768);
    assert_eq!(geometry.num_hidden_layers, 12);
    assert_eq!(geometry.max_position_embeddings, 8192);
    assert_eq!(geometry.type_vocab_size, 2);
    let mut sources = vec![
        &weights.word,
        &weights.token_type,
        &weights.embedding_norm.gamma,
        &weights.embedding_norm.beta,
    ];
    for layer in &weights.layers {
        sources.extend([
            &layer.query.weight,
            &layer.query.bias,
            &layer.key.weight,
            &layer.key.bias,
            &layer.value.weight,
            &layer.value.bias,
            &layer.query_norm.gamma,
            &layer.query_norm.beta,
            &layer.key_norm.gamma,
            &layer.key_norm.beta,
            &layer.attention.weight,
            &layer.attention.bias,
            &layer.attention_norm.gamma,
            &layer.attention_norm.beta,
            &layer.layer_norm_1.gamma,
            &layer.layer_norm_1.beta,
            &layer.layer_norm_2.gamma,
            &layer.layer_norm_2.beta,
            &layer.mlp_up_gated,
            &layer.mlp_down.weight,
            &layer.mlp_down.bias,
        ]);
    }
    let pooler = weights._pooler.as_ref().unwrap();
    sources.extend([&pooler.weight, &pooler.bias]);
    assert_eq!(sources.len(), 258);
    let mut staging = vec![0u16; 65_536];
    for source in sources {
        assert_eq!(source.dtype(), safetensors::Dtype::F16);
        let mut position = 0usize;
        while position < source.len() {
            let count = (source.len() - position).min(staging.len());
            source.read_bf16(position, &mut staging[..count]).unwrap();
            assert!(staging[..count].iter().all(|word| word & 0x7f80 != 0x7f80));
            position += count;
        }
    }
}
