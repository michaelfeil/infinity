// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU-only policy tests for the JinaBase H768 host integration.

use super::*;
use crate::fa4_alibi::{kernel_matches, profile_for_symbol, AlibiProfile};
use crate::model::{EncoderRecipe, NormArithmetic, PositionRecipe};
use serde_json::json;

fn h768_config() -> BertConfig {
    BertConfig::jina_geometry(768, 3072, 12, 12, 8192, 30528, 2, 1e-12)
}

#[test]
fn official_h768_header_routes_to_jina_before_generic_bert() {
    let raw: crate::jina_encoder_model::RawJinaMarker = serde_json::from_value(json!({
        "model_type": "bert",
        "_name_or_path": "jinaai/jina-bert-implementation",
        "auto_map": {
            "AutoConfig": "jinaai/jina-bert-implementation--configuration_bert.JinaBertConfig"
        }
    }))
    .unwrap();
    assert_eq!(
        selects_architecture(&raw).unwrap(),
        CheckpointArchitecture::Jina
    );

    let code: crate::jina_encoder_model::RawJinaMarker = serde_json::from_value(json!({
        "model_type": "bert",
        "_name_or_path": "jinaai/jina-bert-v2-qk-post-norm",
    }))
    .unwrap();
    assert_eq!(
        selects_architecture(&code).unwrap(),
        CheckpointArchitecture::JinaCode
    );
}

#[test]
fn official_h768_geometry_validates_and_stays_distinct_from_jinacode() {
    let config = h768_config();
    config.validate().unwrap();
    assert_eq!(config.family, EncoderFamily::Jina);
    assert_eq!(
        config.recipe,
        EncoderRecipe {
            embedding: NormArithmetic::ExactWordTypeBiased,
            residual: NormArithmetic::RoundedAddBeforeStats,
            positions: PositionRecipe::AlibiModelBias,
            feed_forward: FeedForwardRecipe::GatedGeluUpFirst,
        }
    );

    let jina_code = BertConfig::jina_code_geometry(768, 3072, 12, 12, 8192, 61056, 2, 1e-12);
    assert_eq!(config.recipe.embedding, jina_code.recipe.embedding);
    assert_ne!(config.recipe.residual, jina_code.recipe.residual);
    assert_ne!(config.vocab_size, jina_code.vocab_size);

    for recipe in [
        EncoderRecipe {
            residual: NormArithmetic::ExactBiased,
            ..config.recipe
        },
        EncoderRecipe {
            embedding: NormArithmetic::ExactBiased,
            ..config.recipe
        },
        EncoderRecipe {
            feed_forward: FeedForwardRecipe::DenseGelu,
            ..config.recipe
        },
    ] {
        let mut bad = config.clone();
        bad.recipe = recipe;
        assert!(bad.validate().is_err());
    }

    for (hidden, heads, intermediate, layers, vocab, types) in [
        (768, 8, 3072, 12, 30528, 2),
        (768, 12, 2048, 12, 30528, 2),
        (768, 12, 3072, 11, 30528, 2),
        (768, 12, 3072, 12, 61056, 2),
        (768, 12, 3072, 12, 30528, 3),
        (512, 12, 3072, 12, 30528, 2),
    ] {
        let mut bad = BertConfig::jina_geometry(
            hidden,
            intermediate,
            heads,
            layers,
            8192,
            vocab,
            types,
            1e-12,
        );
        if hidden == 512 {
            bad.recipe = config.recipe;
        }
        assert!(
            bad.validate().is_err(),
            "{hidden}/{heads}/{intermediate}/{layers}/{vocab}/{types}"
        );
    }
}

#[test]
fn h768_kernel_selection_uses_word_type_embedding_and_reuses_deberta_norm() {
    let config = h768_config();
    assert!(selects_jina_h768_word_type_embedding(&config));
    assert!(selects_jina_h768_rounded_norm(&config));
    assert_eq!(
        jina_h768_residual_norm_kernel(),
        "infinity_deberta_layer_norm_768"
    );
    assert!(!selects_mpnet_h768(&config));

    let layout = packed_attention_layout(&config).unwrap().unwrap();
    assert_eq!(layout.strides[0], 3 * 768);
    assert_eq!(layout.strides[1], 64);
    assert_eq!(layout.strides[9], 768);
    assert_eq!(layout.k_offset, 2 * 768);
    assert_eq!(layout.v_offset, 4 * 768);

    let h512 = BertConfig::jina_geometry(512, 2048, 8, 4, 8192, 30528, 2, 1e-12);
    h512.validate().unwrap();
    assert!(!selects_jina_h768_word_type_embedding(&h512));
    assert!(!selects_jina_h768_rounded_norm(&h512));
    assert_eq!(
        packed_attention_layout(&h512).unwrap().unwrap().strides[0],
        1536
    );
}

#[test]
fn alibi12_model_bias_profile_is_additive_and_keeps_eight_heads_legacy() {
    let eight = AlibiProfile::for_heads(8).expect("legacy eight-head profile");
    assert_eq!(AlibiProfile::for_heads(12), None);
    assert_eq!(eight.host_symbol(), b"infinity_fa4_alibi_model_bias_v1\0");

    let native = AlibiProfile::TwelveNativeFlash;
    let base = AlibiProfile::TwelveModelBias;
    assert_eq!(base.heads(), 12);
    assert_eq!(base.token_limit(), i32::MAX as usize / 768);
    assert_eq!(base.slopes(), native.slopes());
    assert_eq!(&base.slopes()[..8], eight.slopes());
    assert_eq!(base.slopes().len(), 12);
    assert_eq!(base.host_symbol(), b"infinity_fa4_alibi12_model_bias_v1\0");
    assert_ne!(base.host_symbol(), native.host_symbol());
    assert_ne!(base.host_symbol(), eight.host_symbol());
    assert_eq!(
        profile_for_symbol("fa4_alibi12_model_bias_bf16", "bf16"),
        Some(base)
    );
    assert_eq!(
        profile_for_symbol("fa4_alibi12_model_bias_bf16", "fp16"),
        None
    );
    assert_eq!(
        profile_for_symbol("fa4_alibi_model_bias_bf16", "bf16"),
        Some(eight)
    );
    assert_eq!(
        profile_for_symbol("fa4_jinacode_flash_alibi_bf16", "bf16"),
        Some(native)
    );
}

#[test]
fn alibi12_model_bias_metadata_rejects_jinacode_fma_and_eight_head_contracts() {
    let base = AlibiProfile::TwelveModelBias;
    let native = AlibiProfile::TwelveNativeFlash;
    let eight = AlibiProfile::Eight;
    let correct = json!({
        "symbol": "fa4_alibi12_model_bias_bf16", "dtype": "bf16",
        "kind": "alibi12_model_bias", "score_source": "alibi12_model_bias_score.py",
        "heads": 12, "kv_heads": 12, "d": 64, "causal": false,
        "window": [null, null], "score_abi": 1, "qk_scale": 0.125,
        "external_softmax_scale": 1.0, "bias_rounding": "model_dtype",
        "alibi_heads": 12, "alibi_slope_count": 12
    });
    assert!(kernel_matches(&correct, "bf16", base));
    assert!(!kernel_matches(&correct, "bf16", native));
    assert!(!kernel_matches(&correct, "bf16", eight));

    let native_metadata = json!({
        "symbol": "fa4_jinacode_flash_alibi_bf16", "dtype": "bf16",
        "kind": "alibi12_native_flash", "score_source": "alibi12_native_score.py",
        "heads": 12, "kv_heads": 12, "d": 64, "causal": false,
        "window": [null, null], "score_abi": 1, "qk_scale": 0.125,
        "external_softmax_scale": 1.0,
        "bias_rounding": "float32_fma_unscaled_then_pow2_scale",
        "alibi_heads": 12, "alibi_slope_count": 12
    });
    assert!(kernel_matches(&native_metadata, "bf16", native));
    assert!(!kernel_matches(&native_metadata, "bf16", base));

    for (field, value) in [
        ("kind", json!("alibi12_native_flash")),
        ("score_source", json!("alibi12_native_score.py")),
        (
            "bias_rounding",
            json!("float32_fma_unscaled_then_pow2_scale"),
        ),
        ("alibi_heads", json!(8)),
        ("alibi_slope_count", json!(8)),
        ("heads", json!(8)),
        ("d", json!(32)),
        ("causal", json!(true)),
        ("score_abi", json!(2)),
        ("external_softmax_scale", json!(0.125)),
    ] {
        let mut changed = correct.clone();
        changed[field] = value;
        assert!(!kernel_matches(&changed, "bf16", base), "{field}");
    }
}
