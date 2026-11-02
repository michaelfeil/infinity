// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU tests against the actual Cloudflare Clef/Clef-flash sidecars.
//!
//! The fixture directory contains verbatim public metadata only: backbone
//! `config.json`, `joint_head_config.json`, the original provenance file, and a
//! local hash manifest. No weight payload is present or synthesized at the
//! production sizes.

use crate::{
    clef_decision::{validate_runtime_dtype, ClefConfig},
    clef_decision_weights::tensor_shapes,
    clef_decision_workspace::ClefCapacity,
};
use std::{collections::BTreeMap, fs, path::Path};

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/clef")
        .join(name)
}

fn backbone_hidden(path: &Path) -> usize {
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("config.json")).unwrap()).unwrap();
    config["text_config"]["hidden_size"].as_u64().unwrap() as usize
}

#[test]
fn actual_clef_and_flash_sidecars_match_the_loader_schema() {
    for (name, hidden) in [
        ("Cloudflare--clef", 5120usize),
        ("Cloudflare--clef-flash", 4096usize),
    ] {
        let path = fixture(name);
        assert_eq!(backbone_hidden(&path), hidden);
        let config = ClefConfig::load(&path, hidden).unwrap();
        assert_eq!(
            (
                config.hidden,
                config.width,
                config.heads,
                config.feedforward
            ),
            (hidden, 1024, 16, 4096)
        );
        assert_eq!((config.layers, config.routing_layers), (4, 2));
        assert_eq!(config.head_dim(), 64);
        assert!(validate_runtime_dtype("bfloat16").is_ok());
        assert!(validate_runtime_dtype("float16").is_err());

        let shapes: BTreeMap<_, _> = tensor_shapes(&config).into_iter().collect();
        assert_eq!(shapes.len(), 122);
        assert_eq!(shapes["hidden_norm.weight"], vec![hidden]);
        assert_eq!(shapes["memory_projection.weight"], vec![1024, hidden]);
        assert_eq!(
            shapes["evidence_layers.1.attention.in_proj_weight"],
            vec![3072, 1024]
        );
        assert_eq!(
            shapes["evidence_layers.1.feedforward.0.weight"],
            vec![4096, 1024]
        );
        assert_eq!(
            shapes["layers.3.multihead_attn.out_proj.weight"],
            vec![1024, 1024]
        );
        assert_eq!(shapes["layers.0.linear1.weight"], vec![4096, 1024]);
        assert_eq!(shapes["residual_scorer.0.weight"], vec![1024, 4096]);

        let capacity = ClefCapacity::new(8, 2, 2, 4, 8, hidden, 1024, 4096).unwrap();
        assert_eq!(capacity.metadata_words(), 57);
        assert!(capacity.device_bytes().unwrap() > 0);
    }
}
