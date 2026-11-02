// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact Clef joint-head schema tests.

use super::*;
use crate::clef_decision::CLEF_WEIGHTS;
use safetensors::tensor::{serialize_to_file, TensorView};
use std::{
    fs,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn config() -> ClefConfig {
    ClefConfig {
        hidden: 16,
        width: 16,
        heads: 2,
        feedforward: 32,
        routing_layers: 1,
        layers: 1,
    }
}

#[test]
fn validates_exact_native_tensor_names_shapes_and_dtypes() {
    let path = std::env::temp_dir().join(format!(
        "infinity-clef-head-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    let shapes = tensor_shapes(&config());
    let data = shapes
        .iter()
        .map(|(_, shape)| {
            let count = shape.iter().product::<usize>();
            vec![0u8; count * 2]
        })
        .collect::<Vec<_>>();
    let views = shapes
        .iter()
        .zip(&data)
        .map(|((name, shape), bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let file = path.join(CLEF_WEIGHTS);
    serialize_to_file(views, None, &file).unwrap();
    assert!(validate_joint_sources(&file, &config()).is_ok());
    assert_eq!(tensor_shapes(&config()).len(), 54);
    let _ = fs::remove_dir_all(path);
}

#[test]
fn scalar_parser_checks_bytes_and_preserves_first_bf16_cast() {
    let two = 2.0f32.to_le_bytes();
    let value = scalar_bf16(Dtype::F32, &[], &two).unwrap();
    assert_eq!(value.to_bits(), 0x4000);

    let negative = (-1.5f32).to_le_bytes();
    let value = scalar_bf16(Dtype::F32, &[], &negative).unwrap();
    assert_eq!(value.to_bits(), 0xbfc0);

    assert!(scalar_bf16(Dtype::BF16, &[], &[0]).is_err());
    assert!(scalar_bf16(Dtype::BF16, &[], &[0, 0, 0]).is_err());
    assert!(scalar_bf16(Dtype::F32, &[], &[0, 1, 2]).is_err());
    assert!(scalar_bf16(Dtype::BF16, &[1], &[0, 0]).is_err());
    assert!(scalar_bf16(Dtype::BF16, &[], &0x7fc0u16.to_le_bytes()).is_err());
}
