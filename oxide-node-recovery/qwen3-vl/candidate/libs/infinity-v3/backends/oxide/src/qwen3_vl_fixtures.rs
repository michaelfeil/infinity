// Real Qwen3-VL processor fixture -> engine `Batch` plumbing.
//
// The fixtures are produced by the cached Rust processor (no Torch, no GPU, no
// download). This module reads the same ids/types/positions/cumulative arrays
// and the prepared f32 patch payloads, so the Oxide consumer and the Native b66
// consumer can be driven from byte-identical inputs.

use crate::checkpoint::start;
use infinity_v3_backend_core::{BackendError, Batch, ImagePatches, MultimodalEncoding, Pool};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureFile {
    pub fixtures: Vec<Fixture>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Fixture {
    pub name: String,
    pub ids: Vec<u32>,
    #[serde(default)]
    pub types: Vec<u32>,
    #[serde(default)]
    pub positions: Vec<u32>,
    pub cumulative: Vec<u32>,
    /// Absent for text-only fixtures. When present it must have three axes.
    #[serde(default)]
    pub media_positions: Option<Vec<Vec<u32>>>,
    #[serde(default)]
    pub images: Vec<FixtureImage>,
    #[serde(default)]
    pub pool: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureImage {
    pub grid_thw: [usize; 3],
    pub merge_size: usize,
    pub patch_dim: usize,
    /// Absolute path in the delivered fixture manifest, or a path relative to
    /// the manifest directory.
    pub pixels: String,
    pub token_start: usize,
    pub token_count: usize,
}

pub fn read_fixture_file(path: &Path) -> Result<FixtureFile, BackendError> {
    let bytes = std::fs::read(path).map_err(start)?;
    serde_json::from_slice(&bytes).map_err(start)
}

pub fn fixture_batch(
    fixture: &Fixture,
    fixture_root: &Path,
    raw: bool,
) -> Result<Batch, BackendError> {
    let tokens = fixture.ids.len();
    let sequences = fixture
        .cumulative
        .len()
        .checked_sub(1)
        .ok_or_else(|| start("fixture is missing cumulative boundaries"))?;
    if sequences == 0
        || fixture.cumulative.first().copied() != Some(0)
        || fixture.cumulative.last().copied() != Some(tokens as u32)
    {
        return Err(start("fixture cumulative boundaries are invalid"));
    }
    if fixture.positions.len() != tokens
        || (!fixture.types.is_empty() && fixture.types.len() != tokens)
    {
        return Err(start("fixture token arrays disagree"));
    }
    let types = if fixture.types.is_empty() {
        vec![0u32; tokens]
    } else {
        fixture.types.clone()
    };
    if let Some(positions) = fixture.media_positions.as_ref() {
        if positions.len() != 3 {
            return Err(start("fixture media_positions must have three axes"));
        }
        if positions.iter().any(|axis| axis.len() != tokens) {
            return Err(start("fixture media_positions do not cover every token"));
        }
    } else if !fixture.images.is_empty() {
        return Err(start(
            "fixture has images but no media_positions; refusing to synthesize positions",
        ));
    }

    let mut multimodal = Vec::with_capacity(sequences);
    for pair in fixture.cumulative.windows(2) {
        let begin = pair[0] as usize;
        let end = pair[1] as usize;
        let mut images = Vec::new();
        for (index, image) in fixture.images.iter().enumerate() {
            if image.token_start < begin || image.token_start >= end {
                continue;
            }
            let relative = image.token_start - begin;
            if relative + image.token_count > end - begin {
                return Err(start(format!(
                    "fixture image {index} crosses a sequence boundary"
                )));
            }
            let rows = image
                .grid_thw
                .iter()
                .try_fold(1usize, |value, dim| value.checked_mul(*dim))
                .ok_or_else(|| start("fixture image rows overflow"))?;
            if image.token_count
                != rows
                    .checked_div(image.merge_size * image.merge_size)
                    .unwrap_or(0)
            {
                return Err(start("fixture image token count disagrees with its grid"));
            }
            let expected = rows
                .checked_mul(image.patch_dim)
                .ok_or_else(|| start("fixture image payload size overflow"))?;
            let path = PathBuf::from(&image.pixels);
            let path = if path.is_absolute() {
                path
            } else {
                fixture_root.join(path)
            };
            let bytes = std::fs::read(&path)
                .map_err(|error| start(format!("fixture pixels {}: {error}", path.display())))?;
            if bytes.len() != expected * 4 {
                return Err(start(format!(
                    "fixture pixels {} have {} bytes, expected {}",
                    path.display(),
                    bytes.len(),
                    expected * 4
                )));
            }
            let pixels = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            images.push((
                relative,
                Arc::new(ImagePatches {
                    pixels,
                    grid_thw: image.grid_thw,
                    patch_dim: image.patch_dim,
                    merge_size: image.merge_size,
                }),
            ));
        }
        let position_ids: [Vec<u32>; 3] = match fixture.media_positions.as_ref() {
            Some(positions) => std::array::from_fn(|axis| positions[axis][begin..end].to_vec()),
            // Plain text rows use the packed local positions for all three axes.
            None => {
                let local = fixture.positions[begin..end].to_vec();
                [local.clone(), local.clone(), local]
            }
        };
        multimodal.push(Some(Arc::new(MultimodalEncoding {
            images,
            audios: vec![],
            position_ids,
            memory: None,
        })));
    }
    let raw_indices = if raw {
        (0..sequences).map(|row| row as u32).collect()
    } else {
        Vec::new()
    };
    Ok(Batch {
        multimodal,
        input_ids: fixture.ids.clone(),
        token_type_ids: types,
        position_ids: fixture.positions.clone(),
        cumulative_seq_lengths: fixture.cumulative.clone(),
        max_length: fixture
            .cumulative
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .unwrap_or(0),
        pooled_indices: if raw {
            Vec::new()
        } else {
            (0..sequences).map(|row| row as u32).collect()
        },
        raw_indices,
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    })
}

pub fn pool_from_name(name: Option<&str>) -> Result<Pool, BackendError> {
    match name.unwrap_or("last_token") {
        "cls" => Ok(Pool::Cls),
        "mean" | "avg" => Ok(Pool::Mean),
        "last_token" | "last" => Ok(Pool::LastToken),
        other => Err(start(format!("unsupported fixture pool {other}"))),
    }
}
