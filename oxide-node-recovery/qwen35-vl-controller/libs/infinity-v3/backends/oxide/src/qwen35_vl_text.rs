// Text-side fusion for Qwen3.5 vision-language roots.
//!
//! Three things differ from Qwen3-VL and are handled explicitly here:
//!
//! 1. Qwen3.5 stores mRoPE under `rope_parameters` and rotates only a
//!    `partial_rotary_factor` prefix (64 of 256 in every published config).
//!    Axis assignment and inverse frequencies therefore cover `rotary/2`, not
//!    `head_dim/2`.
//! 2. Only full-attention layers consume the three multimodal position axes.
//!    Gated-delta (linear-attention) layers have no rotary position input at
//!    all, so a cross-modal span must not be projected into GDN state.
//! 3. Published Qwen3.5 roots have an empty deepstack list; media therefore
//!    contributes only merged rows and one mRoPE block.

use crate::qwen35_vl_config::{Qwen35VlConfig, TextLayerKind};
use crate::qwen35_vl_media::MediaSpan;
use infinity_v3_backend_core::BackendError;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl text: {error}"))
}

/// Native Qwen3.5 interleaved mRoPE axis assignment over `rotary/2` phases.
pub fn mrope_axis_map(rotary_dim: usize, sections: [usize; 3]) -> Result<Vec<u8>, BackendError> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(start("rotary width must be positive and even"));
    }
    let sum = sections
        .iter()
        .try_fold(0usize, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| start("mRoPE section sum overflow"))?;
    if sum != rotary_dim / 2 {
        return Err(start(format!(
            "mRoPE sections {sections:?} do not sum to rotary/2 ({})",
            rotary_dim / 2
        )));
    }
    let s1 = sections[1];
    let s2 = sections[2];
    let mut axes = Vec::with_capacity(rotary_dim / 2);
    for index in 0..rotary_dim / 2 {
        let axis = if index % 3 == 1 && index < s1 * 3 {
            1
        } else if index % 3 == 2 && index < s2 * 3 {
            2
        } else {
            0
        };
        axes.push(axis);
    }
    Ok(axes)
}

/// `freq[j] = 1 / theta^(2j/rotary)` in F32, matching the Native partial-rope
/// inverse frequencies in `qwen35_models.cpp`.
pub fn mrope_frequencies(rotary_dim: usize, theta: f64) -> Result<Vec<f32>, BackendError> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(start("rotary width must be positive and even"));
    }
    if !(theta.is_finite() && theta > 0.0) {
        return Err(start("rope theta must be positive and finite"));
    }
    let mut out = Vec::with_capacity(rotary_dim / 2);
    for index in 0..rotary_dim / 2 {
        out.push((1.0f64 / theta.powf(2.0 * index as f64 / rotary_dim as f64)) as f32);
    }
    Ok(out)
}

/// Build the Native position remap for media spans.
///
/// `tokens` is the full multimodal sequence length. Text gaps advance all three
/// axes together. A media span contributes temporal/row/column positions and
/// advances the next text token by `max(temporal, merged_height, merged_width)`.
pub fn media_position_ids(
    tokens: usize,
    spans: &[MediaSpan],
) -> Result<[Vec<u32>; 3], BackendError> {
    let mut ordered: Vec<&MediaSpan> = spans.iter().collect();
    ordered.sort_unstable_by_key(|span| span.token_start);
    let mut axes: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
    for axis in &mut axes {
        axis.try_reserve_exact(tokens)
            .map_err(|_| start("position allocation"))?;
    }
    let mut cursor = 0usize;
    let mut next = 0u32;
    for span in ordered {
        let grid = span.grid;
        let patches = grid
            .temporal
            .checked_mul(grid.height)
            .and_then(|value| value.checked_mul(grid.width))
            .ok_or_else(|| start("media grid overflow"))?;
        let area = grid
            .merge
            .checked_mul(grid.merge)
            .filter(|value| *value > 0)
            .ok_or_else(|| start("invalid media merge size"))?;
        if patches == 0
            || !grid.height.is_multiple_of(grid.merge)
            || !grid.width.is_multiple_of(grid.merge)
            || !patches.is_multiple_of(area)
        {
            return Err(start("invalid media grid for merge"));
        }
        let length = patches / area;
        if length == 0 || span.token_count != length {
            return Err(start("media token count does not match merged geometry"));
        }
        let end = span
            .token_start
            .checked_add(length)
            .filter(|end| *end <= tokens)
            .ok_or_else(|| start("media token span exceeds token count"))?;
        if span.token_start < cursor {
            return Err(start("overlapping media token spans"));
        }
        for _ in cursor..span.token_start {
            for axis in &mut axes {
                axis.push(next);
            }
            next = next
                .checked_add(1)
                .ok_or_else(|| start("text position overflow"))?;
        }
        let height = grid.height / grid.merge;
        let width = grid.width / grid.merge;
        for temporal in 0..grid.temporal {
            let temporal = next
                .checked_add(u32::try_from(temporal).map_err(start)?)
                .ok_or_else(|| start("temporal position overflow"))?;
            for row in 0..height {
                let row = next
                    .checked_add(u32::try_from(row).map_err(start)?)
                    .ok_or_else(|| start("row position overflow"))?;
                for column in 0..width {
                    let column = next
                        .checked_add(u32::try_from(column).map_err(start)?)
                        .ok_or_else(|| start("column position overflow"))?;
                    axes[0].push(temporal);
                    axes[1].push(row);
                    axes[2].push(column);
                }
            }
        }
        let advance = height.max(width).max(grid.temporal);
        next = next
            .checked_add(u32::try_from(advance).map_err(start)?)
            .ok_or_else(|| start("position advance overflow"))?;
        cursor = end;
    }
    for _ in cursor..tokens {
        for axis in &mut axes {
            axis.push(next);
        }
        next = next
            .checked_add(1)
            .ok_or_else(|| start("text position overflow"))?;
    }
    if axes.iter().any(|axis| axis.len() != tokens) {
        return Err(start("multimodal position length mismatch"));
    }
    Ok(axes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MropePhases {
    pub tokens: usize,
    pub half_dim: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl MropePhases {
    pub fn row(&self, token: usize) -> Option<(&[f32], &[f32])> {
        if token >= self.tokens {
            return None;
        }
        let start = token.checked_mul(self.half_dim)?;
        let end = start.checked_add(self.half_dim)?;
        Some((self.cos.get(start..end)?, self.sin.get(start..end)?))
    }
}

pub fn mrope_phases(
    positions: &[Vec<u32>; 3],
    axes: &[u8],
    frequency: &[f32],
) -> Result<MropePhases, BackendError> {
    if positions[0].len() != positions[1].len() || positions[0].len() != positions[2].len() {
        return Err(start("mRoPE axes have different lengths"));
    }
    if axes.len() != frequency.len() {
        return Err(start("mRoPE axis and frequency widths differ"));
    }
    let tokens = positions[0].len();
    let half_dim = axes.len();
    let mut cos = vec![
        0f32;
        tokens
            .checked_mul(half_dim)
            .ok_or_else(|| start("phase overflow"))?
    ];
    let mut sin = cos.clone();
    for token in 0..tokens {
        for index in 0..half_dim {
            let axis = axes[index] as usize;
            if axis >= 3 {
                return Err(start("mRoPE axis index exceeds three"));
            }
            let angle = positions[axis][token] as f32 * frequency[index];
            let (sine, cosine) = angle.sin_cos();
            cos[token * half_dim + index] = cosine;
            sin[token * half_dim + index] = sine;
        }
    }
    Ok(MropePhases {
        tokens,
        half_dim,
        cos,
        sin,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextFusionPlan {
    pub tokens: usize,
    pub spans: Vec<MediaSpan>,
    pub positions: [Vec<u32>; 3],
    pub axes: Vec<u8>,
    pub frequency: Vec<f32>,
    pub phases: MropePhases,
}

impl TextFusionPlan {
    /// True when the selected text layer consumes external mRoPE positions.
    pub fn layer_uses_mrope(kind: TextLayerKind) -> bool {
        matches!(kind, TextLayerKind::Full)
    }

    pub fn mrope_layers(&self, config: &Qwen35VlConfig) -> Vec<usize> {
        config
            .text
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, kind)| Self::layer_uses_mrope(**kind))
            .map(|(index, _)| index)
            .collect()
    }

    pub fn gdn_layers(&self, config: &Qwen35VlConfig) -> Vec<usize> {
        config
            .text
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, kind)| !Self::layer_uses_mrope(**kind))
            .map(|(index, _)| index)
            .collect()
    }
}

pub fn prepare_text_fusion(
    config: &Qwen35VlConfig,
    tokens: usize,
    spans: &[MediaSpan],
) -> Result<TextFusionPlan, BackendError> {
    let text = &config.text;
    let axes = mrope_axis_map(text.rotary_dim, text.mrope_section)?;
    let frequency = mrope_frequencies(text.rotary_dim, text.rope_theta)?;
    let positions = media_position_ids(tokens, spans)?;
    let phases = mrope_phases(&positions, &axes, &frequency)?;
    Ok(TextFusionPlan {
        tokens,
        spans: spans.to_vec(),
        positions,
        axes,
        frequency,
        phases,
    })
}

/// Replace each media span's embedding rows with the corresponding merged
/// vision rows. This is the actual text fusion mutation used by the decoder
/// prefill; it validates the exact cross-modal geometry before touching rows.
pub fn fuse_visual_rows(
    hidden: &mut [f32],
    width: usize,
    features: &[f32],
    spans: &[MediaSpan],
) -> Result<(), BackendError> {
    if width == 0 || !hidden.len().is_multiple_of(width) {
        return Err(start("text hidden state width mismatch"));
    }
    let tokens = hidden.len() / width;
    if !features.len().is_multiple_of(width) {
        return Err(start("visual feature width mismatch"));
    }
    let mut feature_offset = 0usize;
    let mut cursor = 0usize;
    for span in spans {
        if span.token_start < cursor {
            return Err(start("overlapping media spans"));
        }
        let end = span
            .token_start
            .checked_add(span.token_count)
            .filter(|end| *end <= tokens)
            .ok_or_else(|| start("media span exceeds token count"))?;
        let feature_end = feature_offset
            .checked_add(span.token_count)
            .filter(|end| *end <= features.len() / width)
            .ok_or_else(|| start("media span exceeds visual rows"))?;
        hidden[span.token_start * width..end * width]
            .copy_from_slice(&features[feature_offset * width..feature_end * width]);
        feature_offset = feature_end;
        cursor = end;
    }
    if feature_offset != features.len() / width {
        return Err(start("unused visual rows after fusion"));
    }
    Ok(())
}
