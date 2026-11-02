// Text-side fusion contract for Qwen3-VL: interleaved mRoPE, marker spans,
// deepstack stage mapping and cached-prefix placeholders.
//
// `image_position_ids` is a selective extraction of the frozen media
// foundation v2 implementation (`backends/core/src/media/packed.rs`). It keeps
// the same checked ordering and arithmetic so the Oxide processor and executor
// cannot drift from the shared Qwen3Vl policy.

use crate::checkpoint::start;
use crate::qwen3_vl_config::TextConfig;
use crate::qwen3_vl_vision::ImageGrid;
use infinity_v3_backend_core::BackendError;

/// Native Qwen3-VL interleaved mRoPE axis assignment.
pub fn mrope_axis_map(text_dim: usize, sections: [usize; 3]) -> Result<Vec<u8>, BackendError> {
    if text_dim == 0 || !text_dim.is_multiple_of(2) {
        return Err(start("text head dimension must be positive and even"));
    }
    let sum = sections
        .iter()
        .try_fold(0usize, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| start("mRoPE section sum overflow"))?;
    if sum != text_dim / 2 {
        return Err(start(format!(
            "mRoPE sections {sections:?} do not sum to head_dim/2 ({})",
            text_dim / 2
        )));
    }
    let s1 = sections[1];
    let s2 = sections[2];
    let mut axes = Vec::with_capacity(text_dim / 2);
    for j in 0..text_dim / 2 {
        let axis = if j % 3 == 1 && j < s1 * 3 {
            1
        } else if j % 3 == 2 && j < s2 * 3 {
            2
        } else {
            0
        };
        axes.push(axis);
    }
    Ok(axes)
}

/// `freq[j] = 1 / theta^(2j/text_dim)` in F32, as Native builds the external
/// multimodal table (`rope_theta` default 1e6 for Qwen3-VL).
pub fn mrope_frequencies(text_dim: usize, theta: f64) -> Result<Vec<f32>, BackendError> {
    if text_dim == 0 || !text_dim.is_multiple_of(2) {
        return Err(start("text head dimension must be positive and even"));
    }
    if !(theta.is_finite() && theta > 0.0) {
        return Err(start("rope theta must be positive and finite"));
    }
    let mut out = Vec::with_capacity(text_dim / 2);
    for j in 0..text_dim / 2 {
        out.push((1.0f64 / theta.powf(2.0 * j as f64 / text_dim as f64)) as f32);
    }
    Ok(out)
}

/// Build the Native Qwen mRoPE position remap. Image spans are canonicalized by
/// token start. Text gaps advance all axes together. Returned arrays are
/// `[axis0, axis1, axis2]`, each of length `tokens`.
pub fn image_position_ids(
    tokens: usize,
    images: &[ImageGrid],
) -> Result<[Vec<u32>; 3], BackendError> {
    let mut ordered: Vec<&ImageGrid> = images.iter().collect();
    ordered.sort_unstable_by_key(|image| image.token_start);
    let mut axes: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
    for axis in &mut axes {
        axis.try_reserve_exact(tokens)
            .map_err(|_| start("multimodal position allocation"))?;
    }
    let mut cursor = 0usize;
    let mut next = 0u32;
    for image in ordered {
        let patches = image
            .frames
            .checked_mul(image.height)
            .and_then(|value| value.checked_mul(image.width))
            .ok_or_else(|| start("image grid overflow"))?;
        let merge_area = image
            .merge_size
            .checked_mul(image.merge_size)
            .filter(|value| *value > 0)
            .ok_or_else(|| start("invalid image merge size"))?;
        if patches == 0
            || image.height % image.merge_size != 0
            || image.width % image.merge_size != 0
            || !patches.is_multiple_of(merge_area)
        {
            return Err(start("invalid image grid for merge"));
        }
        let length = patches / merge_area;
        if length == 0 || image.token_count != length {
            return Err(start("image token count does not match merged geometry"));
        }
        let end = image
            .token_start
            .checked_add(length)
            .filter(|end| *end <= tokens)
            .ok_or_else(|| start("image token span exceeds token count"))?;
        if image.token_start < cursor {
            return Err(start("overlapping image token spans"));
        }
        for _ in cursor..image.token_start {
            for axis in &mut axes {
                axis.push(next);
            }
            next = next
                .checked_add(1)
                .ok_or_else(|| start("multimodal position overflow"))?;
        }
        let height = image.height / image.merge_size;
        let width = image.width / image.merge_size;
        for frame in 0..image.frames {
            let temporal = next
                .checked_add(u32::try_from(frame).map_err(start)?)
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
        let advance = height.max(width).max(image.frames);
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
        return Err(start("mRoPE axis/frequency length mismatch"));
    }
    let tokens = positions[0].len();
    let half_dim = axes.len();
    let total = tokens
        .checked_mul(half_dim)
        .ok_or_else(|| start("mRoPE phase size overflow"))?;
    let mut cos = Vec::with_capacity(total);
    let mut sin = Vec::with_capacity(total);
    for ((axis0, axis1), axis2) in positions[0]
        .iter()
        .zip(positions[1].iter())
        .zip(positions[2].iter())
    {
        for (j, axis) in axes.iter().enumerate() {
            let position = match axis {
                0 => *axis0,
                1 => *axis1,
                2 => *axis2,
                _ => return Err(start("invalid mRoPE axis")),
            };
            let angle = position as f32 * frequency[j];
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok(MropePhases {
        tokens,
        half_dim,
        cos,
        sin,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkerSpan {
    pub token_start: usize,
    pub token_count: usize,
}

pub fn marker_spans(images: &[ImageGrid], tokens: usize) -> Result<Vec<MarkerSpan>, BackendError> {
    let mut spans = Vec::with_capacity(images.len());
    let mut previous_end = 0usize;
    let mut ordered: Vec<&ImageGrid> = images.iter().collect();
    ordered.sort_unstable_by_key(|image| image.token_start);
    for image in ordered {
        let end = image
            .token_start
            .checked_add(image.token_count)
            .filter(|end| *end <= tokens)
            .ok_or_else(|| start("image marker span exceeds tokens"))?;
        if image.token_start < previous_end || image.token_count == 0 {
            return Err(start("overlapping or empty image marker span"));
        }
        spans.push(MarkerSpan {
            token_start: image.token_start,
            token_count: image.token_count,
        });
        previous_end = end;
    }
    Ok(spans)
}

/// Native uses the first `deepstack.len()` text layers in capture order.
pub fn deepstack_text_layers(
    text_layers: usize,
    deepstack_tensors: usize,
) -> Result<Vec<usize>, BackendError> {
    if deepstack_tensors > text_layers {
        return Err(start(format!(
            "{deepstack_tensors} deepstack tensors exceed {text_layers} text layers"
        )));
    }
    Ok((0..deepstack_tensors).collect())
}

/// Cached visual-prefix requests with no current images still receive one empty
/// placeholder per configured deepstack stage so the text suffix keeps its
/// original stage accounting.
pub fn empty_prefix_deepstack(
    config: &TextConfig,
    configured: usize,
) -> Result<Vec<[usize; 2]>, BackendError> {
    let layers = deepstack_text_layers(config.num_hidden_layers, configured)?;
    Ok(layers
        .into_iter()
        .map(|_| [0, config.hidden_size])
        .collect::<Vec<[usize; 2]>>())
}

#[derive(Clone, Debug)]
pub struct TextFusionPlan {
    pub tokens: usize,
    pub markers: Vec<MarkerSpan>,
    pub positions: [Vec<u32>; 3],
    pub axes: Vec<u8>,
    pub frequency: Vec<f32>,
    pub phases: MropePhases,
    pub deepstack_layers: Vec<usize>,
}

pub fn prepare_text_fusion(
    config: &TextConfig,
    tokens: usize,
    images: &[ImageGrid],
    deepstack_tensors: usize,
) -> Result<TextFusionPlan, BackendError> {
    if tokens == 0 {
        return Err(start("text fusion requires at least one token"));
    }
    let markers = marker_spans(images, tokens)?;
    let positions = if images.is_empty() {
        let plain: Vec<u32> = (0..tokens)
            .map(|token| u32::try_from(token).map_err(start))
            .collect::<Result<Vec<_>, _>>()?;
        [plain.clone(), plain.clone(), plain]
    } else {
        image_position_ids(tokens, images)?
    };
    let axes = mrope_axis_map(config.text_dim(), config.mrope_section)?;
    let frequency = mrope_frequencies(config.text_dim(), config.rope_theta)?;
    let phases = mrope_phases(&positions, &axes, &frequency)?;
    let deepstack_layers = deepstack_text_layers(config.num_hidden_layers, deepstack_tensors)?;
    Ok(TextFusionPlan {
        tokens,
        markers,
        positions,
        axes,
        frequency,
        phases,
        deepstack_layers,
    })
}
