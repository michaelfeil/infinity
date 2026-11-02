// Actual host reference forward for the Qwen3.5 vision tower.
//!
//! This is not a shape-admission module: it evaluates the complete Native tower
//! over packed patch rows and returns the merged visual rows that the text
//! decoder consumes. The arithmetic follows
//! `backends/libtorch/cpp/models/qwen3_vl_models.cpp` (the tower is shared) with
//! the Qwen3.5-specific fused temporal projection:
//!
//! * patch projection is two `K = 3*16*16` GEMMs over temporal slices 0 and 1,
//!   a rounded BF16 sum, then one BF16 bias;
//! * learned positions are four-corner interpolated from the square
//!   `pos_embed.weight` table and summed in model dtype;
//! * vision rotary is 2-D split-half on `(row, column)` phases;
//! * each frame is an independent non-causal attention sequence;
//! * merger grouping is merge-group-major, final merger normalizes per hidden
//!   row before grouping, deepstack mergers normalize the grouped row and are
//!   absent for published Qwen3.5 roots (`deepstack_visual_indexes: []`).
//!
//! The same row layout is consumed by `qwen35_vl_device` for the device path;
//! sharing `HostImage`/`VisionOutput` keeps CPU tests and device execution on
//! one geometry contract.

use crate::model::Checkpoint;
use crate::qwen35_vl_config::Qwen35VlConfig;
use crate::qwen35_vl_inventory::expected_vision_tensors;
use crate::qwen35_vl_media::MediaGrid;
use half::bf16;
use infinity_v3_backend_core::BackendError;
use std::collections::BTreeMap;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl vision: {error}"))
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start("dimension overflow"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, BackendError> {
        let expected = shape.iter().try_fold(1usize, |total, dim| {
            total
                .checked_mul(*dim)
                .ok_or_else(|| start("tensor shape overflow"))
        })?;
        if expected != data.len() {
            return Err(start(format!(
                "tensor shape {shape:?} needs {expected} values, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisionHostWeights {
    tensors: BTreeMap<String, HostTensor>,
}

impl VisionHostWeights {
    pub fn insert(&mut self, name: impl Into<String>, tensor: HostTensor) {
        self.tensors.insert(name.into(), tensor);
    }

    pub fn get(&self, suffix: &str) -> Result<&HostTensor, BackendError> {
        self.tensors
            .get(&format!("model.visual.{suffix}"))
            .or_else(|| self.tensors.get(&format!("visual.{suffix}")))
            .or_else(|| self.tensors.get(suffix))
            .ok_or_else(|| start(format!("missing vision tensor {suffix}")))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Read every tower tensor from a cached original checkpoint. Payload is
    /// read in one bounded `read_bf16` per tensor; the checkpoint is not
    /// modified. This is the CPU reference loader used by Root's paired run.
    pub fn from_checkpoint(
        path: &std::path::Path,
        config: &Qwen35VlConfig,
    ) -> Result<Self, BackendError> {
        let checkpoint = Checkpoint::open(path)?;
        let mut weights = Self::default();
        for tensor in expected_vision_tensors(config)? {
            let source = checkpoint.source(&tensor.name, &tensor.shape)?;
            let count: usize = tensor.shape.iter().try_fold(1usize, |total, dim| {
                total
                    .checked_mul(*dim)
                    .ok_or_else(|| start("tensor shape overflow"))
            })?;
            let mut bits = vec![0u16; count];
            source.read_bf16(0, &mut bits)?;
            let data = bits
                .into_iter()
                .map(|bits| half::bf16::from_bits(bits).to_f32())
                .collect();
            weights.insert(
                tensor.name.trim_start_matches("model.visual.").to_owned(),
                HostTensor::new(tensor.shape, data)?,
            );
        }
        Ok(weights)
    }
}

#[derive(Clone, Debug)]
pub struct HostImage {
    pub name: String,
    /// Packed rows in `[rows, channels * temporal_patch * patch * patch]` order.
    pub pixels: Vec<f32>,
    pub grid: MediaGrid,
    pub token_start: usize,
}

impl HostImage {
    pub fn patch_rows(&self) -> Result<usize, BackendError> {
        self.grid.patch_rows()
    }

    pub fn merged_rows(&self) -> Result<usize, BackendError> {
        self.grid.merged_tokens()
    }
}

#[derive(Clone, Debug)]
pub struct VisionOutput {
    pub merged: Vec<f32>,
    pub merged_rows: usize,
    pub out_hidden: usize,
    pub deepstack: Vec<Vec<f32>>,
}

impl VisionOutput {
    pub fn image_rows(&self, images: &[HostImage]) -> Result<Vec<(usize, usize)>, BackendError> {
        let mut slices = Vec::with_capacity(images.len());
        let mut offset = 0usize;
        for image in images {
            let count = image.merged_rows()?;
            slices.push((offset, count));
            offset = offset
                .checked_add(count)
                .ok_or_else(|| start("merged offset overflow"))?;
        }
        if offset != self.merged_rows {
            return Err(start("merged row accounting mismatch"));
        }
        Ok(slices)
    }
}

#[inline]
fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

#[inline]
fn bf16_value(value: f32) -> f32 {
    round_bf16(value)
}

#[inline]
fn gelu_tanh(value: f32) -> f32 {
    // torch.nn.functional.gelu(..., approximate="tanh")
    let inner = 0.797_884_6f32 * (value + 0.044_715f32 * value * value * value);
    let half = 0.5f32 * value;
    round_bf16(half * (1.0 + inner.tanh()))
}

#[inline]
fn silu(value: f32) -> f32 {
    round_bf16(value / (1.0 + (-value).exp()))
}

fn linear(
    input: &[f32],
    rows: usize,
    in_width: usize,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    round_input: bool,
    exact_bias: bool,
) -> Result<Vec<f32>, BackendError> {
    if weight.shape.len() != 2 || weight.shape[1] != in_width {
        return Err(start(format!(
            "linear weight shape {:?} does not accept width {in_width}",
            weight.shape
        )));
    }
    let out_width = weight.shape[0];
    if input.len() != product(rows, in_width)? {
        return Err(start("linear input length mismatch"));
    }
    let mut output = vec![0f32; product(rows, out_width)?];
    for row in 0..rows {
        let source = &input[row * in_width..(row + 1) * in_width];
        for unit in 0..out_width {
            let weight_row = &weight.data[unit * in_width..(unit + 1) * in_width];
            let mut total = 0f32;
            for column in 0..in_width {
                let value = if round_input {
                    bf16_value(source[column])
                } else {
                    source[column]
                };
                total += value * weight_row[column];
            }
            let value = match bias {
                Some(bias) => {
                    if bias.data.len() != out_width {
                        return Err(start("linear bias length mismatch"));
                    }
                    if exact_bias {
                        round_bf16(round_bf16(total) + bf16_value(bias.data[unit]))
                    } else {
                        round_bf16(total + bf16_value(bias.data[unit]))
                    }
                }
                None => round_bf16(total),
            };
            output[row * out_width + unit] = value;
        }
    }
    Ok(output)
}

fn layer_norm(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &HostTensor,
    bias: &HostTensor,
    eps: f32,
) -> Result<Vec<f32>, BackendError> {
    if weight.data.len() != width || bias.data.len() != width {
        return Err(start("layer norm affine width mismatch"));
    }
    if input.len() != product(rows, width)? {
        return Err(start("layer norm input length mismatch"));
    }
    let mut output = vec![0f32; input.len()];
    for row in 0..rows {
        let values = &input[row * width..(row + 1) * width];
        let mean = values.iter().sum::<f32>() / width as f32;
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + eps).sqrt();
        for column in 0..width {
            let normalized = (values[column] - mean) * inverse;
            output[row * width + column] = round_bf16(
                normalized * bf16_value(weight.data[column]) + bf16_value(bias.data[column]),
            );
        }
    }
    Ok(output)
}

fn residual_add(left: &[f32], right: &[f32]) -> Result<Vec<f32>, BackendError> {
    if left.len() != right.len() {
        return Err(start("residual add length mismatch"));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| round_bf16(bf16_value(*left) + bf16_value(*right)))
        .collect())
}

fn position_interpolation(
    table: &HostTensor,
    side: usize,
    grid: MediaGrid,
    rows: usize,
    hidden: usize,
    exact: bool,
) -> Result<Vec<f32>, BackendError> {
    if table.shape != [side * side, hidden] || table.data.len() != side * side * hidden {
        return Err(start(
            "pos_embed.weight must be a square [side*side, hidden] table",
        ));
    }
    let row_scale = if grid.height == 1 {
        0.0
    } else {
        (side - 1) as f32 / (grid.height - 1) as f32
    };
    let column_scale = if grid.width == 1 {
        0.0
    } else {
        (side - 1) as f32 / (grid.width - 1) as f32
    };
    let mut output = vec![0f32; product(rows, hidden)?];
    for index in 0..product(grid.height, grid.width)? {
        let (row, column) = grid.merge_major_coords(index)?;
        let row_f = row as f32 * row_scale;
        let column_f = column as f32 * column_scale;
        let r0 = row_f.floor() as usize;
        let c0 = column_f.floor() as usize;
        let r1 = (row_f.ceil() as usize).min(side - 1);
        let c1 = (column_f.ceil() as usize).min(side - 1);
        let dr = row_f - r0 as f32;
        let dc = column_f - c0 as f32;
        for hidden_column in 0..hidden {
            let corner = |r: usize, c: usize, scale: f32| -> f32 {
                let value = table.data[(r * side + c) * hidden + hidden_column];
                round_bf16(value * bf16_value(scale))
            };
            let positional = if exact {
                let first = round_bf16(
                    corner(r0, c0, (1.0 - dr) * (1.0 - dc)) + corner(r1, c0, dr * (1.0 - dc)),
                );
                let second = round_bf16(corner(r0, c1, (1.0 - dr) * dc) + corner(r1, c1, dr * dc));
                round_bf16(first + second)
            } else {
                let values = [
                    corner(r0, c0, (1.0 - dr) * (1.0 - dc)),
                    corner(r0, c1, (1.0 - dr) * dc),
                    corner(r1, c0, dr * (1.0 - dc)),
                    corner(r1, c1, dr * dc),
                ];
                round_bf16(values[0] + values[1] + values[2] + values[3])
            };
            output[index * hidden + hidden_column] = positional;
        }
    }
    Ok(output)
}

fn vision_frequency(dim: usize) -> Result<Vec<f32>, BackendError> {
    if dim == 0 || !dim.is_multiple_of(4) {
        return Err(start("vision head dimension must be divisible by 4"));
    }
    let mut out = Vec::with_capacity(dim / 4);
    for index in 0..dim / 4 {
        out.push(1.0f32 / 10_000f32.powf(2.0 * index as f32 / (dim as f32 / 2.0)));
    }
    Ok(out)
}

fn rotate_vision(
    states: &mut [f32],
    rows: usize,
    heads: usize,
    dim: usize,
    grid: MediaGrid,
) -> Result<(), BackendError> {
    if !dim.is_multiple_of(4) {
        return Err(start("vision head dimension must be divisible by 4"));
    }
    let frequency = vision_frequency(dim)?;
    let frame_rows = product(grid.height, grid.width)?;
    for row in 0..rows {
        let frame = row / frame_rows;
        let spatial = row % frame_rows;
        let (pixel_row, pixel_column) = grid.merge_major_coords(spatial)?;
        let _ = frame;
        let row_phase = pixel_row as f32;
        let column_phase = pixel_column as f32;
        for head in 0..heads {
            let base = (row * heads + head) * dim;
            let half = dim / 2;
            for index in 0..half {
                let angle = if index < half / 2 {
                    row_phase * frequency[index]
                } else {
                    column_phase * frequency[index - half / 2]
                };
                let (sin, cos) = angle.sin_cos();
                let a = states[base + index];
                let b = states[base + half + index];
                states[base + index] = round_bf16(a * cos - b * sin);
                states[base + half + index] = round_bf16(b * cos + a * sin);
            }
        }
    }
    Ok(())
}

fn attention(
    qkv: &[f32],
    rows: usize,
    heads: usize,
    dim: usize,
    grid: MediaGrid,
) -> Result<Vec<f32>, BackendError> {
    let width = product(heads, dim)?;
    if qkv.len() != product(rows, product(width, 3)?)? {
        return Err(start("qkv length mismatch"));
    }
    let frame_rows = product(grid.height, grid.width)?;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut output = vec![0f32; product(rows, width)?];
    for frame_start in (0..rows).step_by(frame_rows) {
        let frame_end = (frame_start + frame_rows).min(rows);
        for head in 0..heads {
            let mut scores = vec![0f32; frame_rows * frame_rows];
            for query in 0..frame_end - frame_start {
                let query_base = (frame_start + query) * (width * 3) + head * dim;
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..frame_end - frame_start {
                    let key_base = (frame_start + key) * (width * 3) + width + head * dim;
                    let mut dot = 0f32;
                    for index in 0..dim {
                        dot += qkv[query_base + index] * qkv[key_base + index];
                    }
                    let value = dot * scale;
                    scores[query * frame_rows + key] = value;
                    maximum = maximum.max(value);
                }
                let mut denominator = 0f32;
                for key in 0..frame_end - frame_start {
                    let value = (scores[query * frame_rows + key] - maximum).exp();
                    scores[query * frame_rows + key] = value;
                    denominator += value;
                }
                for key in 0..frame_end - frame_start {
                    let probability = scores[query * frame_rows + key] / denominator;
                    let value_base = (frame_start + key) * (width * 3) + width * 2 + head * dim;
                    for index in 0..dim {
                        output[(frame_start + query) * width + head * dim + index] +=
                            probability * qkv[value_base + index];
                    }
                }
            }
        }
    }
    for value in &mut output {
        *value = round_bf16(*value);
    }
    Ok(output)
}

fn grouped_view(
    input: &[f32],
    hidden: usize,
    merge: usize,
    rows: usize,
) -> Result<Vec<f32>, BackendError> {
    let area = product(merge, merge)?;
    let grouped_width = product(hidden, area)?;
    if !rows.is_multiple_of(area) || input.len() != product(rows, hidden)? {
        return Err(start("merged row count is not divisible by merge area"));
    }
    // Merge-major rows are already contiguous in group order, so the group
    // view is the identity permutation.
    let _ = grouped_width;
    Ok(input.to_vec())
}

pub fn vision_forward(
    config: &Qwen35VlConfig,
    weights: &VisionHostWeights,
    images: &[HostImage],
) -> Result<VisionOutput, BackendError> {
    if images.is_empty() {
        return Err(start("empty Qwen3.5 image batch"));
    }
    let vision = &config.vision;
    let hidden = vision.hidden_size;
    let heads = vision.num_heads;
    let dim = vision.head_dim();
    let merge = vision.spatial_merge_size;
    let out_hidden = vision.out_hidden_size;
    let patch_dim = vision.patch_dim()?;
    let side = vision.position_side();
    let eps = 1e-6f32;
    let exact = config.text.dtype == crate::qwen35_vl_config::Dtype::F16
        && config.model_type.starts_with("qwen3_5");

    let mut merged_rows = 0usize;
    for image in images {
        if image.grid.merge != merge {
            return Err(start("image merge size does not match config"));
        }
        let rows = image.patch_rows()?;
        if image.pixels.len() != product(rows, patch_dim)? {
            return Err(start("packed pixel payload length mismatch"));
        }
        merged_rows = merged_rows
            .checked_add(image.merged_rows()?)
            .ok_or_else(|| start("merged row overflow"))?;
    }
    let mut merged = vec![0f32; product(merged_rows, out_hidden)?];
    let mut deepstack =
        vec![vec![0f32; product(merged_rows, out_hidden)?]; vision.deepstack_visual_indexes.len()];
    let mut merged_offset = 0usize;

    let patch_weight = weights.get("patch_embed.proj.weight")?;
    if patch_weight.shape
        != [
            hidden,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ]
    {
        return Err(start(format!(
            "patch_embed.proj.weight must be [hidden, channels, temporal, patch, patch], got {:?}",
            patch_weight.shape
        )));
    }
    let patch_bias = weights.get("patch_embed.proj.bias")?;
    if patch_bias.data.len() != hidden {
        return Err(start("patch_embed.proj.bias width mismatch"));
    }
    let position_table = weights.get("pos_embed.weight")?;
    let frequency = vision_frequency(dim)?;
    let mut capture = Vec::new();
    for (name_index, layer) in vision.deepstack_visual_indexes.iter().enumerate() {
        capture.push((*layer, name_index));
    }
    capture.sort_unstable_by_key(|(layer, _)| *layer);

    for image in images {
        let rows = image.patch_rows()?;
        let grid = image.grid;
        // Native BF16 uses one flattened Conv3D GEMM; the F16 qwen3_5 exact path
        // splits the two temporal slices into separate GEMMs with a rounded sum.
        let mut states = vec![0f32; product(rows, hidden)?];
        if exact {
            for slice in 0..vision.temporal_patch_size {
                for row in 0..rows {
                    for unit in 0..hidden {
                        let mut total = 0f32;
                        for channel in 0..vision.in_channels {
                            for y in 0..vision.patch_size {
                                for x in 0..vision.patch_size {
                                    let input_index = ((channel * vision.temporal_patch_size
                                        + slice)
                                        * vision.patch_size
                                        + y)
                                        * vision.patch_size
                                        + x;
                                    let weight_index = ((((unit * vision.in_channels + channel)
                                        * vision.temporal_patch_size
                                        + slice)
                                        * vision.patch_size
                                        + y)
                                        * vision.patch_size)
                                        + x;
                                    total +=
                                        bf16_value(image.pixels[row * patch_dim + input_index])
                                            * patch_weight.data[weight_index];
                                }
                            }
                        }
                        states[row * hidden + unit] =
                            round_bf16(states[row * hidden + unit] + round_bf16(total));
                    }
                }
            }
            for row in 0..rows {
                for unit in 0..hidden {
                    states[row * hidden + unit] =
                        round_bf16(states[row * hidden + unit] + bf16_value(patch_bias.data[unit]));
                }
            }
        } else {
            for row in 0..rows {
                for unit in 0..hidden {
                    let mut total = 0f32;
                    for element in 0..patch_dim {
                        total += bf16_value(image.pixels[row * patch_dim + element])
                            * patch_weight.data[unit * patch_dim + element];
                    }
                    states[row * hidden + unit] =
                        round_bf16(total + bf16_value(patch_bias.data[unit]));
                }
            }
        }
        let positional = position_interpolation(position_table, side, grid, rows, hidden, exact)?;
        for (state, position) in states.iter_mut().zip(&positional) {
            *state = round_bf16(bf16_value(*state) + *position);
        }

        for layer in 0..vision.depth {
            let prefix = format!("blocks.{layer}.");
            let norm1_weight = weights.get(&format!("{prefix}norm1.weight"))?;
            let norm1_bias = weights.get(&format!("{prefix}norm1.bias"))?;
            let normalized = layer_norm(&states, rows, hidden, norm1_weight, norm1_bias, eps)?;
            let qkv_weight = weights.get(&format!("{prefix}attn.qkv.weight"))?;
            if qkv_weight.shape != [product(hidden, 3)?, hidden] {
                return Err(start("attn.qkv.weight shape mismatch"));
            }
            let qkv_bias = weights.get(&format!("{prefix}attn.qkv.bias"))?;
            let mut qkv = linear(
                &normalized,
                rows,
                hidden,
                qkv_weight,
                Some(qkv_bias),
                true,
                exact,
            )?;
            rotate_vision(&mut qkv, rows, heads, dim, grid)?;
            let attended = attention(&qkv, rows, heads, dim, grid)?;
            let proj_weight = weights.get(&format!("{prefix}attn.proj.weight"))?;
            let proj_bias = weights.get(&format!("{prefix}attn.proj.bias"))?;
            let projected = linear(
                &attended,
                rows,
                hidden,
                proj_weight,
                Some(proj_bias),
                true,
                exact,
            )?;
            states = residual_add(&states, &projected)?;

            let norm2_weight = weights.get(&format!("{prefix}norm2.weight"))?;
            let norm2_bias = weights.get(&format!("{prefix}norm2.bias"))?;
            let normalized = layer_norm(&states, rows, hidden, norm2_weight, norm2_bias, eps)?;
            let fc1_weight = weights.get(&format!("{prefix}mlp.linear_fc1.weight"))?;
            let fc1_bias = weights.get(&format!("{prefix}mlp.linear_fc1.bias"))?;
            let mut hidden_values = linear(
                &normalized,
                rows,
                hidden,
                fc1_weight,
                Some(fc1_bias),
                true,
                exact,
            )?;
            match vision.hidden_act.as_str() {
                "silu" => {
                    for value in &mut hidden_values {
                        *value = silu(*value);
                    }
                }
                "gelu" => {
                    for value in &mut hidden_values {
                        *value = round_bf16(*value);
                    }
                }
                _ => {
                    for value in &mut hidden_values {
                        *value = gelu_tanh(*value);
                    }
                }
            }
            let fc2_weight = weights.get(&format!("{prefix}mlp.linear_fc2.weight"))?;
            let fc2_bias = weights.get(&format!("{prefix}mlp.linear_fc2.bias"))?;
            let down = linear(
                &hidden_values,
                rows,
                vision.intermediate_size,
                fc2_weight,
                Some(fc2_bias),
                true,
                exact,
            )?;
            states = residual_add(&states, &down)?;

            if let Some((_, name_index)) = capture
                .iter()
                .find(|(layer_index, _)| *layer_index == layer)
            {
                let rows_now = image.merged_rows()?;
                let grouped = grouped_view(&states, hidden, merge, rows)?;
                let post = true;
                if grouped.len() != product(rows_now, product(hidden, merge * merge)?)? {
                    return Err(start("deepstack grouped width mismatch"));
                }
                let normalized = if post {
                    let norm_weight =
                        weights.get(&format!("deepstack_merger_list.{name_index}.norm.weight"))?;
                    let norm_bias =
                        weights.get(&format!("deepstack_merger_list.{name_index}.norm.bias"))?;
                    layer_norm(
                        &grouped,
                        rows_now,
                        product(hidden, merge * merge)?,
                        norm_weight,
                        norm_bias,
                        eps,
                    )?
                } else {
                    unreachable!()
                };
                let fc1_weight = weights.get(&format!(
                    "deepstack_merger_list.{name_index}.linear_fc1.weight"
                ))?;
                let fc1_bias = weights.get(&format!(
                    "deepstack_merger_list.{name_index}.linear_fc1.bias"
                ))?;
                let mut mid = linear(
                    &normalized,
                    rows_now,
                    product(hidden, merge * merge)?,
                    fc1_weight,
                    Some(fc1_bias),
                    true,
                    exact,
                )?;
                for value in &mut mid {
                    *value = gelu_tanh(*value);
                }
                let fc2_weight = weights.get(&format!(
                    "deepstack_merger_list.{name_index}.linear_fc2.weight"
                ))?;
                let fc2_bias = weights.get(&format!(
                    "deepstack_merger_list.{name_index}.linear_fc2.bias"
                ))?;
                let out = linear(
                    &mid,
                    rows_now,
                    product(hidden, merge * merge)?,
                    fc2_weight,
                    Some(fc2_bias),
                    true,
                    exact,
                )?;
                let start = merged_offset * out_hidden;
                let end = start + out.len();
                deepstack[*name_index][start..end].copy_from_slice(&out);
            }
        }

        let rows_now = image.merged_rows()?;
        let norm_weight = weights.get("merger.norm.weight")?;
        let norm_bias = weights.get("merger.norm.bias")?;
        let normalized = layer_norm(&states, rows, hidden, norm_weight, norm_bias, eps)?;
        let grouped_width = product(hidden, merge * merge)?;
        let grouped = grouped_view(&normalized, hidden, merge, rows)?;
        let fc1_weight = weights.get("merger.linear_fc1.weight")?;
        let fc1_bias = weights.get("merger.linear_fc1.bias")?;
        let mut mid = linear(
            &grouped,
            rows_now,
            grouped_width,
            fc1_weight,
            Some(fc1_bias),
            true,
            exact,
        )?;
        for value in &mut mid {
            *value = gelu_tanh(*value);
        }
        let fc2_weight = weights.get("merger.linear_fc2.weight")?;
        let fc2_bias = weights.get("merger.linear_fc2.bias")?;
        let out = linear(
            &mid,
            rows_now,
            grouped_width,
            fc2_weight,
            Some(fc2_bias),
            true,
            exact,
        )?;
        let start = merged_offset * out_hidden;
        let end = start + out.len();
        merged[start..end].copy_from_slice(&out);
        merged_offset += rows_now;
        let _ = frequency;
    }
    if merged_offset != merged_rows {
        return Err(start("merged row accounting mismatch"));
    }
    Ok(VisionOutput {
        merged,
        merged_rows,
        out_hidden,
        deepstack,
    })
}

#[cfg(test)]
mod native_position_recipe_tests {
    use super::{position_interpolation, HostTensor};
    use crate::qwen35_vl_media::MediaGrid;

    fn bf(value: f32) -> f32 {
        half::bf16::from_f32(value).to_f32()
    }

    #[test]
    fn position_interpolation_matches_native_scale_bf16_and_vertical_grouping() {
        let side = 2usize;
        let hidden = 8usize;
        let table = HostTensor::new(
            vec![side * side, hidden],
            (0..side * side * hidden)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.125)
                .collect(),
        )
        .unwrap();
        let grid = MediaGrid::new(1, 4, 4, 2).unwrap();
        let actual = position_interpolation(&table, side, grid, 16, hidden, true).unwrap();
        let row_scale = (side - 1) as f32 / (grid.height - 1) as f32;
        let column_scale = (side - 1) as f32 / (grid.width - 1) as f32;
        for index in 0..grid.height * grid.width {
            let (row, column) = grid.merge_major_coords(index).unwrap();
            let rf = row as f32 * row_scale;
            let cf = column as f32 * column_scale;
            let r0 = rf.floor() as usize;
            let c0 = cf.floor() as usize;
            let r1 = (rf.ceil() as usize).min(side - 1);
            let c1 = (cf.ceil() as usize).min(side - 1);
            let dr = rf - r0 as f32;
            let dc = cf - c0 as f32;
            for h in 0..hidden {
                let corner = |r: usize, c: usize, scale: f32| {
                    bf(table.data[(r * side + c) * hidden + h] * bf(scale))
                };
                let first =
                    bf(corner(r0, c0, (1.0 - dr) * (1.0 - dc)) + corner(r1, c0, dr * (1.0 - dc)));
                let second = bf(corner(r0, c1, (1.0 - dr) * dc) + corner(r1, c1, dr * dc));
                let expected = bf(first + second);
                assert_eq!(
                    actual[index * hidden + h].to_bits(),
                    expected.to_bits(),
                    "token {index} hidden {h}"
                );
            }
        }
    }
}
