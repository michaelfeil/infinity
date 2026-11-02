// Cold workspace planning for the Oxide Qwen3-VL tower and text fusion.
//
// These are Oxide-owned arena plans derived from the Native stage order. They
// are not a claim that the Native LibTorch allocator uses this layout. Every
// section is 256-byte aligned, every size is checked, and no allocation occurs.

use crate::checkpoint::start;
use crate::qwen3_vl_config::Qwen3VlConfig;
use crate::qwen3_vl_vision::ImageGrid;
use infinity_v3_backend_core::BackendError;
use serde::{Deserialize, Serialize};

pub const ALIGNMENT: u64 = 256;
const BF16: u64 = 2;
const F32: u64 = 4;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArenaSection {
    pub name: String,
    pub elements: u64,
    pub element_bytes: u64,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArenaPlan {
    pub sections: Vec<ArenaSection>,
    pub total_bytes: u64,
}

fn align_up(value: u64, alignment: u64) -> Result<u64, BackendError> {
    if alignment == 0 {
        return Err(start("alignment must be positive"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| start("alignment overflow"))
}

fn checked_mul(left: u64, right: u64, name: &str) -> Result<u64, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

struct Builder {
    sections: Vec<ArenaSection>,
    offset: u64,
}

impl Builder {
    fn new() -> Self {
        Self {
            sections: Vec::new(),
            offset: 0,
        }
    }

    fn push(&mut self, name: &str, elements: u64, element_bytes: u64) -> Result<(), BackendError> {
        let offset = align_up(self.offset, ALIGNMENT)?;
        let bytes = checked_mul(elements, element_bytes, name)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| start(format!("{name} end overflow")))?;
        self.sections.push(ArenaSection {
            name: name.to_owned(),
            elements,
            element_bytes,
            offset,
            bytes,
        });
        self.offset = end;
        Ok(())
    }

    fn finish(self) -> Result<ArenaPlan, BackendError> {
        let total_bytes = align_up(self.offset, ALIGNMENT)?;
        Ok(ArenaPlan {
            sections: self.sections,
            total_bytes,
        })
    }
}

fn u64_of(value: usize) -> Result<u64, BackendError> {
    u64::try_from(value).map_err(start)
}

pub fn vision_arena(
    config: &Qwen3VlConfig,
    images: &[ImageGrid],
) -> Result<ArenaPlan, BackendError> {
    let vision = &config.vision;
    let hidden = u64_of(vision.hidden_size)?;
    let dim = u64_of(vision.head_dim())?;
    let intermediate = u64_of(vision.intermediate_size)?;
    let grouped = checked_mul(hidden, u64_of(vision.merge_area()?)?, "merged vision width")?;
    let patch_dim = u64_of(vision.patch_dim()?)?;
    let mut patch_rows = 0u64;
    let mut merged_rows = 0u64;
    for image in images {
        let rows = u64::try_from(image.patch_rows()?).map_err(start)?;
        patch_rows = patch_rows
            .checked_add(rows)
            .ok_or_else(|| start("vision patch rows overflow"))?;
        merged_rows = merged_rows
            .checked_add(u64_of(image.token_count)?)
            .ok_or_else(|| start("vision merged rows overflow"))?;
    }
    let deepstack = u64_of(vision.deepstack_visual_indexes.len())?;
    let mut builder = Builder::new();
    builder.push(
        "pixels_f32",
        checked_mul(patch_rows, patch_dim, "pixels")?,
        F32,
    )?;
    builder.push(
        "states_bf16",
        checked_mul(patch_rows, hidden, "states")?,
        BF16,
    )?;
    builder.push("norm_bf16", checked_mul(patch_rows, hidden, "norm")?, BF16)?;
    builder.push(
        "qkv_bf16",
        checked_mul(patch_rows, checked_mul(hidden, 3, "qkv width")?, "qkv")?,
        BF16,
    )?;
    builder.push("q_f32", checked_mul(patch_rows, hidden, "q")?, F32)?;
    builder.push("k_f32", checked_mul(patch_rows, hidden, "k")?, F32)?;
    builder.push("v_f32", checked_mul(patch_rows, hidden, "v")?, F32)?;
    builder.push(
        "vision_cos_f32",
        checked_mul(patch_rows, dim / 2, "vision cos")?,
        F32,
    )?;
    builder.push(
        "vision_sin_f32",
        checked_mul(patch_rows, dim / 2, "vision sin")?,
        F32,
    )?;
    builder.push(
        "attention_bf16",
        checked_mul(patch_rows, hidden, "attention")?,
        BF16,
    )?;
    builder.push(
        "projected_bf16",
        checked_mul(patch_rows, hidden, "projected")?,
        BF16,
    )?;
    builder.push(
        "mlp_fc1_bf16",
        checked_mul(patch_rows, intermediate, "mlp fc1")?,
        BF16,
    )?;
    builder.push(
        "mlp_fc2_bf16",
        checked_mul(patch_rows, hidden, "mlp fc2")?,
        BF16,
    )?;
    builder.push(
        "merger_group_bf16",
        checked_mul(merged_rows, grouped, "merger group")?,
        BF16,
    )?;
    builder.push(
        "merger_fc1_bf16",
        checked_mul(merged_rows, grouped, "merger fc1")?,
        BF16,
    )?;
    builder.push(
        "merged_features_bf16",
        checked_mul(
            merged_rows,
            u64_of(vision.out_hidden_size)?,
            "merged features",
        )?,
        BF16,
    )?;
    if deepstack > 0 {
        let deep_elements = checked_mul(
            checked_mul(
                merged_rows,
                u64_of(vision.out_hidden_size)?,
                "deepstack width",
            )?,
            deepstack,
            "deepstack",
        )?;
        builder.push("deepstack_bf16", deep_elements, BF16)?;
    }
    builder.finish()
}

pub fn text_arena(config: &Qwen3VlConfig, tokens: usize) -> Result<ArenaPlan, BackendError> {
    if tokens == 0 {
        return Err(start("text arena requires at least one token"));
    }
    let text = &config.text;
    let tokens = u64_of(tokens)?;
    let hidden = u64_of(text.hidden_size)?;
    let head_dim = u64_of(text.text_dim())?;
    let q_width = u64_of(text.q_width()?)?;
    let kv_width = u64_of(text.kv_width()?)?;
    let intermediate = u64_of(text.intermediate_size)?;
    let mut builder = Builder::new();
    builder.push("embed_bf16", checked_mul(tokens, hidden, "embed")?, BF16)?;
    builder.push(
        "residual_bf16",
        checked_mul(tokens, hidden, "residual")?,
        BF16,
    )?;
    builder.push("norm_bf16", checked_mul(tokens, hidden, "text norm")?, BF16)?;
    let packed_qkv_width = q_width
        .checked_add(
            kv_width
                .checked_mul(2)
                .ok_or_else(|| start("text qkv width overflow"))?,
        )
        .ok_or_else(|| start("text qkv width overflow"))?;
    builder.push(
        "qkv_bf16",
        checked_mul(tokens, packed_qkv_width, "text qkv")?,
        BF16,
    )?;
    builder.push("q_bf16", checked_mul(tokens, q_width, "q")?, BF16)?;
    builder.push("k_bf16", checked_mul(tokens, kv_width, "k")?, BF16)?;
    builder.push("v_bf16", checked_mul(tokens, kv_width, "v")?, BF16)?;
    builder.push("q_norm_bf16", checked_mul(tokens, q_width, "q norm")?, BF16)?;
    builder.push(
        "k_norm_bf16",
        checked_mul(tokens, kv_width, "k norm")?,
        BF16,
    )?;
    builder.push(
        "mrope_cos_f32",
        checked_mul(tokens, head_dim / 2, "mrope cos")?,
        F32,
    )?;
    builder.push(
        "mrope_sin_f32",
        checked_mul(tokens, head_dim / 2, "mrope sin")?,
        F32,
    )?;
    builder.push(
        "attention_bf16",
        checked_mul(tokens, q_width, "text attention")?,
        BF16,
    )?;
    builder.push(
        "mlp_bf16",
        checked_mul(
            tokens,
            checked_mul(intermediate, 2, "text mlp width")?,
            "text mlp",
        )?,
        BF16,
    )?;
    builder.push("final_bf16", checked_mul(tokens, hidden, "final")?, BF16)?;
    builder.finish()
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct FusionLayout {
    pub vision: Option<ArenaPlan>,
    pub text: ArenaPlan,
    pub marker_rows: u64,
    pub deepstack_rows: u64,
}

pub fn fusion_layout(
    config: &Qwen3VlConfig,
    tokens: usize,
    images: &[ImageGrid],
    include_vision: bool,
) -> Result<FusionLayout, BackendError> {
    let vision = if include_vision {
        Some(vision_arena(config, images)?)
    } else {
        None
    };
    let marker_rows = images.iter().try_fold(0u64, |total, image| {
        total
            .checked_add(u64_of(image.token_count)?)
            .ok_or_else(|| start("marker row count overflow"))
    })?;
    let deepstack_rows = if include_vision {
        marker_rows
            .checked_mul(u64_of(config.vision.deepstack_visual_indexes.len())?)
            .ok_or_else(|| start("deepstack row count overflow"))?
    } else {
        0
    };
    Ok(FusionLayout {
        vision,
        text: text_arena(config, tokens)?,
        marker_rows,
        deepstack_rows,
    })
}
