// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use crate::qwen35_config::{Qwen35Config, Qwen35LayerType};
use infinity_v3_backend_core::BackendError;
use std::ops::Range;

const BF16_BYTES: usize = 2;
const LINEAR_HEAD_DIM: usize = 128;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5 layout: {error}"))
}

fn product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CopySpan {
    pub(crate) offset: usize,
    pub(crate) elements: usize,
    pub(crate) row_width: usize,
}

impl CopySpan {
    pub(crate) fn new(
        offset: usize,
        elements: usize,
        row_width: usize,
    ) -> Result<Self, BackendError> {
        if elements != 0 && row_width == 0 {
            return Err(start("copy span has elements but no row width"));
        }
        if row_width != 0 && !elements.is_multiple_of(row_width) {
            return Err(start("copy span elements are not whole rows"));
        }
        add(offset, elements, "copy span end")?;
        Ok(Self {
            offset,
            elements,
            row_width,
        })
    }

    pub(crate) const fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) const fn elements(&self) -> usize {
        self.elements
    }

    pub(crate) const fn row_width(&self) -> usize {
        self.row_width
    }

    pub(crate) fn rows(&self) -> usize {
        if self.row_width == 0 {
            0
        } else {
            self.elements / self.row_width
        }
    }

    pub(crate) fn byte_offset(&self) -> Result<usize, BackendError> {
        product(self.offset, BF16_BYTES, "copy span byte offset")
    }

    pub(crate) fn byte_len(&self) -> Result<usize, BackendError> {
        product(self.elements, BF16_BYTES, "copy span byte length")
    }

    pub(crate) fn byte_range(&self) -> Result<Range<usize>, BackendError> {
        let start = self.byte_offset()?;
        let end = add(start, self.byte_len()?, "copy span byte end")?;
        Ok(start..end)
    }

    pub(crate) fn fits(&self, slab_elements: usize) -> Result<(), BackendError> {
        let end = add(self.offset, self.elements, "copy span end")?;
        if end > slab_elements {
            return Err(start("copy span exceeds its slab"));
        }
        Ok(())
    }

    pub(crate) fn row_span(&self, rows: usize) -> Result<Self, BackendError> {
        if rows > self.rows() {
            return Err(start("copy rows exceed the capacity span"));
        }
        let elements = product(rows, self.row_width, "copy span row elements")?;
        Self::new(self.offset, elements, self.row_width)
    }

    pub(crate) fn validated_ptr(&self, base: *mut u8) -> Result<*mut u8, BackendError> {
        let offset = self.byte_offset()?;
        add(offset, self.byte_len()?, "copy span byte end")?;
        Ok(base.wrapping_add(offset))
    }

    pub(crate) fn validated_const_ptr(&self, base: *const u8) -> Result<*const u8, BackendError> {
        let offset = self.byte_offset()?;
        add(offset, self.byte_len()?, "copy span byte end")?;
        Ok(base.wrapping_add(offset))
    }
}

macro_rules! scalar_accessors {
    ($($name:ident),* $(,)?) => {
        $(
            pub(crate) const fn $name(&self) -> usize {
                self.$name
            }
        )*
    };
}

macro_rules! span_accessors {
    ($($name:ident),* $(,)?) => {
        $(
            pub(crate) const fn $name(&self) -> CopySpan {
                self.$name
            }
        )*
    };
}

/// Minimal attention-phase slab geometry. The dense layout and the MoE
/// attention-only layout both implement it, so the shared prepared attention
/// path never depends on a dense FFN width or on synthetic MLP weights.
pub(crate) trait AttentionSlabLayout {
    fn capacity_tokens(&self) -> usize;
    fn hidden_elements(&self) -> usize;
    fn projection_elements(&self) -> usize;
    fn activation_elements(&self) -> usize;
    fn slab_bytes(&self) -> usize;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35Layout {
    pub(crate) capacity_tokens: usize,
    pub(crate) hidden_elements: usize,
    pub(crate) projection_elements: usize,
    pub(crate) activation_elements: usize,
    pub(crate) hidden_bytes: usize,
    pub(crate) projection_bytes: usize,
    pub(crate) activation_bytes: usize,
    pub(crate) slab_bytes: usize,
    pub(crate) residual: CopySpan,
    pub(crate) normalized: CopySpan,
    pub(crate) full_projection: CopySpan,
    pub(crate) full_q: CopySpan,
    pub(crate) full_k: CopySpan,
    pub(crate) full_attention: CopySpan,
    pub(crate) attention_output: CopySpan,
    pub(crate) linear_qkv: CopySpan,
    pub(crate) linear_z: CopySpan,
    pub(crate) linear_ab: CopySpan,
    pub(crate) ffn_gate_up: CopySpan,
    pub(crate) ffn_activation: CopySpan,
    pub(crate) ffn_output: CopySpan,
    pub(crate) block_linear_output: Option<CopySpan>,
}

impl Qwen35Layout {
    pub(crate) fn new(config: &Qwen35Config, capacity_tokens: usize) -> Result<Self, BackendError> {
        Self::build(config, capacity_tokens, false)
    }

    /// One B1 block plan widens activation by a contiguous normalized GDN
    /// output tail without adding a full `T * V` buffer or a second slab.
    pub(crate) fn new_block(
        config: &Qwen35Config,
        capacity_tokens: usize,
    ) -> Result<Self, BackendError> {
        Self::build(config, capacity_tokens, true)
    }

    fn build(
        config: &Qwen35Config,
        capacity_tokens: usize,
        block: bool,
    ) -> Result<Self, BackendError> {
        if capacity_tokens == 0 {
            return Err(start("capacity_tokens must be nonzero"));
        }
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        if hidden == 0
            || intermediate == 0
            || config.head_dim == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
        {
            return Err(start(
                "hidden, intermediate and attention dimensions must be nonzero",
            ));
        }
        let has_linear = config.layer_types.contains(&Qwen35LayerType::Linear);
        if has_linear && (config.linear_num_key_heads == 0 || config.linear_num_value_heads == 0) {
            return Err(start(
                "linear attention requires nonzero key and value heads",
            ));
        }

        let q_width = product(config.num_attention_heads, config.head_dim, "query width")?;
        let kv_width = product(
            config.num_key_value_heads,
            config.head_dim,
            "key/value width",
        )?;
        let query_gate_width = product(q_width, 2, "query gate width")?;
        let kv_projection_width = product(kv_width, 2, "KV projection width")?;
        let full_projection_width = add(
            query_gate_width,
            kv_projection_width,
            "full projection width",
        )?;
        let linear_key_width = product(
            config.linear_num_key_heads,
            LINEAR_HEAD_DIM,
            "linear key width",
        )?;
        let linear_value_width = product(
            config.linear_num_value_heads,
            LINEAR_HEAD_DIM,
            "linear value width",
        )?;
        let linear_qkv_width = add(
            product(linear_key_width, 2, "linear QKV width")?,
            linear_value_width,
            "linear QKV width",
        )?;
        let linear_ab_width = product(config.linear_num_value_heads, 2, "linear A/B width")?;
        let ffn_gate_up_width = product(intermediate, 2, "FFN gate/up width")?;
        let full_activation_width = add(
            product(q_width, 2, "full attention width")?,
            kv_width,
            "full attention width",
        )?;
        let linear_activation_width = add(
            add(hidden, linear_value_width, "linear activation width")?,
            linear_ab_width,
            "linear activation width",
        )?;
        let projection_width = ffn_gate_up_width
            .max(full_projection_width)
            .max(linear_qkv_width)
            .max(hidden);
        let block_activation_width = if block {
            add(
                add(
                    hidden,
                    product(linear_value_width, 2, "block value width")?,
                    "block activation width",
                )?,
                product(config.linear_num_value_heads, 4, "block A/B width")?,
                "block activation width",
            )?
        } else {
            0
        };
        let activation_width = intermediate
            .max(full_activation_width)
            .max(linear_activation_width)
            .max(block_activation_width)
            .max(hidden);

        let hidden_elements = product(capacity_tokens, hidden, "hidden slab")?;
        let projection_elements = product(capacity_tokens, projection_width, "projection slab")?;
        let activation_elements = product(capacity_tokens, activation_width, "activation slab")?;
        let total_elements = add(
            add(hidden_elements, projection_elements, "slab elements")?,
            activation_elements,
            "slab elements",
        )?;
        let hidden_bytes = product(hidden_elements, BF16_BYTES, "hidden slab bytes")?;
        let projection_bytes = product(projection_elements, BF16_BYTES, "projection slab bytes")?;
        let activation_bytes = product(activation_elements, BF16_BYTES, "activation slab bytes")?;
        let slab_bytes = product(total_elements, BF16_BYTES, "slab bytes")?;

        let residual = capacity_span(0, capacity_tokens, hidden, hidden_elements, "residual")?;
        let normalized = capacity_span(
            0,
            capacity_tokens,
            hidden,
            activation_elements,
            "normalized input",
        )?;
        let full_projection = capacity_span(
            0,
            capacity_tokens,
            full_projection_width,
            projection_elements,
            "full projection",
        )?;
        let full_k_offset = product(capacity_tokens, q_width, "full K offset")?;
        let full_attention_offset = product(
            capacity_tokens,
            add(q_width, kv_width, "full attention offset")?,
            "full attention offset",
        )?;
        let full_q = capacity_span(
            0,
            capacity_tokens,
            q_width,
            activation_elements,
            "full query",
        )?;
        let full_k = capacity_span(
            full_k_offset,
            capacity_tokens,
            kv_width,
            activation_elements,
            "full key",
        )?;
        let full_attention = capacity_span(
            full_attention_offset,
            capacity_tokens,
            q_width,
            activation_elements,
            "full attention",
        )?;
        let attention_output = capacity_span(
            0,
            capacity_tokens,
            hidden,
            projection_elements,
            "attention output",
        )?;

        let linear_qkv = capacity_span(
            0,
            capacity_tokens,
            linear_qkv_width,
            projection_elements,
            "linear QKV",
        )?;
        let linear_z_offset = product(capacity_tokens, hidden, "linear Z offset")?;
        let linear_ab_offset = product(
            capacity_tokens,
            add(hidden, linear_value_width, "linear A/B offset")?,
            "linear A/B offset",
        )?;
        let linear_z = capacity_span(
            linear_z_offset,
            capacity_tokens,
            linear_value_width,
            activation_elements,
            "linear Z",
        )?;
        let linear_ab = capacity_span(
            linear_ab_offset,
            capacity_tokens,
            linear_ab_width,
            activation_elements,
            "linear A/B",
        )?;

        let ffn_gate_up = capacity_span(
            0,
            capacity_tokens,
            ffn_gate_up_width,
            projection_elements,
            "FFN gate/up",
        )?;
        let ffn_activation = capacity_span(
            0,
            capacity_tokens,
            intermediate,
            activation_elements,
            "FFN activation",
        )?;
        let ffn_output = capacity_span(
            0,
            capacity_tokens,
            hidden,
            projection_elements,
            "FFN output",
        )?;
        let block_linear_output = if block {
            let offset = product(
                capacity_tokens,
                add(
                    add(hidden, linear_value_width, "block linear output offset")?,
                    product(
                        config.linear_num_value_heads,
                        4,
                        "block linear output offset",
                    )?,
                    "block linear output offset",
                )?,
                "block linear output offset",
            )?;
            Some(capacity_span(
                offset,
                capacity_tokens,
                linear_value_width,
                activation_elements,
                "block linear output",
            )?)
        } else {
            None
        };

        Ok(Self {
            capacity_tokens,
            hidden_elements,
            projection_elements,
            activation_elements,
            hidden_bytes,
            projection_bytes,
            activation_bytes,
            slab_bytes,
            residual,
            normalized,
            full_projection,
            full_q,
            full_k,
            full_attention,
            attention_output,
            linear_qkv,
            linear_z,
            linear_ab,
            ffn_gate_up,
            ffn_activation,
            ffn_output,
            block_linear_output,
        })
    }

    scalar_accessors!(
        capacity_tokens,
        hidden_elements,
        projection_elements,
        activation_elements,
        hidden_bytes,
        projection_bytes,
        activation_bytes,
        slab_bytes,
    );

    span_accessors!(
        residual,
        normalized,
        full_projection,
        full_q,
        full_k,
        full_attention,
        attention_output,
        linear_qkv,
        linear_z,
        linear_ab,
        ffn_gate_up,
        ffn_activation,
        ffn_output,
    );

    pub(crate) const fn block_linear_output(&self) -> Option<CopySpan> {
        self.block_linear_output
    }

    pub(crate) const fn total_bytes(&self) -> usize {
        self.slab_bytes
    }
}

fn capacity_span(
    offset: usize,
    capacity_tokens: usize,
    row_width: usize,
    slab_elements: usize,
    name: &str,
) -> Result<CopySpan, BackendError> {
    let elements = product(capacity_tokens, row_width, name)?;
    let span = CopySpan::new(offset, elements, row_width)?;
    span.fits(slab_elements)
        .map_err(|_| start(format!("{name} exceeds its slab")))?;
    Ok(span)
}

impl AttentionSlabLayout for Qwen35Layout {
    fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
    fn hidden_elements(&self) -> usize {
        self.hidden_elements
    }
    fn projection_elements(&self) -> usize {
        self.projection_elements
    }
    fn activation_elements(&self) -> usize {
        self.activation_elements
    }
    fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_config::Qwen35LayerType;

    const MIB: usize = 1024 * 1024;

    fn candidate(
        hidden: usize,
        intermediate: usize,
        heads: usize,
        kv_heads: usize,
        value_heads: usize,
        layers: usize,
    ) -> Qwen35Config {
        Qwen35Config {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim: 256,
            rotary_dim: 64,
            vocab_size: 248_320,
            max_position_embeddings: 262_144,
            linear_num_key_heads: 16,
            linear_num_value_heads: value_heads,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            layer_types: (0..layers)
                .map(|index| {
                    if index % 4 == 3 {
                        Qwen35LayerType::Full
                    } else {
                        Qwen35LayerType::Linear
                    }
                })
                .collect(),
            attn_output_gate: true,
        }
    }

    fn elephant() -> Qwen35Config {
        candidate(4096, 12_288, 16, 4, 32, 32)
    }

    fn mammoth() -> Qwen35Config {
        candidate(5120, 17_408, 24, 4, 48, 64)
    }

    fn disjoint(left: CopySpan, right: CopySpan) -> bool {
        left.offset + left.elements <= right.offset || right.offset + right.elements <= left.offset
    }

    #[test]
    fn capacity_sweep_keeps_all_spans_inside_three_slabs() {
        for capacity in [1usize, 7, 128, 129, 2048] {
            let layout = Qwen35Layout::new(&elephant(), capacity).unwrap();
            assert_eq!(layout.capacity_tokens(), capacity);
            assert_eq!(layout.residual().rows(), capacity);
            assert_eq!(layout.normalized().rows(), capacity);
            assert_eq!(layout.full_q().rows(), capacity);
            assert_eq!(layout.full_k().rows(), capacity);
            assert_eq!(layout.full_attention().rows(), capacity);
            assert_eq!(layout.linear_z().rows(), capacity);
            assert_eq!(layout.linear_ab().rows(), capacity);
            assert_eq!(layout.ffn_gate_up().rows(), capacity);
            assert_eq!(layout.ffn_activation().rows(), capacity);

            for span in [
                layout.residual(),
                layout.normalized(),
                layout.full_projection(),
                layout.full_q(),
                layout.full_k(),
                layout.full_attention(),
                layout.attention_output(),
                layout.linear_qkv(),
                layout.linear_z(),
                layout.linear_ab(),
                layout.ffn_gate_up(),
                layout.ffn_activation(),
                layout.ffn_output(),
            ] {
                assert_eq!(span.elements(), span.rows() * span.row_width());
                assert!(span.byte_range().unwrap().end <= layout.slab_bytes());
            }
            assert_eq!(
                layout.slab_bytes(),
                layout.hidden_bytes() + layout.projection_bytes() + layout.activation_bytes()
            );
        }
    }

    #[test]
    fn candidate_capacity_matches_exact_slab_mebibytes() {
        let mammoth = Qwen35Layout::new(&mammoth(), 2048).unwrap();
        assert_eq!(mammoth.hidden_bytes(), 20 * MIB);
        assert_eq!(mammoth.projection_bytes(), 136 * MIB);
        assert_eq!(mammoth.activation_bytes(), 68 * MIB);
        assert_eq!(mammoth.slab_bytes(), 224 * MIB);
        assert_eq!(mammoth.total_bytes(), mammoth.slab_bytes());

        let elephant = Qwen35Layout::new(&elephant(), 2048).unwrap();
        assert_eq!(elephant.hidden_bytes(), 16 * MIB);
        assert_eq!(elephant.projection_bytes(), 96 * MIB);
        assert_eq!(elephant.activation_bytes(), 48 * MIB);
        assert_eq!(elephant.slab_bytes(), 160 * MIB);
    }

    #[test]
    fn full_attention_query_key_and_output_are_simultaneously_disjoint() {
        let layout = Qwen35Layout::new(&elephant(), 7).unwrap();
        let q = layout.full_q();
        let k = layout.full_k();
        let attention = layout.full_attention();
        assert_eq!(q.offset(), 0);
        assert_eq!(q.elements(), 7 * 4096);
        assert_eq!(q.row_width(), 4096);
        assert_eq!(k.offset(), q.offset() + q.elements());
        assert_eq!(k.elements(), 7 * 1024);
        assert_eq!(k.row_width(), 1024);
        assert_eq!(attention.offset(), k.offset() + k.elements());
        assert_eq!(attention.elements(), q.elements());
        assert!(disjoint(q, k));
        assert!(disjoint(q, attention));
        assert!(disjoint(k, attention));
        assert_eq!(layout.full_projection().row_width(), 2 * 4096 + 2 * 1024);
        assert_eq!(layout.attention_output().row_width(), 4096);
    }

    #[test]
    fn linear_normalized_z_and_ab_remain_separate_before_all_projections() {
        let layout = Qwen35Layout::new(&mammoth(), 129).unwrap();
        let normalized = layout.normalized();
        let z = layout.linear_z();
        let ab = layout.linear_ab();
        assert_eq!(normalized.offset(), 0);
        assert_eq!(normalized.elements(), 129 * 5120);
        assert_eq!(z.offset(), normalized.offset() + normalized.elements());
        assert_eq!(z.row_width(), 48 * 128);
        assert_eq!(z.elements(), 129 * 48 * 128);
        assert_eq!(ab.offset(), z.offset() + z.elements());
        assert_eq!(ab.row_width(), 2 * 48);
        assert!(disjoint(normalized, z));
        assert!(disjoint(normalized, ab));
        assert!(disjoint(z, ab));
        assert_eq!(layout.linear_qkv().row_width(), (2 * 16 + 48) * 128);
        assert_eq!(layout.linear_qkv().offset(), 0);
    }

    #[test]
    fn ffn_reuses_projection_after_inputs_and_keeps_residual_dedicated() {
        let layout = Qwen35Layout::new(&elephant(), 1).unwrap();
        assert_eq!(layout.residual().offset(), 0);
        assert_eq!(layout.residual().elements(), 4096);
        assert_eq!(layout.residual().row_width(), 4096);
        assert_eq!(layout.ffn_gate_up().offset(), 0);
        assert_eq!(layout.ffn_gate_up().row_width(), 2 * 12_288);
        assert_eq!(layout.ffn_activation().offset(), 0);
        assert_eq!(layout.ffn_activation().row_width(), 12_288);
        assert_eq!(layout.ffn_output().offset(), 0);
        assert_eq!(layout.ffn_output().row_width(), 4096);
        assert_eq!(layout.attention_output(), layout.ffn_output());
        assert!(layout.ffn_gate_up().elements() <= layout.projection_elements());
        assert!(layout.ffn_activation().elements() <= layout.activation_elements());
        assert!(layout.ffn_output().elements() <= layout.projection_elements());
        assert_eq!(layout.residual().elements(), layout.hidden_elements());
    }

    #[test]
    fn short_actual_rows_fit_capacity_spans_without_padding_tokens() {
        let layout = Qwen35Layout::new(&mammoth(), 128).unwrap();
        for actual in [1usize, 7, 128] {
            for capacity_span in [
                layout.residual(),
                layout.normalized(),
                layout.full_q(),
                layout.full_k(),
                layout.full_attention(),
                layout.linear_z(),
                layout.linear_ab(),
                layout.ffn_gate_up(),
                layout.ffn_activation(),
            ] {
                let span = capacity_span.row_span(actual).unwrap();
                assert_eq!(span.rows(), actual);
                assert_eq!(span.elements(), actual * capacity_span.row_width());
                assert_eq!(span.offset(), capacity_span.offset());
                assert!(
                    span.offset() + span.elements()
                        <= capacity_span.offset() + capacity_span.elements()
                );
            }
        }
        assert!(layout.full_q().row_span(129).is_err());
    }

    #[test]
    fn overflow_in_capacity_or_phase_width_is_rejected() {
        assert!(Qwen35Layout::new(&elephant(), 0).is_err());
        assert!(Qwen35Layout::new(&elephant(), usize::MAX).is_err());

        let mut wide_hidden = elephant();
        wide_hidden.hidden_size = usize::MAX;
        assert!(Qwen35Layout::new(&wide_hidden, 2).is_err());

        let mut wide_query = elephant();
        wide_query.num_attention_heads = usize::MAX;
        assert!(Qwen35Layout::new(&wide_query, 1).is_err());

        let mut wide_linear = mammoth();
        wide_linear.linear_num_value_heads = usize::MAX;
        assert!(Qwen35Layout::new(&wide_linear, 1).is_err());
    }

    #[test]
    fn copy_span_pointer_math_is_byte_checked() {
        let span = CopySpan::new(3, 10, 5).unwrap();
        assert_eq!(span.rows(), 2);
        assert_eq!(span.byte_range().unwrap(), 6..26);
        let base = 0x1000usize as *mut u8;
        assert_eq!(span.validated_ptr(base).unwrap() as usize, 0x1006);
        assert_eq!(
            span.validated_const_ptr(0x1000usize as *const u8).unwrap() as usize,
            0x1006
        );
        assert!(CopySpan::new(0, 0, 0).is_ok());
        assert!(CopySpan::new(0, 1, 0).is_err());
        assert!(CopySpan::new(0, 5, 2).is_err());
        assert!(CopySpan::new(usize::MAX, 1, 1).is_err());
        let overflowing = CopySpan {
            offset: usize::MAX,
            elements: 1,
            row_width: 1,
        };
        assert!(overflowing.byte_range().is_err());
    }

    #[test]
    fn full_only_configuration_does_not_require_linear_widths() {
        let mut config = elephant();
        config.layer_types = vec![Qwen35LayerType::Full; 8];
        config.linear_num_key_heads = 0;
        config.linear_num_value_heads = 0;
        let layout = Qwen35Layout::new(&config, 7).unwrap();
        assert_eq!(layout.linear_z().elements(), 0);
        assert_eq!(layout.linear_ab().elements(), 0);
        assert_eq!(layout.linear_qkv().elements(), 0);
        assert!(layout.full_q().elements() > 0);
    }

    #[test]
    fn block_layout_has_exact_elephant_and_mammoth_bytes_at_2048() {
        let elephant = Qwen35Layout::new_block(&elephant(), 2048).unwrap();
        assert_eq!(elephant.activation_elements(), 2048 * 12_416);
        assert_eq!(elephant.activation_bytes(), 50_855_936);
        assert_eq!(elephant.slab_bytes(), 16 * MIB + 96 * MIB + 50_855_936);
        let output = elephant.block_linear_output().unwrap();
        assert_eq!(output.offset(), 2048 * (4096 + 4096 + 4 * 32));
        assert_eq!(output.elements(), 2048 * 4096);
        assert_eq!(output.row_width(), 4096);
        assert_eq!(
            output.offset() + output.elements(),
            elephant.activation_elements()
        );

        let mammoth = Qwen35Layout::new_block(&mammoth(), 2048).unwrap();
        assert_eq!(mammoth.activation_elements(), 2048 * 17_600);
        assert_eq!(mammoth.activation_bytes(), 72_089_600);
        assert_eq!(mammoth.slab_bytes(), 20 * MIB + 136 * MIB + 72_089_600);
        let output = mammoth.block_linear_output().unwrap();
        assert_eq!(output.offset(), 2048 * (5120 + 6144 + 4 * 48));
        assert_eq!(output.elements(), 2048 * 6144);
        assert_eq!(output.row_width(), 6144);
        assert_eq!(
            output.offset() + output.elements(),
            mammoth.activation_elements()
        );
    }

    #[test]
    fn block_linear_output_is_disjoint_from_inputs_for_actual_rows() {
        for capacity in [1usize, 7, 128, 129, 2048] {
            let layout = Qwen35Layout::new_block(&elephant(), capacity).unwrap();
            let output = layout.block_linear_output().unwrap();
            for actual in [1usize, 7, 128, 129, 2048] {
                if actual > capacity {
                    continue;
                }
                let output = output.row_span(actual).unwrap();
                let vh = 32;
                let at = capacity * (4096 + 4096 + 2 * vh);
                let a_temp = CopySpan::new(at, actual * vh, vh).unwrap();
                let b_temp = CopySpan::new(at + capacity * vh, actual * vh, vh).unwrap();
                for input in [
                    layout.normalized().row_span(actual).unwrap(),
                    layout.linear_z().row_span(actual).unwrap(),
                    layout.linear_ab().row_span(actual).unwrap(),
                    a_temp,
                    b_temp,
                ] {
                    assert!(disjoint(output, input));
                }
                assert!(output.offset() + output.elements() <= layout.activation_elements());
            }
        }
        let layout = Qwen35Layout::new_block(&mammoth(), 2048).unwrap();
        let output = layout.block_linear_output().unwrap();
        assert!(disjoint(output, layout.normalized()));
        assert!(disjoint(output, layout.linear_z()));
        assert!(disjoint(output, layout.linear_ab()));
    }

    #[test]
    fn ordinary_layout_has_no_block_output_and_block_overflow_is_checked() {
        let ordinary = Qwen35Layout::new(&mammoth(), 2048).unwrap();
        assert_eq!(ordinary.block_linear_output(), None);
        assert_eq!(ordinary.activation_bytes(), 68 * MIB);
        assert_eq!(ordinary.slab_bytes(), 224 * MIB);

        assert!(Qwen35Layout::new_block(&elephant(), 0).is_err());
        assert!(Qwen35Layout::new_block(&elephant(), usize::MAX).is_err());
        let mut wide_heads = elephant();
        wide_heads.linear_num_value_heads = usize::MAX;
        assert!(Qwen35Layout::new_block(&wide_heads, 1).is_err());
        let mut wide_hidden = mammoth();
        wide_hidden.hidden_size = usize::MAX;
        assert!(Qwen35Layout::new_block(&wide_hidden, 2).is_err());
    }
}
