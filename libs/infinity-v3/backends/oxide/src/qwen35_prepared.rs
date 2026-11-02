// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Bounded host-side preparation for the seven exact Qwen3.5 dense BF16
//! device exports in `device/src/qwen35_norm.rs` and
//! `device/src/qwen35_attention.rs`.
//!
//! This module is a host wrapper, not an executor and not a model route. It
//! binds the already-built module once, chooses the exact native norm variant
//! on the CPU, and submits fixed-stack launch arguments. The admitted scope is
//! the native BF16 dense Elephant/Mammoth attention shape: 256-wide heads and
//! the native 64-wide partial rotary. FP8, MoE, media, and any other arithmetic
//! or head layout are deliberately not admitted here.
//!
//! The raw bound methods perform no allocation, reference-count operation,
//! CUDA metadata query, or context binding. They only submit through
//! `super::launch_threads`. Geometry, pointer validity, strides, alias safety,
//! and retention must be established by the unsafe caller before submission.

use crate::{arg, launch_threads, product, u32_size};
use cuda_core::{Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

/// Native Qwen3.5 dense attention head width.
pub(crate) const QWEN35_HEAD_DIM: usize = 256;
/// Native Qwen3.5 dense partial rotary width.
pub(crate) const QWEN35_ROTARY_DIM: usize = 64;

const NORM_MAX_WIDTH: usize = 8192;
const NORM_MAX_LOGICAL: usize = 1024;
const NORM_MIN_POINTER_ALIGN: u64 = 2;
const NORM_VECTOR_POINTER_ALIGN: u64 = 8;
const ROPE_THREADS: u32 = 128;
const ROPE_HEAD_ROWS_PER_CTA: u64 = 4;
const GATE_THREADS: u32 = 256;
const GATE_HEAD_WIDTH: u64 = QWEN35_HEAD_DIM as u64;
const MAX_GRID_X: u64 = i32::MAX as u64;

const QWEN35_KERNEL_EXPORTS: [&str; 7] = [
    "infinity_qwen35_norm_vector_bf16",
    "infinity_qwen35_norm_scalar_bf16",
    "infinity_qwen35_residual_norm_vector_bf16",
    "infinity_qwen35_residual_norm_scalar_bf16",
    "infinity_qwen35_partial_rope_256_bf16",
    "infinity_qwen35_attention_gate_bf16",
    "infinity_qwen35_rope_tables_bf16",
];

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide Qwen3.5 prepared: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3.5 prepared: {error}"))
}

fn grid_x(value: u64, name: &str) -> Result<u32, BackendError> {
    if value > MAX_GRID_X {
        return Err(infer(format!(
            "{name} grid {value} exceeds the CUDA x-grid bound {MAX_GRID_X}"
        )));
    }
    u32_size(usize::try_from(value).map_err(|_| infer(format!("{name} grid exceeds usize")))?)
}

/// The exact norm variant and physical CTA width for the native one-row CTA.
///
/// The device `norm_tree<V>` uses `blockDim.x * V` logical lanes. The native
/// host recipe therefore chooses `threads = logical / 4` for the vector entry
/// and `threads = logical` for the scalar fallback, where `logical` is the
/// width rounded up to a power of two and capped at the native 1024-lane tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NormDispatch {
    pub vector: bool,
    pub threads: u32,
}

impl NormDispatch {
    /// Pure CPU selection for one plain or residual norm submission.
    ///
    /// `pointers` must contain every BF16 pointer the selected entry point will
    /// read or write: x, gamma, out for plain norm, and additionally addition
    /// and updated for residual norm. All pointers must be non-null and at
    /// least two-byte aligned. The vector entry additionally requires every
    /// pointer to be eight-byte aligned, `width`, `stride0`, and (for more than
    /// one head) `stride1` to be multiples of four.
    ///
    /// The vector eligibility is only an optimization preflight. Any failed
    /// vector predicate selects the native scalar fallback with the same
    /// arithmetic; it does not introduce a non-native path. Callers precompute
    /// this once, outside warm loops, and reuse the result.
    pub(crate) fn select(
        width: usize,
        heads: usize,
        stride0: usize,
        stride1: usize,
        pointers: &[u64],
    ) -> Result<Self, BackendError> {
        if width == 0 || width > NORM_MAX_WIDTH {
            return Err(start(format!(
                "norm width {width} is outside 1..={NORM_MAX_WIDTH}"
            )));
        }
        if heads == 0 {
            return Err(start("norm head count must be positive"));
        }
        if pointers.is_empty() {
            return Err(start("norm pointer binding must not be empty"));
        }
        for (index, &pointer) in pointers.iter().enumerate() {
            if pointer == 0 {
                return Err(start(format!("norm pointer {index} is null")));
            }
            if pointer & (NORM_MIN_POINTER_ALIGN - 1) != 0 {
                return Err(start(format!(
                    "norm pointer {index} is not two-byte aligned"
                )));
            }
        }

        let vector = width.is_multiple_of(4)
            && stride0.is_multiple_of(4)
            && (heads == 1 || stride1.is_multiple_of(4))
            && pointers
                .iter()
                .all(|pointer| pointer & (NORM_VECTOR_POINTER_ALIGN - 1) == 0);
        let logical = width.next_power_of_two().min(NORM_MAX_LOGICAL);
        let threads = if vector { logical / 4 } else { logical };
        Ok(Self {
            vector,
            threads: u32_size(threads)?,
        })
    }
}

fn rope_grid(tokens: u32, qheads: u32, kvheads: u32) -> Result<(u32, u32, u32), BackendError> {
    if tokens == 0 {
        return Ok((0, 1, 1));
    }
    let heads = u64::from(qheads)
        .checked_add(u64::from(kvheads))
        .ok_or_else(|| infer("RoPE query/key head count overflows"))?;
    if heads == 0 {
        return Err(start("RoPE requires positive query or key heads"));
    }
    let rows = u64::from(tokens)
        .checked_mul(heads)
        .ok_or_else(|| infer("RoPE row extent overflows"))?;
    Ok((grid_x(rows.div_ceil(ROPE_HEAD_ROWS_PER_CTA), "RoPE")?, 1, 1))
}

fn gate_grid(tokens: u32, heads: u32) -> Result<(u32, u32, u32), BackendError> {
    if tokens == 0 {
        return Ok((0, 1, 1));
    }
    if heads == 0 {
        return Err(start("attention gate head count must be positive"));
    }
    let elements = u64::from(tokens)
        .checked_mul(u64::from(heads))
        .and_then(|value| value.checked_mul(GATE_HEAD_WIDTH))
        .ok_or_else(|| infer("attention gate element extent overflows"))?;
    Ok((
        grid_x(elements.div_ceil(u64::from(GATE_THREADS)), "attention gate")?,
        1,
        1,
    ))
}

fn checked_element_offset(
    rows: usize,
    heads: usize,
    stride0: usize,
    stride1: usize,
    width: usize,
) -> Result<usize, BackendError> {
    let full_blocks = rows / heads;
    let tail_rows = rows % heads;
    let mut maximum = 0usize;
    if full_blocks > 0 {
        maximum = product(full_blocks - 1, stride0)?
            .checked_add(product(heads - 1, stride1)?)
            .ok_or_else(|| infer("norm input offset overflows"))?;
    }
    if tail_rows > 0 {
        let tail = product(full_blocks, stride0)?
            .checked_add(product(tail_rows - 1, stride1)?)
            .ok_or_else(|| infer("norm input tail offset overflows"))?;
        maximum = maximum.max(tail);
    }
    maximum
        .checked_add(width)
        .ok_or_else(|| infer("norm input span overflows"))
}

fn byte_end(start_address: u64, elements: usize) -> Result<u64, BackendError> {
    let bytes = u64::try_from(product(elements, 2)?)
        .map_err(|_| infer("norm address byte extent exceeds u64"))?;
    start_address
        .checked_add(bytes)
        .ok_or_else(|| infer("norm address range overflows"))
}

/// Reject a plain strided norm whose output overlaps its input.
///
/// The device plain-norm CTA writes the contiguous `[rows, width]` output but
/// reads the caller's possibly strided input. Separate CTAs have no cross-row
/// barrier, so compaction into an overlapping output is unsafe. In-place output
/// is allowed only when the input layout is exactly the same contiguous row
/// layout, in which case each CTA reads before writing its own cells.
///
/// This function computes checked address spans without scanning rows and
/// makes no claim about GPU view lifetimes. The unsafe owner of the outer arena
/// must retain every allocation and bind the stream through completion.
pub(crate) fn validate_norm_alias(
    input: u64,
    output: u64,
    rows: usize,
    width: usize,
    heads: usize,
    stride0: usize,
    stride1: usize,
) -> Result<(), BackendError> {
    if width == 0 || width > NORM_MAX_WIDTH {
        return Err(start(format!(
            "norm width {width} is outside 1..={NORM_MAX_WIDTH}"
        )));
    }
    if heads == 0 {
        return Err(start("norm head count must be positive"));
    }
    if rows == 0 {
        return Ok(());
    }
    if input == 0 || output == 0 {
        return Err(start("norm alias pointers must be non-null"));
    }

    let input_elements = checked_element_offset(rows, heads, stride0, stride1, width)?;
    let output_elements = product(rows, width)?;
    let input_end = byte_end(input, input_elements)?;
    let output_end = byte_end(output, output_elements)?;
    let overlaps = input < output_end && output < input_end;
    if !overlaps {
        return Ok(());
    }
    if input != output {
        return Err(infer(
            "plain strided norm output overlaps a different input address",
        ));
    }

    let contiguous = if rows == 1 {
        true
    } else if heads == 1 {
        stride0 == width
    } else if rows <= heads {
        stride1 == width
    } else {
        stride1 == width && stride0 == product(heads, width)?
    };
    if contiguous {
        Ok(())
    } else {
        Err(infer(
            "plain strided norm output overlap is allowed only for the exact contiguous row layout",
        ))
    }
}

/// Cold owner of the seven exact Qwen3.5 dense BF16 exports.
pub(crate) struct Qwen35Kernels {
    norm_vector: Function,
    norm_scalar: Function,
    residual_norm_vector: Function,
    residual_norm_scalar: Function,
    partial_rope: Function,
    attention_gate: Function,
    rope_tables: Arc<Function>,
}

impl Qwen35Kernels {
    /// Bind all seven exact names from an already loaded module.
    pub(crate) fn from_module(module: &Arc<cuda_core::Module>) -> Result<Self, BackendError> {
        let [norm_vector, norm_scalar, residual_norm_vector, residual_norm_scalar, partial_rope, attention_gate, rope_tables] =
            QWEN35_KERNEL_EXPORTS;
        Ok(Self {
            norm_vector: module.load_function(norm_vector).map_err(start)?,
            norm_scalar: module.load_function(norm_scalar).map_err(start)?,
            residual_norm_vector: module.load_function(residual_norm_vector).map_err(start)?,
            residual_norm_scalar: module.load_function(residual_norm_scalar).map_err(start)?,
            partial_rope: module.load_function(partial_rope).map_err(start)?,
            attention_gate: module.load_function(attention_gate).map_err(start)?,
            rope_tables: Arc::new(module.load_function(rope_tables).map_err(start)?),
        })
    }

    /// The cold table initializer, owned so a table constructor can retain it.
    pub(crate) fn tables_initializer(&self) -> Arc<Function> {
        Arc::clone(&self.rope_tables)
    }

    /// Exact plain norm entry selected by the CPU dispatch policy.
    ///
    /// # Safety
    /// The caller has admitted positive `rows`, `width`, and `heads`, a
    /// `dispatch` produced for this geometry, and all x/gamma/out allocations
    /// in the active context covering every addressed BF16 element. Plain
    /// strided input requires a disjoint output unless the exact contiguous
    /// layout was validated by [`validate_norm_alias`]. The outer arena owner
    /// retains every allocation and the stream through completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn norm_bound(
        &self,
        stream: &Stream,
        dispatch: NormDispatch,
        x: u64,
        gamma: u64,
        out: u64,
        rows: u32,
        width: u32,
        heads: u32,
        stride0: u64,
        stride1: u64,
        eps: f32,
    ) -> Result<(), BackendError> {
        if rows == 0 {
            return Ok(());
        }
        let function = if dispatch.vector {
            &self.norm_vector
        } else {
            &self.norm_scalar
        };
        let mut x = x;
        let mut gamma = gamma;
        let mut out = out;
        let mut rows = rows;
        let mut width = width;
        let mut heads = heads;
        let mut stride0 = stride0;
        let mut stride1 = stride1;
        let mut eps = eps;
        launch_threads(
            function,
            stream,
            (rows, 1, 1),
            dispatch.threads,
            0,
            &mut [
                arg(&mut x),
                arg(&mut gamma),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut heads),
                arg(&mut stride0),
                arg(&mut stride1),
                arg(&mut eps),
            ],
        )
    }

    /// Exact rounded-residual norm entry selected by the CPU dispatch policy.
    ///
    /// # Safety
    /// The caller has admitted positive `rows` and `width`, a `dispatch`
    /// produced with all five pointers, and contiguous x/addition/gamma/out/
    /// updated allocations in the active context covering `rows * width` BF16
    /// elements. The device reads every cell before writing its own updated or
    /// output cell. The outer arena owner retains every allocation and the
    /// stream through completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn residual_norm_bound(
        &self,
        stream: &Stream,
        dispatch: NormDispatch,
        x: u64,
        addition: u64,
        gamma: u64,
        out: u64,
        updated: u64,
        rows: u32,
        width: u32,
        eps: f32,
    ) -> Result<(), BackendError> {
        if rows == 0 {
            return Ok(());
        }
        let function = if dispatch.vector {
            &self.residual_norm_vector
        } else {
            &self.residual_norm_scalar
        };
        let mut x = x;
        let mut addition = addition;
        let mut gamma = gamma;
        let mut out = out;
        let mut updated = updated;
        let mut rows = rows;
        let mut width = width;
        let mut eps = eps;
        launch_threads(
            function,
            stream,
            (rows, 1, 1),
            dispatch.threads,
            0,
            &mut [
                arg(&mut x),
                arg(&mut addition),
                arg(&mut gamma),
                arg(&mut out),
                arg(&mut updated),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut eps),
            ],
        )
    }

    /// Native staged partial RoPE over packed Q and K head rows.
    ///
    /// # Safety
    /// The caller has admitted positive `tokens`, query/key head counts, an
    /// even rotary width no larger than the 256-wide head, and table stride
    /// covering `rotary / 2`. Q/K, cosine/sine, and positions must cover every
    /// addressed element in the active context. The outer arena owner retains
    /// every allocation and the stream through completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn rope_bound(
        &self,
        stream: &Stream,
        q: u64,
        k: u64,
        cosine: u64,
        sine: u64,
        positions: u64,
        tokens: u32,
        qheads: u32,
        kvheads: u32,
        rotary: u32,
        qstride: u64,
        kstride: u64,
        table_stride: u32,
    ) -> Result<(), BackendError> {
        if tokens == 0 {
            return Ok(());
        }
        let grid = rope_grid(tokens, qheads, kvheads)?;
        let mut q = q;
        let mut k = k;
        let mut cosine = cosine;
        let mut sine = sine;
        let mut positions = positions;
        let mut tokens = tokens;
        let mut qheads = qheads;
        let mut kvheads = kvheads;
        let mut rotary = rotary;
        let mut qstride = qstride;
        let mut kstride = kstride;
        let mut table_stride = table_stride;
        launch_threads(
            &self.partial_rope,
            stream,
            grid,
            ROPE_THREADS,
            0,
            &mut [
                arg(&mut q),
                arg(&mut k),
                arg(&mut cosine),
                arg(&mut sine),
                arg(&mut positions),
                arg(&mut tokens),
                arg(&mut qheads),
                arg(&mut kvheads),
                arg(&mut rotary),
                arg(&mut qstride),
                arg(&mut kstride),
                arg(&mut table_stride),
            ],
        )
    }

    /// Dense BF16 post-attention sigmoid gate.
    ///
    /// `query_gate` is the gate-half base, not the packed query base.
    ///
    /// # Safety
    /// The caller has admitted positive `tokens` and `heads`, and attention,
    /// gate-half, and output allocations in the active context covering the
    /// contiguous attention extent and the strided gate extent. Output may
    /// alias attention only for the same flat extent. The outer arena owner
    /// retains every allocation and the stream through completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gate_bound(
        &self,
        stream: &Stream,
        attention: u64,
        query_gate: u64,
        out: u64,
        tokens: u32,
        heads: u32,
        gate_stride: u64,
    ) -> Result<(), BackendError> {
        if tokens == 0 {
            return Ok(());
        }
        let grid = gate_grid(tokens, heads)?;
        let mut attention = attention;
        let mut query_gate = query_gate;
        let mut out = out;
        let mut tokens = tokens;
        let mut heads = heads;
        let mut gate_stride = gate_stride;
        launch_threads(
            &self.attention_gate,
            stream,
            grid,
            GATE_THREADS,
            0,
            &mut [
                arg(&mut attention),
                arg(&mut query_gate),
                arg(&mut out),
                arg(&mut tokens),
                arg(&mut heads),
                arg(&mut gate_stride),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIGNED: [u64; 5] = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000];

    #[test]
    fn dispatch_uses_the_native_row_block_formula() {
        assert_eq!(
            NormDispatch::select(256, 1, 256, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: true,
                threads: 64,
            }
        );
        assert_eq!(
            NormDispatch::select(4096, 1, 4096, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: true,
                threads: 256,
            }
        );
        assert_eq!(
            NormDispatch::select(5120, 1, 5120, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: true,
                threads: 256,
            }
        );
        assert_eq!(
            NormDispatch::select(255, 1, 255, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: false,
                threads: 256,
            }
        );
        assert_eq!(
            NormDispatch::select(257, 1, 257, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: false,
                threads: 512,
            }
        );
    }

    #[test]
    fn dispatch_caps_the_tree_and_falls_back_for_misalignment() {
        assert_eq!(
            NormDispatch::select(8192, 1, 8192, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: true,
                threads: 256,
            }
        );
        assert_eq!(
            NormDispatch::select(8191, 1, 8191, 0, &ALIGNED).unwrap(),
            NormDispatch {
                vector: false,
                threads: 1024,
            }
        );
        assert_eq!(
            NormDispatch::select(1, 1, 1, 0, &[0x1000, 0x2000, 0x3000]).unwrap(),
            NormDispatch {
                vector: false,
                threads: 1,
            }
        );
        let two_byte_only = [0x1002, 0x2000, 0x3000];
        assert_eq!(
            NormDispatch::select(256, 1, 256, 0, &two_byte_only).unwrap(),
            NormDispatch {
                vector: false,
                threads: 256,
            }
        );
        let stride_misaligned = [0x1000, 0x2000, 0x3000];
        assert!(
            !NormDispatch::select(256, 1, 257, 0, &stride_misaligned)
                .unwrap()
                .vector
        );
        assert!(
            !NormDispatch::select(256, 2, 512, 258, &stride_misaligned)
                .unwrap()
                .vector
        );
        assert!(
            NormDispatch::select(256, 2, 512, 260, &stride_misaligned)
                .unwrap()
                .vector
        );
        assert!(
            NormDispatch::select(256, 1, 256, 3, &stride_misaligned)
                .unwrap()
                .vector
        );
    }

    #[test]
    fn dispatch_rejects_zero_odd_pointers_null_and_excess_width() {
        let odd = [0x1001, 0x2000, 0x3000];
        let null = [0u64, 0x2000, 0x3000];
        assert!(NormDispatch::select(256, 1, 256, 0, &odd).is_err());
        assert!(NormDispatch::select(256, 1, 256, 0, &null).is_err());
        assert!(NormDispatch::select(0, 1, 0, 0, &ALIGNED).is_err());
        assert!(NormDispatch::select(8193, 1, 8193, 0, &ALIGNED).is_err());
        assert!(NormDispatch::select(256, 0, 256, 0, &ALIGNED).is_err());
    }

    #[test]
    fn fixed_launch_grids_round_up_and_reject_excess_blocks() {
        assert_eq!(rope_grid(0, 2, 1).unwrap(), (0, 1, 1));
        assert_eq!(rope_grid(1, 2, 1).unwrap(), (1, 1, 1));
        assert_eq!(rope_grid(5, 2, 1).unwrap(), (4, 1, 1));
        assert_eq!(gate_grid(0, 3).unwrap(), (0, 1, 1));
        assert_eq!(gate_grid(1, 3).unwrap(), (3, 1, 1));
        assert_eq!(gate_grid(2, 3).unwrap(), (6, 1, 1));
        assert!(rope_grid(u32::MAX, u32::MAX, u32::MAX).is_err());
        assert!(rope_grid(u32::MAX, 0, 0).is_err());
        assert!(gate_grid(u32::MAX, u32::MAX).is_err());
        assert!(gate_grid(1, 0).is_err());
    }

    #[test]
    fn alias_allows_only_the_exact_contiguous_in_place_layout() {
        assert!(validate_norm_alias(0x1000, 0x1000, 3, 256, 1, 256, 0).is_ok());
        assert!(validate_norm_alias(0x1000, 0x1000, 4, 8, 2, 16, 8).is_ok());
        assert!(validate_norm_alias(0x1000, 0x1000, 3, 8, 3, 24, 8).is_ok());
        assert!(validate_norm_alias(0x1000, 0x8000, 4, 8, 2, 32, 16).is_ok());
        assert!(validate_norm_alias(0x1000, 0x1000, 4, 8, 2, 32, 16).is_err());
        assert!(validate_norm_alias(0x1000, 0x1002, 4, 8, 2, 16, 8).is_err());
        assert!(validate_norm_alias(0x1004, 0x1000, 4, 8, 2, 16, 8).is_err());
    }

    #[test]
    fn alias_rejects_invalid_geometry_and_checked_span_overflow() {
        assert!(validate_norm_alias(0x1000, 0x8000, 0, 8, 2, 16, 8).is_ok());
        assert!(validate_norm_alias(0x1000, 0x8000, 4, 0, 2, 16, 8).is_err());
        assert!(validate_norm_alias(0x1000, 0x8000, 4, 8193, 2, 16, 8).is_err());
        assert!(validate_norm_alias(0x1000, 0x8000, 4, 8, 0, 16, 8).is_err());
        assert!(validate_norm_alias(0, 0x8000, 4, 8, 2, 16, 8).is_err());
        assert!(validate_norm_alias(0x1000, 0, 4, 8, 2, 16, 8).is_err());
        assert!(validate_norm_alias(0x1000, 0x8000, 2, 8, 1, usize::MAX, 0).is_err());
        assert!(validate_norm_alias(u64::MAX - 1, 0, 1, 2, 1, 2, 0).is_err());
    }
}
