// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//
// The mirrored adapter ABI originates in a source file carrying:
//   Copyright (c) Meta Platforms, Inc. and affiliates.
//   All rights reserved.
//   This source code is licensed under the BSD-style license found in the
//   LICENSE file in the root directory of this source tree.
// Its CUTLASS headers remain under NVIDIA's CUTLASS license.
//!
//! Bounded native MPNet bucket-bias attention provider.
//!
//! This module binds the standalone adapter in `standalone_mpnet_attention.cu`
//! through its exact C ABI. The adapter is derived from PyTorch's public
//! memory-efficient attention forward kernel at commit
//! `5c4886908584029761b579af026dcfb627c84070`; its CUTLASS dependency is the
//! public commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`. The adapter file
//! carries the upstream Meta Platforms copyright and BSD-style license, and
//! the CUTLASS headers remain under NVIDIA's CUTLASS license. Those notices
//! must be preserved when this provider is distributed with the adapter.
//!
//! The provider owns one cold prepared plan per model. It never creates a
//! process-global cache, never binds a context, allocates, synchronizes or
//! queries a symbol on the forward path, and leaves all matrix allocations and
//! context ownership with the caller. As with the existing FA4 provider, the
//! caller must keep every borrowed device allocation and the stream live until
//! the engine's outer drain completes.

use crate::fa4::DType;
use crate::{infer, start};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use serde_json::Value;
use std::{
    ffi::{c_void, CString},
    path::Path,
    sync::Arc,
};

pub(crate) const MPNET_HEADS: usize = 12;
pub(crate) const MPNET_HEAD_DIM: usize = 64;
pub(crate) const MPNET_BUCKETS: usize = 32;
pub(crate) const MPNET_QKV_ROW_STRIDE: usize = 3 * MPNET_HEADS * MPNET_HEAD_DIM;
pub(crate) const MPNET_OUT_WIDTH: usize = MPNET_HEADS * MPNET_HEAD_DIM;
pub(crate) const MPNET_MAX_SEQ: usize = 512;
pub(crate) const MPNET_MAX_BATCHES: usize = 65_535;

pub(crate) const MPNET_ATTN_ABI: u32 = 1;
pub(crate) const MPNET_ATTN_DTYPE_BF16: u32 = 1;
pub(crate) const MPNET_PREPARED_BYTES: u32 = 24;
pub(crate) const MPNET_GEOMETRY_BYTES: usize = 136;

const MPNET_STATUS_OK: i32 = 0;
const MPNET_STATUS_EINVAL: i32 = 1;
const MPNET_STATUS_EUNSUPPORTED: i32 = 2;
const MPNET_STATUS_ECUDA: i32 = 3;
const MPNET_STATUS_EUNPREPARED: i32 = 4;

const MPNET_QKV_ELEMENT_BYTES: u64 = 2;
const MPNET_K_BYTE_OFFSET: u64 = (MPNET_HEADS * MPNET_HEAD_DIM) as u64 * MPNET_QKV_ELEMENT_BYTES;
const MPNET_V_BYTE_OFFSET: u64 =
    2 * (MPNET_HEADS * MPNET_HEAD_DIM) as u64 * MPNET_QKV_ELEMENT_BYTES;
const MPNET_BIAS_BUCKET_STRIDE: u32 = MPNET_HEADS as u32;
const MPNET_BIAS_HEAD_STRIDE: u32 = 1;

const PYTORCH_COMMIT: &str = "5c4886908584029761b579af026dcfb627c84070";
const CUTLASS_COMMIT: &str = "e05f953a5b3d38adc240df2ff928e0421c2abba3";
const SUPPORTED_ARCHITECTURES: [u64; 3] = [90, 100, 120];

const SYMBOL_PREPARE: &str = "mpnet_attn_prepare_bf16_h12d64";
const SYMBOL_VALIDATE: &str = "mpnet_attn_validate";
const SYMBOL_LAUNCH: &str = "mpnet_attn_launch_bf16_h12d64";
const SYMBOL_SMEM_BYTES: &str = "mpnet_attn_smem_bytes";

type Prepare = unsafe extern "C" fn(*mut Prepared) -> i32;
type Validate = unsafe extern "C" fn(*const Geometry) -> i32;
type Launch = unsafe extern "C" fn(*const Prepared, *const Geometry) -> i32;
type SmemBytes = unsafe extern "C" fn() -> usize;

/// Exact C layout of `mpnet_attn_prepared` from `standalone_mpnet_attention.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Prepared {
    pub(crate) abi_version: u32,
    pub(crate) struct_bytes: u32,
    pub(crate) shared_memory_bytes: u64,
    pub(crate) context: *mut c_void,
}

/// Exact C layout of `mpnet_attn_geometry` from `standalone_mpnet_attention.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub(crate) abi_version: u32,
    pub(crate) dtype: u32,
    pub(crate) num_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) qkv_row_stride: u32,
    pub(crate) q_offset: u32,
    pub(crate) k_offset: u32,
    pub(crate) v_offset: u32,
    pub(crate) num_batches: u32,
    pub(crate) max_seqlen_q: u32,
    pub(crate) max_seqlen_k: u32,
    pub(crate) bias_num_buckets: u32,
    pub(crate) bias_num_heads: u32,
    pub(crate) bias_bucket_stride: u32,
    pub(crate) bias_head_stride: u32,
    pub(crate) out_row_stride: u32,
    pub(crate) flags: u32,
    pub(crate) qkv_num_rows: u64,
    pub(crate) out_num_rows: u64,
    pub(crate) qkv: *const c_void,
    pub(crate) seqstart_q: *const i32,
    pub(crate) seqstart_k: *const i32,
    pub(crate) bias_table: *const c_void,
    pub(crate) out: *mut c_void,
    pub(crate) stream: *mut c_void,
}

/// The shared encoder-loop batch contract is deliberately unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeAttentionBatch {
    pub(crate) q: u64,
    pub(crate) k: u64,
    pub(crate) v: u64,
    pub(crate) out: u64,
    pub(crate) offsets: u64,
    pub(crate) table: u64,
    pub(crate) tokens: usize,
    pub(crate) batch: usize,
    pub(crate) max_q: usize,
    pub(crate) strides: [i64; 12],
}

fn require_u64(manifest: &Value, field: &str, expected: u64) -> Result<(), BackendError> {
    if manifest.get(field).and_then(Value::as_u64) == Some(expected) {
        Ok(())
    } else {
        Err(start(format!(
            "MPNet native manifest field '{field}' must be {expected}"
        )))
    }
}

fn validate_source(manifest: &Value) -> Result<(), BackendError> {
    let source = manifest
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| start("MPNet native manifest lacks the source object"))?;
    if source.len() != 2
        || source.get("pytorch_commit").and_then(Value::as_str) != Some(PYTORCH_COMMIT)
        || source.get("cutlass_commit").and_then(Value::as_str) != Some(CUTLASS_COMMIT)
    {
        return Err(start(format!(
            "MPNet native manifest source must be PyTorch {PYTORCH_COMMIT} with CUTLASS {CUTLASS_COMMIT}"
        )));
    }
    Ok(())
}

fn validate_symbols(manifest: &Value) -> Result<(), BackendError> {
    let symbols = manifest
        .get("symbols")
        .and_then(Value::as_object)
        .ok_or_else(|| start("MPNet native manifest lacks the symbols object"))?;
    if symbols.len() != 4
        || symbols.get("prepare").and_then(Value::as_str) != Some(SYMBOL_PREPARE)
        || symbols.get("validate").and_then(Value::as_str) != Some(SYMBOL_VALIDATE)
        || symbols.get("launch").and_then(Value::as_str) != Some(SYMBOL_LAUNCH)
        || symbols.get("smem_bytes").and_then(Value::as_str) != Some(SYMBOL_SMEM_BYTES)
    {
        return Err(start(
            "MPNet native manifest symbols do not match the versioned adapter exports",
        ));
    }
    Ok(())
}

fn validate_architectures(manifest: &Value, capability: u64) -> Result<(), BackendError> {
    if !SUPPORTED_ARCHITECTURES.contains(&capability) {
        return Err(start(format!(
            "MPNet native attention does not support compute capability sm_{capability}"
        )));
    }
    let architectures = manifest
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| start("MPNet native manifest lacks the architectures array"))?;
    if architectures.len() != SUPPORTED_ARCHITECTURES.len() {
        return Err(start(
            "MPNet native manifest architectures must be [90, 100, 120]",
        ));
    }
    for (index, expected) in SUPPORTED_ARCHITECTURES.iter().enumerate() {
        if architectures[index].as_u64() != Some(*expected) {
            return Err(start(
                "MPNet native manifest architectures must be [90, 100, 120]",
            ));
        }
    }
    Ok(())
}

/// Pure manifest validation. No library is loaded and no device is touched.
pub(crate) fn validate_manifest(
    manifest: &Value,
    dtype: DType,
    capability: u64,
) -> Result<(), BackendError> {
    if !matches!(dtype, DType::Bf16) {
        return Err(start(
            "MPNet native attention is BF16-only; F16 is not supported",
        ));
    }
    require_u64(manifest, "abi_version", MPNET_ATTN_ABI as u64)?;
    if manifest.get("dtype").and_then(Value::as_str) != Some("bf16") {
        return Err(start("MPNet native manifest dtype must be bf16"));
    }
    require_u64(manifest, "num_heads", MPNET_HEADS as u64)?;
    require_u64(manifest, "head_dim", MPNET_HEAD_DIM as u64)?;
    require_u64(manifest, "qkv_row_stride", MPNET_QKV_ROW_STRIDE as u64)?;
    require_u64(manifest, "bias_num_buckets", MPNET_BUCKETS as u64)?;
    if manifest.get("causal").and_then(Value::as_bool) != Some(false) {
        return Err(start("MPNet native manifest must declare causal=false"));
    }
    validate_source(manifest)?;
    validate_symbols(manifest)?;
    validate_architectures(manifest, capability)
}

fn valid_packed_stride(stride: &[i64]) -> bool {
    stride.len() == 3
        && stride[0] == MPNET_QKV_ROW_STRIDE as i64
        && stride[1] == MPNET_HEAD_DIM as i64
        && stride[2] == 1
}

fn valid_out_stride(stride: &[i64]) -> bool {
    stride.len() == 3
        && stride[0] >= MPNET_OUT_WIDTH as i64
        && stride[0] <= i32::MAX as i64
        && stride[0] % 8 == 0
        && stride[1] == MPNET_HEAD_DIM as i64
        && stride[2] == 1
}

fn validate_pointer_offsets(batch: &RelativeAttentionBatch) -> Result<(), BackendError> {
    for (name, pointer) in [
        ("Q", batch.q),
        ("K", batch.k),
        ("V", batch.v),
        ("output", batch.out),
    ]
    .iter()
    .copied()
    {
        if pointer == 0 || pointer % 16 != 0 {
            return Err(infer(format!(
                "invalid MPNet native {name} pointer alignment"
            )));
        }
    }
    if batch.offsets == 0 || !batch.offsets.is_multiple_of(4) {
        return Err(infer(
            "invalid MPNet native cumulative-length pointer alignment",
        ));
    }
    if batch.table == 0 || !batch.table.is_multiple_of(2) {
        return Err(infer("invalid MPNet native bias-table pointer alignment"));
    }
    let expected_k = batch
        .q
        .checked_add(MPNET_K_BYTE_OFFSET)
        .ok_or_else(|| infer("MPNet native K byte offset overflows the address"))?;
    if batch.k != expected_k {
        return Err(infer(
            "MPNet native K byte offset does not match packed QKV layout",
        ));
    }
    let expected_v = batch
        .q
        .checked_add(MPNET_V_BYTE_OFFSET)
        .ok_or_else(|| infer("MPNet native V byte offset overflows the address"))?;
    if batch.v != expected_v {
        return Err(infer(
            "MPNet native V byte offset does not match packed QKV layout",
        ));
    }
    Ok(())
}

fn validate_strides(batch: &RelativeAttentionBatch) -> Result<(), BackendError> {
    if !valid_packed_stride(&batch.strides[0..3])
        || !valid_packed_stride(&batch.strides[3..6])
        || !valid_packed_stride(&batch.strides[6..9])
    {
        return Err(infer(
            "MPNet native packed Q/K/V strides must be [2304, 64, 1]",
        ));
    }
    if !valid_out_stride(&batch.strides[9..12]) {
        return Err(infer(
            "MPNet native output strides must be [>=768, 64, 1] with an 8-element aligned int32 row",
        ));
    }
    for base in [0usize, 3, 6, 9] {
        let extent = (batch.tokens as i128 - 1)
            .checked_mul(batch.strides[base] as i128)
            .and_then(|value| {
                value.checked_add((MPNET_HEADS as i128 - 1) * batch.strides[base + 1] as i128)
            })
            .and_then(|value| {
                value.checked_add((MPNET_HEAD_DIM as i128 - 1) * batch.strides[base + 2] as i128)
            });
        if extent.is_none_or(|value| value > i64::MAX as i128) {
            return Err(infer("MPNet native stride extent overflows int64"));
        }
    }
    Ok(())
}

/// Pure batch validation used by CPU tests before any native call.
pub(crate) fn check_batch(batch: &RelativeAttentionBatch) -> Result<(), BackendError> {
    if batch.tokens == 0
        || batch.batch == 0
        || batch.max_q == 0
        || batch.max_q > MPNET_MAX_SEQ
        || batch.max_q > batch.tokens
        || batch.batch > batch.tokens
    {
        return Err(infer("invalid MPNet native attention geometry"));
    }
    if batch.tokens > i32::MAX as usize {
        return Err(infer(
            "MPNet native attention row count exceeds int32 indexing",
        ));
    }
    if batch.batch > MPNET_MAX_BATCHES {
        return Err(infer(
            "MPNet native attention sequence count exceeds the CUDA grid bound",
        ));
    }
    validate_pointer_offsets(batch)?;
    validate_strides(batch)
}

fn status_error(status: i32) -> BackendError {
    let name = match status {
        MPNET_STATUS_EINVAL => "EINVAL",
        MPNET_STATUS_EUNSUPPORTED => "EUNSUPPORTED",
        MPNET_STATUS_ECUDA => "ECUDA",
        MPNET_STATUS_EUNPREPARED => "EUNPREPARED",
        _ => "unknown",
    };
    infer(format!(
        "MPNet native attention provider failed with status {status} ({name})"
    ))
}

fn load_symbol<T: Copy + 'static>(library: &Library, name: &str) -> Result<T, BackendError> {
    let symbol = CString::new(name)
        .map_err(|_| start("MPNet native symbol name contains an interior NUL"))?;
    // SAFETY: the library is retained by the returned provider, and the
    // manifest requires these exact ABI1 symbol names.
    unsafe {
        library
            .get::<T>(symbol.as_bytes_with_nul())
            .map(|value| *value)
            .map_err(|_| start(format!("MPNet native library lacks {name}")))
    }
}

fn pointer_address(value: u64, name: &str) -> Result<usize, BackendError> {
    usize::try_from(value)
        .map_err(|_| infer(format!("MPNet native {name} pointer does not fit usize")))
}

fn build_geometry(
    stream: &Stream,
    batch: &RelativeAttentionBatch,
) -> Result<Geometry, BackendError> {
    let tokens = u64::try_from(batch.tokens)
        .map_err(|_| infer("MPNet native token count does not fit u64"))?;
    let num_batches = u32::try_from(batch.batch)
        .map_err(|_| infer("MPNet native sequence count does not fit u32"))?;
    let max_seq = u32::try_from(batch.max_q)
        .map_err(|_| infer("MPNet native max query length does not fit u32"))?;
    let out_row_stride = u32::try_from(batch.strides[9])
        .map_err(|_| infer("MPNet native output row stride does not fit u32"))?;
    let q = pointer_address(batch.q, "Q")?;
    let offsets = pointer_address(batch.offsets, "cumulative-length")?;
    let table = pointer_address(batch.table, "bias-table")?;
    let out = pointer_address(batch.out, "output")?;
    Ok(Geometry {
        abi_version: MPNET_ATTN_ABI,
        dtype: MPNET_ATTN_DTYPE_BF16,
        num_heads: MPNET_HEADS as u32,
        head_dim: MPNET_HEAD_DIM as u32,
        qkv_row_stride: MPNET_QKV_ROW_STRIDE as u32,
        q_offset: 0,
        k_offset: (MPNET_HEADS * MPNET_HEAD_DIM) as u32,
        v_offset: (2 * MPNET_HEADS * MPNET_HEAD_DIM) as u32,
        num_batches,
        max_seqlen_q: max_seq,
        max_seqlen_k: max_seq,
        bias_num_buckets: MPNET_BUCKETS as u32,
        bias_num_heads: MPNET_HEADS as u32,
        bias_bucket_stride: MPNET_BIAS_BUCKET_STRIDE,
        bias_head_stride: MPNET_BIAS_HEAD_STRIDE,
        out_row_stride,
        flags: 0,
        qkv_num_rows: tokens,
        out_num_rows: tokens,
        qkv: q as *const c_void,
        seqstart_q: offsets as *const i32,
        seqstart_k: offsets as *const i32,
        bias_table: table as *const c_void,
        out: out as *mut c_void,
        stream: stream.cu_stream().cast(),
    })
}

pub(crate) struct NativeMpnetAttention {
    _validate: Validate,
    launch: Launch,
    context: Arc<Context>,
    plan: Prepared,
    _library: Library,
}

// SAFETY: the engine Mutex serializes calls and teardown, guarantees the
// retained context is current for each launch, and drains queued work before
// releasing the caller-owned buffers and this library.
unsafe impl Send for NativeMpnetAttention {}

impl NativeMpnetAttention {
    pub(crate) fn new(
        context: Arc<Context>,
        root: &Path,
        dtype: DType,
    ) -> Result<Self, BackendError> {
        if !matches!(dtype, DType::Bf16) {
            return Err(start(
                "MPNet native attention is BF16-only; F16 is not supported",
            ));
        }
        if std::mem::size_of::<Prepared>() != MPNET_PREPARED_BYTES as usize
            || std::mem::size_of::<Geometry>() != MPNET_GEOMETRY_BYTES
        {
            return Err(start(
                "MPNet native Rust ABI does not match the standalone adapter",
            ));
        }
        context.bind_to_thread().map_err(start)?;
        let (major, minor) = context.compute_capability().map_err(start)?;
        let capability = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| start("device capability overflow"))?;
        let capability =
            u64::try_from(capability).map_err(|_| start("device capability does not fit u64"))?;
        if !SUPPORTED_ARCHITECTURES.contains(&capability) {
            return Err(start(format!(
                "MPNet native attention does not support compute capability sm_{capability}"
            )));
        }

        let directory = root.join(format!("sm_{capability}"));
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).map_err(start)?)
                .map_err(start)?;
        validate_manifest(&manifest, dtype, capability)?;

        // SAFETY: the root is a trusted cold provider selection, and the
        // manifest above pins the ABI1 BF16 adapter symbol set.
        let library =
            unsafe { Library::new(directory.join("libmpnet_attention.so")) }.map_err(start)?;
        let prepare = load_symbol::<Prepare>(&library, SYMBOL_PREPARE)?;
        let validate = load_symbol::<Validate>(&library, SYMBOL_VALIDATE)?;
        let launch = load_symbol::<Launch>(&library, SYMBOL_LAUNCH)?;
        let smem_bytes = load_symbol::<SmemBytes>(&library, SYMBOL_SMEM_BYTES)?;

        let mut plan = Prepared {
            abi_version: 0,
            struct_bytes: 0,
            shared_memory_bytes: 0,
            context: std::ptr::null_mut(),
        };
        // SAFETY: the cold current context was bound above, and the plan is a
        // writable stack POD owned by this call.
        let status = unsafe { prepare(&mut plan) };
        if status != MPNET_STATUS_OK {
            return Err(start(status_error(status)));
        }
        let expected_context: *mut c_void = context.cu_ctx().cast();
        if plan.abi_version != MPNET_ATTN_ABI
            || plan.struct_bytes != MPNET_PREPARED_BYTES
            || plan.context != expected_context
        {
            return Err(start(
                "MPNet native prepared plan does not match the retained ABI1 context",
            ));
        }
        // SAFETY: the scalar query has no borrowed arguments and the library
        // is retained by the returned provider.
        let queried_smem = unsafe { smem_bytes() };
        if queried_smem == 0 || queried_smem > i32::MAX as usize {
            return Err(start("MPNet native shared-memory query is out of range"));
        }
        if plan.shared_memory_bytes != queried_smem as u64 {
            return Err(start(
                "MPNet native prepared plan shared-memory size does not match the query",
            ));
        }

        Ok(Self {
            _validate: validate,
            launch,
            context,
            plan,
            _library: library,
        })
    }

    /// # Safety
    /// The retained context must be current, the stream must belong to that
    /// exact context, and every Q/K/V/output, cumulative-length and bias-table
    /// allocation must stay live and unmodified through completion.
    pub(crate) unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        batch: &RelativeAttentionBatch,
    ) -> Result<(), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(infer("MPNet native stream belongs to another CUDA context"));
        }
        let expected_context: *mut c_void = self.context.cu_ctx().cast();
        if self.plan.context != expected_context {
            return Err(infer(
                "MPNet native prepared plan belongs to another CUDA context",
            ));
        }
        check_batch(batch)?;
        let geometry = build_geometry(stream, batch)?;
        let plan = self.plan;

        // SAFETY: the context, geometry, pointer offsets, alignment and fixed
        // strides were checked above. The caller's lifetime contract covers
        // every borrowed allocation, and this path performs no allocation,
        // synchronization, binding or function query.
        // SAFETY: as above; the native launch validates the prepared plan and
        // issues the asynchronous kernel on the caller's stream.
        let status = unsafe { (self.launch)(&plan, &geometry) };
        if status != MPNET_STATUS_OK {
            return Err(status_error(status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::mem::{align_of, offset_of, size_of};

    fn manifest() -> Value {
        json!({
            "abi_version": 1,
            "dtype": "bf16",
            "num_heads": MPNET_HEADS,
            "head_dim": MPNET_HEAD_DIM,
            "qkv_row_stride": MPNET_QKV_ROW_STRIDE,
            "bias_num_buckets": MPNET_BUCKETS,
            "causal": false,
            "architectures": [90, 100, 120],
            "source": {
                "pytorch_commit": PYTORCH_COMMIT,
                "cutlass_commit": CUTLASS_COMMIT
            },
            "symbols": {
                "prepare": SYMBOL_PREPARE,
                "validate": SYMBOL_VALIDATE,
                "launch": SYMBOL_LAUNCH,
                "smem_bytes": SYMBOL_SMEM_BYTES
            }
        })
    }

    fn batch() -> RelativeAttentionBatch {
        let q = 0x1_0000u64;
        RelativeAttentionBatch {
            q,
            k: q + MPNET_K_BYTE_OFFSET,
            v: q + MPNET_V_BYTE_OFFSET,
            out: 0x2_0000,
            offsets: 0x3_0000,
            table: 0x4_0000,
            tokens: 8,
            batch: 2,
            max_q: 5,
            strides: [
                MPNET_QKV_ROW_STRIDE as i64,
                MPNET_HEAD_DIM as i64,
                1,
                MPNET_QKV_ROW_STRIDE as i64,
                MPNET_HEAD_DIM as i64,
                1,
                MPNET_QKV_ROW_STRIDE as i64,
                MPNET_HEAD_DIM as i64,
                1,
                MPNET_OUT_WIDTH as i64,
                MPNET_HEAD_DIM as i64,
                1,
            ],
        }
    }

    fn error_text(result: Result<(), BackendError>) -> String {
        match result {
            Ok(()) => String::new(),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn accepts_the_strict_manifest_for_every_supported_architecture() {
        for capability in SUPPORTED_ARCHITECTURES.iter().copied() {
            assert!(validate_manifest(&manifest(), DType::Bf16, capability).is_ok());
        }
    }

    #[test]
    fn rejects_wrong_dtype_and_non_bf16_serving() {
        let f16_manifest = json!({
            "abi_version": 1,
            "dtype": "f16",
            "num_heads": 12,
            "head_dim": 64,
            "qkv_row_stride": 2304,
            "bias_num_buckets": 32,
            "causal": false,
            "architectures": [90, 100, 120],
            "source": {
                "pytorch_commit": PYTORCH_COMMIT,
                "cutlass_commit": CUTLASS_COMMIT
            },
            "symbols": {
                "prepare": SYMBOL_PREPARE,
                "validate": SYMBOL_VALIDATE,
                "launch": SYMBOL_LAUNCH,
                "smem_bytes": SYMBOL_SMEM_BYTES
            }
        });
        assert!(error_text(validate_manifest(&manifest(), DType::F16, 90)).contains("BF16-only"));
        assert!(error_text(validate_manifest(&f16_manifest, DType::Bf16, 90)).contains("bf16"));
    }

    #[test]
    fn rejects_wrong_architectures_and_manifest_profiles() {
        let mut wrong_arches = manifest();
        wrong_arches["architectures"] = json!([90, 100, 128]);
        assert!(
            error_text(validate_manifest(&wrong_arches, DType::Bf16, 90))
                .contains("[90, 100, 120]")
        );

        let mut reordered = manifest();
        reordered["architectures"] = json!([100, 90, 120]);
        assert!(
            error_text(validate_manifest(&reordered, DType::Bf16, 100)).contains("[90, 100, 120]")
        );

        assert!(error_text(validate_manifest(&manifest(), DType::Bf16, 80)).contains("sm_80"));

        let mut wrong_heads = manifest();
        wrong_heads["num_heads"] = json!(8);
        assert!(error_text(validate_manifest(&wrong_heads, DType::Bf16, 90)).contains("num_heads"));

        let mut wrong_abi = manifest();
        wrong_abi["abi_version"] = json!(2);
        assert!(error_text(validate_manifest(&wrong_abi, DType::Bf16, 90)).contains("abi_version"));

        let mut wrong_stride = manifest();
        wrong_stride["qkv_row_stride"] = json!(2048);
        assert!(
            error_text(validate_manifest(&wrong_stride, DType::Bf16, 90))
                .contains("qkv_row_stride")
        );

        let mut wrong_buckets = manifest();
        wrong_buckets["bias_num_buckets"] = json!(16);
        assert!(
            error_text(validate_manifest(&wrong_buckets, DType::Bf16, 90))
                .contains("bias_num_buckets")
        );

        let mut causal = manifest();
        causal["causal"] = json!(true);
        assert!(error_text(validate_manifest(&causal, DType::Bf16, 90)).contains("causal=false"));

        let mut wrong_source = manifest();
        wrong_source["source"]["cutlass_commit"] = json!("deadbeef");
        assert!(error_text(validate_manifest(&wrong_source, DType::Bf16, 90)).contains("CUTLASS"));

        let mut wrong_symbol = manifest();
        wrong_symbol["symbols"]["launch"] = json!("mpnet_attn_launch_f16");
        assert!(error_text(validate_manifest(&wrong_symbol, DType::Bf16, 90)).contains("symbols"));
    }

    #[test]
    fn repr_c_layouts_match_the_adapter_abi_exactly() {
        assert_eq!(MPNET_ATTN_ABI, 1);
        assert_eq!(MPNET_ATTN_DTYPE_BF16, 1);
        assert_eq!(size_of::<Prepared>(), MPNET_PREPARED_BYTES as usize);
        assert_eq!(align_of::<Prepared>(), 8);
        assert_eq!(offset_of!(Prepared, abi_version), 0);
        assert_eq!(offset_of!(Prepared, struct_bytes), 4);
        assert_eq!(offset_of!(Prepared, shared_memory_bytes), 8);
        assert_eq!(offset_of!(Prepared, context), 16);

        assert_eq!(size_of::<Geometry>(), MPNET_GEOMETRY_BYTES);
        assert_eq!(align_of::<Geometry>(), 8);
        assert_eq!(offset_of!(Geometry, num_batches), 32);
        assert_eq!(offset_of!(Geometry, flags), 64);
        assert_eq!(offset_of!(Geometry, qkv_num_rows), 72);
        assert_eq!(offset_of!(Geometry, out_num_rows), 80);
        assert_eq!(offset_of!(Geometry, qkv), 88);
        assert_eq!(offset_of!(Geometry, stream), 128);
    }

    #[test]
    fn accepts_valid_packed_geometry() {
        assert!(check_batch(&batch()).is_ok());
    }

    #[test]
    fn rejects_bad_k_and_v_byte_offsets() {
        let mut bad_k = batch();
        bad_k.k += 16;
        assert!(error_text(check_batch(&bad_k)).contains("K byte offset"));

        let mut bad_v = batch();
        bad_v.v -= 16;
        assert!(error_text(check_batch(&bad_v)).contains("V byte offset"));

        let mut overflow = batch();
        overflow.q = u64::MAX - 15;
        overflow.k = 0x1000;
        assert!(error_text(check_batch(&overflow)).contains("K byte offset"));
    }

    #[test]
    fn rejects_misaligned_and_zero_pointers() {
        let mut zero_q = batch();
        zero_q.q = 0;
        assert!(error_text(check_batch(&zero_q)).contains("Q pointer"));

        let mut bad_out = batch();
        bad_out.out += 1;
        assert!(error_text(check_batch(&bad_out)).contains("output pointer"));

        let mut bad_offsets = batch();
        bad_offsets.offsets += 2;
        assert!(error_text(check_batch(&bad_offsets)).contains("cumulative-length"));

        let mut bad_table = batch();
        bad_table.table += 1;
        assert!(error_text(check_batch(&bad_table)).contains("bias-table"));
    }

    #[test]
    fn rejects_non_fixed_qkv_and_out_widths() {
        let mut wide_q = batch();
        wide_q.strides[0] = 2560;
        assert!(error_text(check_batch(&wide_q)).contains("packed Q/K/V strides"));

        let mut head_stride = batch();
        head_stride.strides[4] = 32;
        assert!(error_text(check_batch(&head_stride)).contains("packed Q/K/V strides"));

        let mut narrow_out = batch();
        narrow_out.strides[9] = MPNET_OUT_WIDTH as i64 - 8;
        assert!(error_text(check_batch(&narrow_out)).contains("output strides"));

        let mut unaligned_out = batch();
        unaligned_out.strides[9] = MPNET_OUT_WIDTH as i64 + 4;
        assert!(error_text(check_batch(&unaligned_out)).contains("output strides"));
    }

    #[test]
    fn rejects_max_q_above_512_and_invalid_sequence_counts() {
        let mut long = batch();
        long.tokens = 1024;
        long.batch = 1;
        long.max_q = 513;
        assert!(error_text(check_batch(&long)).contains("geometry"));

        let mut empty = batch();
        empty.tokens = 0;
        assert!(error_text(check_batch(&empty)).contains("geometry"));

        let mut no_sequences = batch();
        no_sequences.batch = 0;
        assert!(error_text(check_batch(&no_sequences)).contains("geometry"));

        let mut no_queries = batch();
        no_queries.max_q = 0;
        assert!(error_text(check_batch(&no_queries)).contains("geometry"));

        let mut too_many_sequences = batch();
        too_many_sequences.tokens = MPNET_MAX_BATCHES + 1;
        too_many_sequences.batch = MPNET_MAX_BATCHES + 1;
        too_many_sequences.max_q = 1;
        assert!(error_text(check_batch(&too_many_sequences)).contains("sequence count"));

        let mut more_sequences_than_tokens = batch();
        more_sequences_than_tokens.tokens = 1;
        more_sequences_than_tokens.batch = 2;
        more_sequences_than_tokens.max_q = 1;
        assert!(error_text(check_batch(&more_sequences_than_tokens)).contains("geometry"));
    }

    #[test]
    fn rejects_int32_and_int64_bound_violations() {
        let too_many_rows = (i32::MAX as usize) + 1;
        let mut tokens = batch();
        tokens.tokens = too_many_rows;
        tokens.batch = 1;
        tokens.max_q = 512;
        assert!(error_text(check_batch(&tokens)).contains("int32"));

        let mut out_stride = batch();
        out_stride.strides[9] = i32::MAX as i64 + 1;
        assert!(error_text(check_batch(&out_stride)).contains("output strides"));

        let mut extent = batch();
        extent.tokens = i32::MAX as usize;
        extent.batch = 1;
        extent.max_q = 512;
        extent.strides[9] = i32::MAX as i64 - 7;
        assert!(check_batch(&extent).is_ok());
    }
}
