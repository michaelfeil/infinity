// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Narrow integration boundary for the native Gemma2 attention provider.
//!
//! The d256 layers use the AOT FA4 `embedding_gemma2_local256` profile, while
//! the four d512 layers use the vendored Native512 AOT provider (the same fast
//! FlashAttention-2.8.3-derived bridge used by the qualified Gemma4 path).
//! Both paths consume packed ragged noncausal BF16 storage with FP32
//! scores/softmax and the exact Native attention scale of 1.0. There is no
//! fallback between the two geometries and no CPU/Torch attention path.
#![cfg_attr(not(test), allow(dead_code))]

use crate::fa4::{AttentionBatch as Fa4Batch, AttentionSpec as Fa4Spec, DType, Fa4, Scale, Window};
use crate::gemma2_head512::Head512V4Provider;
use crate::gemma2_text::{Gemma2AttentionKind as LayerKind, Gemma2TextConfig};
use crate::{infer, start};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use std::path::Path;
use std::sync::Arc;

const LOCAL_HEADS: usize = 4;
const LOCAL_KV_HEADS: usize = 2;
const LOCAL_HEAD_DIM: usize = 256;
const LOCAL_WINDOW: i32 = 512;

const FULL_HEADS: usize = 4;
const FULL_KV_HEADS: usize = 1;
const FULL_HEAD_DIM: usize = 512;

const LOCAL_FA4_SPEC: Fa4Spec = Fa4Spec {
    heads: LOCAL_HEADS,
    kv_heads: LOCAL_KV_HEADS,
    head_dim: LOCAL_HEAD_DIM,
    causal: false,
    window: Window::Local {
        left: LOCAL_WINDOW,
        right: LOCAL_WINDOW,
    },
    dtype: DType::Bf16,
    scale: Scale::Explicit(1.0),
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2AttentionKind {
    Sliding256,
    Full512,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma2AttentionSpec {
    pub(crate) kind: Gemma2AttentionKind,
    pub(crate) q_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) causal: bool,
    pub(crate) window_left: i32,
    pub(crate) window_right: i32,
    pub(crate) scale: f32,
}

impl Gemma2AttentionSpec {
    pub(crate) fn for_layer(config: &Gemma2TextConfig, index: usize) -> Result<Self, BackendError> {
        let layer = config
            .layers
            .get(index)
            .ok_or_else(|| BackendError::Start("Gemma2 layer index out of range".into()))?;
        let kind = match (layer.kind, layer.head_dim) {
            (LayerKind::Sliding, 256) => Gemma2AttentionKind::Sliding256,
            (LayerKind::Full, 512) => Gemma2AttentionKind::Full512,
            (kind, head_dim) => {
                return Err(BackendError::Start(format!(
                    "Gemma2 unsupported attention geometry {kind:?}/d{head_dim}"
                )));
            }
        };
        Ok(Self {
            kind,
            q_heads: config.num_attention_heads,
            kv_heads: layer.kv_heads,
            head_dim: layer.head_dim,
            causal: false,
            window_left: layer.window_left,
            window_right: layer.window_right,
            scale: 1.0,
        })
    }
}

/// Exact Native d256 local geometry. Local extents are symmetric 512/512 and
/// the scale is the model's explicit 1.0; no other local geometry is admitted.
fn is_local_spec(spec: &Gemma2AttentionSpec) -> bool {
    spec.kind == Gemma2AttentionKind::Sliding256
        && spec.q_heads == LOCAL_HEADS
        && spec.kv_heads == LOCAL_KV_HEADS
        && spec.head_dim == LOCAL_HEAD_DIM
        && !spec.causal
        && spec.window_left == LOCAL_WINDOW
        && spec.window_right == LOCAL_WINDOW
        && spec.scale == 1.0
}

/// Exact Native d512 global geometry. Global layers use left/right `-1` and the
/// same explicit scale of 1.0.
fn is_full_spec(spec: &Gemma2AttentionSpec) -> bool {
    spec.kind == Gemma2AttentionKind::Full512
        && spec.q_heads == FULL_HEADS
        && spec.kv_heads == FULL_KV_HEADS
        && spec.head_dim == FULL_HEAD_DIM
        && !spec.causal
        && spec.window_left == -1
        && spec.window_right == -1
        && spec.scale == 1.0
}

fn local_fa4_available() -> bool {
    Fa4::supports(&LOCAL_FA4_SPEC)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma2AttentionBatch {
    pub(crate) q: u64,
    pub(crate) k: u64,
    pub(crate) v: u64,
    pub(crate) out: u64,
    pub(crate) out_bytes: u64,
    /// Work-owned F32 `[query_heads, total_q]` softmax-LSE slab.
    pub(crate) lse: u64,
    pub(crate) lse_bytes: u64,
    pub(crate) q_offsets: u64,
    pub(crate) q_offsets_bytes: usize,
    pub(crate) k_offsets: u64,
    pub(crate) k_offsets_bytes: usize,
    pub(crate) total_q: usize,
    pub(crate) total_k: usize,
    pub(crate) batch: usize,
    pub(crate) max_q: usize,
    /// Explicit K extent. Same-span Gemma2 may pass max_q, but the boundary
    /// never infers it for cross/prefix attention.
    pub(crate) max_k: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) scale: f64,
    /// q_row,q_head,q_d, k_row,k_head,k_d, v_row,v_head,v_d,
    /// o_row,o_head,o_d. Element strides; d strides are one.
    pub(crate) strides: [i64; 12],
}

/// External provider contract. Implementations must not bind the CUDA context,
/// allocate, query descriptors, or read manifests in `forward_batch_bound`; the
/// engine binds once before the request and retains every pointer through the
/// stream completion/error path.
pub(crate) trait Gemma2AttentionProvider: Send + Sync {
    fn supports(&self, spec: &Gemma2AttentionSpec) -> bool;
    /// # Safety
    /// Pointers must be live, aligned BF16 storage in the provider context;
    /// offsets must describe valid ragged spans; output must not alias inputs.
    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        spec: &Gemma2AttentionSpec,
        batch: &Gemma2AttentionBatch,
    ) -> Result<(), BackendError>;
}

/// Concrete provider for the exact EmbeddingGemma2 text geometries. The FA4
/// handle is constructed once for the local d256 profile; the four full d512
/// layers use the vendored, cold-prepared Native512 AOT provider.
pub(crate) struct NativeGemma2Attention {
    context: Arc<Context>,
    local: Fa4,
    full: Head512V4Provider,
}

impl NativeGemma2Attention {
    pub(crate) fn new(
        context: Arc<Context>,
        fa4_root: &Path,
        head512_root: &Path,
        architecture: &str,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, BackendError> {
        if !local_fa4_available() {
            return Err(start(
                "Gemma2 local d256 attention requires the FA4 embedding_gemma2_local256 profile",
            ));
        }
        let local = Fa4::new(Arc::clone(&context), fa4_root, LOCAL_FA4_SPEC)?;
        let full = Head512V4Provider::load_prepared(
            head512_root,
            architecture,
            max_optin_shared_memory_bytes,
            Arc::clone(&context),
        )
        .map_err(|error| start(format!("Gemma2 full d512 attention: {error}")))?;
        Ok(Self {
            context,
            local,
            full,
        })
    }

    fn supports_spec(spec: &Gemma2AttentionSpec) -> bool {
        (is_local_spec(spec) && local_fa4_available()) || is_full_spec(spec)
    }

    fn launch_full(
        &self,
        stream: &Stream,
        spec: &Gemma2AttentionSpec,
        batch: &Gemma2AttentionBatch,
    ) -> Result<(), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(infer(
                "Gemma2 full attention stream belongs to another CUDA context",
            ));
        }
        if !self.full.supports(spec) {
            return Err(infer(
                "Gemma2 full attention provider does not declare the exact d512 spec",
            ));
        }
        // SAFETY: the provider was prepared on this context, and the caller
        // retains every packed region through stream completion or error drain.
        unsafe { self.full.forward_batch_bound(stream, spec, batch) }
    }
}

impl Gemma2AttentionProvider for NativeGemma2Attention {
    fn supports(&self, spec: &Gemma2AttentionSpec) -> bool {
        Self::supports_spec(spec)
    }

    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        spec: &Gemma2AttentionSpec,
        batch: &Gemma2AttentionBatch,
    ) -> Result<(), BackendError> {
        let local = is_local_spec(spec);
        let full = is_full_spec(spec);
        if !local && !full {
            return Err(infer(
                "Gemma2 attention spec is not an exact native geometry",
            ));
        }
        validate_batch(spec, batch)?;
        if local {
            if !local_fa4_available() {
                return Err(infer("Gemma2 local d256 attention profile is unavailable"));
            }
            let attention = Fa4Batch {
                q: batch.q,
                k: batch.k,
                v: batch.v,
                out: batch.out,
                q_offsets: batch.q_offsets,
                k_offsets: batch.k_offsets,
                q_tokens: batch.total_q,
                k_tokens: batch.total_k,
                batch: batch.batch,
                max_q: batch.max_q,
                strides: batch.strides,
            };
            // SAFETY: validate_batch accepted the ragged geometry, the FA4
            // handle was built for this exact local spec, and the caller keeps
            // every pointer live until stream completion.
            return unsafe { self.local.forward_batch_bound(stream, &attention) };
        }
        // SAFETY: validate_batch and the strict full geometry checks above
        // accepted the packed shared-span batch; the caller keeps every pointer
        // live until stream completion.
        self.launch_full(stream, spec, batch)
    }
}

/// Rust-side consumer guard before any provider launch. Strides describe
/// physical padding and may exceed the logical head width, but never underflow
/// it or the d=1 contract. `max_k` is explicit and never inferred from `max_q`.
pub(crate) fn validate_batch(
    spec: &Gemma2AttentionSpec,
    batch: &Gemma2AttentionBatch,
) -> Result<(), BackendError> {
    if spec.causal {
        return Err(BackendError::Inference(
            "Gemma2 text attention must be noncausal".into(),
        ));
    }
    if batch.q == 0
        || batch.k == 0
        || batch.v == 0
        || batch.out == 0
        || batch.lse == 0
        || batch.q_offsets == 0
        || batch.k_offsets == 0
    {
        return Err(BackendError::Inference(
            "Gemma2 attention contains a null pointer".into(),
        ));
    }
    if batch.total_q == 0
        || batch.total_k == 0
        || batch.batch == 0
        || batch.max_q == 0
        || batch.max_k == 0
        || batch.heads == 0
        || batch.kv_heads == 0
        || batch.heads != spec.q_heads
        || batch.kv_heads != spec.kv_heads
    {
        return Err(BackendError::Inference(
            "Gemma2 attention token/batch/head geometry is incomplete".into(),
        ));
    }
    if batch.batch > batch.total_q
        || batch.batch > batch.total_k
        || batch.max_q > batch.total_q
        || batch.max_k > batch.total_k
        || !batch.heads.is_multiple_of(batch.kv_heads)
    {
        return Err(BackendError::Inference(
            "Gemma2 attention extents are inconsistent".into(),
        ));
    }
    let offset_bytes = batch
        .batch
        .checked_add(1)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| BackendError::Inference("Gemma2 offset bytes overflow".into()))?;
    if batch.q_offsets_bytes < offset_bytes || batch.k_offsets_bytes < offset_bytes {
        return Err(BackendError::Inference(
            "Gemma2 cumulative offset capacity is too small".into(),
        ));
    }
    let lse_elements = batch
        .total_q
        .checked_mul(batch.heads)
        .ok_or_else(|| BackendError::Inference("Gemma2 LSE elements overflow".into()))?;
    let lse_bytes = u64::try_from(lse_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| BackendError::Inference("Gemma2 LSE bytes overflow".into()))?;
    if batch.lse_bytes < lse_bytes {
        return Err(BackendError::Inference(
            "Gemma2 Work LSE capacity is too small".into(),
        ));
    }
    let out_elements = batch
        .total_q
        .checked_mul(batch.heads)
        .and_then(|elements| elements.checked_mul(spec.head_dim))
        .ok_or_else(|| BackendError::Inference("Gemma2 output elements overflow".into()))?;
    let out_bytes = u64::try_from(out_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| BackendError::Inference("Gemma2 output bytes overflow".into()))?;
    if batch.out_bytes < out_bytes {
        return Err(BackendError::Inference(
            "Gemma2 Work output capacity is too small".into(),
        ));
    }
    if !batch.scale.is_finite()
        || batch.scale <= 0.0
        || !(batch.scale as f32).is_finite()
        || (batch.scale as f32) <= 0.0
    {
        return Err(BackendError::Inference(
            "Gemma2 attention scale must be finite and positive as f32".into(),
        ));
    }
    let d = i64::try_from(spec.head_dim)
        .map_err(|_| BackendError::Inference("Gemma2 head_dim overflow".into()))?;
    for (tokens, heads, strides) in [
        (batch.total_q, batch.heads, &batch.strides[0..3]),
        (batch.total_k, batch.kv_heads, &batch.strides[3..6]),
        (batch.total_k, batch.kv_heads, &batch.strides[6..9]),
        (batch.total_q, batch.heads, &batch.strides[9..12]),
    ] {
        let logical = i64::try_from(heads)
            .ok()
            .and_then(|heads| heads.checked_mul(d))
            .ok_or_else(|| BackendError::Inference("Gemma2 stride width overflow".into()))?;
        if strides[2] != 1 || strides[0] < logical || strides[1] < d {
            return Err(BackendError::Inference(
                "Gemma2 attention row/head/d stride is invalid".into(),
            ));
        }
        let extent = (tokens as i128 - 1) * strides[0] as i128
            + (heads as i128 - 1) * strides[1] as i128
            + (d as i128 - 1);
        if extent < 0 || extent > i64::MAX as i128 {
            return Err(BackendError::Inference(
                "Gemma2 attention stride extent overflow".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> Gemma2AttentionSpec {
        Gemma2AttentionSpec {
            kind: Gemma2AttentionKind::Sliding256,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 256,
            causal: false,
            window_left: 512,
            window_right: 512,
            scale: 1.0,
        }
    }

    fn batch() -> Gemma2AttentionBatch {
        Gemma2AttentionBatch {
            q: 1,
            k: 2,
            v: 3,
            out: 4,
            out_bytes: 7 * 4 * 256 * 2,
            lse: 7,
            lse_bytes: 7 * 4 * 4,
            q_offsets: 5,
            q_offsets_bytes: 8,
            k_offsets: 6,
            k_offsets_bytes: 8,
            total_q: 7,
            total_k: 7,
            batch: 1,
            max_q: 7,
            max_k: 7,
            heads: 4,
            kv_heads: 2,
            scale: 1.0,
            strides: [1024, 256, 1, 512, 256, 1, 512, 256, 1, 1024, 256, 1],
        }
    }

    #[test]
    fn accepts_exact_and_physically_padded_strides() {
        assert!(validate_batch(&spec(), &batch()).is_ok());
        let mut padded = batch();
        padded.strides = [2048, 512, 1, 768, 512, 1, 768, 512, 1, 2048, 512, 1];
        assert!(validate_batch(&spec(), &padded).is_ok());
    }

    #[test]
    fn rejects_partial_geometry_lse_capacity_and_stride_underflow() {
        for mutate in [
            (|b: &mut Gemma2AttentionBatch| b.batch = 0) as fn(&mut Gemma2AttentionBatch),
            |b: &mut Gemma2AttentionBatch| b.total_q = 0,
            |b: &mut Gemma2AttentionBatch| b.total_k = 6,
            |b: &mut Gemma2AttentionBatch| b.max_q = 8,
            |b: &mut Gemma2AttentionBatch| b.max_k = 0,
            |b: &mut Gemma2AttentionBatch| b.lse = 0,
            |b: &mut Gemma2AttentionBatch| b.out_bytes = 7 * 4 * 256 * 2 - 1,
            |b: &mut Gemma2AttentionBatch| b.lse_bytes = 7 * 4 * 4 - 1,
            |b: &mut Gemma2AttentionBatch| b.q_offsets_bytes = 7,
            |b: &mut Gemma2AttentionBatch| b.strides[0] = 1023,
            |b: &mut Gemma2AttentionBatch| b.strides[4] = 255,
            |b: &mut Gemma2AttentionBatch| b.strides[8] = 2,
            |b: &mut Gemma2AttentionBatch| b.q = 0,
        ] {
            let mut candidate = batch();
            mutate(&mut candidate);
            assert!(validate_batch(&spec(), &candidate).is_err());
        }
        let mut causal = spec();
        causal.causal = true;
        assert!(validate_batch(&causal, &batch()).is_err());
    }

    #[test]
    fn explicit_max_k_is_independent_of_max_q() {
        let mut cross = batch();
        cross.total_k = 11;
        cross.max_k = 11;
        assert!(validate_batch(&spec(), &cross).is_ok());
        cross.max_k = 0;
        assert!(validate_batch(&spec(), &cross).is_err());
    }

    fn full_spec() -> Gemma2AttentionSpec {
        Gemma2AttentionSpec {
            kind: Gemma2AttentionKind::Full512,
            q_heads: 4,
            kv_heads: 1,
            head_dim: 512,
            causal: false,
            window_left: -1,
            window_right: -1,
            scale: 1.0,
        }
    }

    fn full_batch() -> Gemma2AttentionBatch {
        Gemma2AttentionBatch {
            q: 1,
            k: 2,
            v: 3,
            out: 4,
            out_bytes: 7 * 4 * 512 * 2,
            lse: 7,
            lse_bytes: 7 * 4 * 4,
            q_offsets: 5,
            q_offsets_bytes: 8,
            k_offsets: 5,
            k_offsets_bytes: 8,
            total_q: 7,
            total_k: 7,
            batch: 1,
            max_q: 7,
            max_k: 7,
            heads: 4,
            kv_heads: 1,
            scale: 1.0,
            strides: [2048, 512, 1, 512, 512, 1, 512, 512, 1, 2048, 512, 1],
        }
    }

    #[test]
    fn native_dispatch_admits_only_exact_local_and_full_geometries() {
        assert!(NativeGemma2Attention::supports_spec(&spec()));
        assert!(NativeGemma2Attention::supports_spec(&full_spec()));

        for mutate in [
            (|s: &mut Gemma2AttentionSpec| s.window_left = 511) as fn(&mut Gemma2AttentionSpec),
            |s: &mut Gemma2AttentionSpec| s.window_right = 511,
            |s: &mut Gemma2AttentionSpec| s.kv_heads = 1,
            |s: &mut Gemma2AttentionSpec| s.head_dim = 128,
            |s: &mut Gemma2AttentionSpec| s.scale = 0.5,
            |s: &mut Gemma2AttentionSpec| s.causal = true,
        ] {
            let mut local = spec();
            mutate(&mut local);
            assert!(!NativeGemma2Attention::supports_spec(&local));
        }
        for mutate in [
            (|s: &mut Gemma2AttentionSpec| s.kv_heads = 2) as fn(&mut Gemma2AttentionSpec),
            |s: &mut Gemma2AttentionSpec| s.window_left = 512,
            |s: &mut Gemma2AttentionSpec| s.window_right = 512,
            |s: &mut Gemma2AttentionSpec| s.head_dim = 256,
            |s: &mut Gemma2AttentionSpec| s.scale = 4.0,
            |s: &mut Gemma2AttentionSpec| s.causal = true,
            |s: &mut Gemma2AttentionSpec| s.kind = Gemma2AttentionKind::Sliding256,
        ] {
            let mut full = full_spec();
            mutate(&mut full);
            assert!(!NativeGemma2Attention::supports_spec(&full));
        }
    }

    #[test]
    fn full_batch_validation_checks_geometry_and_capacity() {
        assert!(validate_batch(&full_spec(), &full_batch()).is_ok());
        let mut padded = full_batch();
        padded.strides = [4096, 1024, 1, 1024, 1024, 1, 1024, 1024, 1, 4096, 1024, 1];
        assert!(validate_batch(&full_spec(), &padded).is_ok());
        for mutate in [
            (|b: &mut Gemma2AttentionBatch| b.kv_heads = 2) as fn(&mut Gemma2AttentionBatch),
            |b: &mut Gemma2AttentionBatch| b.heads = 8,
            |b: &mut Gemma2AttentionBatch| b.out_bytes = 7 * 4 * 512 * 2 - 1,
            |b: &mut Gemma2AttentionBatch| b.lse_bytes = 7 * 4 * 4 - 1,
            |b: &mut Gemma2AttentionBatch| b.max_k = 0,
            |b: &mut Gemma2AttentionBatch| b.strides[10] = 511,
        ] {
            let mut candidate = full_batch();
            mutate(&mut candidate);
            assert!(validate_batch(&full_spec(), &candidate).is_err());
        }
    }
}
