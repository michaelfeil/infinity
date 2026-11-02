// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Adapter from the Gemma2 typed attention boundary to the vendored
//! `infinity-head512-aot` provider API2/ABI2 source (`vendor/head512-aot`).
//!
//! Cold selection uses caller-supplied actual architecture and opt-in shared
//! memory. `prepare_current_context` runs once before the provider is published
//! behind `Arc`; enqueue performs no allocation, synchronization or device query.
#![allow(dead_code)]

use crate::gemma2_attention::{
    Gemma2AttentionBatch, Gemma2AttentionKind, Gemma2AttentionProvider, Gemma2AttentionSpec,
};
use cuda_core::{Context, Stream};
use infinity_head512_aot::{
    DType, DeviceRegion, Head512Batch, Head512Spec, Head512Work, KernelVariant, Provider, Window,
};
use infinity_v3_backend_core::BackendError;
use std::{fmt, path::Path, sync::Arc};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Head512V4ColdError {
    Context(String),
    Provider(String),
    NotQualified,
}

impl fmt::Display for Head512V4ColdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(message) => write!(formatter, "head512 context bind failed: {message}"),
            Self::Provider(message) => {
                write!(formatter, "head512 provider cold load failed: {message}")
            }
            Self::NotQualified => formatter.write_str(
                "head512 provider ABI2 does not declare BF16 noncausal global d512 H4/KV1",
            ),
        }
    }
}

impl std::error::Error for Head512V4ColdError {}

/// Actual API2/ABI2 provider retained for the lifetime of the executor.
/// `context` is retained to keep the prepared context identity alive.
pub(crate) struct Head512V4Provider {
    provider: Provider,
    _context: Arc<Context>,
}

impl Head512V4Provider {
    pub(crate) fn load_prepared(
        root: &Path,
        architecture: &str,
        max_optin_shared_memory_bytes: u64,
        context: Arc<Context>,
    ) -> Result<Self, Head512V4ColdError> {
        context
            .bind_to_thread()
            .map_err(|error| Head512V4ColdError::Context(error.to_string()))?;
        let mut provider =
            Provider::load_for_device(root, architecture, max_optin_shared_memory_bytes)
                .map_err(|error| Head512V4ColdError::Provider(error.to_string()))?;
        // SAFETY: the caller owns this one current context and keeps it bound
        // and alive until every provider/work owner has drained.
        unsafe {
            provider
                .prepare_current_context()
                .map_err(|error| Head512V4ColdError::Provider(error.to_string()))?;
        }
        match provider.supports(head512_spec_for_kind(Gemma2AttentionKind::Full512)) {
            Ok(true) => Ok(Self {
                provider,
                _context: context,
            }),
            Ok(false) => Err(Head512V4ColdError::NotQualified),
            Err(error) => Err(Head512V4ColdError::Provider(error.to_string())),
        }
    }

    pub(crate) fn kernel_variant(&self) -> KernelVariant {
        self.provider.kernel_variant()
    }

    pub(crate) fn max_optin_shared_memory_bytes(&self) -> u64 {
        self.provider.capabilities().max_optin_shared_memory_bytes
    }
}

fn head512_spec_for_kind(kind: Gemma2AttentionKind) -> Head512Spec {
    let _ = kind;
    Head512Spec {
        dtype: DType::Bf16,
        causal: false,
        window: Window::Global,
    }
}

fn embed_spec_to_head512(spec: &Gemma2AttentionSpec) -> Option<Head512Spec> {
    (spec.kind == Gemma2AttentionKind::Full512
        && spec.q_heads == 4
        && spec.kv_heads == 1
        && spec.head_dim == 512
        && !spec.causal
        && spec.window_left == -1
        && spec.window_right == -1
        && spec.scale == 1.0)
        .then_some(head512_spec_for_kind(spec.kind))
}

fn embed_batch_to_head512(batch: &Gemma2AttentionBatch) -> Result<Head512Batch, BackendError> {
    Ok(Head512Batch {
        q: batch.q,
        k: batch.k,
        v: batch.v,
        q_offsets: batch.q_offsets,
        q_offsets_bytes: batch.q_offsets_bytes,
        k_offsets: batch.k_offsets,
        k_offsets_bytes: batch.k_offsets_bytes,
        total_q: batch.total_q,
        total_k: batch.total_k,
        batch: batch.batch,
        max_q: batch.max_q,
        max_k: batch.max_k,
        heads: batch.heads,
        kv_heads: batch.kv_heads,
        scale: batch.scale,
        dtype: DType::Bf16,
        causal: false,
        window: Window::Global,
        strides: batch.strides,
    })
}

impl Gemma2AttentionProvider for Head512V4Provider {
    fn supports(&self, spec: &Gemma2AttentionSpec) -> bool {
        let Some(spec) = embed_spec_to_head512(spec) else {
            return false;
        };
        self.provider.supports(spec).unwrap_or(false)
    }

    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        spec: &Gemma2AttentionSpec,
        batch: &Gemma2AttentionBatch,
    ) -> Result<(), BackendError> {
        let Some(head512_spec) = embed_spec_to_head512(spec) else {
            return Err(BackendError::Inference(
                "head512 ABI2 accepts only BF16 H4/KV1 d512 noncausal global scale 1.0".into(),
            ));
        };
        if !self
            .provider
            .supports(head512_spec)
            .map_err(|error| BackendError::Inference(error.to_string()))?
        {
            return Err(BackendError::Inference(
                "head512 ABI2 provider does not declare the Gemma2 full-attention spec".into(),
            ));
        }
        let provider_batch = embed_batch_to_head512(batch)?;
        let out = DeviceRegion::new(batch.out, batch.out_bytes)
            .map_err(|error| BackendError::Inference(error.to_string()))?;
        let lse = DeviceRegion::new(batch.lse, batch.lse_bytes)
            .map_err(|error| BackendError::Inference(error.to_string()))?;
        let mut work = Head512Work::new(out, lse);
        // SAFETY: every region is owned by the caller's outer Work and remains
        // live through stream completion or provider error drain. The provider
        // was prepared on this context before it was shared behind Arc.
        unsafe {
            self.provider
                .enqueue(&provider_batch, &mut work, stream.cu_stream() as u64)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            out: 0x4000,
            out_bytes: 7 * 4 * 512 * 2,
            lse: 0x5000,
            lse_bytes: 7 * 4 * 4,
            q_offsets: 0x6000,
            q_offsets_bytes: 16,
            k_offsets: 0x7000,
            k_offsets_bytes: 16,
            total_q: 7,
            total_k: 9,
            batch: 3,
            max_q: 5,
            max_k: 9,
            heads: 4,
            kv_heads: 1,
            scale: 1.0,
            strides: [2048, 512, 1, 512, 512, 1, 512, 512, 1, 2048, 512, 1],
        }
    }

    #[test]
    fn maps_only_native_full512_spec_and_explicit_batch_axes() {
        let spec = embed_spec_to_head512(&full_spec()).expect("full spec");
        assert_eq!(spec.dtype, DType::Bf16);
        assert!(!spec.causal);
        assert_eq!(spec.window, Window::Global);
        let batch = embed_batch_to_head512(&full_batch()).expect("batch mapping");
        assert_eq!(batch.max_q, 5);
        assert_eq!(batch.max_k, 9);
        assert_eq!(batch.total_q, 7);
        assert_eq!(batch.total_k, 9);
        assert_eq!(batch.dtype, DType::Bf16);
        assert_eq!(batch.strides[2], 1);
        assert_eq!(batch.strides[5], 1);
        assert_eq!(batch.strides[8], 1);
        assert_eq!(batch.strides[11], 1);

        let mut local = full_spec();
        local.kind = Gemma2AttentionKind::Sliding256;
        assert!(embed_spec_to_head512(&local).is_none());
        let mut wrong_gqa = full_spec();
        wrong_gqa.kv_heads = 2;
        assert!(embed_spec_to_head512(&wrong_gqa).is_none());
        let mut g4 = full_spec();
        g4.q_heads = 8;
        g4.kv_heads = 2;
        assert!(embed_spec_to_head512(&g4).is_none());
        let mut causal = full_spec();
        causal.causal = true;
        assert!(embed_spec_to_head512(&causal).is_none());
        let mut local_window = full_spec();
        local_window.window_left = 512;
        local_window.window_right = 512;
        assert!(embed_spec_to_head512(&local_window).is_none());
    }

    #[test]
    fn actual_v4_variant_selection_is_cold_and_shared_memory_explicit() {
        assert_eq!(
            KernelVariant::for_architecture("sm_90").unwrap(),
            KernelVariant::Native
        );
        assert_eq!(
            KernelVariant::for_architecture("sm_100").unwrap(),
            KernelVariant::Native
        );
        assert_eq!(
            KernelVariant::for_architecture("sm_120").unwrap(),
            KernelVariant::Compact
        );
        assert_eq!(
            KernelVariant::Native.dynamic_shared_memory_bytes(),
            infinity_head512_aot::NATIVE_DYNAMIC_SHARED_MEMORY_BYTES
        );
        assert_eq!(
            KernelVariant::Compact.dynamic_shared_memory_bytes(),
            infinity_head512_aot::COMPACT_DYNAMIC_SHARED_MEMORY_BYTES
        );
    }
}
