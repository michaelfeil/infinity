// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Qwen3.5 integration for the existing Clef V2 joint-head leaves.
//!
//! This wrapper owns the cold `ClefHead`, the one BF16 lexical owner used by
//! the Native lexical-prior path, and a request wrapper that deliberately
//! leaves completion with the outer Work and its existing pinned input lease.

use crate::{
    clef_decision::{
        output_elements, request_output_count, validate_runtime_dtype, ClefConfig, MAX_CLEF_FIELDS,
        MAX_CLEF_OPTIONS, MAX_CLEF_OPTION_TOKENS,
    },
    clef_decision_head::ClefHead,
    clef_decision_plan::ClefRequestPlan,
    clef_decision_workspace::{ClefCapacity, ClefWorkspace},
    cublas::Blas,
    decision_head::PackedDecisionBatch,
    infer,
    model::WeightSource,
    tensor::Bf16,
    upload::WeightUploader,
    WorkKey,
};
use cuda_core::{Context, DeviceBuffer, Stream};
use infinity_v3_backend_core::{BackendError, DecisionInput, DecisionOutput};
use std::{mem::ManuallyDrop, path::Path, sync::Arc};

enum ClefLexical {
    BorrowedEmbed,
    Owned(ManuallyDrop<DeviceBuffer<Bf16>>),
}

pub(crate) struct ClefQ35Head {
    head: ClefHead,
    lexical: ClefLexical,
}

impl ClefQ35Head {
    /// Cold-load the joint head once and upload an untied LM head once. Tied
    /// checkpoints borrow the already uploaded Qwen3.5 embed_tokens owner.
    pub(crate) fn load(
        path: &Path,
        backbone_hidden: usize,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        fa4_root: &Path,
        lm_head: Option<&WeightSource>,
    ) -> Result<Self, BackendError> {
        validate_runtime_dtype(crate::clef_decision::CLEF_RUNTIME_DTYPE)?;
        let config = ClefConfig::load(path, backbone_hidden)?;
        let head = ClefHead::load(path, config, context, stream, fa4_root)?;
        let lexical = match lm_head {
            Some(source) => {
                let mut uploader = WeightUploader::new(stream.clone())?;
                let buffer = uploader.upload_typed::<Bf16>(&[source])?;
                uploader.finish()?;
                ClefLexical::Owned(ManuallyDrop::new(buffer))
            }
            None => ClefLexical::BorrowedEmbed,
        };
        Ok(Self { head, lexical })
    }

    pub(crate) fn config(&self) -> &ClefConfig {
        &self.head.config
    }

    pub(crate) fn head(&self) -> &ClefHead {
        &self.head
    }

    pub(crate) fn lexical<'a>(&'a self, embed: &'a DeviceBuffer<Bf16>) -> &'a DeviceBuffer<Bf16> {
        match &self.lexical {
            ClefLexical::BorrowedEmbed => embed,
            ClefLexical::Owned(buffer) => buffer,
        }
    }

    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        ids: &[u32],
        requests: &[DecisionInput],
        stream: &Arc<Stream>,
    ) -> Result<ClefWarmPrepared, BackendError> {
        self.head
            .prepare_plan(batch, ids, requests, stream)
            .map(ClefWarmPrepared::new)
    }
}

impl Drop for ClefQ35Head {
    fn drop(&mut self) {
        if self.head.stream.synchronize().is_ok() {
            if let ClefLexical::Owned(buffer) = &mut self.lexical {
                // SAFETY: the cold stream drained, so the lexical upload is idle.
                unsafe { ManuallyDrop::drop(buffer) };
            }
            self.lexical = ClefLexical::BorrowedEmbed;
        } else if let ClefLexical::Owned(buffer) = &mut self.lexical {
            let buffer = unsafe { ManuallyDrop::take(buffer) };
            std::mem::forget(buffer);
            self.lexical = ClefLexical::BorrowedEmbed;
        }
    }
}

/// CPU-only request metadata. The pinned HostIo input lease owns the H2D source;
/// no GPU operation borrows this plan's Vec. Cancellation and validation errors
/// therefore release it normally without draining a stream or leaking owners.
pub(crate) struct ClefWarmPrepared {
    plan: ClefRequestPlan,
}

impl ClefWarmPrepared {
    fn new(plan: ClefRequestPlan) -> Self {
        Self { plan }
    }

    pub(crate) fn metadata(&self) -> &[u32] {
        &self.plan.metadata
    }

    pub(crate) fn metadata_words(&self) -> usize {
        self.plan.metadata_len()
    }

    pub(crate) fn output_elements(&self) -> Result<usize, BackendError> {
        self.plan.output_elements()
    }

    pub(crate) fn plan(&self) -> &ClefRequestPlan {
        &self.plan
    }

    pub(crate) fn finish(&mut self, host: &mut [f32]) -> Result<Vec<DecisionOutput>, BackendError> {
        crate::clef_decision_finish::finish_outputs(&self.plan, host)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_bound(
    head: &ClefQ35Head,
    hidden: &DeviceBuffer<Bf16>,
    embed: &DeviceBuffer<Bf16>,
    workspace: &mut ClefWorkspace,
    prepared: &mut ClefWarmPrepared,
    blas: &Blas,
    stream: &Stream,
    host_out: *mut f32,
    metadata_base: u64,
) -> Result<(), BackendError> {
    let lexical = head.lexical(embed);
    // SAFETY: the outer Work/Engine retains the head, lexical owner, workspace,
    // prepared host tail and output lease until the completion fence.
    unsafe {
        crate::clef_decision_exec::enqueue_bound_plan(
            head.head(),
            hidden,
            lexical,
            workspace,
            prepared.plan(),
            blas,
            stream,
            host_out,
            metadata_base,
        )
    }
}

pub(crate) fn capacity(key: WorkKey, config: &ClefConfig) -> Result<ClefCapacity, BackendError> {
    let requests = key
        .1
        .checked_sub(1)
        .ok_or_else(|| infer("Clef Work key has no sequence capacity"))?;
    ClefCapacity::new(
        key.0,
        requests,
        key.0.min(MAX_CLEF_FIELDS),
        key.0.min(MAX_CLEF_OPTIONS),
        key.0.min(MAX_CLEF_OPTION_TOKENS),
        config.hidden,
        config.width,
        config.feedforward,
    )
}

pub(crate) fn metadata_words(config: &ClefConfig, key: WorkKey) -> Result<usize, BackendError> {
    Ok(capacity(key, config)?.metadata_words())
}

pub(crate) fn output_count(requests: &[DecisionInput]) -> Result<usize, BackendError> {
    output_elements(requests)
}

pub(crate) fn request_count(request: &DecisionInput) -> Result<usize, BackendError> {
    request_output_count(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{clef_decision_plan::plan_requests, decision_head::PackedDecisionBatch};
    use infinity_v3_backend_core::ClefField;

    const KEY: WorkKey = (32, 4, 2);

    fn config() -> ClefConfig {
        ClefConfig {
            hidden: 4,
            width: 8,
            heads: 1,
            feedforward: 16,
            routing_layers: 1,
            layers: 1,
        }
    }

    fn batch() -> PackedDecisionBatch<'static> {
        PackedDecisionBatch::new(&[0, 3, 5]).unwrap()
    }

    fn requests() -> [DecisionInput; 2] {
        [
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 0,
                    question: (0, 1),
                    options: vec![(1, 2), (2, 3)],
                }],
            },
            DecisionInput::Warmup,
        ]
    }

    #[test]
    fn used_path_capacity_tracks_work_key_and_metadata_words() {
        let capacity = capacity(KEY, &config()).unwrap();
        assert_eq!(capacity.tokens(), KEY.0);
        assert_eq!(capacity.requests(), KEY.1 - 1);
        assert_eq!(capacity.fields(), KEY.0.min(MAX_CLEF_FIELDS));
        assert_eq!(capacity.options(), KEY.0.min(MAX_CLEF_OPTIONS));
        assert_eq!(capacity.option_tokens(), KEY.0.min(MAX_CLEF_OPTION_TOKENS));
        assert!(capacity.metadata_words() > 0);
        assert!(capacity.device_bytes().unwrap() > capacity.metadata_words() * 4);
        assert_eq!(
            metadata_words(&config(), KEY).unwrap(),
            capacity.metadata_words()
        );
    }

    #[test]
    fn joint_metadata_fits_the_retained_tail_and_preserves_order() {
        let ids = [3, 4, 5, 6, 7];
        let plan = plan_requests(&batch(), &ids, &requests()).unwrap();
        let capacity = capacity(KEY, &config()).unwrap();
        assert!(plan.metadata_len() <= capacity.metadata_words());
        assert_eq!(plan.counts, vec![2, 1]);
        assert_eq!(plan.options, 3);
        assert_eq!(
            &plan.metadata[plan.layout.option_token_ids..plan.layout.total],
            &[4, 5, 6]
        );
        assert_eq!(output_count(&requests()).unwrap(), 5);
    }

    #[test]
    fn joint_rejects_wrong_kind_and_reversed_spans_before_device_use() {
        let bad_kind = [
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 3,
                    question: (0, 1),
                    options: vec![(1, 2)],
                }],
            },
            DecisionInput::Warmup,
        ];
        assert!(plan_requests(&batch(), &[3, 4, 5, 6, 7], &bad_kind).is_err());
        let reversed = [
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 0,
                    question: (2, 1),
                    options: vec![(1, 2)],
                }],
            },
            DecisionInput::Warmup,
        ];
        assert!(plan_requests(&batch(), &[3, 4, 5, 6, 7], &reversed).is_err());
        assert!(request_count(&DecisionInput::OptionTokens { token_ids: vec![1] }).is_err());
    }

    #[test]
    fn warm_plan_finishes_and_cancels_without_a_stream_owner() {
        let plan = plan_requests(&batch(), &[3, 4, 5, 6, 7], &requests()).unwrap();
        let mut prepared = ClefWarmPrepared::new(plan);
        let mut host = vec![0f32; prepared.output_elements().unwrap()];
        host[0] = 1.5;
        let outputs = prepared.finish(&mut host).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].logits, vec![1.5, 0.0]);
        assert_eq!(outputs[0].action_probability, 1.0);

        let plan = plan_requests(&batch(), &[3, 4, 5, 6, 7], &requests()).unwrap();
        let unsubmitted = ClefWarmPrepared::new(plan);
        drop(unsubmitted);

        let plan = plan_requests(&batch(), &[3, 4, 5, 6, 7], &requests()).unwrap();
        let mut failed = ClefWarmPrepared::new(plan);
        assert!(failed.finish(&mut [0.0f32; 1]).is_err());
        // A failed CPU result-size check neither poisons nor consumes metadata.
        let mut host = vec![0.0; failed.output_elements().unwrap()];
        assert_eq!(failed.finish(&mut host).unwrap().len(), 2);
        drop(failed);
    }
}
