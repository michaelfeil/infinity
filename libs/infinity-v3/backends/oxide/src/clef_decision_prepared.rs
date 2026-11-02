// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Deferred owner for one Clef request plan and its single host metadata tail.
//!
//! `prepare` is a cold Work-miss host operation: it validates spans and builds
//! the packed u32 tail without touching a GPU. The owner must stay alive until
//! the engine's one completion fence proves the asynchronous H2D metadata copy
//! and all dependent work finished. If completion is uncertain, Drop retains
//! the plan rather than freeing host bytes a failed stream may still read.

use crate::{clef_decision_finish::finish_outputs, clef_decision_plan::ClefRequestPlan};
use cuda_core::Stream;
use infinity_v3_backend_core::{BackendError, DecisionOutput};
use std::{mem::ManuallyDrop, sync::Arc};

pub(crate) struct ClefPrepared {
    inner: ManuallyDrop<ClefRequestPlan>,
    drained: bool,
    stream: Arc<Stream>,
}

impl ClefPrepared {
    pub(crate) fn new(plan: ClefRequestPlan, stream: &Arc<Stream>) -> Self {
        Self {
            inner: ManuallyDrop::new(plan),
            drained: false,
            stream: stream.clone(),
        }
    }

    pub(crate) fn plan(&self) -> &ClefRequestPlan {
        &self.inner
    }

    /// The exact u32 tail the Q35 Work owner appends to its single input tail.
    pub(crate) fn metadata(&self) -> &[u32] {
        &self.inner.metadata
    }

    pub(crate) fn metadata_words(&self) -> usize {
        self.inner.metadata.len()
    }

    pub(crate) fn output_elements(&self) -> Result<usize, BackendError> {
        self.inner.output_elements()
    }

    /// Called only after the outer Work/Engine proves the ordered stream and
    /// every dependent event/lease complete.
    pub(crate) fn mark_drained(&mut self) {
        self.drained = true;
    }

    /// Split completed logits into the caller's leased host slice, write Native
    /// action probability `1.0`, and release the host metadata owner.
    pub(crate) fn finish(&mut self, host: &mut [f32]) -> Result<Vec<DecisionOutput>, BackendError> {
        let outputs = finish_outputs(&self.inner, host)?;
        self.drained = true;
        Ok(outputs)
    }
}

impl Drop for ClefPrepared {
    fn drop(&mut self) {
        if !self.drained && self.stream.synchronize().is_err() {
            // Completion is uncertain: retain the host metadata tail.
            let plan = unsafe { ManuallyDrop::take(&mut self.inner) };
            std::mem::forget(plan);
            return;
        }
        // SAFETY: the stream drained just now, or the outer fence already did.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}
