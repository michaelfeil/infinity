// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Work-miss preparation for the Clef joint head.

use crate::{
    clef_decision_head::ClefHead,
    clef_decision_plan::{plan_requests, ClefRequestPlan},
    clef_decision_prepared::ClefPrepared,
    decision_head::PackedDecisionBatch,
    start,
};
use cuda_core::Stream;
use infinity_v3_backend_core::{BackendError, DecisionInput};
use std::sync::Arc;

impl ClefHead {
    /// Validate one request plan at a cold Work miss. This performs no device
    /// allocation or synchronization; `ClefPrepared` owns the host metadata
    /// tail until the outer completion fence.
    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        ids: &[u32],
        requests: &[DecisionInput],
        stream: &Arc<Stream>,
    ) -> Result<ClefPrepared, BackendError> {
        Ok(ClefPrepared::new(
            self.prepare_plan(batch, ids, requests, stream)?,
            stream,
        ))
    }

    /// Plan metadata for a caller that uploads its own pinned input tail.
    /// The returned CPU owner has no stream or completion responsibility.
    pub(crate) fn prepare_plan(
        &self,
        batch: &PackedDecisionBatch<'_>,
        ids: &[u32],
        requests: &[DecisionInput],
        stream: &Arc<Stream>,
    ) -> Result<ClefRequestPlan, BackendError> {
        if !Arc::ptr_eq(stream.context(), &self.context)
            || !std::ptr::eq(stream.as_ref(), self.stream.as_ref())
        {
            return Err(start("Clef preparation requires its cold stream"));
        }
        plan_requests(batch, ids, requests)
    }
}
