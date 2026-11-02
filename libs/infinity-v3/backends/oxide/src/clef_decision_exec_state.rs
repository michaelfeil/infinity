// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Borrowed execution context for the Clef prepared GPU phases.

use crate::{
    clef_decision_head::ClefHead, clef_decision_plan::ClefRequestPlan,
    clef_decision_workspace::ClefWorkspace, cublas::Blas,
};
use cuda_core::Stream;
use infinity_v3_backend_core::BackendError;

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    what: &'static str,
) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| crate::clef_decision::infer(format!("Clef {what} extent overflow")))
}

pub(crate) fn metadata_at(base: u64, words: usize) -> Result<u64, BackendError> {
    let bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| crate::clef_decision::infer("Clef metadata byte offset overflow"))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| crate::clef_decision::infer("Clef metadata offset exceeds u64"))?;
    base.checked_add(bytes)
        .ok_or_else(|| crate::clef_decision::infer("Clef metadata address overflow"))
}

pub(crate) struct ClefRun<'a> {
    pub(crate) head: &'a ClefHead,
    pub(crate) metadata_base: u64,
    pub(crate) workspace: &'a mut ClefWorkspace,
    pub(crate) plan: &'a ClefRequestPlan,
    pub(crate) blas: &'a Blas,
    pub(crate) stream: &'a Stream,
    pub(crate) h: usize,
    pub(crate) width: usize,
    pub(crate) ff: usize,
    pub(crate) tokens: usize,
    pub(crate) requests: usize,
    pub(crate) fields: usize,
    pub(crate) options: usize,
    pub(crate) option_tokens: usize,
}
