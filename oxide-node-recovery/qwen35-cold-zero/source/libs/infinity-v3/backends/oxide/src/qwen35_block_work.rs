// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded B1 block metadata and prefix-state owner for dense Qwen3.5.
//!
//! The ordinary GPU input buffer stays in [`crate::qwen35_work::Qwen35Work`];
//! this leaf adds exactly one external F32 prefix state in block mode. Per
//! request it rebuilds the checked CPU [`Qwen35BlockLayout`], packs only the
//! metadata tail into the pinned input slot after ordinary staging, and
//! exposes the resulting word ranges to a future runtime. No per-layer state
//! or pinned output is allocated here.

use crate::qwen35_block_layout::{Qwen35BlockLayout, Qwen35BlockMode};
use crate::qwen35_config::Qwen35Config;
use crate::qwen35_runtime::Qwen35Model;
use crate::qwen35_work::{allocate_zero_ordered, ColdFenceOutcome, ColdPending};
use crate::{infer, product, staging, HostIoLease, WorkKey};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::{cell::Cell, ops::Range, sync::Arc};

const HEAD_DIM: usize = 128;
use crate::qwen35_block_layout::BLOCK_TAIL_WORDS as TAIL_WORDS;

fn add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| infer(format!("Qwen3.5 block {name} overflow")))
}

fn channels(config: &Qwen35Config) -> Result<usize, BackendError> {
    let units = add(
        product(config.linear_num_key_heads, 2)?,
        config.linear_num_value_heads,
        "projection units",
    )?;
    product(units, HEAD_DIM)
}

pub(crate) fn prefix_state_elements(value_heads: usize) -> Result<usize, BackendError> {
    product(product(value_heads, HEAD_DIM)?, HEAD_DIM)
}

pub(crate) struct BlockWork {
    prefix_state: DeviceBuffer<f32>,
    prefix_state_elements: usize,
    mode: Qwen35BlockMode,
    value_heads: usize,
    channels: usize,
    capacity_tokens: usize,
    metadata_base: usize,
    metadata_capacity: usize,
    layout: Cell<Option<Qwen35BlockLayout>>,
}

/// Engine-only ordered construction state. The prefix buffer is retained here
/// before its fallible zero enqueue; the enclosing Q35 Work takes the single
/// publication fence.
pub(crate) struct BlockWorkBuilding {
    prefix_state: Option<DeviceBuffer<f32>>,
    prefix_state_elements: usize,
    mode: Qwen35BlockMode,
    value_heads: usize,
    channels: usize,
    capacity_tokens: usize,
    metadata_base: usize,
    metadata_capacity: usize,
}

impl BlockWorkBuilding {
    pub(crate) fn new(model: &Qwen35Model, key: WorkKey) -> Result<Option<Self>, BackendError> {
        let Some(mode) = model.block_mode() else {
            return Ok(None);
        };
        let config = &model.weights.config;
        let value_heads = config.linear_num_value_heads;
        let channels = channels(config)?;
        let elements = prefix_state_elements(value_heads)?;
        let metadata_base = staging::input_capacity(key)?;
        let metadata_end = metadata_base
            .checked_add(TAIL_WORDS)
            .ok_or_else(|| infer("Qwen3.5 block metadata capacity overflow"))?;
        if metadata_end > model.input_capacity(key)? {
            return Err(infer(
                "Qwen3.5 block metadata exceeds the actual Work input capacity",
            ));
        }
        Ok(Some(Self {
            prefix_state: None,
            prefix_state_elements: elements,
            mode,
            value_heads,
            channels,
            capacity_tokens: key.0,
            metadata_base,
            metadata_capacity: TAIL_WORDS,
        }))
    }

    pub(crate) fn finish(self) -> Result<BlockWork, BackendError> {
        Ok(BlockWork {
            prefix_state: self
                .prefix_state
                .ok_or_else(|| infer("Qwen3.5 block cold construction lost its prefix state"))?,
            prefix_state_elements: self.prefix_state_elements,
            mode: self.mode,
            value_heads: self.value_heads,
            channels: self.channels,
            capacity_tokens: self.capacity_tokens,
            metadata_base: self.metadata_base,
            metadata_capacity: self.metadata_capacity,
            layout: Cell::new(None),
        })
    }
}

fn settle_block_cold_error(
    building: BlockWorkBuilding,
    stream: &Arc<Stream>,
    original: BackendError,
) -> BackendError {
    match ColdPending::new(building, None).settle(|| stream.synchronize().map_err(infer)) {
        Ok(ColdFenceOutcome::Fenced(building)) => {
            drop(building);
            original
        }
        Ok(ColdFenceOutcome::Quarantined { .. }) => infer(format!(
            "{original}; Qwen3.5 block cold initialization fence failed"
        )),
        Err(invariant) => infer(format!("{original}; {invariant}")),
    }
}

impl BlockWork {
    pub(crate) fn allocation_bytes(
        _key: WorkKey,
        model: &Qwen35Model,
    ) -> Result<usize, BackendError> {
        if model.block_mode().is_none() {
            return Ok(0);
        }
        product(
            prefix_state_elements(model.weights.config.linear_num_value_heads)?,
            4,
        )
    }

    /// Engine-only ordered path: allocate and enqueue the zero without a fence.
    pub(crate) fn build_into(
        building: &mut BlockWorkBuilding,
        stream: &Arc<Stream>,
    ) -> Result<(), BackendError> {
        allocate_zero_ordered(
            stream,
            building.prefix_state_elements,
            &mut building.prefix_state,
        )
    }

    /// Standalone path: one successful fence before the owner is returned.
    #[allow(dead_code)] // Kept for the standalone leaf contract.
    pub(crate) fn new(
        stream: &Arc<Stream>,
        model: &Qwen35Model,
        key: WorkKey,
    ) -> Result<Option<Self>, BackendError> {
        let Some(mut building) = BlockWorkBuilding::new(model, key)? else {
            return Ok(None);
        };
        if let Err(error) = Self::build_into(&mut building, stream) {
            return Err(settle_block_cold_error(building, stream, error));
        }
        match stream.synchronize() {
            Ok(()) => Ok(Some(building.finish()?)),
            Err(fence_error) => {
                let _ = ColdPending::new(building, None).quarantine();
                Err(infer(format!(
                    "Qwen3.5 block cold publication fence failed: {fence_error}"
                )))
            }
        }
    }

    pub(crate) fn prefix_state(&self) -> &DeviceBuffer<f32> {
        &self.prefix_state
    }
    pub(crate) const fn prefix_state_elements(&self) -> usize {
        self.prefix_state_elements
    }
    pub(crate) const fn metadata_base(&self) -> usize {
        self.metadata_base
    }
    pub(crate) const fn metadata_capacity(&self) -> usize {
        self.metadata_capacity
    }
    pub(crate) const fn mode(&self) -> Qwen35BlockMode {
        self.mode
    }
    pub(crate) fn layout(&self) -> Result<Qwen35BlockLayout, BackendError> {
        self.layout
            .get()
            .ok_or_else(|| infer("Qwen3.5 block layout has not been staged"))
    }
    pub(crate) fn packed_range(&self) -> Result<Range<usize>, BackendError> {
        self.layout()?.packed_range(self.metadata_base)
    }

    /// Resolve the actual B1 prefix and validate it against the actual row count
    /// before any host write, then pack exactly the metadata tail into the
    /// already-staged lease. The prefix state allocation is reused unchanged,
    /// so changing the prefix performs no GPU allocation, copy or model reload.
    pub(crate) fn pack_metadata(
        &self,
        lease: &mut HostIoLease,
        tokens: usize,
        request_prefix: Option<usize>,
    ) -> Result<usize, BackendError> {
        if tokens > self.capacity_tokens {
            return Err(infer("Qwen3.5 block tokens exceed retained capacity"));
        }
        let prefix = self.mode.resolve_prefix(request_prefix, tokens)?;
        let layout = Qwen35BlockLayout::new(tokens, prefix, self.value_heads, self.channels)?;
        let host = lease.slot_mut()?.input.as_mut_slice();
        let end = layout.pack_into(host, self.metadata_base)?.end;
        self.layout.set(Some(layout));
        Ok(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANARY: u32 = 0xdead_beef;

    #[test]
    fn metadata_tail_is_exactly_nine_words_after_ordinary_capacity() {
        assert_eq!(TAIL_WORDS, 9);
        for key in [(1usize, 1usize, 1usize), (32, 4, 16), (2048, 9, 9)] {
            assert_eq!(
                staging::input_capacity_with_tail(key, TAIL_WORDS).unwrap(),
                staging::input_capacity(key).unwrap() + TAIL_WORDS
            );
        }
        assert!(staging::input_capacity_with_tail((1, 1, 1), usize::MAX).is_err());
    }

    #[test]
    fn prefix_state_is_two_or_three_mebibytes_for_canonical_heads() {
        assert_eq!(prefix_state_elements(32).unwrap() * 4, 2 * 1024 * 1024);
        assert_eq!(prefix_state_elements(48).unwrap() * 4, 3 * 1024 * 1024);
        assert!(prefix_state_elements(usize::MAX).is_err());
    }

    #[test]
    fn checked_layout_rejects_a_bad_prefix_before_any_pack_write() {
        let layout = Qwen35BlockLayout::new(7, 3, 32, 64 * HEAD_DIM).unwrap();
        assert!(Qwen35BlockLayout::new(7, 8, 32, 64 * HEAD_DIM).is_err());

        let mut words = [CANARY; 9];
        assert_eq!(layout.pack_into(&mut words, 0).unwrap(), 0..8);
        assert_eq!(words[8], CANARY);

        let mut short = [CANARY; 7];
        assert!(layout.pack_into(&mut short, 0).is_err());
        assert_eq!(short, [CANARY; 7]);
    }
}
