// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Checked word geometry for the packed input ABI.
//!
//! The authoritative order is three token-wide `u32` arrays (ids, positions,
//! types), the inclusive cumulative-offset count (`B + 1`), the selected row
//! metadata, and an optional physical trailer. The caller decides whether the
//! trailer stays host-side or is uploaded and read on device; this type makes
//! no residency or ownership claim. It performs no CUDA work or allocation.

use super::infer;
use infinity_v3_backend_core::BackendError;
use std::ops::Range;

const WORD_BYTES: usize = std::mem::size_of::<u32>();

/// A word offset inside one packed input buffer. Bytes are opt-in and checked
/// so a word count cannot silently become a byte address.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct InputWordOffset(usize);

impl InputWordOffset {
    #[inline]
    pub(crate) const fn get(self) -> usize {
        self.0
    }

    #[inline]
    pub(crate) fn bytes(self) -> Result<u64, BackendError> {
        self.0
            .checked_mul(WORD_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| infer("Input staging byte offset overflow"))
    }

    #[inline]
    pub(crate) fn address(self, base: u64) -> Result<u64, BackendError> {
        base.checked_add(self.bytes()?)
            .ok_or_else(|| infer("Input staging device address overflow"))
    }

    #[inline]
    fn checked_add(self, words: usize) -> Result<Self, BackendError> {
        self.0
            .checked_add(words)
            .map(Self)
            .ok_or_else(|| infer("Input staging layout overflow"))
    }
}

/// Checked packed-input geometry for one actual request or one retained key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InputLayout {
    tokens: usize,
    cu_count: usize,
    selections: usize,
    trailer: usize,
    ids: InputWordOffset,
    positions: InputWordOffset,
    types: InputWordOffset,
    boundaries: InputWordOffset,
    selection_start: InputWordOffset,
    trailer_start: InputWordOffset,
    end: InputWordOffset,
}

impl InputLayout {
    /// `cu_count` is the inclusive cumulative-offset count (`B + 1`), matching
    /// the packed `cumulative_seq_lengths` array.
    pub(crate) fn new(
        tokens: usize,
        cu_count: usize,
        selections: usize,
    ) -> Result<Self, BackendError> {
        Self::build(tokens, cu_count, selections, 0)
    }

    /// Opt in to physical words after the packed selection metadata. The caller
    /// may upload and read them on device or keep them host-side. Ordinary
    /// callers keep the unchanged packed capacity by leaving this at zero.
    pub(crate) fn with_trailer(self, trailer: usize) -> Result<Self, BackendError> {
        Self::build(self.tokens, self.cu_count, self.selections, trailer)
    }

    fn build(
        tokens: usize,
        cu_count: usize,
        selections: usize,
        trailer: usize,
    ) -> Result<Self, BackendError> {
        let ids = InputWordOffset(0);
        let positions = ids.checked_add(tokens)?;
        let types = positions.checked_add(tokens)?;
        let boundaries = types.checked_add(tokens)?;
        let selection_start = boundaries.checked_add(cu_count)?;
        let trailer_start = selection_start.checked_add(selections)?;
        let end = trailer_start.checked_add(trailer)?;
        Ok(Self {
            tokens,
            cu_count,
            selections,
            trailer,
            ids,
            positions,
            types,
            boundaries,
            selection_start,
            trailer_start,
            end,
        })
    }

    #[inline]
    pub(crate) const fn tokens(self) -> usize {
        self.tokens
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn cu_count(self) -> usize {
        self.cu_count
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn selection_count(self) -> usize {
        self.selections
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn trailer_words(self) -> usize {
        self.trailer
    }

    #[inline]
    pub(crate) const fn ids(self) -> InputWordOffset {
        self.ids
    }

    #[inline]
    pub(crate) const fn positions(self) -> InputWordOffset {
        self.positions
    }

    #[inline]
    pub(crate) const fn types(self) -> InputWordOffset {
        self.types
    }

    #[inline]
    pub(crate) const fn boundaries(self) -> InputWordOffset {
        self.boundaries
    }

    #[inline]
    pub(crate) const fn selections(self) -> InputWordOffset {
        self.selection_start
    }

    #[inline]
    pub(crate) const fn trailer(self) -> InputWordOffset {
        self.trailer_start
    }

    /// Words occupied by the packed prefix, excluding an optional physical
    /// trailer.
    #[inline]
    pub(crate) const fn packed_words(self) -> usize {
        self.trailer_start.get()
    }

    /// Total words occupied by the packed prefix plus the optional physical
    /// trailer.
    #[inline]
    pub(crate) const fn words(self) -> usize {
        self.end.get()
    }

    #[inline]
    pub(crate) fn ids_range(self) -> Range<usize> {
        self.ids.get()..self.positions.get()
    }

    #[inline]
    pub(crate) fn positions_range(self) -> Range<usize> {
        self.positions.get()..self.types.get()
    }

    #[inline]
    pub(crate) fn types_range(self) -> Range<usize> {
        self.types.get()..self.boundaries.get()
    }

    #[inline]
    pub(crate) fn boundaries_range(self) -> Range<usize> {
        self.boundaries.get()..self.selection_start.get()
    }

    #[inline]
    pub(crate) fn selections_range(self) -> Range<usize> {
        self.selection_start.get()..self.trailer_start.get()
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn trailer_range(self) -> Range<usize> {
        self.trailer_start.get()..self.end.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_input;

    fn boundaries(lengths: &[usize]) -> Vec<u32> {
        let mut out = Vec::with_capacity(lengths.len() + 1);
        out.push(0);
        for &length in lengths {
            let next = out.last().copied().unwrap() + u32::try_from(length).unwrap();
            out.push(next);
        }
        out
    }

    fn values(tokens: usize, base: u32, modulus: u32) -> Vec<u32> {
        (0..tokens)
            .map(|index| base + (index as u32 % modulus))
            .collect()
    }

    #[test]
    fn packed_sections_use_actual_word_offsets_for_ragged_batches() {
        for (tokens, lengths) in [
            (1, &[1][..]),
            (7, &[2, 1, 3, 1][..]),
            (129, &[17, 1, 64, 47][..]),
        ] {
            let cu = boundaries(lengths);
            let selections = if lengths.len() == 1 {
                vec![0, 0]
            } else {
                vec![3, 0, 3, 1]
            };
            let ids = values(tokens, 1_000, 251);
            let positions = values(tokens, 0, 17);
            let types = values(tokens, 0, 2);
            let layout = InputLayout::new(tokens, cu.len(), selections.len()).unwrap();
            let mut out = vec![u32::MAX; layout.words()];

            pack_input(&mut out, tokens, &ids, &positions, &types, &cu, &selections).unwrap();

            assert_eq!(out.get(layout.ids_range()).unwrap(), ids.as_slice());
            assert_eq!(
                out.get(layout.positions_range()).unwrap(),
                positions.as_slice()
            );
            assert_eq!(out.get(layout.types_range()).unwrap(), types.as_slice());
            assert_eq!(out.get(layout.boundaries_range()).unwrap(), cu.as_slice());
            assert_eq!(
                out.get(layout.selections_range()).unwrap(),
                selections.as_slice(),
                "packing preserves reordered and duplicate selection rows"
            );
            assert!(out.get(layout.trailer_range()).unwrap().is_empty());
            assert_eq!(layout.words(), tokens * 3 + cu.len() + selections.len());

            assert_eq!(layout.ids().get(), layout.ids_range().start);
            assert_eq!(layout.positions().get(), layout.positions_range().start);
            assert_eq!(layout.types().get(), layout.types_range().start);
            assert_eq!(layout.boundaries().get(), layout.boundaries_range().start);
            assert_eq!(
                layout.selections().get(),
                layout.boundaries().get() + cu.len()
            );
            assert_eq!(layout.selections().get(), layout.selections_range().start);
            assert_eq!(layout.trailer().get(), layout.trailer_range().start);
            assert_eq!(layout.positions().get(), tokens);
            assert_eq!(layout.positions().bytes().unwrap(), (tokens * 4) as u64);
            assert_eq!(layout.types().get(), tokens * 2);
            assert_eq!(layout.types().bytes().unwrap(), (tokens * 8) as u64);
            assert_eq!(layout.boundaries().get(), tokens * 3);
            assert_eq!(layout.boundaries().bytes().unwrap(), (tokens * 12) as u64);
        }
    }

    #[test]
    fn zero_selection_layout_has_no_selection_words() {
        let cu = boundaries(&[2, 1, 3, 1]);
        let layout = InputLayout::new(7, cu.len(), 0).unwrap();
        let ids = values(7, 100, 11);
        let positions = values(7, 0, 7);
        let types = values(7, 0, 2);
        let mut out = vec![u32::MAX; layout.words()];

        pack_input(&mut out, 7, &ids, &positions, &types, &cu, &[]).unwrap();

        assert!(layout.selections_range().is_empty());
        assert_eq!(layout.selections(), layout.trailer());
        assert_eq!(out.get(layout.boundaries_range()).unwrap(), cu.as_slice());
    }

    #[test]
    fn rejects_caller_lengths_overflow_and_confused_units() {
        let cu = boundaries(&[2, 1, 3, 1]);
        let selections = [2, 0, 2];
        let layout = InputLayout::new(7, cu.len(), selections.len()).unwrap();
        let ids = values(7, 100, 11);
        let positions = values(7, 0, 7);
        let types = values(7, 0, 2);

        let mut short = vec![0; layout.words() - 1];
        assert!(pack_input(&mut short, 7, &ids, &positions, &types, &cu, &selections).is_err());

        let mut long = vec![0; layout.words() + 1];
        assert!(pack_input(&mut long, 7, &ids, &positions, &types, &cu, &selections).is_err());

        let too_many = values(8, 0, 3);
        let mut out = vec![0; layout.words()];
        assert!(pack_input(&mut out, 7, &too_many, &positions, &types, &cu, &selections).is_err());

        assert!(InputLayout::new(usize::MAX, 1, 0).is_err());
        assert!(layout.with_trailer(usize::MAX).is_err());

        assert_eq!(layout.positions().get(), layout.tokens());
        assert_eq!(
            layout.positions().bytes().unwrap(),
            u64::try_from(layout.tokens() * 4).unwrap()
        );
        assert_eq!(
            layout.boundaries().bytes().unwrap(),
            u64::try_from(layout.tokens() * 12).unwrap()
        );
    }

    #[test]
    fn fake_gpu_prefix_and_trailer_do_not_overlap_packed_bytes() {
        let tokens = 129;
        let cu = boundaries(&[17, 1, 64, 47]);
        let selections = [3, 0, 3, 1];
        let raw_rows = 4;
        let gpu = InputLayout::new(tokens, cu.len(), 0).unwrap();
        let host = InputLayout::new(tokens, cu.len(), selections.len())
            .unwrap()
            .with_trailer(raw_rows)
            .unwrap();
        let base = 0x0000_0001_2345_0000u64;
        let ids = values(tokens, 1_000, 251);
        let positions = values(tokens, 0, 17);
        let types = values(tokens, 0, 2);

        assert_eq!(gpu.ids().address(base).unwrap(), base);
        assert_eq!(
            gpu.positions().address(base).unwrap(),
            host.positions().address(base).unwrap()
        );
        assert_eq!(
            gpu.boundaries().address(base).unwrap(),
            host.boundaries().address(base).unwrap()
        );

        let mut packed = vec![0; gpu.words()];
        pack_input(&mut packed, tokens, &ids, &positions, &types, &cu, &[]).unwrap();
        let cu_start = gpu.boundaries().address(base).unwrap();
        let cu_end = cu_start + u64::try_from(gpu.cu_count() * WORD_BYTES).unwrap();
        assert_eq!(
            cu_end,
            base + u64::try_from(packed.len() * WORD_BYTES).unwrap()
        );
        assert_eq!(
            cu_end - base,
            gpu.boundaries().bytes().unwrap() + gpu.cu_count() as u64 * 4
        );

        assert_eq!(host.selection_count(), selections.len());
        assert_eq!(host.trailer_words(), raw_rows);
        assert_eq!(host.packed_words(), gpu.words() + selections.len());
        assert_eq!(host.words() - host.packed_words(), raw_rows);
        assert_eq!(host.selections().get(), host.selections_range().start);
        assert_eq!(host.trailer().get(), host.trailer_range().start);
        assert_eq!(
            host.trailer().address(base).unwrap(),
            base + u64::try_from(host.packed_words() * WORD_BYTES).unwrap()
        );

        let mut storage = vec![u32::MAX; host.words()];
        pack_input(
            &mut storage[..host.packed_words()],
            tokens,
            &ids,
            &positions,
            &types,
            &cu,
            &selections,
        )
        .unwrap();
        assert!(storage
            .get(host.trailer_range())
            .unwrap()
            .iter()
            .all(|&value| value == u32::MAX));
        assert_eq!(
            storage.get(host.selections_range()).unwrap(),
            selections.as_slice()
        );
    }
}
