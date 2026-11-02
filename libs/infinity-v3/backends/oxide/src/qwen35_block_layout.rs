// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Checked host metadata for the first OneJev Qwen3.5 block execution plan.
//! This leaf is B1-only: one actual bidirectional prefix `[0, P)` and an
//! optional suffix `[P, T)`. Every layer keeps one combined `T`-row projection
//! GEMM, never split into prefix/suffix halves. Multiple or shared-prefix
//! requests are unsupported and must be rejected before a layout is built.
//! The prefix uses the Native fresh-prefix AOT path; a present suffix follows
//! existing GDN cached-state policy. Final per-layer state is scratch for the
//! immediate suffix, not a future/cross-request prefix cache. `pack_into` writes
//! only the caller-owned tail: no allocation, device copy or staging policy change.

use infinity_v3_backend_core::BackendError;
use std::ops::Range;
pub(crate) const MAX_BLOCK_TOKENS: usize = 262_144;
const HEAD_DIM: usize = 128;
const KEY_PROJECTION_UNITS: usize = 32;
const MAX_HISTORY_ROWS: usize = 3;
const PREFIX_CU_WORDS: usize = 2;
const SUFFIX_CU_WORDS: usize = 2;
const HISTORY_CU_WORDS: usize = 2;
const SOURCE_WORDS: usize = 2;
const PREFIX_CU_START: usize = 0;
const SUFFIX_CU_START: usize = PREFIX_CU_START + PREFIX_CU_WORDS;
const HISTORY_CU_START: usize = SUFFIX_CU_START + SUFFIX_CU_WORDS;
const SOURCE_START: usize = HISTORY_CU_START + HISTORY_CU_WORDS;
/// Worst-case packed block tail: two CU pairs, history CU, aligned i64 source.
pub(crate) const BLOCK_TAIL_WORDS: usize = SOURCE_START + SOURCE_WORDS + 1;

pub(crate) fn validate_block_token_count(tokens: usize) -> Result<(), BackendError> {
    if tokens == 0 || tokens > MAX_BLOCK_TOKENS {
        return Err(infer(
            "block token count is outside the qualified 1..=262144 range",
        ));
    }
    Ok(())
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3.5 block layout: {error}"))
}

fn product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| infer(format!("{name} overflow")))
}

fn add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| infer(format!("{name} overflow")))
}

macro_rules! scalar_getters {
    ($($name:ident),* $(,)?) => {
        $(pub(crate) const fn $name(&self) -> usize { self.$name })*
    };
}

/// Backend-wide block prefix policy. `Fixed` keeps the original per-backend
/// prefix; `Dynamic` resolves the prefix from each staged request while the
/// actual token count and allocation capacity stay unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35BlockMode {
    Fixed(usize),
    Dynamic,
}

impl Qwen35BlockMode {
    pub(crate) const fn fixed_prefix(self) -> Option<usize> {
        match self {
            Self::Fixed(prefix) => Some(prefix),
            Self::Dynamic => None,
        }
    }

    /// Resolve the actual prefix for one request before any host/device write.
    /// A fixed mode accepts an absent request prefix or the exact fixed value.
    /// A dynamic mode requires an explicit prefix. Both reject zero and any
    /// prefix beyond the actual token count.
    pub(crate) fn resolve_prefix(
        self,
        requested: Option<usize>,
        tokens: usize,
    ) -> Result<usize, BackendError> {
        validate_block_token_count(tokens)?;
        let prefix = match self {
            Self::Fixed(fixed) => match requested {
                None => fixed,
                Some(requested) if requested == fixed => fixed,
                Some(requested) => {
                    return Err(infer(format!(
                        "fixed block prefix {fixed} does not match requested prefix {requested}"
                    )))
                }
            },
            Self::Dynamic => requested
                .ok_or_else(|| infer("dynamic block requires an explicit request prefix"))?,
        };
        if prefix == 0 || prefix > tokens {
            return Err(infer(format!(
                "prefix {prefix} must be nonzero and at most {tokens}"
            )));
        }
        Ok(prefix)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35BlockLayout {
    tokens: usize,
    prefix_tokens: usize,
    suffix_tokens: usize,
    value_heads: usize,
    channels: usize,
    projection_rows: usize,
    projection_elements: usize,
    history_rows: usize,
    history_start_elements: usize,
    history_elements: usize,
    history_end_elements: usize,
    prefix_state_elements: usize,
    prefix_cu: [u32; 2],
    suffix_cu: [u32; 2],
    history_cu: [u32; 2],
    history_cu_i32: [i32; 2],
    source_map_i64: [i64; 1],
    suffix_source: i64,
}

impl Qwen35BlockLayout {
    /// Validate one B1 prefix/suffix split and derive all packed metadata.
    pub(crate) fn new(
        tokens: usize,
        prefix: usize,
        value_heads: usize,
        channels: usize,
    ) -> Result<Self, BackendError> {
        if tokens == 0 || tokens > MAX_BLOCK_TOKENS {
            return Err(infer(format!(
                "token count {tokens} is outside 1..={MAX_BLOCK_TOKENS}"
            )));
        }
        if prefix == 0 || prefix > tokens {
            return Err(infer(format!(
                "prefix {prefix} must be nonzero and at most {tokens}"
            )));
        }
        if !matches!(value_heads, 32 | 48) {
            return Err(infer(format!(
                "value head count {value_heads} is outside 32/48 scope"
            )));
        }
        let units = add(KEY_PROJECTION_UNITS, value_heads, "QKV projection units")?;
        let expected_channels = product(units, HEAD_DIM, "QKV projection width")?;
        if channels != expected_channels {
            return Err(infer(format!(
                "channels {channels} != canonical width {expected_channels}"
            )));
        }
        let suffix = tokens - prefix;
        let prefix_cu = [
            0,
            u32::try_from(prefix).map_err(|_| infer("prefix exceeds u32"))?,
        ];
        let suffix_cu = [
            0,
            u32::try_from(suffix).map_err(|_| infer("suffix exceeds u32"))?,
        ];
        let history_rows = prefix.min(MAX_HISTORY_ROWS);
        let history_cu = [
            0,
            u32::try_from(history_rows).map_err(|_| infer("history rows exceed u32"))?,
        ];
        let history_start_elements = product(prefix - history_rows, channels, "history start")?;
        let history_elements = product(history_rows, channels, "history elements")?;
        let history_end_elements = add(history_start_elements, history_elements, "history end")?;
        let prefix_state_elements = product(
            product(value_heads, HEAD_DIM, "prefix state rows")?,
            HEAD_DIM,
            "prefix state elements",
        )?;
        Ok(Self {
            tokens,
            prefix_tokens: prefix,
            suffix_tokens: suffix,
            value_heads,
            channels,
            projection_rows: tokens,
            projection_elements: product(tokens, channels, "combined projection elements")?,
            history_rows,
            history_start_elements,
            history_elements,
            history_end_elements,
            prefix_state_elements,
            prefix_cu,
            suffix_cu,
            history_cu,
            history_cu_i32: [
                0,
                i32::try_from(history_rows).map_err(|_| infer("history rows exceed i32"))?,
            ],
            source_map_i64: [0],
            suffix_source: 0,
        })
    }

    scalar_getters!(
        tokens,
        prefix_tokens,
        suffix_tokens,
        value_heads,
        channels,
        projection_rows,
        projection_elements,
        history_rows,
        history_start_elements,
        history_elements,
        prefix_state_elements
    );

    /// Prefix-only B1 remains shape-valid, but a suffix launch is forbidden.
    pub(crate) const fn has_suffix(&self) -> bool {
        self.suffix_tokens != 0
    }
    pub(crate) const fn suffix_launch_required(&self) -> bool {
        self.has_suffix()
    }
    pub(crate) const fn history_start(&self) -> usize {
        self.history_start_elements
    }
    pub(crate) const fn prefix_cu(&self) -> [u32; 2] {
        self.prefix_cu
    }
    pub(crate) const fn suffix_cu(&self) -> [u32; 2] {
        self.suffix_cu
    }
    pub(crate) const fn history_cu(&self) -> [u32; 2] {
        self.history_cu
    }
    pub(crate) const fn suffix_source(&self) -> i64 {
        self.suffix_source
    }
    pub(crate) fn history_range(&self) -> Range<usize> {
        self.history_start_elements..self.history_end_elements
    }

    /// Borrowed CPU metadata for the B1 GDN block frames. These references stay
    /// tied to the staged layout; callers must keep it alive through every CPU
    /// enqueue that consumed the frames.
    pub(crate) fn prefix_cumulative(&self) -> &[u32] {
        &self.prefix_cu
    }
    pub(crate) fn suffix_cumulative(&self) -> &[u32] {
        &self.suffix_cu
    }
    pub(crate) fn history_cumulative(&self) -> &[i32] {
        &self.history_cu_i32
    }
    pub(crate) fn sources(&self) -> &[i64] {
        &self.source_map_i64
    }

    pub(crate) fn value_width(&self) -> Result<usize, BackendError> {
        product(self.value_heads, HEAD_DIM, "value width")
    }
    pub(crate) fn ab_width(&self) -> Result<usize, BackendError> {
        product(self.value_heads, 2, "AB width")
    }

    /// Element ranges of the prefix and suffix rows in a combined `T`-row view.
    pub(crate) fn split_rows(
        &self,
        width: usize,
        name: &str,
    ) -> Result<(Range<usize>, Range<usize>), BackendError> {
        let split = product(self.prefix_tokens, width, name)?;
        let end = product(self.tokens, width, name)?;
        Ok((0..split, split..end))
    }

    pub(crate) const fn packed_words(&self) -> usize {
        SOURCE_START + SOURCE_WORDS
    }
    pub(crate) fn packed_end(&self, base: usize) -> Result<usize, BackendError> {
        let source = add(base, SOURCE_START, "packed source alignment")?;
        let words = add(
            SOURCE_START + (source & 1),
            SOURCE_WORDS,
            "packed word count",
        )?;
        add(base, words, "packed metadata end")
    }

    pub(crate) fn packed_range(&self, base: usize) -> Result<Range<usize>, BackendError> {
        Ok(base..self.packed_end(base)?)
    }

    fn word_range(
        &self,
        base: usize,
        start: usize,
        words: usize,
        name: &str,
    ) -> Result<Range<usize>, BackendError> {
        let start = add(base, start, name)?;
        Ok(start..add(start, words, name)?)
    }

    pub(crate) fn prefix_cu_range(&self, base: usize) -> Result<Range<usize>, BackendError> {
        self.word_range(base, PREFIX_CU_START, PREFIX_CU_WORDS, "prefix CU range")
    }

    /// Suffix CU slots are reserved and zero for a prefix-only plan.
    pub(crate) fn suffix_cu_range(&self, base: usize) -> Result<Range<usize>, BackendError> {
        self.word_range(base, SUFFIX_CU_START, SUFFIX_CU_WORDS, "suffix CU range")
    }

    pub(crate) fn history_cu_range(&self, base: usize) -> Result<Range<usize>, BackendError> {
        self.word_range(base, HISTORY_CU_START, HISTORY_CU_WORDS, "history CU range")
    }

    /// Source slots hold one `i64` value as two `u32` words, 8-byte aligned.
    pub(crate) fn source_range(&self, base: usize) -> Result<Range<usize>, BackendError> {
        let start = add(base, SOURCE_START, "source alignment")?;
        let start = add(start, start & 1, "source alignment padding")?;
        Ok(start..add(start, SOURCE_WORDS, "source range")?)
    }

    fn write(
        words: &mut [u32],
        range: Range<usize>,
        values: &[u32],
        name: &str,
    ) -> Result<(), BackendError> {
        let slots = words
            .get_mut(range)
            .ok_or_else(|| infer(format!("{name} exceeds the destination")))?;
        for (slot, value) in slots.iter_mut().zip(values.iter().copied()) {
            *slot = value;
        }
        Ok(())
    }

    /// Write exactly the aligned metadata tail starting at `base`.
    ///
    /// The whole extent is checked before the first write, so an undersized
    /// destination or overflowing base leaves the buffer untouched.
    pub(crate) fn pack_into(
        &self,
        words: &mut [u32],
        base: usize,
    ) -> Result<Range<usize>, BackendError> {
        let end = self.packed_end(base)?;
        if end > words.len() {
            return Err(infer(format!(
                "packed metadata end {end} exceeds destination length {}",
                words.len()
            )));
        }
        let prefix = self.prefix_cu_range(base)?;
        let suffix = self.suffix_cu_range(base)?;
        let history = self.history_cu_range(base)?;
        let source = self.source_range(base)?;
        let low = u32::try_from(self.suffix_source).map_err(|_| infer("source low word"))?;
        let high =
            u32::try_from(self.suffix_source >> 32).map_err(|_| infer("source high word"))?;
        Self::write(words, prefix, &self.prefix_cu, "prefix CU")?;
        Self::write(words, suffix, &self.suffix_cu, "suffix CU")?;
        Self::write(words, history.clone(), &self.history_cu, "history CU")?;
        if source.start != history.end {
            let padding = words
                .get_mut(history.end)
                .ok_or_else(|| infer("source alignment padding exceeds the destination"))?;
            *padding = 0;
        }
        Self::write(words, source, &[low, high], "suffix source")?;
        Ok(base..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANARY: u32 = 0xdead_beef;

    fn channels(value_heads: usize) -> usize {
        (KEY_PROJECTION_UNITS + value_heads) * HEAD_DIM
    }

    #[test]
    fn actual_t7_p3_metadata_is_exact() {
        let layout = Qwen35BlockLayout::new(7, 3, 32, channels(32)).unwrap();
        assert_eq!(layout.tokens(), 7);
        assert_eq!(layout.prefix_tokens(), 3);
        assert_eq!(layout.suffix_tokens(), 4);
        assert!(layout.has_suffix());
        assert!(layout.suffix_launch_required());
        assert_eq!(layout.value_heads(), 32);
        assert_eq!(layout.channels(), 8192);
        assert_eq!(layout.projection_rows(), 7);
        assert_eq!(layout.projection_elements(), 7 * 8192);
        assert_eq!(layout.history_rows(), 3);
        assert_eq!(layout.history_start_elements(), 0);
        assert_eq!(layout.history_start(), 0);
        assert_eq!(layout.history_elements(), 3 * 8192);
        assert_eq!(layout.history_range(), 0..3 * 8192);
        assert_eq!(layout.prefix_state_elements(), 32 * 128 * 128);
        assert_eq!(layout.prefix_cu(), [0, 3]);
        assert_eq!(layout.suffix_cu(), [0, 4]);
        assert_eq!(layout.history_cu(), [0, 3]);
        assert_eq!(layout.suffix_source(), 0);

        let base = 2;
        let mut words = [CANARY; 12];
        let written = layout.pack_into(&mut words, base).unwrap();
        assert_eq!(&words[..base], &[CANARY; 2][..]);
        assert_eq!(&words[base..base + 8], &[0u32, 3, 0, 4, 0, 3, 0, 0][..]);
        assert_eq!(&words[base + 8..], &[CANARY; 2][..]);
        assert_eq!(written, base..base + 8);
    }

    #[test]
    fn actual_t2048_p1536_history_and_packed_words_are_exact() {
        let layout = Qwen35BlockLayout::new(2048, 1536, 48, channels(48)).unwrap();
        assert_eq!(layout.channels(), 10_240);
        assert_eq!(layout.suffix_tokens(), 512);
        assert_eq!(layout.history_rows(), 3);
        assert_eq!(layout.history_start_elements(), 1533 * 10_240);
        assert_eq!(layout.history_elements(), 3 * 10_240);
        assert_eq!(layout.prefix_state_elements(), 48 * 128 * 128);
        assert_eq!(layout.prefix_cu(), [0, 1536]);
        assert_eq!(layout.suffix_cu(), [0, 512]);
        assert_eq!(layout.history_cu(), [0, 3]);
        assert_eq!(layout.packed_end(0).unwrap(), 8);
        assert_eq!(layout.packed_words(), 8);
        assert_eq!(layout.packed_range(0).unwrap(), 0..8);

        let mut words = [CANARY; 10];
        let written = layout.pack_into(&mut words, 1).unwrap();
        assert_eq!(words[0], CANARY);
        assert_eq!(&words[1..10], &[0u32, 1536, 0, 512, 0, 3, 0, 0, 0][..]);
        assert_eq!(written, 1..10);
        assert_eq!(layout.prefix_cu_range(1).unwrap(), 1..3);
        assert_eq!(layout.suffix_cu_range(1).unwrap(), 3..5);
        assert_eq!(layout.history_cu_range(1).unwrap(), 5..7);
        assert_eq!(layout.source_range(1).unwrap(), 8..10);
        assert_eq!((layout.source_range(1).unwrap().start * 4) % 8, 0);
    }

    #[test]
    fn prefix_only_is_valid_but_suffix_is_never_launched() {
        let layout = Qwen35BlockLayout::new(7, 7, 32, channels(32)).unwrap();
        assert!(!layout.has_suffix());
        assert!(!layout.suffix_launch_required());
        assert_eq!(layout.suffix_tokens(), 0);
        assert_eq!(layout.suffix_cu(), [0, 0]);
        assert_eq!(layout.history_rows(), 3);
        assert_eq!(layout.history_start_elements(), 4 * 8192);
        assert_eq!(layout.history_start(), 4 * 8192);
        assert_eq!(layout.history_elements(), 3 * 8192);

        let mut words = [CANARY; 10];
        let written = layout.pack_into(&mut words, 0).unwrap();
        assert_eq!(written, 0..8);
        assert_eq!(&words[..8], &[0u32, 7, 0, 0, 0, 3, 0, 0][..]);
        assert_eq!(&words[8..], &[CANARY; 2][..]);
    }

    #[test]
    fn zero_bounds_and_mismatched_geometry_are_rejected() {
        assert!(Qwen35BlockLayout::new(0, 1, 32, channels(32)).is_err());
        assert!(Qwen35BlockLayout::new(7, 0, 32, channels(32)).is_err());
        assert!(Qwen35BlockLayout::new(7, 8, 32, channels(32)).is_err());
        assert!(Qwen35BlockLayout::new(MAX_BLOCK_TOKENS + 1, 1, 32, channels(32)).is_err());
        for value_heads in [0, 16, 31, 64, usize::MAX] {
            assert!(Qwen35BlockLayout::new(7, 3, value_heads, channels(32)).is_err());
        }
        assert!(Qwen35BlockLayout::new(7, 3, 32, 8193).is_err());
        assert!(Qwen35BlockLayout::new(7, 3, 32, channels(48)).is_err());
        assert!(Qwen35BlockLayout::new(7, 3, 48, 10_239).is_err());
        assert!(Qwen35BlockLayout::new(7, 3, 48, usize::MAX).is_err());
        assert!(Qwen35BlockLayout::new(7, 3, 48, channels(48)).is_ok());
    }

    #[test]
    fn undersized_buffers_and_overflowing_bases_leave_canaries_untouched() {
        let layout = Qwen35BlockLayout::new(7, 3, 32, channels(32)).unwrap();
        let mut short = [CANARY; 7];
        assert!(layout.pack_into(&mut short, 0).is_err());
        assert_eq!(short, [CANARY; 7]);

        let mut odd = [CANARY; 9];
        assert!(layout.pack_into(&mut odd, 1).is_err());
        assert_eq!(odd, [CANARY; 9]);

        assert!(layout.packed_end(usize::MAX).is_err());
        assert!(layout.packed_range(usize::MAX).is_err());
        assert!(layout.pack_into(&mut [], usize::MAX).is_err());
        assert!(layout.source_range(usize::MAX).is_err());
    }

    #[test]
    fn borrowed_state_metadata_and_combined_row_splits_are_exact() {
        let layout = Qwen35BlockLayout::new(7, 3, 32, channels(32)).unwrap();
        assert_eq!(layout.prefix_cumulative(), [0u32, 3].as_slice());
        assert_eq!(layout.suffix_cumulative(), [0u32, 4].as_slice());
        assert_eq!(layout.history_cumulative(), [0i32, 3].as_slice());
        assert_eq!(layout.sources(), [0i64].as_slice());
        assert_eq!(layout.value_width().unwrap(), 4096);
        assert_eq!(layout.ab_width().unwrap(), 64);
        assert_eq!(
            layout.split_rows(8192, "qkv").unwrap(),
            (0..3 * 8192, 3 * 8192..7 * 8192)
        );
        assert_eq!(
            layout.split_rows(4096, "z").unwrap(),
            (0..3 * 4096, 3 * 4096..7 * 4096)
        );
        assert_eq!(
            layout.split_rows(64, "ab").unwrap(),
            (0..3 * 64, 3 * 64..7 * 64)
        );
        assert_eq!(layout.history_range(), 0..3 * 8192);

        let prefix_only = Qwen35BlockLayout::new(7, 7, 32, channels(32)).unwrap();
        assert_eq!(prefix_only.suffix_cumulative(), [0u32, 0].as_slice());
        assert_eq!(
            prefix_only.split_rows(64, "ab").unwrap(),
            (0..7 * 64, 7 * 64..7 * 64)
        );
        assert_eq!(prefix_only.history_range(), 4 * 8192..7 * 8192);
    }

    #[test]
    fn dynamic_mode_resolves_different_prefixes_at_the_same_tokens() {
        let mode = Qwen35BlockMode::Dynamic;
        assert_eq!(mode.fixed_prefix(), None);
        assert_eq!(mode.resolve_prefix(Some(3), 7).unwrap(), 3);
        assert_eq!(mode.resolve_prefix(Some(5), 7).unwrap(), 5);
        for prefix in [1usize, 3, 5, 7] {
            let layout = Qwen35BlockLayout::new(7, prefix, 32, channels(32)).unwrap();
            assert_eq!(layout.tokens(), 7);
            assert_eq!(layout.prefix_tokens(), prefix);
            assert_eq!(layout.suffix_tokens(), 7 - prefix);
        }
    }

    #[test]
    fn invalid_request_prefixes_fail_before_any_pack_write() {
        let mode = Qwen35BlockMode::Dynamic;
        let stage = |words: &mut [u32], requested: Option<usize>| {
            let prefix = mode.resolve_prefix(requested, 7)?;
            let layout = Qwen35BlockLayout::new(7, prefix, 32, channels(32))?;
            layout.pack_into(words, 0)
        };
        for requested in [None, Some(0), Some(8), Some(usize::MAX)] {
            let mut words = [CANARY; 12];
            assert!(stage(&mut words, requested).is_err());
            assert_eq!(words, [CANARY; 12]);
        }
        let mut words = [CANARY; 12];
        assert_eq!(stage(&mut words, Some(3)).unwrap(), 0..8);
        assert_eq!(&words[8..], &[CANARY; 4][..]);
    }

    #[test]
    fn fixed_mode_is_unchanged_and_rejects_other_request_prefixes() {
        let mode = Qwen35BlockMode::Fixed(3);
        assert_eq!(mode.fixed_prefix(), Some(3));
        assert_eq!(mode.resolve_prefix(None, 7).unwrap(), 3);
        assert_eq!(mode.resolve_prefix(Some(3), 7).unwrap(), 3);
        assert!(mode.resolve_prefix(Some(4), 7).is_err());
        assert!(mode.resolve_prefix(None, 2).is_err());

        let mut words = [CANARY; 12];
        let layout = Qwen35BlockLayout::new(7, 3, 32, channels(32)).unwrap();
        assert_eq!(layout.pack_into(&mut words, 0).unwrap(), 0..8);
        assert_eq!(&words[..8], &[0u32, 3, 0, 4, 0, 3, 0, 0][..]);
        assert_eq!(&words[8..], &[CANARY; 4][..]);
    }

    #[test]
    fn every_base_word_gets_an_eight_byte_aligned_source_tail() {
        let layout = Qwen35BlockLayout::new(2048, 1536, 48, channels(48)).unwrap();
        for base in 0..6 {
            let end = layout.packed_end(base).unwrap();
            assert_eq!(end, base + (if base % 2 == 0 { 8 } else { 9 }));
            let mut words = vec![CANARY; end + 1];
            let written = layout.pack_into(&mut words, base).unwrap();
            assert_eq!(written, base..end);
            assert_eq!(words[end], CANARY);
            assert!(words[..base].iter().all(|&word| word == CANARY));
            let source = layout.source_range(base).unwrap();
            assert_eq!(source.len(), SOURCE_WORDS);
            assert_eq!((source.start * 4) % 8, 0);
            assert!(source.start >= layout.history_cu_range(base).unwrap().end);
            assert!(source.start - layout.history_cu_range(base).unwrap().end <= 1);
            assert_eq!(words[source.start], 0);
            assert_eq!(words[source.start + 1], 0);
        }
    }
}
