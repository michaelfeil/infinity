//! Explicit per-family image/audio layout policies.
//!
//! The native towers do not share one patch order. Qwen's preprocessor emits
//! merge-group-major rows (and can carry multiple temporal frames), Mistral and
//! Gemma emit spatial row-major rows, and UltraVBERT tiles a fixed 32x32 grid.
//! Keeping these policies separate prevents a generic loader from silently
//! accepting a layout that a family executor would misinterpret.

use super::MediaError;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaFamily {
    Qwen3Vl,
    Qwen35,
    D3,
    Mistral3,
    Gemma4,
    EmbeddingGemma2,
    UltraVBert,
}

impl MediaFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen35 => "qwen3_5",
            Self::D3 => "d3",
            Self::Mistral3 => "mistral3",
            Self::Gemma4 => "gemma4",
            Self::EmbeddingGemma2 => "embedding_gemma2",
            Self::UltraVBert => "ultravbert",
        }
    }

    /// Vision patch order emitted by the corresponding processor.
    pub const fn image_policy(self) -> ImageLayoutPolicy {
        match self {
            Self::Qwen3Vl | Self::Qwen35 | Self::D3 => ImageLayoutPolicy {
                patch_order: PatchOrder::TemporalMergeGroupMajor,
                temporal_patch_size: 2,
                frames: FramePolicy::Any,
                merge: MergePolicy::Configurable { max: 8 },
                rows: ExtentPolicy::Any,
                columns: ExtentPolicy::Any,
                patch_size: 16,
                patch_size_fixed: false,
            },
            Self::Mistral3 => ImageLayoutPolicy {
                patch_order: PatchOrder::SpatialRowMajor,
                temporal_patch_size: 1,
                frames: FramePolicy::SingleFrame,
                merge: MergePolicy::Configurable { max: 8 },
                rows: ExtentPolicy::Any,
                columns: ExtentPolicy::Any,
                patch_size: 16,
                patch_size_fixed: false,
            },
            Self::Gemma4 | Self::EmbeddingGemma2 => ImageLayoutPolicy {
                patch_order: PatchOrder::SpatialRowMajor,
                temporal_patch_size: 1,
                frames: FramePolicy::SingleFrame,
                merge: MergePolicy::Fixed(3),
                rows: ExtentPolicy::Any,
                columns: ExtentPolicy::Any,
                patch_size: 16,
                patch_size_fixed: true,
            },
            Self::UltraVBert => ImageLayoutPolicy {
                patch_order: PatchOrder::SpatialRowMajor,
                temporal_patch_size: 1,
                frames: FramePolicy::Tiles { max: 5 },
                merge: MergePolicy::Fixed(4),
                rows: ExtentPolicy::Fixed(32),
                columns: ExtentPolicy::Fixed(32),
                patch_size: 16,
                patch_size_fixed: true,
            },
        }
    }

    /// EmbeddingGemma2 is the only native family with an audio tower today.
    pub const fn supports_audio(self) -> bool {
        matches!(self, Self::EmbeddingGemma2)
    }
}

impl fmt::Display for MediaFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchOrder {
    /// Qwen preprocess order: frame, merge block row/column, merge row/column.
    TemporalMergeGroupMajor,
    /// Mistral/Gemma/UltraVBERT preprocess order: frame, patch row, patch column.
    SpatialRowMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePolicy {
    Any,
    SingleFrame,
    /// Axis 0 contains independent image tiles rather than temporal frames.
    Tiles {
        max: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePolicy {
    Configurable { max: usize },
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtentPolicy {
    Any,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLayoutPolicy {
    pub patch_order: PatchOrder,
    pub temporal_patch_size: usize,
    pub frames: FramePolicy,
    pub merge: MergePolicy,
    pub rows: ExtentPolicy,
    pub columns: ExtentPolicy,
    pub patch_size: usize,
    patch_size_fixed: bool,
}

impl ImageLayoutPolicy {
    pub fn validate_grid(&self, family: MediaFamily, grid: [usize; 3]) -> Result<(), MediaError> {
        if grid.contains(&0) {
            return Err(MediaError::ZeroDimension("image grid"));
        }
        match self.frames {
            FramePolicy::Any => {}
            FramePolicy::SingleFrame if grid[0] != 1 => {
                return Err(MediaError::UnsupportedFrames {
                    family,
                    frames: grid[0],
                    allowed: "exactly one frame",
                });
            }
            FramePolicy::Tiles { max } if grid[0] > max => {
                return Err(MediaError::UnsupportedFrames {
                    family,
                    frames: grid[0],
                    allowed: "1..=5 tiles",
                });
            }
            _ => {}
        }
        if let ExtentPolicy::Fixed(expected) = self.rows {
            if grid[1] != expected {
                return Err(MediaError::UnsupportedGrid {
                    family,
                    axis: "height",
                    expected,
                    actual: grid[1],
                });
            }
        }
        if let ExtentPolicy::Fixed(expected) = self.columns {
            if grid[2] != expected {
                return Err(MediaError::UnsupportedGrid {
                    family,
                    axis: "width",
                    expected,
                    actual: grid[2],
                });
            }
        }
        Ok(())
    }

    pub fn validate_merge(&self, family: MediaFamily, merge: usize) -> Result<(), MediaError> {
        if merge == 0 {
            return Err(MediaError::ZeroDimension("image merge size"));
        }
        match self.merge {
            MergePolicy::Fixed(expected) if merge != expected => {
                Err(MediaError::InvalidMergeSize {
                    family,
                    merge,
                    max: expected,
                })
            }
            MergePolicy::Configurable { max } if merge > max => {
                Err(MediaError::InvalidMergeSize { family, merge, max })
            }
            _ => Ok(()),
        }
    }

    /// Use the checkpoint's patch size when the family loader permits it.
    pub fn with_patch_size(self, patch_size: usize) -> Result<Self, MediaError> {
        if patch_size == 0
            || patch_size > 64
            || (self.patch_size_fixed && patch_size != self.patch_size)
        {
            return Err(MediaError::PatchSize {
                expected: self.patch_size,
                actual: patch_size,
                fixed: self.patch_size_fixed,
            });
        }
        Ok(Self { patch_size, ..self })
    }

    pub fn expected_patch_dim(&self) -> Result<usize, MediaError> {
        let area = self
            .patch_size
            .checked_mul(self.patch_size)
            .ok_or(MediaError::DimensionOverflow("image patch area"))?;
        3usize
            .checked_mul(self.temporal_patch_size)
            .and_then(|channels| channels.checked_mul(area))
            .ok_or(MediaError::DimensionOverflow("image patch dimension"))
    }

    pub fn validate_patch_dim(&self, patch_dim: usize) -> Result<(), MediaError> {
        let expected = self.expected_patch_dim()?;
        if patch_dim != expected {
            return Err(MediaError::PatchDimension {
                expected,
                actual: patch_dim,
            });
        }
        Ok(())
    }

    /// Checked number of pre-merge patch rows.
    pub fn patch_rows(&self, grid: [usize; 3]) -> Result<usize, MediaError> {
        grid.iter().try_fold(1usize, |rows, &extent| {
            rows.checked_mul(extent)
                .ok_or(MediaError::DimensionOverflow("image patch rows"))
        })
    }

    /// Checked number of post-merge feature rows.
    pub fn token_rows(
        &self,
        family: MediaFamily,
        grid: [usize; 3],
        merge: usize,
    ) -> Result<usize, MediaError> {
        self.validate_grid(family, grid)?;
        self.validate_merge(family, merge)?;
        if !grid[1].is_multiple_of(merge) || !grid[2].is_multiple_of(merge) {
            return Err(MediaError::MergeMismatch { grid, merge });
        }
        let merge_area = merge
            .checked_mul(merge)
            .ok_or(MediaError::DimensionOverflow("image merge area"))?;
        let rows = self.patch_rows(grid)?;
        if rows % merge_area != 0 {
            return Err(MediaError::MergeMismatch { grid, merge });
        }
        Ok(rows / merge_area)
    }

    pub fn token_rows_per_frame(
        &self,
        family: MediaFamily,
        grid: [usize; 3],
        merge: usize,
    ) -> Result<usize, MediaError> {
        let rows = self.token_rows(family, grid, merge)?;
        let frames = grid[0];
        if frames == 0 {
            return Err(MediaError::ZeroDimension("image frames"));
        }
        Ok(rows / frames)
    }

    /// Write the `merge * merge` pre-merge patch rows consumed by one feature row.
    /// This is the explicit family ordering used by merger/projector code.
    pub fn source_rows_for_token(
        &self,
        family: MediaFamily,
        grid: [usize; 3],
        merge: usize,
        token_row: usize,
        out: &mut [usize],
    ) -> Result<(), MediaError> {
        let token_rows = self.token_rows(family, grid, merge)?;
        if token_row >= token_rows {
            return Err(MediaError::SelectionIndex {
                index: token_row,
                limit: token_rows,
            });
        }
        let merge_area = merge
            .checked_mul(merge)
            .ok_or(MediaError::DimensionOverflow("image merge area"))?;
        if out.len() != merge_area {
            return Err(MediaError::SelectionLength {
                expected: merge_area,
                actual: out.len(),
            });
        }
        let blocks_wide = grid[2] / merge;
        let blocks_high = grid[1] / merge;
        let per_frame = blocks_high
            .checked_mul(blocks_wide)
            .ok_or(MediaError::DimensionOverflow("image merge blocks"))?;
        let frame = token_row / per_frame;
        let local = token_row % per_frame;
        let block_row = local / blocks_wide;
        let block_column = local % blocks_wide;
        let frame_rows = self.patch_rows([1, grid[1], grid[2]])?;
        let frame_base = frame
            .checked_mul(frame_rows)
            .ok_or(MediaError::DimensionOverflow("image frame rows"))?;
        match self.patch_order {
            PatchOrder::TemporalMergeGroupMajor => {
                let block = block_row
                    .checked_mul(blocks_wide)
                    .and_then(|value| value.checked_add(block_column))
                    .ok_or(MediaError::DimensionOverflow("image merge block"))?;
                let base = frame_base
                    .checked_add(
                        block
                            .checked_mul(merge_area)
                            .ok_or(MediaError::DimensionOverflow("image merge row"))?,
                    )
                    .ok_or(MediaError::DimensionOverflow("image merge row"))?;
                for merge_row in 0..merge {
                    for merge_column in 0..merge {
                        out[merge_row * merge + merge_column] = base
                            .checked_add(merge_row * merge + merge_column)
                            .ok_or(MediaError::DimensionOverflow("image source row"))?;
                    }
                }
            }
            PatchOrder::SpatialRowMajor => {
                for merge_row in 0..merge {
                    let row = block_row
                        .checked_mul(merge)
                        .and_then(|value| value.checked_add(merge_row))
                        .ok_or(MediaError::DimensionOverflow("image source row"))?;
                    for merge_column in 0..merge {
                        let column = block_column
                            .checked_mul(merge)
                            .and_then(|value| value.checked_add(merge_column))
                            .ok_or(MediaError::DimensionOverflow("image source column"))?;
                        let offset = row
                            .checked_mul(grid[2])
                            .and_then(|value| value.checked_add(column))
                            .ok_or(MediaError::DimensionOverflow("image source row"))?;
                        out[merge_row * merge + merge_column] = frame_base
                            .checked_add(offset)
                            .ok_or(MediaError::DimensionOverflow("image source row"))?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioLayoutPolicy {
    feature_size: usize,
    subsample_strides: [usize; 2],
}

/// Two related index spaces produced by the native audio subsampler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSelection {
    /// Positions in the projected/subsampled tower output.
    pub subsampled_frames: Vec<usize>,
    /// Corresponding rows in the original F32 feature matrix, clamped exactly
    /// like the native validity-mask gather.
    pub source_frames: Vec<usize>,
}

impl AudioLayoutPolicy {
    pub const fn feature_size(&self) -> usize {
        self.feature_size
    }

    pub const fn subsample_strides(&self) -> [usize; 2] {
        self.subsample_strides
    }

    /// Build a config-driven policy. The processor should pass the checkpoint's
    /// `feature_size` and `sscp_conv_stride_size` values here.
    pub fn new(feature_size: usize, subsample_strides: [usize; 2]) -> Result<Self, MediaError> {
        if feature_size == 0 {
            return Err(MediaError::ZeroDimension("audio feature size"));
        }
        if subsample_strides.contains(&0) {
            return Err(MediaError::ZeroDimension("audio subsample stride"));
        }
        Ok(Self {
            feature_size,
            subsample_strides,
        })
    }

    /// The currently pinned EmbeddingGemma2 checkpoint uses 128 mel bins and two
    /// stride-2 convolutions. Other configurations fail closed until covered by
    /// a native parity fixture.
    pub fn for_embedding_gemma2(
        feature_size: usize,
        subsample_strides: [usize; 2],
    ) -> Result<Self, MediaError> {
        if feature_size != 128 || subsample_strides != [2, 2] {
            return Err(MediaError::UnsupportedAudioConfig {
                feature_size,
                subsample_strides,
            });
        }
        Self::new(feature_size, subsample_strides)
    }

    /// Native Gemma audio applies two stride-2 convolutions. Each stage selects
    /// `ceil(length / stride)` entries before reading the validity mask.
    pub fn subsampled_frames(&self, frames: usize) -> Result<usize, MediaError> {
        if frames == 0 {
            return Err(MediaError::ZeroDimension("audio frames"));
        }
        let mut length = frames;
        for stride in self.subsample_strides {
            if stride == 0 {
                return Err(MediaError::ZeroDimension("audio subsample stride"));
            }
            length = length
                .checked_add(stride - 1)
                .ok_or(MediaError::DimensionOverflow("audio subsample"))?
                / stride;
        }
        Ok(length)
    }

    /// Select valid subsampled frames and their original raw-feature rows.
    pub fn select_frames(&self, mask: &[u8]) -> Result<AudioSelection, MediaError> {
        let length = self.subsampled_frames(mask.len())?;
        let mut subsampled_frames = Vec::new();
        let mut source_frames = Vec::new();
        subsampled_frames
            .try_reserve_exact(length)
            .map_err(|_| MediaError::Overflow("audio selected frame allocation"))?;
        source_frames
            .try_reserve_exact(length)
            .map_err(|_| MediaError::Overflow("audio source frame allocation"))?;
        for frame in 0..length {
            let source = frame
                .checked_mul(self.subsample_strides[0])
                .and_then(|value| value.checked_mul(self.subsample_strides[1]))
                .ok_or(MediaError::DimensionOverflow("audio selected frame"))?
                .min(mask.len() - 1);
            if mask[source] != 0 {
                subsampled_frames.push(frame);
                source_frames.push(source);
            }
        }
        Ok(AudioSelection {
            subsampled_frames,
            source_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_merge_groups_are_not_mistral_spatial_rows() {
        let qwen = MediaFamily::Qwen3Vl.image_policy();
        let mut rows = [usize::MAX; 4];
        qwen.source_rows_for_token(MediaFamily::Qwen3Vl, [1, 4, 6], 2, 0, &mut rows)
            .unwrap();
        assert_eq!(rows, [0, 1, 2, 3]);
        qwen.source_rows_for_token(MediaFamily::Qwen3Vl, [1, 4, 6], 2, 1, &mut rows)
            .unwrap();
        assert_eq!(rows, [4, 5, 6, 7]);

        let mistral = MediaFamily::Mistral3.image_policy();
        mistral
            .source_rows_for_token(MediaFamily::Mistral3, [1, 4, 6], 2, 0, &mut rows)
            .unwrap();
        assert_eq!(rows, [0, 1, 6, 7]);
        mistral
            .source_rows_for_token(MediaFamily::Mistral3, [1, 4, 6], 2, 1, &mut rows)
            .unwrap();
        assert_eq!(rows, [2, 3, 8, 9]);
    }

    #[test]
    fn qwen_temporal_frames_offset_each_frame_in_both_orders() {
        let qwen = MediaFamily::Qwen3Vl.image_policy();
        let mut rows = [0usize; 4];
        qwen.source_rows_for_token(MediaFamily::Qwen3Vl, [2, 4, 6], 2, 6, &mut rows)
            .unwrap();
        assert_eq!(rows, [24, 25, 26, 27]);

        let gemma = MediaFamily::Gemma4.image_policy();
        let mut rows = [0usize; 9];
        gemma
            .source_rows_for_token(MediaFamily::Gemma4, [1, 6, 9], 3, 1, &mut rows)
            .unwrap();
        assert_eq!(rows, [3, 4, 5, 12, 13, 14, 21, 22, 23]);
    }

    #[test]
    fn family_grid_rules_are_explicit_and_typed() {
        let qwen = MediaFamily::Qwen3Vl.image_policy();
        qwen.validate_grid(MediaFamily::Qwen3Vl, [2, 4, 6]).unwrap();
        assert_eq!(
            qwen.token_rows(MediaFamily::Qwen3Vl, [2, 4, 6], 2).unwrap(),
            12
        );

        let gemma = MediaFamily::Gemma4.image_policy();
        assert_eq!(
            gemma.validate_grid(MediaFamily::Gemma4, [2, 6, 9]),
            Err(MediaError::UnsupportedFrames {
                family: MediaFamily::Gemma4,
                frames: 2,
                allowed: "exactly one frame"
            })
        );
        assert!(matches!(
            gemma.validate_merge(MediaFamily::Gemma4, 2),
            Err(MediaError::InvalidMergeSize { max: 3, .. })
        ));

        let ultra = MediaFamily::UltraVBert.image_policy();
        ultra
            .validate_grid(MediaFamily::UltraVBert, [3, 32, 32])
            .unwrap();
        assert!(matches!(
            ultra.validate_grid(MediaFamily::UltraVBert, [6, 32, 32]),
            Err(MediaError::UnsupportedFrames { .. })
        ));
        assert!(matches!(
            ultra.validate_grid(MediaFamily::UltraVBert, [1, 16, 32]),
            Err(MediaError::UnsupportedGrid {
                axis: "height",
                expected: 32,
                ..
            })
        ));
    }

    #[test]
    fn audio_subsampling_matches_the_native_clamped_two_stage_index() {
        let policy = AudioLayoutPolicy::for_embedding_gemma2(128, [2, 2]).unwrap();
        assert_eq!(policy.subsampled_frames(99).unwrap(), 25);
        assert_eq!(policy.subsampled_frames(98).unwrap(), 25);
        let mut mask = vec![1u8; 99];
        for value in &mut mask[40..] {
            *value = 0;
        }
        let selection = policy.select_frames(&mask).unwrap();
        assert_eq!(selection.subsampled_frames, (0..10).collect::<Vec<_>>());
        assert_eq!(
            selection.source_frames,
            (0..10).map(|frame| frame * 4).collect::<Vec<_>>()
        );
        assert!(policy.subsampled_frames(0).is_err());
        assert!(matches!(
            AudioLayoutPolicy::for_embedding_gemma2(128, [2, 3]),
            Err(MediaError::UnsupportedAudioConfig { .. })
        ));
        let generic = AudioLayoutPolicy::new(64, [2, 3]).unwrap();
        let selection = generic.select_frames(&[1; 24]).unwrap();
        assert_eq!(selection.subsampled_frames, (0..4).collect::<Vec<_>>());
        assert_eq!(selection.source_frames, [0, 6, 12, 18]);
    }
}
