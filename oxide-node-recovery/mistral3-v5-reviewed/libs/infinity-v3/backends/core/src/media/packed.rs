//! Packed sequence spans, image/audio descriptors, positions, and borrow views.

use super::{
    AudioLayoutPolicy, AudioSelection, F32View, ImageLayoutPolicy, MediaDevice, MediaDtype,
    MediaError, MediaFamily, MediaLayout,
};
use crate::multimodal::{AudioFeatures, ImagePatches, MultimodalEncoding};
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

/// Validated packed sequence boundaries. Each sequence is nonempty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedSequences<'a> {
    cumulative: &'a [u32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceSpan {
    pub index: usize,
    pub start: usize,
    pub length: usize,
}

impl<'a> PackedSequences<'a> {
    pub fn new(cumulative: &'a [u32]) -> Result<Self, MediaError> {
        if cumulative.len() < 2 {
            return Err(MediaError::Empty("packed sequence boundaries"));
        }
        if cumulative[0] != 0 {
            return Err(MediaError::SequenceSpan {
                sequence_start: cumulative[0] as usize,
                sequence_length: 0,
                tokens: 0,
            });
        }
        for pair in cumulative.windows(2) {
            let start = pair[0] as usize;
            let end = pair[1] as usize;
            if end <= start {
                return Err(MediaError::SequenceSpan {
                    sequence_start: start,
                    sequence_length: end.saturating_sub(start),
                    tokens: *cumulative.last().unwrap_or(&0) as usize,
                });
            }
        }
        let tokens = cumulative.last().copied().unwrap_or(0) as usize;
        if tokens == 0 {
            return Err(MediaError::Empty("packed tokens"));
        }
        Ok(Self { cumulative })
    }

    pub fn batch(&self) -> usize {
        self.cumulative.len() - 1
    }

    pub fn tokens(&self) -> usize {
        self.cumulative.last().copied().unwrap_or(0) as usize
    }

    pub fn offsets(&self) -> &'a [u32] {
        self.cumulative
    }

    pub fn sequence(&self, row: usize) -> Option<SequenceSpan> {
        let start = *self.cumulative.get(row)? as usize;
        let end = *self.cumulative.get(row + 1)? as usize;
        Some(SequenceSpan {
            index: row,
            start,
            length: end - start,
        })
    }

    /// Locate one nonempty token span and reject spans crossing sequence boundaries.
    pub fn span(&self, token_start: usize, token_count: usize) -> Result<SequenceSpan, MediaError> {
        let tokens = self.tokens();
        let end = token_start
            .checked_add(token_count)
            .filter(|&end| token_count != 0 && token_start < tokens && end <= tokens)
            .ok_or(MediaError::TokenSpan {
                token_start,
                token_count,
                tokens,
            })?;
        let row = self
            .cumulative
            .partition_point(|&boundary| (boundary as usize) <= token_start)
            .checked_sub(1)
            .ok_or(MediaError::TokenSpan {
                token_start,
                token_count,
                tokens,
            })?;
        let sequence_start = self.cumulative[row] as usize;
        let sequence_end = self.cumulative[row + 1] as usize;
        if end > sequence_end {
            return Err(MediaError::SpanCrossesSequence {
                token_start,
                token_count,
                sequence_start,
                sequence_length: sequence_end - sequence_start,
            });
        }
        Ok(SequenceSpan {
            index: row,
            start: sequence_start,
            length: sequence_end - sequence_start,
        })
    }
}

/// One validated image descriptor. The payload is not copied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageDescriptor {
    pub family: MediaFamily,
    pub grid_thw: [usize; 3],
    pub patch_dim: usize,
    pub patch_rows: usize,
    pub merge_size: usize,
    pub token_rows: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub sequence_start: usize,
    pub sequence_length: usize,
    pub layout: MediaLayout,
    policy: ImageLayoutPolicy,
}

impl ImageDescriptor {
    pub fn new(
        family: MediaFamily,
        patches: &ImagePatches,
        token_start: usize,
        sequences: &PackedSequences<'_>,
    ) -> Result<Self, MediaError> {
        Self::new_with_policy(
            family,
            patches,
            token_start,
            sequences,
            family.image_policy(),
        )
    }

    /// Use a checkpoint-derived policy when the family patch geometry is
    /// configurable (for example non-16 Pixtral patch sizes).
    pub fn new_with_policy(
        family: MediaFamily,
        patches: &ImagePatches,
        token_start: usize,
        sequences: &PackedSequences<'_>,
        policy: ImageLayoutPolicy,
    ) -> Result<Self, MediaError> {
        policy.validate_grid(family, patches.grid_thw)?;
        policy.validate_merge(family, patches.merge_size)?;
        policy.validate_patch_dim(patches.patch_dim)?;
        let patch_rows = policy.patch_rows(patches.grid_thw)?;
        let token_rows = policy.token_rows(family, patches.grid_thw, patches.merge_size)?;
        let expected = patch_rows
            .checked_mul(patches.patch_dim)
            .ok_or(MediaError::DimensionOverflow("image payload"))?;
        if expected != patches.pixels.len() {
            return Err(MediaError::PayloadLength {
                expected,
                actual: patches.pixels.len(),
            });
        }
        let layout = MediaLayout::row_major(
            patch_rows,
            patches.patch_dim,
            MediaDtype::F32,
            MediaDevice::Cpu,
        )?;
        let sequence = sequences.span(token_start, token_rows)?;
        Ok(Self {
            family,
            grid_thw: patches.grid_thw,
            patch_dim: patches.patch_dim,
            patch_rows,
            merge_size: patches.merge_size,
            token_rows,
            token_start,
            token_count: token_rows,
            sequence_start: sequence.start,
            sequence_length: sequence.length,
            layout,
            policy,
        })
    }

    pub fn source_rows_for_token(
        &self,
        token_row: usize,
        out: &mut [usize],
    ) -> Result<(), MediaError> {
        self.policy.source_rows_for_token(
            self.family,
            self.grid_thw,
            self.merge_size,
            token_row,
            out,
        )
    }

    pub fn policy(&self) -> ImageLayoutPolicy {
        self.policy
    }
}

/// One validated audio span. `subsampled_frames` index the projected tower
/// output; `source_frames` index the original F32 feature rows. They are kept
/// separate so callers cannot accidentally address raw frames with output rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDescriptor {
    pub family: MediaFamily,
    pub frames: usize,
    pub feature_size: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub sequence_start: usize,
    pub sequence_length: usize,
    pub layout: MediaLayout,
    pub subsampled_frames: Vec<usize>,
    pub source_frames: Vec<usize>,
}

impl AudioDescriptor {
    pub fn new(
        family: MediaFamily,
        audio: &AudioFeatures,
        token_start: usize,
        sequences: &PackedSequences<'_>,
        policy: &AudioLayoutPolicy,
    ) -> Result<Self, MediaError> {
        if !family.supports_audio() {
            return Err(MediaError::UnsupportedAudioFamily { family });
        }
        let frames = audio.mask.len();
        if frames == 0 {
            return Err(MediaError::ZeroDimension("audio frames"));
        }
        if audio.feature_size != policy.feature_size() {
            return Err(MediaError::AudioFeatureSize {
                expected: policy.feature_size(),
                actual: audio.feature_size,
            });
        }
        if audio.subsample_strides != policy.subsample_strides() {
            return Err(MediaError::AudioStride {
                expected: policy.subsample_strides(),
                actual: audio.subsample_strides,
            });
        }
        let expected = frames
            .checked_mul(policy.feature_size())
            .ok_or(MediaError::DimensionOverflow("audio payload"))?;
        if expected != audio.values.len() {
            return Err(MediaError::AudioFeatureLength {
                expected,
                actual: audio.values.len(),
            });
        }
        let AudioSelection {
            subsampled_frames,
            source_frames,
        } = policy.select_frames(&audio.mask)?;
        if subsampled_frames.is_empty() {
            return Err(MediaError::AudioEmptySelection);
        }
        if audio.token_count() != subsampled_frames.len() {
            return Err(MediaError::AudioTokenCount {
                expected: subsampled_frames.len(),
                actual: audio.token_count(),
            });
        }
        let layout = MediaLayout::row_major(
            frames,
            policy.feature_size(),
            MediaDtype::F32,
            MediaDevice::Cpu,
        )?;
        let sequence = sequences.span(token_start, subsampled_frames.len())?;
        Ok(Self {
            family,
            frames,
            feature_size: policy.feature_size(),
            token_start,
            token_count: subsampled_frames.len(),
            sequence_start: sequence.start,
            sequence_length: sequence.length,
            layout,
            subsampled_frames,
            source_frames,
        })
    }
}

/// Three-axis mRoPE positions for a packed token sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPositions {
    axes: [Vec<u32>; 3],
}

impl PackedPositions {
    pub fn axes(&self) -> &[Vec<u32>; 3] {
        &self.axes
    }

    pub fn into_axes(self) -> [Vec<u32>; 3] {
        self.axes
    }

    pub fn len(&self) -> usize {
        self.axes[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.axes[0].is_empty()
    }
}

/// Build the native Qwen mRoPE position remap. Image spans are canonicalized by
/// token start, so callers may supply images in any order. Text gaps advance all
/// axes together. This is the single checked implementation used by production
/// Qwen processors through `image_positions`.
pub fn image_position_ids<'a>(
    tokens: usize,
    images: impl IntoIterator<Item = (usize, &'a ImagePatches)>,
) -> Result<PackedPositions, MediaError> {
    let mut ordered: Vec<(usize, &ImagePatches)> = Vec::new();
    let images = images.into_iter();
    let (lower, upper) = images.size_hint();
    ordered
        .try_reserve(upper.unwrap_or(lower))
        .map_err(|_| MediaError::Overflow("multimodal image allocation"))?;
    for image in images {
        ordered
            .try_reserve(1)
            .map_err(|_| MediaError::Overflow("multimodal image allocation"))?;
        ordered.push(image);
    }
    ordered.sort_unstable_by_key(|&(start, _)| start);
    let mut axes: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
    for axis in &mut axes {
        axis.try_reserve_exact(tokens)
            .map_err(|_| MediaError::Overflow("multimodal position allocation"))?;
    }
    let mut cursor = 0usize;
    let mut next = 0u32;
    for (start, image) in ordered {
        let patches = image
            .grid_thw
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or(MediaError::DimensionOverflow("image grid"))?;
        let merge_area = image
            .merge_size
            .checked_mul(image.merge_size)
            .filter(|&merge| merge > 0)
            .ok_or(MediaError::ZeroDimension("image merge size"))?;
        if patches == 0
            || !image.grid_thw[1].is_multiple_of(image.merge_size)
            || !image.grid_thw[2].is_multiple_of(image.merge_size)
            || !patches.is_multiple_of(merge_area)
        {
            return Err(MediaError::MergeMismatch {
                grid: image.grid_thw,
                merge: image.merge_size,
            });
        }
        let length = patches / merge_area;
        let end = start
            .checked_add(length)
            .filter(|&end| length != 0 && end <= tokens)
            .ok_or(MediaError::TokenSpan {
                token_start: start,
                token_count: length,
                tokens,
            })?;
        if start < cursor {
            return Err(MediaError::OverlappingSpans {
                previous_end: cursor,
                next_start: start,
            });
        }
        for _ in cursor..start {
            for axis in &mut axes {
                axis.push(next);
            }
            next = next.checked_add(1).ok_or(MediaError::PositionOverflow)?;
        }
        let height = image.grid_thw[1] / image.merge_size;
        let width = image.grid_thw[2] / image.merge_size;
        for frame in 0..image.grid_thw[0] {
            let temporal = next
                .checked_add(u32::try_from(frame).map_err(|_| MediaError::PositionOverflow)?)
                .ok_or(MediaError::PositionOverflow)?;
            for row in 0..height {
                let row = next
                    .checked_add(u32::try_from(row).map_err(|_| MediaError::PositionOverflow)?)
                    .ok_or(MediaError::PositionOverflow)?;
                for column in 0..width {
                    let column = next
                        .checked_add(
                            u32::try_from(column).map_err(|_| MediaError::PositionOverflow)?,
                        )
                        .ok_or(MediaError::PositionOverflow)?;
                    axes[0].push(temporal);
                    axes[1].push(row);
                    axes[2].push(column);
                }
            }
        }
        let advance = height.max(width).max(image.grid_thw[0]);
        next = next
            .checked_add(u32::try_from(advance).map_err(|_| MediaError::PositionOverflow)?)
            .ok_or(MediaError::PositionOverflow)?;
        cursor = end;
    }
    for _ in cursor..tokens {
        for axis in &mut axes {
            axis.push(next);
        }
        next = next.checked_add(1).ok_or(MediaError::PositionOverflow)?;
    }
    if axes.iter().any(|axis| axis.len() != tokens) {
        return Err(MediaError::PositionLength {
            expected: tokens,
            actual: axes[0].len(),
        });
    }
    Ok(PackedPositions { axes })
}

/// Feature-row gather selection with optional padded destination rows.
/// Reordering is explicit; duplicate destinations are rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureRowSelection {
    sources: Vec<Option<usize>>,
}

impl FeatureRowSelection {
    pub fn new(
        source_rows: usize,
        destination_rows: usize,
        pairs: &[(usize, usize)],
    ) -> Result<Self, MediaError> {
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(destination_rows)
            .map_err(|_| MediaError::Overflow("feature selection allocation"))?;
        sources.resize(destination_rows, None);
        let mut seen = Vec::new();
        seen.try_reserve_exact(destination_rows)
            .map_err(|_| MediaError::Overflow("feature selection seen allocation"))?;
        seen.resize(destination_rows, false);
        for &(source, destination) in pairs {
            if source >= source_rows {
                return Err(MediaError::SelectionIndex {
                    index: source,
                    limit: source_rows,
                });
            }
            if destination >= destination_rows {
                return Err(MediaError::SelectionIndex {
                    index: destination,
                    limit: destination_rows,
                });
            }
            if std::mem::replace(&mut seen[destination], true) {
                return Err(MediaError::DuplicateDestination { destination });
            }
            sources[destination] = Some(source);
        }
        Ok(Self { sources })
    }

    pub fn source_for_destination(&self, destination: usize) -> Option<usize> {
        self.sources.get(destination).copied().flatten()
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn rows(&self) -> &[Option<usize>] {
        &self.sources
    }
}

struct OwnedImage {
    patches: Arc<ImagePatches>,
    descriptor: ImageDescriptor,
}

struct OwnedAudio {
    features: Arc<AudioFeatures>,
    descriptor: AudioDescriptor,
}

/// Owns media payload handles, the processor's memory reservation, and validated
/// descriptor metadata for one packed request. Borrow views never clone storage.
pub struct PackedMedia {
    family: MediaFamily,
    token_count: usize,
    images: Vec<OwnedImage>,
    audios: Vec<OwnedAudio>,
    memory: Option<Arc<OwnedSemaphorePermit>>,
}

impl PackedMedia {
    pub fn new(
        family: MediaFamily,
        cumulative: &[u32],
        images: &[(usize, Arc<ImagePatches>)],
        audios: &[(usize, Arc<AudioFeatures>)],
        audio_policy: Option<AudioLayoutPolicy>,
        memory: Option<Arc<OwnedSemaphorePermit>>,
    ) -> Result<Self, MediaError> {
        Self::new_with_policies(
            family,
            cumulative,
            images,
            audios,
            None,
            audio_policy,
            memory,
        )
    }

    /// Build with a checkpoint-derived image policy (for example non-16
    /// Pixtral patching) while retaining the same checked media ABI.
    pub fn new_with_policies(
        family: MediaFamily,
        cumulative: &[u32],
        images: &[(usize, Arc<ImagePatches>)],
        audios: &[(usize, Arc<AudioFeatures>)],
        image_policy: Option<ImageLayoutPolicy>,
        audio_policy: Option<AudioLayoutPolicy>,
        memory: Option<Arc<OwnedSemaphorePermit>>,
    ) -> Result<Self, MediaError> {
        if !audios.is_empty() && audio_policy.is_none() {
            return Err(MediaError::MissingAudioPolicy);
        }
        let sequences = PackedSequences::new(cumulative)?;
        let token_count = sequences.tokens();
        let mut owned_images = Vec::new();
        owned_images
            .try_reserve_exact(images.len())
            .map_err(|_| MediaError::Overflow("image descriptor allocation"))?;
        for (start, patches) in images {
            let policy = image_policy.unwrap_or_else(|| family.image_policy());
            let descriptor =
                ImageDescriptor::new_with_policy(family, patches, *start, &sequences, policy)?;
            owned_images.push(OwnedImage {
                patches: Arc::clone(patches),
                descriptor,
            });
        }
        let mut owned_audios = Vec::new();
        owned_audios
            .try_reserve_exact(audios.len())
            .map_err(|_| MediaError::Overflow("audio descriptor allocation"))?;
        let audio_policy = match (audios.is_empty(), audio_policy) {
            (true, _) => None,
            (false, Some(policy)) => Some(policy),
            (false, None) => return Err(MediaError::MissingAudioPolicy),
        };
        for (start, features) in audios {
            let policy = audio_policy
                .as_ref()
                .ok_or(MediaError::MissingAudioPolicy)?;
            let descriptor = AudioDescriptor::new(family, features, *start, &sequences, policy)?;
            owned_audios.push(OwnedAudio {
                features: Arc::clone(features),
                descriptor,
            });
        }
        validate_nonoverlap(
            owned_images
                .iter()
                .map(|image| (image.descriptor.token_start, image.descriptor.token_count))
                .chain(
                    owned_audios
                        .iter()
                        .map(|audio| (audio.descriptor.token_start, audio.descriptor.token_count)),
                ),
        )?;
        Ok(Self {
            family,
            token_count,
            images: owned_images,
            audios: owned_audios,
            memory,
        })
    }

    /// Build from the real core `MultimodalEncoding` ABI, retaining its memory
    /// permit and payload handles without copying pixel/feature vectors.
    pub fn from_encoding(
        family: MediaFamily,
        cumulative: &[u32],
        encoding: &MultimodalEncoding,
        audio_policy: Option<AudioLayoutPolicy>,
    ) -> Result<Self, MediaError> {
        Self::new(
            family,
            cumulative,
            &encoding.images,
            &encoding.audios,
            audio_policy,
            encoding.memory.clone(),
        )
    }

    pub fn memory(&self) -> Option<&Arc<OwnedSemaphorePermit>> {
        self.memory.as_ref()
    }

    pub fn family(&self) -> MediaFamily {
        self.family
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn images_len(&self) -> usize {
        self.images.len()
    }

    pub fn audios_len(&self) -> usize {
        self.audios.len()
    }

    pub fn image_descriptors(&self) -> impl Iterator<Item = &ImageDescriptor> {
        self.images.iter().map(|image| &image.descriptor)
    }

    pub fn audio_descriptors(&self) -> impl Iterator<Item = &AudioDescriptor> {
        self.audios.iter().map(|audio| &audio.descriptor)
    }

    /// Constant-cost borrow view. The descriptors and payload `Arc`s stay owned
    /// by `self`; no pixel/feature vector is cloned.
    pub fn view(&self) -> PackedMediaView<'_> {
        PackedMediaView { media: self }
    }

    /// Qwen mRoPE remap for the currently owned image descriptors. Descriptors
    /// may be reordered; the production position function receives token order.
    pub fn position_ids_for_images(&self) -> Result<PackedPositions, MediaError> {
        let mut images = Vec::new();
        images
            .try_reserve_exact(self.images.len())
            .map_err(|_| MediaError::Overflow("multimodal image allocation"))?;
        for image in &self.images {
            images.push((image.descriptor.token_start, image.patches.as_ref()));
        }
        images.sort_unstable_by_key(|&(start, _)| start);
        image_position_ids(self.token_count, images)
    }
}

/// Borrowed request view produced by [`PackedMedia::view`].
#[derive(Clone, Copy)]
pub struct PackedMediaView<'a> {
    media: &'a PackedMedia,
}

impl<'a> PackedMediaView<'a> {
    pub fn family(&self) -> MediaFamily {
        self.media.family
    }

    pub fn token_count(&self) -> usize {
        self.media.token_count
    }

    pub fn images(&self) -> impl Iterator<Item = ImageView<'a>> + 'a {
        self.media.images.iter().map(|image| ImageView {
            data: &image.patches.pixels,
            descriptor: &image.descriptor,
        })
    }

    pub fn audios(&self) -> impl Iterator<Item = AudioView<'a>> + 'a {
        self.media.audios.iter().map(|audio| AudioView {
            data: &audio.features.values,
            mask: &audio.features.mask,
            descriptor: &audio.descriptor,
        })
    }
}

#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    data: &'a [f32],
    descriptor: &'a ImageDescriptor,
}

impl<'a> ImageView<'a> {
    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    pub fn descriptor(&self) -> &'a ImageDescriptor {
        self.descriptor
    }

    pub fn layout(&self) -> MediaLayout {
        self.descriptor.layout
    }

    pub fn as_f32_view(&self) -> Result<F32View<'a>, MediaError> {
        F32View::new(self.data, self.descriptor.layout)
    }

    pub fn row(&self, row: usize) -> Option<&'a [f32]> {
        self.descriptor.layout.row_slice(self.data, row)
    }

    pub fn source_rows_for_token(
        &self,
        token_row: usize,
        out: &mut [usize],
    ) -> Result<(), MediaError> {
        self.descriptor.source_rows_for_token(token_row, out)
    }
}

#[derive(Clone, Copy)]
pub struct AudioView<'a> {
    data: &'a [f32],
    mask: &'a [u8],
    descriptor: &'a AudioDescriptor,
}

impl<'a> AudioView<'a> {
    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    pub fn mask(&self) -> &'a [u8] {
        self.mask
    }

    pub fn descriptor(&self) -> &'a AudioDescriptor {
        self.descriptor
    }

    pub fn layout(&self) -> MediaLayout {
        self.descriptor.layout
    }

    /// Positions in the projected/subsampled tower output.
    pub fn subsampled_frames(&self) -> &'a [usize] {
        &self.descriptor.subsampled_frames
    }

    /// Corresponding rows in `data()` when the caller must address raw frames.
    pub fn source_frames(&self) -> &'a [usize] {
        &self.descriptor.source_frames
    }
}

fn validate_nonoverlap(spans: impl IntoIterator<Item = (usize, usize)>) -> Result<(), MediaError> {
    let mut ordered: Vec<(usize, usize)> = Vec::new();
    let spans = spans.into_iter();
    let (lower, upper) = spans.size_hint();
    ordered
        .try_reserve(upper.unwrap_or(lower))
        .map_err(|_| MediaError::Overflow("media span allocation"))?;
    for span in spans {
        ordered
            .try_reserve(1)
            .map_err(|_| MediaError::Overflow("media span allocation"))?;
        ordered.push(span);
    }
    ordered.sort_unstable_by_key(|&(start, _)| start);
    let mut cursor = 0usize;
    for (start, count) in ordered {
        if start < cursor {
            return Err(MediaError::OverlappingSpans {
                previous_end: cursor,
                next_start: start,
            });
        }
        cursor = start
            .checked_add(count)
            .ok_or(MediaError::DimensionOverflow("media token span"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::PatchOrder;
    use tokio::sync::Semaphore;

    fn image(grid: [usize; 3], merge: usize, patch_dim: usize) -> ImagePatches {
        let rows = grid.iter().product::<usize>();
        ImagePatches {
            pixels: vec![0.25; rows * patch_dim],
            grid_thw: grid,
            patch_dim,
            merge_size: merge,
        }
    }

    fn plain(
        family: MediaFamily,
        cumulative: &[u32],
        images: &[(usize, Arc<ImagePatches>)],
        audios: &[(usize, Arc<AudioFeatures>)],
    ) -> Result<PackedMedia, MediaError> {
        PackedMedia::new(family, cumulative, images, audios, None, None)
    }

    fn audio_policy() -> AudioLayoutPolicy {
        AudioLayoutPolicy::for_embedding_gemma2(128, [2, 2]).unwrap()
    }

    #[test]
    fn qwen_ragged_temporal_grids_and_positions_match_native_contract() {
        let first = Arc::new(image([2, 4, 6], 2, 1536));
        let second = Arc::new(image([1, 4, 4], 2, 1536));
        // 3 text tokens + 12 + 2 text + 4 + 2 text = 23 packed tokens.
        let cumulative = [0u32, 23];
        let media = plain(
            MediaFamily::Qwen3Vl,
            &cumulative,
            &[(3, Arc::clone(&first)), (17, Arc::clone(&second))],
            &[],
        )
        .unwrap();
        assert_eq!(media.token_count(), 23);
        let descriptors: Vec<_> = media.image_descriptors().collect();
        assert_eq!(descriptors[0].patch_rows, 48);
        assert_eq!(descriptors[0].token_rows, 12);
        assert_eq!(descriptors[0].sequence_start, 0);
        assert_eq!(descriptors[1].patch_rows, 16);
        assert_eq!(descriptors[1].token_rows, 4);

        let positions = media.position_ids_for_images().unwrap();
        assert_eq!(positions.len(), 23);
        assert_eq!(&positions.axes()[0][..3], &[0, 1, 2]);
        assert_eq!(&positions.axes()[0][15..17], &[6, 7]);
        assert_eq!(&positions.axes()[0][17..21], &[8; 4]);
        assert_eq!(&positions.axes()[1][3..9], &[3, 3, 3, 4, 4, 4]);
        assert_eq!(&positions.axes()[2][3..9], &[3, 4, 5, 3, 4, 5]);
        assert_eq!(&positions.axes()[0][3..9], &[3, 3, 3, 3, 3, 3]);
        assert_eq!(&positions.axes()[0][9..15], &[4; 6]);
        assert_eq!(&positions.axes()[0][21..], &[10, 11]);
    }

    #[test]
    fn positions_accept_reordered_images_and_reject_bad_spans() {
        let portrait = image([1, 6, 4], 2, 1536);
        let landscape = image([1, 2, 6], 2, 1536);
        let ours = image_position_ids(15, [(10, &landscape), (2, &portrait)]).unwrap();
        assert_eq!(
            ours.axes()[0],
            [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 10, 11]
        );
        assert_eq!(
            ours.axes()[1],
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 10, 11]
        );
        assert_eq!(
            ours.axes()[2],
            [0, 1, 2, 3, 2, 3, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        );
        assert!(image_position_ids(15, [(2, &portrait), (7, &landscape)]).is_err());
        assert!(image_position_ids(7, [(2, &portrait)]).is_err());
    }

    #[test]
    fn reordered_image_descriptors_remap_by_token_start_without_cloning() {
        let first = Arc::new(image([1, 4, 6], 2, 1536));
        let second = Arc::new(image([1, 4, 4], 2, 1536));
        let first_ptr = first.pixels.as_ptr();
        let cumulative = [0u32, 2, 14, 18, 20];
        let media = plain(
            MediaFamily::Qwen3Vl,
            &cumulative,
            &[(2, Arc::clone(&second)), (2, Arc::clone(&first))],
            &[],
        );
        // Both descriptors start at token 2, so the malformed selection is rejected.
        assert!(matches!(media, Err(MediaError::OverlappingSpans { .. })));

        let media = plain(
            MediaFamily::Qwen3Vl,
            &[0u32, 20],
            &[(10, Arc::clone(&second)), (2, Arc::clone(&first))],
            &[],
        )
        .unwrap();
        let view = media.view();
        let views: Vec<_> = view.images().collect();
        let first_view = views
            .iter()
            .find(|view| view.descriptor().token_start == 2)
            .unwrap();
        assert_eq!(first_view.data().as_ptr(), first_ptr);
        let positions = media.position_ids_for_images().unwrap();
        assert_eq!(&positions.axes()[0][2..8], &[2; 6]);
        assert_eq!(&positions.axes()[0][10..14], &[7; 4]);
        assert_eq!(&positions.axes()[0][14..], &[9, 10, 11, 12, 13, 14]);
        assert_eq!(Arc::strong_count(&first), 2);
    }

    #[test]
    fn feature_selection_allows_reorder_and_padding_but_rejects_aliases() {
        let selection = FeatureRowSelection::new(4, 6, &[(3, 0), (1, 2), (0, 5)]).unwrap();
        assert_eq!(
            selection.rows(),
            &[Some(3), None, Some(1), None, None, Some(0)]
        );
        assert_eq!(selection.source_for_destination(2), Some(1));
        assert_eq!(selection.source_for_destination(1), None);
        assert_eq!(
            FeatureRowSelection::new(4, 2, &[(0, 0), (1, 0)]),
            Err(MediaError::DuplicateDestination { destination: 0 })
        );
        assert!(matches!(
            FeatureRowSelection::new(4, 2, &[(4, 0)]),
            Err(MediaError::SelectionIndex { .. })
        ));
    }

    #[test]
    fn sequence_spans_reject_crossing_empty_and_reordered_boundaries() {
        let sequences = PackedSequences::new(&[0, 3, 9]).unwrap();
        assert_eq!(
            sequences.span(1, 2).unwrap(),
            SequenceSpan {
                index: 0,
                start: 0,
                length: 3
            }
        );
        assert_eq!(sequences.span(3, 6).unwrap().index, 1);
        assert!(matches!(
            sequences.span(2, 2),
            Err(MediaError::SpanCrossesSequence { .. })
        ));
        assert!(matches!(
            sequences.span(0, 0),
            Err(MediaError::TokenSpan { .. })
        ));
        assert!(PackedSequences::new(&[0, 3, 3]).is_err());
        assert!(PackedSequences::new(&[0]).is_err());
    }

    #[test]
    fn embedding_gemma2_audio_padding_and_native_subsampling_are_checked() {
        let mut mask = vec![1u8; 99];
        mask[40..].fill(0);
        let audio = Arc::new(AudioFeatures {
            values: vec![0.5; 99 * 128],
            mask,
            feature_size: 128,
            subsample_strides: [2, 2],
        });
        assert_eq!(audio.token_count(), 10);
        let media = PackedMedia::new(
            MediaFamily::EmbeddingGemma2,
            &[0u32, 13],
            &[],
            &[(3, Arc::clone(&audio))],
            Some(audio_policy()),
            None,
        )
        .unwrap();
        let descriptor = media.audio_descriptors().next().unwrap();
        assert_eq!(descriptor.frames, 99);
        assert_eq!(descriptor.feature_size, 128);
        assert_eq!(descriptor.token_count, 10);
        assert_eq!(descriptor.sequence_start, 0);
        assert_eq!(descriptor.subsampled_frames, (0..10).collect::<Vec<_>>());
        assert_eq!(
            descriptor.source_frames,
            (0..10).map(|frame| frame * 4).collect::<Vec<_>>()
        );
        assert_eq!(
            descriptor.layout.shape(),
            [99, 128],
            "audio keeps all padded frames; tokens select valid subsamples"
        );
        let view = media.view();
        let audio_view = view.audios().next().unwrap();
        assert_eq!(audio_view.data().as_ptr(), audio.values.as_ptr());
        assert_eq!(audio_view.subsampled_frames(), &(0..10).collect::<Vec<_>>());

        // A last-frame-only mask exercises the native clamped final subsample.
        let trailing = Arc::new(AudioFeatures {
            values: vec![0.0; 5 * 128],
            mask: vec![0, 0, 0, 0, 1],
            feature_size: 128,
            subsample_strides: [2, 2],
        });
        assert_eq!(trailing.token_count(), 1);
        let descriptor = AudioDescriptor::new(
            MediaFamily::EmbeddingGemma2,
            &trailing,
            0,
            &PackedSequences::new(&[0, 1]).unwrap(),
            &audio_policy(),
        )
        .unwrap();
        assert_eq!(descriptor.subsampled_frames, [1]);
        assert_eq!(descriptor.source_frames, [4]);

        let empty = Arc::new(AudioFeatures {
            values: vec![0.0; 5 * 128],
            mask: vec![0; 5],
            feature_size: 128,
            subsample_strides: [2, 2],
        });
        assert!(matches!(
            AudioDescriptor::new(
                MediaFamily::EmbeddingGemma2,
                &empty,
                0,
                &PackedSequences::new(&[0, 1]).unwrap(),
                &audio_policy()
            ),
            Err(MediaError::AudioEmptySelection)
        ));
        let wrong_stride = Arc::new(AudioFeatures {
            values: vec![0.0; 99 * 128],
            mask: vec![1; 99],
            feature_size: 128,
            subsample_strides: [2, 3],
        });
        assert_eq!(
            AudioDescriptor::new(
                MediaFamily::EmbeddingGemma2,
                &wrong_stride,
                0,
                &PackedSequences::new(&[0, 1]).unwrap(),
                &audio_policy()
            ),
            Err(MediaError::AudioStride {
                expected: [2, 2],
                actual: [2, 3],
            })
        );

        let wrong_size = Arc::new(AudioFeatures {
            values: vec![0.0; 99 * 64],
            mask: vec![1; 99],
            feature_size: 64,
            subsample_strides: [2, 2],
        });
        assert!(matches!(
            AudioDescriptor::new(
                MediaFamily::EmbeddingGemma2,
                &wrong_size,
                0,
                &PackedSequences::new(&[0, 1]).unwrap(),
                &audio_policy()
            ),
            Err(MediaError::AudioFeatureSize { .. })
        ));
        assert!(matches!(
            AudioDescriptor::new(
                MediaFamily::Gemma4,
                &audio,
                0,
                &PackedSequences::new(&[0, 1]).unwrap(),
                &audio_policy()
            ),
            Err(MediaError::UnsupportedAudioFamily { .. })
        ));
        assert!(matches!(
            PackedMedia::new(
                MediaFamily::EmbeddingGemma2,
                &[0u32, 13],
                &[],
                &[(3, Arc::clone(&audio))],
                None,
                None
            ),
            Err(MediaError::MissingAudioPolicy)
        ));
    }

    #[test]
    fn mixed_image_and_audio_spans_are_checked_together() {
        let image = Arc::new(image([1, 6, 9], 3, 768));
        let audio = Arc::new(AudioFeatures {
            values: vec![1.0; 16 * 128],
            mask: vec![1; 16],
            feature_size: 128,
            subsample_strides: [2, 2],
        });
        let result = PackedMedia::new(
            MediaFamily::EmbeddingGemma2,
            &[0u32, 20],
            &[(2, image)],
            &[(4, audio)],
            Some(audio_policy()),
            None,
        );
        assert!(matches!(result, Err(MediaError::OverlappingSpans { .. })));
    }

    #[test]
    fn from_encoding_retains_the_processor_memory_permit() {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = Arc::new(semaphore.clone().try_acquire_owned().unwrap());
        assert_eq!(semaphore.available_permits(), 0);
        let encoding = MultimodalEncoding {
            images: vec![(0, Arc::new(image([1, 4, 6], 2, 1536)))],
            audios: vec![],
            position_ids: std::array::from_fn(|_| vec![0; 6]),
            memory: Some(permit),
        };
        let media =
            PackedMedia::from_encoding(MediaFamily::Qwen3Vl, &[0u32, 6], &encoding, None).unwrap();
        drop(encoding);
        assert_eq!(semaphore.available_permits(), 0);
        assert!(media.memory().is_some());
        drop(media);
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[test]
    fn gemma_and_ultravbert_descriptors_keep_their_own_grid_rules() {
        let gemma = Arc::new(image([1, 6, 9], 3, 768));
        let ultra = Arc::new(image([3, 32, 32], 4, 768));
        let media = plain(
            MediaFamily::Gemma4,
            &[0u32, 8],
            &[(1, Arc::clone(&gemma))],
            &[],
        )
        .unwrap();
        let descriptor = media.image_descriptors().next().unwrap();
        assert_eq!(descriptor.token_rows, 6);
        assert_eq!(descriptor.policy().patch_order, PatchOrder::SpatialRowMajor);

        let media = plain(
            MediaFamily::UltraVBert,
            &[0u32, 193],
            &[(1, Arc::clone(&ultra))],
            &[],
        )
        .unwrap();
        let descriptor = media.image_descriptors().next().unwrap();
        assert_eq!(descriptor.token_rows, 192);
        assert_eq!(descriptor.policy().patch_order, PatchOrder::SpatialRowMajor);
        assert!(matches!(
            plain(
                MediaFamily::UltraVBert,
                &[0u32, 33],
                &[(1, Arc::new(image([1, 32, 32], 3, 768)))],
                &[]
            ),
            Err(MediaError::InvalidMergeSize { .. })
        ));
    }

    #[test]
    fn packed_media_can_use_a_checkpoint_image_policy() {
        let mistral = Arc::new(image([1, 4, 6], 2, 3 * 32 * 32));
        let policy = MediaFamily::Mistral3
            .image_policy()
            .with_patch_size(32)
            .unwrap();
        let media = PackedMedia::new_with_policies(
            MediaFamily::Mistral3,
            &[0u32, 7],
            &[(1, mistral)],
            &[],
            Some(policy),
            None,
            None,
        )
        .unwrap();
        let descriptor = media.image_descriptors().next().unwrap();
        assert_eq!(descriptor.policy().patch_size, 32);
        assert_eq!(descriptor.patch_dim, 3072);
    }

    #[test]
    fn configurable_patch_size_is_family_scoped() {
        let mistral = Arc::new(image([1, 4, 6], 2, 3 * 32 * 32));
        let sequences = PackedSequences::new(&[0, 10]).unwrap();
        assert!(matches!(
            ImageDescriptor::new(MediaFamily::Mistral3, &mistral, 0, &sequences),
            Err(MediaError::PatchDimension { .. })
        ));
        let policy = MediaFamily::Mistral3
            .image_policy()
            .with_patch_size(32)
            .unwrap();
        let descriptor = ImageDescriptor::new_with_policy(
            MediaFamily::Mistral3,
            &mistral,
            0,
            &sequences,
            policy,
        )
        .unwrap();
        assert_eq!(descriptor.patch_dim, 3072);
        assert!(matches!(
            MediaFamily::Gemma4.image_policy().with_patch_size(32),
            Err(MediaError::PatchSize { fixed: true, .. })
        ));
    }

    #[test]
    fn malformed_media_and_checked_overflows_are_typed() {
        let sequences = PackedSequences::new(&[0, 10]).unwrap();
        let mut bad = image([1, 4, 6], 2, 1536);
        bad.pixels.pop();
        assert!(matches!(
            ImageDescriptor::new(MediaFamily::Qwen3Vl, &bad, 0, &sequences),
            Err(MediaError::PayloadLength { .. })
        ));
        let mut bad = image([1, 4, 6], 2, 1536);
        bad.patch_dim = 1535;
        assert!(matches!(
            ImageDescriptor::new(MediaFamily::Qwen3Vl, &bad, 0, &sequences),
            Err(MediaError::PatchDimension { .. })
        ));
        let bad = image([1, 3, 6], 2, 1536);
        assert!(matches!(
            ImageDescriptor::new(MediaFamily::Qwen3Vl, &bad, 0, &sequences),
            Err(MediaError::MergeMismatch { .. })
        ));
        let policy = MediaFamily::Gemma4.image_policy();
        assert!(matches!(
            policy.patch_rows([usize::MAX, 2, 2]),
            Err(MediaError::DimensionOverflow(_))
        ));
    }
}
