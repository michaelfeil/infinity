//! Prepared image tensors aligned to one token sequence.
use std::{fmt, sync::Arc};
use tokio::sync::OwnedSemaphorePermit;

#[derive(Clone)]
pub struct ImagePatches {
    pub pixels: Vec<f32>,
    pub grid_thw: [usize; 3],
    pub patch_dim: usize,
    pub merge_size: usize,
}

impl ImagePatches {
    fn same_content(&self, other: &Self) -> bool {
        self.grid_thw == other.grid_thw
            && self.patch_dim == other.patch_dim
            && self.merge_size == other.merge_size
            && self.pixels.len() == other.pixels.len()
            && self
                .pixels
                .iter()
                .zip(&other.pixels)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    pub fn token_count(&self) -> usize {
        self.grid_thw.iter().product::<usize>() / (self.merge_size * self.merge_size)
    }
}

impl fmt::Debug for ImagePatches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImagePatches")
            .field("grid_thw", &self.grid_thw)
            .field("patch_dim", &self.patch_dim)
            .field("merge_size", &self.merge_size)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct AudioFeatures {
    pub values: Vec<f32>,
    pub mask: Vec<u8>,
    pub feature_size: usize,
}

impl AudioFeatures {
    pub fn token_count(&self) -> usize {
        self.mask
            .iter()
            .step_by(4)
            .filter(|&&valid| valid != 0)
            .count()
    }
}

#[derive(Debug)]
pub struct MultimodalEncoding {
    /// (First image token, patches), in message/content order.
    pub images: Vec<(usize, Arc<ImagePatches>)>,
    pub audios: Vec<(usize, Arc<AudioFeatures>)>,
    pub position_ids: [Vec<u32>; 3],
    /// Keeps the processor's memory reservation alive until the last batch consumer drops it.
    pub memory: Option<Arc<OwnedSemaphorePermit>>,
}

impl MultimodalEncoding {
    /// End of the final image block, checked before using it as a prefix bound.
    pub fn radix_prefix_len(&self) -> Option<usize> {
        self.images.iter().try_fold(0usize, |end, (start, image)| {
            let patches = image
                .grid_thw
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))?;
            let merge = image.merge_size.checked_mul(image.merge_size)?;
            if merge == 0 || patches == 0 || patches % merge != 0 {
                return None;
            }
            Some(end.max(start.checked_add(patches / merge)?))
        })
    }

    /// Token-prefix folding may share rows only within the same image context.
    pub fn same_radix_context(
        a: Option<&Self>,
        b: Option<&Self>,
        a_ids: &[u32],
        b_ids: &[u32],
    ) -> bool {
        if a.is_some_and(|m| !m.audios.is_empty()) || b.is_some_and(|m| !m.audios.is_empty()) {
            return false;
        }
        let a = a.filter(|m| !m.images.is_empty());
        let b = b.filter(|m| !m.images.is_empty());
        let (a, b) = match (a, b) {
            (None, None) => return true,
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };
        let Some(end) = a.radix_prefix_len() else {
            return false;
        };
        if a_ids.get(..end).is_none()
            || a_ids.get(..end) != b_ids.get(..end)
            || (0..3).any(|axis| {
                a.position_ids[axis].get(..end).is_none()
                    || a.position_ids[axis].get(..end) != b.position_ids[axis].get(..end)
            })
        {
            return false;
        }
        a.images.len() == b.images.len()
            && a.images
                .iter()
                .zip(&b.images)
                .all(|((x, a), (y, b))| x == y && (Arc::ptr_eq(a, b) || a.same_content(b)))
    }

    /// Check the actual shared-row mapping rather than requiring one image
    /// context for the entire batch. Unused compact padding rows are ignored.
    pub fn allows_radix_fold(
        media: &[Option<Arc<Self>>],
        ids: &[u32],
        cumulative: &[u32],
        scatter: &[u32],
        fold: &[u32],
    ) -> bool {
        if cumulative.len() != media.len() + 1
            || cumulative.first() != Some(&0)
            || cumulative.last().map(|n| *n as usize) != Some(ids.len())
            || cumulative.windows(2).any(|w| w[0] > w[1])
            || scatter.len() != ids.len()
        {
            return false;
        }
        let mut row_for_token = vec![0usize; ids.len()];
        for (row, span) in cumulative.windows(2).enumerate() {
            row_for_token[span[0] as usize..span[1] as usize].fill(row);
        }
        let mut checked = std::collections::HashSet::new();
        for (i, compact) in scatter.iter().enumerate() {
            let Some(rep) = fold.get(*compact as usize).map(|v| *v as usize) else {
                return false;
            };
            let Some(b) = row_for_token.get(rep).copied() else {
                return false;
            };
            let a = row_for_token[i];
            if checked.insert((a, b))
                && !Self::same_radix_context(
                    media[a].as_deref(),
                    media[b].as_deref(),
                    &ids[cumulative[a] as usize..cumulative[a + 1] as usize],
                    &ids[cumulative[b] as usize..cumulative[b + 1] as usize],
                )
            {
                return false;
            }
        }
        true
    }

    /// Token-only folding is safe for images only when every sequence has the
    /// same prepared pixel bits/layout and context through the final image block.
    /// Equal placeholder IDs alone are insufficient.
    pub fn allows_radix(media: &[Option<Arc<Self>>], ids: &[u32], cumulative: &[u32]) -> bool {
        if media.iter().flatten().any(|m| !m.audios.is_empty()) {
            return false;
        }
        let Some(first) = media.iter().flatten().find(|m| !m.images.is_empty()) else {
            return true;
        };
        if cumulative.len() != media.len() + 1 {
            return false;
        }
        let end = first
            .images
            .iter()
            .map(|(start, image)| start + image.token_count())
            .max()
            .unwrap();
        if first
            .position_ids
            .iter()
            .any(|axis| axis.get(..end).is_none())
        {
            return false;
        }
        let mut prefix: Option<&[u32]> = None;
        for (row, item) in media.iter().enumerate() {
            let Some(item) = item else { return false };
            if item.images.len() != first.images.len()
                || !item
                    .images
                    .iter()
                    .zip(&first.images)
                    .all(|((a, x), (b, y))| a == b && (Arc::ptr_eq(x, y) || x.same_content(y)))
                || (0..3).any(|axis| {
                    item.position_ids[axis].get(..end) != first.position_ids[axis].get(..end)
                })
            {
                return false;
            }
            let start = cumulative[row] as usize;
            if start + end > cumulative[row + 1] as usize {
                return false;
            }
            let Some(current) = ids.get(start..start + end) else {
                return false;
            };
            if prefix.is_some_and(|prefix| prefix != current) {
                return false;
            }
            prefix = Some(current);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radix_requires_identical_image_content_and_complete_prefix() {
        let image = Arc::new(ImagePatches {
            pixels: vec![0.; 12],
            grid_thw: [1, 2, 2],
            patch_dim: 3,
            merge_size: 1,
        });
        let media = |image| {
            Some(Arc::new(MultimodalEncoding {
                audios: vec![],
                images: vec![(1, image)],
                position_ids: std::array::from_fn(|_| (0..6).collect()),
                memory: None,
            }))
        };
        let first = media(image.clone());
        let ids = [1, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 3];
        assert!(MultimodalEncoding::allows_radix(
            &[first.clone(), media(image.clone())],
            &ids,
            &[0, 6, 12]
        ));
        assert!(MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new((*image).clone()))],
            &ids,
            &[0, 6, 12]
        ));
        let mut different = (*image).clone();
        different.pixels[11] = 1.;
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new(different))],
            &ids,
            &[0, 6, 12]
        ));
        let mut different = (*image).clone();
        different.grid_thw = [1, 1, 4];
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new(different))],
            &ids,
            &[0, 6, 12]
        ));
        let mut changed = ids;
        changed[6] = 2;
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(image)],
            &changed,
            &[0, 6, 12]
        ));
        assert!(!MultimodalEncoding::allows_radix(
            &[first, None],
            &ids,
            &[0, 6, 12]
        ));
    }
}
