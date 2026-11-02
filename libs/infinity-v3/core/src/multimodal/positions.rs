use infinity_v3_backend::{media::image_position_ids, ImagePatches};

/// Each span is (first image placeholder token, image). Positions are per sequence,
/// before queue packing. Identical placeholder IDs do not establish image equality.
///
/// The checked implementation lives in the shared backend media foundation so Qwen
/// processors and media-aware executors cannot drift from one another.
pub fn image_positions(
    token_count: usize,
    images: &[(usize, &ImagePatches)],
) -> Result<[Vec<u32>; 3], String> {
    image_position_ids(token_count, images.iter().copied())
        .map(|positions| positions.into_axes())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multiple_images_resume_text_after_largest_spatial_axis() {
        let portrait = ImagePatches {
            pixels: vec![],
            grid_thw: [1, 6, 4],
            patch_dim: 1536,
            merge_size: 2,
        };
        let landscape = ImagePatches {
            grid_thw: [1, 2, 6],
            ..portrait.clone()
        };
        let positions = image_positions(15, &[(2, &portrait), (10, &landscape)]).unwrap();
        assert_eq!(
            positions[0],
            [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 10, 11]
        );
        assert_eq!(
            positions[1],
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 10, 11]
        );
        assert_eq!(
            positions[2],
            [0, 1, 2, 3, 2, 3, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        );
        assert!(image_positions(15, &[(2, &portrait), (7, &landscape)]).is_err());
        assert!(image_positions(7, &[(2, &portrait)]).is_err());
    }

    #[test]
    fn malformed_image_geometry_is_rejected_without_panicking() {
        let zero_merge = ImagePatches {
            pixels: vec![],
            grid_thw: [1, 2, 2],
            patch_dim: 1,
            merge_size: 0,
        };
        assert!(image_positions(4, &[(0, &zero_merge)]).is_err());
        let overflow = ImagePatches {
            pixels: vec![],
            grid_thw: [usize::MAX, 2, 2],
            patch_dim: 1,
            merge_size: 2,
        };
        assert!(image_positions(4, &[(0, &overflow)]).is_err());
    }
}
