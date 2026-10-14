use infinity_v3_backend::ImagePatches;

/// Each span is (first image placeholder token, image). Positions are per sequence,
/// before queue packing. Identical placeholder IDs do not establish image equality.
pub fn image_positions(
    token_count: usize,
    images: &[(usize, &ImagePatches)],
) -> Result<[Vec<u32>; 3], String> {
    let mut result = [
        Vec::with_capacity(token_count),
        Vec::with_capacity(token_count),
        Vec::with_capacity(token_count),
    ];
    let mut cursor = 0;
    let mut next = 0u32;
    for &(start, image) in images {
        let length = image.token_count();
        if start < cursor
            || length == 0
            || start
                .checked_add(length)
                .is_none_or(|end| end > token_count)
        {
            return Err("Image placeholders overlap or exceed the token sequence".into());
        }
        for _ in cursor..start {
            for axis in &mut result {
                axis.push(next);
            }
            next += 1;
        }
        let h = image.grid_thw[1] / image.merge_size;
        let w = image.grid_thw[2] / image.merge_size;
        for t in 0..image.grid_thw[0] {
            for y in 0..h {
                for x in 0..w {
                    result[0].push(next + t as u32);
                    result[1].push(next + y as u32);
                    result[2].push(next + x as u32);
                }
            }
        }
        next += h.max(w).max(image.grid_thw[0]) as u32;
        cursor = start + length;
    }
    for _ in cursor..token_count {
        for axis in &mut result {
            axis.push(next);
        }
        next += 1;
    }
    Ok(result)
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
}
