//! CPU validation of the real processor fixtures used by the Root GPU forward.
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    ids: Vec<u32>,
    cumulative: Vec<u32>,
    media_positions: Option<[Vec<u32>; 3]>,
    images: Vec<Image>,
}

#[derive(Deserialize)]
struct Image {
    pixels: String,
    rows: usize,
    patch_dim: usize,
    grid_thw: [usize; 3],
    merge_size: usize,
    token_start: usize,
    token_count: usize,
}

#[test]
fn real_processor_fixtures_are_self_consistent() {
    let Ok(path) = std::env::var("QWEN3_VL_FIXTURES") else {
        eprintln!("QWEN3_VL_FIXTURES absent; skipping real fixture validation");
        return;
    };
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(Path::new(&path).join("fixtures.json")).unwrap())
            .unwrap();
    assert!(manifest.fixtures.len() >= 8);
    let mut grids = BTreeSet::new();
    let mut saw_frames = false;
    let mut saw_ragged = false;
    let mut saw_empty = false;
    for fixture in &manifest.fixtures {
        assert!(!fixture.ids.is_empty(), "{} is empty", fixture.name);
        assert_eq!(fixture.cumulative.first().copied(), Some(0));
        assert_eq!(
            fixture.cumulative.last().copied(),
            Some(fixture.ids.len() as u32)
        );
        assert!(fixture.cumulative.windows(2).all(|pair| pair[0] <= pair[1]));
        if fixture.cumulative.len() > 2 {
            saw_ragged = true;
        }
        if fixture.images.is_empty() {
            saw_empty = true;
            assert!(fixture.media_positions.is_none());
        } else {
            let positions = fixture.media_positions.as_ref().expect("media positions");
            for axis in positions {
                assert_eq!(axis.len(), fixture.ids.len());
            }
        }
        let mut spans = Vec::new();
        for image in &fixture.images {
            let patches = image.grid_thw.iter().product::<usize>();
            assert_eq!(patches, image.rows);
            let merge = image.merge_size * image.merge_size;
            assert_eq!(image.rows / merge, image.token_count);
            assert!(image.token_start + image.token_count <= fixture.ids.len());
            let bytes = std::fs::metadata(&image.pixels).unwrap().len();
            assert_eq!(bytes, (image.rows * image.patch_dim * 4) as u64);
            grids.insert(image.grid_thw);
            saw_frames |= image.grid_thw[0] > 1;
            spans.push((image.token_start, image.token_start + image.token_count));
        }
        spans.sort_unstable();
        assert!(spans.windows(2).all(|pair| pair[0].1 <= pair[1].0));
    }
    assert!(grids.len() >= 3, "fixtures should vary image shapes");
    assert!(saw_frames, "three-frame fixture missing");
    assert!(saw_ragged, "ragged multi-sequence fixture missing");
    assert!(saw_empty, "empty-image fixture missing");
}
