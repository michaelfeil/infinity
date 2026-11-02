//! Generate real Qwen3-VL processor fixtures for the Oxide and Native b66
//! consumers. This uses the repository's cached Rust processor and local real
//! images; it does not import Torch, create a GPU context, or download anything.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use infinity_v3_backend::ImagePatches;
use infinity_v3_core::input::Message;
use infinity_v3_core::multimodal::image_positions;
use infinity_v3_core::multimodal::{MultimodalConfig, PreparedMultimodal, Qwen3VlProcessor};
use infinity_v3_core::tokenization::EncodingInput;
use serde_json::{json, Value};
use std::error::Error;
use std::path::{Path, PathBuf};
use tokenizers::{Tokenizer, TruncationDirection};

const IMAGES: &[(&str, &str)] = &[
    (
        "cats",
        "/tmp/infinity-plain-cuda/docs/assets/cats_coco_sample.jpg",
    ),
    (
        "qwen256_0",
        "/workspace/model-performance/michaelfeil/benchmark-overview/qwen-vl-256-0.png",
    ),
    (
        "qwen1024_0",
        "/workspace/model-performance/michaelfeil/benchmark-overview/qwen-vl-1024-0.png",
    ),
    (
        "qwen256_1",
        "/workspace/model-performance/michaelfeil/benchmark-overview/qwen-vl-256-1.png",
    ),
    (
        "qwen1024_1",
        "/workspace/model-performance/michaelfeil/benchmark-overview/qwen-vl-1024-1.png",
    ),
];

struct Case {
    name: &'static str,
    images: &'static [&'static str],
    prompt: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "single-cats",
        images: &["cats"],
        prompt: "Represent this image.",
    },
    Case {
        name: "two-images",
        images: &["cats", "qwen1024_0"],
        prompt: "Compare these images.",
    },
    Case {
        name: "varying-shapes",
        images: &["qwen256_0", "qwen1024_1", "qwen256_1", "cats"],
        prompt: "Describe all shapes.",
    },
    Case {
        name: "single-qwen256",
        images: &["qwen256_0"],
        prompt: "Represent the content.",
    },
    Case {
        name: "single-qwen1024",
        images: &["qwen1024_0"],
        prompt: "Represent the content.",
    },
];

fn data_url(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!(
        "data:image/{};base64,{}",
        extension(path),
        STANDARD.encode(bytes)
    ))
}

fn extension(path: &Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()) {
        Some("png") => "png",
        Some("webp") => "webp",
        _ => "jpeg",
    }
}

fn image_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    IMAGES
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, path)| PathBuf::from(path))
        .ok_or_else(|| format!("unknown image {name}").into())
}

fn messages(paths: &[PathBuf], prompt: &str) -> Result<Vec<Message>, Box<dyn Error>> {
    let mut content = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        content.push(json!({"type":"image_url","image_url":{"url": data_url(path)?}}));
        content.push(json!({"type":"text","text": format!("Image {}: ", index + 1)}));
    }
    content.push(json!({"type":"text","text": prompt}));
    Ok(serde_json::from_value(json!([
        {"role":"user","content":content},
        {"role":"assistant","content":"I see the images."},
        {"role":"user","content":"Represent their content."}
    ]))?)
}

async fn prepare(
    processor: &Qwen3VlProcessor,
    paths: &[PathBuf],
    prompt: &str,
) -> Result<PreparedMultimodal, Box<dyn Error>> {
    let messages = messages(paths, prompt)?;
    Ok(processor
        .prepare(
            EncodingInput::Messages(messages),
            false,
            TruncationDirection::Right,
            None,
        )
        .await?)
}

fn write_pixels(
    root: &Path,
    name: &str,
    index: usize,
    patches: &ImagePatches,
) -> Result<String, Box<dyn Error>> {
    let file = format!("{name}.image{index}.f32");
    let mut bytes = Vec::with_capacity(patches.pixels.len() * 4);
    for value in &patches.pixels {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(root.join(&file), bytes)?;
    Ok(file)
}

fn fixture(
    root: &Path,
    name: &str,
    prepared: &PreparedMultimodal,
    sequence_offset: u32,
    image_offset: usize,
) -> Result<Value, Box<dyn Error>> {
    let mut images = Vec::new();
    for (index, (start, patches)) in prepared.media.images.iter().enumerate() {
        let pixels = write_pixels(root, name, image_offset + index, patches)?;
        images.push(json!({
            "name": format!("{name}-{index}"),
            "pixels": root.join(pixels).display().to_string(),
            "rows": patches.pixels.len() / patches.patch_dim,
            "patch_dim": patches.patch_dim,
            "grid_thw": patches.grid_thw,
            "merge_size": patches.merge_size,
            "token_start": sequence_offset as usize + *start,
            "token_count": patches.token_count(),
        }));
    }
    let media_positions = if prepared.media.images.is_empty() {
        None
    } else {
        Some([
            prepared.media.position_ids[0]
                .iter()
                .map(|v| v + sequence_offset)
                .collect::<Vec<_>>(),
            prepared.media.position_ids[1]
                .iter()
                .map(|v| v + sequence_offset)
                .collect::<Vec<_>>(),
            prepared.media.position_ids[2]
                .iter()
                .map(|v| v + sequence_offset)
                .collect::<Vec<_>>(),
        ])
    };
    Ok(json!({
        "ids": prepared.input_ids,
        "types": vec![0u32; prepared.input_ids.len()],
        "positions": (0..prepared.input_ids.len()).collect::<Vec<_>>(),
        "media_positions": media_positions,
        "images": images,
        "tokens": prepared.input_ids.len(),
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .ok_or("usage: qwen3_vl_fixtures CHECKPOINT OUTPUT")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: qwen3_vl_fixtures CHECKPOINT OUTPUT")?,
    );
    std::fs::create_dir_all(&output)?;
    eprintln!(
        "loading tokenizer {}",
        checkpoint.join("tokenizer.json").display()
    );
    let tokenizer = Tokenizer::from_file(checkpoint.join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    eprintln!("loading processor");
    let processor =
        Qwen3VlProcessor::load(&checkpoint, tokenizer, 8192, MultimodalConfig::default())
            .map_err(|error| format!("processor load: {error}"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut fixtures = Vec::new();
    for (case_index, case) in CASES.iter().enumerate() {
        let paths: Vec<PathBuf> = case
            .images
            .iter()
            .map(|name| image_path(name))
            .collect::<Result<Vec<_>, _>>()?;
        eprintln!("preparing {}", case.name);
        let prepared = runtime
            .block_on(prepare(&processor, &paths, case.prompt))
            .map_err(|error| format!("{}: {error}", case.name))?;
        let mut value = fixture(&output, case.name, &prepared, 0, case_index * 8)?;
        value["name"] = json!(case.name);
        value["cumulative"] = json!([0, prepared.input_ids.len()]);
        value["pool"] = json!("last_token");
        fixtures.push(value);
    }
    // Text-only fixture.
    let text = runtime.block_on(async {
        processor
            .prepare(
                EncodingInput::Single("Represent this sentence.".to_owned()),
                false,
                TruncationDirection::Right,
                None,
            )
            .await
    })?;
    fixtures.push(json!({
        "name": "empty-images",
        "ids": text.input_ids,
        "types": vec![0u32; text.input_ids.len()],
        "positions": (0..text.input_ids.len()).collect::<Vec<_>>(),
        "media_positions": null,
        "images": [],
        "cumulative": [0, text.input_ids.len()],
        "pool": "last_token",
    }));
    // Ragged three-sequence fixture: cats, text-only, two images.
    let cats = runtime.block_on(prepare(
        &processor,
        &[PathBuf::from(IMAGES[0].1)],
        "Represent this image.",
    ))?;
    let two = runtime.block_on(prepare(
        &processor,
        &[PathBuf::from(IMAGES[0].1), PathBuf::from(IMAGES[2].1)],
        "Compare these images.",
    ))?;
    let mut ids = Vec::new();
    let mut cumulative = vec![0u32];
    let mut media_axes: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
    let mut images = Vec::new();
    let sequences: [(&PreparedMultimodal, bool); 3] = [(&cats, true), (&text, false), (&two, true)];
    for (sequence, (prepared, has_images)) in sequences.iter().enumerate() {
        let offset = *cumulative.last().unwrap();
        ids.extend_from_slice(&prepared.input_ids);
        cumulative.push(offset + prepared.input_ids.len() as u32);
        if *has_images {
            for axis in 0..3 {
                media_axes[axis].extend(prepared.media.position_ids[axis].iter().copied());
            }
            for (index, (start, patches)) in prepared.media.images.iter().enumerate() {
                let pixels = write_pixels(&output, "ragged", sequence * 4 + index, patches)?;
                images.push(json!({
                    "name": format!("ragged-{sequence}-{index}"),
                    "pixels": output.join(pixels).display().to_string(),
                    "rows": patches.pixels.len() / patches.patch_dim,
                    "patch_dim": patches.patch_dim,
                    "grid_thw": patches.grid_thw,
                    "merge_size": patches.merge_size,
                    "token_start": offset as usize + *start,
                    "token_count": patches.token_count(),
                }));
            }
        } else {
            for axis in 0..3 {
                media_axes[axis].extend((0..prepared.input_ids.len()).map(|v| v as u32));
            }
        }
    }
    fixtures.push(json!({
        "name": "ragged-three-seq",
        "ids": ids,
        "types": vec![0u32; ids.len()],
        "positions": (0..ids.len()).collect::<Vec<_>>(),
        "media_positions": media_axes,
        "images": images,
        "cumulative": cumulative,
        "pool": "last_token",
    }));
    // Real three-frame fixture synthesized from the processed cats patch grid.
    let cats_pixels = &cats.media.images[0].1;
    let grid = cats_pixels.grid_thw;
    let mut frames = Vec::with_capacity(cats_pixels.pixels.len() * 3);
    for _ in 0..3 {
        frames.extend_from_slice(&cats_pixels.pixels);
    }
    let frame_patches = ImagePatches {
        pixels: frames,
        grid_thw: [3, grid[1], grid[2]],
        patch_dim: cats_pixels.patch_dim,
        merge_size: cats_pixels.merge_size,
    };
    let frame_pixels = write_pixels(&output, "three-frame", 0, &frame_patches)?;
    let mut frame_ids = cats.input_ids.clone();
    // Replace the single image placeholder by three copies at the same span.
    let token_count = cats_pixels.token_count();
    let start = cats
        .media
        .images
        .first()
        .map(|(start, _)| *start)
        .ok_or("cats fixture lost its image span")?;
    let image_token = frame_ids[start];
    frame_ids.splice(
        start..start + token_count,
        std::iter::repeat(image_token).take(token_count * 3),
    );
    let frame_positions = image_positions(frame_ids.len(), &[(start, &frame_patches)])?;
    fixtures.push(json!({
        "name": "three-frame-cats",
        "ids": frame_ids,
        "types": vec![0u32; frame_ids.len()],
        "positions": (0..frame_ids.len()).collect::<Vec<_>>(),
        "media_positions": frame_positions,
        "images": [{
            "name": "three-frame-cats",
            "pixels": output.join(frame_pixels).display().to_string(),
            "rows": frame_patches.pixels.len() / frame_patches.patch_dim,
            "patch_dim": frame_patches.patch_dim,
            "grid_thw": frame_patches.grid_thw,
            "merge_size": frame_patches.merge_size,
            "token_start": start,
            "token_count": frame_patches.token_count(),
        }],
        "cumulative": [0, frame_ids.len()],
        "pool": "last_token",
    }));
    let manifest = json!({
        "schema": "qwen3-vl-real-fixtures/v1",
        "checkpoint": checkpoint,
        "source": "cached Rust Qwen3VlProcessor; real local image assets; no Torch/GPU/download",
        "fixtures": fixtures,
    });
    std::fs::write(
        output.join("fixtures.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "{} fixtures written to {}",
        manifest["fixtures"].as_array().map_or(0, Vec::len),
        output.display()
    );
    Ok(())
}
