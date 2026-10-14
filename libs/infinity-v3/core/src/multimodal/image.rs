//! Qwen image preprocessing matching the pinned Transformers Torchvision processor.
use image::{DynamicImage, ImageDecoder, ImageReader};
use infinity_v3_backend::ImagePatches;
use serde::Deserialize;
use std::{io::Cursor, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct QwenImageProcessor {
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    rescale_factor: f64,
    resample: u8,
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    do_convert_rgb: bool,
    #[serde(default)]
    do_center_crop: Option<bool>,
    #[serde(default)]
    do_pad: Option<bool>,
    #[serde(skip)]
    video_frame_count: Option<usize>,
    #[serde(skip)]
    ignore_exif: bool,
}

pub(crate) struct ImagePlan {
    pub output_width: u32,
    pub output_height: u32,
    pub memory_bytes: usize,
    pub token_count: usize,
}

impl QwenImageProcessor {
    pub fn load(root: &Path) -> Result<Self, String> {
        let mut value: serde_json::Value =
            match std::fs::read(root.join("preprocessor_config.json")) {
                Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| e.to_string())?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    let bytes = std::fs::read(root.join("processor_config.json"))
                        .map_err(|e| e.to_string())?;
                    serde_json::from_slice::<serde_json::Value>(&bytes)
                        .map_err(|e| e.to_string())?
                        .get("image_processor")
                        .cloned()
                        .ok_or("Missing image_processor")?
                }
                Err(e) => return Err(e.to_string()),
            };
        if !value.is_object() {
            return Err("Image processor configuration must be an object".into());
        }
        for (name, edge) in [
            ("min_pixels", "shortest_edge"),
            ("max_pixels", "longest_edge"),
        ] {
            if value.get(name).is_none() {
                value[name] = value["size"][edge].clone();
            }
        }
        let config: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    /// Released D3 uses the Qwen defaults but saves only a subset of processor fields.
    pub(super) fn load_d3(root: &Path) -> Result<Self, String> {
        let mut value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("preprocessor_config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        for (key, default) in [
            ("rescale_factor", serde_json::json!(1.0 / 255.0)),
            ("resample", serde_json::json!(3)),
            ("do_resize", serde_json::json!(true)),
            ("do_rescale", serde_json::json!(true)),
            ("do_normalize", serde_json::json!(true)),
            ("do_convert_rgb", serde_json::json!(true)),
        ] {
            if value.get(key).is_none() {
                value[key] = default;
            }
        }
        value["min_pixels"] = serde_json::json!(65_536);
        value["max_pixels"] = serde_json::json!(1_638_400);
        let mut config: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        config.ignore_exif = true;
        config.validate()?;
        if config.patch_size != 16 || config.merge_size != 2 {
            return Err("D3 requires 16-pixel patches and spatial merge size 2".into());
        }
        Ok(config)
    }

    pub(super) fn video_frames(&self, count: usize) -> Self {
        let mut config = self.clone();
        config.min_pixels = 4096usize.div_ceil(count);
        config.video_frame_count = Some(count);
        config.max_pixels = 200_704;
        config
    }

    fn validate(&self) -> Result<(), String> {
        if self.patch_size == 0
            || self.patch_size > 64
            || self.temporal_patch_size != 2
            || self.merge_size == 0
            || self.merge_size > 8
            || self.min_pixels == 0
            || self.min_pixels > self.max_pixels
            || self.max_pixels > 16_777_216
            || self.resample != 3
            || !self.do_resize
            || !self.do_rescale
            || !self.do_normalize
            || !self.do_convert_rgb
            || self.do_center_crop == Some(true)
            || self.do_pad == Some(true)
            || !self.rescale_factor.is_finite()
            || self.rescale_factor <= 0.0
            || self.image_mean.iter().any(|v| !v.is_finite())
            || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("Unsupported Qwen image processor configuration".into());
        }
        let scale = (1.0 / self.rescale_factor) as f32;
        if self
            .image_mean
            .iter()
            .chain(&self.image_std)
            .any(|value| !(value * scale).is_finite())
            || self.image_std.iter().any(|value| value * scale <= 0.0)
        {
            return Err("Invalid image normalization scale".into());
        }
        Ok(())
    }

    pub(crate) fn resized_dimensions(&self, width: u32, height: u32) -> Result<(u32, u32), String> {
        self.validate()?;
        if width == 0
            || height == 0
            || f64::from(width.max(height)) / f64::from(width.min(height)) > 200.0
        {
            return Err("Image aspect ratio must be at most 200 with nonzero dimensions".into());
        }
        let (mut w, mut h) = (f64::from(width), f64::from(height));
        let factor = (self.patch_size * self.merge_size) as f64;
        if self.video_frame_count.is_some() && w.min(h) < factor {
            let scale = factor / w.min(h);
            w = (w * scale).floor();
            h = (h * scale).floor();
        }
        let min_pixels = self
            .video_frame_count
            .map(|count| 4096.0 / count as f64)
            .unwrap_or(self.min_pixels as f64);
        // Python round is ties-to-even; the reference does not clamp this first rounding.
        let (mut rw, mut rh) = (
            (w / factor).round_ties_even() * factor,
            (h / factor).round_ties_even() * factor,
        );
        if rw * rh > self.max_pixels as f64 {
            let scale = (w * h / self.max_pixels as f64).sqrt();
            rw = (w / scale / factor).floor().max(1.0) * factor;
            rh = (h / scale / factor).floor().max(1.0) * factor;
        } else if rw * rh < min_pixels {
            let scale = (min_pixels / (w * h)).sqrt();
            rw = (w * scale / factor).ceil() * factor;
            rh = (h * scale / factor).ceil() * factor;
        }
        if rw <= 0.0 || rh <= 0.0 || rw * rh > 16_777_216.0 {
            return Err("Resized image exceeds the supported pixel budget".into());
        }
        Ok((rw as u32, rh as u32))
    }

    pub(super) fn decoder(bytes: &[u8], max_pixels: u64) -> Result<impl ImageDecoder + '_, String> {
        let mut reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|_| "Invalid image header")?;
        if !matches!(
            reader.format(),
            Some(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)
        ) {
            return Err("Only PNG, JPEG, and WebP images are supported".into());
        }
        if reader.format() == Some(image::ImageFormat::WebP) {
            validate_webp_chunks(bytes)?;
        }
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(16_384);
        limits.max_image_height = Some(16_384);
        limits.max_alloc = Some(max_pixels.saturating_mul(16));
        reader.limits(limits);
        let decoder = reader
            .into_decoder()
            .map_err(|_| "Invalid image or decoder allocation limit exceeded")?;
        // Higher-depth conversion differs from Pillow (notably L16 clamping).
        // Reject it from decoder metadata before allocating/normalizing pixels.
        if !matches!(
            decoder.color_type(),
            image::ColorType::L8
                | image::ColorType::La8
                | image::ColorType::Rgb8
                | image::ColorType::Rgba8
        ) {
            return Err("Only images decoded to 8-bit samples are supported".into());
        }
        let (w, h) = decoder.dimensions();
        if u64::from(w) * u64::from(h) > max_pixels {
            return Err("Image exceeds the decoded pixel limit".into());
        }
        Ok(decoder)
    }

    pub(crate) fn inspect(&self, bytes: &[u8], max_pixels: u64) -> Result<ImagePlan, String> {
        let mut decoder = Self::decoder(bytes, max_pixels)?;
        let (mut width, mut height) = decoder.dimensions();
        let orientation = if self.ignore_exif {
            image::metadata::Orientation::NoTransforms
        } else {
            decoder
                .orientation()
                .map_err(|_| "Invalid image orientation")?
        };
        if matches!(
            orientation,
            image::metadata::Orientation::Rotate90
                | image::metadata::Orientation::Rotate270
                | image::metadata::Orientation::Rotate90FlipH
                | image::metadata::Orientation::Rotate270FlipH
        ) {
            std::mem::swap(&mut width, &mut height);
        }
        let (w, h) = self.resized_dimensions(width, height)?;
        // Include EXIF rotation/conversion, the horizontal resize intermediate, output,
        // patch floats plus backend upload copy, and decoder/resampler scratch. Checked before decode.
        let memory = decoder.total_bytes() * 2
            + u64::from(width) * u64::from(height) * 3
            + u64::from(w) * u64::from(height) * 3
            + u64::from(w) * u64::from(h) * 3
            + u64::from(w) * u64::from(h) * 3 * self.temporal_patch_size as u64 * 4 * 2
            + (u64::from(width) + u64::from(height) + u64::from(w) + u64::from(h)) * 128;
        Ok(ImagePlan {
            token_count: (w as usize / (self.patch_size * self.merge_size))
                * (h as usize / (self.patch_size * self.merge_size)),
            output_width: w,
            output_height: h,
            memory_bytes: usize::try_from(memory).map_err(|_| "Image allocation is too large")?,
        })
    }

    pub(crate) fn prepare(
        &self,
        bytes: &[u8],
        plan: &ImagePlan,
        max_pixels: u64,
    ) -> Result<ImagePatches, String> {
        let (w, h) = (plan.output_width, plan.output_height);
        let output = resize_rgb_with_orientation(bytes, w, h, max_pixels, !self.ignore_exif)?;
        let (p, m) = (self.patch_size, self.merge_size);
        let (gh, gw) = (h as usize / p, w as usize / p);
        let patch_dim = 3 * self.temporal_patch_size * p * p;
        let inverse_scale = (1.0 / self.rescale_factor) as f32;
        let means = self.image_mean.map(|v| v * inverse_scale);
        let stds = self.image_std.map(|v| v * inverse_scale);
        let mut pixels = Vec::with_capacity(gh * gw * patch_dim);
        for block_h in 0..gh / m {
            for block_w in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        for c in 0..3 {
                            for _ in 0..self.temporal_patch_size {
                                for ph in 0..p {
                                    for pw in 0..p {
                                        let x = (block_w * m + mw) * p + pw;
                                        let y = (block_h * m + mh) * p + ph;
                                        let value = output[(y * w as usize + x) * 3 + c] as f32;
                                        pixels.push((value - means[c]) / stds[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(ImagePatches {
            pixels,
            grid_thw: [1, gh, gw],
            patch_dim,
            merge_size: m,
        })
    }
}

pub(super) fn resize_rgb(bytes: &[u8], w: u32, h: u32, max_pixels: u64) -> Result<Vec<u8>, String> {
    resize_rgb_with_orientation(bytes, w, h, max_pixels, true)
}

fn resize_rgb_with_orientation(
    bytes: &[u8],
    w: u32,
    h: u32,
    max_pixels: u64,
    orient: bool,
) -> Result<Vec<u8>, String> {
    let mut decoder = QwenImageProcessor::decoder(bytes, max_pixels)?;
    let orientation = if orient {
        decoder
            .orientation()
            .map_err(|_| "Invalid image orientation")?
    } else {
        image::metadata::Orientation::NoTransforms
    };
    let mut decoded = DynamicImage::from_decoder(decoder).map_err(|_| "Invalid image data")?;
    decoded.apply_orientation(orientation);
    let rgb = decoded.into_rgb8();
    let src = fast_image_resize::images::Image::from_vec_u8(
        rgb.width(),
        rgb.height(),
        rgb.into_raw(),
        fast_image_resize::PixelType::U8x3,
    )
    .map_err(|_| "Invalid RGB image")?;
    let mut intermediate =
        fast_image_resize::images::Image::new(w, src.height(), fast_image_resize::PixelType::U8x3);
    let mut output =
        fast_image_resize::images::Image::new(w, h, fast_image_resize::PixelType::U8x3);
    let options = fast_image_resize::ResizeOptions::new().resize_alg(
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom),
    );
    let mut resizer = fast_image_resize::Resizer::new();
    // Uint8 rounding/clipping makes order observable. The library's default 2D
    // operation is vertical-first; Torchvision performs horizontal-first.
    resizer
        .resize(&src, &mut intermediate, &options)
        .map_err(|_| "Image resize failed")?;
    resizer
        .resize(&intermediate, &mut output, &options)
        .map_err(|_| "Image resize failed")?;
    Ok(output.into_vec())
}

// image-webp's EXIF reader does not inherit ImageReader's allocation limit.
// Check the complete container without allocating before constructing either decoder.
// Metadata copies fit inside the encoded-input copy reservation held by ResolvedImage.
fn validate_webp_chunks(bytes: &[u8]) -> Result<(), String> {
    const MAX_METADATA_BYTES: usize = 64 * 1024;
    let invalid = || "Invalid WebP chunk extents".to_string();
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return Err(invalid());
    }
    let size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if size.checked_add(8) != Some(bytes.len()) {
        return Err(invalid());
    }
    let mut offset = 12usize;
    while offset < bytes.len() {
        let header_end = offset.checked_add(8).ok_or_else(invalid)?;
        let header = bytes.get(offset..header_end).ok_or_else(invalid)?;
        let length = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let end = header_end.checked_add(length).ok_or_else(invalid)?;
        let padded_end = end.checked_add(length & 1).ok_or_else(invalid)?;
        if padded_end > bytes.len() {
            return Err(invalid());
        }
        if matches!(&header[..4], b"EXIF" | b"ICCP" | b"XMP ") && length > MAX_METADATA_BYTES {
            return Err("WebP metadata exceeds the 64 KiB limit".into());
        }
        offset = padded_end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn processor() -> QwenImageProcessor {
        serde_json::from_str(r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"min_pixels":4096,"max_pixels":1310720,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"resample":3,"do_resize":true,"do_rescale":true,"do_normalize":true,"do_convert_rgb":true}"#).unwrap()
    }
    #[test]
    fn d3_keeps_encoded_pixels_without_exif_rotation() {
        let pixels = vec![120u8; 128 * 64 * 3];
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode(&pixels, 128, 64, image::ExtendedColorType::Rgb8)
            .unwrap();
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut oriented = jpeg[..2].to_vec();
        oriented.extend_from_slice(&[0xff, 0xe1]);
        oriented.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        oriented.extend_from_slice(exif);
        oriented.extend_from_slice(&jpeg[2..]);
        let legacy = processor();
        let rotated = legacy.inspect(&oriented, 1_000_000).unwrap();
        assert_eq!((rotated.output_width, rotated.output_height), (64, 128));
        let mut d3 = legacy.clone();
        d3.ignore_exif = true;
        let plan = d3.inspect(&oriented, 1_000_000).unwrap();
        assert_eq!((plan.output_width, plan.output_height), (128, 64));
        let original = d3.inspect(&jpeg, 1_000_000).unwrap();
        assert_eq!(
            d3.prepare(&oriented, &plan, 1_000_000).unwrap().pixels,
            d3.prepare(&jpeg, &original, 1_000_000).unwrap().pixels
        );
    }

    // Little-endian float32 reference patches: Transformers 5.17.0 / Torchvision 0.26.0,
    // Qwen3-VL-Embedding-2B revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda.
    #[test]
    fn patches_match_upstream_reference() {
        let processor = processor();
        for (w, h, rw, rh, expected) in [
            (
                64,
                96,
                64,
                96,
                "718a3e81694ed48829cb277c81c025acceefd01a5bde0bb706e75148f779c1e4",
            ),
            (
                17,
                23,
                64,
                96,
                "90d04c10d68726c95188c3a93701d4ec55cce6844f9f847ae9c5fa9911423a2f",
            ),
            (
                173,
                95,
                160,
                96,
                "baeef7dbc8df44b2553b4e86a8bb75a205cc95f4086fc7ed4a24cc72ee5a3dd4",
            ),
            (
                95,
                173,
                96,
                160,
                "db66afd362c30d0e235d2440b21362b3b2d39b6b36a58a23fb2398ff8c5df6c9",
            ),
            (
                1503,
                1101,
                1312,
                960,
                "bf73763ebd3cb289b25bdb7d1b33123d26add667593ddf00e65c42213d8f7b1d",
            ),
            (
                80,
                80,
                64,
                64,
                "f10927859e2ad9a61b097e6b32b802cc6459d829dfeed9129c82282eb37accfd",
            ),
        ] {
            let source = image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    ((x * 17 + y * 3) % 256) as u8,
                    ((x * 5 + y * 29) % 256) as u8,
                    (((x ^ y) * 23) % 256) as u8,
                ])
            });
            let mut bytes = Cursor::new(Vec::new());
            source
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            let plan = processor.inspect(bytes.get_ref(), 16_777_216).unwrap();
            assert_eq!((plan.output_width, plan.output_height), (rw, rh));
            let prepared = processor
                .prepare(bytes.get_ref(), &plan, 16_777_216)
                .unwrap();
            let mut digest = Sha256::new();
            for pixel in prepared.pixels {
                digest.update(pixel.to_le_bytes());
            }
            assert_eq!(format!("{:x}", digest.finalize()), expected, "{w}x{h}");
        }
    }
    #[test]
    fn rejects_16_bit_samples_before_pixel_conversion() {
        // Pillow clamps integer grayscale values above 255, while image's generic
        // conversion scales u16 values. Do not silently accept that mismatch.
        let gray = image::ImageBuffer::from_fn(32, 32, |x, _| {
            image::Luma([if x == 0 { 255u16 } else { 32768u16 }])
        });
        let rgb = image::ImageBuffer::from_pixel(32, 32, image::Rgb([255u16, 32768, 65535]));
        for source in [
            DynamicImage::ImageLuma16(gray),
            DynamicImage::ImageRgb16(rgb),
        ] {
            let mut bytes = Cursor::new(Vec::new());
            source
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            let result = processor().inspect(bytes.get_ref(), 4096);
            assert!(
                matches!(result, Err(ref error) if error == "Only images decoded to 8-bit samples are supported")
            );
        }
    }

    #[test]
    fn webp_metadata_is_bounded_before_both_orientation_calls() {
        fn webp(chunk: &[u8], declared: u32, payload: &[u8]) -> Vec<u8> {
            let mut bytes = b"RIFF\0\0\0\0WEBP".to_vec();
            bytes.extend_from_slice(chunk);
            bytes.extend_from_slice(&declared.to_le_bytes());
            bytes.extend_from_slice(payload);
            let size = (bytes.len() - 8) as u32;
            bytes[4..8].copy_from_slice(&size.to_le_bytes());
            bytes
        }
        let forged = webp(b"EXIF", 64 * 1024 * 1024, &[]);
        let oversized = webp(b"EXIF", 65538, &vec![0; 65538]);
        let truncated_padding = webp(b"EXIF", 1, &[0]);
        for bytes in [&forged, &oversized, &truncated_padding] {
            assert!(validate_webp_chunks(bytes).is_err());
            let processor = processor();
            assert!(processor.inspect(bytes, 4096).is_err());
            let plan = ImagePlan {
                output_width: 32,
                output_height: 32,
                memory_bytes: 0,
                token_count: 1,
            };
            assert!(processor.prepare(bytes, &plan, 4096).is_err());
        }
        let padded = webp(b"EXIF", 1, &[0, 0]);
        assert!(validate_webp_chunks(&padded).is_ok());
        let mut bad_riff = padded;
        bad_riff[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(validate_webp_chunks(&bad_riff).is_err());
        let mut bytes = Cursor::new(Vec::new());
        image::RgbImage::new(32, 32)
            .write_to(&mut bytes, image::ImageFormat::WebP)
            .unwrap();
        let processor = processor();
        let plan = processor.inspect(bytes.get_ref(), 4096).unwrap();
        assert!(processor.prepare(bytes.get_ref(), &plan, 4096).is_ok());
        // A real image header followed by a forged EXIF length previously reached
        // orientation(), which allocated the declared 64 MiB before failing.
        let mut extended = webp(b"VP8X", 10, &[8, 0, 0, 0, 31, 0, 0, 31, 0, 0]);
        extended.extend_from_slice(&bytes.get_ref()[12..]);
        extended.extend_from_slice(b"EXIF");
        extended.extend_from_slice(&(64u32 * 1024 * 1024).to_le_bytes());
        let size = (extended.len() - 8) as u32;
        extended[4..8].copy_from_slice(&size.to_le_bytes());
        assert!(
            matches!(processor.inspect(&extended,4096), Err(ref error) if error == "Invalid WebP chunk extents")
        );
        assert!(
            matches!(processor.prepare(&extended,&plan,4096), Err(ref error) if error == "Invalid WebP chunk extents")
        );
    }

    #[test]
    fn extreme_aspect_ratio_and_decode_limits() {
        let processor = processor();
        assert_eq!(processor.resized_dimensions(15, 3000).unwrap(), (32, 928));
        assert_eq!(processor.resized_dimensions(3000, 15).unwrap(), (928, 32));
        assert!(processor.resized_dimensions(15, 3001).is_err());
        assert!(processor.resized_dimensions(0, 1).is_err());
        let mut bytes = Cursor::new(Vec::new());
        image::RgbImage::new(64, 64)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        assert!(processor.inspect(bytes.get_ref(), 4095).is_err());
        assert!(processor.inspect(b"not an image", 4096).is_err());
    }
}
