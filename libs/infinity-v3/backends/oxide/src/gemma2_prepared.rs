// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Gemma2 arithmetic admission and rotary-table construction.
//!
//! The exact candidate is the released H512/I2048/24-layer text backbone with
//! PLE width 512 and output width 768. Norm scales are expanded BF16 -> F32
//! with no offset. Rotary tables are built once on the device with the Native
//! FP32 inverse frequencies and BF16-rounded coefficients.
#![allow(dead_code)]

use crate::gemma2_text::{Gemma2Activation, Gemma2AttentionKind, Gemma2Config};
use crate::model::WeightSource;
use crate::{arg, launch_threads, product, start, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::path::Path;
use std::sync::Arc;

pub(crate) const GEMMA2_KERNEL_EXPORTS: [&str; 8] = crate::gemma2_math::EXPORTS;
pub(crate) const GEMMA2_MAX_POSITIONS: usize = 262_144;
pub(crate) const GEMMA2_TABLE_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 2048;
const LAYERS: usize = 24;
const PLE_WIDTH: usize = 512;
const EMBEDDING_DIM: usize = 768;
const Q_HEADS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma2PreparedMath {
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) layers: usize,
    pub(crate) ple_width: usize,
    pub(crate) embedding_dim: usize,
    pub(crate) q_heads: u32,
    pub(crate) norm_eps: f32,
    pub(crate) embedding_scale: f32,
    pub(crate) ple_scale: f32,
}

impl Gemma2PreparedMath {
    pub(crate) fn admit(config: &Gemma2Config) -> Result<Self, BackendError> {
        let text = &config.text;
        for (name, found, expected) in [
            ("hidden_size", text.hidden_size, HIDDEN),
            ("intermediate_size", text.intermediate_size, INTERMEDIATE),
            ("num_hidden_layers", text.num_hidden_layers, LAYERS),
            ("hidden_size_per_layer_input", text.ple_width, PLE_WIDTH),
            ("embedding_dim", text.embedding_dim, EMBEDDING_DIM),
            ("num_attention_heads", text.num_attention_heads, Q_HEADS),
        ] {
            if found != expected {
                return Err(start(format!(
                    "only {name} {expected} is prepared, found {found}"
                )));
            }
        }
        if text.sliding_window != 512 {
            return Err(start(format!(
                "only sliding_window 512 is prepared, found {}",
                text.sliding_window
            )));
        }
        if text.max_position_embeddings != GEMMA2_MAX_POSITIONS {
            return Err(start(format!(
                "only max_position_embeddings {GEMMA2_MAX_POSITIONS} is prepared, found {}",
                text.max_position_embeddings
            )));
        }
        if text.vocab_size == 0 {
            return Err(start("vocabulary count must be positive"));
        }
        for (index, layer) in text.layers.iter().enumerate() {
            match (layer.kind, layer.head_dim, layer.kv_heads) {
                (Gemma2AttentionKind::Sliding, 256, 2) | (Gemma2AttentionKind::Full, 512, 1) => {}
                _ => {
                    return Err(start(format!(
                        "unsupported layer {index} attention geometry {:?}/d{}/kv{}",
                        layer.kind, layer.head_dim, layer.kv_heads
                    )))
                }
            }
            if layer.window_left
                != if matches!(layer.kind, Gemma2AttentionKind::Sliding) {
                    512
                } else {
                    -1
                }
                || layer.window_right
                    != if matches!(layer.kind, Gemma2AttentionKind::Sliding) {
                        512
                    } else {
                        -1
                    }
            {
                return Err(start(format!("unsupported layer {index} mask geometry")));
            }
        }
        if text.activation != Gemma2Activation::GeluTanh {
            return Err(start(
                "only the Native gelu_pytorch_tanh Gemma2 activation is prepared",
            ));
        }
        let norm_eps = text.rms_norm_eps as f32;
        if !text.rms_norm_eps.is_finite()
            || text.rms_norm_eps <= 0.0
            || !norm_eps.is_finite()
            || norm_eps <= 0.0
        {
            return Err(start("rms_norm_eps must be finite and positive as f32"));
        }
        let embedding_scale = bf16_scalar((text.hidden_size as f32).sqrt())?;
        let ple_scale = bf16_scalar((text.hidden_size as f64).sqrt().recip() as f32)?;
        Ok(Self {
            hidden: text.hidden_size,
            intermediate: text.intermediate_size,
            layers: text.num_hidden_layers,
            ple_width: text.ple_width,
            embedding_dim: text.embedding_dim,
            q_heads: u32_size(text.num_attention_heads)?,
            norm_eps,
            embedding_scale,
            ple_scale,
        })
    }

    pub(crate) fn ple_rows(&self, tokens: usize) -> Result<usize, BackendError> {
        product(tokens, self.layers)
    }

    pub(crate) fn ple_projection_elements(&self, tokens: usize) -> Result<usize, BackendError> {
        product(product(tokens, self.layers)?, self.ple_width)
    }
}

pub(crate) fn bf16_scalar(value: f32) -> Result<f32, BackendError> {
    if !value.is_finite() {
        return Err(start("BF16 scalar must be finite"));
    }
    let rounded = half::bf16::from_f32(value).to_f32();
    if !rounded.is_finite() {
        return Err(start("BF16 scalar is not finite after rounding"));
    }
    Ok(rounded)
}

/// Expand a BF16 checkpoint norm scale to F32 with no +1 offset.
pub(crate) fn norm_values(source: &WeightSource, width: usize) -> Result<Vec<f32>, BackendError> {
    if width == 0 || source.shape() != [width].as_slice() {
        return Err(start("norm source shape disagrees with width"));
    }
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(
            "Gemma2 norm scales must be BF16 in the released checkpoint",
        ));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(width)
        .map_err(|_| start("norm BF16 staging allocation failed"))?;
    bits.resize(width, 0u16);
    source
        .read_bf16(0, &mut bits)
        .map_err(|error| start(format!("norm source read failed: {error}")))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(width)
        .map_err(|_| start("norm F32 allocation failed"))?;
    for bits in bits {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(start("norm source contains nonfinite values"));
        }
        values.push(value);
    }
    Ok(values)
}

pub(crate) fn upload_norm(
    stream: &Arc<Stream>,
    source: &WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let values = norm_values(source, width)?;
    DeviceBuffer::from_host(stream, &values).map_err(start)
}

/// Compare Rust-constructed inverse-frequency bits against Native-supplied
/// F32 words. This is a Root-side qualification hook only; the device table
/// initializer owns cos/sin, so no CPU cosine stand-in can qualify the table.
pub(crate) fn compare_inverse_words(
    rust: &[f32],
    native_words: &[u32],
) -> Result<(), BackendError> {
    if rust.len() != native_words.len() {
        return Err(start("rotary inverse word count mismatch"));
    }
    for (index, (value, expected)) in rust.iter().zip(native_words).enumerate() {
        if value.to_bits() != *expected {
            return Err(start(format!(
                "rotary inverse word {index} differs: rust={:#010x} native={expected:#010x}",
                value.to_bits()
            )));
        }
    }
    Ok(())
}

/// Parse an externally supplied Native inverse-word list. JSON arrays and
/// whitespace-separated decimal/hex u32 words are accepted. This control never
/// constructs its own expected words and therefore is not a self-comparison.
pub(crate) fn parse_external_inverse_words(text: &str) -> Result<Vec<u32>, BackendError> {
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        let words: Vec<u32> = serde_json::from_str(trimmed)
            .map_err(|error| start(format!("invalid Native inverse JSON: {error}")))?;
        return Ok(words);
    }
    let mut words = Vec::new();
    for token in trimmed.split_ascii_whitespace() {
        let word = if let Some(hex) = token
            .strip_prefix("0x")
            .or_else(|| token.strip_prefix("0X"))
        {
            u32::from_str_radix(hex, 16)
        } else {
            token.parse::<u32>()
        }
        .map_err(|_| start(format!("invalid Native inverse word {token}")))?;
        words.push(word);
    }
    Ok(words)
}

pub(crate) fn compare_inverse_words_to_external(
    theta: f64,
    dim: usize,
    native_words: &[u32],
) -> Result<usize, BackendError> {
    let rust = inverse_frequencies(theta, dim)?;
    compare_inverse_words(&rust, native_words)?;
    Ok(rust.len())
}

pub(crate) fn compare_inverse_words_file(
    theta: f64,
    dim: usize,
    path: &Path,
) -> Result<usize, BackendError> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| start(format!("native inverse word file failed: {error}")))?;
    let words = parse_external_inverse_words(&text)?;
    compare_inverse_words_to_external(theta, dim, &words)
}

pub(crate) fn inverse_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(crate) fn inverse_frequencies(theta: f64, dim: usize) -> Result<Vec<f32>, BackendError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(start("rotary theta must be finite and positive"));
    }
    let base = theta as f32;
    if !base.is_finite() || base <= 0.0 {
        return Err(start("rotary theta must be representable as positive f32"));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(start("rotary dimension must be positive and even"));
    }
    let pairs = dim / 2;
    let mut result = Vec::new();
    result.try_reserve_exact(pairs).map_err(start)?;
    for index in 0..pairs {
        let exponent = (index * 2) as f32 / dim as f32;
        let frequency = 1.0f32 / base.powf(exponent);
        if !frequency.is_finite() {
            return Err(start("rotary inverse frequency is not finite"));
        }
        result.push(frequency);
    }
    Ok(result)
}

fn table_elements(positions: usize, pairs: usize) -> Result<usize, BackendError> {
    let elements = product(positions, pairs)?;
    let bytes = product(elements, core::mem::size_of::<f32>())?;
    let total = bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(product(pairs, 4).ok()?))
        .ok_or_else(|| start("rotary table byte size overflow"))?;
    if total > GEMMA2_TABLE_BUDGET_BYTES {
        return Err(start("rotary tables exceed the 1 GiB candidate budget"));
    }
    Ok(elements)
}

pub(crate) struct Gemma2RoPe {
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) stride: u32,
    _inverse: DeviceBuffer<f32>,
    _initializer: Arc<Function>,
}

struct PendingRoPe {
    stream: Arc<Stream>,
    table: Option<Gemma2RoPe>,
}

impl Drop for PendingRoPe {
    fn drop(&mut self) {
        if self.table.is_some() && self.stream.synchronize().is_err() {
            if let Some(table) = self.table.take() {
                std::mem::forget(table);
            }
        }
    }
}

impl Gemma2RoPe {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        initializer: Arc<Function>,
        dim: usize,
        theta: f64,
        positions: usize,
    ) -> Result<Self, BackendError> {
        if positions == 0 {
            return Err(start("rotary positions must be positive"));
        }
        let pairs = dim / 2;
        let elements = table_elements(positions, pairs)?;
        let positions = u32_size(positions)?;
        let pairs = u32_size(pairs)?;
        let inverse =
            DeviceBuffer::from_host(stream, &inverse_frequencies(theta, dim)?).map_err(start)?;
        let cosine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let sine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let mut pending = PendingRoPe {
            stream: stream.clone(),
            table: Some(Self {
                cosine,
                sine,
                stride: pairs,
                _inverse: inverse,
                _initializer: initializer,
            }),
        };
        let table = pending
            .table
            .as_ref()
            .ok_or_else(|| start("missing rotary table owner"))?;
        let mut inv = table._inverse.cu_deviceptr();
        let mut cos = table.cosine.cu_deviceptr();
        let mut sin = table.sine.cu_deviceptr();
        let mut positions = positions;
        let mut pairs = pairs;
        let grid = u32_size(elements.div_ceil(256))?;
        launch_threads(
            &table._initializer,
            stream,
            (grid, 1, 1),
            256,
            0,
            &mut [
                arg(&mut inv),
                arg(&mut cos),
                arg(&mut sin),
                arg(&mut positions),
                arg(&mut pairs),
            ],
        )
        .map_err(start)?;
        stream.synchronize().map_err(start)?;
        pending
            .table
            .take()
            .ok_or_else(|| start("missing initialized rotary table"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma2_text::{Gemma2Activation, Gemma2LayerConfig, Gemma2Media, Gemma2TextConfig};

    /// Synthetic text geometry derived from the released Gemma2 fixture:
    /// H512/I2048/24 layers with the 05/11/17/23 global-width overrides.
    fn fixture_text_config() -> Gemma2TextConfig {
        let mut layers = Vec::new();
        for index in 0..24usize {
            let full = matches!(index, 5 | 11 | 17 | 23);
            let head_dim = if full { 512 } else { 256 };
            let kv_heads = if full { 1 } else { 2 };
            layers.push(Gemma2LayerConfig {
                index,
                kind: if full {
                    Gemma2AttentionKind::Full
                } else {
                    Gemma2AttentionKind::Sliding
                },
                head_dim,
                kv_heads,
                q_width: 4 * head_dim,
                kv_width: kv_heads * head_dim,
                window_left: if full { -1 } else { 512 },
                window_right: if full { -1 } else { 512 },
                rope_theta: if full { 1_000_000.0 } else { 10_000.0 },
                shared: false,
                store_kv: false,
                producer: None,
            });
        }
        Gemma2TextConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 4,
            vocab_size: 262_144,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            ple_width: 512,
            embedding_dim: 768,
            sliding_window: 512,
            rope_sliding_theta: 10_000.0,
            rope_full_theta: 1_000_000.0,
            activation: Gemma2Activation::GeluTanh,
            layers,
        }
    }

    fn fixture_config() -> Gemma2Config {
        Gemma2Config {
            text: fixture_text_config(),
            media: Gemma2Media {
                tokens: None,
                vision_layers: 0,
                audio_layers: 0,
                processor: None,
                vision_config: None,
                audio_config: None,
            },
        }
    }

    #[test]
    fn prepared_math_admits_native_mixed_width_geometry() {
        let config = fixture_config();
        let math = Gemma2PreparedMath::admit(&config).expect("prepared math");
        assert_eq!(math.hidden, 512);
        assert_eq!(math.intermediate, 2048);
        assert_eq!(math.layers, 24);
        assert_eq!(math.ple_width, 512);
        assert_eq!(math.embedding_dim, 768);
        assert_eq!(math.q_heads, 4);
        assert_eq!(math.norm_eps, 1e-6);
        assert_eq!(math.embedding_scale, 22.625);
        assert_eq!(math.ple_rows(7).unwrap(), 168);
        assert_eq!(math.ple_projection_elements(7).unwrap(), 86_016);
    }

    #[test]
    fn prepared_math_rejects_non_native_geometry() {
        let mut config = fixture_config();
        config.text.hidden_size = 768;
        let error = Gemma2PreparedMath::admit(&config).expect_err("non-native hidden size");
        assert!(error.to_string().contains("hidden_size"), "{error}");

        let mut config = fixture_config();
        config.text.layers[0].head_dim = 128;
        let error = Gemma2PreparedMath::admit(&config).expect_err("non-native head dim");
        assert!(error.to_string().contains("attention geometry"), "{error}");
    }

    #[test]
    fn native_rotary_frequencies_and_bf16_scalars_are_exact() {
        let sliding = inverse_frequencies(10_000.0, 256).expect("d256 frequencies");
        assert_eq!(sliding.len(), 128);
        assert_eq!(sliding[0], 1.0);
        let full = inverse_frequencies(1_000_000.0, 512).expect("d512 frequencies");
        assert_eq!(full.len(), 256);
        assert_eq!(full[0], 1.0);
        assert_eq!(bf16_scalar((512f32).sqrt()).unwrap(), 22.625);
        assert_eq!(
            bf16_scalar((512f64).sqrt().recip() as f32).unwrap(),
            0.044_189_453
        );
    }

    #[test]
    fn native_inverse_word_comparison_is_exact_and_rejects_substitutes() {
        let values = inverse_frequencies(10_000.0, 256).expect("d256 inverse");
        let words = inverse_words(&values);
        compare_inverse_words(&values, &words).expect("exact words");
        let mut changed = words.clone();
        changed[7] ^= 1;
        assert!(compare_inverse_words(&values, &changed).is_err());
        assert!(compare_inverse_words(&values, &words[..4]).is_err());

        // This is a constructor readiness hook, not a table oracle. The device
        // initializer remains the only source of cos/sin table values.
        assert_eq!(
            inverse_words(&[1.0f32, 0.5, 0.25]),
            vec![1.0f32.to_bits(), 0.5f32.to_bits(), 0.25f32.to_bits()]
        );
    }

    #[test]
    fn external_inverse_parser_accepts_only_supplied_words() {
        let words = parse_external_inverse_words("[1065353216, 1056964608]").unwrap();
        assert_eq!(words, vec![0x3f80_0000, 0x3f00_0000]);
        let words = parse_external_inverse_words("0x3f800000 0x3f000000").unwrap();
        assert_eq!(words, vec![0x3f80_0000, 0x3f00_0000]);
        let words = parse_external_inverse_words("1065353216 1056964608").unwrap();
        assert_eq!(words, vec![0x3f80_0000, 0x3f00_0000]);
        assert!(parse_external_inverse_words("0xzz").is_err());
        assert!(parse_external_inverse_words("not-a-word").is_err());
    }

    #[test]
    fn external_inverse_parser_positive_both_lengths_and_source_params() {
        fn check(theta: f64, dim: usize, pairs: usize) {
            let generated = inverse_frequencies(theta, dim).expect("Rust inverse for parser test");
            let words = inverse_words(&generated);
            assert_eq!(words.len(), pairs);
            let json = serde_json::to_string(&words).expect("encode parser words");
            assert_eq!(parse_external_inverse_words(&json).unwrap(), words);

            let path = std::env::temp_dir().join(format!(
                "infinity-gemma2-inverse-{}-{dim}.txt",
                std::process::id()
            ));
            std::fs::write(&path, json.as_bytes()).expect("write parser fixture");
            assert_eq!(
                compare_inverse_words_file(theta, dim, &path).expect("external parser compare"),
                pairs
            );
            let _ = std::fs::remove_file(path);
        }
        check(10_000.0, 256, 128);
        check(1_000_000.0, 512, 256);
    }

    #[test]
    #[ignore = "Requires a Native host-source inverse word file supplied locally"]
    fn native_host_source_inverse_words_consumer_control() {
        let path256 = std::env::var_os("INFINITY_OXIDE_GEMMA2_NATIVE_INVERSE256_WORDS")
            .map(std::path::PathBuf::from)
            .expect("set INFINITY_OXIDE_GEMMA2_NATIVE_INVERSE256_WORDS");
        let path512 = std::env::var_os("INFINITY_OXIDE_GEMMA2_NATIVE_INVERSE512_WORDS")
            .map(std::path::PathBuf::from)
            .expect("set INFINITY_OXIDE_GEMMA2_NATIVE_INVERSE512_WORDS");
        assert_ne!(
            path256, path512,
            "d256 and d512 word files must be independent"
        );
        assert_eq!(
            compare_inverse_words_file(10_000.0, 256, &path256).expect("Native d256 inverse words"),
            128
        );
        assert_eq!(
            compare_inverse_words_file(1_000_000.0, 512, &path512)
                .expect("Native d512 inverse words"),
            256
        );
    }
}
