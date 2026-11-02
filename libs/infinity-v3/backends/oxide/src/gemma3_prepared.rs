// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Bounded host-side preparation for the BF16 Gemma3 text device ABI in
//! `device/src/gemma3.rs`. This is not an executor and does not choose a model
//! route: it validates the already-parsed dense Gemma3 text geometry, prepares
//! the native offset scale once, and binds the three Gemma3 kernels plus the shared
//! Gemma2 gated-GELU kernel.
//!
//! The prepared recipe supports any positive hidden/intermediate width up to
//! the kernel limits, 128- or 256-wide rotary heads, and any GQA query/key
//! grouping. Gemma4, EmbeddingGemma2, and FP16 arithmetic are deliberately
//! outside this helper. The upstream Candle MIT notices for the device source
//! remain in `device/src/gemma3.rs`; no device source is copied here.

use crate::gemma3_model::Gemma3Config;
use crate::model::WeightSource;
use crate::{arg, infer, launch_threads, product, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use half::bf16;
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const MAX_OFFSET_NORM_WIDTH: usize = 8192;
/// The device Q/K shared row holds at most this many normalized values.
const MAX_QK_HEAD_DIM: usize = 256;

/// The three Gemma3 exports plus the shared Gemma2 gated-GELU export. The
/// Gemma3 ABI is defined by `device/src/gemma3.rs`.
pub(crate) const GEMMA3_KERNEL_EXPORTS: [&str; 4] = [
    "infinity_gemma3_rms_norm_bf16_f32scale",
    "infinity_gemma3_residual_norm_bf16_f32scale",
    "infinity_gemma3_qk_norm_rope_bf16",
    "infinity_gemma2_gated_gelu_bf16",
];

/// Packed gated-GELU work uses one 256-thread CTA per 256 packed elements.
/// Norm and Q/K launches choose the Native power-of-two reduction width.
pub(crate) const GEMMA3_KERNEL_THREADS: u32 = 256;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide gemma3 prepared: {error}"))
}

/// Cold geometry policy for the exact first scope. All products are checked
/// here, not on a request path, and no device query is made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma3PreparedMath {
    embedding_scale: f32,
    norm_eps: f32,
    hidden: usize,
    intermediate: usize,
    q_width: usize,
    qk_stride: usize,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    qk_head_rows: usize,
}

impl Gemma3PreparedMath {
    /// Pure CPU admission. Configuration shape is already validated by the
    /// shared Gemma3 loader; this only enforces the bounded prepared recipe.
    pub(crate) fn admit(config: &Gemma3Config) -> Result<Self, BackendError> {
        if config.num_hidden_layers == 0 || config.vocab_size == 0 {
            return Err(start("Gemma3 layer and vocabulary counts must be positive"));
        }
        if config.hidden_size == 0 || config.hidden_size > MAX_OFFSET_NORM_WIDTH {
            return Err(start(format!(
                "Gemma3 hidden width must be in 1..={MAX_OFFSET_NORM_WIDTH}"
            )));
        }
        if config.intermediate_size == 0 {
            return Err(start("Gemma3 intermediate width must be positive"));
        }
        if config.num_attention_heads == 0 || config.num_key_value_heads == 0 {
            return Err(start("Gemma3 attention head counts must be positive"));
        }
        if config.head_dim == 0
            || config.head_dim > MAX_QK_HEAD_DIM
            || !config.head_dim.is_multiple_of(2)
        {
            return Err(start(format!(
                "Gemma3 rotary head_dim must be even and in 1..={MAX_QK_HEAD_DIM}"
            )));
        }

        let eps = config.rms_norm_eps as f32;
        if !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.0
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(start("rms_norm_eps must be finite and positive as f32"));
        }
        let q_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| start("query width overflow"))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| start("key/value width overflow"))?;
        let qk_stride = q_width
            .checked_add(
                kv_width
                    .checked_mul(2)
                    .ok_or_else(|| start("packed KV width overflow"))?,
            )
            .ok_or_else(|| start("packed QKV stride overflow"))?;
        let qk_head_rows = config
            .num_attention_heads
            .checked_add(config.num_key_value_heads)
            .ok_or_else(|| start("query/key head count overflow"))?;
        let embedding_scale = bf16::from_f32((config.hidden_size as f32).sqrt()).to_f32();
        if !embedding_scale.is_finite() {
            return Err(start("embedding scale is not finite"));
        }
        Ok(Self {
            embedding_scale,
            norm_eps: eps,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            q_width,
            qk_stride,
            q_heads: u32::try_from(config.num_attention_heads)
                .map_err(|_| start("query head count exceeds u32"))?,
            kv_heads: u32::try_from(config.num_key_value_heads)
                .map_err(|_| start("KV head count exceeds u32"))?,
            head_dim: u32::try_from(config.head_dim)
                .map_err(|_| start("head dimension exceeds u32"))?,
            qk_head_rows,
        })
    }

    pub(crate) const fn embedding_scale(self) -> f32 {
        self.embedding_scale
    }

    pub(crate) const fn norm_eps(self) -> f32 {
        self.norm_eps
    }

    pub(crate) const fn q_width(self) -> usize {
        self.q_width
    }

    pub(crate) const fn qk_stride(self) -> usize {
        self.qk_stride
    }

    /// Native RMS-norm reduction width: next power of two up to 256, at least 32.
    pub(crate) fn norm_threads(&self) -> u32 {
        let mut threads = 1u32;
        while threads < self.hidden as u32 && threads < 256 {
            threads *= 2;
        }
        threads.max(32)
    }

    /// Native Q/K reduction width: next power of two up to 256, at least 32.
    pub(crate) fn qk_threads(&self) -> u32 {
        let mut threads = 32u32;
        while threads < self.head_dim && threads < 256 {
            threads *= 2;
        }
        threads
    }

    /// One CTA per normalized Q or K head row.
    pub(crate) fn qk_grid(&self, tokens: usize) -> Result<(u32, u32, u32), BackendError> {
        Ok((u32_size(product(tokens, self.qk_head_rows)?)?, 1, 1))
    }

    /// One CTA per norm row at the Native reduction width.
    pub(crate) fn norm_grid(&self, rows: usize) -> Result<(u32, u32, u32), BackendError> {
        Ok((u32_size(rows)?, 1, 1))
    }

    /// One CTA per 256 activation columns, including the final partial group.
    pub(crate) fn activation_grid(&self, rows: usize) -> Result<(u32, u32, u32), BackendError> {
        let columns = product(rows, self.intermediate)?;
        Ok((
            u32_size(columns.div_ceil(GEMMA3_KERNEL_THREADS as usize))?,
            1,
            1,
        ))
    }
}

/// Cold owner of the four exact Gemma3 functions. The shared decoder can load
/// the module once and pass the functions here, avoiding a second environment,
/// filesystem, or device-capability resolution in this helper.
pub(crate) struct Gemma3KernelSet {
    norm: Function,
    residual_norm: Function,
    qk: Function,
    gate: Function,
}

impl Gemma3KernelSet {
    pub(crate) fn from_functions(
        norm: Function,
        residual_norm: Function,
        qk: Function,
        gate: Function,
    ) -> Self {
        Self {
            norm,
            residual_norm,
            qk,
            gate,
        }
    }

    /// In-place BF16 RMS norm with an already expanded FP32 offset scale.
    ///
    /// # Safety
    /// `input`, `scale`, and `out` must be allocations in the active context
    /// covering `rows * math.hidden` BF16 elements and `math.hidden` FP32
    /// elements respectively. `out` may alias `input` for the same extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_norm(
        &self,
        math: &Gemma3PreparedMath,
        stream: &Stream,
        input: u64,
        scale: u64,
        out: u64,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        if width != math.hidden {
            return Err(infer(format!(
                "norm width must be {}, found {width}",
                math.hidden
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let grid = math.norm_grid(rows)?;
        let mut input = input;
        let mut scale = scale;
        let mut out = out;
        let mut rows = grid.0;
        let mut width = u32_size(width)?;
        let mut eps = math.norm_eps;
        launch_threads(
            &self.norm,
            stream,
            grid,
            math.norm_threads(),
            0,
            &mut [
                arg(&mut input),
                arg(&mut scale),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut eps),
            ],
        )
    }

    /// Residual BF16 norm; statistics are read from `input` only.
    ///
    /// # Safety
    /// All three BF16 buffers must cover `rows * math.hidden` elements in the
    /// active context. `out` may alias `input` or `residual` for that extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_residual_norm(
        &self,
        math: &Gemma3PreparedMath,
        stream: &Stream,
        input: u64,
        residual: u64,
        scale: u64,
        out: u64,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        if width != math.hidden {
            return Err(infer(format!(
                "residual norm width must be {}, found {width}",
                math.hidden
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let grid = math.norm_grid(rows)?;
        let mut input = input;
        let mut residual = residual;
        let mut scale = scale;
        let mut out = out;
        let mut rows = grid.0;
        let mut width = u32_size(width)?;
        let mut eps = math.norm_eps;
        launch_threads(
            &self.residual_norm,
            stream,
            grid,
            math.norm_threads(),
            0,
            &mut [
                arg(&mut input),
                arg(&mut residual),
                arg(&mut scale),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut eps),
            ],
        )
    }

    /// In-place packed Q/K norm and rotary for the configured even head width.
    /// The caller may pass `k = q + math.q_width * 2` from the original QKV
    /// allocation; both strides must equal the original packed row stride. V is
    /// never written.
    ///
    /// # Safety
    /// `q` and `k` must cover `tokens * stride` BF16 elements from their head
    /// bases, scale tables must cover the two head widths in FP32, cosine/sine
    /// must cover `positions` table rows, and positions must cover `tokens`
    /// entries in the active context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_qk(
        &self,
        math: &Gemma3PreparedMath,
        stream: &Stream,
        q: u64,
        k: u64,
        q_scale: u64,
        k_scale: u64,
        cosine: u64,
        sine: u64,
        positions: u64,
        tokens: usize,
        q_stride: usize,
        k_stride: usize,
        table_stride: usize,
    ) -> Result<(), BackendError> {
        if q_stride != math.qk_stride || k_stride != math.qk_stride {
            return Err(infer(format!(
                "Q/K strides must equal packed stride {}, found {q_stride}/{k_stride}",
                math.qk_stride
            )));
        }
        if table_stride != math.head_dim as usize / 2 {
            return Err(infer(format!(
                "rotary table stride must be {}, found {table_stride}",
                math.head_dim / 2
            )));
        }
        if tokens == 0 {
            return Ok(());
        }
        let grid = math.qk_grid(tokens)?;
        let mut q = q;
        let mut k = k;
        let mut q_scale = q_scale;
        let mut k_scale = k_scale;
        let mut cosine = cosine;
        let mut sine = sine;
        let mut positions = positions;
        let mut tokens = u32_size(tokens)?;
        let mut q_heads = math.q_heads;
        let mut kv_heads = math.kv_heads;
        let mut head_dim = math.head_dim;
        let mut q_stride = u64::try_from(q_stride).map_err(infer)?;
        let mut k_stride = u64::try_from(k_stride).map_err(infer)?;
        let mut table_stride = math.head_dim / 2;
        let mut eps = math.norm_eps;
        launch_threads(
            &self.qk,
            stream,
            grid,
            math.qk_threads(),
            0,
            &mut [
                arg(&mut q),
                arg(&mut k),
                arg(&mut q_scale),
                arg(&mut k_scale),
                arg(&mut cosine),
                arg(&mut sine),
                arg(&mut positions),
                arg(&mut tokens),
                arg(&mut q_heads),
                arg(&mut kv_heads),
                arg(&mut head_dim),
                arg(&mut q_stride),
                arg(&mut k_stride),
                arg(&mut table_stride),
                arg(&mut eps),
            ],
        )
    }

    /// Packed `[gate, up]` gated GELU activation. Input and output must not
    /// overlap because each input row is twice the output row width.
    ///
    /// # Safety
    /// `input` must cover `rows * 2 * math.intermediate` BF16 elements and
    /// `out` must cover `rows * math.intermediate` distinct elements.
    pub(crate) unsafe fn launch_gate(
        &self,
        math: &Gemma3PreparedMath,
        stream: &Stream,
        input: u64,
        out: u64,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        if width != math.intermediate {
            return Err(infer(format!(
                "activation width must be {}, found {width}",
                math.intermediate
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let grid = math.activation_grid(rows)?;
        let mut input = input;
        let mut out = out;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        launch_threads(
            &self.gate,
            stream,
            grid,
            GEMMA3_KERNEL_THREADS,
            0,
            &mut [
                arg(&mut input),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
            ],
        )
    }
}

/// Expand checkpoint BF16 target bits to FP32 and add the native offset one.
/// The addition happens only after BF16 rounding, exactly as the device ABI
/// expects; a raw FP32 source is never incremented before the BF16 cast.
pub(crate) fn offset_norm_values(
    source: &WeightSource,
    width: usize,
) -> Result<Vec<f32>, BackendError> {
    if width == 0 {
        return Err(infer("offset norm width must be positive"));
    }
    if width > MAX_OFFSET_NORM_WIDTH {
        return Err(infer(format!(
            "offset norm width {width} exceeds bounded maximum {MAX_OFFSET_NORM_WIDTH}"
        )));
    }
    if source.shape() != [width].as_slice() {
        return Err(infer(format!(
            "offset norm source must have shape [{width}], found {:?}",
            source.shape()
        )));
    }
    if source.len() != width {
        return Err(infer(format!(
            "offset norm source length {} disagrees with width {width}",
            source.len()
        )));
    }

    let mut bits = Vec::new();
    bits.try_reserve_exact(width)
        .map_err(|_| infer("offset norm BF16 staging allocation failed"))?;
    bits.resize(width, 0u16);
    source
        .read_bf16(0, &mut bits)
        .map_err(|error| infer(format!("offset norm source read failed: {error}")))?;

    let mut values = Vec::new();
    values
        .try_reserve_exact(width)
        .map_err(|_| infer("offset norm FP32 allocation failed"))?;
    for bits in bits {
        let expanded = f32::from_bits(u32::from(bits) << 16);
        let value = expanded + 1.0f32;
        if !expanded.is_finite() || !value.is_finite() {
            return Err(infer("offset norm source contains nonfinite values"));
        }
        values.push(value);
    }
    Ok(values)
}

/// Cold constructor-only upload of the native BF16-rounded offset scale. It
/// allocates no model-sized staging buffer and performs no weight-name rewrite.
pub(crate) fn upload_offset_norm(
    stream: &Arc<Stream>,
    source: &WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let values = offset_norm_values(source, width)?;
    DeviceBuffer::from_host(stream, &values).map_err(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Checkpoint;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn config() -> Gemma3Config {
        Gemma3Config {
            hidden_size: 640,
            intermediate_size: 2048,
            num_hidden_layers: 18,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 256,
            vocab_size: 262_144,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 512,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
            is_causal: true,
            global_rope_factor: 1.0,
        }
    }

    #[test]
    fn admission_keeps_independent_q_width_and_prepares_native_scale() {
        let prepared = Gemma3PreparedMath::admit(&config()).unwrap();
        assert_eq!(prepared.q_width(), 1024);
        assert_ne!(prepared.q_width(), config().hidden_size);
        assert_eq!(prepared.qk_stride(), 1536);
        assert_eq!(prepared.norm_eps().to_bits(), 1e-6f32.to_bits());
        let expected = 25.25f32;
        assert_eq!(prepared.embedding_scale().to_bits(), expected.to_bits());
        assert_eq!(prepared.qk_grid(3).unwrap(), (15, 1, 1));
        assert_eq!(prepared.norm_grid(7).unwrap(), (7, 1, 1));
        assert_eq!(prepared.activation_grid(3).unwrap(), (24, 1, 1));
        assert_eq!(prepared.activation_grid(1).unwrap(), (8, 1, 1));
        assert_eq!(prepared.norm_threads(), 256);
        assert_eq!(prepared.qk_threads(), 256);
        let mut narrow = config();
        narrow.head_dim = 128;
        assert_eq!(
            Gemma3PreparedMath::admit(&narrow).unwrap().qk_threads(),
            128
        );
        let mut tiny = config();
        tiny.head_dim = 2;
        assert_eq!(Gemma3PreparedMath::admit(&tiny).unwrap().qk_threads(), 32);
    }

    #[test]
    fn admission_rejects_scope_escapes_and_unchecked_grid_overflow() {
        let mutations: [fn(&mut Gemma3Config); 12] = [
            |c| c.hidden_size = 0,
            |c| c.hidden_size = 8193,
            |c| c.intermediate_size = 0,
            |c| c.head_dim = 0,
            |c| c.head_dim = 255,
            |c| c.head_dim = 258,
            |c| c.num_attention_heads = 0,
            |c| c.num_key_value_heads = 0,
            |c| c.num_hidden_layers = 0,
            |c| c.vocab_size = 0,
            |c| c.rms_norm_eps = 0.0,
            |c| c.rms_norm_eps = f64::NAN,
        ];
        for mutate in mutations {
            let mut config = config();
            mutate(&mut config);
            assert!(Gemma3PreparedMath::admit(&config).is_err());
        }

        let prepared = Gemma3PreparedMath::admit(&config()).unwrap();
        assert!(prepared.qk_grid(u32::MAX as usize / 5 + 1).is_err());
        assert!(prepared
            .activation_grid(usize::MAX / config().intermediate_size + 1)
            .is_err());
        assert!(prepared.norm_grid(u32::MAX as usize + 1).is_err());
    }

    struct NormFixture(PathBuf);
    impl NormFixture {
        fn new(values: &[f32]) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-gemma3-offset-norm-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(path.join("config.json"), b"{}").unwrap();
            let data = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let view = TensorView::new(Dtype::F32, vec![values.len()], &data).unwrap();
            let bytes = serialize([("norm.weight".to_string(), view)], None).unwrap();
            fs::write(path.join("model.safetensors"), bytes).unwrap();
            Self(path)
        }

        fn source(&self, width: usize) -> WeightSource {
            Checkpoint::open(&self.0)
                .unwrap()
                .source("norm.weight", &[width])
                .unwrap()
        }
    }
    impl Drop for NormFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn offset_norm_rounds_f32_to_bf16_before_adding_one() {
        let values = [
            1.0f32,
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f80_8001),
            f32::from_bits(0x3f80_7fff),
            f32::from_bits(0xbf80_8000),
            0.1f32,
        ];
        let fixture = NormFixture::new(&values);
        let actual = offset_norm_values(&fixture.source(values.len()), values.len()).unwrap();
        assert_eq!(actual.len(), values.len());
        let expected = [
            2.0f32,
            2.0,
            2.0078125,
            2.0,
            0.0,
            f32::from_bits(0x3f8c_d000),
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(actual[1].to_bits(), 2.0f32.to_bits());
        assert_eq!(actual[2].to_bits(), (1.007_812_5f32 + 1.0).to_bits());
        assert_eq!(actual[3].to_bits(), 2.0f32.to_bits());
    }

    #[test]
    fn offset_norm_rejects_wrong_shape_width_and_nonfinite_values() {
        let wrong_shape = WeightSource::zeroed(vec![8]).unwrap();
        assert!(offset_norm_values(&wrong_shape, 7).is_err());

        let too_wide = WeightSource::zeroed(vec![MAX_OFFSET_NORM_WIDTH + 1]).unwrap();
        assert!(offset_norm_values(&too_wide, MAX_OFFSET_NORM_WIDTH + 1).is_err());
        assert!(offset_norm_values(&wrong_shape, 0).is_err());

        for values in [
            [f32::NAN, 1.0],
            [f32::INFINITY, 1.0],
            [f32::NEG_INFINITY, 1.0],
        ] {
            let fixture = NormFixture::new(&values);
            assert!(offset_norm_values(&fixture.source(2), 2).is_err());
        }
    }

    #[test]
    #[ignore = "Root GPU: actual Gemma3 native gold and additive76 module; plain/mem/init"]
    fn prepared_methods_match_actual_native_gold_and_alias_boundaries() {
        use cuda_core::Context;
        use sha2::{Digest, Sha256};
        let root = PathBuf::from(std::env::var_os("INFINITY_OXIDE_GEMMA3_GOLD").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "gemma3-native-primitive-gold/v1");
        assert_eq!(manifest["status"], "passed");
        assert_eq!(
            manifest["native_sdk_sha256"],
            "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
        );
        assert_eq!(
            manifest["native_helper_sha256"],
            "8a213c56abd561a7d6329f0bd11b7148e687866d495827ba1e63d64abbcfe855"
        );
        let cases = manifest["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 50);
        // Read/hash the actual selected payloads before starting CUDA.
        let mut fixture = Vec::new();
        for case in cases {
            let operation = case["operation"].as_str().unwrap();
            let width = case["width"].as_u64().unwrap() as usize;
            if (matches!(operation, "norm" | "residual_norm") && width != 640)
                || (operation == "gate" && width != 2048)
            {
                continue;
            }
            let name = case["name"].as_str().unwrap();
            let mut buffers = std::collections::BTreeMap::new();
            for (key, info) in case["buffers"].as_object().unwrap() {
                let bytes = fs::read(root.join(name).join(info["file"].as_str().unwrap())).unwrap();
                assert_eq!(bytes.len(), info["bytes"].as_u64().unwrap() as usize);
                assert_eq!(
                    format!("{:x}", Sha256::digest(&bytes)),
                    info["sha256"].as_str().unwrap()
                );
                buffers.insert(key.clone(), bytes);
            }
            fixture.push((case.clone(), buffers));
        }
        assert_eq!(fixture.len(), 27);
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let path = PathBuf::from(std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE").unwrap());
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().unwrap();
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let module = context
            .load_module_from_image(&fs::read(path).unwrap())
            .unwrap();
        let [norm, residual_norm, qk, gate] = GEMMA3_KERNEL_EXPORTS;
        let kernels = Gemma3KernelSet::from_functions(
            module.load_function(norm).unwrap(),
            module.load_function(residual_norm).unwrap(),
            module.load_function(qk).unwrap(),
            module.load_function(gate).unwrap(),
        );
        let math = Gemma3PreparedMath::admit(&config()).unwrap();
        let guarded = |data: &[u16]| {
            let mut result = vec![0x5a5au16; 16];
            result.extend_from_slice(data);
            result.extend([0x5a5a; 16]);
            result
        };
        let pointer = |data: &DeviceBuffer<u16>| data.cu_deviceptr() + 32;
        let mut checks = 0;
        for (case, buffers) in fixture {
            let operation = case["operation"].as_str().unwrap();
            let rows = case["rows"].as_u64().unwrap() as usize;
            let width = case["width"].as_u64().unwrap() as usize;
            let read_u16 = |name: &str| {
                buffers[name]
                    .chunks_exact(2)
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            let read_f32 = |name: &str| {
                buffers[name]
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            let input = read_u16("input");
            let expected = read_u16("expected");
            let aliases: &[u32] = match operation {
                "norm" => &[0, 1],
                "residual_norm" => &[0, 1, 2],
                "qk" => &[1],
                _ => &[0],
            };
            for alias in aliases {
                let source = DeviceBuffer::from_host(&stream, &guarded(&input)).unwrap();
                let residual_values = if operation == "residual_norm" {
                    read_u16("residual")
                } else {
                    Vec::new()
                };
                let residual =
                    DeviceBuffer::from_host(&stream, &guarded(&residual_values)).unwrap();
                let output =
                    DeviceBuffer::from_host(&stream, &guarded(&vec![0x7fc1; expected.len()]))
                        .unwrap();
                let out = match alias {
                    1 => pointer(&source),
                    2 => pointer(&residual),
                    _ => pointer(&output),
                };
                match operation {
                    "norm" | "residual_norm" => {
                        let scales = read_f32("scale");
                        let original = scales.iter().map(|x| *x - 1.0).collect::<Vec<_>>();
                        let norm_source = NormFixture::new(&original);
                        let uploaded =
                            upload_offset_norm(&stream, &norm_source.source(width), width).unwrap();
                        assert_eq!(
                            uploaded.to_host_vec(&stream).unwrap(),
                            scales,
                            "cold offset-scale conversion changed native values"
                        );
                        unsafe {
                            if operation == "norm" {
                                kernels
                                    .launch_norm(
                                        &math,
                                        &stream,
                                        pointer(&source),
                                        uploaded.cu_deviceptr(),
                                        out,
                                        rows,
                                        width,
                                    )
                                    .unwrap();
                            } else {
                                kernels
                                    .launch_residual_norm(
                                        &math,
                                        &stream,
                                        pointer(&source),
                                        pointer(&residual),
                                        uploaded.cu_deviceptr(),
                                        out,
                                        rows,
                                        width,
                                    )
                                    .unwrap();
                            }
                        }
                        stream.synchronize().unwrap();
                    }
                    "qk" => {
                        let qs = DeviceBuffer::from_host(&stream, &read_f32("qscale")).unwrap();
                        let ks = DeviceBuffer::from_host(&stream, &read_f32("kscale")).unwrap();
                        let cosine = DeviceBuffer::from_host(&stream, &read_f32("cosine")).unwrap();
                        let sine = DeviceBuffer::from_host(&stream, &read_f32("sine")).unwrap();
                        let positions = buffers["positions"]
                            .chunks_exact(4)
                            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                            .collect::<Vec<_>>();
                        let positions = DeviceBuffer::from_host(&stream, &positions).unwrap();
                        unsafe {
                            kernels
                                .launch_qk(
                                    &math,
                                    &stream,
                                    pointer(&source),
                                    pointer(&source) + 2048,
                                    qs.cu_deviceptr(),
                                    ks.cu_deviceptr(),
                                    cosine.cu_deviceptr(),
                                    sine.cu_deviceptr(),
                                    positions.cu_deviceptr(),
                                    rows,
                                    1536,
                                    1536,
                                    128,
                                )
                                .unwrap();
                        }
                        stream.synchronize().unwrap();
                    }
                    "gate" => {
                        unsafe {
                            kernels
                                .launch_gate(&math, &stream, pointer(&source), out, rows, width)
                                .unwrap();
                        }
                        stream.synchronize().unwrap();
                    }
                    other => panic!("unexpected prepared operation {other}"),
                }
                let actual = match alias {
                    1 => source.to_host_vec(&stream).unwrap(),
                    2 => residual.to_host_vec(&stream).unwrap(),
                    _ => output.to_host_vec(&stream).unwrap(),
                };
                assert_eq!(&actual[..16], &[0x5a5a; 16], "prefix canary");
                assert_eq!(&actual[actual.len() - 16..], &[0x5a5a; 16], "suffix canary");
                assert_eq!(
                    &actual[16..actual.len() - 16],
                    &expected,
                    "{} alias{alias}",
                    case["name"]
                );
                if *alias != 1 {
                    assert_eq!(source.to_host_vec(&stream).unwrap(), guarded(&input));
                }
                if *alias != 2 {
                    assert_eq!(
                        residual.to_host_vec(&stream).unwrap(),
                        guarded(&residual_values)
                    );
                }
                checks += 1;
            }
        }
        assert_eq!(checks, 57);
    }
}
