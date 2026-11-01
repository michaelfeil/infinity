// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Dense encoder/decoder executors with oxide-compiled Rust kernels and BF16
//! cuBLASLt projections.
mod cache;
mod cublas;
mod decoder;
mod decoder_model;
mod dense;
mod fa4;
pub(crate) mod lt;
pub mod math;
mod model;
mod output;
mod rotary;
mod rotary_encoder;
mod rotary_encoder_model;
mod staging;
pub mod tensor;
mod upload;

use cublas::Blas;
use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Stream};
use decoder_model::{DecoderConfig, DecoderFamily, DecoderWeights};
use dense::{DenseChainHost, DenseHead, DenseWork};
use infinity_v3_backend_core::{
    result_memory, Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
    Predictions, TokenPredictions,
};
use lt::{LtPlan, LtProvider, ShapeError, NATIVE_DEFAULT_SEED_ROWS};
use model::{
    BertConfig, BertWeights, EncoderFamily, FeedForwardRecipe, LayerWeights, LinearWeights,
    NormArithmetic, PositionRecipe,
};
use rotary_encoder::{RotaryEncoder, RotaryFunctions};
use staging::{HostIo, HostIoLease, SlotPool};
use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use tensor::Bf16;
use upload::WeightUploader;

// Model warmup precedes the HTTP metrics recorder. Track physical ownership
// independently so publishing metrics later cannot lose startup allocations.
static LIVE_ARENA_BYTES: AtomicUsize = AtomicUsize::new(0);
fn record_live_arena_bytes() {
    metrics::gauge!("oxide_arena_live_bytes").set(LIVE_ARENA_BYTES.load(Ordering::Relaxed) as f64);
}

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(error.to_string())
}
fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(error.to_string())
}
fn product(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| infer("CUDA tensor size overflow"))
}
fn u32_size(value: usize) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(infer)
}

/// Private Engine classification. Rejected shape work has not submitted this
/// request's kernels and can recover after one failure-owned drain; admission
/// pressure remains non-fatal; every other error closes the Engine as before.
#[derive(Debug)]
enum EnqueueError {
    /// Pure preflight rejection before any arena reservation or request body.
    PreflightRejected(BackendError),
    /// Post-reservation shape rejection whose failure owns one stream drain.
    Rejected(BackendError),
    Fatal(BackendError),
    Admission(BackendError),
}
impl EnqueueError {
    fn classify_backend(error: BackendError) -> Self {
        match error {
            BackendError::Overloaded(_) => Self::Admission(error),
            _ => Self::Fatal(error),
        }
    }
    #[cfg(test)]
    fn poisons(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }
}
impl From<ShapeError> for EnqueueError {
    fn from(error: ShapeError) -> Self {
        match error {
            ShapeError::Rejected(error) => Self::Rejected(error),
            ShapeError::Fatal(error) => Self::classify_backend(error),
        }
    }
}
impl From<BackendError> for EnqueueError {
    fn from(error: BackendError) -> Self {
        Self::classify_backend(error)
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RequestCounters {
    pub(crate) lease_begins: usize,
    pub(crate) uploads: usize,
    pub(crate) launches: usize,
    pub(crate) engine_context_binds: usize,
    pub(crate) provider_context_queries: usize,
    pub(crate) recovery_drains: usize,
}
#[cfg(test)]
thread_local! {
    static REQUEST_COUNTERS: std::cell::Cell<RequestCounters> =
        const { std::cell::Cell::new(RequestCounters {
            lease_begins: 0,
            uploads: 0,
            launches: 0,
            engine_context_binds: 0,
            provider_context_queries: 0,
            recovery_drains: 0,
        }) };
}
#[cfg(test)]
fn reset_request_counters() {
    REQUEST_COUNTERS.with(|counters| counters.set(RequestCounters::default()));
}
#[cfg(test)]
fn bump_lease_begin() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.lease_begins += 1;
        counters.set(value);
    });
}
#[cfg(test)]
fn bump_upload() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.uploads += 1;
        counters.set(value);
    });
}
#[cfg(test)]
fn bump_launch() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.launches += 1;
        counters.set(value);
    });
}
#[cfg(test)]
fn bump_engine_context_bind() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.engine_context_binds += 1;
        counters.set(value);
    });
}
#[cfg(test)]
fn bump_recovery_drain() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.recovery_drains += 1;
        counters.set(value);
    });
}
#[cfg(test)]
pub(crate) fn bump_provider_context_query() {
    REQUEST_COUNTERS.with(|counters| {
        let mut value = counters.get();
        value.provider_context_queries += 1;
        counters.set(value);
    });
}
#[cfg(test)]
fn request_counters() -> RequestCounters {
    REQUEST_COUNTERS.with(std::cell::Cell::get)
}

pub fn device_count() -> Result<usize, BackendError> {
    // Check the shared library before upstream's lazy CUDA symbol loader.
    let _driver = unsafe { libloading::Library::new("libcuda.so.1") }.map_err(start)?;
    usize::try_from(Context::device_count().map_err(start)?).map_err(start)
}

struct Linear {
    weight: DeviceBuffer<Bf16>,
    bias: DeviceBuffer<Bf16>,
}
struct Layer {
    qkv: Linear,
    attention: Linear,
    intermediate: Linear,
    output: Linear,
    attention_gamma: DeviceBuffer<Bf16>,
    attention_beta: DeviceBuffer<Bf16>,
    output_gamma: DeviceBuffer<Bf16>,
    output_beta: DeviceBuffer<Bf16>,
}
struct Weights {
    word: DeviceBuffer<Bf16>,
    position: DeviceBuffer<Bf16>,
    token_type: DeviceBuffer<Bf16>,
    gamma: DeviceBuffer<Bf16>,
    beta: DeviceBuffer<Bf16>,
    qkv_zero_bias: DeviceBuffer<Bf16>,
    layers: Vec<Layer>,
}
impl Linear {
    fn upload(u: &mut WeightUploader, w: &LinearWeights) -> Result<Self, BackendError> {
        Ok(Self {
            weight: u.upload(&[&w.weight])?,
            bias: u.upload(&[&w.bias])?,
        })
    }
}
impl Layer {
    fn upload(u: &mut WeightUploader, w: &LayerWeights) -> Result<Self, BackendError> {
        Ok(Self {
            qkv: Linear {
                weight: u.upload(&[&w.query.weight, &w.key.weight, &w.value.weight])?,
                bias: u.upload(&[&w.query.bias, &w.key.bias, &w.value.bias])?,
            },
            attention: Linear::upload(u, &w.attention)?,
            intermediate: Linear::upload(u, &w.intermediate)?,
            output: Linear::upload(u, &w.output)?,
            attention_gamma: u.upload(&[&w.attention_gamma])?,
            attention_beta: u.upload(&[&w.attention_beta])?,
            output_gamma: u.upload(&[&w.output_gamma])?,
            output_beta: u.upload(&[&w.output_beta])?,
        })
    }
}

#[derive(Clone, Copy)]
struct KernelGeometry {
    rows_per_cta: u32,
    threads: u32,
}
const GENERIC_GEOMETRY: KernelGeometry = KernelGeometry {
    rows_per_cta: 1,
    threads: 256,
};
const H768_GEOMETRY: KernelGeometry = KernelGeometry {
    rows_per_cta: 4,
    threads: 128,
};

struct Kernels {
    embedding: Function,
    embedding_384: Option<Function>,
    norm: Function,
    norm_384: Option<Function>,
    embedding_geometry: KernelGeometry,
    norm_geometry: KernelGeometry,
    qkv: Function,
    attention: Function,
    attention_length_limit: Option<usize>,
    pool: Function,
    mean_pool: Function,
    raw_cast: Option<Function>,
    rotary: Option<RotaryFunctions>,
}
impl Kernels {
    fn load(ctx: &Arc<Context>, config: &BertConfig, fa4: bool) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = ctx.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).map_err(|e| start(format!("{}: {e}", path.display())))?;
        let module = ctx.load_module_from_image(&data).map_err(start)?;
        // Old encoder-only bundles may omit the token-output export. Preserve
        // pooled startup while reporting that capability before submitting a
        // raw request. Other driver errors still fail construction.
        let raw_cast = match module.load_function("infinity_bf16_to_f32") {
            Ok(function) => Some(function),
            Err(error) if error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_FOUND => None,
            Err(error) => return Err(start(error)),
        };
        let pool = module.load_function("infinity_pool").map_err(start)?;
        let attention = module.load_function("infinity_attention").map_err(start)?;
        let attention_length_limit = if fa4 {
            None
        } else {
            let bytes = attention.max_dynamic_shared_memory_bytes().map_err(start)? as usize;
            let limit = bytes / std::mem::size_of::<f32>();
            if limit == 0 {
                return Err(start(format!(
                    "SIMT attention kernel reports {bytes} bytes of dynamic shared memory"
                )));
            }
            Some(limit)
        };
        let exact_768_embedding =
            config.hidden_size == 768 && config.recipe.embedding == NormArithmetic::ExactBiased;
        let exact_768_norm =
            config.hidden_size == 768 && config.recipe.residual == NormArithmetic::ExactBiased;
        let rounded_1024_embedding = config.hidden_size == 1024
            && config.recipe.embedding == NormArithmetic::RoundedAddBeforeStats;
        let rounded_1024_norm = config.hidden_size == 1024
            && config.recipe.residual == NormArithmetic::RoundedAddBeforeStats;
        let aten_768_embedding =
            config.hidden_size == 768 && config.recipe.embedding == NormArithmetic::AtenEmbedding;
        let embedding_name = match config.recipe.embedding {
            NormArithmetic::ExactBiased if exact_768_embedding => {
                "infinity_embedding_norm_768_exact_vector"
            }
            NormArithmetic::RoundedAddBeforeStats if rounded_1024_embedding => {
                "infinity_embedding_norm_1024_rounded_native"
            }
            NormArithmetic::AtenEmbedding if aten_768_embedding => {
                "infinity_embedding_norm_word_768_aten"
            }
            NormArithmetic::AtenEmbedding => {
                return Err(start(
                    "AtenEmbedding requires the GTE 768 word-only embedding kernel",
                ))
            }
            NormArithmetic::ExactBiased => "infinity_embedding_norm",
            NormArithmetic::RoundedAddBeforeStats => "infinity_embedding_norm_rounded",
        };
        let norm_name = match config.recipe.residual {
            NormArithmetic::ExactBiased if exact_768_norm => "infinity_layer_norm_768_exact_vector",
            NormArithmetic::RoundedAddBeforeStats if rounded_1024_norm => {
                "infinity_layer_norm_1024_rounded_native"
            }
            NormArithmetic::ExactBiased => "infinity_layer_norm",
            NormArithmetic::RoundedAddBeforeStats => "infinity_layer_norm_rounded",
            NormArithmetic::AtenEmbedding => {
                return Err(start(
                    "AtenEmbedding is only valid for the GTE word-only embedding norm",
                ))
            }
        };
        let rotary = match config.recipe.positions {
            PositionRecipe::Absolute => None,
            PositionRecipe::Rotary(_) => Some(RotaryFunctions {
                rope_hfma: module
                    .load_function("infinity_rope_neox_hfma")
                    .map_err(start)?,
                gated_gelu: module
                    .load_function("infinity_gated_gelu_up_first")
                    .map_err(start)?,
            }),
        };
        Ok(Self {
            raw_cast,
            embedding_384: if config.hidden_size == 384
                && config.recipe.embedding == NormArithmetic::ExactBiased
            {
                Some(
                    module
                        .load_function("infinity_embedding_norm_384")
                        .map_err(start)?,
                )
            } else {
                None
            },
            norm_384: if config.hidden_size == 384
                && config.recipe.residual == NormArithmetic::ExactBiased
            {
                Some(
                    module
                        .load_function("infinity_layer_norm_384")
                        .map_err(start)?,
                )
            } else {
                None
            },
            mean_pool: module.load_function("infinity_mean_pool").map_err(start)?,
            embedding: module.load_function(embedding_name).map_err(start)?,
            norm: module.load_function(norm_name).map_err(start)?,
            embedding_geometry: if exact_768_embedding {
                H768_GEOMETRY
            } else {
                GENERIC_GEOMETRY
            },
            norm_geometry: if exact_768_norm {
                H768_GEOMETRY
            } else {
                GENERIC_GEOMETRY
            },
            qkv: module.load_function("infinity_qkv").map_err(start)?,
            attention,
            attention_length_limit,
            pool,
            rotary,
        })
    }
}
// Driver launch parameters are pointers to host scalar values copied at launch.
// The Work owner below retains every device allocation until the stream drains.
fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}
fn launch(
    k: &Function,
    s: &Stream,
    grid: (u32, u32, u32),
    shared: u32,
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    launch_threads(k, s, grid, 256, shared, args)
}
fn launch_threads(
    k: &Function,
    s: &Stream,
    grid: (u32, u32, u32),
    threads: u32,
    shared: u32,
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    #[cfg(test)]
    bump_launch();
    // SAFETY: fixed ABI signatures and validated dimensions; arenas and modules
    // stay retained. Prepared kernel geometry supplies the block width.
    unsafe {
        launch_kernel(
            k.cu_function(),
            grid,
            (threads, 1, 1),
            shared,
            s.cu_stream(),
            args,
        )
    }
    .map_err(infer)
}
fn blocks(elements: usize) -> Result<(u32, u32, u32), BackendError> {
    Ok((u32_size(elements.div_ceil(256))?, 1, 1))
}
#[allow(clippy::too_many_arguments)]
fn norm(
    k: &Kernels,
    s: &Stream,
    mut input: u64,
    mut residual: u64,
    mut bias: u64,
    gamma: &DeviceBuffer<Bf16>,
    beta: &DeviceBuffer<Bf16>,
    mut out: u64,
    mut rows: u32,
    mut width: u32,
    mut eps: f32,
) -> Result<(), BackendError> {
    let (mut gamma, mut beta) = (gamma.cu_deviceptr(), beta.cu_deviceptr());
    if let Some(kernel) = k.norm_384.as_ref().filter(|_| width == 384) {
        return launch_threads(
            kernel,
            s,
            (rows, 1, 1),
            256,
            0,
            &mut [
                arg(&mut input),
                arg(&mut residual),
                arg(&mut bias),
                arg(&mut gamma),
                arg(&mut beta),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut eps),
            ],
        );
    }
    launch_threads(
        &k.norm,
        s,
        (rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
        k.norm_geometry.threads,
        0,
        &mut [
            arg(&mut input),
            arg(&mut residual),
            arg(&mut bias),
            arg(&mut gamma),
            arg(&mut beta),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
struct PackedAttention {
    strides: [i64; 12],
    k_offset: u64,
    v_offset: u64,
}
fn packed_attention_layout(c: &BertConfig) -> Result<Option<PackedAttention>, BackendError> {
    let row = c
        .hidden_size
        .checked_mul(3)
        .ok_or_else(|| start("Packed attention stride overflow"))?;
    if !row.is_multiple_of(8) || !c.hidden_size.is_multiple_of(8) {
        return Ok(None);
    }
    let head_dim = c
        .hidden_size
        .checked_div(c.num_attention_heads)
        .ok_or_else(|| start("Packed attention head dimension overflow"))?;
    let row = i64::try_from(row).map_err(start)?;
    let width = i64::try_from(c.hidden_size).map_err(start)?;
    let head_dim = i64::try_from(head_dim).map_err(start)?;
    let k_offset = u64::try_from(
        c.hidden_size
            .checked_mul(2)
            .ok_or_else(|| start("Packed attention offset overflow"))?,
    )
    .map_err(start)?;
    let v_offset = u64::try_from(
        c.hidden_size
            .checked_mul(4)
            .ok_or_else(|| start("Packed attention offset overflow"))?,
    )
    .map_err(start)?;
    Ok(Some(PackedAttention {
        strides: [
            row, head_dim, 1, row, head_dim, 1, row, head_dim, 1, width, head_dim, 1,
        ],
        k_offset,
        v_offset,
    }))
}

enum IntermediatePlan {
    Dense(Box<LtPlan<Bf16>>),
    Gated {
        plan: Box<Option<math::LinearPlan<Bf16>>>,
        input: usize,
        width: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DownPolicy {
    NativeDefault,
    Existing,
}

fn down_policy(c: &BertConfig) -> DownPolicy {
    if c.family == EncoderFamily::Gte {
        DownPolicy::NativeDefault
    } else {
        DownPolicy::Existing
    }
}

#[derive(Clone, Copy)]
enum DownInvocation {
    M1(cublas::NativeM1Arguments),
    Lt(lt::NativeDefaultPrepared),
}

enum PreparedDown {
    NativeDefault {
        m1: Box<math::LinearPlan<Bf16>>,
        lt_native: Box<LtPlan<Bf16>>,
        invocations: Vec<Option<DownInvocation>>,
        pending: Box<lt::NativeDefaultPending>,
    },
    Existing(Box<LtPlan<Bf16>>),
}
impl PreparedDown {
    fn new(provider: &LtProvider, c: &BertConfig) -> Result<Self, BackendError> {
        Ok(match down_policy(c) {
            DownPolicy::NativeDefault => Self::NativeDefault {
                m1: Box::new(
                    math::LinearPlan::contiguous(1, c.intermediate_size, c.hidden_size)
                        .map_err(infer)?,
                ),
                lt_native: Box::new(provider.prepare_native_default_bias_shape::<Bf16>(
                    c.intermediate_size,
                    c.hidden_size,
                    NATIVE_DEFAULT_SEED_ROWS,
                )?),
                invocations: vec![None; c.num_hidden_layers],
                pending: Box::<lt::NativeDefaultPending>::default(),
            },
            DownPolicy::Existing => Self::Existing(Box::new(
                provider
                    .prepare_native_small_bias_shape::<Bf16>(c.intermediate_size, c.hidden_size)?,
            )),
        })
    }

    fn set_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        match self {
            Self::NativeDefault { .. } => Ok(()),
            Self::Existing(plan) => provider.set_logical_rows(plan, rows),
        }
    }

    fn begin_prepare(&mut self) {
        if let Self::NativeDefault {
            invocations,
            pending,
            ..
        } = self
        {
            invocations.fill(None);
            pending.clear();
        }
    }

    fn end_prepare(&mut self) {
        if let Self::NativeDefault { pending, .. } = self {
            pending.clear();
        }
    }

    /// Validate and preselect one layer's actual-M down invocation. The caller
    /// supplies the already-reserved arena buffers, so all pointers, bounds and
    /// alignments are known before any request kernel is submitted.
    ///
    /// # Safety
    /// The provider/engine CUDA context must already be bound on this thread
    /// and remain current through the later bound down launch. This method
    /// performs no context bind or `cuCtxGetCurrent` query.
    #[allow(clippy::too_many_arguments)]
    unsafe fn prepare_layer(
        &mut self,
        provider: &LtProvider,
        blas: Option<&Blas>,
        tokens: usize,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        stream: &Stream,
        width: usize,
        index: usize,
    ) -> Result<(), ShapeError> {
        let Self::NativeDefault {
            m1,
            lt_native,
            invocations,
            pending,
        } = self
        else {
            return Ok(());
        };
        let invocation = if tokens == 1 {
            let blas =
                blas.ok_or_else(|| ShapeError::Fatal(infer("GTE down M1 requires plain cuBLAS")))?;
            DownInvocation::M1(
                blas.preflight_native_m1(m1, x, weight, bias, output, stream, width)?,
            )
        } else {
            // SAFETY: the caller's bound-context contract covers this prepare.
            DownInvocation::Lt(unsafe {
                provider.prepare_native_default_bound(
                    lt_native, pending, tokens, x, weight, bias, output,
                )?
            })
        };
        *invocations.get_mut(index).ok_or_else(|| {
            ShapeError::Fatal(infer("GTE down layer index exceeds prepared slots"))
        })? = Some(invocation);
        Ok(())
    }
}

/// Launch the preselected down invocation.
///
/// # Safety
/// The caller must have bound the provider/engine CUDA context once on this
/// thread, keep every buffer alive without conflicting access until the stream
/// drains, and pass the same buffers used during `prepare_layer`.
#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_down(
    prepared: &mut PreparedDown,
    provider: &LtProvider,
    blas: Option<&Blas>,
    tokens: usize,
    index: usize,
    x: &DeviceBuffer<Bf16>,
    weight: &DeviceBuffer<Bf16>,
    bias: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
    stream: &Stream,
) -> Result<(), ShapeError> {
    match prepared {
        PreparedDown::NativeDefault {
            m1,
            lt_native,
            invocations,
            ..
        } => {
            let invocation = invocations
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    ShapeError::Fatal(infer(
                        "GTE down metadata was not prepared before the prefix",
                    ))
                })?;
            match invocation {
                DownInvocation::M1(plan) if tokens == 1 => {
                    let blas = blas.ok_or_else(|| {
                        ShapeError::Fatal(infer("GTE down M1 requires plain cuBLAS"))
                    })?;
                    // SAFETY: the caller's bound-context and retained-buffer
                    // contract covers the revalidated M1 invocation.
                    unsafe { blas.launch_native_m1(m1, plan, x, weight, bias, output, stream) }
                }
                DownInvocation::Lt(plan) if tokens != 1 => {
                    // SAFETY: the caller's bound-context and retained-buffer
                    // contract covers the preselected invocation.
                    unsafe {
                        provider.launch_native_default_bound(lt_native, plan, x, weight, output)
                    }
                }
                _ => Err(ShapeError::Fatal(infer(
                    "GTE down metadata does not match the launched M",
                ))),
            }
        }
        PreparedDown::Existing(plan) => {
            // SAFETY: the caller retains every buffer through stream drain.
            unsafe { provider.enqueue(plan, x, weight, bias, output) }.map_err(ShapeError::Fatal)
        }
    }
}

struct Plans {
    rows: Option<usize>,
    qkv: LtPlan<Bf16>,
    attention: LtPlan<Bf16>,
    intermediate: IntermediatePlan,
    output: PreparedDown,
}
impl Plans {
    fn new(provider: &LtProvider, c: &BertConfig) -> Result<Self, BackendError> {
        let h = c.hidden_size;
        let intermediate = match c.recipe.feed_forward {
            FeedForwardRecipe::DenseGelu => IntermediatePlan::Dense(Box::new(
                provider.prepare_native_small_gelu_bias_shape::<Bf16>(h, c.intermediate_size)?,
            )),
            FeedForwardRecipe::GatedGeluUpFirst => IntermediatePlan::Gated {
                plan: Box::new(None),
                input: h,
                width: product(c.intermediate_size, 2)?,
            },
        };
        Ok(Self {
            rows: None,
            qkv: provider.prepare_native_small_bias_shape::<Bf16>(h, product(h, 3)?)?,
            attention: provider.prepare_native_small_bias_shape::<Bf16>(h, h)?,
            intermediate,
            output: PreparedDown::new(provider, c)?,
        })
    }

    fn set_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        math::refresh_plan_rows(&mut self.rows, rows, || {
            provider.set_logical_rows(&mut self.qkv, rows)?;
            provider.set_logical_rows(&mut self.attention, rows)?;
            match &mut self.intermediate {
                IntermediatePlan::Dense(plan) => provider.set_logical_rows(plan, rows)?,
                IntermediatePlan::Gated { plan, input, width } => {
                    **plan = Some(
                        math::LinearPlan::contiguous(rows, *input, *width)
                            .map_err(|error| ShapeError::Fatal(infer(error)))?,
                    );
                }
            }
            self.output.set_rows(provider, rows)?;
            Ok(())
        })
    }

    /// Preselect every layer's down invocation after the request arenas are
    /// reserved. This runs before `lease.begin()`, H2D upload, or any kernel.
    ///
    /// # Safety
    /// The provider/engine CUDA context must already be bound on this thread.
    #[allow(clippy::too_many_arguments)]
    unsafe fn prepare(
        &mut self,
        provider: &LtProvider,
        blas: Option<&Blas>,
        tokens: usize,
        layers: &[Layer],
        x: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        stream: &Stream,
        width: usize,
    ) -> Result<(), ShapeError> {
        self.output.begin_prepare();
        let result = (|| {
            for (index, layer) in layers.iter().enumerate() {
                // SAFETY: the caller's bound-context contract covers every
                // layer preselection.
                unsafe {
                    self.output.prepare_layer(
                        provider,
                        blas,
                        tokens,
                        x,
                        &layer.output.weight,
                        &layer.output.bias,
                        output,
                        stream,
                        width,
                        index,
                    )?;
                }
            }
            Ok(())
        })();
        self.output.end_prepare();
        result
    }
}
#[cfg(test)]
impl IntermediatePlan {
    fn logical_rows(&self) -> i32 {
        match self {
            Self::Dense(plan) => plan.logical_rows(),
            Self::Gated { plan, .. } => plan
                .as_ref()
                .as_ref()
                .map_or(0, |plan| plan.arguments(0, 0, 0).rows),
        }
    }
}

/// Concrete executable selected once at load. Encoder and decoder share the
/// Engine's context, ordered stream, provider, host I/O pool and admission.
enum ExecutableModel {
    Encoder(Box<EncoderModel>),
    Decoder(Box<decoder::DecoderModel>),
}
impl ExecutableModel {
    fn supports_raw_output(&self) -> bool {
        match self {
            Self::Encoder(model) => model.kernels.raw_cast.is_some(),
            Self::Decoder(_) => true,
        }
    }
    fn preflight(
        &mut self,
        provider: &LtProvider,
        batch: &Batch,
        pool: &Pool,
    ) -> Result<(), ShapeError> {
        match self {
            Self::Encoder(model) => model.plans.set_rows(provider, batch.input_ids.len()),
            Self::Decoder(model) => model.preflight(provider, batch, pool),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Encoder(model) => model.config.hidden_size,
            Self::Decoder(model) => model.config.hidden_size,
        }
    }
    /// # Safety
    /// The provider/engine CUDA context must already be bound on this thread.
    unsafe fn prepare(
        &mut self,
        provider: &LtProvider,
        blas: Option<&Blas>,
        tokens: usize,
        workspace: &mut Workspace,
    ) -> Result<(), ShapeError> {
        match (self, workspace) {
            (Self::Encoder(model), Workspace::Encoder(work)) => {
                let width = model.config.hidden_size;
                // SAFETY: the caller's bound-context contract covers preselection.
                unsafe {
                    model.plans.prepare(
                        provider,
                        blas,
                        tokens,
                        &model.weights.layers,
                        &work.intermediate,
                        &mut work.temp,
                        &work.stream,
                        width,
                    )
                }
            }
            _ => Ok(()),
        }
    }
    fn workspace_bytes(&self, key: WorkKey) -> Result<usize, BackendError> {
        match self {
            Self::Encoder(model) => Work::allocation_bytes(key, &model.config),
            Self::Decoder(model) => decoder::DecoderWork::allocation_bytes(key, &model.config),
        }
    }
    fn workspace(&self, stream: Arc<Stream>, key: WorkKey) -> Result<Workspace, BackendError> {
        match self {
            Self::Encoder(model) => Ok(Workspace::Encoder(Work::new(stream, &model.config, key)?)),
            Self::Decoder(model) => Ok(Workspace::Decoder(decoder::DecoderWork::new(
                stream,
                &model.config,
                key,
            )?)),
        }
    }
}

struct EncoderModel {
    config: BertConfig,
    plans: Plans,
    kernels: Kernels,
    weights: Weights,
    rotary: Option<RotaryEncoder>,
    fa4: Option<fa4::Fa4>,
    fa4_layout: Option<PackedAttention>,
}

enum Workspace {
    Encoder(Work),
    Decoder(decoder::DecoderWork),
}
impl Workspace {
    fn key(&self) -> WorkKey {
        match self {
            Self::Encoder(work) => work.key(),
            Self::Decoder(work) => work.key(),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Self::Encoder(work) => work.bytes,
            Self::Decoder(work) => work.bytes(),
        }
    }
    fn fits(&self, key: WorkKey) -> bool {
        match self {
            Self::Encoder(work) => work.fits(key),
            Self::Decoder(work) => work.fits(key),
        }
    }
    fn stage(
        &self,
        batch: &Batch,
        geometry: output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        match self {
            Self::Encoder(work) => work.stage(batch, geometry, lease),
            Self::Decoder(work) => work.stage(batch, geometry, lease),
        }
    }
    fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        match self {
            Self::Encoder(work) => work.upload(host, elements),
            Self::Decoder(work) => work.upload(host, elements),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        dense: Option<(&mut DenseHead, &mut DenseWork)>,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        pooled_rows: usize,
        raw_rows: usize,
        backbone_width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        match self {
            Self::Encoder(work) => work.finish(
                dense,
                lt,
                plain_blas,
                pooled_rows,
                raw_rows,
                backbone_width,
                lease,
            ),
            Self::Decoder(work) => work.finish(
                dense,
                lt,
                plain_blas,
                pooled_rows,
                raw_rows,
                backbone_width,
                lease,
            ),
        }
    }
}

struct Engine {
    lt: ManuallyDrop<LtProvider>,
    model: ManuallyDrop<ExecutableModel>,
    dense: ManuallyDrop<Option<DenseHead>>,
    plain_blas: ManuallyDrop<Option<Blas>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    io: Arc<SlotPool<HostIo>>,
    workspace: Option<Workspace>,
    dense_workspace: Option<DenseWork>,
    cached: cache::WorkspaceCache<WorkKey, Workspace>,
    failed: bool,
}
impl Drop for Engine {
    fn drop(&mut self) {
        let complete = self.stream.synchronize().is_ok();
        self.workspace.take();
        drop(self.cached.drain());
        if complete {
            if let Some(mut dense) = self.dense_workspace.take() {
                // SAFETY: the drain above proved every queued Dense use done.
                unsafe { dense.drop_after_drain() };
            }
            // SAFETY: Engine is dropped once, and all GPU uses have finished.
            // If drain fails, retaining these owners also retains their CUDA
            // context/modules/libraries while completion is uncertain.
            unsafe {
                ManuallyDrop::drop(&mut self.model);
                ManuallyDrop::drop(&mut self.dense);
                ManuallyDrop::drop(&mut self.plain_blas);
                ManuallyDrop::drop(&mut self.lt);
            }
        } else if let Some(dense) = self.dense_workspace.take() {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(dense.bytes() as u64);
            std::mem::forget(dense);
        }
    }
}
/// Stable device arenas; pinned host staging lives in `staging`.
/// Raw launches retain all allocations.
type WorkKey = (usize, usize, usize);
const MIN_TOKEN_CAPACITY: usize = 32;
fn bucketed(value: usize, minimum: usize) -> Result<usize, BackendError> {
    value
        .max(minimum)
        .checked_next_power_of_two()
        .ok_or_else(|| infer("Batch arena capacity overflow"))
}
fn work_key(b: &Batch) -> Result<WorkKey, BackendError> {
    Ok((
        bucketed(b.input_ids.len(), MIN_TOKEN_CAPACITY)?,
        bucketed(b.cumulative_seq_lengths.len(), 2)?,
        bucketed(output::rows(b)?, 1)?,
    ))
}
/// Dense scratch follows pooled selection capacity only. A raw-only request
/// (`pooled_rows == 0`) retains any existing allocation without executing or
/// growing it, and pooled growth is independent of the backbone key.
fn plan_dense_workspace(
    pooled_rows: usize,
    existing: Option<(usize, usize)>,
    bytes_for: impl FnOnce(usize) -> Result<usize, BackendError>,
) -> Result<(Option<usize>, usize), BackendError> {
    match existing {
        Some((capacity, bytes)) if pooled_rows <= capacity => Ok((None, bytes)),
        _ if pooled_rows == 0 => Ok((None, 0)),
        _ => {
            let capacity = bucketed(pooled_rows, 1)?;
            Ok((Some(capacity), bytes_for(capacity)?))
        }
    }
}
fn capacity_fits(capacity: WorkKey, request: WorkKey, bytes: usize) -> bool {
    capacity.0 >= request.0 && capacity.1 >= request.1 && capacity.2 >= request.2
        // Release an oversized token arena when traffic returns to a smaller
        // token class. Changing only sequence/output counts can reuse its
        // bounded metadata without reallocating the much larger GPU scratch.
        && (bytes <= cache::BYTE_BUDGET || capacity.0 == request.0)
}
fn needs_exclusive_staging(workspace_bytes: usize, pinned_bytes: usize) -> bool {
    // An oversized GPU arena is still shared by ordered submissions. Only
    // staging must stay bounded independently while that arena is active.
    pinned_bytes > cache::BYTE_BUDGET
        || (workspace_bytes <= cache::BYTE_BUDGET
            && workspace_bytes.saturating_add(pinned_bytes) > cache::BYTE_BUDGET)
}
fn input_elements(
    tokens: usize,
    boundaries: usize,
    selected: usize,
) -> Result<usize, BackendError> {
    product(tokens, 3)?
        .checked_add(boundaries)
        .and_then(|n| n.checked_add(selected))
        .ok_or_else(|| infer("Input staging layout overflow"))
}
fn pack_input(
    out: &mut [u32],
    tokens: usize,
    input_ids: &[u32],
    position_ids: &[u32],
    token_type_ids: &[u32],
    boundaries: &[u32],
    selected: &[u32],
) -> Result<(), BackendError> {
    let layout = product(tokens, 3)?
        .checked_add(boundaries.len())
        .and_then(|n| n.checked_add(selected.len()))
        .ok_or_else(|| infer("Input staging layout overflow"))?;
    if out.len() != layout
        || input_ids.len() > tokens
        || position_ids.len() > tokens
        || token_type_ids.len() > tokens
    {
        return Err(infer("Invalid input staging layout"));
    }
    let mut rest = out;
    for source in [input_ids, position_ids, token_type_ids] {
        let (row, tail) = rest
            .split_at_mut_checked(tokens)
            .ok_or_else(|| infer("Invalid input staging layout"))?;
        let (head, padding) = row
            .split_at_mut_checked(source.len())
            .ok_or_else(|| infer("Invalid input staging layout"))?;
        head.copy_from_slice(source);
        padding.fill(0);
        rest = tail;
    }
    let (boundary_rows, selection_rows) = rest
        .split_at_mut_checked(boundaries.len())
        .ok_or_else(|| infer("Invalid input staging layout"))?;
    if selection_rows.len() != selected.len() {
        return Err(infer("Invalid input staging layout"));
    }
    boundary_rows.copy_from_slice(boundaries);
    selection_rows.copy_from_slice(selected);
    Ok(())
}

/// Pack the request metadata into one bounded pinned slot. Both executors use
/// the same word/position/type/boundary/selection layout, differing only in how
/// the model consumes it.
fn stage_batch(
    b: &Batch,
    geometry: output::OutputGeometry,
    lease: &mut HostIoLease,
) -> Result<(usize, usize), BackendError> {
    let rows = output::rows(b)?;
    let elements = input_elements(b.input_ids.len(), b.cumulative_seq_lengths.len(), rows)?;
    // The backbone keeps every selected row at its native width, while the
    // host payload uses the narrower/wider Dense width for the pooled prefix.
    let backbone = geometry.backbone_elements(b)?;
    let output = geometry.host_elements(b)?;
    u32_size(backbone)?;
    u32_size(output)?;
    let host = lease
        .slot_mut()?
        .input
        .as_mut_slice()
        .get_mut(..elements)
        .ok_or_else(|| infer("Input exceeds retained host I/O arena"))?;
    let pooled_metadata = input_elements(
        b.input_ids.len(),
        b.cumulative_seq_lengths.len(),
        b.pooled_indices.len(),
    )?;
    let (pooled_input, raw_input) = host
        .split_at_mut_checked(pooled_metadata)
        .ok_or_else(|| infer("Invalid raw selection staging layout"))?;
    pack_input(
        pooled_input,
        b.input_ids.len(),
        &b.input_ids,
        &b.position_ids,
        &b.token_type_ids,
        &b.cumulative_seq_lengths,
        &b.pooled_indices,
    )?;
    output::write_raw_indices(b, raw_input)?;
    if output > lease.slot()?.output.len() {
        return Err(infer("Output exceeds retained host I/O arena"));
    }
    Ok((elements, output))
}

fn upload_input(
    stream: &Stream,
    buffer: &DeviceBuffer<u32>,
    host: &[u32],
    elements: usize,
) -> Result<(), BackendError> {
    if elements > host.len() {
        return Err(infer("Input exceeds host I/O arena"));
    }
    #[cfg(test)]
    bump_upload();
    // SAFETY: pinned staging and destination are retained by the workspace;
    // no host edits occur before the lease drains, including error paths.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buffer.cu_deviceptr(),
            host.as_ptr(),
            product(elements, 4)?,
            stream.cu_stream(),
        )
    }
    .map_err(infer)
}

/// Copy one checked subrange to a checked destination inside the pinned output.
/// Mixed Dense requests chain the projected prefix and raw suffix on the same
/// ordered stream before recording a single completion event.
fn download_output_range(
    stream: &Stream,
    buffer: &DeviceBuffer<f32>,
    source_offset: usize,
    host: &mut [f32],
    host_offset: usize,
    elements: usize,
) -> Result<(), BackendError> {
    source_offset
        .checked_add(elements)
        .filter(|end| *end <= buffer.len())
        .ok_or_else(|| infer("Output source range exceeds retained arena"))?;
    let destination = host_offset
        .checked_add(elements)
        .ok_or_else(|| infer("Output host range overflow"))?;
    let slice = host
        .get_mut(host_offset..destination)
        .ok_or_else(|| infer("Output exceeds retained host I/O arena"))?;
    let bytes = product(elements, std::mem::size_of::<f32>())?;
    let offset = product(source_offset, std::mem::size_of::<f32>())?;
    let address = u64::try_from(offset)
        .ok()
        .and_then(|offset| buffer.cu_deviceptr().checked_add(offset))
        .ok_or_else(|| infer("Output source address overflow"))?;
    // SAFETY: the source range was checked against the retained device buffer
    // and the destination slice is borrowed from the lease's pinned output.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(slice.as_mut_ptr(), address, bytes, stream.cu_stream())
    }
    .map_err(infer)
}

/// Emit only the requested token rows after the pooled prefix. Both selection
/// tables and output buffers belong to the request's retained arena/lease.
fn enqueue_raw_output(
    function: &Function,
    stream: &Stream,
    input: u64,
    output: u64,
    selected: u64,
    batch: &Batch,
    hidden: usize,
) -> Result<(), BackendError> {
    let mut count = u32_size(output::raw_rows(batch)?)?;
    let mut width = u32_size(hidden)?;
    let mut input = input;
    let mut output = output
        .checked_add(
            u64::try_from(product(product(batch.pooled_indices.len(), hidden)?, 4)?)
                .map_err(infer)?,
        )
        .ok_or_else(|| infer("Raw output address overflow"))?;
    let mut selected = selected
        .checked_add(u64::try_from(product(batch.pooled_indices.len(), 4)?).map_err(infer)?)
        .ok_or_else(|| infer("Raw selection address overflow"))?;
    launch(
        function,
        stream,
        blocks(product(count as usize, hidden)?)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut output),
            arg(&mut selected),
            arg(&mut count),
            arg(&mut width),
        ],
    )
}

struct Work {
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    bytes: usize,
    input: ManuallyDrop<DeviceBuffer<u32>>,
    x: ManuallyDrop<DeviceBuffer<Bf16>>,
    temp: ManuallyDrop<DeviceBuffer<Bf16>>,
    qkv: ManuallyDrop<DeviceBuffer<Bf16>>,
    query: ManuallyDrop<DeviceBuffer<Bf16>>,
    key: ManuallyDrop<DeviceBuffer<Bf16>>,
    value: ManuallyDrop<DeviceBuffer<Bf16>>,
    attention: ManuallyDrop<DeviceBuffer<Bf16>>,
    intermediate: ManuallyDrop<DeviceBuffer<Bf16>>,
    gated: ManuallyDrop<Option<DeviceBuffer<Bf16>>>,
    pooled: ManuallyDrop<DeviceBuffer<f32>>,
}
#[cfg(test)]
struct LinearPlans {
    qkv: math::LinearPlan<Bf16>,
    attention: math::LinearPlan<Bf16>,
    intermediate: math::LinearPlan<Bf16>,
    output: math::LinearPlan<Bf16>,
}
#[cfg(test)]
impl LinearPlans {
    fn new(tokens: usize, c: &BertConfig) -> Result<Self, BackendError> {
        let h = c.hidden_size;
        Ok(Self {
            qkv: math::LinearPlan::contiguous(tokens, h, product(h, 3)?).map_err(infer)?,
            attention: math::LinearPlan::contiguous(tokens, h, h).map_err(infer)?,
            intermediate: math::LinearPlan::contiguous(tokens, h, c.intermediate_size)
                .map_err(infer)?,
            output: math::LinearPlan::contiguous(tokens, c.intermediate_size, h).map_err(infer)?,
        })
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: every field is dropped once after completion. Failed
            // drains deliberately retain device allocations.
            unsafe {
                ManuallyDrop::drop(&mut self.input);
                ManuallyDrop::drop(&mut self.x);
                ManuallyDrop::drop(&mut self.temp);
                ManuallyDrop::drop(&mut self.qkv);
                ManuallyDrop::drop(&mut self.query);
                ManuallyDrop::drop(&mut self.key);
                ManuallyDrop::drop(&mut self.value);
                ManuallyDrop::drop(&mut self.attention);
                ManuallyDrop::drop(&mut self.intermediate);
                ManuallyDrop::drop(&mut self.gated);
                ManuallyDrop::drop(&mut self.pooled);
            }
            LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            record_live_arena_bytes();
        } else {
            // Live bytes includes retained allocations when completion is
            // uncertain; expose that exceptional subset separately.
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}
impl Work {
    fn key(&self) -> WorkKey {
        (self.tokens, self.sequences, self.selections)
    }
    fn allocation_bytes(key: WorkKey, c: &BertConfig) -> Result<usize, BackendError> {
        let (tokens, _, _) = key;
        i32::try_from(tokens).map_err(infer)?;
        u32_size(product(tokens, c.hidden_size)?)?;
        u32_size(product(tokens, c.intermediate_size)?)?;
        // Device-only input/output and nine BF16 hidden-sized buffers
        // (including the packed QKV projection) plus the intermediate tensor.
        // The two pinned host slots are charged separately by `staging`.
        let input = staging::input_capacity(key)?;
        let output = product(key.2, c.hidden_size)?;
        let hidden = product(product(tokens, c.hidden_size)?, 18)?;
        let mut intermediate = product(product(tokens, c.intermediate_size)?, 2)?;
        if c.recipe.feed_forward == FeedForwardRecipe::GatedGeluUpFirst {
            let gated_elements = product(tokens, product(c.intermediate_size, 2)?)?;
            u32_size(gated_elements)?;
            intermediate = intermediate
                .checked_add(product(gated_elements, 2)?)
                .ok_or_else(|| infer("Workspace arena overflow"))?;
        }
        product(input, 4)?
            .checked_add(product(output, 4)?)
            .and_then(|n| n.checked_add(hidden))
            .and_then(|n| n.checked_add(intermediate))
            .ok_or_else(|| infer("Workspace arena overflow"))
    }
    fn fits(&self, key: WorkKey) -> bool {
        capacity_fits(self.key(), key, self.bytes)
    }
    fn new(s: Arc<Stream>, c: &BertConfig, key: WorkKey) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, c)?;
        let (tokens, sequences, selections) = key;
        let hidden = product(tokens, c.hidden_size)?;
        let input_len = staging::input_capacity(key)?;
        let output = product(selections, c.hidden_size)?;
        let alloc = |len| {
            // Initialize scratch/padding once, without a host drain per buffer.
            // SAFETY: private allocation; initialization precedes every use on
            // this stream, including vectorized library reads of scratch.
            let buffer =
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&s, len) }.map_err(infer)?;
            unsafe {
                cuda_core::memory::memset_d8_async(
                    buffer.cu_deviceptr(),
                    0,
                    buffer.num_bytes(),
                    s.cu_stream(),
                )
            }
            .map_err(infer)?;
            Ok::<_, BackendError>(buffer)
        };
        // Stage fallible construction in ordinary owners before transferring
        // to ManuallyDrop, so allocation failures still clean up prior owners.
        let input =
            unsafe { DeviceBuffer::<u32>::uninitialized_async(&s, input_len) }.map_err(infer)?;
        let x = alloc(hidden)?;
        let temp = alloc(hidden)?;
        let qkv = alloc(product(hidden, 3)?)?;
        let query = alloc(hidden)?;
        let key_buffer = alloc(hidden)?;
        let value = alloc(hidden)?;
        let attention = alloc(hidden)?;
        let intermediate = alloc(product(tokens, c.intermediate_size)?)?;
        let gated = match c.recipe.feed_forward {
            FeedForwardRecipe::DenseGelu => None,
            FeedForwardRecipe::GatedGeluUpFirst => {
                Some(alloc(product(product(tokens, c.intermediate_size)?, 2)?)?)
            }
        };
        // SAFETY: the pooling kernel writes every element before D2H.
        let pooled =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&s, output) }.map_err(infer)?;
        metrics::counter!("oxide_arena_allocations_total").increment(1);
        LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        record_live_arena_bytes();
        Ok(Self {
            tokens,
            sequences,
            selections,
            bytes,
            input: ManuallyDrop::new(input),
            x: ManuallyDrop::new(x),
            temp: ManuallyDrop::new(temp),
            qkv: ManuallyDrop::new(qkv),
            query: ManuallyDrop::new(query),
            key: ManuallyDrop::new(key_buffer),
            value: ManuallyDrop::new(value),
            attention: ManuallyDrop::new(attention),
            intermediate: ManuallyDrop::new(intermediate),
            gated: ManuallyDrop::new(gated),
            pooled: ManuallyDrop::new(pooled),
            stream: s,
        })
    }
    fn stage(
        &self,
        b: &Batch,
        geometry: output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        if geometry.backbone_elements(b)? > self.pooled.len() {
            return Err(infer("Backbone output exceeds retained arena"));
        }
        stage_batch(b, geometry, lease)
    }
    fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        upload_input(&self.stream, &self.input, host, elements)
    }
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        dense: Option<(&mut DenseHead, &mut DenseWork)>,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        pooled_rows: usize,
        raw_rows: usize,
        backbone_width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        finish_pooled(
            dense,
            lt,
            plain_blas,
            &self.stream,
            &self.pooled,
            pooled_rows,
            raw_rows,
            backbone_width,
            lease,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_pooled(
    dense: Option<(&mut DenseHead, &mut DenseWork)>,
    lt: &LtProvider,
    plain_blas: Option<&Blas>,
    stream: &Stream,
    source: &DeviceBuffer<f32>,
    pooled_rows: usize,
    raw_rows: usize,
    backbone_width: usize,
    lease: &mut HostIoLease,
) -> Result<(), BackendError> {
    let pooled_width = dense
        .as_ref()
        .map_or(backbone_width, |(head, _)| head.output_width());
    let geometry = output::OutputGeometry {
        pooled_width,
        raw_width: backbone_width,
    };
    let raw_elements = product(raw_rows, geometry.raw_width)?;
    let raw_offset = geometry.raw_source_offset(pooled_rows)?;
    let raw_host_offset = geometry.raw_host_offset(pooled_rows)?;
    let host = lease.slot_mut()?.output.as_mut_slice();
    match dense {
        Some((head, work)) if pooled_rows > 0 => {
            if !work.fits(pooled_rows) {
                return Err(infer("Dense request exceeds retained arena capacity"));
            }
            head.apply(lt, plain_blas, stream, source, pooled_rows, work)?;
            let pooled_elements = product(pooled_rows, head.output_width())?;
            download_output_range(stream, work.output(), 0, host, 0, pooled_elements)?;
            if raw_rows > 0 {
                download_output_range(
                    stream,
                    source,
                    raw_offset,
                    host,
                    raw_host_offset,
                    raw_elements,
                )?;
            }
        }
        _ => {
            // Raw-only Dense requests execute no Dense work and keep the raw
            // rows native. Without Dense, D == H so the combined prefix and
            // suffix remain one contiguous transfer as before.
            let total = product(
                pooled_rows
                    .checked_add(raw_rows)
                    .ok_or_else(|| infer("Output row count overflow"))?,
                backbone_width,
            )?;
            download_output_range(stream, source, 0, host, 0, total)?;
        }
    }
    lease.record()
}
impl Engine {
    fn retain_uncertain_workspaces(&mut self) {
        if let Some(work) = self.workspace.take() {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(work.bytes() as u64);
            std::mem::forget(work);
        }
        if let Some(dense) = self.dense_workspace.take() {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(dense.bytes() as u64);
            std::mem::forget(dense);
        }
    }
    fn fail_closed(&mut self) {
        self.failed = true;
        // Close before the fallback drain so blocked acquirers wake even while
        // this thread waits for an uncertain stream.
        self.io.close();
        if self.stream.synchronize().is_err() {
            self.retain_uncertain_workspaces();
        }
    }
    fn enqueue(
        &mut self,
        b: &Batch,
        pool: &Pool,
        max_length: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        if self.failed || self.io.poisoned() {
            return Err(BackendError::Unhealthy);
        }
        #[cfg(test)]
        reset_request_counters();
        match self.enqueue_request(b, pool, max_length, lease) {
            Ok(()) => Ok(()),
            Err(EnqueueError::PreflightRejected(error)) => Err(error),
            Err(EnqueueError::Admission(error)) => Err(error),
            Err(EnqueueError::Rejected(error)) => {
                // The request submitted no body kernels. One failure-owned
                // drain covers arena initialization queued by reservation and
                // earlier held Deferred work; success preserves Engine/Io.
                #[cfg(test)]
                bump_recovery_drain();
                if self.stream.synchronize().is_err() {
                    self.failed = true;
                    self.io.close();
                    self.retain_uncertain_workspaces();
                    return Err(BackendError::Unhealthy);
                }
                Err(error)
            }
            Err(EnqueueError::Fatal(error)) => {
                self.fail_closed();
                Err(error)
            }
        }
    }
    fn enqueue_request(
        &mut self,
        b: &Batch,
        pool: &Pool,
        max_length: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), EnqueueError> {
        // Descriptor-only preflight precedes arena allocation/zeroing. The
        // actual-M down metadata is prepared later, after the reserved arena
        // exposes every scratch/weight/bias pointer but before lease.begin.
        let preflight =
            self.model
                .preflight(&self.lt, b, pool)
                .and_then(|()| match (*self.dense).as_mut() {
                    Some(dense) => dense.preflight(&self.lt, b.pooled_indices.len()),
                    None => Ok(()),
                });
        match preflight {
            Ok(()) => {}
            Err(ShapeError::Rejected(error)) => return Err(EnqueueError::PreflightRejected(error)),
            Err(ShapeError::Fatal(error)) => return Err(EnqueueError::classify_backend(error)),
        }
        #[cfg(test)]
        bump_engine_context_bind();
        self.context.bind_to_thread().map_err(infer)?;
        self.enqueue_inner(b, pool, max_length, lease)
    }
    fn enqueue_inner(
        &mut self,
        b: &Batch,
        pool: &Pool,
        max_length: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), EnqueueError> {
        metrics::counter!("oxide_arena_requests_total").increment(1);
        record_live_arena_bytes();
        let hidden = self.model.hidden_size();
        let pooled_width = (*self.dense)
            .as_ref()
            .map_or(hidden, DenseHead::output_width);
        let geometry = output::OutputGeometry::new(pooled_width, hidden)?;
        let pooled_rows = b.pooled_indices.len();
        let raw_rows = output::raw_rows(b)?;
        let host_elements = geometry.host_elements(b)?;
        let plan = work_key(b)?;
        let host_before = lease.bytes()?;
        let host_after = lease
            .required_bytes(plan, host_elements)?
            .unwrap_or(host_before);
        let mut pinned = self.io.pinned_after(host_before, host_after)?;
        let reuse_active = self.workspace.as_ref().is_some_and(|w| w.fits(plan));
        let mut next = None;
        let workspace_bytes = if reuse_active {
            self.workspace
                .as_ref()
                .map(|work| work.bytes())
                .ok_or_else(|| infer("Missing oxide workspace"))?
        } else {
            // A cached fit can be larger than the requested size class: reserve
            // its actual allocation, not the smaller request estimate.
            next = self
                .cached
                .take_best_fit(|key| capacity_fits(*key, plan, 0));
            match next.as_ref() {
                Some(work) => work.bytes(),
                None => self.model.workspace_bytes(plan)?,
            }
        };
        // Dense scratch is sized by pooled selections only. Raw token rows may
        // grow without touching this allocation, and raw-only requests execute
        // no Dense work while retaining any previous pooled capacity.
        let (dense_capacity, dense_bytes) = match (*self.dense).as_ref() {
            Some(head) => plan_dense_workspace(
                pooled_rows,
                self.dense_workspace
                    .as_ref()
                    .map(|work| (work.capacity(), work.bytes())),
                |capacity| head.work_bytes(capacity),
            )?,
            None => (None, 0),
        };
        let gpu_bytes = workspace_bytes
            .checked_add(dense_bytes)
            .ok_or_else(|| infer("Workspace arena overflow"))?;
        let mut exclusive = needs_exclusive_staging(gpu_bytes, pinned);
        if exclusive {
            let reclaimed = self.io.shrink_free();
            metrics::counter!("oxide_host_io_shrunk_bytes_total").increment(reclaimed as u64);
            pinned = self.io.pinned_after(host_before, host_after)?;
            exclusive = needs_exclusive_staging(gpu_bytes, pinned);
            if exclusive && self.io.active_leases() > 1 {
                if let Some(work) = next.take() {
                    let (key, bytes) = (work.key(), work.bytes());
                    let evicted = self.cached.put(key, work, bytes);
                    metrics::counter!("oxide_arena_evictions_total")
                        .increment(evicted.len() as u64);
                    drop(evicted);
                }
                return Err(BackendError::Overloaded(
                    "Oxide workspace and host staging memory is busy".into(),
                )
                .into());
            }
        }
        let reserve = gpu_bytes
            .checked_add(pinned)
            .ok_or_else(|| infer("Workspace arena overflow"))?;
        // Drop mismatched idle pins before allocating future GPU scratch, then
        // allocate future pins only after the previous scratch is accounted.
        lease.discard_for_resize(plan, host_elements)?;
        let evicted = self.cached.reserve(reserve);
        metrics::counter!("oxide_arena_evictions_total").increment(evicted.len() as u64);
        drop(evicted);
        if !reuse_active {
            if let Some(previous) = self.workspace.take() {
                let (key, bytes) = (previous.key(), previous.bytes());
                let evicted = self.cached.put(key, previous, bytes);
                metrics::counter!("oxide_arena_evictions_total").increment(evicted.len() as u64);
                drop(evicted);
            }
            self.workspace = Some(match next {
                Some(work) => {
                    metrics::counter!("oxide_arena_cached_reuses_total").increment(1);
                    work
                }
                None => self.model.workspace(self.stream.clone(), plan)?,
            });
        }
        if let Some(capacity) = dense_capacity {
            let head = (*self.dense)
                .as_ref()
                .ok_or_else(|| infer("Missing Dense chain"))?;
            // Dense growth is independent of the backbone arena. Dropping the
            // previous workspace drains its ordered uses before reallocation.
            drop(self.dense_workspace.take());
            self.dense_workspace = Some(head.work(self.stream.clone(), capacity)?);
        }
        // All request arena pointers are now fixed. Preselect the actual-M
        // GTE down metadata before host staging, H2D upload, or request kernels.
        let tokens = b.input_ids.len();
        let w = self
            .workspace
            .as_mut()
            .ok_or_else(|| infer("Missing oxide workspace"))?;
        let lt = &*self.lt;
        let plain_blas = (*self.plain_blas).as_ref();
        // SAFETY: enqueue_request bound the engine context above and keeps it
        // current through the ordered body.
        unsafe { self.model.prepare(lt, plain_blas, tokens, w)? };
        // Retire/cache the previous GPU arena before allocating larger pins;
        // otherwise old scratch plus future staging can exceed the projection.
        let host_oversized = pinned > cache::BYTE_BUDGET;
        if host_oversized {
            lease.mark_exclusive();
        }
        lease.prepare(plan, host_elements, host_oversized)?;
        let (input_elements, _) = w.stage(b, geometry, lease)?;
        let s = &*self.stream;
        // Ordinary ordered-stream dispatch; concrete plans were refreshed for
        // the exact logical row count and down metadata was preselected above.
        lease.begin();
        #[cfg(test)]
        bump_lease_begin();
        w.upload(lease.slot()?.input.as_slice(), input_elements)?;
        match (&mut *self.model, &mut *w) {
            (ExecutableModel::Decoder(model), Workspace::Decoder(work)) => {
                model.enqueue(lt, plain_blas, s, work, b, pool, max_length, tokens)?;
                Ok(())
            }
            (ExecutableModel::Encoder(model), Workspace::Encoder(w)) => {
                let c = &model.config;
                let k = &model.kernels;
                let weights = &model.weights;
                let fa4 = model.fa4.as_ref();
                let fa4_layout = model.fa4_layout;
                let rotary = model.rotary.as_ref();
                let plans = &mut model.plans;
                let h = c.hidden_size;
                let batch_count = b
                    .cumulative_seq_lengths
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| infer("Missing oxide sequence boundaries"))?;
                let (mut t, mut width) = (u32_size(tokens)?, u32_size(h)?);
                let (mut count, mut heads, mut dim) = (
                    u32_size(batch_count)?,
                    u32_size(c.num_attention_heads)?,
                    u32_size(
                        h.checked_div(c.num_attention_heads)
                            .ok_or_else(|| infer("Invalid oxide attention head count"))?,
                    )?,
                );
                #[cfg(test)]
                let mut trace = trace::Trace::from_env(c, pool, tokens, batch_count);
                let zero_bias = weights.qkv_zero_bias.cu_deviceptr();
                let base = w.input.cu_deviceptr();
                let (mut ids, mut pos, mut types) = (
                    base,
                    base + product(tokens, 4)? as u64,
                    base + product(tokens, 8)? as u64,
                );
                let boundaries = base + product(tokens, 12)? as u64;
                let selected = boundaries + product(b.cumulative_seq_lengths.len(), 4)? as u64;
                let (mut word, mut position, mut type_weight) = (
                    weights.word.cu_deviceptr(),
                    weights.position.cu_deviceptr(),
                    weights.token_type.cu_deviceptr(),
                );
                let (mut gamma, mut beta) =
                    (weights.gamma.cu_deviceptr(), weights.beta.cu_deviceptr());
                let (mut x, mut eps) = (w.x.cu_deviceptr(), c.layer_norm_eps as f32);
                if rotary.is_some() {
                    launch_threads(
                        &k.embedding,
                        s,
                        (t, 1, 1),
                        128,
                        0,
                        &mut [
                            arg(&mut ids),
                            arg(&mut word),
                            arg(&mut gamma),
                            arg(&mut beta),
                            arg(&mut x),
                            arg(&mut t),
                            arg(&mut width),
                            arg(&mut eps),
                        ],
                    )?;
                } else if let Some(kernel) = &k.embedding_384 {
                    launch_threads(
                        kernel,
                        s,
                        (t, 1, 1),
                        256,
                        0,
                        &mut [
                            arg(&mut ids),
                            arg(&mut pos),
                            arg(&mut types),
                            arg(&mut word),
                            arg(&mut position),
                            arg(&mut type_weight),
                            arg(&mut gamma),
                            arg(&mut beta),
                            arg(&mut x),
                            arg(&mut t),
                            arg(&mut eps),
                        ],
                    )?;
                } else {
                    launch_threads(
                        &k.embedding,
                        s,
                        (t.div_ceil(k.embedding_geometry.rows_per_cta), 1, 1),
                        k.embedding_geometry.threads,
                        0,
                        &mut [
                            arg(&mut ids),
                            arg(&mut pos),
                            arg(&mut types),
                            arg(&mut word),
                            arg(&mut position),
                            arg(&mut type_weight),
                            arg(&mut gamma),
                            arg(&mut beta),
                            arg(&mut x),
                            arg(&mut t),
                            arg(&mut width),
                            arg(&mut eps),
                        ],
                    )?;
                }
                #[cfg(test)]
                if let Some(trace) = trace.as_mut() {
                    trace.tensor(s, &w.x, tokens, h, "embedding");
                }
                for (layer_index, layer) in weights.layers.iter().enumerate() {
                    // SAFETY: this thread bound the engine context once above. The
                    // engine retains all buffers until stream completion/error drain;
                    // input/output buffers are distinct and access is stream-ordered.
                    unsafe {
                        lt.enqueue(
                            &plans.qkv,
                            &w.x,
                            &layer.qkv.weight,
                            &layer.qkv.bias,
                            &mut w.qkv,
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.qkv,
                            tokens,
                            product(h, 3)?,
                            &format!("layer_{layer_index:02}_qkv_pre_rope"),
                        );
                    }
                    if let Some(rotary) = rotary {
                        let mut q = w.qkv.cu_deviceptr();
                        let mut key = q
                            .checked_add(
                                u64::try_from(product(h, 2)?)
                                    .map_err(|_| infer("GTE rotary K offset overflow"))?,
                            )
                            .ok_or_else(|| infer("GTE rotary K offset overflow"))?;
                        let mut cosine = rotary.cosine.cu_deviceptr();
                        let mut sine = rotary.sine.cu_deviceptr();
                        let mut positions = pos;
                        let mut rotary_tokens = t;
                        let mut q_heads = heads;
                        let mut kv_heads = heads;
                        let mut head_dim = rotary.head_dim;
                        let mut rotary_dim = rotary.head_dim;
                        let row = product(h, 3)?;
                        let mut q_stride =
                            u64::try_from(row).map_err(|_| infer("GTE rotary stride overflow"))?;
                        let mut k_stride = q_stride;
                        let mut table_stride = rotary.table_stride;
                        let twice_heads = c
                            .num_attention_heads
                            .checked_mul(2)
                            .ok_or_else(|| infer("GTE rotary grid overflow"))?;
                        let rotary_work =
                            product(product(tokens, twice_heads)?, rotary.head_dim as usize / 2)?;
                        let rope_grid = blocks(rotary_work)?;
                        launch_threads(
                            &rotary.rope_hfma,
                            s,
                            rope_grid,
                            256,
                            0,
                            &mut [
                                arg(&mut q),
                                arg(&mut key),
                                arg(&mut cosine),
                                arg(&mut sine),
                                arg(&mut positions),
                                arg(&mut rotary_tokens),
                                arg(&mut q_heads),
                                arg(&mut kv_heads),
                                arg(&mut head_dim),
                                arg(&mut rotary_dim),
                                arg(&mut q_stride),
                                arg(&mut k_stride),
                                arg(&mut table_stride),
                            ],
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.qkv,
                            tokens,
                            product(h, 3).unwrap(),
                            &format!("layer_{layer_index:02}_qkv"),
                        );
                    }
                    if let Some(provider) = fa4 {
                        let layout =
                            fa4_layout.ok_or_else(|| infer("Missing FA4 packed layout"))?;
                        let q = w.qkv.cu_deviceptr();
                        let attention = fa4::AttentionBatch {
                            q,
                            k: q.checked_add(layout.k_offset)
                                .ok_or_else(|| infer("FA4 packed K address overflow"))?,
                            v: q.checked_add(layout.v_offset)
                                .ok_or_else(|| infer("FA4 packed V address overflow"))?,
                            out: w.attention.cu_deviceptr(),
                            q_offsets: boundaries,
                            k_offsets: boundaries,
                            q_tokens: tokens,
                            k_tokens: tokens,
                            batch: batch_count,
                            max_q: max_length,
                            strides: layout.strides,
                        };
                        // SAFETY: Engine::enqueue_inner bound this context once on
                        // this thread; packed Q/K/V and the separate output remain
                        // retained by the engine, and the cumulative boundaries are
                        // shared int32 data.
                        unsafe { provider.forward_batch_bound(s, &attention)? };
                    } else {
                        let mut qkv = w.qkv.cu_deviceptr();
                        let (mut q, mut key, mut v) = (
                            w.query.cu_deviceptr(),
                            w.key.cu_deviceptr(),
                            w.value.cu_deviceptr(),
                        );
                        let (mut qkv_bias, mut n) = (zero_bias, u32_size(product(tokens, h)?)?);
                        launch(
                            &k.qkv,
                            s,
                            blocks(product(tokens, h)?)?,
                            0,
                            &mut [
                                arg(&mut qkv),
                                arg(&mut qkv_bias),
                                arg(&mut q),
                                arg(&mut key),
                                arg(&mut v),
                                arg(&mut n),
                                arg(&mut width),
                            ],
                        )?;
                        let (mut bounds, mut attn) = (boundaries, w.attention.cu_deviceptr());
                        launch(
                            &k.attention,
                            s,
                            (u32_size(max_length)?, heads, count),
                            u32_size(product(max_length, 4)?)?,
                            &mut [
                                arg(&mut q),
                                arg(&mut key),
                                arg(&mut v),
                                arg(&mut bounds),
                                arg(&mut attn),
                                arg(&mut count),
                                arg(&mut heads),
                                arg(&mut dim),
                                arg(&mut width),
                            ],
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.attention,
                            tokens,
                            h,
                            &format!("layer_{layer_index:02}_attention"),
                        );
                    }
                    // SAFETY: same retained-buffer/context contract as the QKV call.
                    unsafe {
                        lt.enqueue(
                            &plans.attention,
                            &w.attention,
                            &layer.attention.weight,
                            &layer.attention.bias,
                            &mut w.temp,
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.temp,
                            tokens,
                            h,
                            &format!("layer_{layer_index:02}_attention_projection"),
                        );
                    }
                    let temp = w.temp.cu_deviceptr();
                    norm(
                        k,
                        s,
                        temp,
                        x,
                        0,
                        &layer.attention_gamma,
                        &layer.attention_beta,
                        x,
                        t,
                        width,
                        eps,
                    )?;
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.x,
                            tokens,
                            h,
                            &format!("layer_{layer_index:02}_attention_norm"),
                        );
                    }
                    if let Some(rotary) = rotary {
                        let IntermediatePlan::Gated { plan, .. } = &plans.intermediate else {
                            return Err(infer(
                                "GTE gated feed-forward requires a gated intermediate plan",
                            )
                            .into());
                        };
                        let plan = plan
                            .as_ref()
                            .as_ref()
                            .ok_or_else(|| infer("Gated feed-forward plan is not prepared"))?;
                        let blas = plain_blas
                            .ok_or_else(|| infer("Gated feed-forward requires plain cuBLAS"))?;
                        let mut input = {
                            let scratch = (*w.gated)
                                .as_mut()
                                .ok_or_else(|| infer("Missing gated feed-forward scratch"))?;
                            // SAFETY: same retained-buffer/context contract as the QKV call.
                            unsafe {
                                blas.linear(plan, &w.x, &layer.intermediate.weight, scratch)?;
                            }
                            #[cfg(test)]
                            if let Some(trace) = trace.as_mut() {
                                trace.tensor(
                                    s,
                                    scratch,
                                    tokens,
                                    product(c.intermediate_size, 2)?,
                                    &format!("layer_{layer_index:02}_up_gate"),
                                );
                            }
                            scratch.cu_deviceptr()
                        };
                        let mut output = w.intermediate.cu_deviceptr();
                        let mut rows = t;
                        let mut gate_width = u32_size(c.intermediate_size)?;
                        launch(
                            &rotary.gated_gelu,
                            s,
                            blocks(product(tokens, c.intermediate_size)?)?,
                            0,
                            &mut [
                                arg(&mut input),
                                arg(&mut output),
                                arg(&mut rows),
                                arg(&mut gate_width),
                            ],
                        )?;
                    } else {
                        let IntermediatePlan::Dense(plan) = &plans.intermediate else {
                            return Err(infer(
                                "Dense feed-forward requires a dense intermediate plan",
                            )
                            .into());
                        };
                        // SAFETY: same retained-buffer/context contract as the QKV call.
                        unsafe {
                            lt.enqueue(
                                plan,
                                &w.x,
                                &layer.intermediate.weight,
                                &layer.intermediate.bias,
                                &mut w.intermediate,
                            )?;
                        }
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.intermediate,
                            tokens,
                            c.intermediate_size,
                            &format!("layer_{layer_index:02}_gelu"),
                        );
                    }
                    // SAFETY: same retained-buffer/context contract as the QKV call.
                    unsafe {
                        enqueue_down(
                            &mut plans.output,
                            lt,
                            plain_blas,
                            tokens,
                            layer_index,
                            &w.intermediate,
                            &layer.output.weight,
                            &layer.output.bias,
                            &mut w.temp,
                            s,
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.temp,
                            tokens,
                            h,
                            &format!("layer_{layer_index:02}_output_projection"),
                        );
                    }
                    norm(
                        k,
                        s,
                        temp,
                        x,
                        0,
                        &layer.output_gamma,
                        &layer.output_beta,
                        x,
                        t,
                        width,
                        eps,
                    )?;
                    #[cfg(test)]
                    if let Some(trace) = trace.as_mut() {
                        trace.tensor(
                            s,
                            &w.x,
                            tokens,
                            h,
                            &format!("layer_{layer_index:02}_output_norm"),
                        );
                    }
                }
                #[cfg(test)]
                if let Some(trace) = trace.take() {
                    trace.finish();
                }
                let (mut bounds, mut indices, mut out) =
                    (boundaries, selected, w.pooled.cu_deviceptr());
                let mut count = u32_size(b.pooled_indices.len())?;
                let mut mode = match pool {
                    Pool::Cls => 0u32,
                    Pool::Mean => 1,
                    Pool::LastToken => 2,
                    Pool::Splade => return Err(infer("Oxide SPLADE unsupported").into()),
                };
                if count > 0 && mode == 1 {
                    launch(
                        &k.mean_pool,
                        s,
                        (count, u32_size(h.div_ceil(32))?, 1),
                        0,
                        &mut [
                            arg(&mut x),
                            arg(&mut bounds),
                            arg(&mut indices),
                            arg(&mut out),
                            arg(&mut count),
                            arg(&mut width),
                        ],
                    )?;
                } else if count > 0 {
                    launch(
                        &k.pool,
                        s,
                        blocks(product(count as usize, h)?)?,
                        0,
                        &mut [
                            arg(&mut x),
                            arg(&mut bounds),
                            arg(&mut indices),
                            arg(&mut out),
                            arg(&mut count),
                            arg(&mut width),
                            arg(&mut mode),
                        ],
                    )?;
                }
                if !b.raw_indices.is_empty() {
                    let function = k
                        .raw_cast
                        .as_ref()
                        .ok_or_else(|| infer("Raw output kernel is unavailable"))?;
                    enqueue_raw_output(function, s, x, w.pooled.cu_deviceptr(), selected, b, h)?;
                }
                Ok(())
            }
            _ => Err(infer("Oxide model/workspace mismatch")),
        }?;
        let dense = match ((*self.dense).as_mut(), self.dense_workspace.as_mut()) {
            (Some(_), _) if pooled_rows == 0 => None,
            (Some(head), Some(work)) => {
                if !work.fits(pooled_rows) {
                    return Err(infer("Dense request exceeds retained arena capacity").into());
                }
                Some((head, work))
            }
            (Some(_), None) => return Err(infer("Missing Dense workspace").into()),
            (None, _) => None,
        };
        Ok(w.finish(dense, lt, plain_blas, pooled_rows, raw_rows, hidden, lease)?)
    }
}

enum LoadedConfig {
    Encoder(BertConfig),
    Decoder(DecoderConfig),
}
impl LoadedConfig {
    #[cfg_attr(not(test), allow(dead_code))]
    fn hidden_size(&self) -> usize {
        match self {
            Self::Encoder(config) => config.hidden_size,
            Self::Decoder(config) => config.hidden_size,
        }
    }
}

fn detects_decoder(path: &Path) -> Result<bool, BackendError> {
    #[derive(serde::Deserialize)]
    struct Head {
        model_type: String,
    }
    let head: Head = model::read_json(&path.join("config.json"))?;
    Ok(DecoderFamily::from_model_type(&head.model_type).is_some())
}

pub struct OxideBackend {
    config: LoadedConfig,
    output_width: usize,
    pool: Pool,
    attention_length_limit: Option<usize>,
    raw_output_supported: bool,
    engine: Arc<Mutex<Engine>>,
    stream: Arc<Stream>,
    io: Arc<SlotPool<HostIo>>,
}
impl OxideBackend {
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_with_dense_paths(path, dtype, model_type, device_id, None)
    }

    /// Construct an Oxide backend with an explicit ordered Dense module list.
    /// `None` discovers admitted Dense entries from `modules.json`.
    pub fn new_with_dense_paths(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device_id: usize,
        dense_paths: Option<Vec<String>>,
    ) -> Result<Self, BackendError> {
        if dtype != "bfloat16" {
            return Err(start("Oxide currently requires --dtype bfloat16"));
        }
        let pool = match model_type {
            ModelType::Embedding(p @ (Pool::Cls | Pool::Mean | Pool::LastToken)) => p,
            _ => return Err(start("Oxide currently supports dense embeddings only")),
        };
        if device_id >= device_count()? {
            return Err(start("CUDA device ordinal is not visible"));
        }
        let context = Context::new(device_id).map_err(start)?;
        let stream = context.new_stream().map_err(start)?;
        let decoder = detects_decoder(path)?;
        let native_default = if decoder {
            false
        } else {
            model::read_json::<BertConfig>(&path.join("config.json"))?.family == EncoderFamily::Gte
        };
        let lt = if native_default {
            LtProvider::new_native_default(context.clone(), stream.clone())?
        } else {
            LtProvider::new(context.clone(), stream.clone())?
        };
        let (model, config, plain_blas, attention_length_limit, dense, output_width) = if decoder {
            let host = DecoderWeights::load(path)?;
            let config = host.config.clone();
            let dense_host =
                DenseChainHost::load_with_paths(path, config.hidden_size, dense_paths.clone())?;
            let output_width = dense_host
                .as_ref()
                .map_or(config.hidden_size, DenseChainHost::output_width);
            let model = decoder::DecoderModel::new(host, &context, &stream, &lt)?;
            let needs_plain =
                model.needs_plain() || dense_host.as_ref().is_some_and(DenseChainHost::uses_plain);
            let dense = dense_host
                .map(|host| DenseHead::new(host, &context, &stream, &lt))
                .transpose()?;
            let plain_blas = if needs_plain {
                Some(Blas::new(context.clone(), stream.clone())?)
            } else {
                None
            };
            (
                ExecutableModel::Decoder(Box::new(model)),
                LoadedConfig::Decoder(config),
                plain_blas,
                None,
                dense,
                output_width,
            )
        } else {
            let host = BertWeights::load(path)?;
            let config = host.config.clone();
            let dense_host =
                DenseChainHost::load_with_paths(path, config.hidden_size, dense_paths.clone())?;
            let output_width = dense_host
                .as_ref()
                .map_or(config.hidden_size, DenseChainHost::output_width);
            let model = encoder_model(host, &context, &stream, &lt)?;
            let dense = dense_host
                .map(|host| DenseHead::new(host, &context, &stream, &lt))
                .transpose()?;
            let plain_blas = if dense.as_ref().is_some_and(DenseHead::uses_plain)
                || config.recipe.feed_forward == FeedForwardRecipe::GatedGeluUpFirst
            {
                Some(Blas::new(context.clone(), stream.clone())?)
            } else {
                None
            };
            let attention_length_limit = model.kernels.attention_length_limit;
            (
                ExecutableModel::Encoder(Box::new(model)),
                LoadedConfig::Encoder(config),
                plain_blas,
                attention_length_limit,
                dense,
                output_width,
            )
        };
        let io = HostIo::pool(&context)?;
        let raw_output_supported = model.supports_raw_output();
        Ok(Self {
            config,
            output_width,
            pool,
            attention_length_limit,
            raw_output_supported,
            engine: Arc::new(Mutex::new(Engine {
                lt: ManuallyDrop::new(lt),
                model: ManuallyDrop::new(model),
                dense: ManuallyDrop::new(dense),
                plain_blas: ManuallyDrop::new(plain_blas),
                context,
                stream: stream.clone(),
                io: io.clone(),
                workspace: None,
                dense_workspace: None,
                cached: cache::WorkspaceCache::new(),
                failed: false,
            })),
            stream,
            io,
        })
    }
}

fn encoder_model(
    host: BertWeights,
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    lt: &LtProvider,
) -> Result<EncoderModel, BackendError> {
    let config = host.config.clone();
    let attention_mode = std::env::var(infinity_v3_config::infinity_env!("OXIDE_ATTENTION"))
        .unwrap_or_else(|_| "auto".into());
    let fa4_root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"));
    let head_dim = config
        .hidden_size
        .checked_div(config.num_attention_heads)
        .ok_or_else(|| start("Invalid oxide attention head count"))?;
    let attention_spec = fa4::AttentionSpec {
        heads: config.num_attention_heads,
        kv_heads: config.num_attention_heads,
        head_dim,
        causal: false,
        window: fa4::Window::Global,
        dtype: fa4::DType::Bf16,
        scale: fa4::Scale::RsqrtHeadDim,
    };
    let packed = packed_attention_layout(&config)?;
    let fa4 = match attention_mode.as_str() {
        "fa4" => {
            if packed.is_none() {
                return Err(start(
                    "FA4 requires packed QKV and output row strides divisible by 8 elements",
                ));
            }
            Some(fa4::Fa4::new(
                context.clone(),
                Path::new(
                    &fa4_root.ok_or_else(|| start("FA4 attention requires INFINITY_FA4_ROOT"))?,
                ),
                attention_spec,
            )?)
        }
        "auto" if fa4_root.is_some() && fa4::Fa4::supports(&attention_spec) && packed.is_some() => {
            Some(fa4::Fa4::new(
                context.clone(),
                Path::new(&fa4_root.ok_or_else(|| start("Missing FA4 root"))?),
                attention_spec,
            )?)
        }
        "auto" | "simt" => None,
        _ => return Err(start("INFINITY_OXIDE_ATTENTION must be auto,simt or fa4")),
    };
    let fa4_layout = if fa4.is_some() { packed } else { None };
    let mut kernels = Kernels::load(context, &config, fa4.is_some())?;
    let rotary = match config.recipe.positions {
        PositionRecipe::Absolute => None,
        PositionRecipe::Rotary(recipe) => {
            let functions = kernels
                .rotary
                .take()
                .ok_or_else(|| start("Missing prepared GTE rotary kernels"))?;
            Some(RotaryEncoder::new(recipe, &config, stream, functions)?)
        }
    };
    let plans = Plans::new(lt, &config)?;
    let mut uploader = WeightUploader::new(stream.clone())?;
    let weights = Weights {
        word: uploader.upload(&[&host.word])?,
        position: uploader.upload(&[&host.position])?,
        token_type: uploader.upload(&[&host.token_type])?,
        gamma: uploader.upload(&[&host.gamma])?,
        beta: uploader.upload(&[&host.beta])?,
        qkv_zero_bias: DeviceBuffer::zeroed(stream, product(config.hidden_size, 3)?)
            .map_err(start)?,
        layers: host
            .layers
            .iter()
            .map(|l| Layer::upload(&mut uploader, l))
            .collect::<Result<_, _>>()?,
    };
    uploader.finish()?;
    Ok(EncoderModel {
        config,
        plans,
        kernels,
        weights,
        rotary,
        fa4,
        fa4_layout,
    })
}

fn validate(batch: &Batch, c: &BertConfig) -> Result<usize, BackendError> {
    let fail = || infer("Invalid or unsupported packed oxide BERT batch");
    let tokens = batch.input_ids.len();
    let sequences = batch
        .cumulative_seq_lengths
        .len()
        .checked_sub(1)
        .filter(|n| *n > 0)
        .ok_or_else(fail)?;
    if tokens == 0
        || batch.cumulative_seq_lengths.first() != Some(&0)
        || batch
            .cumulative_seq_lengths
            .last()
            .copied()
            .map(|n| n as usize)
            != Some(tokens)
        || batch.position_ids.len() != tokens
        || batch.token_type_ids.len() != tokens
        || batch.multimodal.iter().any(Option::is_some)
        || batch.compact_input_ids.is_some()
        || batch.compact_position_ids.is_some()
        || batch.scatter_unfold.is_some()
        || batch.fold_gather.is_some()
    {
        return Err(fail());
    }
    let mut max_length = 0;
    for pair in batch.cumulative_seq_lengths.windows(2) {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            return Err(fail());
        };
        let length = end.checked_sub(start).filter(|n| *n > 0).ok_or_else(fail)? as usize;
        if length > c.max_position_embeddings {
            return Err(fail());
        }
        max_length = max_length.max(length);
    }
    let ignores_token_type_ids = c.family == model::EncoderFamily::Gte
        && c.recipe.embedding == NormArithmetic::AtenEmbedding;
    if sequences > 65535
        || batch
            .pooled_indices
            .iter()
            .chain(&batch.raw_indices)
            .any(|&i| i as usize >= sequences)
        || batch.input_ids.iter().any(|&i| i as usize >= c.vocab_size)
        || (!ignores_token_type_ids
            && batch
                .token_type_ids
                .iter()
                .any(|&i| i as usize >= c.type_vocab_size))
        || batch
            .position_ids
            .iter()
            .any(|&i| i as usize >= c.max_position_embeddings)
    {
        return Err(fail());
    }
    u32_size(product(tokens, c.intermediate_size)?)?;
    if c.recipe.feed_forward == FeedForwardRecipe::GatedGeluUpFirst {
        let gate_width = product(c.intermediate_size, 2)?;
        u32_size(product(tokens, gate_width)?)?;
        // Bucket padding is admitted here as well: rejecting it after engine
        // submission would turn an unsupported request into a poisoned engine.
        u32_size(product(bucketed(tokens, MIN_TOKEN_CAPACITY)?, gate_width)?)?;
    }
    u32_size(product(tokens, c.hidden_size)?)?;
    Ok(max_length)
}

fn validate_loaded(batch: &Batch, config: &LoadedConfig) -> Result<usize, BackendError> {
    match config {
        LoadedConfig::Encoder(config) => validate(batch, config),
        LoadedConfig::Decoder(config) => validate_decoder(batch, config),
    }
}

fn validate_decoder(batch: &Batch, c: &DecoderConfig) -> Result<usize, BackendError> {
    let fail = || infer("Invalid or unsupported packed oxide decoder batch");
    let tokens = batch.input_ids.len();
    let sequences = batch
        .cumulative_seq_lengths
        .len()
        .checked_sub(1)
        .filter(|n| *n > 0)
        .ok_or_else(fail)?;
    if tokens == 0
        || batch.cumulative_seq_lengths.first() != Some(&0)
        || batch
            .cumulative_seq_lengths
            .last()
            .copied()
            .map(|n| n as usize)
            != Some(tokens)
        || batch.position_ids.len() != tokens
        || batch.token_type_ids.len() != tokens
        || batch.multimodal.iter().any(Option::is_some)
        || batch.compact_input_ids.is_some()
        || batch.compact_position_ids.is_some()
        || batch.scatter_unfold.is_some()
        || batch.fold_gather.is_some()
        || batch.token_type_ids.iter().any(|&kind| kind != 0)
    {
        return Err(fail());
    }
    let mut max_length = 0;
    for pair in batch.cumulative_seq_lengths.windows(2) {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            return Err(fail());
        };
        let length = end.checked_sub(start).filter(|n| *n > 0).ok_or_else(fail)? as usize;
        if length > c.max_position_embeddings {
            return Err(fail());
        }
        max_length = max_length.max(length);
    }
    if sequences > 65535
        || batch
            .pooled_indices
            .iter()
            .chain(&batch.raw_indices)
            .any(|&i| i as usize >= sequences)
        || batch.input_ids.iter().any(|&i| i as usize >= c.vocab_size)
        || batch
            .position_ids
            .iter()
            .any(|&i| i as usize >= c.max_position_embeddings)
    {
        return Err(fail());
    }
    u32_size(product(tokens, c.q_width())?)?;
    u32_size(product(tokens, c.kv_width())?)?;
    u32_size(product(tokens, c.intermediate_size)?)?;
    u32_size(product(tokens, c.hidden_size)?)?;
    Ok(max_length)
}

impl Backend for OxideBackend {
    fn health(&self) -> Result<(), BackendError> {
        if self.io.poisoned() {
            return Err(BackendError::Unhealthy);
        }
        let mut engine = match self.engine.lock() {
            Ok(engine) => engine,
            Err(_) => {
                self.io.close();
                return Err(BackendError::Unhealthy);
            }
        };
        if engine.failed || self.io.poisoned() {
            self.io.close();
            return Err(BackendError::Unhealthy);
        }
        if let Err(error) = engine.stream.synchronize() {
            engine.failed = true;
            self.io.close();
            return Err(infer(error));
        }
        Ok(())
    }
    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch)?.complete()
    }
    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        let max_length = validate_loaded(&batch, &self.config)?;
        if let Some(limit) = self.attention_length_limit {
            if max_length > limit {
                return Err(infer(format!(
                    "SIMT attention sequence length {max_length} exceeds kernel limit {limit}"
                )));
            }
        }
        if batch.pooled_indices.is_empty() && batch.raw_indices.is_empty() {
            return Ok(Inference::Ready(Embeddings::default()));
        }
        if !batch.raw_indices.is_empty() && !self.raw_output_supported {
            return Err(infer(
                "This oxide kernel bundle does not support token outputs",
            ));
        }
        let hidden = self.config.hidden_size();
        let output_width = self.output_width;
        let geometry = output::OutputGeometry::new(output_width, hidden)?;
        let elements = geometry.host_elements(&batch)?;
        u32_size(elements)?;
        // Only the packed result rows are result-memory. Input/output pins are
        // charged to the backend's bounded host I/O arena.
        let memory = result_memory::global_pool()?.reserve(output::result_bytes(
            &batch,
            output_width,
            hidden,
        )?)?;
        let slot = self.io.acquire()?;
        let mut lease = HostIoLease::new(
            slot,
            self.io.clone(),
            self.engine.clone(),
            self.stream.clone(),
        );
        {
            let mut engine = match self.engine.lock() {
                Ok(engine) => engine,
                Err(_) => {
                    self.io.close();
                    return Err(BackendError::Unhealthy);
                }
            };
            if engine.failed || self.io.poisoned() {
                self.io.close();
                return Err(BackendError::Unhealthy);
            }
            engine.enqueue(&batch, &self.pool, max_length, &mut lease)?;
        }
        if !batch.raw_indices.is_empty() {
            let pooled = batch.pooled_indices;
            let raw = batch.raw_indices;
            let boundaries = batch.cumulative_seq_lengths;
            return Ok(Inference::Deferred(Box::new(move || {
                lease.wait()?;
                let output = lease.output(elements)?;
                if output.iter().any(|value| !value.is_finite()) {
                    lease.poison();
                    return Err(infer("Invalid oxide token output"));
                }
                let mut result =
                    output::pack(output, output_width, hidden, &pooled, &raw, &boundaries)?;
                result.memory = Some(memory);
                Ok(result)
            })));
        }
        let pooled = batch.pooled_indices;
        Ok(Inference::Deferred(Box::new(move || {
            // The lease owns the exact output size and pins; never consult the
            // shared Work after the next submission may have replaced it.
            lease.wait()?;
            let output = lease.output(elements)?;
            if output.iter().any(|value| !value.is_finite()) {
                lease.poison();
                return Err(infer("Invalid oxide embedding output"));
            }
            let mut result = Embeddings::default();
            result.memory = Some(memory);
            for (&index, row) in pooled.iter().zip(output.chunks_exact(output_width)) {
                result.insert(index as usize, Embedding::Pooled(row.to_vec()));
            }
            Ok(result)
        })))
    }
    fn predict(&self, _: Batch) -> Result<Predictions, BackendError> {
        Err(infer("Oxide classifiers are not implemented"))
    }
    fn predict_tokens(&self, _: Batch) -> Result<TokenPredictions, BackendError> {
        Err(infer("Oxide token predictions are not implemented"))
    }
}

#[cfg(test)]
mod trace {
    use super::*;

    pub(super) struct Trace {
        dir: std::path::PathBuf,
        entries: Vec<serde_json::Value>,
        model: serde_json::Value,
        config: serde_json::Value,
        pool: &'static str,
        tokens: usize,
        sequences: usize,
    }

    impl Trace {
        pub(super) fn from_env(
            c: &BertConfig,
            pool: &Pool,
            tokens: usize,
            sequences: usize,
        ) -> Option<Self> {
            let dir = std::path::PathBuf::from(std::env::var_os("INFINITY_OXIDE_TRACE_DIR")?);
            std::fs::create_dir_all(&dir).unwrap();
            Some(Self {
                dir,
                entries: Vec::new(),
                model: serde_json::json!({
                    "family": format!("{:?}", c.family),
                    "embedding_arithmetic": format!("{:?}", c.recipe.embedding),
                    "residual_arithmetic": format!("{:?}", c.recipe.residual),
                }),
                config: serde_json::json!({
                    "hidden_size": c.hidden_size,
                    "intermediate_size": c.intermediate_size,
                    "num_attention_heads": c.num_attention_heads,
                    "num_hidden_layers": c.num_hidden_layers,
                    "max_position_embeddings": c.max_position_embeddings,
                    "vocab_size": c.vocab_size,
                    "type_vocab_size": c.type_vocab_size,
                    "layer_norm_eps": c.layer_norm_eps,
                }),
                pool: match pool {
                    Pool::Cls => "cls",
                    Pool::Mean => "mean",
                    Pool::LastToken => "last_token",
                    Pool::Splade => "splade",
                },
                tokens,
                sequences,
            })
        }

        pub(super) fn tensor(
            &mut self,
            stream: &Stream,
            buffer: &DeviceBuffer<Bf16>,
            tokens: usize,
            width: usize,
            name: &str,
        ) {
            assert!(
                width > 0 && tokens * width <= buffer.len(),
                "trace shape exceeds {name}"
            );
            stream.synchronize().unwrap();
            let mut host = vec![0u16; tokens * width];
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    host.as_mut_ptr(),
                    buffer.cu_deviceptr(),
                    host.len() * 2,
                    stream.cu_stream(),
                )
            }
            .unwrap();
            stream.synchronize().unwrap();
            let mut bytes = Vec::with_capacity(host.len() * 2);
            for value in host {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(self.dir.join(format!("{name}.bf16")), &bytes).unwrap();
            self.entries.push(serde_json::json!({
                "file": format!("{name}.bf16"),
                "shape": [tokens, width],
            }));
        }

        pub(super) fn finish(self) {
            let manifest = serde_json::json!({
                "model": self.model,
                "pool": self.pool,
                "config": self.config,
                "tokens": self.tokens,
                "sequences": self.sequences,
                "tensors": self.entries,
            });
            std::fs::write(
                self.dir.join("manifest.json"),
                serde_json::to_vec_pretty(&manifest).unwrap(),
            )
            .unwrap();
        }
    }

    pub(super) fn write_pooled(dir: &Path, embeddings: &Embeddings, hidden: usize) {
        let mut rows: Vec<_> = embeddings
            .iter()
            .map(|(&index, row)| (index, row))
            .collect();
        rows.sort_unstable_by_key(|(index, _)| *index);
        let mut bytes = Vec::with_capacity(rows.len() * hidden * 4);
        let mut indices = Vec::with_capacity(rows.len());
        for (index, embedding) in rows {
            let Embedding::Pooled(values) = embedding else {
                panic!("oxide trace requires pooled embeddings");
            };
            assert_eq!(values.len(), hidden);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            indices.push(index);
        }
        std::fs::write(dir.join("pooled.f32"), &bytes).unwrap();
        let manifest_path = dir.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["output"] = serde_json::json!({
            "file": "pooled.f32",
            "dtype": "f32",
            "shape": [indices.len(), hidden],
            "indices": indices,
        });
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn raw_span(batch: &Batch, index: u32) -> Result<usize, String> {
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| "raw trace batch has no sequences".to_string())?;
        let index = usize::try_from(index)
            .map_err(|_| format!("raw index {index} is out of bounds for {sequences} sequences"))?;
        if index >= sequences {
            return Err(format!(
                "raw index {index} is out of bounds for {sequences} sequences"
            ));
        }
        let start = batch.cumulative_seq_lengths[index] as usize;
        let end = batch.cumulative_seq_lengths[index + 1] as usize;
        let span = end
            .checked_sub(start)
            .ok_or_else(|| format!("raw index {index} has a reversed token span"))?;
        if end > batch.input_ids.len() {
            return Err(format!(
                "raw index {index} span ends at {end}, beyond {} tokens",
                batch.input_ids.len()
            ));
        }
        Ok(span)
    }

    pub(super) fn validate_outputs(
        batch: &Batch,
        result: &Embeddings,
        output_width: usize,
        hidden: usize,
    ) -> Result<(), String> {
        let mut by_index = std::collections::BTreeMap::new();
        for (&index, embedding) in result.iter() {
            if by_index.insert(index, embedding).is_some() {
                return Err(format!("duplicate trace result index {index}"));
            }
        }
        let mut expected: std::collections::BTreeMap<u32, &'static str> =
            std::collections::BTreeMap::new();
        for &index in &batch.pooled_indices {
            expected.insert(index, "pooled");
        }
        for &index in &batch.raw_indices {
            expected.insert(index, "raw");
        }
        if by_index.len() != expected.len() {
            return Err(format!(
                "trace result has {} indices, expected {}",
                by_index.len(),
                expected.len()
            ));
        }
        for (&index, &kind) in &expected {
            let Some(embedding) = by_index.get(&(index as usize)).copied() else {
                return Err(format!("missing {kind} trace index {index}"));
            };
            match (kind, embedding) {
                ("pooled", Embedding::Pooled(values)) => {
                    if values.len() != output_width {
                        return Err(format!(
                            "pooled trace index {index} has width {}, expected {output_width}",
                            values.len()
                        ));
                    }
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(format!("pooled trace index {index} is not finite"));
                    }
                }
                ("pooled", Embedding::All(_)) => {
                    return Err(format!("pooled trace index {index} has raw kind"));
                }
                ("raw", Embedding::All(rows)) => {
                    let span = raw_span(batch, index)?;
                    if rows.len() != span {
                        return Err(format!(
                            "raw trace index {index} has {} rows, expected {span}",
                            rows.len()
                        ));
                    }
                    for row in rows {
                        if row.len() != hidden {
                            return Err(format!(
                                "raw trace index {index} has width {}, expected {hidden}",
                                row.len()
                            ));
                        }
                        if row.iter().any(|value| !value.is_finite()) {
                            return Err(format!("raw trace index {index} is not finite"));
                        }
                    }
                }
                ("raw", Embedding::Pooled(_)) => {
                    return Err(format!("raw trace index {index} has pooled kind"));
                }
                _ => unreachable!("trace kind is pooled or raw"),
            }
        }
        for &index in by_index.keys() {
            if !expected.contains_key(&(index as u32)) {
                return Err(format!("unrequested trace result index {index}"));
            }
        }
        Ok(())
    }

    pub(super) fn outputs_json(
        batch: &Batch,
        result: &Embeddings,
        output_width: usize,
        hidden: usize,
        physical_batch: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        validate_outputs(batch, result, output_width, hidden)?;
        let mut rows: Vec<_> = result.iter().collect();
        rows.sort_unstable_by_key(|(&index, _)| index);
        let mut outputs = Vec::with_capacity(rows.len());
        for (&index, embedding) in rows {
            let (kind, shape, values) = match embedding {
                Embedding::Pooled(values) => ("pooled", vec![1, output_width], values.clone()),
                Embedding::All(rows) => {
                    let mut values = Vec::with_capacity(rows.len() * hidden);
                    for row in rows {
                        values.extend_from_slice(row);
                    }
                    ("raw", vec![rows.len(), hidden], values)
                }
            };
            outputs.push(serde_json::json!({
                "index": index,
                "kind": kind,
                "shape": shape,
                "values": values,
            }));
        }
        Ok(serde_json::json!({
            "outputs": outputs,
            "physical_batch": physical_batch,
        }))
    }

    pub(super) fn write_outputs(
        dir: &Path,
        batch: &Batch,
        result: &Embeddings,
        output_width: usize,
        hidden: usize,
        physical_batch: serde_json::Value,
    ) {
        std::fs::create_dir_all(dir).unwrap();
        let outputs = outputs_json(batch, result, output_width, hidden, physical_batch).unwrap();
        std::fs::write(
            dir.join("outputs.json"),
            serde_json::to_vec_pretty(&outputs).unwrap(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn gte_request_validation_and_arena_account_for_packed_gates() {
        let config: BertConfig = serde_json::from_value(serde_json::json!({
            "model_type": "new", "architectures": ["NewModel"],
            "hidden_size": 768, "intermediate_size": 3072,
            "num_attention_heads": 12, "num_hidden_layers": 12,
            "max_position_embeddings": 8192, "vocab_size": 30528,
            "type_vocab_size": 0, "rope_theta": 500000.0,
            "rope_scaling": {"type": "ntk", "factor": 2.0}
        }))
        .unwrap();
        let mut request = batch();
        request.token_type_ids.fill(0);
        request.position_ids[0] = 8191;
        assert!(validate(&request, &config).is_ok());
        request.position_ids[0] = 8192;
        assert!(validate(&request, &config).is_err());
        request.position_ids[0] = 0;
        request.token_type_ids = vec![1, 0, u32::MAX];
        assert!(validate(&request, &config).is_ok());
        request.token_type_ids.pop();
        assert!(validate(&request, &config).is_err());
        request.token_type_ids = vec![0; 3];
        assert!(validate(&request, &config).is_ok());

        let bert = self::config();
        let mut bert_request = batch();
        bert_request.token_type_ids = vec![1, 0, 0];
        assert!(validate(&bert_request, &bert).is_ok());
        bert_request.token_type_ids = vec![u32::MAX, 0, 0];
        assert!(validate(&bert_request, &bert).is_err());

        let tokens = (1usize << 19) + 1;
        let mut oversized = batch();
        oversized.input_ids = vec![1; tokens];
        oversized.position_ids = (0..tokens).map(|index| (index % 8192) as u32).collect();
        oversized.token_type_ids = vec![0; tokens];
        oversized.cumulative_seq_lengths = (0..tokens)
            .step_by(8192)
            .map(|index| index as u32)
            .collect();
        oversized.cumulative_seq_lengths.push(tokens as u32);
        oversized.max_length = 8192;
        // Actual gates fit u32; their next-power-of-two arena does not. This
        // must fail in CPU admission, before submit_embed touches the engine.
        assert!(u32::try_from(tokens * 6144).is_ok());
        assert!(validate(&oversized, &config).is_err());
        assert!(validate(&request, &config).is_ok());

        let key = (32, 4, 2);
        let mut dense = config.clone();
        dense.recipe.feed_forward = FeedForwardRecipe::DenseGelu;
        let gate_bytes = Work::allocation_bytes(key, &config).unwrap();
        let dense_bytes = Work::allocation_bytes(key, &dense).unwrap();
        assert_eq!(
            gate_bytes - dense_bytes,
            32 * 2 * 3072 * size_of::<half::bf16>()
        );
    }

    fn config() -> BertConfig {
        serde_json::from_value(serde_json::json!({"model_type":"bert","hidden_size":4,
            "intermediate_size":8,"num_attention_heads":2,"num_hidden_layers":1,
            "max_position_embeddings":8,"vocab_size":4,"type_vocab_size":2}))
        .unwrap()
    }
    fn batch() -> Batch {
        Batch {
            input_ids: vec![1, 2, 3],
            position_ids: vec![0, 0, 1],
            token_type_ids: vec![0, 0, 0],
            cumulative_seq_lengths: vec![0, 1, 3],
            pooled_indices: vec![0, 1],
            max_length: 2,
            multimodal: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    fn same_bucket_batch() -> Batch {
        let mut b = batch();
        b.input_ids = vec![1, 2, 3, 1, 2];
        b.position_ids = vec![0, 1, 0, 1, 2];
        b.token_type_ids = vec![0; 5];
        b.cumulative_seq_lengths = vec![0, 2, 5];
        b.pooled_indices = vec![0, 1];
        b.max_length = 3;
        b
    }
    fn changed_shape_batch() -> Batch {
        let tokens = 40;
        let mut b = batch();
        b.input_ids = (0..tokens).map(|i| (i % 3 + 1) as u32).collect();
        b.position_ids = (0..tokens).map(|i| (i % 10) as u32).collect();
        b.token_type_ids = vec![0; tokens];
        b.cumulative_seq_lengths = vec![0, 10, 20, 30, 40];
        b.pooled_indices = vec![0, 1, 2, 3];
        b.max_length = 10;
        b
    }
    fn repeated_sequences(sequences: usize, length: usize) -> Batch {
        let tokens = sequences * length;
        let mut b = batch();
        b.input_ids = (0..tokens).map(|i| (i % 3 + 1) as u32).collect();
        b.position_ids = (0..tokens).map(|i| (i % length) as u32).collect();
        b.token_type_ids = vec![0; tokens];
        b.cumulative_seq_lengths = (0..=sequences).map(|i| (i * length) as u32).collect();
        b.pooled_indices = (0..sequences).map(|i| i as u32).collect();
        b.max_length = length as u32;
        b
    }
    fn decoder_config() -> DecoderConfig {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "infinity-decoder-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(
            path.join("config.json"),
            r#"{
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 151646,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "hidden_act": "silu",
                "is_causal": false
            }"#,
        )
        .unwrap();
        let config = DecoderConfig::load(&path).unwrap();
        std::fs::remove_dir_all(&path).unwrap();
        config
    }
    fn decoder_batch(lengths: &[usize]) -> Batch {
        let tokens = lengths.iter().sum::<usize>();
        let mut boundaries = Vec::with_capacity(lengths.len() + 1);
        boundaries.push(0u32);
        let mut position_ids = Vec::with_capacity(tokens);
        for (index, &length) in lengths.iter().enumerate() {
            boundaries.push(boundaries[index] + length as u32);
            position_ids.extend(0..length as u32);
        }
        Batch {
            input_ids: (0..tokens).map(|i| (1000 + i % 97) as u32).collect(),
            position_ids,
            token_type_ids: vec![0; tokens],
            cumulative_seq_lengths: boundaries,
            pooled_indices: (0..lengths.len()).map(|i| i as u32).collect(),
            max_length: lengths.iter().copied().max().unwrap_or(0) as u32,
            multimodal: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    #[test]
    fn decoder_validation_enforces_word_only_positions_and_boundaries() {
        let c = decoder_config();
        let b = decoder_batch(&[3, 5, 2]);
        assert_eq!(validate_decoder(&b, &c).unwrap(), 5);
        for mutate in [
            |b: &mut Batch| b.token_type_ids[2] = 1,
            |b: &mut Batch| b.position_ids[0] = 131072,
            |b: &mut Batch| b.input_ids[0] = 151646,
            |b: &mut Batch| b.pooled_indices = vec![3],
            |b: &mut Batch| b.cumulative_seq_lengths = vec![0, 4, 3, 10],
            |b: &mut Batch| b.cumulative_seq_lengths = vec![1, 3, 10],
        ] {
            let mut invalid = b.clone();
            mutate(&mut invalid);
            assert!(validate_decoder(&invalid, &c).is_err());
        }
    }

    #[test]
    fn decoder_duplicate_excess_keeps_the_canonical_readout_fast_path() {
        let c = decoder_config();
        let mut b = decoder_batch(&[3, 5, 2]);
        // Three sequences with three nonordered duplicate selections: valid.
        b.pooled_indices = vec![2, 0, 2];
        assert!(validate_decoder(&b, &c).is_ok());
        assert_eq!(
            decoder::selected_readout(&Pool::LastToken, b.pooled_indices.len()),
            Some(decoder::SelectedReadout::LastToken)
        );
        assert_eq!(
            decoder::selected_readout(&Pool::Cls, b.pooled_indices.len()),
            Some(decoder::SelectedReadout::Cls)
        );
        // Duplicate excess is valid and does not leave the fast path: canonical
        // B rows are projected, then the cast selects S requested rows.
        b.pooled_indices = vec![0, 1, 2, 0];
        assert!(validate_decoder(&b, &c).is_ok());
        assert_eq!(
            decoder::selected_readout(&Pool::LastToken, b.pooled_indices.len()),
            Some(decoder::SelectedReadout::LastToken)
        );
        assert_eq!(
            decoder::selected_readout(&Pool::Cls, b.pooled_indices.len()),
            Some(decoder::SelectedReadout::Cls)
        );
        assert_eq!(
            decoder::selected_readout(&Pool::Mean, b.pooled_indices.len()),
            None
        );
    }

    fn pooled_bits(embeddings: Embeddings) -> Vec<(usize, Vec<u32>)> {
        let mut rows = Vec::with_capacity(embeddings.len());
        for (index, embedding) in embeddings {
            let Embedding::Pooled(values) = embedding else {
                panic!("oxide must return pooled embeddings");
            };
            rows.push((index, values.into_iter().map(f32::to_bits).collect()));
        }
        rows.sort_unstable_by_key(|(index, _)| *index);
        rows
    }
    fn assert_pooled_bits(
        actual: &[(usize, Vec<u32>)],
        expected: &[(usize, Vec<u32>)],
        hidden: usize,
    ) {
        assert_eq!(actual.len(), expected.len());
        for ((actual_index, actual_row), (expected_index, expected_row)) in
            actual.iter().zip(expected)
        {
            assert_eq!(actual_index, expected_index);
            assert_eq!(actual_row.len(), hidden);
            assert_eq!(actual_row, expected_row);
            assert!(actual_row
                .iter()
                .all(|bits| f32::from_bits(*bits).is_finite()));
        }
    }
    #[test]
    fn rejects_bad_ragged_boundaries_and_lookup_ids_before_device_access() {
        let c = config();
        assert_eq!(validate(&batch(), &c).unwrap(), 2);
        for lengths in [
            vec![],
            vec![0],
            vec![0, 2, 1, 3],
            vec![0, 0, 3],
            vec![1, 3],
            vec![0, 4],
        ] {
            let mut b = batch();
            b.cumulative_seq_lengths = lengths;
            assert!(validate(&b, &c).is_err());
        }
        let mut b = batch();
        b.input_ids = vec![4, 2, 3];
        assert!(validate(&b, &c).is_err());
        let mut b = batch();
        b.pooled_indices = vec![2];
        assert!(validate(&b, &c).is_err());
        let mut b = batch();
        b.position_ids = vec![8, 0, 1];
        assert!(validate(&b, &c).is_err());
        let mut b = batch();
        b.token_type_ids = vec![2, 0, 0];
        assert!(validate(&b, &c).is_err());
        let mut b = batch();
        b.pooled_indices.clear();
        // Empty pooled output is valid and embed returns without GPU work.
        assert!(validate(&b, &c).is_ok());
        let mut b = batch();
        b.raw_indices = vec![0];
        assert!(validate(&b, &c).is_ok());
        b.raw_indices = vec![2];
        assert!(validate(&b, &c).is_err());
    }
    #[test]
    fn arenas_fit_changing_shapes_without_length_in_the_key() {
        let mut b = batch();
        assert_eq!(work_key(&b).unwrap(), (32, 4, 2));
        b.cumulative_seq_lengths = vec![0, 2, 3];
        assert_eq!(work_key(&b).unwrap(), (32, 4, 2));
        let capacity = (256, 16, 16);
        for request in [(32, 2, 1), (64, 8, 4), capacity] {
            assert!(capacity_fits(capacity, request, 1024));
        }
        for request in [(512, 2, 1), (32, 32, 1), (32, 2, 32)] {
            assert!(!capacity_fits(capacity, request, 1024));
        }
        assert!(!capacity_fits(capacity, (32, 2, 1), cache::BYTE_BUDGET + 1));
        assert!(capacity_fits(capacity, capacity, cache::BYTE_BUDGET + 1));
        assert!(capacity_fits(
            capacity,
            (capacity.0, 2, 1),
            cache::BYTE_BUDGET + 1
        ));
        assert_eq!(bucketed(33, MIN_TOKEN_CAPACITY).unwrap(), 64);
        assert_eq!(bucketed(65, MIN_TOKEN_CAPACITY).unwrap(), 128);
    }
    #[test]
    fn compact_metadata_does_not_touch_the_unused_arena_tail() {
        let mut storage = vec![u32::MAX; 256];
        let n = input_elements(3, 3, 1).unwrap();
        pack_input(
            &mut storage[..n],
            3,
            &[7, 8, 9],
            &[0, 1, 2],
            &[0, 0, 0],
            &[0, 1, 3],
            &[1],
        )
        .unwrap();
        assert_eq!(&storage[..n], &[7, 8, 9, 0, 1, 2, 0, 0, 0, 0, 1, 3, 1]);
        assert!(storage[n..].iter().all(|&x| x == u32::MAX));
    }
    #[test]
    fn stages_actual_metadata_into_padded_sections() {
        let tokens = 64;
        let boundaries = [0, 1, 3];
        let selected = [1];
        let mut out = vec![u32::MAX; tokens * 3 + boundaries.len() + selected.len()];
        pack_input(
            &mut out,
            tokens,
            &[7, 8, 9],
            &[0, 1],
            &[1],
            &boundaries,
            &selected,
        )
        .unwrap();
        assert_eq!(&out[..3], &[7, 8, 9]);
        assert!(out[3..tokens].iter().all(|&value| value == 0));
        assert_eq!(&out[tokens..tokens + 2], &[0, 1]);
        assert!(out[tokens + 2..tokens * 2].iter().all(|&value| value == 0));
        assert_eq!(&out[tokens * 2..tokens * 2 + 1], &[1]);
        assert!(out[tokens * 2 + 1..tokens * 3]
            .iter()
            .all(|&value| value == 0));
        assert_eq!(&out[tokens * 3..], &[0, 1, 3, 1]);
        let mut short = vec![0; tokens * 3 + boundaries.len()];
        assert!(pack_input(
            &mut short,
            tokens,
            &[7, 8, 9],
            &[0, 1],
            &[1],
            &boundaries,
            &selected,
        )
        .is_err());
        let mut long = vec![0; tokens * 3 + boundaries.len() + selected.len()];
        assert!(pack_input(
            &mut long,
            tokens,
            &vec![7; tokens + 1],
            &[0, 1],
            &[1],
            &boundaries,
            &selected,
        )
        .is_err());
    }
    #[test]
    fn allocation_rounding_does_not_expand_linear_rows() {
        let b = batch();
        let c = config();
        let capacity = work_key(&b).unwrap();
        assert_eq!(capacity.0, 32);
        let plans = LinearPlans::new(b.input_ids.len(), &c).unwrap();
        for plan in [
            &plans.qkv,
            &plans.attention,
            &plans.intermediate,
            &plans.output,
        ] {
            assert_eq!(plan.arguments(0, 0, 0).rows, 3);
            assert_eq!(plan.input_layout().shape()[0], 3);
            assert_eq!(plan.output_layout().shape()[0], 3);
        }
    }
    #[test]
    #[ignore = "Requires CUDA, MiniLM INFINITY_OXIDE_TEST_MODEL and INFINITY_OXIDE_KERNEL_MODULE"]
    fn rejected_shape_preflight_preserves_arena_and_recovers_same_or_previous_shape() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("set INFINITY_OXIDE_TEST_MODEL to the real MiniLM checkpoint");
        let new_backend = || {
            OxideBackend::new(
                Path::new(&path),
                "bfloat16",
                ModelType::Embedding(Pool::Mean),
                0,
            )
            .unwrap()
        };
        let a = batch();
        let b = repeated_sequences(1, 65);
        let reference = new_backend();
        let hidden = reference.config.hidden_size();
        assert_eq!(
            hidden, 384,
            "this regression requires the qualified MiniLM shape"
        );
        let expected_a = pooled_bits(reference.embed(a.clone()).unwrap());
        let expected_b = pooled_bits(reference.embed(b.clone()).unwrap());
        drop(reference);
        let backend = new_backend();

        // First retry the rejected B; then reject B again and immediately
        // return to A, which would expose a stale successful-A marker.
        for (successful_checks, retry_b) in [(1usize, true), (2, false)] {
            let actual_a = pooled_bits(backend.embed(a.clone()).unwrap());
            assert_pooled_bits(&actual_a, &expected_a, hidden);
            let before = {
                let engine = backend.engine.lock().unwrap();
                let work = match engine.workspace.as_ref().unwrap() {
                    Workspace::Encoder(work) => work,
                    Workspace::Decoder(_) => panic!("MiniLM must use the encoder workspace"),
                };
                engine
                    .lt
                    .reject_shape_once_after(b.input_ids.len(), successful_checks);
                (work.x.cu_deviceptr(), work.key(), backend.io.pinned_bytes())
            };
            let failed = backend.submit_embed(b.clone());
            assert!(matches!(failed, Err(BackendError::Inference(_))));
            {
                let engine = backend.engine.lock().unwrap();
                assert!(!engine.failed);
                assert!(!engine.io.poisoned());
                let work = match engine.workspace.as_ref().unwrap() {
                    Workspace::Encoder(work) => work,
                    Workspace::Decoder(_) => panic!("MiniLM must use the encoder workspace"),
                };
                assert_eq!(
                    (work.x.cu_deviceptr(), work.key(), backend.io.pinned_bytes()),
                    before,
                    "preflight rejection must precede arena growth/zeroing/staging"
                );
                let model = match &*engine.model {
                    ExecutableModel::Encoder(model) => model,
                    ExecutableModel::Decoder(_) => panic!("MiniLM must use encoder plans"),
                };
                assert_eq!(model.plans.rows, None);
                assert_eq!(model.plans.qkv.logical_rows(), b.input_ids.len() as i32);
                assert_eq!(
                    model.plans.attention.logical_rows(),
                    if successful_checks == 1 {
                        a.input_ids.len()
                    } else {
                        b.input_ids.len()
                    } as i32
                );
                assert_eq!(
                    model.plans.intermediate.logical_rows(),
                    a.input_ids.len() as i32
                );
            }
            assert_eq!(
                backend.io.active_leases(),
                0,
                "rejected submission returns its unused lease"
            );
            backend.health().unwrap();
            let (request, expected) = if retry_b {
                (&b, &expected_b)
            } else {
                (&a, &expected_a)
            };
            let recovered = pooled_bits(backend.embed(request.clone()).unwrap());
            assert_pooled_bits(&recovered, expected, hidden);
            backend.health().unwrap();
        }
        let final_b = pooled_bits(backend.embed(b).unwrap());
        assert_pooled_bits(&final_b, &expected_b, hidden);
        backend.health().unwrap();
    }

    #[test]
    #[ignore = "Requires an NVIDIA GPU, oxide kernels, and INFINITY_OXIDE_TEST_MODEL"]
    fn deferred_slots_complete_out_of_order_and_recover_after_drop() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_MODEL").expect(
            "Set INFINITY_OXIDE_TEST_MODEL to a BERT checkpoint supporting 256-token sequences",
        );
        let new_backend = || {
            OxideBackend::new(
                std::path::Path::new(&path),
                "bfloat16",
                ModelType::Embedding(Pool::Mean),
                0,
            )
            .unwrap()
        };
        let backend = new_backend();
        let hidden = backend.config.hidden_size();
        let first = batch();
        let second = same_bucket_batch();
        let changed = changed_shape_batch();
        // Same capacity bucket as `first`, different exact token/sequence counts.
        assert_eq!(work_key(&first).unwrap(), work_key(&second).unwrap());
        assert_ne!(work_key(&first).unwrap(), work_key(&changed).unwrap());
        let expected_first = pooled_bits(backend.embed(first.clone()).unwrap());
        let expected_second = pooled_bits(backend.embed(second.clone()).unwrap());
        let expected_changed = pooled_bits(backend.embed(changed.clone()).unwrap());
        assert_pooled_bits(&expected_first, &expected_first, hidden);
        assert_pooled_bits(&expected_second, &expected_second, hidden);
        assert_pooled_bits(&expected_changed, &expected_changed, hidden);
        assert_eq!(expected_first.len(), first.pooled_indices.len());
        assert_eq!(expected_second.len(), second.pooled_indices.len());
        assert_eq!(expected_changed.len(), changed.pooled_indices.len());

        let held_first = backend.submit_embed(first.clone()).unwrap();
        let held_second = backend.submit_embed(second.clone()).unwrap();
        assert!(matches!(&held_first, Inference::Deferred(_)));
        assert!(matches!(&held_second, Inference::Deferred(_)));
        let actual_second = pooled_bits(held_second.complete().unwrap());
        let actual_first = pooled_bits(held_first.complete().unwrap());
        assert_pooled_bits(&actual_second, &expected_second, hidden);
        assert_pooled_bits(&actual_first, &expected_first, hidden);

        let held_small = backend.submit_embed(first.clone()).unwrap();
        let held_changed = backend.submit_embed(changed.clone()).unwrap();
        let actual_changed = pooled_bits(held_changed.complete().unwrap());
        let actual_small = pooled_bits(held_small.complete().unwrap());
        assert_pooled_bits(&actual_changed, &expected_changed, hidden);
        assert_pooled_bits(&actual_small, &expected_first, hidden);

        drop(backend.submit_embed(changed.clone()).unwrap());
        let recovered = pooled_bits(backend.submit_embed(changed).unwrap().complete().unwrap());
        assert_pooled_bits(&recovered, &expected_changed, hidden);
        let recovered_small = pooled_bits(backend.submit_embed(first).unwrap().complete().unwrap());
        assert_pooled_bits(&recovered_small, &expected_first, hidden);

        let started = Instant::now();
        let single_first = backend.submit_embed(batch()).unwrap();
        let depth1_first_submit = started.elapsed();
        let started = Instant::now();
        let single_first = pooled_bits(single_first.complete().unwrap());
        let depth1_first_complete = started.elapsed();
        let started = Instant::now();
        let single_second = backend.submit_embed(same_bucket_batch()).unwrap();
        let depth1_second_submit = started.elapsed();
        let started = Instant::now();
        let single_second = pooled_bits(single_second.complete().unwrap());
        let depth1_second_complete = started.elapsed();
        assert_pooled_bits(&single_first, &expected_first, hidden);
        assert_pooled_bits(&single_second, &expected_second, hidden);

        let started = Instant::now();
        let depth2_first = backend.submit_embed(batch()).unwrap();
        let depth2_first_submit = started.elapsed();
        let started = Instant::now();
        let depth2_second = backend.submit_embed(same_bucket_batch()).unwrap();
        let depth2_second_submit = started.elapsed();
        let started = Instant::now();
        let depth2_second = pooled_bits(depth2_second.complete().unwrap());
        let depth2_second_complete = started.elapsed();
        let started = Instant::now();
        let depth2_first = pooled_bits(depth2_first.complete().unwrap());
        let depth2_first_complete = started.elapsed();
        assert_pooled_bits(&depth2_second, &expected_second, hidden);
        assert_pooled_bits(&depth2_first, &expected_first, hidden);

        eprintln!(
            "oxide depth1 submit_first_us={} complete_first_us={} \
             submit_second_us={} complete_second_us={}",
            depth1_first_submit.as_micros(),
            depth1_first_complete.as_micros(),
            depth1_second_submit.as_micros(),
            depth1_second_complete.as_micros()
        );
        eprintln!(
            "oxide depth2 submit_first_us={} submit_second_us={} \
             complete_second_us={} complete_first_us={}",
            depth2_first_submit.as_micros(),
            depth2_second_submit.as_micros(),
            depth2_second_complete.as_micros(),
            depth2_first_complete.as_micros()
        );
        drop(backend);

        let pending = {
            let dropped_backend = new_backend();
            dropped_backend.submit_embed(batch()).unwrap()
        };
        let after_backend_drop = pooled_bits(pending.complete().unwrap());
        assert_pooled_bits(&after_backend_drop, &expected_first, hidden);

        let small = repeated_sequences(64, 256);
        let large = repeated_sequences(128, 256);
        let reference = new_backend();
        let small_key = work_key(&small).unwrap();
        let large_key = work_key(&large).unwrap();
        let encoder_config = match &reference.config {
            LoadedConfig::Encoder(config) => config,
            LoadedConfig::Decoder(_) => panic!("this test requires an encoder checkpoint"),
        };
        let small_bytes = Work::allocation_bytes(small_key, encoder_config).unwrap();
        let large_bytes = Work::allocation_bytes(large_key, encoder_config).unwrap();
        assert!(small_bytes <= cache::BYTE_BUDGET);
        assert!(
            large_bytes > cache::BYTE_BUDGET,
            "This test requires a checkpoint whose 32768-token GPU arena exceeds 256 MiB"
        );
        let expected_small = pooled_bits(reference.embed(small.clone()).unwrap());
        let expected_large = pooled_bits(reference.embed(large.clone()).unwrap());
        drop(reference);

        let fresh = new_backend();
        let arena = || {
            let engine = fresh.engine.lock().unwrap();
            let Workspace::Encoder(work) = engine.workspace.as_ref().unwrap() else {
                panic!("this test requires an encoder workspace");
            };
            (work.key(), work.bytes, work.input.cu_deviceptr())
        };
        let held_small = fresh.submit_embed(small).unwrap();
        assert_eq!((arena().0, arena().1), (small_key, small_bytes));
        let held_large = fresh.submit_embed(large.clone()).unwrap();
        assert_eq!((arena().0, arena().1), (large_key, large_bytes));
        let actual_large = pooled_bits(held_large.complete().unwrap());
        let actual_small = pooled_bits(held_small.complete().unwrap());
        assert_pooled_bits(&actual_large, &expected_large, hidden);
        assert_pooled_bits(&actual_small, &expected_small, hidden);

        // Two retained results share the same oversized device arena; only
        // their pinned I/O slots differ, and admission must not reject either.
        let shared = arena();
        let first_large = fresh.submit_embed(large.clone()).unwrap();
        let second_large = fresh.submit_embed(large.clone()).unwrap();
        assert_eq!(arena(), shared);
        let second_large = pooled_bits(second_large.complete().unwrap());
        let first_large = pooled_bits(first_large.complete().unwrap());
        assert_pooled_bits(&second_large, &expected_large, hidden);
        assert_pooled_bits(&first_large, &expected_large, hidden);

        // Selecting fewer rows changes only metadata capacity. The oversized
        // token scratch stays shared, and the result contains no stale rows.
        let mut narrowed = large;
        narrowed.pooled_indices.truncate(1);
        let expected_narrowed = expected_large.into_iter().take(1).collect::<Vec<_>>();
        let actual_narrowed = pooled_bits(fresh.embed(narrowed).unwrap());
        assert_eq!(arena(), shared);
        assert_pooled_bits(&actual_narrowed, &expected_narrowed, hidden);
    }
    #[test]
    fn oversized_gpu_scratch_is_shared_while_host_staging_stays_bounded() {
        let budget = cache::BYTE_BUDGET;
        assert!(!needs_exclusive_staging(budget + 1, budget));
        assert!(!needs_exclusive_staging(budget * 4, budget / 8));
        assert!(needs_exclusive_staging(budget + 1, budget + 1));
        assert!(!needs_exclusive_staging(budget / 2, budget / 2));
        assert!(needs_exclusive_staging(budget / 2, budget / 2 + 1));
    }
    #[test]
    fn dense_capacity_tracks_pooled_selections_not_raw_rows_or_backbone_key() {
        let bytes = |capacity: usize| Ok::<usize, BackendError>(capacity * 100);
        assert_eq!(plan_dense_workspace(0, None, bytes).unwrap(), (None, 0));
        assert_eq!(
            plan_dense_workspace(3, None, bytes).unwrap(),
            (Some(4), 400)
        );
        // Raw growth (R) never changes the pooled capacity P=3.
        assert_eq!(
            plan_dense_workspace(3, Some((4, 400)), bytes).unwrap(),
            (None, 400)
        );
        // Raw-only requests retain an existing allocation without growth.
        assert_eq!(
            plan_dense_workspace(0, Some((4, 400)), bytes).unwrap(),
            (None, 400)
        );
        // Only pooled capacity growth reallocates Dense scratch.
        assert_eq!(
            plan_dense_workspace(6, Some((4, 400)), bytes).unwrap(),
            (Some(8), 800)
        );
        assert_eq!(
            plan_dense_workspace(2, Some((8, 800)), bytes).unwrap(),
            (None, 800)
        );
    }

    #[test]
    fn gpu_and_both_host_slots_are_charged_once() {
        let c = config();
        let key = (32, 4, 2);
        let gpu = Work::allocation_bytes(key, &c).unwrap();
        let pin = staging::pinned_bytes(key, key.2 * c.hidden_size).unwrap();
        let input = staging::input_capacity(key).unwrap();
        let output = key.2 * c.hidden_size;
        // Old Work bytes included one host input/output pair. The split charges
        // that pair to the slot instead; two slots add one more pair.
        let old = input * 8 + output * 8 + 32 * 4 * 18 + 32 * 8 * 2;
        assert_eq!(gpu + pin, old);
        assert_eq!(gpu + 2 * pin, old + pin);
    }
    #[test]
    fn rejects_bucket_and_capacity_overflow() {
        assert!(bucketed(usize::MAX, MIN_TOKEN_CAPACITY).is_err());
        let c = config();
        let oversized = u32::MAX as usize + 1;
        assert!(Work::allocation_bytes((oversized, 2, 1), &c).is_err());
        assert!(Work::allocation_bytes((usize::MAX, 2, 1), &c).is_err());
    }
    #[derive(serde::Deserialize)]
    struct TraceBatch {
        input_ids: Vec<u32>,
        position_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        cumulative_seq_lengths: Vec<u32>,
        #[serde(default)]
        max_length: u32,
        #[serde(default)]
        pooled_indices: Vec<u32>,
        #[serde(default)]
        raw_indices: Vec<u32>,
        #[serde(default)]
        compact_input_ids: Option<Vec<u32>>,
        #[serde(default)]
        compact_position_ids: Option<Vec<u32>>,
        #[serde(default)]
        scatter_unfold: Option<Vec<u32>>,
        #[serde(default)]
        fold_gather: Option<Vec<u32>>,
        #[serde(default)]
        tokens: Vec<String>,
        #[serde(default)]
        offsets: Vec<(usize, usize)>,
        #[serde(default)]
        multimodal: Vec<serde_json::Value>,
    }
    impl TraceBatch {
        fn into_batch(self) -> Batch {
            assert!(
                self.multimodal.is_empty(),
                "raw trace requires a text-only batch"
            );
            Batch {
                multimodal: vec![],
                input_ids: self.input_ids,
                position_ids: self.position_ids,
                token_type_ids: self.token_type_ids,
                cumulative_seq_lengths: self.cumulative_seq_lengths,
                max_length: self.max_length,
                pooled_indices: self.pooled_indices,
                raw_indices: self.raw_indices,
                compact_input_ids: self.compact_input_ids,
                compact_position_ids: self.compact_position_ids,
                scatter_unfold: self.scatter_unfold,
                fold_gather: self.fold_gather,
                tokens: self.tokens,
                offsets: self.offsets,
            }
        }
    }
    fn trace_pool() -> Pool {
        match std::env::var("INFINITY_OXIDE_TRACE_POOL")
            .unwrap_or_else(|_| "mean".into())
            .as_str()
        {
            "mean" => Pool::Mean,
            "cls" => Pool::Cls,
            "last_token" | "last-token" => Pool::LastToken,
            other => {
                panic!("INFINITY_OXIDE_TRACE_POOL must be mean, cls, or last_token, got {other}")
            }
        }
    }
    #[test]
    #[ignore = "Requires CUDA plus INFINITY_OXIDE_TEST_MODEL, INFINITY_OXIDE_TRACE_DIR, and INFINITY_OXIDE_TRACE_BATCH_JSON"]
    fn trace_encoder_batch() {
        let model = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("Set INFINITY_OXIDE_TEST_MODEL to a BERT checkpoint");
        let batch_path = std::env::var_os("INFINITY_OXIDE_TRACE_BATCH_JSON")
            .expect("Set INFINITY_OXIDE_TRACE_BATCH_JSON to a batch JSON file");
        let trace_dir = std::env::var_os("INFINITY_OXIDE_TRACE_DIR")
            .expect("Set INFINITY_OXIDE_TRACE_DIR for encoder trace output");
        let input: TraceBatch =
            serde_json::from_slice(&std::fs::read(batch_path).unwrap()).unwrap();
        let max_length = input
            .cumulative_seq_lengths
            .windows(2)
            .map(|pair| pair[1].saturating_sub(pair[0]))
            .max()
            .unwrap_or(0);
        let batch = Batch {
            multimodal: vec![],
            input_ids: input.input_ids,
            position_ids: input.position_ids,
            token_type_ids: input.token_type_ids,
            cumulative_seq_lengths: input.cumulative_seq_lengths,
            max_length,
            pooled_indices: input.pooled_indices,
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let backend = OxideBackend::new(
            std::path::Path::new(&model),
            "bfloat16",
            ModelType::Embedding(trace_pool()),
            0,
        )
        .unwrap();
        let hidden = backend.config.hidden_size();
        let embeddings = backend.embed(batch).unwrap();
        trace::write_pooled(std::path::Path::new(&trace_dir), &embeddings, hidden);
    }
    #[test]
    #[ignore = "Requires CUDA plus INFINITY_OXIDE_TEST_MODEL, INFINITY_OXIDE_TRACE_DIR, and INFINITY_OXIDE_TRACE_BATCH_JSON"]
    fn trace_gte_raw_batch() {
        let model = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("Set INFINITY_OXIDE_TEST_MODEL to a GTE checkpoint");
        let batch_path = std::env::var_os("INFINITY_OXIDE_TRACE_BATCH_JSON")
            .expect("Set INFINITY_OXIDE_TRACE_BATCH_JSON to a batch JSON file");
        let trace_dir = std::env::var_os("INFINITY_OXIDE_TRACE_DIR")
            .expect("Set INFINITY_OXIDE_TRACE_DIR for raw trace output");
        let bytes = std::fs::read(batch_path).unwrap();
        let physical_batch: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let input: TraceBatch = serde_json::from_slice(&bytes).unwrap();
        let batch = input.into_batch();
        let backend = OxideBackend::new(
            std::path::Path::new(&model),
            "bfloat16",
            ModelType::Embedding(trace_pool()),
            0,
        )
        .unwrap();
        let hidden = backend.config.hidden_size();
        assert_eq!(
            hidden, 768,
            "the GTE raw trace requires the qualified hidden size"
        );
        let embeddings = backend.embed(batch.clone()).unwrap();
        trace::write_outputs(
            std::path::Path::new(&trace_dir),
            &batch,
            &embeddings,
            backend.output_width,
            hidden,
            physical_batch,
        );
    }
    #[test]
    fn raw_and_pooled_trace_outputs_use_actual_widths_and_validate() {
        let mut request = batch();
        request.pooled_indices = vec![0];
        request.raw_indices = vec![1];
        let mut result = Embeddings::default();
        result.insert(0, Embedding::Pooled(vec![1.25, -2.5, 3.75]));
        result.insert(
            1,
            Embedding::All(vec![vec![4.0, 5.0, 6.0, 7.0], vec![8.0, 9.0, 10.0, 11.0]]),
        );
        let physical = serde_json::json!({
            "input_ids": request.input_ids.clone(),
            "max_length": request.max_length,
            "raw_indices": request.raw_indices.clone(),
            "extra": {"preserved": true},
        });
        let outputs = trace::outputs_json(&request, &result, 3, 4, physical.clone()).unwrap();
        assert_eq!(outputs["physical_batch"], physical);
        let entries = outputs["outputs"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["index"], 0);
        assert_eq!(entries[0]["kind"], "pooled");
        assert_eq!(entries[0]["shape"], serde_json::json!([1, 3]));
        assert_eq!(entries[0]["values"], serde_json::json!([1.25, -2.5, 3.75]));
        assert_eq!(entries[1]["index"], 1);
        assert_eq!(entries[1]["kind"], "raw");
        assert_eq!(entries[1]["shape"], serde_json::json!([2, 4]));
        assert_eq!(
            entries[1]["values"],
            serde_json::json!([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        );
        assert!(trace::validate_outputs(&request, &result, 3, 4).is_ok());
        assert!(trace::validate_outputs(&request, &result, 4, 4).is_err());
        let mut missing = Embeddings::default();
        missing.insert(0, Embedding::Pooled(vec![1.25, -2.5, 3.75]));
        assert!(trace::validate_outputs(&request, &missing, 3, 4).is_err());
        let mut short = Embeddings::default();
        short.insert(0, Embedding::Pooled(vec![1.25, -2.5, 3.75]));
        short.insert(1, Embedding::All(vec![vec![4.0, 5.0, 6.0, 7.0]]));
        assert!(trace::validate_outputs(&request, &short, 3, 4).is_err());
        let mut out_of_bounds = request.clone();
        out_of_bounds.raw_indices = vec![2];
        assert!(trace::validate_outputs(&out_of_bounds, &result, 3, 4).is_err());
        let mut duplicate = request.clone();
        duplicate.raw_indices = vec![1, 1];
        assert!(trace::validate_outputs(&duplicate, &result, 3, 4).is_ok());
        let mut duplicate_pooled = request;
        duplicate_pooled.pooled_indices = vec![0, 0];
        assert!(trace::validate_outputs(&duplicate_pooled, &result, 3, 4).is_ok());
    }
    #[test]
    fn raw_trace_canonical_map_deduplicates_and_overrides_pooled() {
        let mut request = changed_shape_batch();
        request.pooled_indices = vec![2];
        request.raw_indices = vec![1, 0, 2, 1];
        let raw = |index: usize, rows: usize| {
            Embedding::All(
                (0..rows)
                    .map(|row| vec![(index * 100 + row) as f32; 4])
                    .collect(),
            )
        };
        let mut result = Embeddings::default();
        for index in 0..3 {
            result.insert(index, raw(index, 10));
        }
        assert!(trace::validate_outputs(&request, &result, 3, 4).is_ok());
        let physical = serde_json::json!({
            "pooled_indices": request.pooled_indices.clone(),
            "raw_indices": request.raw_indices.clone(),
        });
        let outputs = trace::outputs_json(&request, &result, 3, 4, physical.clone()).unwrap();
        assert_eq!(outputs["physical_batch"], physical);
        let entries = outputs["outputs"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        for (entry, index) in entries.iter().zip(0..3usize) {
            assert_eq!(entry["index"], index);
            assert_eq!(entry["kind"], "raw");
            assert_eq!(entry["shape"], serde_json::json!([10, 4]));
            assert_eq!(entry["values"].as_array().unwrap().len(), 40);
        }

        let mut pooled_request = request.clone();
        pooled_request.pooled_indices = vec![2, 2];
        pooled_request.raw_indices.clear();
        let mut pooled = Embeddings::default();
        pooled.insert(2, Embedding::Pooled(vec![1.0, 2.0, 3.0]));
        assert!(trace::validate_outputs(&pooled_request, &pooled, 3, 4).is_ok());

        let mut missing = Embeddings::default();
        for index in 0..2 {
            missing.insert(index, raw(index, 10));
        }
        assert!(trace::validate_outputs(&request, &missing, 3, 4).is_err());

        let mut extra = Embeddings::default();
        for index in 0..4 {
            extra.insert(index, raw(index, 10));
        }
        assert!(trace::validate_outputs(&request, &extra, 3, 4).is_err());

        let mut wrong_shape = Embeddings::default();
        for index in 0..3 {
            wrong_shape.insert(index, raw(index, if index == 1 { 9 } else { 10 }));
        }
        assert!(trace::validate_outputs(&request, &wrong_shape, 3, 4).is_err());
    }
    #[test]
    #[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_MODEL (MiniLM) and INFINITY_OXIDE_KERNEL_MODULE"]
    fn simt_attention_rejects_oversized_sequence_without_poisoning() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("set INFINITY_OXIDE_TEST_MODEL to a MiniLM checkpoint directory");
        let attention_env = infinity_v3_config::infinity_env!("OXIDE_ATTENTION");
        let previous = std::env::var_os(attention_env);
        std::env::set_var(attention_env, "simt");
        let mut backend = OxideBackend::new(
            std::path::Path::new(&path),
            "bfloat16",
            ModelType::Embedding(Pool::Mean),
            0,
        )
        .unwrap();
        match previous {
            Some(value) => std::env::set_var(attention_env, value),
            None => std::env::remove_var(attention_env),
        }
        let hidden = backend.config.hidden_size();
        let limit = {
            let engine = backend.engine.lock().unwrap();
            let ExecutableModel::Encoder(model) = &*engine.model else {
                panic!("this test requires an encoder checkpoint");
            };
            let bytes = model
                .kernels
                .attention
                .max_dynamic_shared_memory_bytes()
                .unwrap() as usize;
            let limit = bytes / std::mem::size_of::<f32>();
            assert_eq!(model.kernels.attention_length_limit, Some(limit));
            limit
        };
        assert!(limit > 0);
        assert_eq!(backend.attention_length_limit, Some(limit));
        let valid = batch();
        let expected = pooled_bits(backend.embed(valid.clone()).unwrap());
        assert_pooled_bits(&expected, &expected, hidden);

        let oversized_length = limit.checked_add(1).unwrap();
        let vocab_size = match &backend.config {
            LoadedConfig::Encoder(config) => config.vocab_size,
            LoadedConfig::Decoder(_) => panic!("this test requires an encoder checkpoint"),
        };
        assert!(vocab_size > 0);
        match &mut backend.config {
            LoadedConfig::Encoder(config) => {
                config.max_position_embeddings =
                    config.max_position_embeddings.max(oversized_length);
            }
            LoadedConfig::Decoder(_) => unreachable!(),
        }
        let boundary = u32::try_from(oversized_length).unwrap();
        let oversized = Batch {
            input_ids: vec![0; oversized_length],
            position_ids: vec![0; oversized_length],
            token_type_ids: vec![0; oversized_length],
            cumulative_seq_lengths: vec![0, boundary],
            pooled_indices: vec![0],
            max_length: boundary,
            multimodal: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let error = match backend.embed(oversized) {
            Ok(_) => panic!("oversized SIMT attention sequence was admitted"),
            Err(error) => error,
        };
        assert!(matches!(&error, BackendError::Inference(_)));
        assert!(error.to_string().contains("SIMT attention"), "{error}");
        backend.health().unwrap();
        let actual = pooled_bits(backend.embed(valid).unwrap());
        assert_pooled_bits(&actual, &expected, hidden);
    }

    #[test]
    #[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_LLAMA_MODEL (TinyLlama checkpoint), INFINITY_OXIDE_KERNEL_MODULE and FA4_ROOT"]
    fn llama_tiny_deferred_aba_drop_subset_and_shape() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_LLAMA_MODEL")
            .expect("set INFINITY_OXIDE_TEST_LLAMA_MODEL to the TinyLlama checkpoint directory");
        let backend = OxideBackend::new(
            std::path::Path::new(&path),
            "bfloat16",
            ModelType::Embedding(Pool::LastToken),
            0,
        )
        .unwrap();
        assert_eq!(backend.config.hidden_size(), 2048);
        match &backend.config {
            LoadedConfig::Decoder(config) => {
                assert_eq!(config.family, DecoderFamily::Llama);
                assert_eq!(config.intermediate_size, 5632);
                assert_eq!(config.num_attention_heads, 32);
                assert_eq!(config.num_key_value_heads, 4);
                assert_eq!(config.head_dim, 64);
                assert!(config.is_causal);
            }
            LoadedConfig::Encoder(_) => panic!("TinyLlama must load as a decoder"),
        }

        let a = decoder_batch(&[3, 5, 2]);
        let b = decoder_batch(&[7, 11]);
        let expected_a = pooled_bits(backend.embed(a.clone()).unwrap());
        let expected_b = pooled_bits(backend.embed(b.clone()).unwrap());
        assert_eq!(expected_a.len(), 3);
        assert_eq!(expected_b.len(), 2);
        for (_, row) in expected_a.iter().chain(&expected_b) {
            assert_eq!(row.len(), 2048);
            assert!(row.iter().all(|bits| f32::from_bits(*bits).is_finite()));
        }

        // ABA on the public path before the deferred slots are exercised.
        let repeated_a = pooled_bits(backend.embed(a.clone()).unwrap());
        assert_pooled_bits(&repeated_a, &expected_a, 2048);

        // At most two leases: both submissions stay deferred, then complete B
        // before A to force independent retained workspaces.
        let held_a = backend.submit_embed(a.clone()).unwrap();
        let held_b = backend.submit_embed(b.clone()).unwrap();
        assert!(matches!(&held_a, Inference::Deferred(_)));
        assert!(matches!(&held_b, Inference::Deferred(_)));
        let actual_b = pooled_bits(held_b.complete().unwrap());
        let actual_a = pooled_bits(held_a.complete().unwrap());
        assert_pooled_bits(&actual_b, &expected_b, 2048);
        assert_pooled_bits(&actual_a, &expected_a, 2048);

        // Dropping one deferred lease releases it without disturbing the other
        // slot, and the dropped shape remains reproducible.
        drop(backend.submit_embed(b.clone()).unwrap());
        let recovered_b = pooled_bits(backend.submit_embed(b.clone()).unwrap().complete().unwrap());
        assert_pooled_bits(&recovered_b, &expected_b, 2048);

        // Shape change plus canonical all-B readout with subset selection.
        let base = decoder_batch(&[4, 6, 2, 5]);
        let mut subset = base.clone();
        subset.pooled_indices = vec![0, 2];
        let expected_base = pooled_bits(backend.embed(base.clone()).unwrap());
        let expected_subset = pooled_bits(backend.embed(subset.clone()).unwrap());
        let subset_from_base = expected_base
            .iter()
            .filter(|(index, _)| subset.pooled_indices.contains(&(*index as u32)))
            .cloned()
            .collect::<Vec<_>>();
        assert_pooled_bits(&expected_subset, &subset_from_base, 2048);

        let held_base = backend.submit_embed(base.clone()).unwrap();
        let held_subset = backend.submit_embed(subset.clone()).unwrap();
        assert!(matches!(&held_base, Inference::Deferred(_)));
        assert!(matches!(&held_subset, Inference::Deferred(_)));
        let actual_subset = pooled_bits(held_subset.complete().unwrap());
        let held_base_repeat = backend.submit_embed(base.clone()).unwrap();
        assert!(matches!(&held_base_repeat, Inference::Deferred(_)));
        let actual_base_repeat = pooled_bits(held_base_repeat.complete().unwrap());
        let actual_base = pooled_bits(held_base.complete().unwrap());
        assert_pooled_bits(&actual_subset, &expected_subset, 2048);
        assert_pooled_bits(&actual_base_repeat, &expected_base, 2048);
        assert_pooled_bits(&actual_base, &expected_base, 2048);
    }

    #[test]
    #[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_DECODER_MODEL, FA4_ROOT and oxide decoder kernels"]
    fn decoder_gte_deferred_aba_stability_and_drop() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_DECODER_MODEL")
            .expect("set INFINITY_OXIDE_TEST_DECODER_MODEL to the GTE Qwen2 checkpoint");
        let backend = OxideBackend::new(
            std::path::Path::new(&path),
            "bfloat16",
            ModelType::Embedding(Pool::LastToken),
            0,
        )
        .unwrap();
        let hidden = backend.config.hidden_size();
        assert_eq!(hidden, 1536);
        let a = decoder_batch(&[3, 5, 2]);
        let b = decoder_batch(&[7, 11]);
        let expected_a = pooled_bits(backend.embed(a.clone()).unwrap());
        let expected_b = pooled_bits(backend.embed(b.clone()).unwrap());
        assert_eq!(expected_a.len(), a.pooled_indices.len());
        assert_eq!(expected_b.len(), b.pooled_indices.len());
        for (_, row) in expected_a.iter().chain(&expected_b) {
            assert_eq!(row.len(), hidden);
            assert!(row.iter().all(|bits| f32::from_bits(*bits).is_finite()));
        }
        let again = pooled_bits(backend.embed(a.clone()).unwrap());
        assert_pooled_bits(&again, &expected_a, hidden);

        let held_a = backend.submit_embed(a.clone()).unwrap();
        let held_b = backend.submit_embed(b.clone()).unwrap();
        assert!(matches!(&held_a, Inference::Deferred(_)));
        assert!(matches!(&held_b, Inference::Deferred(_)));
        let actual_b = pooled_bits(held_b.complete().unwrap());
        let actual_a = pooled_bits(held_a.complete().unwrap());
        assert_pooled_bits(&actual_b, &expected_b, hidden);
        assert_pooled_bits(&actual_a, &expected_a, hidden);

        drop(backend.submit_embed(b.clone()).unwrap());
        let recovered = pooled_bits(backend.submit_embed(b).unwrap().complete().unwrap());
        assert_pooled_bits(&recovered, &expected_b, hidden);
        let repeat = pooled_bits(backend.embed(a).unwrap());
        assert_pooled_bits(&repeat, &expected_a, hidden);

        // Same T and B, changed request selection: full plans stay T while
        // readout plans stay B (canonical order). Duplicate excess is valid.
        let base = decoder_batch(&[4, 6, 2, 5]);
        let mut subset = base.clone();
        subset.pooled_indices = vec![0, 2];
        let mut duplicate = base.clone();
        duplicate.pooled_indices = vec![2, 0, 2, 1, 0];
        assert_eq!(base.input_ids.len(), subset.input_ids.len());
        assert_eq!(base.cumulative_seq_lengths, subset.cumulative_seq_lengths);
        assert_eq!(base.input_ids.len(), duplicate.input_ids.len());
        assert_eq!(
            base.cumulative_seq_lengths,
            duplicate.cumulative_seq_lengths
        );
        assert_ne!(base.pooled_indices.len(), subset.pooled_indices.len());
        assert!(duplicate.pooled_indices.len() > base.cumulative_seq_lengths.len() - 1);
        let expected_base = pooled_bits(backend.embed(base.clone()).unwrap());
        let expected_subset = pooled_bits(backend.embed(subset.clone()).unwrap());
        let expected_duplicate = pooled_bits(backend.embed(duplicate.clone()).unwrap());
        let subset_from_base = expected_base
            .iter()
            .filter(|(index, _)| subset.pooled_indices.contains(&(*index as u32)))
            .cloned()
            .collect::<Vec<_>>();
        assert_pooled_bits(&expected_subset, &subset_from_base, hidden);
        // Public embeddings are keyed by sequence, so duplicate request rows
        // collapse to the same canonical state. The device cast tests cover
        // the raw S-row duplicate order independently.
        let duplicate_from_base = expected_base
            .iter()
            .filter(|(index, _)| duplicate.pooled_indices.contains(&(*index as u32)))
            .cloned()
            .collect::<Vec<_>>();
        assert_pooled_bits(&expected_duplicate, &duplicate_from_base, hidden);

        // Two slots only: hold base+subset, complete subset, then submit the
        // base repeat while retaining base, and complete both.
        let held_base = backend.submit_embed(base.clone()).unwrap();
        let held_subset = backend.submit_embed(subset.clone()).unwrap();
        assert!(matches!(&held_base, Inference::Deferred(_)));
        assert!(matches!(&held_subset, Inference::Deferred(_)));
        let actual_subset = pooled_bits(held_subset.complete().unwrap());
        let held_base_repeat = backend.submit_embed(base.clone()).unwrap();
        assert!(matches!(&held_base_repeat, Inference::Deferred(_)));
        let actual_base_repeat = pooled_bits(held_base_repeat.complete().unwrap());
        let actual_base = pooled_bits(held_base.complete().unwrap());
        assert_pooled_bits(&actual_subset, &expected_subset, hidden);
        assert_pooled_bits(&actual_base_repeat, &expected_base, hidden);
        assert_pooled_bits(&actual_base, &expected_base, hidden);

        // Duplicate excess in two deferred slots, completed in reverse order.
        let held_duplicate = backend.submit_embed(duplicate.clone()).unwrap();
        let held_base_again = backend.submit_embed(base.clone()).unwrap();
        assert!(matches!(&held_duplicate, Inference::Deferred(_)));
        assert!(matches!(&held_base_again, Inference::Deferred(_)));
        let actual_base_again = pooled_bits(held_base_again.complete().unwrap());
        let actual_duplicate = pooled_bits(held_duplicate.complete().unwrap());
        assert_pooled_bits(&actual_base_again, &expected_base, hidden);
        assert_pooled_bits(&actual_duplicate, &expected_duplicate, hidden);

        drop(backend.submit_embed(duplicate.clone()).unwrap());
        let recovered_duplicate = pooled_bits(
            backend
                .submit_embed(duplicate.clone())
                .unwrap()
                .complete()
                .unwrap(),
        );
        assert_pooled_bits(&recovered_duplicate, &expected_duplicate, hidden);
        let recovered_subset = pooled_bits(
            backend
                .submit_embed(subset.clone())
                .unwrap()
                .complete()
                .unwrap(),
        );
        assert_pooled_bits(&recovered_subset, &expected_subset, hidden);
        let recovered_base = pooled_bits(
            backend
                .submit_embed(base.clone())
                .unwrap()
                .complete()
                .unwrap(),
        );
        assert_pooled_bits(&recovered_base, &expected_base, hidden);
        let repeated_subset = pooled_bits(backend.embed(subset).unwrap());
        assert_pooled_bits(&repeated_subset, &expected_subset, hidden);
        let repeated_base = pooled_bits(backend.embed(base).unwrap());
        assert_pooled_bits(&repeated_base, &expected_base, hidden);
        let repeated_duplicate = pooled_bits(backend.embed(duplicate).unwrap());
        assert_pooled_bits(&repeated_duplicate, &expected_duplicate, hidden);
    }

    #[test]
    #[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_QWEN3_MODEL, FA4_ROOT and oxide Qwen3 kernels"]
    fn qwen3_deferred_aba_drop_subset_and_recovery() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_QWEN3_MODEL")
            .expect("set INFINITY_OXIDE_TEST_QWEN3_MODEL to a Qwen3-Embedding checkpoint");
        let backend = OxideBackend::new(
            std::path::Path::new(&path),
            "bfloat16",
            ModelType::Embedding(Pool::LastToken),
            0,
        )
        .unwrap();
        let (hidden, vocab_size, max_positions) = match &backend.config {
            LoadedConfig::Decoder(config) => {
                assert_eq!(config.family, DecoderFamily::Qwen3);
                assert!(matches!(config.hidden_size, 1024 | 4096));
                match config.hidden_size {
                    1024 => assert_eq!(config.q_width(), 2048),
                    4096 => assert_eq!(config.q_width(), 4096),
                    _ => unreachable!(),
                }
                (
                    config.hidden_size,
                    config.vocab_size,
                    config.max_position_embeddings,
                )
            }
            LoadedConfig::Encoder(_) => panic!("Qwen3 must load as a decoder"),
        };

        let a = decoder_batch(&[3, 5, 2]);
        let b = decoder_batch(&[7, 11]);
        let expected_a = pooled_bits(backend.embed(a.clone()).unwrap());
        let expected_b = pooled_bits(backend.embed(b.clone()).unwrap());
        assert_eq!(expected_a.len(), a.pooled_indices.len());
        assert_eq!(expected_b.len(), b.pooled_indices.len());
        for (_, row) in expected_a.iter().chain(&expected_b) {
            assert_eq!(row.len(), hidden);
            assert!(row.iter().all(|bits| f32::from_bits(*bits).is_finite()));
        }
        let again_a = pooled_bits(backend.embed(a.clone()).unwrap());
        assert_pooled_bits(&again_a, &expected_a, hidden);

        // ABA with B completing out of order before A; both held submissions
        // must be deferred and use at most the two existing slots.
        let held_a = backend.submit_embed(a.clone()).unwrap();
        let held_b = backend.submit_embed(b.clone()).unwrap();
        assert!(matches!(&held_a, Inference::Deferred(_)));
        assert!(matches!(&held_b, Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 2);
        let actual_b = pooled_bits(held_b.complete().unwrap());
        let actual_a = pooled_bits(held_a.complete().unwrap());
        assert_pooled_bits(&actual_b, &expected_b, hidden);
        assert_pooled_bits(&actual_a, &expected_a, hidden);
        backend.health().unwrap();

        // Drop one deferred request while the other is retained, then
        // resubmit the dropped shape and recover both slots.
        let retained_a = backend.submit_embed(a.clone()).unwrap();
        let dropped_b = backend.submit_embed(b.clone()).unwrap();
        assert!(matches!(&retained_a, Inference::Deferred(_)));
        assert!(matches!(&dropped_b, Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 2);
        drop(dropped_b);
        let recovered_b = backend.submit_embed(b.clone()).unwrap();
        assert!(matches!(&recovered_b, Inference::Deferred(_)));
        let recovered_b = pooled_bits(recovered_b.complete().unwrap());
        let retained_a = pooled_bits(retained_a.complete().unwrap());
        assert_pooled_bits(&recovered_b, &expected_b, hidden);
        assert_pooled_bits(&retained_a, &expected_a, hidden);
        backend.health().unwrap();

        // Same token/batch shape, full versus subset [0, 2] readout.
        let base = decoder_batch(&[4, 6, 2, 5]);
        let mut subset = base.clone();
        subset.pooled_indices = vec![0, 2];
        assert_eq!(base.input_ids.len(), subset.input_ids.len());
        assert_eq!(base.cumulative_seq_lengths, subset.cumulative_seq_lengths);
        let expected_base = pooled_bits(backend.embed(base.clone()).unwrap());
        let expected_subset = pooled_bits(backend.embed(subset.clone()).unwrap());
        let subset_from_base = expected_base
            .iter()
            .filter(|(index, _)| subset.pooled_indices.contains(&(*index as u32)))
            .cloned()
            .collect::<Vec<_>>();
        assert_pooled_bits(&expected_subset, &subset_from_base, hidden);

        let held_base = backend.submit_embed(base.clone()).unwrap();
        let held_subset = backend.submit_embed(subset.clone()).unwrap();
        assert!(matches!(&held_base, Inference::Deferred(_)));
        assert!(matches!(&held_subset, Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 2);
        let actual_subset = pooled_bits(held_subset.complete().unwrap());
        let actual_base = pooled_bits(held_base.complete().unwrap());
        assert_pooled_bits(&actual_subset, &expected_subset, hidden);
        assert_pooled_bits(&actual_base, &expected_base, hidden);

        // Existing validation rejects an out-of-vocabulary token and an
        // out-of-range position before any GPU work; valid work then recovers.
        let mut invalid_token = base.clone();
        invalid_token.input_ids[0] = u32::try_from(vocab_size).unwrap();
        assert!(backend.submit_embed(invalid_token).is_err());
        let mut invalid_position = base.clone();
        invalid_position.position_ids[0] = u32::try_from(max_positions).unwrap();
        assert!(backend.submit_embed(invalid_position).is_err());
        backend.health().unwrap();
        let recovered_base = pooled_bits(backend.embed(base).unwrap());
        assert_pooled_bits(&recovered_base, &expected_base, hidden);

        let final_a = pooled_bits(backend.embed(a).unwrap());
        assert_pooled_bits(&final_a, &expected_a, hidden);
        backend.health().unwrap();
    }

    #[test]
    fn only_the_gte_family_selects_the_native_default_down_policy() {
        let gte: BertConfig = serde_json::from_value(serde_json::json!({
            "model_type": "new", "architectures": ["NewModel"],
            "hidden_size": 768, "intermediate_size": 3072,
            "num_attention_heads": 12, "num_hidden_layers": 12,
            "max_position_embeddings": 8192, "vocab_size": 30528,
            "type_vocab_size": 0, "rope_theta": 500000.0,
            "rope_scaling": {"type": "ntk", "factor": 2.0}
        }))
        .unwrap();
        assert_eq!(down_policy(&gte), DownPolicy::NativeDefault);
        assert_eq!(down_policy(&config()), DownPolicy::Existing);
        let roberta: BertConfig = serde_json::from_value(serde_json::json!({
            "model_type": "roberta", "architectures": ["RobertaModel"],
            "hidden_size": 4, "intermediate_size": 8,
            "num_attention_heads": 2, "num_hidden_layers": 1,
            "max_position_embeddings": 8, "vocab_size": 4, "type_vocab_size": 1
        }))
        .unwrap();
        assert_eq!(down_policy(&roberta), DownPolicy::Existing);
    }

    #[test]
    fn enqueue_classification_keeps_rejections_recoverable_and_fatal_pressure_typed() {
        let preflight = EnqueueError::PreflightRejected(infer("pure preflight rejected"));
        assert!(matches!(preflight, EnqueueError::PreflightRejected(_)));
        assert!(!preflight.poisons());

        let rejected: EnqueueError =
            ShapeError::Rejected(infer("cold GTE down shape rejected")).into();
        assert!(matches!(rejected, EnqueueError::Rejected(_)));
        assert!(!rejected.poisons());

        let fatal: EnqueueError = ShapeError::Fatal(infer("descriptor rollback failed")).into();
        assert!(matches!(fatal, EnqueueError::Fatal(_)));
        assert!(fatal.poisons());

        let admission: EnqueueError = BackendError::Overloaded("staging busy".into()).into();
        assert!(matches!(admission, EnqueueError::Admission(_)));
        assert!(!admission.poisons());

        let fatal_overloaded: EnqueueError =
            ShapeError::Fatal(BackendError::Overloaded("fatal pressure".into())).into();
        assert!(matches!(
            fatal_overloaded,
            EnqueueError::Admission(BackendError::Overloaded(_))
        ));
        assert!(!fatal_overloaded.poisons());

        let internal: EnqueueError = infer("unexpected runtime failure").into();
        assert!(matches!(internal, EnqueueError::Fatal(_)));
        assert!(internal.poisons());
    }

    fn bf16_bit_mismatch(actual: &[u16], gold: &[u16]) -> Option<(usize, u16, u16)> {
        if actual.len() != gold.len() {
            return Some((actual.len().min(gold.len()), 0, 0));
        }
        actual
            .iter()
            .zip(gold)
            .enumerate()
            .find_map(|(index, (&actual, &gold))| (actual != gold).then_some((index, actual, gold)))
    }

    #[test]
    fn negative_gold_checker_rejects_one_corrupted_bf16_bit() {
        let actual = (0..9216u32)
            .map(|index| index.wrapping_mul(37).rotate_left(3) as u16)
            .collect::<Vec<_>>();
        assert!(bf16_bit_mismatch(&actual, &actual).is_none());
        let mut corrupted = actual.clone();
        corrupted[9215] ^= 1;
        assert_eq!(
            bf16_bit_mismatch(&actual, &corrupted).map(|mismatch| mismatch.0),
            Some(9215)
        );
        assert!(bf16_bit_mismatch(&actual[..9216], &actual[..9215]).is_some());
    }

    fn read_bf16_bits(path: &Path) -> Vec<u16> {
        let bytes =
            std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(
            bytes.len() % 2,
            0,
            "{} is not a raw little-endian BF16 file",
            path.display()
        );
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    fn bf16_bits_finite(bits: u16) -> bool {
        f32::from_bits(u32::from(bits) << 16).is_finite()
    }

    fn native_down_queries(provider: &LtProvider, prepared: &PreparedDown) -> usize {
        match prepared {
            PreparedDown::NativeDefault { lt_native, .. } => {
                provider.native_default_queries(lt_native)
            }
            PreparedDown::Existing(_) => {
                panic!("GTE down primitive requires a native-default plan")
            }
        }
    }

    fn native_down_cache_len(provider: &LtProvider, prepared: &PreparedDown) -> usize {
        match prepared {
            PreparedDown::NativeDefault { lt_native, .. } => {
                provider.native_default_cache_len(lt_native)
            }
            PreparedDown::Existing(_) => {
                panic!("GTE down primitive requires a native-default plan")
            }
        }
    }

    fn engine_down_state(backend: &OxideBackend) -> (usize, usize) {
        let engine = backend.engine.lock().unwrap();
        let model = match &*engine.model {
            ExecutableModel::Encoder(model) => model,
            ExecutableModel::Decoder(_) => panic!("GTE must use encoder plans"),
        };
        (
            native_down_queries(&engine.lt, &model.plans.output),
            native_down_cache_len(&engine.lt, &model.plans.output),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_down_result(
        prepared: &mut PreparedDown,
        provider: &LtProvider,
        blas: Option<&Blas>,
        tokens: usize,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        stream: &Stream,
        width: usize,
    ) -> Result<(), ShapeError> {
        prepared.begin_prepare();
        provider.bind_context().unwrap();
        // SAFETY: the primitive helper bound the provider context once above;
        // every buffer remains alive until the readback stream drains.
        let result = unsafe {
            prepared.prepare_layer(
                provider, blas, tokens, x, weight, bias, output, stream, width, 0,
            )
        };
        prepared.end_prepare();
        result?;
        // SAFETY: all buffers belong to this provider context and remain alive
        // until the readback stream drains.
        unsafe {
            enqueue_down(
                prepared, provider, blas, tokens, 0, x, weight, bias, output, stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_down(
        prepared: &mut PreparedDown,
        provider: &LtProvider,
        blas: Option<&Blas>,
        tokens: usize,
        x: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        stream: &Stream,
        width: usize,
    ) -> Vec<u16> {
        run_down_result(
            prepared, provider, blas, tokens, x, weight, bias, output, stream, width,
        )
        .unwrap();
        output
            .to_host_vec(stream)
            .unwrap()
            .iter()
            .take(tokens * width)
            .map(|value| value.to_bits())
            .collect()
    }

    fn assert_gold_bits(actual: &[u16], gold: &[u16], label: &str) {
        if let Some((index, actual, gold)) = bf16_bit_mismatch(actual, gold) {
            panic!("{label} BF16 mismatch at {index}: actual {actual:#06x}, gold {gold:#06x}");
        }
    }

    #[test]
    #[ignore = "Requires CUDA, an actual GTE INFINITY_OXIDE_TEST_MODEL, and INFINITY_OXIDE_KERNEL_MODULE"]
    fn gte_down_cold_rejection_preserves_held_deferred_and_recovers() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("Set INFINITY_OXIDE_TEST_MODEL to a real GTE checkpoint");
        let new_backend = || {
            OxideBackend::new(
                Path::new(&path),
                "bfloat16",
                ModelType::Embedding(Pool::Mean),
                0,
            )
            .unwrap()
        };
        let a = batch();
        let b = repeated_sequences(1, 65);
        let reference = new_backend();
        let hidden = reference.config.hidden_size();
        assert_eq!(hidden, 768, "this recovery test requires the GTE 768 shape");
        assert_eq!(
            down_policy(match &reference.config {
                LoadedConfig::Encoder(config) => config,
                LoadedConfig::Decoder(_) => panic!("GTE must use the encoder executor"),
            }),
            DownPolicy::NativeDefault
        );
        let expected_a = pooled_bits(reference.embed(a.clone()).unwrap());
        let expected_b = pooled_bits(reference.embed(b.clone()).unwrap());
        drop(reference);

        let backend = new_backend();
        let held_a = backend.submit_embed(a.clone()).unwrap();
        assert!(matches!(&held_a, Inference::Deferred(_)));
        let (queries_before, cache_before) = engine_down_state(&backend);
        {
            let engine = backend.engine.lock().unwrap();
            engine.lt.reject_down_shape_once_after(b.input_ids.len(), 0);
        }
        let failed = backend.submit_embed(b.clone());
        let message = match &failed {
            Err(BackendError::Inference(message)) => message.clone(),
            _ => panic!("cold B must be rejected as a recoverable inference error"),
        };
        assert!(
            message.contains("algo check status 15"),
            "the down-only injected AlgoCheck must be the rejection source: {message}"
        );
        let rejected = request_counters();
        assert_eq!(
            rejected.recovery_drains, 1,
            "the post-reservation rejection must own exactly one recovery drain"
        );
        assert_eq!(
            (rejected.lease_begins, rejected.uploads, rejected.launches),
            (0, 0, 0),
            "rejected B must not reach lease.begin, H2D upload, or a request kernel"
        );
        assert_eq!(
            rejected.engine_context_binds, 1,
            "the rejected request still uses the regular outer Engine binding"
        );
        {
            let engine = backend.engine.lock().unwrap();
            assert!(!engine.failed, "recoverable down rejection must not poison");
            assert!(
                !engine.io.poisoned(),
                "recoverable down rejection keeps Io open"
            );
            let work = match engine.workspace.as_ref().unwrap() {
                Workspace::Encoder(work) => work,
                Workspace::Decoder(_) => panic!("GTE must use the encoder workspace"),
            };
            assert!(
                work.key().0 >= b.input_ids.len(),
                "the rejected cold shape must have reserved the grown arena first"
            );
            assert!(
                work.key().0 > a.input_ids.len(),
                "the held result must exercise the arena initialization growth path"
            );
        }
        let (queries_after, cache_after) = engine_down_state(&backend);
        assert_eq!(
            queries_after,
            queries_before + 1,
            "the rejected cold B must query the heuristic exactly once"
        );
        assert_eq!(
            cache_after, cache_before,
            "the rejected cold B must not publish or evict persistent cache metadata"
        );

        let actual_a_held = pooled_bits(held_a.complete().unwrap());
        assert_pooled_bits(&actual_a_held, &expected_a, hidden);
        backend.health().unwrap();

        let first_b = pooled_bits(backend.embed(b.clone()).unwrap());
        assert_pooled_bits(&first_b, &expected_b, hidden);
        let repeat_b = pooled_bits(backend.embed(b.clone()).unwrap());
        assert_pooled_bits(&repeat_b, &expected_b, hidden);
        let hot = request_counters();
        assert_eq!(
            hot.recovery_drains, 0,
            "a hot successful request must not perform a recovery drain"
        );
        assert_eq!(
            hot.engine_context_binds, 1,
            "the regular outer Engine binding must remain for every request"
        );
        assert_eq!(
            hot.provider_context_queries, 0,
            "a hot Engine request must not query provider context from any down layer"
        );
        assert_eq!(
            (hot.lease_begins, hot.uploads),
            (1, 1),
            "the hot request still reaches exactly one lease begin and H2D upload"
        );
        assert!(hot.launches > 0, "the hot request must execute its body");
        let a_again = pooled_bits(backend.embed(a).unwrap());
        assert_pooled_bits(&a_again, &expected_a, hidden);
        backend.health().unwrap();
    }

    #[test]
    #[ignore = "Requires CUDA, an actual GTE INFINITY_OXIDE_TEST_MODEL, and INFINITY_OXIDE_KERNEL_MODULE"]
    fn gte_preflight_rejection_on_fresh_thread_returns_without_recovery_drain() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("Set INFINITY_OXIDE_TEST_MODEL to a real GTE checkpoint");
        std::thread::spawn(move || {
            let backend = OxideBackend::new(
                Path::new(&path),
                "bfloat16",
                ModelType::Embedding(Pool::Mean),
                0,
            )
            .unwrap();
            let request = batch();
            {
                let engine = backend.engine.lock().unwrap();
                engine
                    .lt
                    .reject_shape_once_after(request.input_ids.len(), 0);
            }
            let failed = backend.submit_embed(request.clone());
            let message = match &failed {
                Err(BackendError::Inference(message)) => message.clone(),
                _ => panic!("fresh preflight rejection must be a recoverable inference error"),
            };
            assert!(
                message.contains("algo check status 15"),
                "the generic preflight AlgoCheck must be the rejection source: {message}"
            );
            let counters = request_counters();
            assert_eq!(
                counters.recovery_drains, 0,
                "a pure preflight rejection must return immediately without draining"
            );
            assert_eq!(
                (counters.lease_begins, counters.uploads, counters.launches),
                (0, 0, 0)
            );
            assert_eq!(
                counters.engine_context_binds, 0,
                "preflight rejection returns before the outer Engine bind"
            );
            {
                let engine = backend.engine.lock().unwrap();
                assert!(!engine.failed);
                assert!(!engine.io.poisoned());
                assert!(
                    engine.workspace.is_none(),
                    "preflight rejection must not reserve an arena"
                );
            }
            backend.health().unwrap();
            let recovered = pooled_bits(backend.embed(request).unwrap());
            assert_eq!(recovered.len(), 2);
            backend.health().unwrap();
        })
        .join()
        .unwrap();
    }

    #[test]
    #[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_MODEL (GTE), INFINITY_OXIDE_GTE_DOWN_INPUT and INFINITY_OXIDE_GTE_DOWN_GOLD"]
    fn trained_gte_down_default_matches_full_bf16_gold() {
        const PRESSURE_ROWS: usize = 70;
        let model = std::env::var_os("INFINITY_OXIDE_TEST_MODEL")
            .expect("Set INFINITY_OXIDE_TEST_MODEL to a GTE checkpoint");
        let input_path = std::env::var_os("INFINITY_OXIDE_GTE_DOWN_INPUT")
            .expect("Set INFINITY_OXIDE_GTE_DOWN_INPUT to raw little-endian BF16 input");
        let gold_path = std::env::var_os("INFINITY_OXIDE_GTE_DOWN_GOLD")
            .expect("Set INFINITY_OXIDE_GTE_DOWN_GOLD to raw little-endian BF16 gold");
        let tokens = std::env::var("INFINITY_OXIDE_GTE_DOWN_TOKENS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("GTE down tokens must be a usize")
            })
            .unwrap_or(12);
        assert!(tokens > 0, "GTE down tokens must be nonzero");
        let hidden: usize = 768;
        let intermediate: usize = 3072;
        let host = BertWeights::load(Path::new(&model)).unwrap();
        assert_eq!(host.config.family, EncoderFamily::Gte);
        assert_eq!(
            (host.config.intermediate_size, host.config.hidden_size),
            (intermediate, hidden)
        );
        let input = read_bf16_bits(Path::new(&input_path));
        let gold = read_bf16_bits(Path::new(&gold_path));
        assert_eq!(input.len(), tokens * intermediate);
        assert_eq!(gold.len(), tokens * hidden);

        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let provider = LtProvider::new_native_default(context.clone(), stream.clone()).unwrap();
        let mut prepared = PreparedDown::new(&provider, &host.config).unwrap();
        assert!(matches!(prepared, PreparedDown::NativeDefault { .. }));
        let mut uploader = WeightUploader::new(stream.clone()).unwrap();
        let layer = host.layers.first().unwrap();
        let weight = uploader.upload(&[&layer.output.weight]).unwrap();
        let bias = uploader.upload(&[&layer.output.bias]).unwrap();
        uploader.finish().unwrap();
        let host_input = input.into_iter().map(Bf16::from_bits).collect::<Vec<_>>();
        let capacity_rows = PRESSURE_ROWS.max(tokens);
        let mut pressure = Vec::with_capacity(capacity_rows * intermediate);
        for row in 0..capacity_rows {
            let start = (row % tokens) * intermediate;
            pressure.extend_from_slice(&host_input[start..start + intermediate]);
        }
        let x = DeviceBuffer::from_host(&stream, &pressure).unwrap();
        let mut output = DeviceBuffer::<Bf16>::zeroed(&stream, capacity_rows * hidden).unwrap();
        let blas = (tokens == 1).then(|| Blas::new(context.clone(), stream.clone()).unwrap());

        let actual = run_down(
            &mut prepared,
            &provider,
            blas.as_ref(),
            tokens,
            &x,
            &weight,
            &bias,
            &mut output,
            &stream,
            hidden,
        );
        assert_gold_bits(&actual, &gold, &format!("GTE down M={tokens}"));

        if tokens == 1 {
            assert_eq!(
                native_down_queries(&provider, &prepared),
                0,
                "M1 must use GemmEx without an Lt algorithm query"
            );
            let repeated = run_down(
                &mut prepared,
                &provider,
                blas.as_ref(),
                tokens,
                &x,
                &weight,
                &bias,
                &mut output,
                &stream,
                hidden,
            );
            assert_eq!(native_down_queries(&provider, &prepared), 0);
            assert_gold_bits(&repeated, &gold, "repeated GTE down M=1");
            let mut corrupted = gold.clone();
            let corrupted_index = gold.len() / 2;
            corrupted[corrupted_index] ^= 1;
            assert_eq!(
                bf16_bit_mismatch(&repeated, &corrupted).map(|mismatch| mismatch.0),
                Some(corrupted_index),
                "a one-bit corrupted M1 gold must be rejected"
            );
            let mut wrong_output =
                DeviceBuffer::<Bf16>::zeroed(&stream, capacity_rows * hidden).unwrap();
            // SAFETY: the prepared invocation still belongs to this provider
            // context; the deliberately different output must be rejected before
            // any bias copy or GEMM.
            let mismatch = unsafe {
                enqueue_down(
                    &mut prepared,
                    &provider,
                    blas.as_ref(),
                    tokens,
                    0,
                    &x,
                    &weight,
                    &bias,
                    &mut wrong_output,
                    &stream,
                )
            };
            assert!(
                matches!(&mismatch, Err(ShapeError::Fatal(_))),
                "a mismatched M1 launch buffer must fail closed: {mismatch:?}"
            );
            assert!(
                wrong_output
                    .to_host_vec(&stream)
                    .unwrap()
                    .iter()
                    .all(|value| value.to_bits() == 0),
                "a mismatched M1 launch must not copy bias into the wrong output"
            );
            return;
        }

        let queries = native_down_queries(&provider, &prepared);
        assert_eq!(queries, 1, "first actual M must query the heuristic once");
        let repeated = run_down(
            &mut prepared,
            &provider,
            blas.as_ref(),
            tokens,
            &x,
            &weight,
            &bias,
            &mut output,
            &stream,
            hidden,
        );
        assert_eq!(
            native_down_queries(&provider, &prepared),
            queries,
            "repeated actual M must reuse the prepared cache entry"
        );
        assert_gold_bits(&repeated, &gold, "repeated GTE down M");

        let rejected_rows = if tokens == 65 { 66 } else { 65 };
        provider.reject_down_shape_once_after(rejected_rows, 0);
        let rejected = run_down_result(
            &mut prepared,
            &provider,
            blas.as_ref(),
            rejected_rows,
            &x,
            &weight,
            &bias,
            &mut output,
            &stream,
            hidden,
        );
        match &rejected {
            Err(ShapeError::Rejected(BackendError::Inference(message))) => assert!(
                message.contains("algo check status 15"),
                "injected AlgoCheck must be the rejection source: {message}"
            ),
            _ => panic!("injected AlgoCheck rejection must roll back: {rejected:?}"),
        }
        let after_reject = native_down_queries(&provider, &prepared);
        assert_eq!(
            after_reject,
            queries + 1,
            "the rejected cold shape must make exactly one heuristic query"
        );
        let recovered = run_down(
            &mut prepared,
            &provider,
            blas.as_ref(),
            tokens,
            &x,
            &weight,
            &bias,
            &mut output,
            &stream,
            hidden,
        );
        assert_eq!(
            native_down_queries(&provider, &prepared),
            after_reject,
            "a rejected cold entry must not publish or evict the cached shape"
        );
        assert_gold_bits(&recovered, &gold, "recovered GTE down M");

        let queries_before_pressure = native_down_queries(&provider, &prepared);
        let order = (2..=PRESSURE_ROWS)
            .filter(|rows| *rows != tokens)
            .chain(std::iter::once(tokens))
            .collect::<Vec<_>>();
        for rows in &order {
            let bits = run_down(
                &mut prepared,
                &provider,
                blas.as_ref(),
                *rows,
                &x,
                &weight,
                &bias,
                &mut output,
                &stream,
                hidden,
            );
            if let Some(index) = bits.iter().position(|bits| !bf16_bits_finite(*bits)) {
                panic!("GTE down M={rows} pressure output is non-finite at {index}");
            }
            if *rows == tokens {
                assert_gold_bits(&bits, &gold, "pressure-recovered GTE down M");
            }
        }
        let after_pressure = native_down_queries(&provider, &prepared);
        assert_eq!(
            after_pressure,
            queries_before_pressure + order.len(),
            "each distinct pressure shape must make exactly one heuristic query"
        );
        assert_eq!(
            native_down_cache_len(&provider, &prepared),
            64,
            "bounded metadata cache must evict without rejecting legal shapes"
        );
        let returned = run_down(
            &mut prepared,
            &provider,
            blas.as_ref(),
            tokens,
            &x,
            &weight,
            &bias,
            &mut output,
            &stream,
            hidden,
        );
        assert_eq!(
            native_down_queries(&provider, &prepared),
            after_pressure,
            "returning to the trained M must hit the bounded cache"
        );
        assert_gold_bits(&returned, &gold, "cache-hit GTE down M");

        let mut corrupted = gold.clone();
        let corrupted_index = gold.len() / 2;
        corrupted[corrupted_index] ^= 1;
        assert_eq!(
            bf16_bit_mismatch(&returned, &corrupted).map(|mismatch| mismatch.0),
            Some(corrupted_index),
            "a one-bit corrupted gold must be rejected"
        );
    }
}
