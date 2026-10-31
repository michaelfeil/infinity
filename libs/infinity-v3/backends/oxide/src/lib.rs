// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! BERT executor with oxide-compiled Rust kernels and BF16 cuBLAS projections.
mod cache;
mod cublas;
mod fa4;
pub mod math;
mod model;
pub mod tensor;
mod upload;

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::{
    result_memory, Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool,
    Predictions, TokenPredictions,
};
use math::LinearPlan;
use model::{BertConfig, BertWeights, LayerWeights, LinearWeights};
use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    path::Path,
    sync::{Arc, Mutex},
};
use tensor::Bf16;
use upload::WeightUploader;

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

struct Kernels {
    embedding: Function,
    embedding_384: Function,
    norm: Function,
    norm_384: Function,
    qkv: Function,
    gelu: Function,
    attention: Function,
    pool: Function,
    mean_pool: Function,
}
impl Kernels {
    fn load(ctx: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start(
                    "Set INFINITY_OXIDE_KERNEL_MODULE to the BERT artifact compiled by cargo oxide",
                )
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
        let pool = module.load_function("infinity_pool").map_err(start)?;
        Ok(Self {
            embedding_384: module
                .load_function("infinity_embedding_norm_384")
                .map_err(start)?,
            norm_384: module
                .load_function("infinity_layer_norm_384")
                .map_err(start)?,
            mean_pool: module.load_function("infinity_mean_pool").map_err(start)?,
            embedding: module
                .load_function("infinity_embedding_norm")
                .map_err(start)?,
            norm: module.load_function("infinity_layer_norm").map_err(start)?,
            qkv: module.load_function("infinity_qkv").map_err(start)?,
            gelu: module.load_function("infinity_gelu_bias").map_err(start)?,
            attention: module.load_function("infinity_attention").map_err(start)?,
            pool,
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
    // SAFETY: fixed ABI signatures and validated dimensions; arenas and modules
    // stay retained. Every entry currently uses 256 threads per block.
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
    if width == 384 {
        return launch_threads(
            &k.norm_384,
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
    launch(
        &k.norm,
        s,
        (rows, 1, 1),
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
fn bias(
    k: &Function,
    s: &Stream,
    mut input: u64,
    b: &DeviceBuffer<Bf16>,
    mut out: u64,
    elements: usize,
    mut width: u32,
) -> Result<(), BackendError> {
    let (mut b, mut n) = (b.cu_deviceptr(), u32_size(elements)?);
    launch(
        k,
        s,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut b),
            arg(&mut out),
            arg(&mut n),
            arg(&mut width),
        ],
    )
}

struct Engine {
    blas: ManuallyDrop<cublas::Blas>,
    kernels: ManuallyDrop<Kernels>,
    weights: ManuallyDrop<Weights>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    workspace: Option<Work>,
    cached: cache::WorkspaceCache<WorkKey, Work>,
    failed: bool,
    fa4: ManuallyDrop<Option<fa4::Fa4>>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        let complete = self.stream.synchronize().is_ok();
        self.workspace.take();
        drop(self.cached.drain());
        if complete {
            // SAFETY: Engine is dropped once, and all GPU uses have finished.
            // If drain fails, retaining these owners also retains their CUDA
            // context/modules/libraries while completion is uncertain.
            unsafe {
                ManuallyDrop::drop(&mut self.blas);
                ManuallyDrop::drop(&mut self.kernels);
                ManuallyDrop::drop(&mut self.weights);
                ManuallyDrop::drop(&mut self.fa4);
            }
        }
    }
}
/// Stable device and pinned-host arenas; raw launches retain all allocations.
type WorkKey = (usize, usize, usize, usize);
const BUCKET: usize = 32;
fn bucketed(value: usize) -> Result<usize, BackendError> {
    value
        .checked_add(BUCKET - 1)
        .map(|rounded| rounded / BUCKET * BUCKET)
        .ok_or_else(|| infer("Batch bucket capacity overflow"))
}
fn work_key(b: &Batch, max_length: usize) -> Result<WorkKey, BackendError> {
    Ok((
        bucketed(b.input_ids.len())?,
        b.cumulative_seq_lengths.len(),
        b.pooled_indices.len(),
        bucketed(max_length)?,
    ))
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
        row.fill(0);
        let (head, _) = row
            .split_at_mut_checked(source.len())
            .ok_or_else(|| infer("Invalid input staging layout"))?;
        head.copy_from_slice(source);
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
struct Work {
    linear: LinearPlans,
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    max_length: usize,
    bytes: usize,
    input: ManuallyDrop<DeviceBuffer<u32>>,
    host_input: ManuallyDrop<PinnedHostBuffer<u32>>,
    host_output: ManuallyDrop<PinnedHostBuffer<f32>>,
    x: ManuallyDrop<DeviceBuffer<Bf16>>,
    temp: ManuallyDrop<DeviceBuffer<Bf16>>,
    qkv: ManuallyDrop<DeviceBuffer<Bf16>>,
    query: ManuallyDrop<DeviceBuffer<Bf16>>,
    key: ManuallyDrop<DeviceBuffer<Bf16>>,
    value: ManuallyDrop<DeviceBuffer<Bf16>>,
    attention: ManuallyDrop<DeviceBuffer<Bf16>>,
    intermediate: ManuallyDrop<DeviceBuffer<Bf16>>,
    pooled: ManuallyDrop<DeviceBuffer<f32>>,
}
struct LinearPlans {
    qkv: LinearPlan<Bf16>,
    attention: LinearPlan<Bf16>,
    intermediate: LinearPlan<Bf16>,
    output: LinearPlan<Bf16>,
}
impl LinearPlans {
    fn new(tokens: usize, c: &BertConfig) -> Result<Self, BackendError> {
        let h = c.hidden_size;
        Ok(Self {
            qkv: LinearPlan::contiguous(tokens, h, product(h, 3)?).map_err(infer)?,
            attention: LinearPlan::contiguous(tokens, h, h).map_err(infer)?,
            intermediate: LinearPlan::contiguous(tokens, h, c.intermediate_size).map_err(infer)?,
            output: LinearPlan::contiguous(tokens, c.intermediate_size, h).map_err(infer)?,
        })
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: every field is dropped once after completion. Failed
            // drains deliberately retain device and pinned allocations.
            unsafe {
                ManuallyDrop::drop(&mut self.input);
                ManuallyDrop::drop(&mut self.host_input);
                ManuallyDrop::drop(&mut self.host_output);
                ManuallyDrop::drop(&mut self.x);
                ManuallyDrop::drop(&mut self.temp);
                ManuallyDrop::drop(&mut self.qkv);
                ManuallyDrop::drop(&mut self.query);
                ManuallyDrop::drop(&mut self.key);
                ManuallyDrop::drop(&mut self.value);
                ManuallyDrop::drop(&mut self.attention);
                ManuallyDrop::drop(&mut self.intermediate);
                ManuallyDrop::drop(&mut self.pooled);
            }
        }
    }
}
impl Work {
    fn key(&self) -> WorkKey {
        (
            self.tokens,
            self.sequences,
            self.selections,
            self.max_length,
        )
    }
    fn allocation_bytes(key: WorkKey, c: &BertConfig) -> Result<usize, BackendError> {
        let (tokens, sequences, selections, max_length) = key;
        i32::try_from(tokens).map_err(infer)?;
        u32_size(product(tokens, c.hidden_size)?)?;
        u32_size(product(tokens, c.intermediate_size)?)?;
        u32_size(product(max_length, 4)?)?;
        let input = product(tokens, 3)?
            .checked_add(sequences)
            .and_then(|n| n.checked_add(selections))
            .ok_or_else(|| infer("Input arena overflow"))?;
        // Device+host input/output staging and nine BF16 hidden-sized buffers
        // (including the packed QKV projection) plus the intermediate tensor.
        let output = product(product(selections, c.hidden_size)?, 8)?;
        let hidden = product(product(tokens, c.hidden_size)?, 18)?;
        let intermediate = product(product(tokens, c.intermediate_size)?, 2)?;
        product(input, 8)?
            .checked_add(output)
            .and_then(|n| n.checked_add(hidden))
            .and_then(|n| n.checked_add(intermediate))
            .ok_or_else(|| infer("Workspace arena overflow"))
    }
    fn matches(&self, key: &WorkKey) -> bool {
        self.key() == *key
    }
    fn new(s: Arc<Stream>, c: &BertConfig, key: WorkKey) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, c)?;
        let (tokens, sequences, selections, max_length) = key;
        let hidden = product(tokens, c.hidden_size)?;
        let input_len = product(tokens, 3)?
            .checked_add(sequences)
            .and_then(|n| n.checked_add(selections))
            .ok_or_else(|| infer("Input arena overflow"))?;
        let output = product(selections, c.hidden_size)?;
        let linear = LinearPlans::new(tokens, c)?;
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
        let host_input = PinnedHostBuffer::zeroed(s.context(), input_len).map_err(infer)?;
        let host_output = PinnedHostBuffer::zeroed(s.context(), output).map_err(infer)?;
        let x = alloc(hidden)?;
        let temp = alloc(hidden)?;
        let qkv = alloc(product(hidden, 3)?)?;
        let query = alloc(hidden)?;
        let key_buffer = alloc(hidden)?;
        let value = alloc(hidden)?;
        let attention = alloc(hidden)?;
        let intermediate = alloc(product(tokens, c.intermediate_size)?)?;
        // SAFETY: the pooling kernel writes every element before D2H.
        let pooled =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&s, output) }.map_err(infer)?;
        Ok(Self {
            linear,
            tokens,
            sequences,
            selections,
            max_length,
            bytes,
            input: ManuallyDrop::new(input),
            host_input: ManuallyDrop::new(host_input),
            host_output: ManuallyDrop::new(host_output),
            x: ManuallyDrop::new(x),
            temp: ManuallyDrop::new(temp),
            qkv: ManuallyDrop::new(qkv),
            query: ManuallyDrop::new(query),
            key: ManuallyDrop::new(key_buffer),
            value: ManuallyDrop::new(value),
            attention: ManuallyDrop::new(attention),
            intermediate: ManuallyDrop::new(intermediate),
            pooled: ManuallyDrop::new(pooled),
            stream: s,
        })
    }
    fn stage(&mut self, b: &Batch) -> Result<(), BackendError> {
        pack_input(
            self.host_input.as_mut_slice(),
            self.tokens,
            &b.input_ids,
            &b.position_ids,
            &b.token_type_ids,
            &b.cumulative_seq_lengths,
            &b.pooled_indices,
        )
    }
    fn upload(&self) -> Result<(), BackendError> {
        // SAFETY: pinned staging and destination are retained in this workspace;
        // no host edits occur before the final drain, including error paths.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.input.cu_deviceptr(),
                self.host_input.as_ptr(),
                self.input.num_bytes(),
                self.stream.cu_stream(),
            )
        }
        .map_err(infer)
    }
    fn download(&mut self) -> Result<(), BackendError> {
        // SAFETY: output sizes agree; pinned staging stays owned by Engine even
        // if synchronization fails. Engine leaks a failed arena rather than
        // freeing host memory still potentially being written by CUDA.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                self.host_output.as_mut_ptr(),
                self.pooled.cu_deviceptr(),
                self.pooled.num_bytes(),
                self.stream.cu_stream(),
            )
        }
        .map_err(infer)
    }
    fn complete(&self) -> Result<Vec<f32>, BackendError> {
        self.stream.synchronize().map_err(infer)?;
        Ok(self.host_output.as_slice().to_vec())
    }
}
impl Engine {
    fn forward(
        &mut self,
        b: &Batch,
        c: &BertConfig,
        pool: &Pool,
        max_length: usize,
    ) -> Result<Vec<f32>, BackendError> {
        if self.failed {
            return Err(BackendError::Unhealthy);
        }
        let result = self.forward_inner(b, c, pool, max_length);
        if result.is_err() {
            self.failed = true;
            if self.stream.synchronize().is_err() {
                if let Some(work) = self.workspace.take() {
                    std::mem::forget(work);
                }
            }
        }
        result
    }
    fn forward_inner(
        &mut self,
        b: &Batch,
        c: &BertConfig,
        pool: &Pool,
        max_length: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.context.bind_to_thread().map_err(infer)?;
        let plan = work_key(b, max_length)?;
        if !self.workspace.as_ref().is_some_and(|w| w.matches(&plan)) {
            let next = self.cached.take(&plan);
            // Reserve the next active arena before retaining the previous one.
            // Oversized active requests empty the inactive cache.
            drop(self.cached.reserve(Work::allocation_bytes(plan, c)?));
            if let Some(previous) = self.workspace.take() {
                let (key, bytes) = (previous.key(), previous.bytes);
                drop(self.cached.put(key, previous, bytes));
            }
            self.workspace = Some(match next {
                Some(work) => work,
                None => Work::new(self.stream.clone(), c, plan)?,
            });
        }
        let w = self
            .workspace
            .as_mut()
            .ok_or_else(|| infer("Missing oxide workspace"))?;
        w.stage(b)?;
        let s = &*self.stream;
        // Ordinary ordered-stream dispatch; plans are prepared with the arena.
        w.upload()?;
        let k = &self.kernels;
        let weights = &self.weights;
        let (tokens, h) = (w.tokens, c.hidden_size);
        let (mut t, mut width) = (u32_size(tokens)?, u32_size(h)?);
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
        let (mut gamma, mut beta) = (weights.gamma.cu_deviceptr(), weights.beta.cu_deviceptr());
        let (mut x, mut eps) = (w.x.cu_deviceptr(), c.layer_norm_eps as f32);
        if h == 384 {
            launch_threads(
                &k.embedding_384,
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
            launch(
                &k.embedding,
                s,
                (t, 1, 1),
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
        for layer in &weights.layers {
            let mut qkv = w.qkv.cu_deviceptr();
            // SAFETY: this thread bound the engine context once above. The
            // engine retains all buffers until stream completion/error drain;
            // input/output buffers are distinct and access is stream-ordered.
            unsafe {
                self.blas
                    .linear(&w.linear.qkv, &w.x, &layer.qkv.weight, &mut w.qkv)?;
            }
            let (mut q, mut key, mut v) = (
                w.query.cu_deviceptr(),
                w.key.cu_deviceptr(),
                w.value.cu_deviceptr(),
            );
            let (mut qkv_bias, mut n) = (
                layer.qkv.bias.cu_deviceptr(),
                u32_size(product(tokens, h)?)?,
            );
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
            let (mut count, mut heads, mut dim) = (
                u32_size(b.cumulative_seq_lengths.len() - 1)?,
                u32_size(c.num_attention_heads)?,
                u32_size(h / c.num_attention_heads)?,
            );
            if let Some(provider) = self.fa4.as_ref() {
                provider.forward(
                    s,
                    q,
                    key,
                    v,
                    attn,
                    boundaries,
                    tokens,
                    b.cumulative_seq_lengths.len() - 1,
                    w.max_length,
                )?;
            } else {
                launch(
                    &k.attention,
                    s,
                    (u32_size(w.max_length)?, heads, count),
                    u32_size(product(w.max_length, 4)?)?,
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
            let temp = w.temp.cu_deviceptr();
            // SAFETY: same retained-buffer/context contract as the QKV call.
            unsafe {
                self.blas.linear(
                    &w.linear.attention,
                    &w.attention,
                    &layer.attention.weight,
                    &mut w.temp,
                )?;
            }
            norm(
                k,
                s,
                temp,
                x,
                layer.attention.bias.cu_deviceptr(),
                &layer.attention_gamma,
                &layer.attention_beta,
                x,
                t,
                width,
                eps,
            )?;
            let intermediate = w.intermediate.cu_deviceptr();
            // SAFETY: same retained-buffer/context contract as the QKV call.
            unsafe {
                self.blas.linear(
                    &w.linear.intermediate,
                    &w.x,
                    &layer.intermediate.weight,
                    &mut w.intermediate,
                )?;
            }
            bias(
                &k.gelu,
                s,
                intermediate,
                &layer.intermediate.bias,
                intermediate,
                product(tokens, c.intermediate_size)?,
                u32_size(c.intermediate_size)?,
            )?;
            // SAFETY: same retained-buffer/context contract as the QKV call.
            unsafe {
                self.blas.linear(
                    &w.linear.output,
                    &w.intermediate,
                    &layer.output.weight,
                    &mut w.temp,
                )?;
            }
            norm(
                k,
                s,
                temp,
                x,
                layer.output.bias.cu_deviceptr(),
                &layer.output_gamma,
                &layer.output_beta,
                x,
                t,
                width,
                eps,
            )?;
        }
        let (mut bounds, mut indices, mut out) = (boundaries, selected, w.pooled.cu_deviceptr());
        let mut count = u32_size(b.pooled_indices.len())?;
        let mut mode = match pool {
            Pool::Cls => 0u32,
            Pool::Mean => 1,
            Pool::LastToken => 2,
            Pool::Splade => return Err(infer("Oxide SPLADE unsupported")),
        };
        if mode == 1 {
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
        } else {
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
        w.download()?;
        w.complete()
    }
}

pub struct OxideBackend {
    config: BertConfig,
    pool: Pool,
    engine: Mutex<Engine>,
}
impl OxideBackend {
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        if dtype != "bfloat16" {
            return Err(start("Oxide currently requires --dtype bfloat16"));
        }
        let pool = match model_type {
            ModelType::Embedding(p @ (Pool::Cls | Pool::Mean | Pool::LastToken)) => p,
            _ => {
                return Err(start(
                    "Oxide currently supports BERT pooled embeddings only",
                ))
            }
        };
        let host = BertWeights::load(path)?;
        let config = host.config.clone();
        if device_id >= device_count()? {
            return Err(start("CUDA device ordinal is not visible"));
        }
        let context = Context::new(device_id).map_err(start)?;
        let stream = context.new_stream().map_err(start)?;
        let attention_mode = std::env::var(infinity_v3_config::infinity_env!("OXIDE_ATTENTION"))
            .unwrap_or_else(|_| "auto".into());
        let fa4_root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"));
        let fa4 = match attention_mode.as_str() {
            "fa4" => Some(fa4::Fa4::new(
                context.clone(),
                Path::new(
                    &fa4_root.ok_or_else(|| start("FA4 attention requires INFINITY_FA4_ROOT"))?,
                ),
                config.num_attention_heads,
                config.hidden_size / config.num_attention_heads,
            )?),
            "auto"
                if fa4_root.is_some()
                    && config.num_attention_heads == 12
                    && config.hidden_size == 384 =>
            {
                Some(fa4::Fa4::new(
                    context.clone(),
                    Path::new(&fa4_root.ok_or_else(|| start("Missing FA4 root"))?),
                    12,
                    32,
                )?)
            }
            "auto" | "simt" => None,
            _ => return Err(start("INFINITY_OXIDE_ATTENTION must be auto,simt or fa4")),
        };
        let kernels = Kernels::load(&context)?;
        let blas = cublas::Blas::new(context.clone(), stream.clone())?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let weights = Weights {
            word: uploader.upload(&[&host.word])?,
            position: uploader.upload(&[&host.position])?,
            token_type: uploader.upload(&[&host.token_type])?,
            gamma: uploader.upload(&[&host.gamma])?,
            beta: uploader.upload(&[&host.beta])?,
            layers: host
                .layers
                .iter()
                .map(|l| Layer::upload(&mut uploader, l))
                .collect::<Result<_, _>>()?,
        };
        uploader.finish()?;
        Ok(Self {
            config,
            pool,
            engine: Mutex::new(Engine {
                blas: ManuallyDrop::new(blas),
                kernels: ManuallyDrop::new(kernels),
                weights: ManuallyDrop::new(weights),
                context,
                stream,
                workspace: None,
                cached: cache::WorkspaceCache::new(),
                failed: false,
                fa4: ManuallyDrop::new(fa4),
            }),
        })
    }
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
        || !batch.raw_indices.is_empty()
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
    if sequences > 65535
        || batch
            .pooled_indices
            .iter()
            .any(|&i| i as usize >= sequences)
        || batch.input_ids.iter().any(|&i| i as usize >= c.vocab_size)
        || batch
            .token_type_ids
            .iter()
            .any(|&i| i as usize >= c.type_vocab_size)
        || batch
            .position_ids
            .iter()
            .any(|&i| i as usize >= c.max_position_embeddings)
    {
        return Err(fail());
    }
    u32_size(product(tokens, c.intermediate_size)?)?;
    u32_size(product(tokens, c.hidden_size)?)?;
    Ok(max_length)
}
impl Backend for OxideBackend {
    fn health(&self) -> Result<(), BackendError> {
        let engine = self.engine.lock().map_err(|_| BackendError::Unhealthy)?;
        if engine.failed {
            return Err(BackendError::Unhealthy);
        }
        engine.stream.synchronize().map_err(infer)
    }
    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let max_length = validate(&batch, &self.config)?;
        if batch.pooled_indices.is_empty() {
            return Ok(Embeddings::default());
        }
        let elements = product(batch.pooled_indices.len(), self.config.hidden_size)?;
        // Account for the D2H staging vector and the simultaneously owned rows.
        let memory = result_memory::global_pool()?.reserve(product(elements, 8)?)?;
        let output = self
            .engine
            .lock()
            .map_err(|_| BackendError::Unhealthy)?
            .forward(&batch, &self.config, &self.pool, max_length)?;
        if output.len() != elements || output.iter().any(|v| !v.is_finite()) {
            return Err(infer("Invalid oxide embedding output"));
        }
        let mut result = Embeddings::default();
        result.memory = Some(memory);
        for (&index, row) in batch
            .pooled_indices
            .iter()
            .zip(output.chunks_exact(self.config.hidden_size))
        {
            result.insert(index as usize, Embedding::Pooled(row.to_vec()));
        }
        Ok(result)
    }
    fn predict(&self, _: Batch) -> Result<Predictions, BackendError> {
        Err(infer("Oxide classifiers are not implemented"))
    }
    fn predict_tokens(&self, _: Batch) -> Result<TokenPredictions, BackendError> {
        Err(infer("Oxide token predictions are not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        b.raw_indices = vec![0];
        assert!(validate(&b, &c).is_err());
    }
    #[test]
    fn buckets_token_and_length_capacities_to_round_up_32() {
        fn keyed(tokens: usize, max_length: usize) -> WorkKey {
            let mut b = batch();
            b.input_ids = vec![1; tokens];
            b.position_ids = vec![0; tokens];
            b.token_type_ids = vec![0; tokens];
            b.cumulative_seq_lengths = vec![0, tokens as u32];
            work_key(&b, max_length).unwrap()
        }
        assert_eq!(keyed(31, 31), keyed(32, 32));
        assert_ne!(keyed(32, 32), keyed(33, 33));
        assert_eq!(keyed(33, 33), keyed(64, 64));
        assert_ne!(keyed(64, 64), keyed(65, 65));
        assert_eq!(keyed(31, 64), keyed(32, 64));
        assert_ne!(keyed(32, 32), keyed(32, 33));
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
    fn rejects_bucket_and_capacity_overflow() {
        assert!(bucketed(usize::MAX).is_err());
        assert!(work_key(&batch(), usize::MAX).is_err());
        let c = config();
        let oversized = u32::MAX as usize + 1;
        assert!(Work::allocation_bytes((oversized, 2, 1, BUCKET), &c).is_err());
        assert!(Work::allocation_bytes((usize::MAX, 2, 1, BUCKET), &c).is_err());
    }
}
