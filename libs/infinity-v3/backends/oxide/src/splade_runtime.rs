// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use crate::lt::{LtPlan, LtProvider, NativeDefaultPending, NativeDefaultPrepared, ShapeError};
use crate::model::WeightSource;
use crate::splade::{SpladeDecoderHost, SpladeHeadHost};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::{blocks, infer, product, start, u32_size};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch};
use std::mem::{size_of, ManuallyDrop};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn reject(m: impl std::fmt::Display) -> ShapeError {
    ShapeError::Rejected(infer(m))
}
fn fatal(m: impl std::fmt::Display) -> ShapeError {
    ShapeError::Fatal(infer(m))
}

fn checked_i32(value: usize, label: &str) -> Result<i32, ShapeError> {
    i32::try_from(value).map_err(|_| fatal(format!("{label} exceeds i32 range")))
}

fn ceil_div(value: usize, divisor: usize) -> Result<usize, ShapeError> {
    if divisor == 0 {
        return Err(fatal("SPLADE grid divisor is zero"));
    }
    let bias = divisor
        .checked_sub(1)
        .ok_or_else(|| fatal("SPLADE grid overflow"))?;
    value
        .checked_add(bias)
        .map(|sum| sum / divisor)
        .ok_or_else(|| fatal("SPLADE grid overflow"))
}

fn checked_bytes(elements: usize, width: usize, label: &str) -> Result<usize, BackendError> {
    elements
        .checked_mul(width)
        .ok_or_else(|| infer(format!("{label} byte size overflow")))
}

fn work_bytes_for(
    hidden: usize,
    vocab: usize,
    rows: usize,
    pooled: usize,
) -> Result<usize, BackendError> {
    if pooled == 0 {
        return Ok(0);
    }
    if rows == 0 || hidden == 0 || vocab == 0 {
        return Err(infer("SPLADE work dimensions must be non-zero"));
    }
    if rows > i32::MAX as usize || pooled > i32::MAX as usize {
        return Err(infer("SPLADE work shape exceeds i32 launch range"));
    }
    let scratch = product(rows, hidden)?;
    let logits = product(rows, vocab)?;
    let output = product(pooled, vocab)?;
    let pairs = pooled
        .checked_mul(2)
        .ok_or_else(|| infer("SPLADE metadata element overflow"))?;
    let metadata = rows
        .checked_add(pairs)
        .ok_or_else(|| infer("SPLADE metadata element overflow"))?;
    let mut bytes = checked_bytes(scratch, 2, "SPLADE first scratch")?;
    bytes = bytes
        .checked_add(checked_bytes(scratch, 2, "SPLADE second scratch")?)
        .ok_or_else(|| infer("SPLADE work byte overflow"))?;
    bytes = bytes
        .checked_add(checked_bytes(logits, 2, "SPLADE logits")?)
        .ok_or_else(|| infer("SPLADE work byte overflow"))?;
    bytes = bytes
        .checked_add(checked_bytes(output, size_of::<f32>(), "SPLADE output")?)
        .ok_or_else(|| infer("SPLADE work byte overflow"))?;
    bytes = bytes
        .checked_add(checked_bytes(
            metadata,
            size_of::<u32>(),
            "SPLADE metadata",
        )?)
        .ok_or_else(|| infer("SPLADE work byte overflow"))?;
    u32_size(metadata)?;
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpladeSelection {
    rows: usize,
    pooled: usize,
    tokens: usize,
    metadata_elements: usize,
}

fn cumulative_span(
    batch: &Batch,
    sequences: usize,
    row: u32,
    tokens: u32,
) -> Result<(u32, u32), ShapeError> {
    let index = usize::try_from(row).map_err(|_| reject("SPLADE pooled index is invalid"))?;
    if index >= sequences {
        return Err(reject("SPLADE pooled index is outside the batch"));
    }
    let end_index = index
        .checked_add(1)
        .ok_or_else(|| reject("SPLADE pooled index overflow"))?;
    let start = batch
        .cumulative_seq_lengths
        .get(index)
        .copied()
        .ok_or_else(|| reject("SPLADE cumulative start is missing"))?;
    let end = batch
        .cumulative_seq_lengths
        .get(end_index)
        .copied()
        .ok_or_else(|| reject("SPLADE cumulative end is missing"))?;
    if end <= start {
        return Err(reject("SPLADE pooled selection has an empty span"));
    }
    if end > tokens {
        return Err(reject("SPLADE pooled span exceeds the token extent"));
    }
    Ok((start, end - start))
}

fn cumulative_bounds(batch: &Batch, tokens: u32) -> Result<usize, ShapeError> {
    let cumulative = &batch.cumulative_seq_lengths;
    if cumulative.first().copied() != Some(0) || cumulative.last().copied() != Some(tokens) {
        return Err(reject(
            "SPLADE cumulative lengths do not cover the input tokens",
        ));
    }
    if cumulative
        .windows(2)
        .any(|pair| pair.first().zip(pair.get(1)).is_some_and(|(a, b)| b <= a))
    {
        return Err(reject("SPLADE cumulative lengths are not monotonic"));
    }
    Ok(cumulative.len().saturating_sub(1))
}

impl SpladeSelection {
    pub(crate) fn new(batch: &Batch) -> Result<Self, ShapeError> {
        let tokens = batch.input_ids.len();
        let pooled = batch.pooled_indices.len();
        if pooled == 0 {
            return Ok(Self {
                rows: 0,
                pooled: 0,
                tokens,
                metadata_elements: 0,
            });
        }
        if tokens == 0 {
            return Err(reject("SPLADE selection requires at least one input token"));
        }
        let tokens_u32 = u32_size(tokens).map_err(ShapeError::Rejected)?;
        if tokens > i32::MAX as usize || pooled > i32::MAX as usize {
            return Err(reject("SPLADE selection exceeds i32 range"));
        }
        u32_size(pooled).map_err(ShapeError::Rejected)?;
        let sequences = cumulative_bounds(batch, tokens_u32)?;
        let mut rows = 0usize;
        for &row in &batch.pooled_indices {
            let (_, length) = cumulative_span(batch, sequences, row, tokens_u32)?;
            let length_usize =
                usize::try_from(length).map_err(|_| reject("SPLADE span length is invalid"))?;
            rows = rows
                .checked_add(length_usize)
                .ok_or_else(|| reject("SPLADE selected row count overflow"))?;
        }
        u32_size(rows).map_err(ShapeError::Rejected)?;
        if rows > i32::MAX as usize {
            return Err(reject("SPLADE selected row count exceeds i32 range"));
        }
        let pairs = pooled
            .checked_mul(2)
            .ok_or_else(|| reject("SPLADE metadata element overflow"))?;
        let metadata_elements = rows
            .checked_add(pairs)
            .ok_or_else(|| reject("SPLADE metadata element overflow"))?;
        u32_size(metadata_elements).map_err(ShapeError::Rejected)?;
        if metadata_elements > i32::MAX as usize {
            return Err(reject("SPLADE metadata exceeds i32 range"));
        }
        Ok(Self {
            rows,
            pooled,
            tokens,
            metadata_elements,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn pooled(&self) -> usize {
        self.pooled
    }
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }
    pub(crate) fn metadata_elements(&self) -> usize {
        self.metadata_elements
    }

    fn validate_batch(&self, batch: &Batch) -> Result<(), ShapeError> {
        if Self::new(batch)? != *self {
            return Err(reject("SPLADE selection geometry changed"));
        }
        Ok(())
    }

    pub(crate) fn write_metadata(&self, batch: &Batch, out: &mut [u32]) -> Result<(), ShapeError> {
        if out.len() != self.metadata_elements {
            return Err(reject("SPLADE metadata buffer has the wrong length"));
        }
        self.validate_batch(batch)?;
        let mut token_cursor = 0usize;
        let mut compact_cursor = 0u32;
        if self.pooled == 0 {
            return Ok(());
        }
        let tokens = u32_size(self.tokens).map_err(ShapeError::Rejected)?;
        let sequences = batch.cumulative_seq_lengths.len().saturating_sub(1);
        for (index, &row) in batch.pooled_indices.iter().enumerate() {
            let (start, length) = cumulative_span(batch, sequences, row, tokens)?;
            let length_usize =
                usize::try_from(length).map_err(|_| reject("SPLADE span length is invalid"))?;
            for offset in 0..length_usize {
                let out_index = token_cursor
                    .checked_add(offset)
                    .ok_or_else(|| reject("SPLADE metadata index overflow"))?;
                let offset_u32 = u32::try_from(offset)
                    .map_err(|_| reject("SPLADE token offset exceeds u32 range"))?;
                let value = start
                    .checked_add(offset_u32)
                    .ok_or_else(|| reject("SPLADE token row overflow"))?;
                *out.get_mut(out_index)
                    .ok_or_else(|| reject("SPLADE metadata token row is out of bounds"))? = value;
            }
            token_cursor = token_cursor
                .checked_add(length_usize)
                .ok_or_else(|| reject("SPLADE metadata index overflow"))?;
            let pair_offset = index
                .checked_mul(2)
                .ok_or_else(|| reject("SPLADE metadata span offset overflow"))?;
            let pair_index = self
                .rows
                .checked_add(pair_offset)
                .ok_or_else(|| reject("SPLADE metadata span index overflow"))?;
            let length_index = pair_index
                .checked_add(1)
                .ok_or_else(|| reject("SPLADE metadata span index overflow"))?;
            *out.get_mut(pair_index)
                .ok_or_else(|| reject("SPLADE metadata compact start is out of bounds"))? =
                compact_cursor;
            *out.get_mut(length_index)
                .ok_or_else(|| reject("SPLADE metadata compact length is out of bounds"))? = length;
            compact_cursor = compact_cursor
                .checked_add(length)
                .ok_or_else(|| reject("SPLADE compact span overflow"))?;
        }
        if token_cursor != self.rows || usize::try_from(compact_cursor).ok() != Some(self.rows) {
            return Err(reject("SPLADE metadata row count is inconsistent"));
        }
        Ok(())
    }
}

struct SpladeKernels {
    layer_norm: Function,
    pool: Function,
    gather: Function,
    relu: Option<Function>,
}

impl SpladeKernels {
    fn load(context: &Arc<Context>, separate_relu: bool) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| start(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        Ok(Self {
            layer_norm: module
                .load_function("infinity_plain_layer_norm_768_bf16_aten")
                .map_err(start)?,
            pool: module
                .load_function("infinity_splade_pool_bf16_f32")
                .map_err(start)?,
            gather: module
                .load_function("infinity_embedding_gather")
                .map_err(start)?,
            relu: if separate_relu {
                Some(
                    module
                        .load_function("infinity_encoder_activation_relu_bf16")
                        .map_err(start)?,
                )
            } else {
                None
            },
        })
    }
}

#[derive(Clone, Copy)]
struct Invocation {
    rows: usize,
    pooled: usize,
    source: u64,
    word: u64,
    decoder: u64,
    first: u64,
    second: u64,
    logits: u64,
    output: u64,
    metadata: u64,
    gelu: NativeDefaultPrepared,
    relu: NativeDefaultPrepared,
}

pub(crate) struct SpladeHead {
    context: Arc<Context>,
    stream: Arc<Stream>,
    transform_weight: ManuallyDrop<DeviceBuffer<Bf16>>,
    transform_bias: ManuallyDrop<DeviceBuffer<Bf16>>,
    gamma: ManuallyDrop<DeviceBuffer<Bf16>>,
    beta: ManuallyDrop<DeviceBuffer<Bf16>>,
    decoder_bias: ManuallyDrop<DeviceBuffer<Bf16>>,
    explicit_decoder: Option<ManuallyDrop<DeviceBuffer<Bf16>>>,
    hidden: usize,
    vocab: usize,
    eps: f32,
    gelu_plan: ManuallyDrop<LtPlan<Bf16>>,
    relu_plan: ManuallyDrop<LtPlan<Bf16>>,
    gelu_pending: NativeDefaultPending,
    relu_pending: NativeDefaultPending,
    invocation: Option<Invocation>,
    kernels: ManuallyDrop<SpladeKernels>,
    bytes: usize,
    released: bool,
}

fn retain_all_undrained<T>(stream: &Stream, buffers: &mut Vec<T>) {
    if stream.synchronize().is_err() {
        for buffer in buffers.drain(..) {
            std::mem::forget(buffer);
        }
    }
}

fn upload_keep(
    stream: &Stream,
    uploader: &mut WeightUploader,
    sources: &[&WeightSource],
    buffers: &mut Vec<DeviceBuffer<Bf16>>,
) -> Result<(), BackendError> {
    match uploader.upload(sources) {
        Ok(buffer) => {
            buffers.push(buffer);
            Ok(())
        }
        Err(error) => {
            retain_all_undrained(stream, buffers);
            Err(error)
        }
    }
}

fn next_uploaded(
    iterator: &mut std::vec::IntoIter<DeviceBuffer<Bf16>>,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    iterator
        .next()
        .ok_or_else(|| start("SPLADE weight staging ended early"))
}

impl SpladeHead {
    pub(crate) fn new(
        host: SpladeHeadHost,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(stream.context(), context) {
            return Err(start("SPLADE stream belongs to a different CUDA context"));
        }
        if !Arc::ptr_eq(lt.context(), context) {
            return Err(start("SPLADE provider belongs to a different CUDA context"));
        }
        if !std::ptr::eq(lt.stream(), stream.as_ref()) {
            return Err(start("SPLADE provider belongs to a different CUDA stream"));
        }
        if host.hidden != 768 || host.vocab == 0 {
            return Err(start(
                "SPLADE runtime requires hidden width768 and a positive vocabulary",
            ));
        }
        if !host.eps.is_finite() || host.eps <= 0.0 {
            return Err(start("SPLADE head requires a positive finite epsilon"));
        }
        let kernels = SpladeKernels::load(context, host.decoder_relu_separate)?;
        let gelu_plan =
            lt.prepare_native_default_gelu_bias_shape::<Bf16>(host.hidden, host.hidden, 1)?;
        let relu_plan = if host.decoder_relu_separate {
            lt.prepare_native_default_bias_shape::<Bf16>(host.hidden, host.vocab, 1)?
        } else {
            lt.prepare_native_default_relu_bias_shape::<Bf16>(host.hidden, host.vocab, 1)?
        };
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut uploaded = Vec::new();
        uploaded
            .try_reserve_exact(6)
            .map_err(|_| start("SPLADE weight staging allocation failed"))?;
        for source in [
            &host.transform.weight,
            &host.transform.bias,
            &host.gamma,
            &host.beta,
            &host.decoder_bias,
        ] {
            upload_keep(stream, &mut uploader, &[source], &mut uploaded)?;
        }
        let has_explicit = match &host.decoder {
            SpladeDecoderHost::TiedWord => false,
            SpladeDecoderHost::Explicit(source) => {
                upload_keep(stream, &mut uploader, &[&**source], &mut uploaded)?;
                true
            }
        };
        if let Err(error) = uploader.finish() {
            retain_all_undrained(stream, &mut uploaded);
            return Err(error);
        }
        let expected = if has_explicit { 6 } else { 5 };
        if uploaded.len() != expected {
            retain_all_undrained(stream, &mut uploaded);
            return Err(start("SPLADE weight staging count mismatch"));
        }
        let bytes = match uploaded.iter().try_fold(0usize, |total, buffer| {
            total.checked_add(buffer.num_bytes())
        }) {
            Some(bytes) => bytes,
            None => {
                retain_all_undrained(stream, &mut uploaded);
                return Err(start("SPLADE weight byte size overflow"));
            }
        };
        let mut iterator = uploaded.into_iter();
        let transform_weight = next_uploaded(&mut iterator)?;
        let transform_bias = next_uploaded(&mut iterator)?;
        let gamma = next_uploaded(&mut iterator)?;
        let beta = next_uploaded(&mut iterator)?;
        let decoder_bias = next_uploaded(&mut iterator)?;
        let explicit_decoder = if has_explicit {
            Some(ManuallyDrop::new(next_uploaded(&mut iterator)?))
        } else {
            None
        };
        Ok(Self {
            context: context.clone(),
            stream: stream.clone(),
            transform_weight: ManuallyDrop::new(transform_weight),
            transform_bias: ManuallyDrop::new(transform_bias),
            gamma: ManuallyDrop::new(gamma),
            beta: ManuallyDrop::new(beta),
            decoder_bias: ManuallyDrop::new(decoder_bias),
            explicit_decoder,
            hidden: host.hidden,
            vocab: host.vocab,
            eps: host.eps,
            gelu_plan: ManuallyDrop::new(gelu_plan),
            relu_plan: ManuallyDrop::new(relu_plan),
            gelu_pending: NativeDefaultPending::default(),
            relu_pending: NativeDefaultPending::default(),
            invocation: None,
            kernels: ManuallyDrop::new(kernels),
            bytes,
            released: false,
        })
    }

    pub(crate) fn output_width(&self) -> usize {
        self.vocab
    }
    pub(crate) fn work_bytes(&self, rows: usize, pooled: usize) -> Result<usize, BackendError> {
        work_bytes_for(self.hidden, self.vocab, rows, pooled)
    }
    pub(crate) fn work(
        &self,
        stream: Arc<Stream>,
        rows: usize,
        pooled: usize,
    ) -> Result<SpladeWork, BackendError> {
        SpladeWork::new(stream, self, rows, pooled)
    }

    pub(crate) unsafe fn prepare(
        &mut self,
        lt: &LtProvider,
        selection: SpladeSelection,
        source: &DeviceBuffer<Bf16>,
        word: &DeviceBuffer<Bf16>,
        work: &mut SpladeWork,
    ) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("SPLADE head was already released"));
        }
        self.invocation = None;
        self.gelu_pending.clear();
        self.relu_pending.clear();
        let rows = selection.rows();
        let pooled = selection.pooled();
        if pooled == 0 {
            return Ok(());
        }
        let input_elements = product(selection.tokens(), self.hidden).map_err(ShapeError::Fatal)?;
        if source.len() < input_elements {
            return Err(reject(
                "SPLADE source buffer does not hold every input token",
            ));
        }
        if !Arc::ptr_eq(source.context(), &self.context)
            || !Arc::ptr_eq(word.context(), &self.context)
        {
            return Err(fatal("SPLADE input buffer belongs to a different context"));
        }
        let decoder_elements = product(self.vocab, self.hidden).map_err(ShapeError::Fatal)?;
        let decoder = match self.explicit_decoder.as_ref() {
            Some(buffer) if buffer.len() == decoder_elements => &**buffer,
            Some(_) => return Err(reject("SPLADE explicit decoder has the wrong shape")),
            None if word.len() == decoder_elements => word,
            None => return Err(reject("SPLADE tied word embedding has the wrong shape")),
        };
        work.preflight(self, selection)?;
        let (first_ptr, second_ptr, logits_ptr, output_ptr, metadata_ptr, decoder_ptr) = (
            work.first_ptr(),
            work.second_ptr(),
            work.logits_ptr(),
            work.output().cu_deviceptr(),
            work.metadata_ptr(),
            decoder.cu_deviceptr(),
        );
        let gelu = unsafe {
            let (first, second) = work.first_second_mut();
            lt.prepare_native_default_bound(
                &mut *self.gelu_plan,
                &mut self.gelu_pending,
                rows,
                first,
                &*self.transform_weight,
                &*self.transform_bias,
                second,
            )?
        };
        let relu = unsafe {
            let (second, logits) = work.second_logits_mut();
            lt.prepare_native_default_bound(
                &mut *self.relu_plan,
                &mut self.relu_pending,
                rows,
                second,
                decoder,
                &*self.decoder_bias,
                logits,
            )?
        };
        self.invocation = Some(Invocation {
            rows,
            pooled,
            source: source.cu_deviceptr(),
            word: word.cu_deviceptr(),
            decoder: decoder_ptr,
            first: first_ptr,
            second: second_ptr,
            logits: logits_ptr,
            output: output_ptr,
            metadata: metadata_ptr,
            gelu,
            relu,
        });
        Ok(())
    }

    /// Execute only an already prepared invocation after the request prefix.
    /// Every failure is fatal; recoverable admission belongs to prepare().
    pub(crate) unsafe fn enqueue(
        &mut self,
        lt: &LtProvider,
        stream: &Stream,
        selection: SpladeSelection,
        source: &DeviceBuffer<Bf16>,
        word: &DeviceBuffer<Bf16>,
        work: &mut SpladeWork,
    ) -> Result<(), ShapeError> {
        // SAFETY: the caller's bound-context and retained-owner contract is
        // forwarded unchanged to the prepared executor.
        unsafe { self.enqueue_prepared(lt, stream, selection, source, word, work) }
            .map_err(|error| ShapeError::Fatal(error.into_backend()))
    }

    unsafe fn enqueue_prepared(
        &mut self,
        lt: &LtProvider,
        stream: &Stream,
        selection: SpladeSelection,
        source: &DeviceBuffer<Bf16>,
        word: &DeviceBuffer<Bf16>,
        work: &mut SpladeWork,
    ) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("SPLADE head was already released"));
        }
        let rows = selection.rows();
        let pooled = selection.pooled();
        if pooled == 0 {
            return Ok(());
        }
        if !std::ptr::eq(stream, &*self.stream) {
            return Err(fatal("SPLADE enqueue stream differs from the head stream"));
        }
        let invocation = match self.invocation {
            Some(invocation) => invocation,
            None => return Err(fatal("SPLADE head has no prepared invocation")),
        };
        if invocation.rows != rows || invocation.pooled != pooled {
            return Err(reject("SPLADE invocation shape changed"));
        }
        let decoder = match self.explicit_decoder.as_ref() {
            Some(buffer) => &**buffer,
            None => word,
        };
        if invocation.source != source.cu_deviceptr()
            || invocation.word != word.cu_deviceptr()
            || invocation.decoder != decoder.cu_deviceptr()
            || invocation.first != work.first_ptr()
            || invocation.second != work.second_ptr()
            || invocation.logits != work.logits_ptr()
            || invocation.output != work.output().cu_deviceptr()
            || invocation.metadata != work.metadata_ptr()
        {
            return Err(reject("SPLADE invocation buffer identity changed"));
        }
        work.preflight(self, selection)?;
        let gather_elements = product(rows, self.hidden).map_err(ShapeError::Fatal)?;
        let gather_grid = blocks(gather_elements).map_err(ShapeError::Fatal)?;
        let mut tokens_ptr = work.metadata_ptr();
        let mut source_ptr = source.cu_deviceptr();
        let mut first_ptr = work.first_ptr();
        let mut rows_arg = checked_i32(rows, "SPLADE gather rows")?;
        let mut hidden_arg = checked_i32(self.hidden, "SPLADE hidden width")?;
        crate::launch_threads(
            &self.kernels.gather,
            stream,
            gather_grid,
            256,
            0,
            &mut [
                crate::arg(&mut tokens_ptr),
                crate::arg(&mut source_ptr),
                crate::arg(&mut first_ptr),
                crate::arg(&mut rows_arg),
                crate::arg(&mut hidden_arg),
            ],
        )
        .map_err(ShapeError::Fatal)?;
        unsafe {
            let (first, second) = work.first_second_mut();
            lt.launch_native_default_bound(
                &mut *self.gelu_plan,
                &invocation.gelu,
                first,
                &*self.transform_weight,
                second,
            )?;
        }
        let second_ptr = work.second_ptr();
        let mut layer_input = second_ptr;
        let mut gamma_ptr = self.gamma.cu_deviceptr();
        let mut beta_ptr = self.beta.cu_deviceptr();
        let mut layer_output = second_ptr;
        let mut layer_rows = checked_i32(rows, "SPLADE layer-norm rows")?;
        let mut layer_width = checked_i32(self.hidden, "SPLADE layer-norm width")?;
        let mut epsilon = self.eps;
        let layer_grid_x = u32_size(rows).map_err(ShapeError::Fatal)?;
        crate::launch_threads(
            &self.kernels.layer_norm,
            stream,
            (layer_grid_x, 1, 1),
            128,
            0,
            &mut [
                crate::arg(&mut layer_input),
                crate::arg(&mut gamma_ptr),
                crate::arg(&mut beta_ptr),
                crate::arg(&mut layer_output),
                crate::arg(&mut layer_rows),
                crate::arg(&mut layer_width),
                crate::arg(&mut epsilon),
            ],
        )
        .map_err(ShapeError::Fatal)?;
        unsafe {
            let (second, logits) = work.second_logits_mut();
            lt.launch_native_default_bound(
                &mut *self.relu_plan,
                &invocation.relu,
                second,
                decoder,
                logits,
            )?;
        }
        if let Some(relu) = self.kernels.relu.as_ref() {
            // Native Distil materializes BF16 logits before its separate ReLU.
            // Reuse the existing pointwise activation without a fused epilogue.
            let mut input = work.logits_ptr();
            let mut output = input;
            let mut bias = 0u64;
            let mut count = u32_size(rows).map_err(ShapeError::Fatal)?;
            let mut width = u32_size(self.vocab).map_err(ShapeError::Fatal)?;
            let mut input_stride = width;
            let mut output_stride = width;
            crate::launch_threads(
                relu,
                stream,
                crate::blocks(product(rows, self.vocab).map_err(ShapeError::Fatal)?)
                    .map_err(ShapeError::Fatal)?,
                256,
                0,
                &mut [
                    crate::arg(&mut input),
                    crate::arg(&mut bias),
                    crate::arg(&mut output),
                    crate::arg(&mut count),
                    crate::arg(&mut width),
                    crate::arg(&mut input_stride),
                    crate::arg(&mut output_stride),
                ],
            )
            .map_err(ShapeError::Fatal)?;
        }
        let compact_offset = rows
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| fatal("SPLADE compact span offset overflow"))?;
        let compact_offset = u64::try_from(compact_offset)
            .map_err(|_| fatal("SPLADE compact span offset exceeds u64 range"))?;
        let mut logits_ptr = work.logits_ptr();
        let mut compact_ptr = work
            .metadata_ptr()
            .checked_add(compact_offset)
            .ok_or_else(|| fatal("SPLADE compact span pointer overflow"))?;
        let mut output_ptr = work.output().cu_deviceptr();
        let mut vocab_arg = checked_i32(self.vocab, "SPLADE vocabulary width")?;
        let mut pooled_arg = checked_i32(pooled, "SPLADE pooled rows")?;
        let pool_items = ceil_div(self.vocab, 32)?
            .checked_mul(pooled)
            .ok_or_else(|| fatal("SPLADE pool grid overflow"))?;
        let pool_grid_x = u32_size(pool_items).map_err(ShapeError::Fatal)?;
        crate::launch_threads(
            &self.kernels.pool,
            stream,
            (pool_grid_x, 1, 1),
            128,
            0,
            &mut [
                crate::arg(&mut logits_ptr),
                crate::arg(&mut compact_ptr),
                crate::arg(&mut output_ptr),
                crate::arg(&mut vocab_arg),
                crate::arg(&mut pooled_arg),
            ],
        )
        .map_err(ShapeError::Fatal)?;
        Ok(())
    }

    unsafe fn drop_after_drain(&mut self) {
        if self.released {
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.transform_weight);
            ManuallyDrop::drop(&mut self.transform_bias);
            ManuallyDrop::drop(&mut self.gamma);
            ManuallyDrop::drop(&mut self.beta);
            ManuallyDrop::drop(&mut self.decoder_bias);
            if let Some(decoder) = self.explicit_decoder.as_mut() {
                ManuallyDrop::drop(decoder);
            }
            ManuallyDrop::drop(&mut self.gelu_plan);
            ManuallyDrop::drop(&mut self.relu_plan);
            ManuallyDrop::drop(&mut self.kernels);
        }
        self.released = true;
    }
}

impl Drop for SpladeHead {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            unsafe { self.drop_after_drain() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

struct WorkBuilder {
    stream: Arc<Stream>,
    bf16: Vec<DeviceBuffer<Bf16>>,
    f32_out: Option<DeviceBuffer<f32>>,
    u32_out: Option<DeviceBuffer<u32>>,
    committed: bool,
}

impl WorkBuilder {
    fn new(stream: Arc<Stream>) -> Result<Self, BackendError> {
        let mut bf16 = Vec::new();
        bf16.try_reserve_exact(3)
            .map_err(|_| infer("SPLADE work allocation failed"))?;
        Ok(Self {
            stream,
            bf16,
            f32_out: None,
            u32_out: None,
            committed: false,
        })
    }
    fn bf16(&mut self, elements: usize) -> Result<(), BackendError> {
        let buffer = DeviceBuffer::<Bf16>::zeroed(&self.stream, elements).map_err(infer)?;
        self.bf16.push(buffer);
        Ok(())
    }
    fn f32(&mut self, elements: usize) -> Result<(), BackendError> {
        self.f32_out = Some(DeviceBuffer::<f32>::zeroed(&self.stream, elements).map_err(infer)?);
        Ok(())
    }
    fn u32(&mut self, elements: usize) -> Result<(), BackendError> {
        self.u32_out = Some(DeviceBuffer::<u32>::zeroed(&self.stream, elements).map_err(infer)?);
        Ok(())
    }
}

impl Drop for WorkBuilder {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if self.stream.synchronize().is_err() {
            for buffer in self.bf16.drain(..) {
                std::mem::forget(buffer);
            }
            if let Some(buffer) = self.f32_out.take() {
                std::mem::forget(buffer);
            }
            if let Some(buffer) = self.u32_out.take() {
                std::mem::forget(buffer);
            }
        }
    }
}

pub(crate) struct SpladeWork {
    stream: Arc<Stream>,
    rows_capacity: usize,
    pooled_capacity: usize,
    bytes: usize,
    first: ManuallyDrop<DeviceBuffer<Bf16>>,
    second: ManuallyDrop<DeviceBuffer<Bf16>>,
    logits: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<f32>>,
    metadata: ManuallyDrop<DeviceBuffer<u32>>,
    released: bool,
}

impl SpladeWork {
    pub(crate) fn new(
        stream: Arc<Stream>,
        head: &SpladeHead,
        rows_capacity: usize,
        pooled_capacity: usize,
    ) -> Result<Self, BackendError> {
        if pooled_capacity == 0 || rows_capacity == 0 {
            return Err(infer("SPLADE work requires positive capacities"));
        }
        if !Arc::ptr_eq(&stream, &head.stream) {
            return Err(infer("SPLADE work stream differs from the head stream"));
        }
        let bytes = head.work_bytes(rows_capacity, pooled_capacity)?;
        let scratch = product(rows_capacity, head.hidden)?;
        let logits = product(rows_capacity, head.vocab)?;
        let output = product(pooled_capacity, head.vocab)?;
        let pairs = pooled_capacity
            .checked_mul(2)
            .ok_or_else(|| infer("SPLADE metadata element overflow"))?;
        let metadata = rows_capacity
            .checked_add(pairs)
            .ok_or_else(|| infer("SPLADE metadata element overflow"))?;
        let mut builder = WorkBuilder::new(stream.clone())?;
        builder.bf16(scratch)?;
        builder.bf16(scratch)?;
        builder.bf16(logits)?;
        builder.f32(output)?;
        builder.u32(metadata)?;
        if builder.bf16.len() != 3 || builder.f32_out.is_none() || builder.u32_out.is_none() {
            return Err(infer("SPLADE work allocation count mismatch"));
        }
        let mut bf16 = std::mem::take(&mut builder.bf16).into_iter();
        let first = match bf16.next() {
            Some(buffer) => buffer,
            None => {
                builder.committed = true;
                return Err(infer("SPLADE work allocation ended early"));
            }
        };
        let second = match bf16.next() {
            Some(buffer) => buffer,
            None => {
                builder.committed = true;
                return Err(infer("SPLADE work allocation ended early"));
            }
        };
        let logits = match bf16.next() {
            Some(buffer) => buffer,
            None => {
                builder.committed = true;
                return Err(infer("SPLADE work allocation ended early"));
            }
        };
        let output = builder
            .f32_out
            .take()
            .ok_or_else(|| infer("SPLADE work output is missing"))?;
        let metadata = builder
            .u32_out
            .take()
            .ok_or_else(|| infer("SPLADE work metadata is missing"))?;
        builder.committed = true;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            rows_capacity,
            pooled_capacity,
            bytes,
            first: ManuallyDrop::new(first),
            second: ManuallyDrop::new(second),
            logits: ManuallyDrop::new(logits),
            output: ManuallyDrop::new(output),
            metadata: ManuallyDrop::new(metadata),
            released: false,
        })
    }

    pub(crate) fn fits(&self, rows: usize, pooled: usize) -> bool {
        if rows > self.rows_capacity || pooled > self.pooled_capacity {
            return false;
        }
        let Some(metadata) = pooled
            .checked_mul(2)
            .and_then(|pairs| rows.checked_add(pairs))
        else {
            return false;
        };
        metadata <= self.metadata().len()
    }
    pub(crate) fn capacity(&self) -> (usize, usize) {
        (self.rows_capacity, self.pooled_capacity)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }
    pub(crate) fn metadata(&self) -> &DeviceBuffer<u32> {
        &self.metadata
    }
    fn first(&self) -> &DeviceBuffer<Bf16> {
        &self.first
    }
    fn second(&self) -> &DeviceBuffer<Bf16> {
        &self.second
    }
    fn logits(&self) -> &DeviceBuffer<Bf16> {
        &self.logits
    }
    fn first_ptr(&self) -> u64 {
        self.first().cu_deviceptr()
    }
    fn second_ptr(&self) -> u64 {
        self.second().cu_deviceptr()
    }
    fn logits_ptr(&self) -> u64 {
        self.logits().cu_deviceptr()
    }
    fn metadata_ptr(&self) -> u64 {
        self.metadata().cu_deviceptr()
    }
    fn first_second_mut(&mut self) -> (&DeviceBuffer<Bf16>, &mut DeviceBuffer<Bf16>) {
        (&*self.first, &mut *self.second)
    }
    fn second_logits_mut(&mut self) -> (&DeviceBuffer<Bf16>, &mut DeviceBuffer<Bf16>) {
        (&*self.second, &mut *self.logits)
    }

    fn preflight(&self, head: &SpladeHead, selection: SpladeSelection) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("SPLADE workspace was already released"));
        }
        let (context, stream) = (&head.context, &head.stream);
        let (rows, pooled, hidden, vocab, metadata) = (
            selection.rows(),
            selection.pooled(),
            head.hidden,
            head.vocab,
            selection.metadata_elements(),
        );
        if !Arc::ptr_eq(&self.stream, stream) {
            return Err(fatal("SPLADE work belongs to a different CUDA stream"));
        }
        for buffer_context in [
            self.first().context(),
            self.second().context(),
            self.logits().context(),
            self.output().context(),
            self.metadata().context(),
        ] {
            if !Arc::ptr_eq(buffer_context, context) {
                return Err(fatal(
                    "SPLADE work buffer belongs to a different CUDA context",
                ));
            }
        }
        if !self.fits(rows, pooled) {
            return Err(reject("SPLADE work capacity is too small"));
        }
        let scratch = product(rows, hidden).map_err(ShapeError::Fatal)?;
        let logits = product(rows, vocab).map_err(ShapeError::Fatal)?;
        let output = product(pooled, vocab).map_err(ShapeError::Fatal)?;
        if self.first().len() < scratch
            || self.second().len() < scratch
            || self.logits().len() < logits
            || self.output().len() < output
            || self.metadata().len() < metadata
        {
            return Err(reject("SPLADE work buffer extent is too small"));
        }
        Ok(())
    }

    pub(crate) unsafe fn drop_after_drain(&mut self) {
        if self.released {
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.first);
            ManuallyDrop::drop(&mut self.second);
            ManuallyDrop::drop(&mut self.logits);
            ManuallyDrop::drop(&mut self.output);
            ManuallyDrop::drop(&mut self.metadata);
        }
        self.released = true;
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }
}

impl Drop for SpladeWork {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            unsafe { self.drop_after_drain() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(
        input_ids: Vec<u32>,
        pooled_indices: Vec<u32>,
        cumulative_seq_lengths: Vec<u32>,
        raw_indices: Vec<u32>,
    ) -> Batch {
        Batch {
            multimodal: Vec::new(),
            input_ids,
            token_type_ids: Vec::new(),
            position_ids: Vec::new(),
            cumulative_seq_lengths,
            max_length: 0,
            pooled_indices,
            raw_indices,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    #[test]
    fn selection_ragged_repeated_metadata() {
        let batch = batch(vec![0; 6], vec![1, 0, 1], vec![0, 2, 5, 6], vec![4, 3]);
        let selection = SpladeSelection::new(&batch).unwrap();
        assert_eq!(
            (
                selection.rows(),
                selection.pooled(),
                selection.tokens(),
                selection.metadata_elements()
            ),
            (8, 3, 6, 14)
        );
        let mut metadata = vec![0u32; selection.metadata_elements()];
        selection.write_metadata(&batch, &mut metadata).unwrap();
        assert_eq!(metadata, vec![2, 3, 4, 0, 1, 2, 3, 4, 0, 3, 3, 2, 5, 3]);
    }

    #[test]
    fn selection_duplicates_ignore_raw_indices() {
        let batch = batch(vec![0; 6], vec![0, 0], vec![0, 2, 5, 6], vec![2, 1]);
        let selection = SpladeSelection::new(&batch).unwrap();
        assert_eq!((selection.rows(), selection.pooled()), (4, 2));
        let mut metadata = vec![0u32; selection.metadata_elements()];
        selection.write_metadata(&batch, &mut metadata).unwrap();
        assert_eq!(metadata, vec![0, 1, 0, 1, 0, 2, 2, 2]);
    }

    #[test]
    fn selection_empty_pooled_has_no_metadata() {
        let batch = batch(vec![1, 2], Vec::new(), vec![0, 2], vec![0]);
        let selection = SpladeSelection::new(&batch).unwrap();
        assert_eq!(
            (
                selection.rows(),
                selection.pooled(),
                selection.metadata_elements()
            ),
            (0, 0, 0)
        );
        selection.write_metadata(&batch, &mut []).unwrap();
        let mut metadata = [9u32];
        assert!(selection.write_metadata(&batch, &mut metadata).is_err());
        assert_eq!(metadata, [9u32]);
    }

    #[test]
    fn selection_rejects_malformed_batches() {
        assert!(
            SpladeSelection::new(&batch(vec![0; 5], vec![2], vec![0, 2, 5], Vec::new())).is_err()
        );
        assert!(
            SpladeSelection::new(&batch(vec![0; 2], vec![0], vec![0, 3, 2], Vec::new())).is_err()
        );
        assert!(
            SpladeSelection::new(&batch(vec![0; 2], vec![1], vec![0, 2, 2], Vec::new())).is_err()
        );
        assert!(
            SpladeSelection::new(&batch(vec![0; 4], vec![0], vec![0, 2, 5], Vec::new())).is_err()
        );
    }

    #[test]
    fn selection_short_metadata_buffer_is_untouched() {
        let batch = batch(vec![0; 6], vec![1, 0, 1], vec![0, 2, 5, 6], Vec::new());
        let selection = SpladeSelection::new(&batch).unwrap();
        let mut metadata = vec![7u32; selection.metadata_elements() - 1];
        assert!(selection.write_metadata(&batch, &mut metadata).is_err());
        assert!(metadata.iter().all(|&value| value == 7));
    }

    #[test]
    fn work_bytes_are_the_sum_of_the_five_buffers() {
        let expected = 2 * (2 * 768) * 2 + 2 * 30522 * 2 + 3 * 30522 * 4 + (2 + 2 * 3) * 4;
        assert_eq!(work_bytes_for(768, 30522, 2, 3).unwrap(), expected);
        assert_eq!(work_bytes_for(768, 30522, 2, 0).unwrap(), 0);
    }

    #[test]
    fn work_bytes_checked_overflow() {
        assert!(work_bytes_for(768, 30522, usize::MAX, 1).is_err());
        assert!(work_bytes_for(usize::MAX, usize::MAX, 2, 1).is_err());
    }
    #[test]
    #[ignore = "Root GPU: actual trained SPLADE head golden and kernel71"]
    fn trained_runtime_head_matches_native_gold_with_in_place_ln_and_borrowed_word() {
        use crate::model::{BertConfig, Checkpoint};
        use sha2::{Digest, Sha256};
        use std::{fs, path::PathBuf};
        let gold_root = PathBuf::from(std::env::var_os("INFINITY_OXIDE_SPLADE_GOLD").unwrap());
        let model = PathBuf::from(std::env::var_os("INFINITY_OXIDE_TEST_MODEL").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(gold_root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["status"], "passed");
        let cases = manifest["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 7);
        let config = BertConfig::load(&model).unwrap();
        let checkpoint = Checkpoint::open(&model).unwrap();
        let word = checkpoint
            .source(
                "bert.embeddings.word_embeddings.weight",
                &[config.vocab_size, config.hidden_size],
            )
            .unwrap();
        let decoder = checkpoint
            .source(
                "cls.predictions.decoder.weight",
                &[config.vocab_size, config.hidden_size],
            )
            .unwrap();
        // Qualify the tied path only after checking every real decoder/word
        // element locally. The immutable original checkpoint is not rewritten.
        let mut offset = 0;
        let mut a = vec![0u16; 16384];
        let mut b = vec![0u16; 16384];
        while offset < word.len() {
            let n = (word.len() - offset).min(a.len());
            word.read_bf16(offset, &mut a[..n]).unwrap();
            decoder.read_bf16(offset, &mut b[..n]).unwrap();
            assert_eq!(
                &a[..n],
                &b[..n],
                "trained decoder is not tied at offset{offset}"
            );
            offset += n;
        }
        // Validate every actual reference file before CUDA starts.
        let mut data = Vec::new();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let rows = case["rows"].as_u64().unwrap() as usize;
            let read = |stage: &str| {
                let info = &case["stages"][stage];
                let path = gold_root.join(name).join(info["file"].as_str().unwrap());
                let bytes = fs::read(path).unwrap();
                assert_eq!(bytes.len(), info["bytes"].as_u64().unwrap() as usize);
                assert_eq!(
                    format!("{:x}", Sha256::digest(&bytes)),
                    info["sha256"].as_str().unwrap()
                );
                bytes
            };
            let input = read("input")
                .chunks_exact(2)
                .map(|x| Bf16::from_bits(u16::from_le_bytes([x[0], x[1]])))
                .collect::<Vec<_>>();
            assert_eq!(input.len(), rows * 768);
            let pooled = read("pooled")
                .chunks_exact(4)
                .take(config.vocab_size)
                .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect::<Vec<_>>();
            assert_eq!(pooled.len(), config.vocab_size);
            data.push((name.to_owned(), rows, input, pooled));
        }
        assert!(Context::device_count().unwrap() > 0);
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let lt = LtProvider::new_native_default(context.clone(), stream.clone()).unwrap();
        let mut uploader = WeightUploader::new(stream.clone()).unwrap();
        let device_word = uploader.upload(&[&word]).unwrap();
        uploader.finish().unwrap();
        for tied in [false, true] {
            let mut host = SpladeHeadHost::load(&checkpoint, &config, &word).unwrap();
            if tied {
                host.decoder = SpladeDecoderHost::TiedWord;
            }
            let mut head = SpladeHead::new(host, &context, &stream, &lt).unwrap();
            assert_eq!(head.explicit_decoder.is_none(), tied);
            for (name, rows, original, expected) in &data {
                let b = batch(vec![0; *rows], vec![0], vec![0, *rows as u32], vec![]);
                let selection = SpladeSelection::new(&b).unwrap();
                let mut metadata = vec![0; selection.metadata_elements()];
                selection.write_metadata(&b, &mut metadata).unwrap();
                let mut source = original.clone();
                source.resize(
                    rows.next_power_of_two().max(32) * 768,
                    Bf16::from_bits(0x4a42),
                );
                let input = DeviceBuffer::from_host(&stream, &source).unwrap();
                let mut work = head
                    .work(stream.clone(), rows.next_power_of_two(), 1)
                    .unwrap();
                context.bind_to_thread().unwrap();
                unsafe { head.prepare(&lt, selection, &input, &device_word, &mut work) }.unwrap();
                let queries = (
                    lt.native_default_queries(&head.gelu_plan),
                    lt.native_default_queries(&head.relu_plan),
                );
                for _ in 0..2 {
                    unsafe {
                        cuda_core::memory::memcpy_htod_async(
                            work.metadata().cu_deviceptr(),
                            metadata.as_ptr(),
                            metadata.len() * 4,
                            stream.cu_stream(),
                        )
                    }
                    .unwrap();
                    unsafe {
                        head.enqueue(&lt, &stream, selection, &input, &device_word, &mut work)
                    }
                    .unwrap();
                    let actual = work.output().to_host_vec(&stream).unwrap();
                    let words = actual[..config.vocab_size]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    let differences = words
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(8)
                        .map(|(i, (a, b))| (i, *a, *b))
                        .collect::<Vec<_>>();
                    assert!(
                        differences.is_empty(),
                        "{name} tied={tied} differing F32 words{differences:?}"
                    );
                    assert_eq!(
                        input.to_host_vec(&stream).unwrap(),
                        source,
                        "source overwritten"
                    );
                    unsafe { head.prepare(&lt, selection, &input, &device_word, &mut work) }
                        .unwrap();
                    assert_eq!(
                        (
                            lt.native_default_queries(&head.gelu_plan),
                            lt.native_default_queries(&head.relu_plan)
                        ),
                        queries,
                        "steady replay reran heuristic"
                    );
                }
            }
        }
        stream.synchronize().unwrap();
    }
}
