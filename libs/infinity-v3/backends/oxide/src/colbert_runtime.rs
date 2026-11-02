// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! GPU projection, model-dtype L2 normalization and attention-key gathering
//! for the BF16 BERT ColBERT head.

use crate::{
    colbert::{ColbertHeadHost, ColbertSelection},
    cublas::Blas,
    infer,
    math::{GemmArguments, LinearPlan},
    product, start, u32_size,
};
use crate::{
    lt::{LtPlan, LtProvider, NativeDefaultPending, ShapeError},
    tensor::Bf16,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::{
    mem::{size_of, ManuallyDrop},
    sync::{atomic::Ordering, Arc},
};

fn fatal(m: impl std::fmt::Display) -> ShapeError {
    ShapeError::Fatal(infer(m))
}

fn checked_bytes(elements: usize, width: usize, label: &str) -> Result<usize, BackendError> {
    elements
        .checked_mul(width)
        .ok_or_else(|| infer(format!("{label} byte size overflow")))
}

pub(crate) fn work_bytes_for(
    output: usize,
    tokens: usize,
    sequences: usize,
) -> Result<usize, BackendError> {
    if output == 0 || tokens == 0 {
        return Err(infer(
            "ColBERT work requires positive token and output widths",
        ));
    }
    if tokens > i32::MAX as usize || sequences > i32::MAX as usize {
        return Err(infer("ColBERT work shape exceeds i32 launch range"));
    }
    let projected = product(tokens, output)?;
    let mut bytes = checked_bytes(projected, size_of::<u16>(), "ColBERT projection")?;
    bytes = bytes
        .checked_add(checked_bytes(
            projected,
            size_of::<f32>(),
            "ColBERT normalized output",
        )?)
        .ok_or_else(|| infer("ColBERT work byte overflow"))?;
    bytes = bytes
        .checked_add(checked_bytes(
            tokens,
            size_of::<u32>(),
            "ColBERT key indices",
        )?)
        .ok_or_else(|| infer("ColBERT work byte overflow"))?;
    bytes = bytes
        .checked_add(checked_bytes(
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("ColBERT key offset count overflow"))?,
            size_of::<u32>(),
            "ColBERT key offsets",
        )?)
        .ok_or_else(|| infer("ColBERT work byte overflow"))?;
    u32_size(projected)?;
    Ok(bytes)
}

struct ColbertKernels {
    gather: Function,
    l2: Function,
}

impl ColbertKernels {
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
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
            gather: module
                .load_function("infinity_attention_row_gather_bf16")
                .map_err(start)?,
            l2: module
                .load_function("infinity_colbert_l2_bf16_f32")
                .map_err(start)?,
        })
    }
}

pub(crate) struct ColbertHead {
    context: Arc<Context>,
    stream: Arc<Stream>,
    projection_weight: ManuallyDrop<DeviceBuffer<Bf16>>,
    projection_bias: ManuallyDrop<Option<DeviceBuffer<Bf16>>>,
    hidden: usize,
    output: usize,
    eps: f32,
    plan: ManuallyDrop<Option<LtPlan<Bf16>>>,
    pending: NativeDefaultPending,
    prepared: Option<PreparedProjection>,
    kernels: ManuallyDrop<ColbertKernels>,
    bytes: usize,
    released: bool,
}

enum PreparedProjection {
    Biased(crate::lt::NativeDefaultPrepared),
    Unbiased(GemmArguments),
}

fn retain_all_undrained<T>(stream: &Stream, buffers: &mut Vec<T>) {
    if stream.synchronize().is_err() {
        for buffer in buffers.drain(..) {
            std::mem::forget(buffer);
        }
    }
}

fn next_uploaded(
    iterator: &mut std::vec::IntoIter<DeviceBuffer<Bf16>>,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    iterator
        .next()
        .ok_or_else(|| start("ColBERT weight staging ended early"))
}

impl ColbertHead {
    pub(crate) fn new(
        host: ColbertHeadHost,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(stream.context(), context) {
            return Err(start("ColBERT stream belongs to a different CUDA context"));
        }
        if !Arc::ptr_eq(lt.context(), context) {
            return Err(start(
                "ColBERT provider belongs to a different CUDA context",
            ));
        }
        if !std::ptr::eq(lt.stream(), stream.as_ref()) {
            return Err(start("ColBERT provider belongs to a different CUDA stream"));
        }
        if host.hidden == 0 || host.output == 0 {
            return Err(start(
                "ColBERT runtime requires positive hidden and output widths",
            ));
        }
        if !host.eps.is_finite() || host.eps <= 0.0 {
            return Err(start("ColBERT head requires a positive finite epsilon"));
        }
        let kernels = ColbertKernels::load(context)?;
        // Native calls plain matmul when linear.bias is absent. A fabricated
        // zero bias would select a different Lt epilogue/algorithm.
        let plan = if host.has_bias {
            Some(lt.prepare_native_default_bias_shape::<Bf16>(host.hidden, host.output, 1)?)
        } else {
            None
        };
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut uploaded = Vec::new();
        uploaded
            .try_reserve_exact(2)
            .map_err(|_| start("ColBERT weight staging allocation failed"))?;
        let sources = [&host.projection.weight, &host.projection.bias];
        let source_count = if host.has_bias { 2 } else { 1 };
        for source in sources.into_iter().take(source_count) {
            match uploader.upload(&[source]) {
                Ok(buffer) => uploaded.push(buffer),
                Err(error) => {
                    retain_all_undrained(stream, &mut uploaded);
                    return Err(error);
                }
            }
        }
        if let Err(error) = uploader.finish() {
            retain_all_undrained(stream, &mut uploaded);
            return Err(error);
        }
        if uploaded.len() != source_count {
            retain_all_undrained(stream, &mut uploaded);
            return Err(start("ColBERT weight staging count mismatch"));
        }
        let bytes = match uploaded.iter().try_fold(0usize, |total, buffer| {
            total.checked_add(buffer.num_bytes())
        }) {
            Some(bytes) => bytes,
            None => {
                retain_all_undrained(stream, &mut uploaded);
                return Err(start("ColBERT weight byte size overflow"));
            }
        };
        let mut iterator = uploaded.into_iter();
        let projection_weight = next_uploaded(&mut iterator)?;
        let projection_bias = if host.has_bias {
            Some(next_uploaded(&mut iterator)?)
        } else {
            None
        };
        Ok(Self {
            context: context.clone(),
            stream: stream.clone(),
            projection_weight: ManuallyDrop::new(projection_weight),
            projection_bias: ManuallyDrop::new(projection_bias),
            hidden: host.hidden,
            output: host.output,
            eps: host.eps,
            plan: ManuallyDrop::new(plan),
            pending: NativeDefaultPending::default(),
            prepared: None,
            kernels: ManuallyDrop::new(kernels),
            bytes,
            released: false,
        })
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output
    }

    pub(crate) fn work_bytes(
        &self,
        tokens: usize,
        sequences: usize,
    ) -> Result<usize, BackendError> {
        work_bytes_for(self.output, tokens, sequences)
    }

    pub(crate) fn work(
        &self,
        stream: Arc<Stream>,
        tokens: usize,
        sequences: usize,
    ) -> Result<ColbertWork, BackendError> {
        ColbertWork::new(stream, self, tokens, sequences)
    }

    pub(crate) unsafe fn prepare(
        &mut self,
        lt: &LtProvider,
        selection: &ColbertSelection,
        source: &DeviceBuffer<Bf16>,
        work: &mut ColbertWork,
    ) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("ColBERT head was already released"));
        }
        self.prepared = None;
        self.pending.clear();
        work.preflight(self, selection)?;
        let tokens = selection.tokens();
        let input_elements = product(tokens, self.hidden).map_err(ShapeError::Fatal)?;
        if source.len() < input_elements {
            return Err(ShapeError::Rejected(infer(
                "ColBERT source buffer does not hold every input token",
            )));
        }
        if !Arc::ptr_eq(source.context(), &self.context) {
            return Err(fatal("ColBERT input buffer belongs to a different context"));
        }
        // SAFETY: the caller retains the source, weights and work buffers
        // through the bound launch on the same stream.
        self.prepared = Some(if let Some(bias) = self.projection_bias.as_ref() {
            let plan = self
                .plan
                .as_mut()
                .ok_or_else(|| fatal("Missing biased ColBERT plan"))?;
            PreparedProjection::Biased(unsafe {
                lt.prepare_native_default_bound(
                    plan,
                    &mut self.pending,
                    tokens,
                    source,
                    &*self.projection_weight,
                    bias,
                    work.projected_mut(),
                )?
            })
        } else {
            let plan = LinearPlan::<Bf16>::contiguous(tokens, self.hidden, self.output)
                .map_err(|error| ShapeError::Rejected(infer(error)))?;
            PreparedProjection::Unbiased(
                plan.bind(source, &*self.projection_weight, work.projected_mut())
                    .map_err(|error| ShapeError::Rejected(infer(error)))?,
            )
        });
        Ok(())
    }

    pub(crate) fn upload_selection(
        &self,
        stream: &Stream,
        selection: &ColbertSelection,
        work: &mut ColbertWork,
    ) -> Result<(), ShapeError> {
        if !std::ptr::eq(stream, &*self.stream) {
            return Err(fatal("ColBERT upload stream differs from the head stream"));
        }
        work.preflight(self, selection)?;
        work.upload_selection(stream, selection)
    }

    /// Gather K and V rows for the selected attention keys. `source_k` and
    /// `source_v` are [tokens, heads, dim] views inside the packed QKV buffer;
    /// `row_stride` is the packed stride in elements.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gather_kv<E: crate::upload::UploadElement>(
        &self,
        stream: &Stream,
        source_k: u64,
        source_v: u64,
        row_stride: usize,
        out_k: &DeviceBuffer<E>,
        out_v: &DeviceBuffer<E>,
        selection: &ColbertSelection,
        work: &ColbertWork,
    ) -> Result<(), ShapeError> {
        if !selection.excludes_attention_keys() {
            return Ok(());
        }
        if !std::ptr::eq(stream, &*self.stream) {
            return Err(fatal("ColBERT gather stream differs from the head stream"));
        }
        let rows = selection.keys();
        let elements = product(rows, self.hidden).map_err(ShapeError::Fatal)?;
        if out_k.len() < elements || out_v.len() < elements {
            return Err(ShapeError::Rejected(infer(
                "ColBERT gathered K/V buffer is too small",
            )));
        }
        let grid = crate::blocks(elements).map_err(ShapeError::Fatal)?;
        let mut indices = work.key_indices_ptr();
        let mut count = u32_size(rows).map_err(ShapeError::Fatal)?;
        let mut width = u32_size(self.hidden).map_err(ShapeError::Fatal)?;
        let mut stride = u32_size(row_stride).map_err(ShapeError::Fatal)?;
        for (source, output) in [(source_k, out_k), (source_v, out_v)] {
            let mut source = source;
            let mut output = output.cu_deviceptr();
            crate::launch_threads(
                &self.kernels.gather,
                stream,
                grid,
                256,
                0,
                &mut [
                    crate::arg(&mut source),
                    crate::arg(&mut indices),
                    crate::arg(&mut output),
                    crate::arg(&mut count),
                    crate::arg(&mut width),
                    crate::arg(&mut stride),
                ],
            )
            .map_err(ShapeError::Fatal)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn enqueue(
        &mut self,
        lt: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        selection: &ColbertSelection,
        source: &DeviceBuffer<Bf16>,
        work: &mut ColbertWork,
    ) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("ColBERT head was already released"));
        }
        if !std::ptr::eq(stream, &*self.stream) {
            return Err(fatal("ColBERT enqueue stream differs from the head stream"));
        }
        work.preflight(self, selection)?;
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| fatal("ColBERT head has no prepared projection"))?;
        // SAFETY: prepare() built the actual-M plan from these exact buffers.
        match prepared {
            PreparedProjection::Biased(prepared) => unsafe {
                let plan = self
                    .plan
                    .as_mut()
                    .ok_or_else(|| fatal("Missing biased ColBERT plan"))?;
                lt.launch_native_default_bound(
                    plan,
                    prepared,
                    source,
                    &*self.projection_weight,
                    work.projected_mut(),
                )?
            },
            PreparedProjection::Unbiased(arguments) => unsafe {
                blas.linear_native_default_bound(arguments)
                    .map_err(ShapeError::Fatal)?;
            },
        }
        let mut input = work.projected_ptr();
        let mut output = work.output().cu_deviceptr();
        let mut rows = u32_size(selection.tokens()).map_err(ShapeError::Fatal)?;
        let mut width = u32_size(self.output).map_err(ShapeError::Fatal)?;
        let mut eps = self.eps;
        crate::launch_threads(
            &self.kernels.l2,
            stream,
            (rows, 1, 1),
            256,
            0,
            &mut [
                crate::arg(&mut input),
                crate::arg(&mut output),
                crate::arg(&mut rows),
                crate::arg(&mut width),
                crate::arg(&mut eps),
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
            ManuallyDrop::drop(&mut self.projection_weight);
            ManuallyDrop::drop(&mut self.projection_bias);
            ManuallyDrop::drop(&mut self.plan);
            ManuallyDrop::drop(&mut self.kernels);
        }
        self.released = true;
    }
}

impl Drop for ColbertHead {
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
    bf16: Option<DeviceBuffer<Bf16>>,
    f32: Option<DeviceBuffer<f32>>,
    u32: Vec<DeviceBuffer<u32>>,
    committed: bool,
}

impl WorkBuilder {
    fn new(stream: Arc<Stream>) -> Result<Self, BackendError> {
        Ok(Self {
            stream,
            bf16: None,
            f32: None,
            u32: Vec::new(),
            committed: false,
        })
    }
}

impl Drop for WorkBuilder {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if self.stream.synchronize().is_err() {
            if let Some(buffer) = self.bf16.take() {
                std::mem::forget(buffer);
            }
            if let Some(buffer) = self.f32.take() {
                std::mem::forget(buffer);
            }
            for buffer in self.u32.drain(..) {
                std::mem::forget(buffer);
            }
        }
    }
}

pub(crate) struct ColbertWork {
    stream: Arc<Stream>,
    tokens_capacity: usize,
    sequences_capacity: usize,
    bytes: usize,
    projected: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<f32>>,
    key_indices: ManuallyDrop<DeviceBuffer<u32>>,
    key_offsets: ManuallyDrop<DeviceBuffer<u32>>,
    released: bool,
}

impl ColbertWork {
    fn new(
        stream: Arc<Stream>,
        head: &ColbertHead,
        tokens: usize,
        sequences: usize,
    ) -> Result<Self, BackendError> {
        if tokens == 0 || sequences == 0 {
            return Err(infer(
                "ColBERT work requires positive token and sequence counts",
            ));
        }
        if !Arc::ptr_eq(&stream, &head.stream) {
            return Err(infer("ColBERT work stream differs from the head stream"));
        }
        let bytes = head.work_bytes(tokens, sequences)?;
        let projected = product(tokens, head.output)?;
        let mut builder = WorkBuilder::new(stream.clone())?;
        builder.bf16 = Some(DeviceBuffer::<Bf16>::zeroed(&stream, projected).map_err(infer)?);
        builder.f32 = Some(DeviceBuffer::<f32>::zeroed(&stream, projected).map_err(infer)?);
        builder
            .u32
            .try_reserve_exact(2)
            .map_err(|_| infer("ColBERT work allocation failed"))?;
        builder
            .u32
            .push(DeviceBuffer::<u32>::zeroed(&stream, tokens).map_err(infer)?);
        builder.u32.push(
            DeviceBuffer::<u32>::zeroed(
                &stream,
                sequences
                    .checked_add(1)
                    .ok_or_else(|| infer("ColBERT key offset count overflow"))?,
            )
            .map_err(infer)?,
        );
        let projected = builder
            .bf16
            .take()
            .ok_or_else(|| infer("ColBERT projection allocation is missing"))?;
        let output = builder
            .f32
            .take()
            .ok_or_else(|| infer("ColBERT output allocation is missing"))?;
        let mut offsets = builder.u32.drain(..);
        let key_indices = offsets
            .next()
            .ok_or_else(|| infer("ColBERT key index allocation is missing"))?;
        let key_offsets = offsets
            .next()
            .ok_or_else(|| infer("ColBERT key offset allocation is missing"))?;
        builder.committed = true;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            tokens_capacity: tokens,
            sequences_capacity: sequences,
            bytes,
            projected: ManuallyDrop::new(projected),
            output: ManuallyDrop::new(output),
            key_indices: ManuallyDrop::new(key_indices),
            key_offsets: ManuallyDrop::new(key_offsets),
            released: false,
        })
    }

    pub(crate) fn fits(&self, tokens: usize, sequences: usize) -> bool {
        tokens <= self.tokens_capacity
            && sequences <= self.sequences_capacity
            && tokens <= self.key_indices.len()
            && sequences < self.key_offsets.len()
    }

    pub(crate) fn capacity(&self) -> (usize, usize) {
        (self.tokens_capacity, self.sequences_capacity)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    fn projected_mut(&mut self) -> &mut DeviceBuffer<Bf16> {
        &mut self.projected
    }

    fn projected_ptr(&self) -> u64 {
        self.projected.cu_deviceptr()
    }

    pub(crate) fn key_indices_ptr(&self) -> u64 {
        self.key_indices.cu_deviceptr()
    }

    pub(crate) fn key_offsets_ptr(&self) -> u64 {
        self.key_offsets.cu_deviceptr()
    }

    fn upload_selection(
        &mut self,
        stream: &Stream,
        selection: &ColbertSelection,
    ) -> Result<(), ShapeError> {
        if !std::ptr::eq(stream, &*self.stream) {
            return Err(fatal("ColBERT selection upload uses a different stream"));
        }
        if selection.keys() > 0 {
            let bytes = checked_bytes(selection.keys(), size_of::<u32>(), "ColBERT key indices")
                .map_err(ShapeError::Fatal)?;
            // SAFETY: the destination capacity was checked by preflight and
            // the host selection remains alive through the async copy.
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    self.key_indices.cu_deviceptr(),
                    selection.key_indices().as_ptr(),
                    bytes,
                    stream.cu_stream(),
                )
            }
            .map_err(|error| ShapeError::Fatal(infer(error)))?;
        }
        let bytes = checked_bytes(
            selection.key_offsets().len(),
            size_of::<u32>(),
            "ColBERT key offsets",
        )
        .map_err(ShapeError::Fatal)?;
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.key_offsets.cu_deviceptr(),
                selection.key_offsets().as_ptr(),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|error| ShapeError::Fatal(infer(error)))?;
        Ok(())
    }

    fn preflight(
        &self,
        head: &ColbertHead,
        selection: &ColbertSelection,
    ) -> Result<(), ShapeError> {
        if self.released {
            return Err(fatal("ColBERT workspace was already released"));
        }
        if !Arc::ptr_eq(&self.stream, &head.stream) {
            return Err(fatal("ColBERT work belongs to a different CUDA stream"));
        }
        for buffer_context in [
            self.projected.context(),
            self.output.context(),
            self.key_indices.context(),
            self.key_offsets.context(),
        ] {
            if !Arc::ptr_eq(buffer_context, &head.context) {
                return Err(fatal(
                    "ColBERT work buffer belongs to a different CUDA context",
                ));
            }
        }
        if !self.fits(selection.tokens(), selection.sequences()) {
            return Err(ShapeError::Rejected(infer(
                "ColBERT work capacity is too small",
            )));
        }
        let projected = product(selection.tokens(), head.output).map_err(ShapeError::Fatal)?;
        if self.projected.len() < projected
            || self.output.len() < projected
            || self.key_indices.len() < selection.keys()
            || self.key_offsets.len() < selection.key_offsets().len()
        {
            return Err(ShapeError::Rejected(infer(
                "ColBERT work buffer extent is too small",
            )));
        }
        Ok(())
    }

    pub(crate) unsafe fn drop_after_drain(&mut self) {
        if self.released {
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.projected);
            ManuallyDrop::drop(&mut self.output);
            ManuallyDrop::drop(&mut self.key_indices);
            ManuallyDrop::drop(&mut self.key_offsets);
        }
        self.released = true;
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }
}

impl Drop for ColbertWork {
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

    #[test]
    fn work_bytes_cover_projection_output_and_key_metadata() {
        let expected = 2 * 6 * 3 + 4 * 6 * 3 + 4 * 6 + 4 * 3;
        assert_eq!(work_bytes_for(3, 6, 2).unwrap(), expected);
        assert!(work_bytes_for(0, 6, 2).is_err());
        assert!(work_bytes_for(3, 0, 2).is_err());
        assert!(work_bytes_for(usize::MAX, 2, 2).is_err());
    }
}
