// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Owned canonical quantized rows and checked, allocation-free projection metadata.
//!
//! Quantization and execution use the existing context, stream, tensor views and
//! executor lifetime boundary. A provider wraps the existing LtCore implementation
//! and owns one workspace shared by its plans. No separate runtime, FFI loader,
//! plan cache, graph or per-layer workspace is introduced. GPU owners remain live
//! through completion, as required by the unsafe raw enqueue boundaries.

use crate::{
    math::MathError,
    tensor::{E4M3Fn, Element, Layout, ScalarFormat, TensorError, TensorMut, TensorRef, F32},
};
use cuda_core::{DeviceBuffer, DriverError, Stream};
use std::{fmt, marker::PhantomData, sync::Arc};

#[derive(Debug)]
pub enum Fp8Error {
    Tensor(TensorError),
    Driver(DriverError),
    Dimension,
    IncompleteRows,
}
impl fmt::Display for Fp8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "invalid quantized tensor: {e}"),
            Self::Driver(e) => write!(f, "quantized storage allocation failed: {e}"),
            Self::Dimension => f.write_str("invalid quantized row dimensions"),
            Self::IncompleteRows => f.write_str("requested rows exceed initialized quantized rows"),
        }
    }
}
impl std::error::Error for Fp8Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => Some(e),
            Self::Driver(e) => Some(e),
            _ => None,
        }
    }
}
impl From<TensorError> for Fp8Error {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}
impl From<DriverError> for Fp8Error {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

fn row_layout(rows: usize, width: usize) -> Result<Layout<2>, Fp8Error> {
    if width == 0 || rows > i32::MAX as usize || width > i32::MAX as usize {
        return Err(Fp8Error::Dimension);
    }
    Ok(Layout::contiguous([rows, width])?)
}
fn prefix(initialized: usize, rows: usize) -> Result<(), Fp8Error> {
    if rows > initialized {
        Err(Fp8Error::IncompleteRows)
    } else {
        Ok(())
    }
}

/// Allocate only during cold checkpoint loading or request-arena preparation.
/// Logical rows are explicit and are never padded for GEMM eligibility.
pub struct QuantizedRowStorage {
    values: DeviceBuffer<E4M3Fn>,
    scales: DeviceBuffer<f32>,
    capacity: usize,
    width: usize,
}
impl QuantizedRowStorage {
    pub fn allocate(stream: &Stream, capacity: usize, width: usize) -> Result<Self, Fp8Error> {
        let layout = row_layout(capacity, width)?;
        ScalarFormat::F32.storage_bytes(capacity)?;
        Ok(Self {
            values: DeviceBuffer::zeroed(stream, layout.num_elements())?,
            scales: DeviceBuffer::zeroed(stream, capacity)?,
            capacity,
            width,
        })
    }
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Borrow the actual prepared quantizer destinations without marking them
    /// initialized or making a second copy of the checkpoint weights.
    pub fn destinations(&mut self) -> (&mut DeviceBuffer<E4M3Fn>, &mut DeviceBuffer<f32>) {
        (&mut self.values, &mut self.scales)
    }
    /// Borrow an arena after an ordered prepared quantizer write.
    ///
    /// # Safety
    /// The same initialization, ordering and retention obligations as
    /// `assume_quantized` apply to these borrowed logical rows.
    pub unsafe fn initialized_view(&self, rows: usize) -> Result<QuantizedRowsRef<'_>, Fp8Error> {
        prefix(self.capacity, rows)?;
        Ok(QuantizedRowsRef {
            values: TensorRef::new(&self.values, row_layout(rows, self.width)?)?,
            scales: TensorRef::new(&self.scales, Layout::contiguous([rows, 1])?)?,
        })
    }

    /// Publish owned checkpoint rows after cold quantization has drained.
    ///
    /// # Safety
    /// Quantization and all earlier raw uses must have completed before this
    /// ownership transfer. Values/scales must follow the native recipe. Retain
    /// the returned owner through all subsequent raw GPU uses. Warm arenas use
    /// `initialized_view`, which preserves their owner even on a bounds error.
    pub unsafe fn assume_quantized(self, rows: usize) -> Result<QuantizedRows, Fp8Error> {
        prefix(self.capacity, rows)?;
        Ok(QuantizedRows {
            storage: self,
            rows,
        })
    }
}

/// Canonical E4M3FN `[rows,width]` values plus one F32 scale per real row.
pub struct QuantizedRows {
    storage: QuantizedRowStorage,
    rows: usize,
}
impl QuantizedRows {
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn width(&self) -> usize {
        self.storage.width
    }
    pub fn into_storage(self) -> QuantizedRowStorage {
        self.storage
    }
    /// Checked borrowed prefix; no allocation, CUDA query, refcount or copy.
    pub fn view(&self, rows: usize) -> Result<QuantizedRowsRef<'_>, Fp8Error> {
        prefix(self.rows, rows)?;
        Ok(QuantizedRowsRef {
            values: TensorRef::new(&self.storage.values, row_layout(rows, self.width())?)?,
            scales: TensorRef::new(&self.storage.scales, Layout::contiguous([rows, 1])?)?,
        })
    }
    pub fn full_view(&self) -> Result<QuantizedRowsRef<'_>, Fp8Error> {
        self.view(self.rows)
    }
}
pub struct QuantizedRowsRef<'a> {
    values: TensorRef<'a, E4M3Fn, 2>,
    scales: TensorRef<'a, F32, 2>,
}
impl QuantizedRowsRef<'_> {
    pub const fn values(&self) -> &TensorRef<'_, E4M3Fn, 2> {
        &self.values
    }
    pub const fn scales(&self) -> &TensorRef<'_, F32, 2> {
        &self.scales
    }
}

/// Pure metadata for a single real `(M,N,K)` projection. The existing work
/// cache may own prepared execution plans; this type imposes no arbitrary
/// number-of-distinct-shapes limit and makes no CUDA call.
pub struct Fp8LinearPlan<E: Element> {
    input: Layout<2>,
    weight: Layout<2>,
    output: Layout<2>,
    rows: i32,
    input_width: i32,
    output_width: i32,
    marker: PhantomData<E>,
}
#[derive(Clone, Copy, Debug)]
pub struct Fp8Arguments {
    pub input: u64,
    pub weight: u64,
    pub input_scales: u64,
    pub weight_scales: u64,
    pub output: u64,
    pub rows: i32,
    pub input_width: i32,
    pub output_width: i32,
}
impl<E: Element> Fp8LinearPlan<E> {
    pub fn new(rows: usize, input: usize, output: usize) -> Result<Self, MathError> {
        if !matches!(E::FORMAT, ScalarFormat::Bf16 | ScalarFormat::F16) {
            return Err(MathError::UnsupportedFormat);
        }
        if input == 0 || output == 0 || !input.is_multiple_of(16) || !output.is_multiple_of(16) {
            return Err(MathError::UnsupportedLayout);
        }
        let dim = |n| i32::try_from(n).map_err(|_| MathError::DimensionOverflow);
        Ok(Self {
            input: Layout::contiguous([rows, input]).map_err(MathError::Tensor)?,
            weight: Layout::contiguous([output, input]).map_err(MathError::Tensor)?,
            output: Layout::contiguous([rows, output]).map_err(MathError::Tensor)?,
            rows: dim(rows)?,
            input_width: dim(input)?,
            output_width: dim(output)?,
            marker: PhantomData,
        })
    }
    pub const fn input_layout(&self) -> Layout<2> {
        self.input
    }
    pub const fn weight_layout(&self) -> Layout<2> {
        self.weight
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
    /// CPU-only tensor address binding. Raw enqueues are unsafe: retain all
    /// owners through completion and order scales/values on the chosen stream.
    pub fn bind(
        &self,
        input: &QuantizedRowsRef<'_>,
        weight: &QuantizedRowsRef<'_>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<Fp8Arguments, MathError> {
        if input.values.layout() != &self.input || weight.values.layout() != &self.weight {
            return Err(MathError::UnsupportedLayout);
        }
        let context = input.values.context();
        for other in [
            input.scales.context(),
            weight.values.context(),
            weight.scales.context(),
            output.context(),
        ] {
            if !Arc::ptr_eq(context, other) {
                return Err(MathError::ContextMismatch);
            }
        }
        let output = TensorMut::<E, 2>::new(output, self.output).map_err(MathError::Tensor)?;
        Ok(Fp8Arguments {
            input: input.values.device_ptr(),
            weight: weight.values.device_ptr(),
            input_scales: input.scales.device_ptr(),
            weight_scales: weight.scales.device_ptr(),
            output: output.device_ptr(),
            rows: self.rows,
            input_width: self.input_width,
            output_width: self.output_width,
        })
    }
}

use crate::lt::{Fp8Geometry, Fp8Plan, Fp8Recipe, LtProvider};
use cuda_core::{launch_kernel, Context, Function, Module};
use infinity_v3_backend_core::BackendError;
use std::{ffi::c_void, mem::ManuallyDrop};

/// Owns one instance of the existing cuBLASLt Core implementation. Its plans
/// share one initialized workspace. It is movable, externally serialized, and
/// uses the same cuda-core runtime.
pub struct Fp8Provider {
    inner: ManuallyDrop<LtProvider>,
    unit: ManuallyDrop<Option<DeviceBuffer<f32>>>,
    recipe: Fp8Recipe,
    failed: bool,
}
/// One model-typed quantized checkpoint projection. Only canonical FP8 bytes
/// and F32 row scales are retained; the original checkpoint buffer can be
/// released after its cold quantization stream drains.
pub struct Fp8Linear<E: Element> {
    weights: QuantizedRows,
    element: PhantomData<E>,
}
impl<E: Element> Fp8Linear<E> {
    /// # Safety
    /// These rows must have been quantized from finite checkpoint values after
    /// conversion to the actual model dtype `E`; all initialization/retention
    /// obligations of `QuantizedRows` continue to apply. Cold initialization
    /// and all earlier raw uses must have drained before this admission.
    pub unsafe fn from_quantized_weights(weights: QuantizedRows) -> Result<Self, MathError> {
        Fp8LinearPlan::<E>::new(0, weights.width(), weights.rows())?;
        Ok(Self {
            weights,
            element: PhantomData,
        })
    }
    pub fn input_width(&self) -> usize {
        self.weights.width()
    }
    pub fn output_width(&self) -> usize {
        self.weights.rows()
    }
    pub fn weights(&self) -> Result<QuantizedRowsRef<'_>, Fp8Error> {
        self.weights.full_view()
    }
}

pub struct PreparedFp8<E: Element> {
    gpu: Fp8Plan,
    metadata: Fp8LinearPlan<E>,
}
impl Fp8Provider {
    pub fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        expected_version: Option<usize>,
    ) -> Result<Self, BackendError> {
        let capability = context.compute_capability().map_err(crate::start)?;
        let recipe = match capability {
            (9, 0) => Fp8Recipe::RowwiseSm90,
            (10, 0) | (12, 0) => Fp8Recipe::ScalarF32Sm100Or120,
            _ => return Err(crate::start("FP8 methods require SM90, SM100 or SM120")),
        };
        let inner = LtProvider::new_fp8(context, stream)?;
        let version = inner.fp8_vendor_version()?;
        if version < 130100 || expected_version.is_some_and(|v| v != version) {
            return Err(crate::start(format!(
                "FP8 cuBLASLt version {version} does not match the requested CUDA13.1+ provider"
            )));
        }
        let unit = if recipe == Fp8Recipe::ScalarF32Sm100Or120 {
            Some(DeviceBuffer::from_host(inner.stream(), &[1.0f32]).map_err(crate::start)?)
        } else {
            None
        };
        Ok(Self {
            inner: ManuallyDrop::new(inner),
            unit: ManuallyDrop::new(unit),
            recipe,
            failed: false,
        })
    }
    pub fn context(&self) -> &Arc<Context> {
        self.inner.context()
    }
    pub fn stream(&self) -> &Stream {
        self.inner.stream()
    }
    /// Bind once at the standalone/request boundary. Engine callers then use
    /// prepared bound operations without per-layer context queries.
    pub fn bind_context(&self) -> Result<(), BackendError> {
        self.inner.bind_context()
    }
    fn scale_pointers(&self, x: u64, w: u64) -> Result<(u64, u64), BackendError> {
        match self.recipe {
            Fp8Recipe::RowwiseSm90 => Ok((x, w)),
            Fp8Recipe::ScalarF32Sm100Or120 => {
                let unit = self
                    .unit
                    .as_ref()
                    .ok_or_else(|| crate::infer("FP8 scalar unit storage missing"))?;
                Ok((unit.cu_deviceptr(), unit.cu_deviceptr()))
            }
        }
    }
    fn validate_context(
        &self,
        input: &QuantizedRowsRef<'_>,
        weight: &QuantizedRowsRef<'_>,
    ) -> Result<(), BackendError> {
        for context in [
            input.values.context(),
            input.scales.context(),
            weight.values.context(),
            weight.scales.context(),
        ] {
            if !Arc::ptr_eq(self.context(), context) {
                return Err(crate::infer("FP8 tensor belongs to a different context"));
            }
        }
        Ok(())
    }
    /// Allocate/choose descriptors once for one real prepared work shape.
    pub fn prepare<E: Element>(
        &self,
        metadata: Fp8LinearPlan<E>,
        input: &QuantizedRowsRef<'_>,
        weight: &QuantizedRowsRef<'_>,
    ) -> Result<PreparedFp8<E>, BackendError> {
        if self.failed {
            return Err(crate::infer("FP8 provider is failed"));
        }
        self.validate_context(input, weight)?;
        if input.values.layout() != &metadata.input || weight.values.layout() != &metadata.weight {
            return Err(crate::infer("FP8 preparation tensor geometry differs"));
        }
        self.bind_context()?;
        let model_type = match E::FORMAT {
            ScalarFormat::Bf16 => 14,
            ScalarFormat::F16 => 2,
            _ => return Err(crate::infer("FP8 model output must be BF16 or F16")),
        };
        let geometry = Fp8Geometry {
            rows: metadata.rows,
            input: metadata.input_width,
            output: metadata.output_width,
            model_type,
        };
        let (x, w) = self.scale_pointers(input.scales.device_ptr(), weight.scales.device_ptr())?;
        let gpu = self.inner.prepare_fp8_bound(geometry, self.recipe, x, w)?;
        Ok(PreparedFp8 { gpu, metadata })
    }
    /// Prepared projection, with optional separate post-model-round bias.
    ///
    /// # Safety
    /// The provider context must already be current. Retain inputs, output,
    /// scratch, bias, methods, provider and plan through stream completion and
    /// serialize every use. After any error this provider cannot be reused; the
    /// executor must drain or retain all potentially in-flight owners.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn project_bound<E: Element>(
        &mut self,
        plan: &mut PreparedFp8<E>,
        methods: &Fp8Methods<E>,
        input: &QuantizedRowsRef<'_>,
        weight: &QuantizedRowsRef<'_>,
        output: &mut DeviceBuffer<E::Storage>,
        scratch: Option<&mut DeviceBuffer<f32>>,
        bias: Option<&DeviceBuffer<E::Storage>>,
    ) -> Result<(), BackendError> {
        if self.failed {
            return Err(crate::infer("FP8 provider is failed"));
        }
        self.validate_context(input, weight)?;
        if !Arc::ptr_eq(self.context(), &methods.context) {
            return Err(crate::infer("FP8 kernel context differs"));
        }
        let args = plan
            .metadata
            .bind(input, weight, output)
            .map_err(crate::infer)?;
        let expected = Fp8Geometry {
            rows: args.rows,
            input: args.input_width,
            output: args.output_width,
            model_type: if E::FORMAT == ScalarFormat::Bf16 {
                14
            } else {
                2
            },
        };
        if plan.gpu.geometry() != expected || plan.gpu.recipe() != self.recipe {
            return Err(crate::infer("FP8 prepared geometry/recipe differs"));
        }
        let mut y = args.output;
        if self.recipe == Fp8Recipe::ScalarF32Sm100Or120 {
            let scratch =
                scratch.ok_or_else(|| crate::infer("FP8 F32 postscale scratch missing"))?;
            if !Arc::ptr_eq(self.context(), scratch.context()) {
                return Err(crate::infer("FP8 scratch context differs"));
            }
            y = TensorMut::<F32, 2>::new(scratch, plan.metadata.output)
                .map_err(crate::infer)?
                .device_ptr();
        }
        // Validate the scalar tail's launch range before queuing the GEMM.
        let tail_count = u64::from(args.rows as u32) * u64::from(args.output_width as u32);
        if tail_count.div_ceil(256) > i32::MAX as u64 {
            return Err(crate::infer(
                "FP8 elementwise launch grid exceeds the device ABI",
            ));
        }
        let bias_pointer = if let Some(bias) = bias {
            if !Arc::ptr_eq(self.context(), bias.context()) {
                return Err(crate::infer("FP8 bias context differs"));
            }
            Some(
                TensorRef::<E, 1>::new(
                    bias,
                    Layout::contiguous([args.output_width as usize]).map_err(crate::infer)?,
                )
                .map_err(crate::infer)?
                .device_ptr(),
            )
        } else {
            None
        };
        let (xs, ws) = self.scale_pointers(args.input_scales, args.weight_scales)?;
        // SAFETY: checked same-context complete tensor views above; the caller
        // retains owners and has bound the context for this stream.
        let submitted = unsafe {
            self.inner
                .launch_fp8_bound(&mut plan.gpu, args.input, args.weight, xs, ws, y)
        };
        if let Err(error) = submitted {
            self.failed = true;
            return Err(error);
        }
        let tail = unsafe {
            methods.finish_bound(
                self.stream(),
                self.recipe,
                y,
                args.input_scales,
                args.weight_scales,
                args.output,
                bias_pointer,
                args.rows as u32,
                args.output_width as u32,
            )
        };
        if tail.is_err() {
            self.failed = true;
        }
        tail
    }
}
impl Drop for Fp8Provider {
    fn drop(&mut self) {
        if self.inner.context().bind_to_thread().is_err()
            || self.inner.stream().synchronize().is_err()
        {
            return;
        }
        // SAFETY: the retained ordered stream drained. An uncertain completion
        // leaves both existing Core and unit allocation deliberately retained.
        unsafe {
            ManuallyDrop::drop(&mut self.unit);
            ManuallyDrop::drop(&mut self.inner);
        }
    }
}

/// Selected AOT methods over the existing cuda-core module/function owners.
pub struct Fp8Methods<E: Element> {
    quantize: Function,
    postscale: Function,
    bias: Function,
    context: Arc<Context>,
    element: PhantomData<E>,
}
fn argument<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast()
}
impl<E: Element> Fp8Methods<E> {
    /// Resolve once after the standard bundle loader validated source/hash/ABI.
    ///
    /// # Safety
    /// The quantizer module must contain the qualified five-argument row ABI
    /// `(model,bytes,F32scales,rows,width)` with128threads. The math module must
    /// contain the qualified six-argument postscale and four-argument model-bias
    /// ABIs with256threads. Both modules must be precompiled for this device.
    pub unsafe fn from_modules(
        quantizer: &Arc<Module>,
        math: &Arc<Module>,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(quantizer.context(), math.context()) {
            return Err(crate::start("FP8 modules belong to different contexts"));
        }
        let suffix = match E::FORMAT {
            ScalarFormat::Bf16 => "bf16",
            ScalarFormat::F16 => "f16",
            _ => {
                return Err(crate::start(
                    "FP8 source/output methods require BF16 or F16",
                ))
            }
        };
        Ok(Self {
            quantize: quantizer
                .load_function(&format!("infinity_fp8_quant_{suffix}_rows"))
                .map_err(crate::start)?,
            postscale: math
                .load_function(&format!("infinity_fp8_postscale_{suffix}"))
                .map_err(crate::start)?,
            bias: math
                .load_function(&format!("infinity_fp8_model_bias_{suffix}"))
                .map_err(crate::start)?,
            context: quantizer.context().clone(),
            element: PhantomData,
        })
    }
    /// Enqueue the native quantizer into an existing arena/checkpoint owner.
    ///
    /// # Safety
    /// The context must be current, finite model sources must be valid and kept
    /// live until completion, destinations must not be used concurrently, and
    /// all later consumers must be ordered on this same stream. On error retain
    /// potentially in-flight storage until completion is established.
    pub unsafe fn quantize_bound(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<E::Storage>,
        destination: &mut QuantizedRowStorage,
        rows: usize,
    ) -> Result<(), BackendError> {
        prefix(destination.capacity, rows).map_err(crate::infer)?;
        if !Arc::ptr_eq(&self.context, stream.context())
            || !Arc::ptr_eq(&self.context, source.context())
            || !Arc::ptr_eq(&self.context, destination.values.context())
        {
            return Err(crate::infer("FP8 quantization context differs"));
        }
        let layout = row_layout(rows, destination.width).map_err(crate::infer)?;
        let source = TensorRef::<E, 2>::new(source, layout).map_err(crate::infer)?;
        if rows == 0 {
            return Ok(());
        }
        let (mut src, mut dst, mut scale, mut m, mut k) = (
            source.device_ptr(),
            destination.values.cu_deviceptr(),
            destination.scales.cu_deviceptr(),
            u32::try_from(rows).map_err(crate::infer)?,
            u32::try_from(destination.width).map_err(crate::infer)?,
        );
        unsafe {
            launch_kernel(
                self.quantize.cu_function(),
                (m, 1, 1),
                (128, 1, 1),
                0,
                stream.cu_stream(),
                &mut [
                    argument(&mut src),
                    argument(&mut dst),
                    argument(&mut scale),
                    argument(&mut m),
                    argument(&mut k),
                ],
            )
        }
        .map_err(crate::infer)
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn finish_bound(
        &self,
        stream: &Stream,
        recipe: Fp8Recipe,
        mut scratch: u64,
        mut xs: u64,
        mut ws: u64,
        mut output: u64,
        bias: Option<u64>,
        mut rows: u32,
        mut width: u32,
    ) -> Result<(), BackendError> {
        let count = u64::from(rows) * u64::from(width);
        if count == 0 {
            return Ok(());
        }
        let grid = u32::try_from(count.div_ceil(256)).map_err(crate::infer)?;
        if recipe == Fp8Recipe::ScalarF32Sm100Or120 {
            unsafe {
                launch_kernel(
                    self.postscale.cu_function(),
                    (grid, 1, 1),
                    (256, 1, 1),
                    0,
                    stream.cu_stream(),
                    &mut [
                        argument(&mut scratch),
                        argument(&mut xs),
                        argument(&mut ws),
                        argument(&mut output),
                        argument(&mut rows),
                        argument(&mut width),
                    ],
                )
            }
            .map_err(crate::infer)?;
        }
        if let Some(mut bias) = bias {
            unsafe {
                launch_kernel(
                    self.bias.cu_function(),
                    (grid, 1, 1),
                    (256, 1, 1),
                    0,
                    stream.cu_stream(),
                    &mut [
                        argument(&mut output),
                        argument(&mut bias),
                        argument(&mut rows),
                        argument(&mut width),
                    ],
                )
            }
            .map_err(crate::infer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Bf16, F16};
    #[test]
    fn real_changing_rows_and_empty_rows_do_not_get_padded_or_hit_a_shape_count_limit() {
        for m in 0..257 {
            let p = Fp8LinearPlan::<Bf16>::new(m, 1024, 3072).unwrap();
            assert_eq!(*p.input_layout().shape(), [m, 1024]);
            assert_eq!(*p.output_layout().shape(), [m, 3072]);
        }
        assert!(prefix(128, 129).is_err());
        assert!(prefix(129, 129).is_ok());
    }
    #[test]
    fn fp32_or_quantized_output_and_invalid_feature_geometry_are_rejected() {
        assert!(matches!(
            crate::math::LinearPlan::<E4M3Fn>::contiguous(1, 16, 16),
            Err(MathError::UnsupportedFormat)
        ));
        assert!(matches!(
            Fp8LinearPlan::<F32>::new(1, 16, 16),
            Err(MathError::UnsupportedFormat)
        ));
        assert!(matches!(
            Fp8LinearPlan::<E4M3Fn>::new(1, 16, 16),
            Err(MathError::UnsupportedFormat)
        ));
        for (n, k) in [(0, 16), (16, 0), (17, 16), (16, 17)] {
            assert!(Fp8LinearPlan::<F16>::new(7, k, n).is_err());
        }
        assert!(Fp8LinearPlan::<F16>::new(i32::MAX as usize + 1, 16, 16).is_err());
        assert!(row_layout(1, 0).is_err());
    }
}
