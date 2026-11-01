// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Sentence Transformers Dense module loading and pooled-device execution.
//!
//! Module metadata is read from `modules.json` in array order. Every admitted
//! projection must stay inside the checkpoint root after canonicalization, and
//! its safetensors payload is exposed through the same bounded `WeightSource`
//! streaming path as the backbone weights. Execution rounds the FP32 pooled
//! rows to BF16 once, applies each prepared linear projection (with a fused
//! bias epilogue when configured), applies optional Tanh in FP32 then rounds
//! back to BF16, and widens the final rows to FP32 for the existing readback.

use crate::{
    arg, blocks,
    cublas::Blas,
    infer, launch,
    lt::{LtPlan, LtProvider, ShapeError},
    math::{refresh_plan_rows, LinearPlan},
    model::{path_present, read_json, Checkpoint, WeightSource},
    product, start,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use serde::Deserialize;
use std::{
    mem::ManuallyDrop,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

const DENSE_PLAIN_SEED_ROWS: usize = 128;
const IDENTITY_ACTIVATION: &str = "torch.nn.modules.linear.Identity";
const TANH_ACTIVATION: &str = "torch.nn.modules.activation.Tanh";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DenseActivation {
    Identity,
    Tanh,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseModuleConfig {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) bias: bool,
    pub(crate) activation: DenseActivation,
}

#[derive(Clone, Debug)]
pub(crate) struct DenseModule {
    pub(crate) config: DenseModuleConfig,
    pub(crate) weight: WeightSource,
    pub(crate) bias: Option<WeightSource>,
}

#[derive(Clone, Debug)]
pub(crate) struct DenseChainHost {
    modules: Vec<DenseModule>,
    input_width: usize,
    output_width: usize,
}

impl DenseChainHost {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn load(
        root: &Path,
        backbone_width: usize,
    ) -> Result<Option<Self>, crate::BackendError> {
        Self::load_with_paths(root, backbone_width, None)
    }

    /// Load Dense modules from an explicit ordered path list, or discover them
    /// from `modules.json` when no override list is supplied. An explicit empty
    /// list means the caller already resolved that this checkpoint has none.
    pub(crate) fn load_with_paths(
        root: &Path,
        backbone_width: usize,
        paths: Option<Vec<String>>,
    ) -> Result<Option<Self>, crate::BackendError> {
        // The native loader validates every declared Dense path before applying
        // an override. A downloader's empty result must not hide malformed
        // metadata and silently turn a projected model into its backbone.
        let declared_paths = match read_modules(root)? {
            Some(entries) => entries
                .iter()
                .filter(|module| is_dense_module_type(&module.kind))
                .map(|module| {
                    module
                        .path
                        .clone()
                        .ok_or_else(|| start("Dense module is missing its path"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let relative_paths = paths.unwrap_or(declared_paths);
        if relative_paths.is_empty() {
            return Ok(None);
        }
        if backbone_width == 0 {
            return Err(start("Dense modules require a nonzero backbone width"));
        }
        let root = root.canonicalize().map_err(start)?;
        let mut modules = Vec::new();
        modules
            .try_reserve_exact(relative_paths.len())
            .map_err(start)?;
        let mut expected_input = backbone_width;
        for relative in relative_paths {
            let module_root = checked_module_root(&root, &relative)?;
            let config = read_dense_config(&module_root.join("config.json"))?;
            if config.in_features != expected_input {
                return Err(start(
                    "Dense module chain dimensions are not continuous in modules.json order",
                ));
            }
            let checkpoint = open_dense_checkpoint(&root, &module_root, &relative)?;
            let weight =
                checkpoint.source("linear.weight", &[config.out_features, config.in_features])?;
            let bias = if config.bias {
                Some(checkpoint.source("linear.bias", &[config.out_features])?)
            } else {
                None
            };
            expected_input = config.out_features;
            modules.push(DenseModule {
                config,
                weight,
                bias,
            });
        }
        Ok(Some(Self {
            modules,
            input_width: backbone_width,
            output_width: expected_input,
        }))
    }

    #[cfg(test)]
    pub(crate) fn input_width(&self) -> usize {
        self.input_width
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }

    #[cfg(test)]
    pub(crate) fn modules(&self) -> &[DenseModule] {
        &self.modules
    }

    pub(crate) fn uses_plain(&self) -> bool {
        self.modules.iter().any(|module| !module.config.bias)
    }
}

#[derive(Deserialize)]
struct ModuleEntry {
    #[serde(rename = "type")]
    kind: String,
    path: Option<String>,
}

fn read_modules(root: &Path) -> Result<Option<Vec<ModuleEntry>>, crate::BackendError> {
    let file = root.join("modules.json");
    if !path_present(&file)? {
        return Ok(None);
    }
    Ok(Some(read_json(&file)?))
}

/// Native counts projections by the last type component, including the modern
/// `sentence_transformers.base.modules.dense.Dense` namespace.
pub(crate) fn is_dense_module_type(kind: &str) -> bool {
    kind.starts_with("sentence_transformers.") && kind.rsplit('.').next() == Some("Dense")
}

fn checked_module_root(root: &Path, relative: &str) -> Result<PathBuf, crate::BackendError> {
    let path = Path::new(relative);
    if relative.is_empty()
        || relative.contains('\\')
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(start(
            "Dense module path must be a nonempty relative path inside the model directory",
        ));
    }
    let module = root
        .join(path)
        .canonicalize()
        .map_err(|error| start(format!("Dense module path could not be resolved: {error}")))?;
    if !module.starts_with(root) {
        return Err(start("Dense module path escapes the model directory"));
    }
    if !module.is_dir() {
        return Err(start("Dense module path is not a directory"));
    }
    Ok(module)
}

fn read_dense_config(path: &Path) -> Result<DenseModuleConfig, crate::BackendError> {
    let value: serde_json::Value = read_json(path)?;
    let object = value
        .as_object()
        .ok_or_else(|| start("Dense config must be a JSON object"))?;
    let in_features = dense_dimension(object.get("in_features"), "in_features")?;
    let out_features = dense_dimension(object.get("out_features"), "out_features")?;
    let bias = object
        .get("bias")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| start("Dense config bias must be a boolean"))?;
    let activation = match object.get("activation_function") {
        None | Some(serde_json::Value::Null) => DenseActivation::Identity,
        Some(serde_json::Value::String(name)) if name == IDENTITY_ACTIVATION => {
            DenseActivation::Identity
        }
        Some(serde_json::Value::String(name)) if name == TANH_ACTIVATION => DenseActivation::Tanh,
        Some(serde_json::Value::String(name)) => {
            return Err(start(format!(
                "Unsupported Sentence Transformers Dense activation: {name}"
            )))
        }
        Some(_) => {
            return Err(start(
                "Sentence Transformers Dense activation_function must be a string",
            ))
        }
    };
    Ok(DenseModuleConfig {
        in_features,
        out_features,
        bias,
        activation,
    })
}

fn dense_dimension(
    value: Option<&serde_json::Value>,
    name: &str,
) -> Result<usize, crate::BackendError> {
    let value = value
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| start(format!("Dense config {name} must be a positive integer")))?;
    let value = usize::try_from(value)
        .map_err(|_| start(format!("Dense config {name} exceeds the host index range")))?;
    if value == 0 {
        return Err(start(format!("Dense config {name} must be positive")));
    }
    u32::try_from(value).map_err(|_| start(format!("Dense config {name} exceeds u32")))?;
    i32::try_from(value).map_err(|_| start(format!("Dense config {name} exceeds i32")))?;
    Ok(value)
}

fn open_dense_checkpoint(
    model_root: &Path,
    module_root: &Path,
    relative: &str,
) -> Result<Checkpoint, crate::BackendError> {
    let single = path_present(&module_root.join("model.safetensors"))?;
    let sharded = path_present(&module_root.join("model.safetensors.index.json"))?;
    if !single && !sharded {
        if path_present(&module_root.join("pytorch_model.bin"))? {
            return Err(start(format!(
                "Dense module {relative} uses pytorch_model.bin; only safetensors Dense checkpoints are implemented"
            )));
        }
        return Err(start(format!(
            "Dense module {relative} has no model.safetensors checkpoint"
        )));
    }
    Checkpoint::open_module(model_root, module_root)
}

struct DenseDeviceModule {
    weight: DeviceBuffer<Bf16>,
    bias: Option<DeviceBuffer<Bf16>>,
    in_features: usize,
    out_features: usize,
    tanh: bool,
}

struct DenseModulePlan {
    bias: Option<LtPlan<Bf16>>,
    plain: Option<LinearPlan<Bf16>>,
}

struct DensePlans {
    rows: Option<usize>,
    modules: Vec<DenseModulePlan>,
}

impl DensePlans {
    fn set_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        refresh_plan_rows(&mut self.rows, rows, || {
            for plan in &mut self.modules {
                if let Some(bias) = plan.bias.as_mut() {
                    provider.set_logical_rows(bias, rows)?;
                } else if let Some(plain) = plan.plain.as_mut() {
                    plain
                        .set_rows(rows)
                        .map_err(|error| ShapeError::Rejected(infer(error)))?;
                }
            }
            Ok(())
        })
    }
}

struct DenseKernels {
    f32_to_bf16: Function,
    tanh: Option<Function>,
    bf16_to_f32: Function,
}

impl DenseKernels {
    fn load(context: &Arc<Context>, tanh: bool) -> Result<Self, crate::BackendError> {
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
            f32_to_bf16: module
                .load_function("infinity_f32_to_bf16")
                .map_err(start)?,
            tanh: tanh
                .then(|| module.load_function("infinity_dense_tanh_bf16_inplace"))
                .transpose()
                .map_err(start)?,
            bf16_to_f32: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
        })
    }
}

pub(crate) struct DenseHead {
    modules: Vec<DenseDeviceModule>,
    plans: DensePlans,
    kernels: DenseKernels,
    input_width: usize,
    output_width: usize,
    uses_plain: bool,
}

impl DenseHead {
    pub(crate) fn new(
        host: DenseChainHost,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, crate::BackendError> {
        let tanh = host
            .modules
            .iter()
            .any(|module| module.config.activation == DenseActivation::Tanh);
        let kernels = DenseKernels::load(context, tanh)?;
        let mut plans = Vec::new();
        plans.try_reserve_exact(host.modules.len()).map_err(start)?;
        for module in &host.modules {
            let config = &module.config;
            plans.push(DenseModulePlan {
                bias: config
                    .bias
                    .then(|| lt.prepare_bias_shape::<Bf16>(config.in_features, config.out_features))
                    .transpose()?,
                plain: (!config.bias)
                    .then(|| {
                        LinearPlan::<Bf16>::contiguous(
                            DENSE_PLAIN_SEED_ROWS,
                            config.in_features,
                            config.out_features,
                        )
                        .map_err(infer)
                    })
                    .transpose()?,
            });
        }
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut modules = Vec::new();
        modules
            .try_reserve_exact(host.modules.len())
            .map_err(start)?;
        for module in &host.modules {
            modules.push(DenseDeviceModule {
                weight: uploader.upload(&[&module.weight])?,
                bias: module
                    .bias
                    .as_ref()
                    .map(|bias| uploader.upload(&[bias]))
                    .transpose()?,
                in_features: module.config.in_features,
                out_features: module.config.out_features,
                tanh: module.config.activation == DenseActivation::Tanh,
            });
        }
        uploader.finish()?;
        Ok(Self {
            modules,
            plans: DensePlans {
                rows: None,
                modules: plans,
            },
            kernels,
            input_width: host.input_width,
            output_width: host.output_width,
            uses_plain: host.uses_plain(),
        })
    }

    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }

    pub(crate) fn uses_plain(&self) -> bool {
        self.uses_plain
    }

    pub(crate) fn preflight(
        &mut self,
        provider: &LtProvider,
        rows: usize,
    ) -> Result<(), ShapeError> {
        if rows == 0 {
            return Ok(());
        }
        self.plans.set_rows(provider, rows)
    }

    fn max_width(&self) -> usize {
        self.modules
            .iter()
            .map(|module| module.out_features.max(module.in_features))
            .fold(self.input_width, usize::max)
    }

    pub(crate) fn work_bytes(&self, selections: usize) -> Result<usize, crate::BackendError> {
        let rows = selections;
        let scratch = product(rows, self.max_width())?
            .checked_mul(4)
            .ok_or_else(|| infer("Dense workspace arena overflow"))?;
        let output = product(rows, self.output_width)?
            .checked_mul(4)
            .ok_or_else(|| infer("Dense workspace arena overflow"))?;
        scratch
            .checked_add(output)
            .ok_or_else(|| infer("Dense workspace arena overflow"))
    }

    pub(crate) fn work(
        &self,
        stream: Arc<Stream>,
        selections: usize,
    ) -> Result<DenseWork, crate::BackendError> {
        DenseWork::new(stream, self, selections)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        source: &DeviceBuffer<f32>,
        rows: usize,
        work: &mut DenseWork,
    ) -> Result<(), crate::BackendError> {
        if rows == 0 {
            return Ok(());
        }
        if self.input_width == 0 || self.output_width == 0 {
            return Err(infer("Dense chain has zero geometry"));
        }
        if !work.fits(rows) {
            return Err(infer("Dense request exceeds retained arena capacity"));
        }
        let source_elements = product(rows, self.input_width)?;
        if source.len() < source_elements {
            return Err(infer("Dense input exceeds retained pooling buffer"));
        }
        let mut current = 0usize;
        for (index, module) in self.modules.iter().enumerate() {
            if index == 0 {
                convert_f32_to_bf16(&self.kernels, stream, source, &work.first, source_elements)?;
            }
            if current == 0 {
                self.project(index, lt, plain_blas, module, &work.first, &mut work.second)?;
                if module.tanh {
                    apply_tanh(
                        &self.kernels,
                        stream,
                        &mut work.second,
                        rows,
                        module.out_features,
                    )?;
                }
                current = 1;
            } else {
                self.project(index, lt, plain_blas, module, &work.second, &mut work.first)?;
                if module.tanh {
                    apply_tanh(
                        &self.kernels,
                        stream,
                        &mut work.first,
                        rows,
                        module.out_features,
                    )?;
                }
                current = 0;
            }
        }
        let final_input = if current == 0 {
            &*work.first
        } else {
            &*work.second
        };
        cast_bf16_to_f32(
            &self.kernels,
            stream,
            final_input,
            &mut work.output,
            rows,
            self.output_width,
        )
    }

    fn project(
        &self,
        index: usize,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        module: &DenseDeviceModule,
        input: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<(), crate::BackendError> {
        let plan = self
            .plans
            .modules
            .get(index)
            .ok_or_else(|| infer("Missing Dense module plan"))?;
        if let Some(bias) = module.bias.as_ref() {
            let plan = plan
                .bias
                .as_ref()
                .ok_or_else(|| infer("Missing Dense bias plan"))?;
            // SAFETY: the engine bound the provider context and retains every
            // buffer until the ordered stream completes or retains on failure.
            unsafe { lt.enqueue(plan, input, &module.weight, bias, output) }
        } else {
            let blas = plain_blas.ok_or_else(|| infer("Missing Dense plain cuBLAS handle"))?;
            let plan = plan
                .plain
                .as_ref()
                .ok_or_else(|| infer("Missing Dense plain plan"))?;
            // SAFETY: same retained-buffer and ordered-stream contract as above.
            unsafe { blas.linear(plan, input, &module.weight, output) }
        }
    }
}

fn convert_f32_to_bf16(
    kernels: &DenseKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<Bf16>,
    count: usize,
) -> Result<(), crate::BackendError> {
    let mut input = input.cu_deviceptr();
    let mut output = output.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        &kernels.f32_to_bf16,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut output), arg(&mut count)],
    )
}

fn apply_tanh(
    kernels: &DenseKernels,
    stream: &Stream,
    buffer: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), crate::BackendError> {
    let kernel = kernels
        .tanh
        .as_ref()
        .ok_or_else(|| infer("Missing Dense Tanh kernel"))?;
    let count = product(rows, width)?;
    let mut values = buffer.cu_deviceptr();
    let mut count = u32_size(count)?;
    launch(
        kernel,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut values), arg(&mut count)],
    )
}

fn cast_bf16_to_f32(
    kernels: &DenseKernels,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
) -> Result<(), crate::BackendError> {
    let elements = product(rows, width)?;
    let mut input = input.cu_deviceptr();
    let mut output = output.cu_deviceptr();
    let mut selected = 0u64;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.bf16_to_f32,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut output),
            arg(&mut selected),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

pub(crate) struct DenseWork {
    stream: Arc<Stream>,
    selections: usize,
    bytes: usize,
    released: bool,
    first: ManuallyDrop<DeviceBuffer<Bf16>>,
    second: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<f32>>,
}

fn alloc_initialized_bf16(
    stream: &Arc<Stream>,
    elements: usize,
) -> Result<DeviceBuffer<Bf16>, crate::BackendError> {
    // SAFETY: private allocation initialized before any vectorized library or
    // kernel read on this ordered stream.
    let buffer =
        unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(infer)?;
    let memset = unsafe {
        cuda_core::memory::memset_d8_async(
            buffer.cu_deviceptr(),
            0,
            buffer.num_bytes(),
            stream.cu_stream(),
        )
    };
    if let Err(error) = memset {
        // Completion is uncertain if the stream cannot be drained. Retain the
        // allocation instead of freeing while a queued memset might reference it.
        if stream.synchronize().is_err() {
            std::mem::forget(buffer);
        }
        return Err(infer(error));
    }
    Ok(buffer)
}

fn drop_owners_after_failure<T>(stream: &Stream, owners: Vec<T>) {
    if stream.synchronize().is_err() {
        for owner in owners {
            std::mem::forget(owner);
        }
    }
}

impl DenseWork {
    fn new(
        stream: Arc<Stream>,
        head: &DenseHead,
        selections: usize,
    ) -> Result<Self, crate::BackendError> {
        let rows = selections;
        let max_width = head.max_width();
        let width_elements = product(rows, max_width)?;
        let output_elements = product(rows, head.output_width)?;
        let bytes = product(width_elements, 4)?
            .checked_add(product(output_elements, 4)?)
            .ok_or_else(|| infer("Dense workspace arena overflow"))?;
        let first = alloc_initialized_bf16(&stream, width_elements)?;
        let second = match alloc_initialized_bf16(&stream, width_elements) {
            Ok(buffer) => buffer,
            Err(error) => {
                drop_owners_after_failure(&stream, vec![first]);
                return Err(error);
            }
        };
        // SAFETY: the cast kernel writes every output element before readback.
        let output =
            match unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, output_elements) } {
                Ok(buffer) => buffer,
                Err(error) => {
                    drop_owners_after_failure(&stream, vec![first, second]);
                    return Err(infer(error));
                }
            };
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            selections,
            bytes,
            released: false,
            first: ManuallyDrop::new(first),
            second: ManuallyDrop::new(second),
            output: ManuallyDrop::new(output),
        })
    }

    pub(crate) fn fits(&self, selections: usize) -> bool {
        selections <= self.selections
    }

    pub(crate) fn capacity(&self) -> usize {
        self.selections
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    /// Release Dense-owned device buffers after the parent workspace has proved
    /// the shared stream drained. No synchronization or driver query occurs.
    ///
    /// # Safety
    /// Every queued use of these buffers must have completed.
    pub(crate) unsafe fn drop_after_drain(&mut self) {
        if self.released {
            return;
        }
        // SAFETY: the caller proved completion, and `released` makes this
        // idempotent against the subsequent ordinary destructor.
        unsafe {
            ManuallyDrop::drop(&mut self.first);
            ManuallyDrop::drop(&mut self.second);
            ManuallyDrop::drop(&mut self.output);
        }
        self.released = true;
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }
}

impl Drop for DenseWork {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            // SAFETY: this destructor just drained the owning stream. A failed
            // drain leaves the ManuallyDrop owners retained.
            unsafe { self.drop_after_drain() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-oxide-dense-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write_modules(&self, modules: serde_json::Value) {
            fs::write(
                self.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
        }

        fn module(
            &self,
            relative: &str,
            config: serde_json::Value,
            dtype: Dtype,
            weight_shape: [usize; 2],
            bias: bool,
        ) -> PathBuf {
            let root = self.0.join(relative);
            fs::create_dir_all(&root).unwrap();
            fs::write(
                root.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let weight_count = weight_shape.iter().product::<usize>();
            let weight_data = vec![0u8; weight_count * (dtype.bitsize() / 8)];
            let views = vec![(
                "linear.weight".to_owned(),
                TensorView::new(dtype, weight_shape.to_vec(), &weight_data).unwrap(),
            )];
            if bias {
                let bias_data = vec![0u8; weight_shape[0] * (dtype.bitsize() / 8)];
                let mut views = views;
                views.push((
                    "linear.bias".to_owned(),
                    TensorView::new(dtype, vec![weight_shape[0]], &bias_data).unwrap(),
                ));
                fs::write(
                    root.join("model.safetensors"),
                    serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                )
                .unwrap();
            } else {
                fs::write(
                    root.join("model.safetensors"),
                    serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                )
                .unwrap();
            }
            root
        }

        fn error(&self, width: usize) -> String {
            match DenseChainHost::load(&self.0, width) {
                Ok(_) => panic!("invalid Dense fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn dense_config(input: usize, output: usize, bias: bool) -> serde_json::Value {
        serde_json::json!({
            "in_features": input,
            "out_features": output,
            "bias": bias,
            "activation_function": "torch.nn.modules.linear.Identity"
        })
    }

    fn modules(paths: &[&str]) -> serde_json::Value {
        let mut entries = vec![serde_json::json!({
            "type": "sentence_transformers.models.Transformer",
            "path": ""
        })];
        for path in paths {
            entries.push(serde_json::json!({
                "type": "sentence_transformers.models.Dense",
                "path": path
            }));
        }
        entries.push(serde_json::json!({
            "type": "sentence_transformers.models.Pooling",
            "path": "1_Pooling"
        }));
        serde_json::Value::Array(entries)
    }

    #[test]
    fn loads_ordered_chain_with_decreasing_and_increasing_widths() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 6, true),
            Dtype::F32,
            [6, 8],
            true,
        );
        fixture.module(
            "3_Dense",
            dense_config(6, 12, false),
            Dtype::F16,
            [12, 6],
            false,
        );
        fixture.write_modules(modules(&["2_Dense", "3_Dense"]));

        let chain = DenseChainHost::load(&fixture.0, 8).unwrap().unwrap();
        assert_eq!(chain.input_width(), 8);
        assert_eq!(chain.output_width(), 12);
        assert!(chain.uses_plain());
        assert_eq!(chain.modules().len(), 2);
        assert_eq!(chain.modules()[0].config.in_features, 8);
        assert_eq!(chain.modules()[0].config.out_features, 6);
        assert_eq!(chain.modules()[1].config.in_features, 6);
        assert_eq!(chain.modules()[1].config.out_features, 12);
    }

    #[test]
    fn accepts_modern_dense_namespace_and_missing_activation_default() {
        let fixture = Fixture::new();
        let config = serde_json::json!({"in_features": 8, "out_features": 4, "bias": false});
        fixture.module("2_Dense", config, Dtype::BF16, [4, 8], false);
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.base.modules.dense.Dense", "path": "2_Dense"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]));

        let chain = DenseChainHost::load(&fixture.0, 8).unwrap().unwrap();
        assert_eq!(
            chain.modules()[0].config.activation,
            DenseActivation::Identity
        );
        assert!(!chain.modules()[0].config.bias);
        assert_eq!(chain.output_width(), 4);
    }

    #[test]
    fn no_dense_modules_means_no_head() {
        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]));
        assert!(DenseChainHost::load(&fixture.0, 8).unwrap().is_none());
    }

    #[test]
    fn explicit_paths_use_exact_order_and_override_discovery() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 6, true),
            Dtype::F32,
            [6, 8],
            true,
        );
        fixture.module(
            "2_Dense_alt",
            dense_config(8, 4, false),
            Dtype::F32,
            [4, 8],
            false,
        );
        fixture.write_modules(modules(&["2_Dense"]));

        let chain =
            DenseChainHost::load_with_paths(&fixture.0, 8, Some(vec!["2_Dense_alt".into()]))
                .unwrap()
                .unwrap();
        assert_eq!(chain.modules()[0].config.out_features, 4);
        assert_eq!(chain.output_width(), 4);
        assert!(
            DenseChainHost::load_with_paths(&fixture.0, 8, Some(Vec::new()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn automatic_discovery_with_pathless_transformer_requires_dense_weights() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 4, true),
            Dtype::F32,
            [4, 8],
            true,
        );
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"}
        ]));

        let chain = DenseChainHost::load_with_paths(&fixture.0, 8, None)
            .unwrap()
            .unwrap();
        assert_ne!(chain.output_width(), 8);

        fs::remove_file(fixture.0.join("2_Dense").join("model.safetensors")).unwrap();
        let error = match DenseChainHost::load_with_paths(&fixture.0, 8, None) {
            Ok(_) => panic!("missing Dense weights were silently ignored"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("no model.safetensors"), "{error}");
    }

    #[test]
    fn malformed_declared_dense_paths_cannot_be_hidden_by_empty_or_valid_overrides() {
        for declared in [serde_json::json!({}), serde_json::json!({"path": 7})] {
            let fixture = Fixture::new();
            fixture.module(
                "override_Dense",
                dense_config(8, 4, true),
                Dtype::F32,
                [4, 8],
                true,
            );
            let mut module = declared;
            module["type"] = serde_json::json!("sentence_transformers.models.Dense");
            fixture.write_modules(serde_json::json!([
                {"type": "sentence_transformers.models.Transformer"},
                {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                module
            ]));
            for paths in [None, Some(Vec::new()), Some(vec!["override_Dense".into()])] {
                assert!(DenseChainHost::load_with_paths(&fixture.0, 8, paths).is_err());
            }
        }
    }

    #[test]
    fn explicit_paths_work_without_sentence_transformer_metadata() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 4, true),
            Dtype::F32,
            [4, 8],
            true,
        );
        let chain = DenseChainHost::load_with_paths(&fixture.0, 8, Some(vec!["2_Dense".into()]))
            .unwrap()
            .unwrap();
        assert_eq!(chain.output_width(), 4);
    }

    #[test]
    fn rejects_discontinuous_widths_and_backbone_mismatch() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 6, true),
            Dtype::F32,
            [6, 8],
            true,
        );
        fixture.module(
            "3_Dense",
            dense_config(7, 4, false),
            Dtype::F32,
            [4, 7],
            false,
        );
        fixture.write_modules(modules(&["2_Dense", "3_Dense"]));
        let error = fixture.error(8);
        assert!(error.contains("not continuous"), "{error}");

        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 6, true),
            Dtype::F32,
            [6, 8],
            true,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        let error = fixture.error(7);
        assert!(error.contains("not continuous"), "{error}");
    }

    #[test]
    fn rejects_nonpositive_dimensions_missing_bias_boolean_and_bad_activation() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            serde_json::json!({"in_features": 0, "out_features": 4, "bias": true}),
            Dtype::F32,
            [4, 1],
            true,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        assert!(fixture.error(8).contains("positive"));

        for config in [
            serde_json::json!({"in_features": 8, "out_features": 4}),
            serde_json::json!({"in_features": 8, "out_features": 4, "bias": "true"}),
        ] {
            let fixture = Fixture::new();
            fixture.module("2_Dense", config, Dtype::F32, [4, 8], false);
            fixture.write_modules(modules(&["2_Dense"]));
            assert!(fixture.error(8).contains("bias must be a boolean"));
        }

        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            serde_json::json!({
                "in_features": 8,
                "out_features": 4,
                "bias": true,
                "activation_function": "torch.nn.modules.activation.ReLU"
            }),
            Dtype::F32,
            [4, 8],
            true,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        assert!(fixture.error(8).contains("Unsupported"));
    }

    #[test]
    fn rejects_wrong_tensor_shapes_missing_bias_and_unsupported_dtype() {
        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 4, true),
            Dtype::F32,
            [3, 8],
            true,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        assert!(fixture.error(8).contains("wrong shape"));

        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 4, true),
            Dtype::F32,
            [4, 8],
            false,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        assert!(fixture.error(8).contains("missing weight linear.bias"));

        let fixture = Fixture::new();
        fixture.module(
            "2_Dense",
            dense_config(8, 4, true),
            Dtype::I32,
            [4, 8],
            true,
        );
        fixture.write_modules(modules(&["2_Dense"]));
        assert!(fixture.error(8).contains("unsupported weight dtype"));
    }

    #[test]
    fn rejects_path_traversal_absolute_and_symlink_escape() {
        for relative in [
            "",
            "../2_Dense",
            "/tmp/2_Dense",
            "2_Dense/../escape",
            "2_Dense\\..",
        ] {
            let fixture = Fixture::new();
            fixture.write_modules(serde_json::json!([
                {"type": "sentence_transformers.models.Transformer"},
                {"type": "sentence_transformers.models.Dense", "path": relative}
            ]));
            let error = fixture.error(8);
            assert!(
                error.contains("path") || error.contains("missing"),
                "{relative}: {error}"
            );
        }

        #[cfg(unix)]
        {
            let fixture = Fixture::new();
            let outside = std::env::temp_dir().join(format!(
                "infinity-oxide-dense-outside-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(outside.join("2_Dense")).unwrap();
            std::os::unix::fs::symlink(&outside, fixture.0.join("2_Dense")).unwrap();
            fixture.write_modules(modules(&["2_Dense"]));
            let error = fixture.error(8);
            assert!(error.contains("escapes the model directory"), "{error}");
            let _ = fs::remove_dir_all(outside);
        }
    }

    #[test]
    fn rejects_pytorch_bin_only_modules_without_claiming_support() {
        let fixture = Fixture::new();
        let root = fixture.0.join("2_Dense");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&dense_config(8, 4, true)).unwrap(),
        )
        .unwrap();
        fs::write(root.join("pytorch_model.bin"), b"not a pickle").unwrap();
        fixture.write_modules(modules(&["2_Dense"]));
        let error = fixture.error(8);
        assert!(error.contains("only safetensors"), "{error}");
    }

    #[test]
    fn rejects_oversized_dense_config_before_reading_payload() {
        let fixture = Fixture::new();
        let root = fixture.0.join("2_Dense");
        fs::create_dir_all(&root).unwrap();
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(root.join("config.json"))
            .unwrap();
        file.set_len(16 * 1024 * 1024 + 1).unwrap();
        fixture.write_modules(modules(&["2_Dense"]));
        let error = fixture.error(8);
        assert!(error.contains("16 MiB"), "{error}");
    }

    #[test]
    fn dense_rows_refresh_retries_after_partial_failure() {
        // The shared refresh helper invalidates the cached shape before the
        // closure starts, so a partially updated chain is never reused.
        let mut cached = Some(2usize);
        let mut calls = 0usize;
        let failed = refresh_plan_rows(&mut cached, 3, || {
            calls += 1;
            Err(infer("injected Dense plan failure"))
        });
        assert!(failed.is_err());
        assert_eq!(cached, None);
        refresh_plan_rows(&mut cached, 3, || {
            calls += 1;
            Ok::<(), crate::BackendError>(())
        })
        .unwrap();
        assert_eq!((cached, calls), (Some(3), 2));
    }

    #[test]
    fn module_type_detection_matches_native_last_component() {
        for kind in [
            "sentence_transformers.models.Dense",
            "sentence_transformers.base.modules.dense.Dense",
        ] {
            assert!(is_dense_module_type(kind), "{kind}");
        }
        for kind in [
            "sentence_transformers.models.DenseCustom",
            "custom.Dense",
            "sentence_transformers.models.Pooling",
        ] {
            assert!(!is_dense_module_type(kind), "{kind}");
        }
    }
}
