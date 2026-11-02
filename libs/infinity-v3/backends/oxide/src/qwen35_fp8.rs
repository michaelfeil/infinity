// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Selective dynamic runtime FP8 linears for dense Qwen3.5 text.
//!
//! The native `models/qwen35_models.cpp` path enables dynamic FP8 for the dense
//! MLP unconditionally and for full/GDN attention projections when the scope is
//! `all`. DeltaNet `in_proj_a`/`in_proj_b`, convolution, recurrent state, norms,
//! rotary, vision and the LM head stay outside the FP8 scope. This module owns
//! exactly that subset over the shared Oxide FP8 framework: checkpoint weights
//! are quantized once with the qualified row E4M3FN recipe, activations are
//! quantized into one request Work, and each real `(M,N,K)` projection reuses a
//! prepared cuBLASLt plan. No tensor framework, runtime, graph, or per-layer
//! allocation is introduced.

use crate::{
    fp8::{Fp8Linear, Fp8LinearPlan, Fp8Methods, Fp8Provider, PreparedFp8, QuantizedRowStorage},
    infer,
    model::WeightSource,
    qwen35_config::{Qwen35Config, Qwen35LayerType},
    start,
    tensor::Bf16,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, DeviceBufferView, Module, Stream};
use infinity_v3_backend_core::BackendError;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Native scope policy: `mlp` quantizes only the dense FFN; `all` also
/// quantizes the full-attention and GDN projections (never the GDN A/B gates).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Qwen35Fp8Scope {
    Mlp,
    #[default]
    All,
}
impl Qwen35Fp8Scope {
    pub fn resolve(requested: Option<&str>) -> Result<Self, BackendError> {
        match requested.unwrap_or("all") {
            "mlp" => Ok(Self::Mlp),
            "all" => Ok(Self::All),
            _ => Err(start("Qwen3.5 FP8 scope must be mlp or all")),
        }
    }
    pub(crate) const fn quantizes_attention(self) -> bool {
        matches!(self, Self::All)
    }
}

/// Cold module root plus exact Native scope. The root must contain
/// `sm_90|sm_100|sm_120/{quant.cubin,math.cubin}` with the pinned hashes.
#[derive(Clone, Debug)]
pub struct Qwen35Fp8Options {
    pub module_root: PathBuf,
    pub scope: Qwen35Fp8Scope,
    pub expected_cublaslt_version: Option<usize>,
}
impl Qwen35Fp8Options {
    pub fn dynamic(module_root: impl Into<PathBuf>) -> Self {
        Self {
            module_root: module_root.into(),
            scope: Qwen35Fp8Scope::All,
            expected_cublaslt_version: None,
        }
    }
}

const QUANT_SM90: &str = "43ded6a32288359cf164e5ec7b554f392470c3c9fee5cf4245b857b767ad5e71";
const QUANT_SM100: &str = "458f5c6fbc3121cf4a75b84278dd0ad0096d26678776e006fadc5dba51cfa4ce";
const QUANT_SM120: &str = "c109c821450e2196930a1b1e5112a6bf7ecbecaf0cf44300b5eff0179ee09e2d";
const MATH_SM90: &str = "7c1884403045ba915a5a24162c794e081bef8843b1dccc53843062b176a7c730";
const MATH_SM100: &str = "5944fa2748496bc08aa0e1b59ea8a78eccdd3a43af8beedb1853ad710943afb8";
const MATH_SM120: &str = "e755b0976258608e706a088255892ef6ed75f7803f67ff4c4f089352772b89d2";

fn load_verified(
    context: &Arc<Context>,
    root: &Path,
    arch: &str,
    file: &str,
    expected: &str,
) -> Result<Arc<Module>, BackendError> {
    let path = root.join(arch).join(file);
    let bytes = std::fs::read(&path).map_err(start)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(start(format!(
            "Qwen3.5 FP8 AOT digest mismatch: {}",
            path.display()
        )));
    }
    context.load_module_from_image(&bytes).map_err(start)
}

fn module_identity(
    (major, minor): (i32, i32),
) -> Result<(&'static str, &'static str, &'static str), BackendError> {
    match (major, minor) {
        (9, 0) => Ok(("sm_90", QUANT_SM90, MATH_SM90)),
        (10, 0) => Ok(("sm_100", QUANT_SM100, MATH_SM100)),
        (12, 0) => Ok(("sm_120", QUANT_SM120, MATH_SM120)),
        _ => Err(start(
            "Dynamic FP8 Qwen3.5 requires H100 SM90, B200 SM100 or RTX PRO 6000 SM120",
        )),
    }
}

fn load_methods(
    context: &Arc<Context>,
    capability: (i32, i32),
    root: &Path,
) -> Result<Fp8Methods<Bf16>, BackendError> {
    let (arch, quant, math) = module_identity(capability)?;
    let quantizer = load_verified(context, root, arch, "quant.cubin", quant)?;
    let math = load_verified(context, root, arch, "math.cubin", math)?;
    // SAFETY: the pinned hashes bind the qualified native row-quantizer,
    // postscale and model-bias ABIs and launch geometry.
    unsafe { Fp8Methods::from_modules(&quantizer, &math) }
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("FP8 Qwen3.5 {name} overflow")))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("FP8 Qwen3.5 {name} overflow")))
}

fn linear_value_width(config: &Qwen35Config) -> Result<usize, BackendError> {
    checked_product(
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        "linear value width",
    )
}

fn linear_qkv_width(config: &Qwen35Config) -> Result<usize, BackendError> {
    let key = checked_product(
        config.linear_num_key_heads,
        config.linear_key_head_dim,
        "linear key width",
    )?;
    let value = linear_value_width(config)?;
    checked_add(
        checked_product(key, 2, "linear QKV width")?,
        value,
        "linear QKV width",
    )
}

/// Unique `(input, output)` projections Native quantizes for this scope.
fn selected_geometries(
    config: &Qwen35Config,
    scope: Qwen35Fp8Scope,
) -> Result<Vec<(usize, usize)>, BackendError> {
    let mut out = Vec::with_capacity(8);
    let mut push = |input: usize, output: usize| {
        if !out.contains(&(input, output)) {
            out.push((input, output));
        }
    };
    let h = config.hidden_size;
    let i = config.intermediate_size;
    push(h, checked_product(i, 2, "gate/up width")?);
    push(i, h);
    if scope.quantizes_attention() {
        let has_full = config
            .layer_types
            .iter()
            .any(|kind| matches!(kind, Qwen35LayerType::Full));
        let has_linear = config
            .layer_types
            .iter()
            .any(|kind| matches!(kind, Qwen35LayerType::Linear));
        if has_full {
            let q = config.q_width();
            let kv = config.kv_width();
            push(h, checked_product(q, 2, "query-gate width")?);
            push(h, kv);
            push(q, h);
        }
        if has_linear {
            push(h, linear_qkv_width(config)?);
            push(h, linear_value_width(config)?);
            push(linear_value_width(config)?, h);
        }
    }
    Ok(out)
}

fn validate_geometries(
    config: &Qwen35Config,
    scope: Qwen35Fp8Scope,
) -> Result<Vec<(usize, usize)>, BackendError> {
    let geometries = selected_geometries(config, scope)?;
    for &(input, output) in &geometries {
        Fp8LinearPlan::<Bf16>::new(0, input, output).map_err(start)?;
    }
    Ok(geometries)
}

fn arena_widths(config: &Qwen35Config, scope: Qwen35Fp8Scope) -> Result<Vec<usize>, BackendError> {
    let geometries = validate_geometries(config, scope)?;
    let mut widths = Vec::with_capacity(4);
    for &(input, _) in &geometries {
        if !widths.contains(&input) {
            widths.push(input);
        }
    }
    Ok(widths)
}

fn max_output_width(geometries: &[(usize, usize)]) -> usize {
    geometries
        .iter()
        .map(|&(_, output)| output)
        .max()
        .unwrap_or(0)
}

pub(crate) enum Qwen35Fp8Attention {
    Full {
        query_gate: Fp8Linear<Bf16>,
        key: Fp8Linear<Bf16>,
        value: Fp8Linear<Bf16>,
        output: Fp8Linear<Bf16>,
    },
    Linear {
        qkv: Fp8Linear<Bf16>,
        z: Fp8Linear<Bf16>,
        output: Fp8Linear<Bf16>,
    },
}

pub(crate) struct Qwen35Fp8Layer {
    pub(crate) mlp_gate_up: Fp8Linear<Bf16>,
    pub(crate) mlp_down: Fp8Linear<Bf16>,
    pub(crate) attention: Option<Qwen35Fp8Attention>,
}

/// Cold provider, methods and canonical quantized checkpoint weights.
pub(crate) struct Qwen35Fp8 {
    provider: Fp8Provider,
    methods: Fp8Methods<Bf16>,
    scope: Qwen35Fp8Scope,
    major: i32,
    geometries: Vec<(usize, usize)>,
    layers: Vec<Qwen35Fp8Layer>,
}
impl Qwen35Fp8 {
    pub(crate) fn new(
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        config: &Qwen35Config,
        options: &Qwen35Fp8Options,
    ) -> Result<Self, BackendError> {
        let geometries = validate_geometries(config, options.scope)?;
        let capability = context.compute_capability().map_err(start)?;
        let methods = load_methods(context, capability, &options.module_root)?;
        let provider = Fp8Provider::new(
            Arc::clone(context),
            Arc::clone(stream),
            options.expected_cublaslt_version,
        )?;
        Ok(Self {
            provider,
            methods,
            scope: options.scope,
            major: capability.0,
            geometries,
            layers: Vec::new(),
        })
    }

    pub(crate) const fn scope(&self) -> Qwen35Fp8Scope {
        self.scope
    }

    pub(crate) const fn quantizes_attention(&self) -> bool {
        self.scope.quantizes_attention()
    }

    pub(crate) fn provider(&self) -> &Fp8Provider {
        &self.provider
    }

    pub(crate) const fn methods(&self) -> &Fp8Methods<Bf16> {
        &self.methods
    }

    pub(crate) fn layers(&self) -> &[Qwen35Fp8Layer] {
        &self.layers
    }

    /// Split the provider from its immutable methods and canonical weights so
    /// the request path can bind the provider mutably without re-reading it.
    pub(crate) fn split_mut(&mut self) -> (&mut Fp8Provider, &Fp8Methods<Bf16>, &[Qwen35Fp8Layer]) {
        (&mut self.provider, &self.methods, &self.layers)
    }

    pub(crate) fn finish(&self, expected: usize) -> Result<(), BackendError> {
        if self.layers.len() != expected {
            return Err(start(format!(
                "FP8 Qwen3.5 has {} quantized layers, expected {expected}",
                self.layers.len()
            )));
        }
        Ok(())
    }

    pub(crate) fn push_layer(&mut self, layer: Qwen35Fp8Layer) -> Result<(), BackendError> {
        self.layers
            .try_reserve(1)
            .map_err(|_| start("FP8 Qwen3.5 layer list allocation failed"))?;
        self.layers.push(layer);
        Ok(())
    }

    pub(crate) fn needs_postscale_scratch(&self) -> bool {
        self.major != 9
    }

    pub(crate) fn max_output_width(&self) -> usize {
        max_output_width(&self.geometries)
    }

    /// Cold quantization from the shared bounded uploader. The cold stream is
    /// drained exactly once per matrix before the BF16 staging owner is freed;
    /// no request path calls this function.
    pub(crate) fn quantize_linear(
        &self,
        stream: &Arc<Stream>,
        uploader: &mut WeightUploader,
        sources: &[&WeightSource],
        rows: usize,
        width: usize,
    ) -> Result<Fp8Linear<Bf16>, BackendError> {
        let elements = checked_product(rows, width, "weight extent")?;
        let source = uploader.upload(sources)?;
        if source.len() != elements {
            return Err(start(format!(
                "FP8 Qwen3.5 weight upload has {} elements, expected {elements}",
                source.len()
            )));
        }
        let mut storage = QuantizedRowStorage::allocate(stream, rows, width).map_err(infer)?;
        // SAFETY: cold checkpoint staging, one ordered stream, every owner kept
        // live until the explicit cold drain below.
        let queued = unsafe {
            self.methods
                .quantize_bound(stream, &source, &mut storage, rows)
        };
        if let Err(error) = queued.and_then(|()| stream.synchronize().map_err(infer)) {
            if stream.synchronize().is_err() {
                std::mem::forget(storage);
                std::mem::forget(source);
            }
            return Err(error);
        }
        // SAFETY: the ordered quantizer drained; canonical rows are complete.
        let rows = unsafe { storage.assume_quantized(rows) }.map_err(infer)?;
        // SAFETY: drained cold initialization and native row recipe verified.
        let linear = unsafe { Fp8Linear::from_quantized_weights(rows) }.map_err(infer)?;
        drop(source);
        Ok(linear)
    }
}

type PlanKey = (usize, usize, usize);

/// One request Work's FP8 activation arenas, optional portable F32 postscale
/// scratch and a bounded real-M prepared-plan cache. The enclosing Work owns
/// every allocation, error lifetime and completion boundary.
pub(crate) struct Qwen35Fp8Work {
    widths: Vec<usize>,
    arenas: Vec<QuantizedRowStorage>,
    scratch: Option<DeviceBuffer<f32>>,
    plans: Vec<(PlanKey, PreparedFp8<Bf16>)>,
}
/// Hard prepared-plan bound per Work. Reaching it rejects the new real-M shape
/// instead of evicting a plan (whose drop would drain the outer stream).
const MAX_PLANS: usize = 1024;

impl Qwen35Fp8Work {
    pub(crate) fn allocation_bytes(
        capacity: usize,
        config: &Qwen35Config,
        scope: Qwen35Fp8Scope,
        needs_scratch: bool,
    ) -> Result<usize, BackendError> {
        let widths = arena_widths(config, scope)?;
        let mut bytes = 0usize;
        for width in widths {
            let values = checked_product(capacity, width, "arena values")?;
            let scales = checked_product(capacity, 4, "arena scales")?;
            bytes = checked_add(
                bytes,
                checked_add(values, scales, "arena bytes")?,
                "arena bytes",
            )?;
        }
        if needs_scratch {
            let width = max_output_width(&validate_geometries(config, scope)?);
            let f32_bytes = checked_product(width, 4, "scratch width")?;
            let scratch = checked_product(capacity, f32_bytes, "scratch bytes")?;
            bytes = checked_add(bytes, scratch, "arena bytes")?;
        }
        Ok(bytes)
    }

    pub(crate) fn new(
        stream: &Arc<Stream>,
        capacity: usize,
        config: &Qwen35Config,
        scope: Qwen35Fp8Scope,
        needs_scratch: bool,
    ) -> Result<Self, BackendError> {
        Self::allocation_bytes(capacity, config, scope, needs_scratch)?;
        let widths = arena_widths(config, scope)?;
        let mut arenas = Vec::new();
        arenas.try_reserve_exact(widths.len()).map_err(infer)?;
        for &width in &widths {
            arenas.push(QuantizedRowStorage::allocate(stream, capacity, width).map_err(infer)?);
        }
        let scratch = if needs_scratch {
            let width = max_output_width(&validate_geometries(config, scope)?);
            let count = checked_product(capacity, width, "scratch elements")?;
            Some(DeviceBuffer::zeroed(stream, count).map_err(infer)?)
        } else {
            None
        };
        Ok(Self {
            widths,
            arenas,
            scratch,
            plans: Vec::new(),
        })
    }

    pub(crate) fn arena_for_width(&self, width: usize) -> Result<usize, BackendError> {
        self.widths
            .iter()
            .position(|&existing| existing == width)
            .ok_or_else(|| {
                infer(format!(
                    "FP8 Qwen3.5 has no activation arena of width {width}"
                ))
            })
    }

    fn arena(&self, arena: usize) -> Result<&QuantizedRowStorage, BackendError> {
        self.arenas
            .get(arena)
            .ok_or_else(|| infer("FP8 Qwen3.5 activation arena index is invalid"))
    }

    fn arena_mut(&mut self, arena: usize) -> Result<&mut QuantizedRowStorage, BackendError> {
        self.arenas
            .get_mut(arena)
            .ok_or_else(|| infer("FP8 Qwen3.5 activation arena index is invalid"))
    }

    fn ensure_plan(
        &mut self,
        provider: &Fp8Provider,
        rows: usize,
        weight: &Fp8Linear<Bf16>,
    ) -> Result<(), BackendError> {
        let key: PlanKey = (rows, weight.input_width(), weight.output_width());
        if self.plans.iter().any(|(existing, _)| *existing == key) {
            return Ok(());
        }
        if self.plans.len() >= MAX_PLANS {
            return Err(infer(
                "FP8 Qwen3.5 prepared-plan cache is full; split the Work shape class",
            ));
        }
        let arena = self.arena_for_width(key.1)?;
        let metadata = Fp8LinearPlan::<Bf16>::new(key.0, key.1, key.2).map_err(infer)?;
        let weight = weight.weights().map_err(infer)?;
        let input = self.arena(arena)?.planned_view(rows).map_err(infer)?;
        let plan = provider.prepare(metadata, &input, &weight)?;
        self.plans.push((key, plan));
        Ok(())
    }

    /// Prepare the real row count once per Work cache entry. Duplicate layer
    /// geometries share one descriptor plan; no per-layer plan or context query
    /// is created.
    pub(crate) fn prepare(&mut self, fp8: &Qwen35Fp8, rows: usize) -> Result<(), BackendError> {
        if rows == 0 {
            return Err(infer("FP8 Qwen3.5 does not admit zero token rows"));
        }
        if rows > self.arena(0)?.capacity() {
            return Err(infer("FP8 Qwen3.5 real rows exceed the Work arena"));
        }
        let first = fp8
            .layers()
            .first()
            .ok_or_else(|| infer("FP8 Qwen3.5 has no quantized layer"))?;
        self.ensure_plan(fp8.provider(), rows, &first.mlp_gate_up)?;
        self.ensure_plan(fp8.provider(), rows, &first.mlp_down)?;
        if let Some(layer) = fp8
            .layers()
            .iter()
            .find(|layer| matches!(layer.attention, Some(Qwen35Fp8Attention::Full { .. })))
        {
            let Some(Qwen35Fp8Attention::Full {
                query_gate,
                key,
                value,
                output,
            }) = &layer.attention
            else {
                return Err(infer("FP8 Qwen3.5 full attention layer mismatch"));
            };
            self.ensure_plan(fp8.provider(), rows, query_gate)?;
            self.ensure_plan(fp8.provider(), rows, key)?;
            self.ensure_plan(fp8.provider(), rows, value)?;
            self.ensure_plan(fp8.provider(), rows, output)?;
        }
        if let Some(layer) = fp8
            .layers()
            .iter()
            .find(|layer| matches!(layer.attention, Some(Qwen35Fp8Attention::Linear { .. })))
        {
            let Some(Qwen35Fp8Attention::Linear { qkv, z, output }) = &layer.attention else {
                return Err(infer("FP8 Qwen3.5 linear attention layer mismatch"));
            };
            self.ensure_plan(fp8.provider(), rows, qkv)?;
            self.ensure_plan(fp8.provider(), rows, z)?;
            self.ensure_plan(fp8.provider(), rows, output)?;
        }
        Ok(())
    }

    /// Enqueue one row quantizer into the selected Work arena.
    ///
    /// # Safety
    /// The producer already wrote the view on the enclosing ordered stream and
    /// every owner survives completion.
    pub(crate) unsafe fn quantize(
        &mut self,
        stream: &Stream,
        methods: &Fp8Methods<Bf16>,
        arena: usize,
        source: &DeviceBufferView<'_, Bf16>,
        rows: usize,
    ) -> Result<(), BackendError> {
        let width = self.arena(arena)?.width();
        if source.len() < rows.saturating_mul(width) {
            return Err(infer("FP8 Qwen3.5 activation view is too small"));
        }
        let storage = self.arena_mut(arena)?;
        unsafe { methods.quantize_view_bound(stream, source, storage, rows) }
    }

    /// Submit one prepared projection into a retained phase slab at a checked
    /// element offset.
    ///
    /// # Safety
    /// The enclosing Work already bound the context, the activation arena was
    /// quantized on this stream, and every owner survives completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn project(
        &mut self,
        provider: &mut Fp8Provider,
        methods: &Fp8Methods<Bf16>,
        arena: usize,
        weight: &Fp8Linear<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        output_offset: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        let key: PlanKey = (rows, weight.input_width(), weight.output_width());
        let Self {
            arenas,
            plans,
            scratch,
            ..
        } = self;
        let arena = arenas
            .get(arena)
            .ok_or_else(|| infer("FP8 Qwen3.5 activation arena index is invalid"))?;
        if arena.width() != key.1 {
            return Err(infer("FP8 Qwen3.5 arena width differs from the projection"));
        }
        let input = arena.planned_view(rows).map_err(infer)?;
        let weight = weight.weights().map_err(infer)?;
        let plan = plans
            .iter_mut()
            .find_map(|(existing, plan)| (*existing == key).then_some(plan))
            .ok_or_else(|| infer("FP8 Qwen3.5 real-row plan was not prepared"))?;
        unsafe {
            provider.project_bound_at(
                plan,
                methods,
                &input,
                &weight,
                output,
                output_offset,
                scratch.as_mut(),
                None,
            )
        }
    }
}
