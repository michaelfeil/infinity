// Copyright (c) 2026 Oxide MoE Authors. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//
// Original Rust source offered under the Infinity Restricted MIT License 1.0
// (see LICENSE). This is not standard MIT; Appendix A provider use is prohibited.

//! Owning, validating loader for the `oxide_moe.h` ABI1 boundary.
//!
//! Build/model/workspace/prepare metadata stay CPU-only. A prepared plan can
//! only be enqueued through the `unsafe` [`PreparedPlan::forward_bound`] method,
//! which accepts the exact ABI1 `Buffers` descriptor and keeps the loaded
//! library and plan alive. No safe API accepts caller raw pointers, and a
//! failed enqueue never retries or falls back.

use crate::moe_abi::*;
use libloading::Library;
use std::{ffi::OsStr, fmt, mem::size_of, sync::Arc};

macro_rules! ensure {
    ($condition:expr, $error:expr) => {
        if !$condition {
            return Err($error);
        }
    };
}

#[derive(Debug)]
pub enum Error {
    Library(libloading::Error),
    MissingSymbol(&'static str),
    AbiMismatch {
        expected: u32,
        actual: u32,
    },
    UnsupportedAbi,
    InvalidArch(Arch),
    ArchMismatch {
        expected: Arch,
        built: Arch,
    },
    UnsupportedModel(ModelKind),
    UnsupportedKernel(KernelKind),
    ShapeMismatch(&'static str),
    InvalidArgument(&'static str),
    Overflow,
    WorkspaceTooSmall,
    KernelFailure,
    Internal,
    Malformed(&'static str),
    Status {
        operation: &'static str,
        code: Status,
    },
    ForwardFailed {
        status: Status,
        kernel_status: i32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub model: ModelKind,
    pub tokens: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub experts: i32,
    pub top_k: i32,
    pub renormalize: i32,
}

/// An owned prepared plan. The private ABI plan cannot be mutated or read as
/// raw pointers; enqueue goes through the checked `forward_bound` contract, and
/// the retained `Arc<Api>` keeps the library loaded.
pub struct PreparedPlan {
    _api: Arc<Api>,
    plan: Plan,
}

impl PreparedPlan {
    pub fn workspace_bytes(&self) -> usize {
        self.plan.workspace_bytes
    }

    pub fn selected_kernel(&self) -> KernelKind {
        self.plan.kernel
    }

    pub fn geometry(&self) -> Geometry {
        Geometry {
            model: self.plan.model,
            tokens: self.plan.tokens,
            hidden: self.plan.hidden,
            intermediate: self.plan.intermediate,
            experts: self.plan.experts,
            top_k: self.plan.top_k,
            renormalize: self.plan.renormalize,
        }
    }

    /// Enqueue the cached native routed operation for this retained plan.
    ///
    /// # Safety
    ///
    /// The caller must bind the plan's CUDA context on this thread. Every
    /// device pointer in `buffers` must address live storage in that context
    /// for the full geometry of this plan, `workspace` must be a writable
    /// 256-byte-aligned range of at least `workspace_bytes`, and every owner,
    /// including the stream, must stay alive without conflicting access until
    /// that stream drains. The caller retains ownership of completion; this
    /// method never synchronizes. On any non-`OK` status the failure is final:
    /// there is no retry or fallback, and the outer owner must drain.
    pub(crate) unsafe fn forward_bound(&self, buffers: &Buffers) -> Result<(), Error> {
        validate_forward_buffers(buffers, self.plan.workspace_bytes)?;
        let mut kernel_status = 0i32;
        // SAFETY: the caller guarantees the borrowed buffers and retained plan;
        // the library stays loaded through the retained Arc above.
        let code = unsafe { (self._api.forward)(&self.plan, buffers, &mut kernel_status) };
        if code != STATUS_OK {
            return Err(Error::ForwardFailed {
                status: code,
                kernel_status,
            });
        }
        Ok(())
    }
}

/// A fully validated ABI1 library. No unsafe `Send`/`Sync` implementation is
/// declared: `libloading::Library` and copied function pointers already are
/// both `Send` and `Sync`, and no mutable state is shared.
pub struct Api {
    _library: Library,
    abi_version: AbiVersionFn,
    query_model: QueryModelFn,
    query_workspace: QueryWorkspaceFn,
    prepare: PrepareFn,
    forward: ForwardFn,
    build: BuildInfo,
}

impl Api {
    /// Load and validate an externally supplied native ABI1 library.
    ///
    /// # Safety
    ///
    /// `path` must name trusted native code compiled for the matching
    /// `oxide_moe.h` ABI. Loading runs platform library initializers before any
    /// of the ABI checks below, so untrusted files must never be passed. The
    /// resolved functions are trusted to implement the C signatures exactly;
    /// function pointers are copied out and are never transmuted into static
    /// references. The library stays alive inside every `Api`/`PreparedPlan`.
    pub unsafe fn load<P: AsRef<OsStr>>(path: P, expected_arch: Arch) -> Result<Arc<Self>, Error> {
        if !arch_valid(expected_arch) {
            return Err(Error::InvalidArch(expected_arch));
        }
        // SAFETY: the caller promises trusted native code.
        let library = unsafe { Library::new(path) }.map_err(Error::Library)?;
        macro_rules! symbol {
            ($ty:ty, $name:literal) => {
                // SAFETY: the trusted library exports the header's exact C signature.
                unsafe {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|_| Error::MissingSymbol($name))?
                }
            };
        }
        let abi_version = symbol!(AbiVersionFn, "oxide_moe_abi_version");
        let query_build = symbol!(QueryBuildFn, "oxide_moe_query_build");
        let query_model = symbol!(QueryModelFn, "oxide_moe_query_model");
        let query_workspace = symbol!(QueryWorkspaceFn, "oxide_moe_query_workspace");
        let prepare = symbol!(PrepareFn, "oxide_moe_prepare");
        let forward = symbol!(ForwardFn, "oxide_moe_forward_bf16");

        // SAFETY: copied function pointer with the declared C signature.
        let actual = unsafe { abi_version() };
        if actual != ABI_VERSION {
            return Err(Error::AbiMismatch {
                expected: ABI_VERSION,
                actual,
            });
        }
        let mut build = BuildInfo::new();
        // SAFETY: `build` is a valid writable ABI1 structure.
        let code = unsafe { query_build(&mut build) };
        if code != STATUS_OK {
            return Err(status_error("oxide_moe_query_build", code, -1, -1));
        }
        validate_build(&build, expected_arch)?;
        Ok(Arc::new(Self {
            _library: library,
            abi_version,
            query_model,
            query_workspace,
            prepare,
            forward,
            build,
        }))
    }

    pub fn abi_version(&self) -> u32 {
        // SAFETY: validated pure getter with the declared C signature.
        unsafe { (self.abi_version)() }
    }

    pub fn build_info(&self) -> &BuildInfo {
        &self.build
    }

    pub fn query_model(&self, model: ModelKind) -> Result<ModelInfo, Error> {
        let bit = model_bit(model).ok_or(Error::UnsupportedModel(model))?;
        if self.build.model_mask & bit == 0 {
            return Err(Error::UnsupportedModel(model));
        }
        let mut out = ModelInfo::new();
        // SAFETY: `out` is a valid writable ABI1 structure; the library is retained.
        let code = unsafe { (self.query_model)(model, &mut out) };
        if code != STATUS_OK {
            return Err(status_error("oxide_moe_query_model", code, model, -1));
        }
        validate_model_info(&out, model, self.build.kernel_mask)?;
        Ok(out)
    }

    pub fn query_workspace(&self, shape: &Shape) -> Result<WorkspaceInfo, Error> {
        self.validate_shape(shape)?;
        let mut out = WorkspaceInfo::new();
        // SAFETY: `shape` is validated and `out` is a writable ABI1 structure.
        let code = unsafe { (self.query_workspace)(shape, &mut out) };
        if code != STATUS_OK {
            return Err(status_error(
                "oxide_moe_query_workspace",
                code,
                shape.model,
                shape.kernel,
            ));
        }
        check_header(
            out.struct_size,
            out.abi_version,
            size_of::<WorkspaceInfo>(),
            "workspace header",
        )?;
        check_reserved(&out.reserved, "workspace reserved")?;
        ensure!(
            out.arch == self.build.arch,
            Error::ArchMismatch {
                expected: self.build.arch,
                built: out.arch,
            }
        );
        let bit = kernel_bit(out.kernel).ok_or(Error::Malformed("workspace kernel"))?;
        ensure!(
            self.build.kernel_mask & bit != 0,
            Error::Malformed("workspace kernel mask")
        );
        ensure!(
            out.kernel == selected_kernel(shape.kernel, &self.build),
            Error::Malformed("workspace selected kernel")
        );
        ensure!(
            out.workspace_bytes != 0,
            Error::Malformed("workspace bytes")
        );
        ensure!(out.flags == 0, Error::Malformed("workspace flags"));
        Ok(out)
    }

    pub fn prepare(self: &Arc<Self>, shape: &Shape) -> Result<PreparedPlan, Error> {
        let info = self.validate_shape(shape)?;
        let workspace = self.query_workspace(shape)?;
        let mut plan = Plan::new();
        // SAFETY: `shape` is validated and `plan` is a writable ABI1 structure.
        let code = unsafe { (self.prepare)(shape, &mut plan) };
        if code != STATUS_OK {
            return Err(status_error(
                "oxide_moe_prepare",
                code,
                shape.model,
                shape.kernel,
            ));
        }
        validate_plan(&plan, shape, &info, &self.build)?;
        ensure!(
            plan.workspace_bytes == workspace.workspace_bytes,
            Error::Malformed("plan workspace differs from query")
        );
        Ok(PreparedPlan {
            _api: Arc::clone(self),
            plan,
        })
    }

    fn validate_shape(&self, shape: &Shape) -> Result<ModelInfo, Error> {
        check_header(
            shape.struct_size,
            shape.abi_version,
            size_of::<Shape>(),
            "shape header",
        )?;
        check_reserved(&shape.reserved, "shape reserved")?;
        let model = model_bit(shape.model).ok_or(Error::UnsupportedModel(shape.model))?;
        ensure!(
            self.build.model_mask & model != 0,
            Error::UnsupportedModel(shape.model)
        );
        if shape.kernel != KERNEL_AUTO {
            let kernel = kernel_bit(shape.kernel).ok_or(Error::UnsupportedKernel(shape.kernel))?;
            ensure!(
                self.build.kernel_mask & kernel != 0,
                Error::UnsupportedKernel(shape.kernel)
            );
        }
        ensure!(
            shape.tokens > 0 && shape.hidden > 0 && shape.intermediate > 0,
            Error::InvalidArgument("shape dimensions")
        );
        ensure!(
            shape.hidden & 7 == 0 && shape.intermediate & 7 == 0,
            Error::ShapeMismatch("shape dimensions must be multiples of 8")
        );
        ensure!(
            shape.renormalize == 0 || shape.renormalize == 1,
            Error::InvalidArgument("shape renormalize")
        );
        let info = self.query_model(shape.model)?;
        ensure!(
            shape.experts == info.experts && shape.top_k == info.top_k,
            Error::ShapeMismatch("experts/top_k")
        );
        ensure!(
            info.hidden == 0 || shape.hidden == info.hidden,
            Error::ShapeMismatch("fixed hidden")
        );
        ensure!(
            info.intermediate == 0 || shape.intermediate == info.intermediate,
            Error::ShapeMismatch("fixed intermediate")
        );
        Ok(info)
    }
}

fn arch_valid(arch: Arch) -> bool {
    matches!(arch, ARCH_SM90 | ARCH_SM100 | ARCH_SM120)
}

fn selected_kernel(request: KernelKind, build: &BuildInfo) -> KernelKind {
    if request == KERNEL_AUTO {
        build.default_kernel
    } else {
        request
    }
}

fn model_bit(model: ModelKind) -> Option<u32> {
    match model {
        MODEL_QWEN3_E128_TOP8 => Some(MODEL_MASK_QWEN3),
        MODEL_QWEN35_E256_TOP8 => Some(MODEL_MASK_QWEN35),
        MODEL_GEMMA4_E128_TOP8 => Some(MODEL_MASK_GEMMA4),
        _ => None,
    }
}

fn kernel_bit(kernel: KernelKind) -> Option<u32> {
    match kernel {
        KERNEL_PORTABLE => Some(KERNEL_MASK_PORTABLE),
        KERNEL_HOPPER => Some(KERNEL_MASK_HOPPER),
        _ => None,
    }
}

fn check_header(size: u32, abi: u32, expected: usize, what: &'static str) -> Result<(), Error> {
    ensure!(abi == ABI_VERSION, Error::UnsupportedAbi);
    ensure!(size == expected as u32, Error::Malformed(what));
    Ok(())
}

const FORWARD_WORKSPACE_ALIGNMENT: usize = 256;

fn validate_forward_buffers(buffers: &Buffers, required_workspace: usize) -> Result<(), Error> {
    check_header(
        buffers.struct_size,
        buffers.abi_version,
        size_of::<Buffers>(),
        "forward buffers header",
    )?;
    check_reserved(&buffers.reserved, "forward buffers reserved")?;
    ensure!(
        buffers.flags == 0,
        Error::Malformed("forward buffers flags")
    );
    ensure!(
        required_workspace != 0,
        Error::Malformed("forward plan workspace")
    );
    ensure!(
        buffers.workspace_bytes >= required_workspace,
        Error::WorkspaceTooSmall
    );
    ensure!(
        !buffers.logits.is_null()
            && !buffers.input_bf16.is_null()
            && !buffers.gate_up_weight_bf16.is_null()
            && !buffers.down_weight_bf16.is_null()
            && !buffers.output_bf16.is_null()
            && !buffers.workspace.is_null()
            && !buffers.stream.is_null(),
        Error::InvalidArgument("forward buffer pointer")
    );
    ensure!(
        (buffers.workspace as usize).is_multiple_of(FORWARD_WORKSPACE_ALIGNMENT),
        Error::InvalidArgument("forward workspace alignment")
    );
    Ok(())
}

fn check_reserved(reserved: &[u32], what: &'static str) -> Result<(), Error> {
    ensure!(
        !reserved.iter().any(|&value| value != 0),
        Error::Malformed(what)
    );
    Ok(())
}

fn status_error(
    operation: &'static str,
    code: Status,
    model: ModelKind,
    kernel: KernelKind,
) -> Error {
    match code {
        STATUS_UNSUPPORTED_MODEL => Error::UnsupportedModel(model),
        STATUS_UNSUPPORTED_KERNEL => Error::UnsupportedKernel(kernel),
        STATUS_UNSUPPORTED_ABI => Error::UnsupportedAbi,
        STATUS_SHAPE_MISMATCH => Error::ShapeMismatch(operation),
        STATUS_OVERFLOW => Error::Overflow,
        STATUS_WORKSPACE_TOO_SMALL => Error::WorkspaceTooSmall,
        STATUS_KERNEL_ERROR => Error::KernelFailure,
        STATUS_INTERNAL_ERROR => Error::Internal,
        other => Error::Status {
            operation,
            code: other,
        },
    }
}

fn validate_build(build: &BuildInfo, expected: Arch) -> Result<(), Error> {
    check_header(
        build.struct_size,
        build.abi_version,
        size_of::<BuildInfo>(),
        "build header",
    )?;
    check_reserved(&build.reserved, "build reserved")?;
    ensure!(arch_valid(build.arch), Error::InvalidArch(build.arch));
    ensure!(
        build.arch == expected,
        Error::ArchMismatch {
            expected,
            built: build.arch
        }
    );
    let default =
        kernel_bit(build.default_kernel).ok_or(Error::Malformed("build default kernel"))?;
    ensure!(
        build.kernel_mask != 0
            && build.kernel_mask & !(KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER) == 0
            && build.kernel_mask & default != 0,
        Error::Malformed("build kernel mask")
    );
    ensure!(
        build.model_mask != 0
            && build.model_mask & !(MODEL_MASK_QWEN3 | MODEL_MASK_QWEN35 | MODEL_MASK_GEMMA4) == 0,
        Error::Malformed("build model mask")
    );
    ensure!(
        build.flags & !(BUILD_FLAG_PORTABLE | BUILD_FLAG_HOPPER | BUILD_FLAG_COMPILE_ONLY) == 0,
        Error::Malformed("build flags")
    );
    let (kernel, mask, flags) = if build.arch == ARCH_SM90 {
        (
            KERNEL_HOPPER,
            KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER,
            BUILD_FLAG_PORTABLE | BUILD_FLAG_HOPPER,
        )
    } else {
        (
            KERNEL_PORTABLE,
            KERNEL_MASK_PORTABLE,
            BUILD_FLAG_PORTABLE | BUILD_FLAG_COMPILE_ONLY,
        )
    };
    ensure!(
        (build.default_kernel, build.kernel_mask, build.flags) == (kernel, mask, flags),
        Error::Malformed("build architecture policy")
    );
    ensure!(
        build.model_mask == MODEL_MASK_QWEN3 | MODEL_MASK_QWEN35 | MODEL_MASK_GEMMA4,
        Error::Malformed("build model policy")
    );
    Ok(())
}

fn validate_model_info(
    info: &ModelInfo,
    model: ModelKind,
    build_kernel_mask: u32,
) -> Result<(), Error> {
    check_header(
        info.struct_size,
        info.abi_version,
        size_of::<ModelInfo>(),
        "model header",
    )?;
    check_reserved(&info.reserved, "model reserved")?;
    ensure!(info.model == model, Error::Malformed("model identity"));
    ensure!(
        info.experts > 0 && info.top_k > 0 && info.top_k <= info.experts,
        Error::Malformed("model experts/top_k")
    );
    ensure!(
        info.hidden >= 0 && info.intermediate >= 0,
        Error::Malformed("model dimensions")
    );
    ensure!(
        info.requires_expert_scales & !1 == 0 && info.supports_renormalize & !1 == 0,
        Error::Malformed("model capability flags")
    );
    ensure!(
        info.kernel_mask != 0
            && info.kernel_mask & !(KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER) == 0
            && info.kernel_mask & !build_kernel_mask == 0,
        Error::Malformed("model kernel mask")
    );
    let expected = match model {
        MODEL_QWEN3_E128_TOP8 => (128, 8, 0, 0, 0, 1),
        MODEL_QWEN35_E256_TOP8 => (256, 8, 0, 0, 0, 1),
        MODEL_GEMMA4_E128_TOP8 => (128, 8, 2816, 704, 1, 0),
        _ => return Err(Error::UnsupportedModel(model)),
    };
    ensure!(
        (
            info.experts,
            info.top_k,
            info.hidden,
            info.intermediate,
            info.requires_expert_scales,
            info.supports_renormalize
        ) == expected
            && info.kernel_mask == build_kernel_mask,
        Error::Malformed("model ABI1 policy")
    );
    Ok(())
}

fn validate_plan(
    plan: &Plan,
    shape: &Shape,
    info: &ModelInfo,
    build: &BuildInfo,
) -> Result<(), Error> {
    check_header(
        plan.struct_size,
        plan.abi_version,
        size_of::<Plan>(),
        "plan header",
    )?;
    check_reserved(&plan.reserved, "plan reserved")?;
    ensure!(
        plan.arch == build.arch,
        Error::ArchMismatch {
            expected: build.arch,
            built: plan.arch,
        }
    );
    let kernel = kernel_bit(plan.kernel).ok_or(Error::Malformed("plan kernel"))?;
    ensure!(
        build.kernel_mask & kernel != 0,
        Error::Malformed("plan kernel mask")
    );
    ensure!(
        plan.kernel == selected_kernel(shape.kernel, build),
        Error::Malformed("plan selected kernel")
    );
    ensure!(
        plan.model == shape.model
            && plan.tokens == shape.tokens
            && plan.hidden == shape.hidden
            && plan.intermediate == shape.intermediate
            && plan.experts == info.experts
            && plan.top_k == info.top_k,
        Error::Malformed("plan geometry")
    );
    let expected_renormalize = if info.supports_renormalize != 0 {
        shape.renormalize
    } else {
        1
    };
    ensure!(
        plan.renormalize == expected_renormalize,
        Error::Malformed("plan renormalize")
    );
    ensure!(
        plan.workspace_bytes != 0,
        Error::Malformed("plan workspace bytes")
    );
    ensure!(
        plan.flags & !(PLAN_FLAG_RENORMALIZE | PLAN_FLAG_EXPERT_SCALES) == 0,
        Error::Malformed("plan flags")
    );
    ensure!(
        (plan.flags & PLAN_FLAG_RENORMALIZE != 0) == (plan.renormalize == 1),
        Error::Malformed("plan renormalize flag")
    );
    ensure!(
        (plan.flags & PLAN_FLAG_EXPERT_SCALES != 0) == (info.requires_expert_scales == 1),
        Error::Malformed("plan expert scales flag")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn forward_buffers() -> Buffers {
        Buffers {
            struct_size: size_of::<Buffers>() as u32,
            abi_version: ABI_VERSION,
            logits: 0x1000usize as *const f32,
            input_bf16: 0x1100usize as *const core::ffi::c_void,
            gate_up_weight_bf16: 0x1200usize as *const core::ffi::c_void,
            down_weight_bf16: 0x1300usize as *const core::ffi::c_void,
            expert_scales: std::ptr::null(),
            output_bf16: 0x1400usize as *mut core::ffi::c_void,
            workspace: 0x2000usize as *mut core::ffi::c_void,
            workspace_bytes: 1024,
            stream: 0x3000usize as *mut core::ffi::c_void,
            flags: 0,
            reserved: [0; 3],
        }
    }

    #[test]
    fn forward_buffer_guard_rejects_header_pointer_and_workspace_damage() {
        let good = forward_buffers();
        assert!(validate_forward_buffers(&good, 512).is_ok());
        assert!(validate_forward_buffers(&good, 0).is_err());
        assert!(matches!(
            validate_forward_buffers(&good, 2048),
            Err(Error::WorkspaceTooSmall)
        ));
        let mutators: [fn(&mut Buffers); 9] = [
            |b| b.struct_size = 40,
            |b| b.abi_version = 2,
            |b| b.reserved[1] = 1,
            |b| b.flags = 1,
            |b| b.logits = std::ptr::null(),
            |b| b.output_bf16 = std::ptr::null_mut(),
            |b| b.workspace = std::ptr::null_mut(),
            |b| b.stream = std::ptr::null_mut(),
            |b| b.workspace = 0x2001usize as *mut core::ffi::c_void,
        ];
        for mutate in mutators {
            let mut damaged = forward_buffers();
            mutate(&mut damaged);
            assert!(validate_forward_buffers(&damaged, 512).is_err());
        }
    }

    fn build() -> BuildInfo {
        BuildInfo {
            arch: ARCH_SM90,
            default_kernel: KERNEL_HOPPER,
            kernel_mask: KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER,
            model_mask: MODEL_MASK_QWEN3 | MODEL_MASK_QWEN35 | MODEL_MASK_GEMMA4,
            flags: BUILD_FLAG_PORTABLE | BUILD_FLAG_HOPPER,
            ..BuildInfo::new()
        }
    }

    fn qwen_info() -> ModelInfo {
        ModelInfo {
            model: MODEL_QWEN3_E128_TOP8,
            experts: 128,
            top_k: 8,
            requires_expert_scales: 0,
            supports_renormalize: 1,
            kernel_mask: KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER,
            ..ModelInfo::new()
        }
    }

    fn qwen_plan(shape: &Shape) -> Plan {
        Plan {
            arch: ARCH_SM90,
            kernel: KERNEL_HOPPER,
            model: MODEL_QWEN3_E128_TOP8,
            tokens: shape.tokens,
            hidden: shape.hidden,
            intermediate: shape.intermediate,
            experts: 128,
            top_k: 8,
            renormalize: shape.renormalize,
            workspace_bytes: 4096,
            flags: PLAN_FLAG_RENORMALIZE,
            ..Plan::new()
        }
    }

    #[test]
    fn build_and_model_validation_reject_damaged_success_fields() {
        let good = build();
        assert!(validate_build(&good, ARCH_SM90).is_ok());
        let mut bad = good;
        bad.reserved[2] = 1;
        assert!(matches!(
            validate_build(&bad, ARCH_SM90),
            Err(Error::Malformed(_))
        ));
        let mut bad = good;
        bad.arch = ARCH_SM100;
        assert!(matches!(
            validate_build(&bad, ARCH_SM90),
            Err(Error::ArchMismatch { .. })
        ));
        let mut bad = good;
        bad.kernel_mask = 0;
        assert!(matches!(
            validate_build(&bad, ARCH_SM90),
            Err(Error::Malformed(_))
        ));
        assert!(matches!(
            validate_build(&good, 70),
            Err(Error::ArchMismatch { .. })
        ));

        let info = qwen_info();
        assert!(validate_model_info(&info, MODEL_QWEN3_E128_TOP8, good.kernel_mask).is_ok());
        let mutators: [fn(&mut ModelInfo); 4] = [
            |m| m.reserved[0] = 9,
            |m| m.model = MODEL_QWEN35_E256_TOP8,
            |m| m.top_k = 129,
            |m| m.supports_renormalize = 2,
        ];
        for mutate in mutators {
            let mut bad = info;
            mutate(&mut bad);
            assert!(validate_model_info(&bad, MODEL_QWEN3_E128_TOP8, good.kernel_mask).is_err());
        }
    }

    #[test]
    fn plan_validation_rejects_bogus_success_geometry_and_flags() {
        let b = build();
        let info = qwen_info();
        let shape = Shape::new(MODEL_QWEN3_E128_TOP8, KERNEL_AUTO, 1, 128, 64, 128, 8, 1);
        let good = qwen_plan(&shape);
        assert!(validate_plan(&good, &shape, &info, &b).is_ok());
        let mutators: [fn(&mut Plan); 5] = [
            |p| p.workspace_bytes = 0,
            |p| p.reserved[1] = 1,
            |p| p.flags = 0,
            |p| p.tokens = 2,
            |p| p.arch = ARCH_SM100,
        ];
        for mutate in mutators {
            let mut bad = good;
            mutate(&mut bad);
            assert!(validate_plan(&bad, &shape, &info, &b).is_err());
        }
    }

    #[test]
    fn shape_header_and_reserved_are_rejected() {
        let mut shape = Shape::new(MODEL_QWEN3_E128_TOP8, KERNEL_AUTO, 1, 128, 64, 128, 8, 0);
        shape.abi_version = 2;
        assert!(check_header(
            shape.struct_size,
            shape.abi_version,
            size_of::<Shape>(),
            "shape"
        )
        .is_err());
        shape.abi_version = ABI_VERSION;
        shape.struct_size = 40;
        assert!(check_header(
            shape.struct_size,
            shape.abi_version,
            size_of::<Shape>(),
            "shape"
        )
        .is_err());
        shape.reserved[0] = 1;
        assert!(check_reserved(&shape.reserved, "shape reserved").is_err());
    }

    #[test]
    fn statuses_map_to_typed_errors() {
        let model = status_error("p", STATUS_UNSUPPORTED_MODEL, 1, 0);
        assert!(matches!(model, Error::UnsupportedModel(1)));
        let kernel = status_error("p", STATUS_UNSUPPORTED_KERNEL, 1, 2);
        assert!(matches!(kernel, Error::UnsupportedKernel(2)));
        assert!(matches!(
            status_error("p", STATUS_OVERFLOW, 1, 0),
            Error::Overflow
        ));
    }

    fn actual_library() -> PathBuf {
        if let Some(path) = std::env::var_os("OXIDE_MOE_LIBRARY") {
            return PathBuf::from(path);
        }
        PathBuf::from(
            "/tmp/infinity-oxide-parity-20261101/moe-standalone-root-build-v1/liboxide_moe_sm90.so",
        )
    }

    /// Run with `CUDA_VISIBLE_DEVICES` empty. This never calls forward.
    #[test]
    #[ignore = "loads the externally built native ABI library; CPU-only queries, never forward"]
    fn load_actual_sm90_query_and_prepare() {
        assert!(std::env::var_os("CUDA_VISIBLE_DEVICES").is_none_or(|v| v.is_empty()));
        let path = actual_library();
        assert!(
            path.is_file(),
            "actual ABI1 library is a required test input"
        );
        // SAFETY: this test deliberately loads the trusted build artifact named
        // above. No forward call and no CUDA context/stream is created.
        let api = unsafe { Api::load(&path, ARCH_SM90) }.expect("load trusted ABI1 library");
        assert_eq!(api.abi_version(), ABI_VERSION);
        assert_eq!(api.build_info().arch, ARCH_SM90);
        let info = api
            .query_model(MODEL_QWEN3_E128_TOP8)
            .expect("Qwen3 model info");
        assert_eq!(
            (info.experts, info.top_k, info.supports_renormalize),
            (128, 8, 1)
        );
        let gemma = api
            .query_model(MODEL_GEMMA4_E128_TOP8)
            .expect("Gemma4 model info");
        assert_eq!(
            (
                gemma.hidden,
                gemma.intermediate,
                gemma.requires_expert_scales
            ),
            (2816, 704, 1)
        );

        let mut last = None;
        for renormalize in [0, 1] {
            let shape = Shape::new(
                MODEL_QWEN3_E128_TOP8,
                KERNEL_AUTO,
                1,
                128,
                64,
                128,
                8,
                renormalize,
            );
            let workspace = api.query_workspace(&shape).expect("CPU workspace query");
            assert!(workspace.workspace_bytes > 0);
            let plan = api.prepare(&shape).expect("CPU prepare");
            assert_eq!(plan.workspace_bytes(), workspace.workspace_bytes);
            assert!(matches!(
                plan.selected_kernel(),
                KERNEL_PORTABLE | KERNEL_HOPPER
            ));
            let g = plan.geometry();
            assert_eq!((g.tokens, g.hidden, g.intermediate), (1, 128, 64));
            assert_eq!((g.experts, g.top_k, g.renormalize), (128, 8, renormalize));
            last = Some(plan);
        }
        drop(api);
        assert!(last.expect("prepared plan").workspace_bytes() > 0);
    }

    #[test]
    #[ignore = "requires the three actual compiled libraries; CPU-only queries, never forward"]
    fn actual_all_three_arch_policies_and_plan_library_lifetime() {
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
        let root = std::env::var_os("OXIDE_MOE_LIBRARY_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/tmp/infinity-oxide-parity-20261101/moe-standalone-root-build-v1")
            });
        for arch in [ARCH_SM90, ARCH_SM100, ARCH_SM120] {
            let path = root.join(format!("liboxide_moe_sm{arch}.so"));
            assert!(path.is_file(), "actual architecture library required");
            // SAFETY: exact trusted Root-built ABI1 source and ELF artifact;
            // only CPU metadata/prepare functions are called in this test.
            let api = unsafe { Api::load(path, arch) }.expect("actual ABI library");
            let weak = Arc::downgrade(&api);
            let shape = Shape::new(
                MODEL_QWEN3_E128_TOP8,
                KERNEL_PORTABLE,
                7,
                128,
                64,
                128,
                8,
                0,
            );
            let plan = api.prepare(&shape).expect("actual portable CPU plan");
            assert_eq!(plan.selected_kernel(), KERNEL_PORTABLE);
            let automatic = Shape {
                kernel: KERNEL_AUTO,
                ..shape
            };
            assert_eq!(
                api.prepare(&automatic)
                    .expect("actual default plan")
                    .selected_kernel(),
                if arch == ARCH_SM90 {
                    KERNEL_HOPPER
                } else {
                    KERNEL_PORTABLE
                }
            );
            if arch != ARCH_SM90 {
                assert!(matches!(
                    api.prepare(&Shape {
                        kernel: KERNEL_HOPPER,
                        ..shape
                    }),
                    Err(Error::UnsupportedKernel(KERNEL_HOPPER))
                ));
            }
            drop(api);
            assert!(
                weak.upgrade().is_some(),
                "prepared plan must retain loaded library"
            );
            drop(plan);
            assert!(
                weak.upgrade().is_none(),
                "no extra unbounded owner retention"
            );
        }
    }
}
