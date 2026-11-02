// Copyright (c) 2026 Oxide MoE Authors. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//
// Original Rust source offered under the Infinity Restricted MIT License 1.0
// (see LICENSE). This is not standard MIT; Appendix A provider use is prohibited.

//! Exact C ABI1 mirror of `oxide_moe.h`. No CUDA, Torch or ATen types appear.
//! Struct fields, sizes, offsets, constant values and all six function aliases
//! follow the supplied header. The header itself remains MIT licensed.

use core::{ffi::c_void, mem::size_of};

pub const ABI_VERSION: u32 = 1;

pub type Status = i32;
pub const STATUS_OK: Status = 0;
pub const STATUS_INVALID_ARGUMENT: Status = 1;
pub const STATUS_UNSUPPORTED_ABI: Status = 2;
pub const STATUS_UNSUPPORTED_MODEL: Status = 3;
pub const STATUS_UNSUPPORTED_ARCH: Status = 4;
pub const STATUS_UNSUPPORTED_KERNEL: Status = 5;
pub const STATUS_SHAPE_MISMATCH: Status = 6;
pub const STATUS_OVERFLOW: Status = 7;
pub const STATUS_WORKSPACE_TOO_SMALL: Status = 8;
pub const STATUS_KERNEL_ERROR: Status = 9;
pub const STATUS_INTERNAL_ERROR: Status = 10;

pub type ModelKind = i32;
pub const MODEL_QWEN3_E128_TOP8: ModelKind = 1;
pub const MODEL_QWEN35_E256_TOP8: ModelKind = 2;
pub const MODEL_GEMMA4_E128_TOP8: ModelKind = 3;

pub type KernelKind = i32;
pub const KERNEL_AUTO: KernelKind = 0;
pub const KERNEL_PORTABLE: KernelKind = 1;
pub const KERNEL_HOPPER: KernelKind = 2;

pub type Arch = i32;
pub const ARCH_SM90: Arch = 90;
pub const ARCH_SM100: Arch = 100;
pub const ARCH_SM120: Arch = 120;

pub const KERNEL_MASK_PORTABLE: u32 = 1 << KERNEL_PORTABLE;
pub const KERNEL_MASK_HOPPER: u32 = 1 << KERNEL_HOPPER;
pub const MODEL_MASK_QWEN3: u32 = 1 << MODEL_QWEN3_E128_TOP8;
pub const MODEL_MASK_QWEN35: u32 = 1 << MODEL_QWEN35_E256_TOP8;
pub const MODEL_MASK_GEMMA4: u32 = 1 << MODEL_GEMMA4_E128_TOP8;

pub const BUILD_FLAG_PORTABLE: u32 = 1 << 0;
pub const BUILD_FLAG_HOPPER: u32 = 1 << 1;
pub const BUILD_FLAG_COMPILE_ONLY: u32 = 1 << 2;

pub const PLAN_FLAG_RENORMALIZE: u32 = 1 << 0;
pub const PLAN_FLAG_EXPERT_SCALES: u32 = 1 << 1;

macro_rules! with_header {
    ($($ty:ty),* $(,)?) => {$(
        impl $ty {
            pub fn new() -> Self {
                Self { struct_size: size_of::<Self>() as u32, abi_version: ABI_VERSION, ..Self::default() }
            }
        }
    )*};
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BuildInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub arch: Arch,
    pub default_kernel: KernelKind,
    pub kernel_mask: u32,
    pub model_mask: u32,
    pub flags: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub model: ModelKind,
    pub experts: i32,
    pub top_k: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub requires_expert_scales: i32,
    pub supports_renormalize: i32,
    pub kernel_mask: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Shape {
    pub struct_size: u32,
    pub abi_version: u32,
    pub model: ModelKind,
    pub kernel: KernelKind,
    pub tokens: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub experts: i32,
    pub top_k: i32,
    pub renormalize: i32,
    pub reserved: [u32; 2],
}

impl Shape {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelKind,
        kernel: KernelKind,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        experts: i32,
        top_k: i32,
        renormalize: i32,
    ) -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION,
            model,
            kernel,
            tokens,
            hidden,
            intermediate,
            experts,
            top_k,
            renormalize,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub arch: Arch,
    pub kernel: KernelKind,
    pub workspace_bytes: usize,
    pub flags: u32,
    pub reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Plan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub arch: Arch,
    pub kernel: KernelKind,
    pub model: ModelKind,
    pub tokens: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub experts: i32,
    pub top_k: i32,
    pub renormalize: i32,
    pub workspace_bytes: usize,
    pub flags: u32,
    pub reserved: [u32; 3],
}

with_header!(BuildInfo, ModelInfo, WorkspaceInfo, Plan);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub struct_size: u32,
    pub abi_version: u32,
    pub logits: *const f32,
    pub input_bf16: *const c_void,
    pub gate_up_weight_bf16: *const c_void,
    pub down_weight_bf16: *const c_void,
    pub expert_scales: *const f32,
    pub output_bf16: *mut c_void,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub stream: *mut c_void,
    pub flags: u32,
    pub reserved: [u32; 3],
}

pub type AbiVersionFn = unsafe extern "C" fn() -> u32;
pub type QueryBuildFn = unsafe extern "C" fn(*mut BuildInfo) -> Status;
pub type QueryModelFn = unsafe extern "C" fn(ModelKind, *mut ModelInfo) -> Status;
pub type QueryWorkspaceFn = unsafe extern "C" fn(*const Shape, *mut WorkspaceInfo) -> Status;
pub type PrepareFn = unsafe extern "C" fn(*const Shape, *mut Plan) -> Status;
pub type ForwardFn = unsafe extern "C" fn(*const Plan, *const Buffers, *mut i32) -> Status;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;

    #[test]
    fn abi_layout_matches_c_on_64_bit() {
        if size_of::<usize>() != 8 {
            return;
        }
        assert_eq!(size_of::<BuildInfo>(), 40);
        assert_eq!(size_of::<ModelInfo>(), 52);
        assert_eq!(size_of::<Shape>(), 48);
        assert_eq!(size_of::<WorkspaceInfo>(), 40);
        assert_eq!(size_of::<Plan>(), 72);
        assert_eq!(offset_of!(Plan, workspace_bytes), 48);
        assert_eq!(size_of::<Buffers>(), 96);
        assert_eq!(offset_of!(Buffers, stream), 72);
    }

    #[test]
    fn constructor_headers_are_canonical() {
        let shape = Shape::new(MODEL_QWEN3_E128_TOP8, KERNEL_AUTO, 1, 128, 64, 128, 8, 0);
        assert_eq!((shape.struct_size, shape.abi_version), (48, 1));
        assert_eq!((Plan::new().struct_size, Plan::new().abi_version), (72, 1));
        assert_eq!(
            (BuildInfo::new().struct_size, BuildInfo::new().abi_version),
            (40, 1)
        );
        assert_eq!(
            (ModelInfo::new().struct_size, ModelInfo::new().abi_version),
            (52, 1)
        );
        assert_eq!(
            (
                WorkspaceInfo::new().struct_size,
                WorkspaceInfo::new().abi_version
            ),
            (40, 1)
        );
    }

    #[test]
    fn constant_values_match_header() {
        assert_eq!((ABI_VERSION, STATUS_INTERNAL_ERROR), (1, 10));
        assert_eq!((MODEL_QWEN3_E128_TOP8, MODEL_GEMMA4_E128_TOP8), (1, 3));
        assert_eq!((KERNEL_AUTO, KERNEL_HOPPER), (0, 2));
        assert_eq!((ARCH_SM90, ARCH_SM120), (90, 120));
        assert_eq!(KERNEL_MASK_HOPPER, 4);
        assert_eq!(MODEL_MASK_GEMMA4, 8);
        assert_eq!((PLAN_FLAG_RENORMALIZE, PLAN_FLAG_EXPERT_SCALES), (1, 2));
    }
}
