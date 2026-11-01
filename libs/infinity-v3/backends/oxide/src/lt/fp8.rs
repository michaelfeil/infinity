// Copyright Infinity Core Contributors. Original Rust contributions use the repository LICENSE.
// The native PyTorch-derived scaled-mm recipe retains its BSD-3-Clause
// attribution and license in NOTICE and device/LICENSE-PyTorch-BSD3.
//! Prepared FP8 cuBLASLt plans reusing the existing public `lt` provider core.
//!
//! Public native `fp8_linear.cpp` is the semantic reference: rowwise SM90 uses
//! OUTER_VEC_32F operand scales and writes model dtype; ScalarF32Sm100Or120 uses
//! unit SCALAR_32F scales and writes F32 scratch for two ordered F32 products and
//! one model-dtype cast outside cuBLASLt. Bias stays separate after rounding.
//!
//! No loader, handle, runtime, workspace, cache, or global shape registry is
//! added. Each plan owns descriptors for one work shape and reuses the existing
//! API/context/stream/owner token/32 MiB zeroed workspace. Launch performs no
//! heuristic, metadata query, or allocation.

use super::super::infer;
use super::{
    check, create_layout, destroy_descriptors, retain_failed_resources, set_desc, set_preference,
    Algo, Api, Descriptor, HeuristicResult, LtProvider, CUBLAS_COMPUTE_32F, CUBLAS_OP_T,
    CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F, MATMUL_DESC_EPILOGUE, MATMUL_DESC_POINTER_MODE,
    MATMUL_DESC_TRANSA, MATMUL_DESC_TRANSB, NATIVE_DEFAULT_WORKSPACE_BYTES,
    PREFERENCE_MAX_WORKSPACE_BYTES, STATUS_NOT_SUPPORTED, STATUS_SUCCESS,
};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use std::{
    ffi::c_void,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const CUBLAS_OP_N: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const MATMUL_DESC_A_SCALE_POINTER: i32 = 17;
const MATMUL_DESC_B_SCALE_POINTER: i32 = 18;
const MATMUL_DESC_FAST_ACCUM: i32 = 25;
const MATMUL_DESC_A_SCALE_MODE: i32 = 31;
const MATMUL_DESC_B_SCALE_MODE: i32 = 32;
const OUTER_VEC_32F: i32 = 3;
const SCALAR_32F: i32 = 0;
const EPILOGUE_DEFAULT: i32 = 1;
const POINTER_MODE_HOST: i32 = 0;
const ROWWISE_FAST_ACCUM: [i8; 2] = [1, 0];
const SCALAR_FAST_ACCUM: [i8; 1] = [0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fp8Recipe {
    RowwiseSm90,
    ScalarF32Sm100Or120,
}

impl Fp8Recipe {
    const fn scale_mode(self) -> i32 {
        match self {
            Self::RowwiseSm90 => OUTER_VEC_32F,
            Self::ScalarF32Sm100Or120 => SCALAR_32F,
        }
    }

    fn fast_accum_attempts(self) -> &'static [i8] {
        match self {
            Self::RowwiseSm90 => &ROWWISE_FAST_ACCUM,
            Self::ScalarF32Sm100Or120 => &SCALAR_FAST_ACCUM,
        }
    }

    const fn output_type(self, model_type: i32) -> i32 {
        match self {
            Self::RowwiseSm90 => model_type,
            Self::ScalarF32Sm100Or120 => CUDA_R_32F,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp8Geometry {
    pub(crate) rows: i32,
    pub(crate) input: i32,
    pub(crate) output: i32,
    pub(crate) model_type: i32,
}

impl Fp8Geometry {
    fn validate(&self, _recipe: Fp8Recipe) -> Result<(), BackendError> {
        if self.model_type != CUDA_R_16F && self.model_type != CUDA_R_16BF {
            return Err(infer(format!(
                "FP8 model output type {} is not F16 or BF16",
                self.model_type
            )));
        }
        if self.rows < 0 {
            return Err(infer(format!(
                "FP8 token rows {} must be nonnegative",
                self.rows
            )));
        }
        if self.input <= 0 || self.input % 16 != 0 {
            return Err(infer(format!(
                "FP8 input width {} must be positive and divisible by 16",
                self.input
            )));
        }
        if self.output <= 0 || self.output % 16 != 0 {
            return Err(infer(format!(
                "FP8 output width {} must be positive and divisible by 16",
                self.output
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeuristicDisposition {
    Accept,
    Fallback,
    Fatal,
}

fn heuristic_disposition(
    status: i32,
    count: i32,
    result: &HeuristicResult,
    owned: usize,
) -> HeuristicDisposition {
    if status == STATUS_NOT_SUPPORTED {
        return HeuristicDisposition::Fallback;
    }
    if status != STATUS_SUCCESS {
        return HeuristicDisposition::Fatal;
    }
    if count == 0 {
        return HeuristicDisposition::Fallback;
    }
    if count != 1 {
        return HeuristicDisposition::Fatal;
    }
    if result.state == STATUS_NOT_SUPPORTED {
        return HeuristicDisposition::Fallback;
    }
    if result.state != STATUS_SUCCESS {
        return HeuristicDisposition::Fatal;
    }
    if result.workspace_size > owned {
        return HeuristicDisposition::Fallback;
    }
    HeuristicDisposition::Accept
}

fn same_owner(owner: &Arc<()>, expected: &Arc<()>) -> bool {
    Arc::ptr_eq(owner, expected)
}

pub(crate) struct Fp8Plan {
    api: Arc<Api>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    owner: Arc<()>,
    descriptors_safe: Arc<AtomicBool>,
    op: Descriptor,
    a: Descriptor,
    b: Descriptor,
    y: Descriptor,
    pref: Descriptor,
    algo: Algo,
    workspace: usize,
    geometry: Fp8Geometry,
    recipe: Fp8Recipe,
    usable: bool,
}

// SAFETY: moving a plan transfers exclusive host-descriptor ownership, as with
// `LtPlan`. Raw pointers keep it `!Sync`; provider use is externally serialized.
unsafe impl Send for Fp8Plan {}

impl Fp8Plan {
    pub(crate) const fn geometry(&self) -> Fp8Geometry {
        self.geometry
    }
    pub(crate) const fn recipe(&self) -> Fp8Recipe {
        self.recipe
    }
}

impl Drop for Fp8Plan {
    fn drop(&mut self) {
        let mut safe = self.descriptors_safe.load(Ordering::Acquire);
        if self.context.bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            self.descriptors_safe.store(false, Ordering::Release);
            safe = false;
        }
        if !safe {
            retain_failed_resources(&self.api, &self.context, &self.stream);
            return;
        }
        let destroyed = destroy_descriptors(
            &self.api.descriptor_destroy,
            &mut self.op,
            &mut self.a,
            &mut self.b,
            &mut self.y,
            &mut self.pref,
        );
        if !destroyed {
            self.descriptors_safe.store(false, Ordering::Release);
            retain_failed_resources(&self.api, &self.context, &self.stream);
        }
    }
}

impl LtProvider {
    pub(crate) fn fp8_vendor_version(&self) -> Result<usize, BackendError> {
        type Version = unsafe extern "C" fn() -> usize;
        // SAFETY: CUDA13.1 cublasLtGetVersion has this no-argument ABI.
        let version = unsafe {
            self.core
                .api
                ._library
                .get::<Version>(b"cublasLtGetVersion\0")
        }
        .map_err(infer)?;
        Ok(unsafe { version() })
    }

    fn allocate_fp8_plan(&self, geometry: Fp8Geometry, recipe: Fp8Recipe) -> Fp8Plan {
        Fp8Plan {
            api: self.core.api.clone(),
            context: self.core.context.clone(),
            stream: self.core.stream.clone(),
            owner: self.core.owner_token.clone(),
            descriptors_safe: self.core.descriptors_safe.clone(),
            op: ptr::null_mut(),
            a: ptr::null_mut(),
            b: ptr::null_mut(),
            y: ptr::null_mut(),
            pref: ptr::null_mut(),
            algo: Algo::default(),
            workspace: 0,
            geometry,
            recipe,
            usable: true,
        }
    }

    /// Prepare one FP8 plan for one CPU geometry. The caller must have bound this
    /// provider's context and keep both scale pointers valid through the last
    /// launch. `rows == 0` is an explicit no-op plan and runs no heuristic. A cold
    /// rowwise plan tries FAST_ACCUM=1 then FAST_ACCUM=0 only for NOT_SUPPORTED or
    /// an unsuccessful/incomplete heuristic result; other statuses are fatal.
    pub(crate) fn prepare_fp8_bound(
        &self,
        geometry: Fp8Geometry,
        recipe: Fp8Recipe,
        input_scale_ptr: u64,
        weight_scale_ptr: u64,
    ) -> Result<Fp8Plan, BackendError> {
        geometry.validate(recipe)?;
        let owned = self.core.workspace.bytes;
        if owned < NATIVE_DEFAULT_WORKSPACE_BYTES {
            return Err(infer(format!(
                "FP8 cuBLASLt plan requires a 32 MiB zeroed provider workspace, found {owned} bytes"
            )));
        }
        let mut plan = self.allocate_fp8_plan(geometry, recipe);
        if geometry.rows == 0 {
            return Ok(plan);
        }

        let mut op = ptr::null_mut();
        let status = unsafe { (plan.api.desc_create)(&mut op, CUBLAS_COMPUTE_32F, CUDA_R_32F) };
        plan.op = op;
        check(status, "FP8 matmul descriptor create")?;
        if plan.op.is_null() {
            return Err(infer("cuBLASLt returned a null FP8 matmul descriptor"));
        }
        let (pointer_mode, transa, transb) = (POINTER_MODE_HOST, CUBLAS_OP_T, CUBLAS_OP_N);
        let (epilogue, mode) = (EPILOGUE_DEFAULT, recipe.scale_mode());
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_POINTER_MODE,
            &pointer_mode,
            "FP8 pointer mode",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_TRANSA,
            &transa,
            "FP8 transpose A",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_TRANSB,
            &transb,
            "FP8 transpose B",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_EPILOGUE,
            &epilogue,
            "FP8 epilogue",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_A_SCALE_MODE,
            &mode,
            "FP8 A scale mode",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_B_SCALE_MODE,
            &mode,
            "FP8 B scale mode",
        )?;
        let (a_scale, b_scale) = (
            weight_scale_ptr as *const c_void,
            input_scale_ptr as *const c_void,
        );
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_A_SCALE_POINTER,
            &a_scale,
            "FP8 A scale pointer",
        )?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_B_SCALE_POINTER,
            &b_scale,
            "FP8 B scale pointer",
        )?;

        plan.a = create_layout(
            &plan.api,
            CUDA_R_8F_E4M3,
            geometry.input,
            geometry.output,
            geometry.input,
            "FP8 layout A",
        )?;
        plan.b = create_layout(
            &plan.api,
            CUDA_R_8F_E4M3,
            geometry.input,
            geometry.rows,
            geometry.input,
            "FP8 layout B",
        )?;
        plan.y = create_layout(
            &plan.api,
            recipe.output_type(geometry.model_type),
            geometry.output,
            geometry.rows,
            geometry.output,
            "FP8 layout C/D",
        )?;

        let mut pref = ptr::null_mut();
        let status = unsafe { (plan.api.preference_create)(&mut pref) };
        plan.pref = pref;
        check(status, "FP8 preference create")?;
        if plan.pref.is_null() {
            return Err(infer("cuBLASLt returned a null FP8 preference descriptor"));
        }
        let workspace_limit = NATIVE_DEFAULT_WORKSPACE_BYTES as u64;
        set_preference(
            &plan.api,
            plan.pref,
            PREFERENCE_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            "FP8 workspace preference",
        )?;

        let attempts = recipe.fast_accum_attempts();
        for (index, fast) in attempts.iter().copied().enumerate() {
            set_desc(
                &plan.api,
                plan.op,
                MATMUL_DESC_FAST_ACCUM,
                &fast,
                "FP8 fast accumulation",
            )?;
            let mut choice = HeuristicResult::default();
            let mut count = 0i32;
            let status = unsafe {
                (plan.api.heuristic)(
                    self.core.handle,
                    plan.op,
                    plan.a,
                    plan.b,
                    plan.y,
                    plan.y,
                    plan.pref,
                    1,
                    &mut choice,
                    &mut count,
                )
            };
            match heuristic_disposition(status, count, &choice, owned) {
                HeuristicDisposition::Accept => {
                    plan.algo = choice.algo;
                    plan.workspace = choice.workspace_size;
                    return Ok(plan);
                }
                HeuristicDisposition::Fallback if index + 1 < attempts.len() => {}
                HeuristicDisposition::Fallback => {
                    return Err(infer(format!(
                        "cuBLASLt returned no successful FP8 heuristic for {recipe:?} M={} K={} N={}",
                        geometry.rows, geometry.input, geometry.output
                    )));
                }
                HeuristicDisposition::Fatal => {
                    return Err(infer(format!(
                        "cuBLASLt FP8 heuristic status {status} for {recipe:?} M={} K={} N={}",
                        geometry.rows, geometry.input, geometry.output
                    )));
                }
            }
        }
        Err(infer("cuBLASLt FP8 heuristic produced no attempts"))
    }

    /// Submit one prepared FP8 matmul with new scale pointers.
    ///
    /// # Safety
    /// The caller must bind this provider's context on the current thread and keep
    /// it current through this call. It must keep `x_ptr` (X[M,K] canonical E4M3),
    /// `weight_ptr` (W[N,K] canonical E4M3), the scales, and `output_ptr` free of
    /// conflicting access until this provider's stream drains. `output_ptr` is
    /// model-dtype [M,N] for `RowwiseSm90`, or F32 [M,N] scratch for
    /// `ScalarF32Sm100Or120`; the parent validates tensor lengths/lifetimes. The
    /// scalar caller applies two ordered F32 products then one model-dtype cast,
    /// and either recipe adds bias separately after model rounding.
    pub(crate) unsafe fn launch_fp8_bound(
        &self,
        plan: &mut Fp8Plan,
        x_ptr: u64,
        weight_ptr: u64,
        input_scale_ptr: u64,
        weight_scale_ptr: u64,
        output_ptr: u64,
    ) -> Result<(), BackendError> {
        if !same_owner(&plan.owner, &self.core.owner_token) {
            return Err(infer("FP8 plan belongs to a different cuBLASLt provider"));
        }
        if !plan.usable {
            return Err(infer(
                "FP8 plan is unusable after a previous submission or descriptor failure",
            ));
        }
        plan.geometry.validate(plan.recipe)?;
        if plan.geometry.rows == 0 {
            return Ok(());
        }
        if plan.workspace > self.core.workspace.bytes {
            return Err(infer(format!(
                "FP8 plan requires {} workspace bytes, provider owns {}",
                plan.workspace, self.core.workspace.bytes
            )));
        }

        let a_scale = weight_scale_ptr as *const c_void;
        if let Err(error) = set_desc(
            &self.core.api,
            plan.op,
            MATMUL_DESC_A_SCALE_POINTER,
            &a_scale,
            "FP8 A scale pointer",
        ) {
            plan.usable = false;
            return Err(error);
        }
        let b_scale = input_scale_ptr as *const c_void;
        if let Err(error) = set_desc(
            &self.core.api,
            plan.op,
            MATMUL_DESC_B_SCALE_POINTER,
            &b_scale,
            "FP8 B scale pointer",
        ) {
            plan.usable = false;
            return Err(error);
        }

        let (alpha, beta) = (1.0f32, 0.0f32);
        let workspace = self.core.workspace.pointer as *mut c_void;
        let stream = self.core.stream.cu_stream().cast();
        // SAFETY: the caller's contract covers every raw pointer; descriptors and
        // workspace belong to this provider and use its single ordered stream.
        let status = unsafe {
            (self.core.api.matmul)(
                self.core.handle,
                plan.op,
                &alpha,
                weight_ptr as *const c_void,
                plan.a,
                x_ptr as *const c_void,
                plan.b,
                &beta,
                output_ptr as *const c_void,
                plan.y,
                output_ptr as *mut c_void,
                plan.y,
                &plan.algo,
                workspace,
                self.core.workspace.bytes,
                stream,
            )
        };
        if status != STATUS_SUCCESS {
            plan.usable = false;
            return Err(infer(format!("FP8 cuBLASLt matmul status {status}")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heuristic_fallback_is_limited_to_eligibility_not_vendor_faults() {
        let mut h = HeuristicResult::default();
        assert!(matches!(
            heuristic_disposition(0, 0, &h, 32),
            HeuristicDisposition::Fallback
        ));
        assert!(matches!(
            heuristic_disposition(STATUS_NOT_SUPPORTED, 0, &h, 32),
            HeuristicDisposition::Fallback
        ));
        for (status, count, state) in [(1, 0, 0), (0, -1, 0), (0, 2, 0), (0, 1, 3)] {
            h.state = state;
            assert!(matches!(
                heuristic_disposition(status, count, &h, 32),
                HeuristicDisposition::Fatal
            ));
        }
        h.state = 0;
        h.workspace_size = 32;
        assert!(matches!(
            heuristic_disposition(0, 1, &h, 32),
            HeuristicDisposition::Accept
        ));
        h.workspace_size = 33;
        assert!(matches!(
            heuristic_disposition(0, 1, &h, 32),
            HeuristicDisposition::Fallback
        ));
    }
    #[test]
    fn retained_provider_identity_rejects_another_handle_even_for_identical_geometry() {
        let first = Arc::new(());
        let retained = first.clone();
        let other = Arc::new(());
        assert!(same_owner(&first, &retained));
        assert!(!same_owner(&first, &other));
    }
}
