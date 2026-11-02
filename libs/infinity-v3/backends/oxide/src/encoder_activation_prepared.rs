// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Optional cold resolution and safe-view binding for the four pointwise
//! encoder-activation exports.
//!
//! The safe API accepts existing `TensorRef`/`TensorMut` views, so capacity,
//! address span, layout and dtype were already checked by `crate::tensor`.
//! `BoundActivation` has private fields and can only be constructed by
//! `ActivationPlan::bind`, which also requires one device context. The actual
//! enqueue is `unsafe`: the caller must keep every bound view, the prepared
//! module/function owner, and the stream's completion alive until the stream
//! drains. No raw pointer or unchecked address can enter the safe API.

use crate::encoder_activation::Activation;
use crate::encoder_activation_plan::{ActivationPlan, PlanError};
use crate::tensor::{Bf16, TensorMut, TensorRef};
use cuda_core::{launch_kernel, Context, Function, Module, Stream};
use infinity_v3_backend_core::BackendError;
use std::{ffi::c_void, sync::Arc};

/// The four additive AOT exports. The original 86-entry list is unchanged.
pub const EXPORTS: [&str; 4] = [
    "infinity_encoder_activation_gelu_tanh_bf16",
    "infinity_encoder_activation_relu_bf16",
    "infinity_encoder_activation_silu_bf16",
    "infinity_encoder_activation_tanh_bf16",
];

/// Safe-view binding for one prepared plan. Fields are deliberately private:
/// callers cannot manufacture unchecked or stale pointers.
pub struct BoundActivation<'a> {
    plan: ActivationPlan,
    input: TensorRef<'a, Bf16, 2>,
    output: TensorMut<'a, Bf16, 2>,
    bias: Option<TensorRef<'a, Bf16, 1>>,
}

impl ActivationPlan {
    /// Bind already-validated tensor views to this plan. The views keep their
    /// `DeviceBuffer`s alive for `'a`; the plan checks exact layouts and that
    /// input, output and optional bias share one context.
    pub fn bind<'a>(
        &self,
        input: TensorRef<'a, Bf16, 2>,
        output: TensorMut<'a, Bf16, 2>,
        bias: Option<TensorRef<'a, Bf16, 1>>,
    ) -> Result<BoundActivation<'a>, PlanError> {
        let same_context = Arc::ptr_eq(input.context(), output.context())
            && bias
                .as_ref()
                .is_none_or(|bias| Arc::ptr_eq(input.context(), bias.context()));
        self.validate_bound_layouts(
            input.layout(),
            output.layout(),
            bias.as_ref().map(TensorRef::layout),
            same_context,
        )?;
        Ok(BoundActivation {
            plan: *self,
            input,
            output,
            bias,
        })
    }
}

impl BoundActivation<'_> {
    #[inline]
    fn pointers(&self) -> (u64, u64, u64) {
        (
            self.input.device_ptr(),
            self.bias.as_ref().map_or(0, TensorRef::device_ptr),
            self.output.device_ptr(),
        )
    }
}

/// Cold-resolved functions. Each `Function` retains its owning `Arc<Module>`,
/// and the explicit module clone keeps the context owner alive independently of
/// the individual function handles. Construction is the only driver-touching
/// step; ordinary model startup never calls it.
pub struct PreparedEncoderActivations {
    module: Arc<Module>,
    context: Arc<Context>,
    gelu_tanh: Function,
    relu: Function,
    silu: Function,
    tanh: Function,
}

impl PreparedEncoderActivations {
    /// Resolve the four exports from `module`, which must have been loaded on
    /// `context`. The explicit context owner lets `enqueue` reject a stream or
    /// bound buffers that belong to a different context.
    pub fn from_module(module: &Arc<Module>, context: &Arc<Context>) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(module.context(), context) {
            return Err(BackendError::Start(
                "encoder activation module context differs from supplied context".into(),
            ));
        }
        let load = |name: &'static str| {
            module.load_function(name).map_err(|error| {
                BackendError::Start(format!("encoder activation export {name}: {error}"))
            })
        };
        Ok(Self {
            gelu_tanh: load(EXPORTS[0])?,
            relu: load(EXPORTS[1])?,
            silu: load(EXPORTS[2])?,
            tanh: load(EXPORTS[3])?,
            module: Arc::clone(module),
            context: Arc::clone(context),
        })
    }

    /// First required export absent from an old bundle's names.
    pub fn missing_export<'a>(
        available: impl IntoIterator<Item = &'a str>,
    ) -> Option<&'static str> {
        let available: std::collections::BTreeSet<&str> = available.into_iter().collect();
        EXPORTS.into_iter().find(|name| !available.contains(name))
    }

    #[inline]
    fn function(&self, activation: Activation) -> &Function {
        match activation {
            Activation::GeluTanh => &self.gelu_tanh,
            Activation::Relu => &self.relu,
            Activation::Silu => &self.silu,
            Activation::Tanh => &self.tanh,
        }
    }

    /// Warm enqueue with a stack-only argument array. This method does not
    /// allocate, query, change the context, or drain.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * the bound views and their `DeviceBuffer` owners stay alive and must not
    ///   be mutated (input/bias) or aliased (output) until the stream completes;
    /// * `stream` and the prepared module belong to the same context as the
    ///   bound views (the explicit `from_module` context is checked here);
    /// * this prepared owner and the module/function remain alive until the
    ///   stream completes;
    /// * the bound plan is the plan used to construct the views.
    pub unsafe fn enqueue(
        &self,
        bound: &BoundActivation<'_>,
        stream: &Stream,
    ) -> Result<(), BackendError> {
        if !Arc::ptr_eq(bound.input.context(), stream.context())
            || !Arc::ptr_eq(bound.input.context(), &self.context)
        {
            return Err(BackendError::Start(
                "encoder activation module/stream context differs from bound buffers".into(),
            ));
        }
        if bound.plan.elements() == 0 {
            return Ok(());
        }
        let (input, bias, output) = bound.pointers();
        let mut input = input;
        let mut bias = bias;
        let mut output = output;
        let mut rows = u32::try_from(bound.plan.rows())
            .map_err(|_| BackendError::Start("encoder activation rows overflow".into()))?;
        let mut width = u32::try_from(bound.plan.width())
            .map_err(|_| BackendError::Start("encoder activation width overflow".into()))?;
        let mut input_stride = u32::try_from(bound.plan.input_layout().strides()[0])
            .map_err(|_| BackendError::Start("encoder activation input stride overflow".into()))?;
        let mut output_stride = u32::try_from(bound.plan.output_layout().strides()[0])
            .map_err(|_| BackendError::Start("encoder activation output stride overflow".into()))?;
        let mut args = [
            arg(&mut input),
            arg(&mut bias),
            arg(&mut output),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut input_stride),
            arg(&mut output_stride),
        ];
        let [gx, gy, gz] = bound.plan.grid();
        let [bx, by, bz] = bound.plan.block();
        // SAFETY: the caller accepted the retention/context contract above and
        // the fixed ABI matches the four device exports.
        unsafe {
            launch_kernel(
                self.function(bound.plan.activation()).cu_function(),
                (gx, gy, gz),
                (bx, by, bz),
                0,
                stream.cu_stream(),
                &mut args,
            )
        }
        .map_err(|error| BackendError::Start(format!("encoder activation launch: {error}")))
    }

    /// The module handle retained by this preparation. Exposed for owner audits.
    pub fn module(&self) -> &Arc<Module> {
        &self.module
    }
}

#[inline]
fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_are_unique_and_additive() {
        let unique: std::collections::BTreeSet<_> = EXPORTS.into_iter().collect();
        assert_eq!(unique.len(), EXPORTS.len());
        assert!(EXPORTS.iter().all(|name| name.starts_with("infinity_")));
        assert_eq!(PreparedEncoderActivations::missing_export(EXPORTS), None);
        assert_eq!(
            PreparedEncoderActivations::missing_export(EXPORTS[..3].iter().copied()),
            Some(EXPORTS[3])
        );
    }
}
