// Native ABI1 provider selection for the actual Qwen3.5-MoE routed FFN.
//
// The existing private Qwen3 MoE framework already owns the validated ABI1
// loader (`moe_api::Api`) and the C mirror (`moe_abi`). This module only adds
// Qwen3.5 geometry, architecture policy, and a typed prepared-plan wrapper.
// It performs CPU metadata queries only; it never creates a CUDA context,
// binds a stream, allocates device storage, or calls forward.

use crate::math::LinearPlan;
use crate::moe_abi::{
    Arch, BuildInfo, ModelInfo, Shape, ARCH_SM100, ARCH_SM120, ARCH_SM90, BUILD_FLAG_COMPILE_ONLY,
    BUILD_FLAG_HOPPER, KERNEL_HOPPER, KERNEL_MASK_HOPPER, KERNEL_MASK_PORTABLE, KERNEL_PORTABLE,
    MODEL_QWEN35_E256_TOP8,
};
use crate::moe_api::{Api, Error, PreparedPlan};
use crate::tensor::Bf16;
use infinity_v3_backend_core::BackendError;
use std::path::Path;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide Qwen3.5-MoE provider: {error}"))
}

/// Explicit architecture contract difference. SM90 builds contain the Hopper
/// grouped-GEMM path (default HOPPER) plus the portable path. SM100/SM120
/// builds contain only the portable SM80-CUTLASS path and are compile-only
/// until Root qualifies them on a real GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35MoeArchContract {
    Sm90Hopper,
    PortableSm100,
    PortableSm120,
}

impl Qwen35MoeArchContract {
    pub(crate) fn from_build(arch: Arch, build: &BuildInfo) -> Result<Self, BackendError> {
        match arch {
            ARCH_SM90 => {
                if build.default_kernel != KERNEL_HOPPER
                    || build.kernel_mask != KERNEL_MASK_PORTABLE | KERNEL_MASK_HOPPER
                    || build.flags & (BUILD_FLAG_HOPPER | BUILD_FLAG_COMPILE_ONLY)
                        != BUILD_FLAG_HOPPER
                {
                    return Err(start("SM90 build policy is not Hopper plus portable"));
                }
                Ok(Self::Sm90Hopper)
            }
            ARCH_SM100 => {
                Self::validate_portable(build)?;
                Ok(Self::PortableSm100)
            }
            ARCH_SM120 => {
                Self::validate_portable(build)?;
                Ok(Self::PortableSm120)
            }
            other => Err(start(format!("unsupported MoE arch {other}"))),
        }
    }

    fn validate_portable(build: &BuildInfo) -> Result<(), BackendError> {
        if build.default_kernel != KERNEL_PORTABLE
            || build.kernel_mask != KERNEL_MASK_PORTABLE
            || build.flags & BUILD_FLAG_COMPILE_ONLY == 0
            || build.flags & BUILD_FLAG_HOPPER != 0
        {
            return Err(start(
                "SM100/SM120 build policy must be portable-only and compile-only",
            ));
        }
        Ok(())
    }

    /// Routing/combine contract name. The portable path keeps FP32 expert
    /// outputs and multiplies by FP32 weights in the combine kernel. The
    /// Hopper path fuses the routing weight into the down-GEMM epilogue and
    /// stores BF16 expert outputs, then sums those BF16 values. Both keep the
    /// route weights FP32; neither is bit-identical to the other.
    pub(crate) const fn expert_output_contract(self) -> &'static str {
        match self {
            Self::Sm90Hopper => "BF16 weighted expert output (Hopper epilogue) + BF16 sum",
            Self::PortableSm100 | Self::PortableSm120 => {
                "FP32 expert output + FP32*FP32 weighted term rounded to BF16 + FP32 sum"
            }
        }
    }

    pub(crate) const fn selected_kernel(self) -> crate::moe_abi::KernelKind {
        match self {
            Self::Sm90Hopper => KERNEL_HOPPER,
            Self::PortableSm100 | Self::PortableSm120 => KERNEL_PORTABLE,
        }
    }

    pub(crate) const fn compile_only(self) -> bool {
        !matches!(self, Self::Sm90Hopper)
    }
}

pub(crate) struct Qwen35MoeProvider {
    api: Arc<Api>,
    arch: Arch,
    model: ModelInfo,
    contract: Qwen35MoeArchContract,
}

impl Qwen35MoeProvider {
    /// Load a trusted ABI1 artifact for a known architecture. The caller
    /// resolves the artifact from the independently reviewed AOT packet.
    ///
    /// # Safety
    /// `path` must name trusted native code compiled for the matching
    /// `oxide_moe.h` ABI and architecture. Loading runs native initializers
    /// before validation; never pass untrusted files.
    pub(crate) unsafe fn load(path: &Path, arch: Arch) -> Result<Self, BackendError> {
        // SAFETY: the caller promises trusted, architecture-matched native code.
        let api = unsafe { Api::load(path, arch) }.map_err(classify)?;
        let contract = Qwen35MoeArchContract::from_build(arch, api.build_info())?;
        let model = api.query_model(MODEL_QWEN35_E256_TOP8).map_err(classify)?;
        if model.experts != 256 || model.top_k != 8 || model.supports_renormalize != 1 {
            return Err(start("ABI1 Qwen35 model metadata differs from E256/top8"));
        }
        if model.kernel_mask & (1 << contract.selected_kernel()) == 0 {
            return Err(start(
                "selected Qwen35 kernel is absent from the model mask",
            ));
        }
        Ok(Self {
            api,
            arch,
            model,
            contract,
        })
    }

    pub(crate) const fn contract(&self) -> Qwen35MoeArchContract {
        self.contract
    }

    pub(crate) const fn arch(&self) -> Arch {
        self.arch
    }

    pub(crate) fn model(&self) -> &ModelInfo {
        &self.model
    }

    pub(crate) fn prepare(
        &self,
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        renormalize: bool,
    ) -> Result<PreparedQwen35MoeRouted, BackendError> {
        let tokens = i32::try_from(tokens).map_err(|_| start("token count does not fit i32"))?;
        let hidden = i32::try_from(hidden).map_err(|_| start("hidden does not fit i32"))?;
        let intermediate =
            i32::try_from(intermediate).map_err(|_| start("intermediate does not fit i32"))?;
        let shape = Shape::new(
            MODEL_QWEN35_E256_TOP8,
            contract_kernel(self.contract),
            tokens,
            hidden,
            intermediate,
            256,
            8,
            i32::from(renormalize),
        );
        let workspace = self.api.query_workspace(&shape).map_err(classify)?;
        let plan = self.api.prepare(&shape).map_err(classify)?;
        if plan.selected_kernel() != contract_kernel(self.contract)
            || plan.workspace_bytes() != workspace.workspace_bytes
        {
            return Err(start("prepared Qwen35 plan differs from its query"));
        }
        let router =
            LinearPlan::<Bf16>::contiguous(tokens as usize, hidden as usize, 256).map_err(start)?;
        Ok(PreparedQwen35MoeRouted {
            plan,
            shape,
            contract: self.contract,
            router,
            hidden: hidden as usize,
            intermediate: intermediate as usize,
        })
    }
}

fn contract_kernel(contract: Qwen35MoeArchContract) -> crate::moe_abi::KernelKind {
    contract.selected_kernel()
}

fn classify(error: Error) -> BackendError {
    start(error)
}

/// One provider-owned prepared plan. It retains the loaded library through the
/// plan's `Arc<Api>`; no pointers are exposed and no forward is possible
/// without the crate-private enqueue guard.
pub(crate) struct PreparedQwen35MoeRouted {
    plan: PreparedPlan,
    shape: Shape,
    contract: Qwen35MoeArchContract,
    router: LinearPlan<Bf16>,
    hidden: usize,
    intermediate: usize,
}

impl PreparedQwen35MoeRouted {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.plan.workspace_bytes()
    }

    pub(crate) fn contract(&self) -> Qwen35MoeArchContract {
        self.contract
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    pub(crate) fn selected_kernel(&self) -> crate::moe_abi::KernelKind {
        self.plan.selected_kernel()
    }

    pub(crate) fn plan(&self) -> &PreparedPlan {
        &self.plan
    }

    pub(crate) fn router_plan(&self) -> &LinearPlan<Bf16> {
        &self.router
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub(crate) fn tokens(&self) -> usize {
        self.shape.tokens as usize
    }

    pub(crate) fn forward_admitted(&self) -> bool {
        !self.contract.compile_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn required_aot_root() -> Option<PathBuf> {
        let root = std::env::var_os("QWEN35_MOE_AOT_ROOT").map(PathBuf::from);
        if root.is_none() && std::env::var_os("QWEN35_MOE_REQUIRE_AOT").is_some() {
            panic!("QWEN35_MOE_AOT_ROOT required for provider metadata tests");
        }
        root
    }

    #[test]
    fn actual_aot_arch_policies_and_qwen35_workspace_are_distinct() {
        let Some(root) = required_aot_root() else {
            return;
        };
        let mut workspaces = Vec::new();
        for arch in [ARCH_SM90, ARCH_SM100, ARCH_SM120] {
            let path = root.join(format!("liboxide_moe_sm{arch}.so"));
            assert!(path.is_file(), "missing actual AOT library {path:?}");
            // SAFETY: trusted reviewed AOT artifact; CPU metadata only.
            let provider = unsafe { Qwen35MoeProvider::load(&path, arch) }.unwrap();
            assert_eq!(provider.arch(), arch);
            assert_eq!(provider.model().experts, 256);
            assert_eq!(provider.model().top_k, 8);
            let mut previous = None;
            for tokens in [1usize, 7, 128, 2048] {
                let prepared = provider
                    .prepare(tokens, 2048, 512, true)
                    .expect("CPU Qwen35 prepare");
                assert_eq!(prepared.tokens(), tokens);
                assert_eq!(
                    prepared.selected_kernel(),
                    provider.contract().selected_kernel()
                );
                assert!(prepared.workspace_bytes() > 0);
                assert_eq!(
                    prepared.contract().compile_only(),
                    arch != ARCH_SM90,
                    "SM100/120 must remain compile-only"
                );
                if let Some(previous) = previous {
                    assert!(prepared.workspace_bytes() > previous);
                }
                eprintln!(
                    "arch sm{arch}: tokens {tokens} kernel {:?} compile_only {} workspace {}",
                    prepared.selected_kernel(),
                    prepared.contract().compile_only(),
                    prepared.workspace_bytes()
                );
                previous = Some(prepared.workspace_bytes());
                if tokens == 7 {
                    let workspace =
                        crate::qwen35_moe_enqueue::RoutedWorkspacePlan::from_prepared(&prepared)
                            .unwrap();
                    crate::qwen35_moe_enqueue::validate_ffn_contract(workspace).unwrap();
                    assert_eq!(workspace.bytes(), prepared.workspace_bytes());
                    assert_eq!(
                        crate::qwen35_moe_enqueue::ensure_forward_admitted(&prepared).is_ok(),
                        prepared.forward_admitted()
                    );
                    assert_eq!(prepared.forward_admitted(), arch == ARCH_SM90);
                    workspaces.push((arch, prepared.workspace_bytes()));
                }
            }
        }
        let sm90 = workspaces
            .iter()
            .find(|(arch, _)| *arch == ARCH_SM90)
            .unwrap()
            .1;
        let sm100 = workspaces
            .iter()
            .find(|(arch, _)| *arch == ARCH_SM100)
            .unwrap()
            .1;
        assert_ne!(
            sm90, sm100,
            "Hopper and portable workspace ABIs must not be conflated"
        );
    }
}
