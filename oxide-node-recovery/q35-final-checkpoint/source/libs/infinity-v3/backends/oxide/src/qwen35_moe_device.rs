// Cold device FFN bindings for the shared Qwen3.5 backbone.
//
// Dense and routed layers use the same typed enum, so the backbone never needs
// a fake dense MLP config to host a MoE layer. Uploads are cold (once per
// model) through the existing bounded WeightUploader; no per-request or
// per-expert allocation, host readback, sort, context bind, or synchronization
// is introduced here.

use crate::model::{start, WeightSource};
use crate::qwen35_backbone::Qwen35BackboneFfn;
use crate::qwen35_moe_weights::GateUpSources;
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use cuda_core::DeviceBuffer;
use infinity_v3_backend_core::BackendError;

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

pub(crate) struct Qwen35MoeDeviceFfn {
    pub(crate) router: DeviceBuffer<Bf16>,
    pub(crate) gate_up: DeviceBuffer<Bf16>,
    pub(crate) down: DeviceBuffer<Bf16>,
    pub(crate) shared_gate_up: DeviceBuffer<Bf16>,
    pub(crate) shared_down: DeviceBuffer<Bf16>,
    pub(crate) shared_gate_logit: DeviceBuffer<Bf16>,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) shared_intermediate: usize,
    pub(crate) experts: usize,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum DeviceFfnPlan {
    Dense {
        gate_up: DeviceBuffer<Bf16>,
        down: DeviceBuffer<Bf16>,
        intermediate_size: usize,
    },
    Routed(Qwen35MoeDeviceFfn),
}

impl DeviceFfnPlan {
    /// Cold upload of one layer. The host manifest has already validated every
    /// shape; this function re-checks before each device allocation.
    pub(crate) fn upload(
        uploader: &mut WeightUploader,
        ffn: &Qwen35BackboneFfn,
    ) -> Result<Self, BackendError> {
        match ffn {
            Qwen35BackboneFfn::Dense {
                gate,
                up,
                down,
                intermediate_size,
            } => {
                let hidden = gate.shape()[1];
                if gate.shape() != [*intermediate_size, hidden]
                    || up.shape() != [*intermediate_size, hidden]
                    || down.shape() != [hidden, *intermediate_size]
                {
                    return Err(start("dense FFN weight shape mismatch"));
                }
                let gate_up = uploader.upload(&[gate, up])?;
                let down = uploader.upload(&[down])?;
                Ok(Self::Dense {
                    gate_up,
                    down,
                    intermediate_size: *intermediate_size,
                })
            }
            Qwen35BackboneFfn::Routed {
                routed,
                shared,
                intermediate_size,
                shared_intermediate_size,
            } => {
                let hidden = routed.hidden;
                let experts = routed.experts;
                let two_i = checked_product(*intermediate_size, 2, "gate/up width")?;
                if routed.router.shape() != [experts, hidden] {
                    return Err(start("routed router shape mismatch"));
                }
                if shared.gate.shape() != [*shared_intermediate_size, hidden]
                    || shared.up.shape() != [*shared_intermediate_size, hidden]
                    || shared.down.shape() != [hidden, *shared_intermediate_size]
                    || shared.gate_logit.shape() != [1, hidden]
                {
                    return Err(start("shared expert weight shape mismatch"));
                }

                let mut views = Vec::new();
                views
                    .try_reserve_exact(experts)
                    .map_err(|_| start("routed expert view allocation failed"))?;
                for index in 0..experts {
                    views.push(routed.expert(index)?);
                }
                let mut gate_sources = Vec::new();
                gate_sources
                    .try_reserve_exact(experts * 2)
                    .map_err(|_| start("routed gate/up source allocation failed"))?;
                let mut down_sources = Vec::new();
                down_sources
                    .try_reserve_exact(experts)
                    .map_err(|_| start("routed down source allocation failed"))?;
                let mut packed = None;
                let packed_shape = [two_i, hidden];
                let separate_shape = [*intermediate_size, hidden];
                let down_shape = [hidden, *intermediate_size];
                for view in &views {
                    match &view.gate_up {
                        GateUpSources::Packed(source) => {
                            if source.shape() != packed_shape.as_slice() || packed == Some(false) {
                                return Err(start("packed gate/up expert shape mismatch"));
                            }
                            packed = Some(true);
                            gate_sources.push(source);
                        }
                        GateUpSources::Separate { gate, up } => {
                            if gate.shape() != separate_shape.as_slice()
                                || up.shape() != separate_shape.as_slice()
                                || packed == Some(true)
                            {
                                return Err(start("separate gate/up expert shape mismatch"));
                            }
                            packed = Some(false);
                            gate_sources.push(gate);
                            gate_sources.push(up);
                        }
                    }
                    if view.down.shape() != down_shape.as_slice() {
                        return Err(start("routed down expert shape mismatch"));
                    }
                    down_sources.push(&view.down);
                }
                let expected_sources = if packed == Some(true) {
                    experts
                } else {
                    experts * 2
                };
                if gate_sources.len() != expected_sources {
                    return Err(start("routed gate/up source count mismatch"));
                }
                let router = uploader.upload(&[&routed.router])?;
                let gate_up = uploader.upload(&gate_sources)?;
                let down = uploader.upload(&down_sources)?;
                let shared_gate_up = uploader.upload(&[&shared.gate, &shared.up])?;
                let shared_down = uploader.upload(&[&shared.down])?;
                let shared_gate_logit = uploader.upload(&[&shared.gate_logit])?;
                Ok(Self::Routed(Qwen35MoeDeviceFfn {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate_logit,
                    hidden,
                    intermediate: *intermediate_size,
                    shared_intermediate: *shared_intermediate_size,
                    experts,
                }))
            }
        }
    }

    pub(crate) fn routed(&self) -> Option<&Qwen35MoeDeviceFfn> {
        match self {
            Self::Routed(ffn) => Some(ffn),
            Self::Dense { .. } => None,
        }
    }

    pub(crate) fn intermediate_size(&self) -> usize {
        match self {
            Self::Dense {
                intermediate_size, ..
            } => *intermediate_size,
            Self::Routed(ffn) => ffn.intermediate,
        }
    }
}

/// Validate a host FFN manifest without uploading. Keeps the same guards used
/// by upload so tests can cover shape rejection without a CUDA context.
pub(crate) fn validate_ffn_host(ffn: &Qwen35BackboneFfn) -> Result<(), BackendError> {
    match ffn {
        Qwen35BackboneFfn::Dense {
            gate,
            up,
            down,
            intermediate_size,
        } => {
            let hidden = gate.shape()[1];
            if gate.shape() != [*intermediate_size, hidden]
                || up.shape() != [*intermediate_size, hidden]
                || down.shape() != [hidden, *intermediate_size]
            {
                return Err(start("dense FFN weight shape mismatch"));
            }
        }
        Qwen35BackboneFfn::Routed {
            routed,
            shared,
            intermediate_size,
            shared_intermediate_size,
        } => {
            let hidden = routed.hidden;
            if routed.router.shape() != [routed.experts, hidden]
                || shared.gate.shape() != [*shared_intermediate_size, hidden]
                || shared.up.shape() != [*shared_intermediate_size, hidden]
                || shared.down.shape() != [hidden, *shared_intermediate_size]
                || shared.gate_logit.shape() != [1, hidden]
            {
                return Err(start("routed/shared FFN weight shape mismatch"));
            }
            for index in 0..routed.experts {
                let expert = routed.expert(index)?;
                if expert.down.shape() != [hidden, *intermediate_size] {
                    return Err(start("routed down expert shape mismatch"));
                }
                match expert.gate_up {
                    GateUpSources::Packed(source)
                        if source.shape()
                            == [
                                checked_product(*intermediate_size, 2, "gate/up width")?,
                                hidden,
                            ] => {}
                    GateUpSources::Separate { gate, up }
                        if gate.shape() == [*intermediate_size, hidden]
                            && up.shape() == [*intermediate_size, hidden] => {}
                    _ => return Err(start("routed gate/up expert shape mismatch")),
                }
            }
        }
    }
    Ok(())
}

/// Shape-only helper used by the shared backbone before any upload.
pub(crate) fn source_elements(source: &WeightSource) -> Result<usize, BackendError> {
    source
        .shape()
        .iter()
        .try_fold(1usize, |value, dim| value.checked_mul(*dim))
        .ok_or_else(|| start("weight element count overflow"))
}
