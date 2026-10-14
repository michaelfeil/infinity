//! CUDA expert execution for Qwen3-MoE. Routing metadata stays on device.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor};
use std::ffi::c_void;

unsafe extern "C" {
    fn qwen3_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn qwen3_moe_forward_bf16(
        logits: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        renormalize: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(gemma4_moe_hopper)]
unsafe extern "C" {
    fn hopper_qwen3_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn hopper_qwen3_moe_forward_bf16(
        logits: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        renormalize: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}
unsafe extern "C" {
    fn qwen35_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn qwen35_moe_forward_bf16(
        logits: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        renormalize: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(gemma4_moe_hopper)]
unsafe extern "C" {
    fn hopper_qwen35_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn hopper_qwen35_moe_forward_bf16(
        logits: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        renormalize: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

pub fn experts(
    input: &Tensor,
    logits: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
    renormalize: bool,
) -> Result<Tensor> {
    let (tokens, hidden) = input.dims2()?;
    let (experts, _, intermediate) = down.dims3()?;
    if tokens == 0
        || tokens > i32::MAX as usize / 8
        || hidden == 0
        || !hidden.is_multiple_of(8)
        || intermediate == 0
        || !intermediate.is_multiple_of(8)
        || hidden > i32::MAX as usize
        || intermediate > i32::MAX as usize / 2
        || !matches!(experts, 128 | 256)
        || logits.dims() != [tokens, experts]
        || gate_up.dims() != [experts, 2 * intermediate, hidden]
        || down.dims() != [experts, hidden, intermediate]
    {
        candle::bail!("Qwen3 MoE requires aligned BF16 expert dimensions, 128 or 256 experts and eight routes per token");
    }
    for t in [input, gate_up, down] {
        if t.dtype() != DType::BF16 {
            candle::bail!("Qwen3 MoE expert tensors must be BF16");
        }
    }
    if logits.dtype() != DType::F32 {
        candle::bail!("Qwen3 MoE router tensors must be FP32");
    }
    for t in [input, logits, gate_up, down] {
        if !t.is_contiguous() || !t.device().same_device(input.device()) {
            candle::bail!("Qwen3 MoE tensors must be contiguous and on the same CUDA device");
        }
    }
    input.apply_op3_no_bwd(
        gate_up,
        down,
        &Experts {
            logits: logits.clone(),
            renormalize,
            intermediate,
        },
    )
}

struct Experts {
    logits: Tensor,
    renormalize: bool,
    intermediate: usize,
}
impl CustomOp3 for Experts {
    fn name(&self) -> &'static str {
        "qwen3-moe"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Qwen3 MoE expert execution requires CUDA")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        gate: &CudaStorage,
        gl: &Layout,
        down: &CudaStorage,
        dl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, hidden) = il.shape().dims2()?;
        let device = input.device();
        let stream = device.cuda_stream();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(il.start_offset()..il.start_offset() + il.shape().elem_count());
        let gate = gate.as_cuda_slice::<half::bf16>()?;
        let gate = gate.slice(gl.start_offset()..gl.start_offset() + gl.shape().elem_count());
        let down = down.as_cuda_slice::<half::bf16>()?;
        let down = down.slice(dl.start_offset()..dl.start_offset() + dl.shape().elem_count());
        let (ls, ll) = self.logits.storage_and_layout();
        let Storage::Cuda(ls) = &*ls else {
            candle::bail!("router logits must be CUDA");
        };
        let logits = ls.as_cuda_slice::<f32>()?;
        let logits = logits.slice(ll.start_offset()..ll.start_offset() + ll.shape().elem_count());
        #[cfg(gemma4_moe_hopper)]
        let use_hopper = if tokens >= 2048 {
            use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
            };
            let context = stream.context();
            context.attribute(MAJOR).map_err(candle::Error::wrap)? == 9
                && context.attribute(MINOR).map_err(candle::Error::wrap)? == 0
        } else {
            false
        };
        #[cfg(gemma4_moe_hopper)]
        let workspace_fn = if use_hopper {
            hopper_qwen3_moe_workspace_bytes
        } else {
            qwen3_moe_workspace_bytes
        };
        #[cfg(not(gemma4_moe_hopper))]
        let workspace_fn = qwen3_moe_workspace_bytes;
        #[cfg(gemma4_moe_hopper)]
        let launch_fn = if use_hopper {
            hopper_qwen3_moe_forward_bf16
        } else {
            qwen3_moe_forward_bf16
        };
        #[cfg(not(gemma4_moe_hopper))]
        let launch_fn = qwen3_moe_forward_bf16;
        let experts = gl.dims()[0];
        let workspace_fn = if experts == 256 {
            #[cfg(gemma4_moe_hopper)]
            if use_hopper {
                hopper_qwen35_moe_workspace_bytes
            } else {
                qwen35_moe_workspace_bytes
            }
            #[cfg(not(gemma4_moe_hopper))]
            {
                qwen35_moe_workspace_bytes
            }
        } else {
            workspace_fn
        };
        let launch_fn = if experts == 256 {
            #[cfg(gemma4_moe_hopper)]
            if use_hopper {
                hopper_qwen35_moe_forward_bf16
            } else {
                qwen35_moe_forward_bf16
            }
            #[cfg(not(gemma4_moe_hopper))]
            {
                qwen35_moe_forward_bf16
            }
        } else {
            launch_fn
        };
        let bytes = unsafe { workspace_fn(tokens as i32, hidden as i32, self.intermediate as i32) };
        if bytes == 0 {
            candle::bail!("Invalid Qwen3 MoE workspace shape");
        }
        // Kernels initialize all scratch values before reads and every output element.
        let mut scratch = unsafe { device.alloc::<u8>(bytes)? };
        let mut output = unsafe { device.alloc::<half::bf16>(tokens * hidden)? };
        {
            let (ip, _ig) = input.device_ptr(&stream);
            let (gp, _gg) = gate.device_ptr(&stream);
            let (dp, _dg) = down.device_ptr(&stream);
            let (lp, _lg) = logits.device_ptr(&stream);
            let (wp, _wg) = scratch.device_ptr_mut(&stream);
            let (op, _og) = output.device_ptr_mut(&stream);
            let status = unsafe {
                launch_fn(
                    lp as *const f32,
                    ip as *const c_void,
                    gp as *const c_void,
                    dp as *const c_void,
                    op as *mut c_void,
                    tokens as i32,
                    hidden as i32,
                    self.intermediate as i32,
                    i32::from(self.renormalize),
                    wp as *mut c_void,
                    bytes,
                    stream.cu_stream() as *mut c_void,
                )
            };
            if status != 0 {
                candle::bail!("Qwen3 grouped expert execution failed: {status}");
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            il.shape().clone(),
        ))
    }
}
