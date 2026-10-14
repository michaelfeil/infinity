//! CUDA expert execution for Gemma4-26B-A4B. Routing metadata stays on device.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor};
use std::ffi::c_void;

unsafe extern "C" {
    fn gemma4_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn gemma4_moe_forward_bf16(
        logits: *const f32,
        scales: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(gemma4_moe_hopper)]
unsafe extern "C" {
    fn hopper_gemma4_moe_workspace_bytes(tokens: i32, hidden: i32, intermediate: i32) -> usize;
    fn hopper_gemma4_moe_forward_bf16(
        logits: *const f32,
        scales: *const f32,
        input: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        output: *mut c_void,
        tokens: i32,
        hidden: i32,
        intermediate: i32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

pub fn experts(
    input: &Tensor,
    logits: &Tensor,
    scales: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
) -> Result<Tensor> {
    let (tokens, hidden) = input.dims2()?;
    if tokens == 0
        || tokens > i32::MAX as usize / 8
        || hidden != 2816
        || logits.dims() != [tokens, 128]
        || scales.dims() != [128]
        || gate_up.dims() != [128, 1408, 2816]
        || down.dims() != [128, 2816, 704]
    {
        candle::bail!("Gemma4 MoE requires [T,2816] input, 128 experts and eight routes per token");
    }
    for t in [input, gate_up, down] {
        if t.dtype() != DType::BF16 {
            candle::bail!("Gemma4 MoE expert tensors must be BF16");
        }
    }
    for t in [logits, scales] {
        if t.dtype() != DType::F32 {
            candle::bail!("Gemma4 MoE router tensors must be FP32");
        }
    }
    for t in [input, logits, scales, gate_up, down] {
        if !t.is_contiguous() || !t.device().same_device(input.device()) {
            candle::bail!("Gemma4 MoE tensors must be contiguous and on the same CUDA device");
        }
    }
    input.apply_op3_no_bwd(
        gate_up,
        down,
        &Experts {
            logits: logits.clone(),
            scales: scales.clone(),
        },
    )
}

struct Experts {
    logits: Tensor,
    scales: Tensor,
}
impl CustomOp3 for Experts {
    fn name(&self) -> &'static str {
        "gemma4-moe"
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
        candle::bail!("Gemma4 MoE expert execution requires CUDA")
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
        let (ss, sl) = self.scales.storage_and_layout();
        let Storage::Cuda(ls) = &*ls else {
            candle::bail!("router logits must be CUDA");
        };
        let Storage::Cuda(ss) = &*ss else {
            candle::bail!("expert scales must be CUDA");
        };
        let logits = ls.as_cuda_slice::<f32>()?;
        let logits = logits.slice(ll.start_offset()..ll.start_offset() + ll.shape().elem_count());
        let scales = ss.as_cuda_slice::<f32>()?;
        let scales = scales.slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
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
            hopper_gemma4_moe_workspace_bytes
        } else {
            gemma4_moe_workspace_bytes
        };
        #[cfg(not(gemma4_moe_hopper))]
        let workspace_fn = gemma4_moe_workspace_bytes;
        #[cfg(gemma4_moe_hopper)]
        let launch_fn = if use_hopper {
            hopper_gemma4_moe_forward_bf16
        } else {
            gemma4_moe_forward_bf16
        };
        #[cfg(not(gemma4_moe_hopper))]
        let launch_fn = gemma4_moe_forward_bf16;
        let bytes = unsafe { workspace_fn(tokens as i32, hidden as i32, 704) };
        if bytes == 0 {
            candle::bail!("Invalid Gemma4 MoE workspace shape");
        }
        // Kernels initialize all scratch values before reads and every output element.
        let mut scratch = unsafe { device.alloc::<u8>(bytes)? };
        let mut output = unsafe { device.alloc::<half::bf16>(tokens * hidden)? };
        {
            let (ip, _ig) = input.device_ptr(&stream);
            let (gp, _gg) = gate.device_ptr(&stream);
            let (dp, _dg) = down.device_ptr(&stream);
            let (lp, _lg) = logits.device_ptr(&stream);
            let (sp, _sg) = scales.device_ptr(&stream);
            let (wp, _wg) = scratch.device_ptr_mut(&stream);
            let (op, _og) = output.device_ptr_mut(&stream);
            let status = unsafe {
                launch_fn(
                    lp as *const f32,
                    sp as *const f32,
                    ip as *const c_void,
                    gp as *const c_void,
                    dp as *const c_void,
                    op as *mut c_void,
                    tokens as i32,
                    hidden as i32,
                    704,
                    wp as *mut c_void,
                    bytes,
                    stream.cu_stream() as *mut c_void,
                )
            };
            if status != 0 {
                candle::bail!("Gemma4 grouped expert execution failed: {status}");
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            il.shape().clone(),
        ))
    }
}
