//! Varlen Gated DeltaNet prefill. State is local to each sequence and invocation.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/qwen35_gdn_ptx.rs"));
}

#[derive(Clone)]
pub(crate) struct GatedDelta {
    pub conv: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm: Tensor,
    pub key_heads: usize,
    pub value_heads: usize,
    pub epsilon: f32,
}
impl GatedDelta {
    pub fn forward(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        ab: &Tensor,
        lengths: &[u32],
    ) -> Result<Tensor> {
        let (tokens, channels) = qkv.dims2()?;
        let (kh, vh) = (self.key_heads, self.value_heads);
        if kh == 0
            || vh == 0
            || vh > 128
            || !vh.is_multiple_of(kh)
            || tokens == 0
            || tokens > i32::MAX as usize
            || tokens
                .checked_mul(channels)
                .is_none_or(|n| n > u32::MAX as usize)
            || lengths.len() < 2
            || lengths[0] != 0
            || lengths.last() != Some(&(tokens as u32))
            || lengths.windows(2).any(|w| w[0] >= w[1])
            || channels != (2 * kh + vh) * 128
            || z.dims() != [tokens, vh * 128]
            || ab.dims() != [tokens, 2 * vh]
            || self.conv.dim(0)? != channels
            || self.conv.dim(1)? == 0
            || self.conv.dim(1)? > 16
            || self.a_log.dims() != [vh]
            || self.dt_bias.dims() != [vh]
            || self.norm.dims() != [128]
        {
            candle::bail!("Unsupported Qwen3.5 Gated DeltaNet shape or sequence boundaries");
        }
        for t in [qkv, z, ab, &self.conv, &self.norm] {
            if t.dtype() != DType::BF16
                || !t.is_contiguous()
                || !t.device().same_device(qkv.device())
            {
                candle::bail!("Gated DeltaNet requires contiguous BF16 tensors on one CUDA device");
            }
        }
        for t in [&self.a_log, &self.dt_bias] {
            if t.dtype() != DType::F32
                || !t.is_contiguous()
                || !t.device().same_device(qkv.device())
            {
                candle::bail!("Gated DeltaNet decay parameters require contiguous FP32 tensors on the input device");
            }
        }
        let cu = Tensor::new(lengths, qkv.device())?;
        qkv.apply_op3_no_bwd(
            z,
            ab,
            &Forward {
                params: self.clone(),
                cu,
            },
        )
    }
}
struct Forward {
    params: GatedDelta,
    cu: Tensor,
}
impl CustomOp3 for Forward {
    fn name(&self) -> &'static str {
        "qwen35-gated-delta"
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
        candle::bail!("Qwen3.5 Gated DeltaNet requires CUDA")
    }
    fn cuda_fwd(
        &self,
        qkv: &CudaStorage,
        ql: &Layout,
        z: &CudaStorage,
        zl: &Layout,
        ab: &CudaStorage,
        al: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = qkv.device();
        let (tokens, channels) = ql.shape().dims2()?;
        let (kh, vh) = (self.params.key_heads as i32, self.params.value_heads as i32);
        let sequences = (self.cu.elem_count() - 1) as i32;
        let kernel = self.params.conv.dim(1)? as i32;
        let qkv = qkv.as_cuda_slice::<half::bf16>()?;
        let qkv = qkv.slice(ql.start_offset()..ql.start_offset() + ql.shape().elem_count());
        let z = z.as_cuda_slice::<half::bf16>()?;
        let z = z.slice(zl.start_offset()..zl.start_offset() + zl.shape().elem_count());
        let ab = ab.as_cuda_slice::<half::bf16>()?;
        let ab = ab.slice(al.start_offset()..al.start_offset() + al.shape().elem_count());
        macro_rules! cuda_view {
            ($name:ident,$storage:ident,$tensor:expr,$ty:ty) => {
                let ($storage, layout) = $tensor.storage_and_layout();
                let Storage::Cuda(storage) = &*$storage else {
                    candle::bail!("Gated DeltaNet parameter is not CUDA")
                };
                let $name = storage.as_cuda_slice::<$ty>()?;
                let $name = $name.slice(
                    layout.start_offset()..layout.start_offset() + layout.shape().elem_count(),
                );
            };
        }
        cuda_view!(conv, conv_storage, self.params.conv, half::bf16);
        cuda_view!(norm, norm_storage, self.params.norm, half::bf16);
        cuda_view!(alog, alog_storage, self.params.a_log, f32);
        cuda_view!(dt, dt_storage, self.params.dt_bias, f32);
        cuda_view!(cu, cu_storage, self.cu, u32);
        let mut mixed = unsafe { device.alloc::<half::bf16>(tokens * channels)? };
        let mut qk = unsafe { device.alloc::<f32>(tokens * kh as usize * 256)? };
        let mut gates = unsafe { device.alloc::<f32>(tokens * vh as usize * 2)? };
        let mut recurrent = unsafe { device.alloc::<half::bf16>(tokens * vh as usize * 128)? };
        let mut output = unsafe { device.alloc::<half::bf16>(tokens * vh as usize * 128)? };
        let f =
            device.get_or_load_custom_func("qwen35_conv_bf16", "qwen35-gdn", ptx::QWEN35_GDN)?;
        unsafe {
            f.builder()
                .arg(&qkv)
                .arg(&conv)
                .arg(&cu)
                .arg(&mut mixed)
                .arg(&(tokens as i32))
                .arg(&(channels as i32))
                .arg(&kernel)
                .arg(&sequences)
                .launch(LaunchConfig::for_num_elems((tokens * channels) as u32))
        }
        .map_err(candle::Error::wrap)?;
        let f =
            device.get_or_load_custom_func("qwen35_gdn_prepare", "qwen35-gdn", ptx::QWEN35_GDN)?;
        unsafe {
            f.builder()
                .arg(&mixed)
                .arg(&ab)
                .arg(&alog)
                .arg(&dt)
                .arg(&mut qk)
                .arg(&mut gates)
                .arg(&(tokens as i32))
                .arg(&kh)
                .arg(&vh)
                .launch(LaunchConfig {
                    grid_dim: ((tokens * kh as usize) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(candle::Error::wrap)?;
        let f = device.get_or_load_custom_func(
            "qwen35_gdn_recurrent",
            "qwen35-gdn",
            ptx::QWEN35_GDN,
        )?;
        unsafe {
            f.builder()
                .arg(&mixed)
                .arg(&qk)
                .arg(&gates)
                .arg(&cu)
                .arg(&mut recurrent)
                .arg(&kh)
                .arg(&vh)
                .launch(LaunchConfig {
                    grid_dim: (sequences as u32, vh as u32, 16),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(candle::Error::wrap)?;
        let f = device.get_or_load_custom_func("qwen35_gdn_norm", "qwen35-gdn", ptx::QWEN35_GDN)?;
        unsafe {
            f.builder()
                .arg(&recurrent)
                .arg(&z)
                .arg(&norm)
                .arg(&mut output)
                .arg(&self.params.epsilon)
                .launch(LaunchConfig {
                    grid_dim: ((tokens * vh as usize) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((tokens, vh as usize * 128)),
        ))
    }
}
