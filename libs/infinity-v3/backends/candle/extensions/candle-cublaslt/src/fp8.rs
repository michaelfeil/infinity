//! Experimental row-scaled E4M3 GEMM for opt-in dynamic FP8 MLP inference.
use candle::{DType, Result, Storage, Tensor};
use cudarc::cublaslt::{result as lt, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::{bf16, f16};
use std::mem::size_of;

/// Architectures covered by the FP8 build matrix. SM90 supports cuBLASLt
/// outer-vector scales; the others use FP32 output followed by row scaling.
pub fn supported_compute_cap(cap: usize) -> bool {
    matches!(cap, 89 | 90 | 100 | 120)
}

fn err(e: lt::CublasError) -> candle::Error {
    candle::Error::Cuda(Box::new(e))
}
macro_rules! owned {
    ($name:ident, $ty:ty, $destroy:path) => {
        struct $name($ty);
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    let _ = $destroy(self.0);
                }
            }
        }
    };
}
owned!(Handle, sys::cublasLtHandle_t, lt::destroy_handle);
owned!(Desc, sys::cublasLtMatmulDesc_t, lt::destroy_matmul_desc);
owned!(
    Layout,
    sys::cublasLtMatrixLayout_t,
    lt::destroy_matrix_layout
);
owned!(
    Preference,
    sys::cublasLtMatmulPreference_t,
    lt::destroy_matmul_pref
);
impl Desc {
    // Only used with the attribute's documented scalar or device-pointer type.
    fn set<T>(&self, attr: sys::cublasLtMatmulDescAttributes_t, value: &T) -> Result<()> {
        unsafe {
            lt::set_matmul_desc_attribute(
                self.0,
                attr,
                value as *const T as *const _,
                size_of::<T>(),
            )
            .map_err(err)
        }
    }
}

struct Plan {
    desc: Desc,
    a: Layout,
    b: Layout,
    c: Layout,
    algo: sys::cublasLtMatmulHeuristicResult_t,
    #[cfg(test)]
    fast_accum: bool,
}

/// Experimental GEMM executor tied to one Candle CUDA device/stream.
/// Exclusive mutable access serializes host submission and workspace reuse.
/// No unsafe Send/Sync implementation: create/use it on its owning worker thread.
pub struct Fp8Matmul {
    handle: Handle,
    plans: std::collections::HashMap<(usize, usize, usize, DType), Plan>,
    workspace: cudarc::driver::CudaSlice<u8>,
    device: candle::Device,
    fused_row_scaling: bool,
}
impl Fp8Matmul {
    pub fn new(device: &candle::Device) -> Result<Self> {
        let dev = device.as_cuda_device()?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let stream = dev.cuda_stream();
        let major = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(candle::Error::wrap)?;
        let minor = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(candle::Error::wrap)?;
        if !supported_compute_cap(major as usize * 10 + minor as usize) {
            candle::bail!("Dynamic FP8 requires Ada SM89, Hopper SM90, or Blackwell SM100/SM120");
        }
        dev.cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)?;
        let handle = Handle(lt::create_handle().map_err(err)?);
        let workspace = unsafe { dev.alloc::<u8>(32usize << 20)? };
        Ok(Self {
            handle,
            plans: Default::default(),
            workspace,
            device: device.clone(),
            fused_row_scaling: (major, minor) == (9, 0),
        })
    }
    /// Compute FP16 `(x * sx) @ (w * sw).T` from contiguous E4M3 matrices.
    /// Scales are FP32, one per row. Plans are bounded to sixteen shapes and
    /// never retain input pointers between submissions without refreshing them.
    pub fn scaled_mm(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        sx: &Tensor,
        sw: &Tensor,
    ) -> Result<Tensor> {
        self.scaled_mm_dtype(x, w, sx, sw, DType::F16)
    }

    /// Keep FP8 GEMM output in the model's native 16-bit dtype.
    pub fn scaled_mm_dtype(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        sx: &Tensor,
        sw: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        if !matches!(dtype, DType::F16 | DType::BF16) {
            candle::bail!("FP8 GEMM output must be FP16 or BF16");
        }
        if !self.fused_row_scaling {
            // cuBLASLt OUTER_VEC_32F is SM90-only. Tensorwide FP8 GEMM
            // with unit scales works on Ada and Blackwell too. Keep the
            // unscaled accumulation in FP32 to avoid FP16 overflow, apply
            // the original row scales, and only then round to model dtype.
            return self
                .scaled_mm_typed::<f32>(x, w, sx, sw, DType::F32)?
                .broadcast_mul(&sx.unsqueeze(1)?)?
                .broadcast_mul(&sw.unsqueeze(0)?)?
                .to_dtype(dtype);
        }
        match dtype {
            DType::F16 => self.scaled_mm_typed::<f16>(x, w, sx, sw, dtype),
            DType::BF16 => self.scaled_mm_typed::<bf16>(x, w, sx, sw, dtype),
            _ => candle::bail!("FP8 GEMM output must be FP16 or BF16"),
        }
    }

    fn scaled_mm_typed<T: candle::cuda_backend::CudaDType + cudarc::driver::DeviceRepr>(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        sx: &Tensor,
        sw: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        if !self.device.same_device(x.device()) {
            candle::bail!("FP8 executor belongs to a different CUDA device")
        }
        let (m, k) = x.dims2()?;
        let (n, wk) = w.dims2()?;
        if m == 0 || n == 0 || k == 0 || wk != k || k % 16 != 0 || n % 16 != 0 {
            candle::bail!("FP8 GEMM requires nonzero matching dimensions and N/K multiples of 16")
        }
        // cuBLAS uses signed dimensions internally even though layout construction is u64.
        for d in [m, n, k] {
            i32::try_from(d).map_err(candle::Error::wrap)?;
        }
        let count = m
            .checked_mul(n)
            .ok_or_else(|| candle::Error::Msg("FP8 output size overflow".into()))?;
        for (t, dt, len) in [
            (x, DType::F8E4M3, m * k),
            (w, DType::F8E4M3, n * k),
            (sx, DType::F32, m),
            (sw, DType::F32, n),
        ] {
            if t.dtype() != dt
                || !t.is_contiguous()
                || t.elem_count() != len
                || !t.device().same_device(x.device())
            {
                candle::bail!("FP8 matrices/scales must have matching CUDA devices, contiguous layouts and expected types/shapes")
            }
            if t.storage_and_layout().1.start_offset() != 0 {
                candle::bail!("FP8 prototype requires zero-offset tensors")
            }
        }
        let dev = x.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)?;
        let (xs, _) = x.storage_and_layout();
        let (ws, _) = w.storage_and_layout();
        let (ssx, _) = sx.storage_and_layout();
        let (ssw, _) = sw.storage_and_layout();
        let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ssx), Storage::Cuda(ssw)) =
            (&*xs, &*ws, &*ssx, &*ssw)
        else {
            candle::bail!("FP8 GEMM requires CUDA")
        };
        let xa = xs.as_cuda_slice::<float8::F8E4M3>()?;
        let wa = ws.as_cuda_slice::<float8::F8E4M3>()?;
        let xscale = ssx.as_cuda_slice::<f32>()?;
        let wscale = ssw.as_cuda_slice::<f32>()?;
        let mut output = unsafe { dev.alloc::<T>(count)? };
        let workspace_bytes = 32usize << 20;
        // Keep all pointer records alive until after enqueueing the GEMM. This is
        // essential for producer-stream waits and safe asynchronous buffer reuse.
        let (xp, _rx) = xa.device_ptr(&stream);
        let (wp, _rw) = wa.device_ptr(&stream);
        let (spx, _rsx) = xscale.device_ptr(&stream);
        let (spw, _rsw) = wscale.device_ptr(&stream);
        let (op, ro) = output.device_ptr_mut(&stream);
        let (scratch, rwork) = self.workspace.device_ptr_mut(&stream);
        let key = (m, n, k, dtype);
        if !self.plans.contains_key(&key) {
            if self.plans.len() >= 16 {
                self.plans.clear();
            }
            let d = Desc(
                lt::create_matmul_desc(
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType::CUDA_R_32F,
                )
                .map_err(err)?,
            );
            use sys::cublasLtMatmulDescAttributes_t::*;
            d.set(
                CUBLASLT_MATMUL_DESC_TRANSA,
                &cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            )?;
            // Unscaled FP8 products can be large. The portable path needs
            // full accumulation before applying row scales to FP32 output.
            d.set(
                CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                &(self.fused_row_scaling as i8),
            )?;
            if self.fused_row_scaling {
                let mode =
                    sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                d.set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode)?;
                d.set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode)?;
                d.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &spw)?;
                d.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &spx)?;
            }
            // Otherwise leave scalar scale pointers unset (cuBLASLt defaults to 1).
            let a = Layout(
                lt::create_matrix_layout(
                    sys::cudaDataType::CUDA_R_8F_E4M3,
                    k as u64,
                    n as u64,
                    k as i64,
                )
                .map_err(err)?,
            );
            let b = Layout(
                lt::create_matrix_layout(
                    sys::cudaDataType::CUDA_R_8F_E4M3,
                    k as u64,
                    m as u64,
                    k as i64,
                )
                .map_err(err)?,
            );
            let c = Layout(
                lt::create_matrix_layout(
                    match dtype {
                        DType::BF16 => sys::cudaDataType::CUDA_R_16BF,
                        DType::F16 => sys::cudaDataType::CUDA_R_16F,
                        DType::F32 => sys::cudaDataType::CUDA_R_32F,
                        _ => unreachable!(),
                    },
                    n as u64,
                    m as u64,
                    n as i64,
                )
                .map_err(err)?,
            );
            let pref = Preference(lt::create_matmul_pref().map_err(err)?);
            unsafe {
                let attr = sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
                lt::set_matmul_pref_attribute(
                    pref.0,
                    attr,
                    &workspace_bytes as *const _ as *const _,
                    size_of::<usize>(),
                )
                .map_err(err)?;
            }
            let query = || unsafe {
                lt::get_matmul_algo_heuristic(self.handle.0, d.0, a.0, b.0, c.0, c.0, pref.0)
            };
            let (algo, _fast_accum) = match query() {
                Ok(algo) => (algo, self.fused_row_scaling),
                Err(error) if error.0 == sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => {
                    // CUDA 12.9 lacks fast-accumulation tactics for some short,
                    // wide-K projections. Keep the same FP8 inputs and scales,
                    // and request full accumulation instead. Other errors propagate.
                    d.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &0i8)?;
                    (query().map_err(err)?, false)
                }
                Err(error) => return Err(err(error)),
            };
            self.plans.insert(
                key,
                Plan {
                    desc: d,
                    a,
                    b,
                    c,
                    algo,
                    #[cfg(test)]
                    fast_accum: _fast_accum,
                },
            );
        }
        let plan = self.plans.get(&key).expect("plan inserted above");
        if self.fused_row_scaling {
            // Scale allocations may change even when dimensions do not.
            plan.desc.set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &spw,
            )?;
            plan.desc.set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &spx,
            )?;
        }
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            lt::matmul(
                self.handle.0,
                plan.desc.0,
                &alpha as *const _ as *const _,
                &beta as *const _ as *const _,
                wp as *const _,
                plan.a.0,
                xp as *const _,
                plan.b.0,
                op as *const _,
                plan.c.0,
                op as *mut _,
                plan.c.0,
                &plan.algo.algo,
                scratch as *mut _,
                workspace_bytes,
                stream.cu_stream() as *mut _,
            )
            .map_err(err)?;
        }
        drop(ro);
        drop(rwork);
        let storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        Ok(Tensor::from((
            Storage::Cuda(storage),
            candle::Shape::from((m, n)),
        )))
    }
}
/// Convenience entry point; inference should retain `Fp8Matmul` across calls.
pub fn scaled_mm(x: &Tensor, w: &Tensor, sx: &Tensor, sw: &Tensor) -> Result<Tensor> {
    Fp8Matmul::new(x.device())?.scaled_mm(x, w, sx, sw)
}

/// Quantize finite contiguous FP16/BF16 rows to E4M3 and FP32 row scales.
/// Matches the dynamic-row research recipe, including scale rounding. The same
/// operation quantizes weight rows once at load time and activations per batch.
pub fn quantize_rows(x: &Tensor) -> Result<(Tensor, Tensor)> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let (m, k) = x.dims2()?;
    if !matches!(x.dtype(), DType::F16 | DType::BF16)
        || !x.is_contiguous()
        || x.storage_and_layout().1.start_offset() != 0
        || m == 0
        || k == 0
    {
        candle::bail!("FP8 conversion requires nonempty zero-offset contiguous FP16/BF16 matrices")
    }
    let rows = i32::try_from(m).map_err(candle::Error::wrap)?;
    let width = i32::try_from(k).map_err(candle::Error::wrap)?;
    let dev = x.device().as_cuda_device()?;
    use cudarc::driver::sys::CUdevice_attribute::*;
    let ctx = dev.cuda_stream();
    let major = ctx
        .context()
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(candle::Error::wrap)?;
    let minor = ctx
        .context()
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(candle::Error::wrap)?;
    if major * 10 + minor < 89 {
        candle::bail!("FP8 conversion requires compute capability 8.9 or newer")
    }
    let (storage, _) = x.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle::bail!("FP8 conversion requires CUDA")
    };
    let mut out = unsafe { dev.alloc::<float8::F8E4M3>(x.elem_count())? };
    let mut scales = unsafe { dev.alloc::<f32>(m)? };
    let (name, threads, special) = match k {
        768 => ("quant_f16_768", 64, true),
        1024 => ("quant_f16_1024", 64, true),
        1152 => ("quant_f16_1152", 64, true),
        3072 => ("quant_f16_3072", 128, true),
        4096 => ("quant_f16_4096", 128, true),
        8192 => ("quant_f16_8192", 128, true),
        12288 => ("quant_f16_12288", 128, true),
        _ => ("quant_f16_generic", 128, false),
    };
    // Hopper measurements: short matrices benefit from more threads per row;
    // large matrices need enough resident rows to sustain memory throughput.
    let (name, threads, special) = if major == 9 && minor == 0 {
        match k {
            1024 if m <= 1024 => ("quant_f16_1024_t256", 256, true),
            4096 if m <= 512 => ("quant_f16_4096_t512", 512, true),
            4096 if m <= 1024 => ("quant_f16_4096_t256", 256, true),
            8192 if m <= 128 => ("quant_f16_8192_t1024", 1024, true),
            8192 if m <= 1024 => ("quant_f16_8192_t512", 512, true),
            8192 => ("quant_f16_8192_t256", 256, true),
            12288 if m <= 128 => ("quant_f16_12288_t1024", 1024, true),
            12288 if m <= 1024 => ("quant_f16_12288_t512", 512, true),
            12288 => ("quant_f16_12288_t256", 256, true),
            _ => (name, threads, special),
        }
    } else {
        (name, threads, special)
    };
    let name = if x.dtype() == DType::BF16 {
        "quant_bf16_generic"
    } else {
        name
    };
    let special = special && x.dtype() == DType::F16;
    let threads = if x.dtype() == DType::BF16 {
        128
    } else {
        threads
    };
    let function = dev.get_or_load_custom_func(
        name,
        "tei-fp8-quantize",
        include_str!(concat!(env!("OUT_DIR"), "/fp8_quant.ptx")),
    )?;
    let mut builder = function.builder();
    match x.dtype() {
        DType::F16 => {
            builder.arg(storage.as_cuda_slice::<f16>()?);
        }
        DType::BF16 => {
            builder.arg(storage.as_cuda_slice::<bf16>()?);
        }
        _ => unreachable!(),
    }
    builder.arg(&mut out).arg(&mut scales);
    if !special {
        builder.arg(&width);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(candle::Error::wrap)?;
    let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());
    let scales = candle::CudaStorage::wrap_cuda_slice(scales, dev.clone());
    Ok((
        Tensor::from((Storage::Cuda(out), candle::Shape::from((m, k)))),
        Tensor::from((Storage::Cuda(scales), candle::Shape::from(m))),
    ))
}

/// Experimental fused packed SwiGLU and row conversion. Unsupported shapes
/// return None so the caller can retain the established separate operations.
pub fn quantize_packed_swiglu(x: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    if x.rank() != 2
        || x.dtype() != DType::F16
        || !x.is_contiguous()
        || x.storage_and_layout().1.start_offset() != 0
    {
        return Ok(None);
    }
    let (m, packed) = x.dims2()?;
    if m == 0 || m > i32::MAX as usize || packed % 2 != 0 {
        return Ok(None);
    }
    let k = packed / 2;
    let threads = match k {
        3072 => {
            if m <= 128 {
                1024
            } else {
                256
            }
        }
        8192 => 1024,
        12288 => {
            if m <= 128 {
                1024
            } else {
                512
            }
        }
        _ => return Ok(None),
    };
    let dev = x.device().as_cuda_device()?;
    let (storage, _) = x.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle::bail!("FP8 SwiGLU requires CUDA")
    };
    let input = storage.as_cuda_slice::<f16>()?;
    let mut out = unsafe { dev.alloc::<float8::F8E4M3>(m * k)? };
    let mut scales = unsafe { dev.alloc::<f32>(m)? };
    let name = format!("swiglu_quant_{k}_{threads}");
    let function = dev.get_or_load_custom_func(
        &name,
        "tei-fp8-quantize",
        include_str!(concat!(env!("OUT_DIR"), "/fp8_quant.ptx")),
    )?;
    let mut builder = function.builder();
    builder.arg(input).arg(&mut out).arg(&mut scales);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(candle::Error::wrap)?;
    Ok(Some((
        Tensor::from((
            Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(out, dev.clone())),
            candle::Shape::from((m, k)),
        )),
        Tensor::from((
            Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(scales, dev.clone())),
            candle::Shape::from(m),
        )),
    )))
}

/// Experimental bias-free FP8 linear layer. Weight conversion happens once;
/// the caller retains a worker-owned GEMM executor across layer invocations.
pub struct Fp8Linear {
    dtype: DType,
    weight: Tensor,
    scales: Tensor,
}
impl Fp8Linear {
    pub fn new(weight: &Tensor) -> Result<Self> {
        let (n, k) = weight.dims2()?;
        if n == 0 || k == 0 || n % 16 != 0 || k % 16 != 0 {
            candle::bail!("FP8 linear dimensions must be nonzero multiples of 16")
        }
        let dtype = weight.dtype();
        let (weight, scales) = quantize_rows(weight)?;
        Ok(Self {
            weight,
            scales,
            dtype,
        })
    }
    pub fn forward_packed_swiglu(
        &self,
        x: &Tensor,
        executor: &mut Fp8Matmul,
    ) -> Result<Option<Tensor>> {
        if x.dtype() != self.dtype {
            candle::bail!("FP8 linear input dtype must match weights");
        }
        if x.dim(1)? != self.weight.dim(1)? * 2 || !x.device().same_device(self.weight.device()) {
            candle::bail!("FP8 packed SwiGLU input dimension/device mismatch")
        }
        let Some((x, sx)) = quantize_packed_swiglu(x)? else {
            return Ok(None);
        };
        Ok(Some(executor.scaled_mm(
            &x,
            &self.weight,
            &sx,
            &self.scales,
        )?))
    }
    pub fn forward(&self, x: &Tensor, executor: &mut Fp8Matmul) -> Result<Tensor> {
        if x.dtype() != self.dtype {
            candle::bail!("FP8 linear input dtype must match weights");
        }
        let (_, k) = x.dims2()?;
        if k != self.weight.dim(1)? || !x.device().same_device(self.weight.device()) {
            candle::bail!("FP8 linear input dimension/device mismatch")
        }
        let (x, sx) = quantize_rows(x)?;
        executor
            .scaled_mm_dtype(&x, &self.weight, &sx, &self.scales, self.dtype)
            .map_err(|error| {
                error.context(format!(
                    "dynamic FP8 linear: input {:?}, weight {:?}",
                    x.dims(),
                    self.weight.dims()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_row_scaling_preserves_scales_and_avoids_fp16_overflow() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        // Exercise the Ada/Blackwell path even on the Hopper test host.
        executor.fused_row_scaling = false;
        for m in [1, 17, 129] {
            let (n, k) = (32, 1024);
            let x = Tensor::from_vec(vec![float8::F8E4M3::from_f32(448.); m * k], (m, k), &dev)?;
            let w = Tensor::from_vec(vec![float8::F8E4M3::from_f32(448.); n * k], (n, k), &dev)?;
            for factor in [1., 2.] {
                let xs: Vec<f32> = (0..m)
                    .map(|i| factor * (i % 4 + 1) as f32 / 4096.)
                    .collect();
                let ws: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32 / 2048.).collect();
                let sx = Tensor::from_vec(xs.clone(), m, &dev)?;
                let sw = Tensor::from_vec(ws.clone(), n, &dev)?;
                for dtype in [DType::F16, DType::BF16] {
                    let y = executor
                        .scaled_mm_dtype(&x, &w, &sx, &sw, dtype)?
                        .to_dtype(DType::F32)?
                        .to_vec2::<f32>()?;
                    for i in 0..m {
                        for j in 0..n {
                            let expected = (k as f32 * 448. * 448.) * xs[i] * ws[j];
                            let expected = match dtype {
                                DType::F16 => f16::from_f32(expected).to_f32(),
                                _ => bf16::from_f32(expected).to_f32(),
                            };
                            assert_eq!(y[i][j], expected);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_quantization_and_gemm_preserve_range() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        for k in [17, 1024, 3072] {
            let values = (0..3 * k)
                .map(|i| {
                    bf16::from_f32(if i < k {
                        0.
                    } else {
                        ((i * 37 % 513) as f32 - 256.) * 4096.
                    })
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values.clone(), (3, k), &dev)?;
            let (q, s) = quantize_rows(&input)?;
            let q = q.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
            let s = s.to_vec1::<f32>()?;
            for row in 0..3 {
                let amax = values[row * k..(row + 1) * k]
                    .iter()
                    .map(|x| x.to_f32().abs())
                    .fold(1e-12f32, f32::max);
                let scale = amax * (1f32 / 448f32);
                assert_eq!(s[row].to_bits(), scale.to_bits());
                for col in 0..k {
                    let expected = float8::F8E4M3::from_f32(
                        (values[row * k + col].to_f32() / scale).clamp(-448., 448.),
                    );
                    assert_eq!(q[row * k + col].to_bits(), expected.to_bits());
                }
            }
        }
        let weight = Tensor::ones((16, 32), DType::BF16, &dev)?;
        let layer = Fp8Linear::new(&weight)?;
        let input = Tensor::full(bf16::from_f32(131072.), (17, 32), &dev)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        let output = layer.forward(&input, &mut executor)?;
        assert_eq!(output.dtype(), DType::BF16);
        for y in output.flatten_all()?.to_vec1::<bf16>()? {
            assert_eq!(y.to_f32(), 4194304.);
        }
        assert!(layer
            .forward(&input.to_dtype(DType::F16)?, &mut executor)
            .is_err());
        // Reuse the same GEMM shape with both output dtypes: cache plans must differ.
        let (q, sx) = quantize_rows(&Tensor::ones((17, 32), DType::BF16, &dev)?)?;
        let (w, sw) = quantize_rows(&weight)?;
        for dtype in [DType::F16, DType::BF16, DType::F16] {
            let result = executor.scaled_mm_dtype(&q, &w, &sx, &sw, dtype)?;
            assert_eq!(result.dtype(), dtype);
            assert!(result
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|x| *x == 32.));
        }
        assert_eq!(
            executor.plans.len(),
            if executor.fused_row_scaling { 2 } else { 1 }
        );
        Ok(())
    }

    #[test]
    fn fused_swiglu_matches_candle_conversion() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        for k in [3072, 8192, 12288] {
            for m in [1, 17, 129] {
                let values = (0..m * 2 * k)
                    .map(|i| f16::from_f32(((i * 37 % 257) as f32 - 128.) / 16.))
                    .collect::<Vec<_>>();
                let x = Tensor::from_vec(values, (m, 2 * k), &dev)?;
                let gate = x.narrow(1, 0, k)?.contiguous()?;
                let up = x.narrow(1, k, k)?.contiguous()?;
                let reference = quantize_rows(&(gate.silu()? * up)?)?;
                let fused = quantize_packed_swiglu(&x)?.expect("supported shape");
                let bytes = |q: &Tensor| -> Result<Vec<u8>> {
                    Ok(q.flatten_all()?
                        .to_vec1::<float8::F8E4M3>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect())
                };
                assert_eq!(bytes(&reference.0)?, bytes(&fused.0)?, "m={m} k={k}");
                assert_eq!(
                    reference
                        .1
                        .to_vec1::<f32>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    fused
                        .1
                        .to_vec1::<f32>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(quantize_packed_swiglu(&Tensor::zeros((1, 2048), DType::F16, &dev)?)?.is_none());
        Ok(())
    }

    #[test]
    fn row_dispatch_boundaries_preserve_bytes_and_scales() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        for k in [1024, 4096, 8192, 12288] {
            let mut values = (0..3 * k)
                .map(|i| f16::from_f32(((i * 37 % 513) as f32 - 256.) / 32.))
                .collect::<Vec<_>>();
            values[..k].fill(f16::ZERO);
            values[k] = f16::MAX;
            let (q, scales) = quantize_rows(&Tensor::from_vec(values.clone(), (3, k), &dev)?)?;
            let reference = q.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
            let scales = scales.to_vec1::<f32>()?;
            for m in [128, 129, 512, 513, 1024, 1025] {
                let data = values
                    .iter()
                    .copied()
                    .cycle()
                    .take(m * k)
                    .collect::<Vec<_>>();
                let (q, s) = quantize_rows(&Tensor::from_vec(data, (m, k), &dev)?)?;
                let q = q.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
                let s = s.to_vec1::<f32>()?;
                assert!(
                    q.iter()
                        .zip(reference.iter().cycle())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "M={m} K={k} bytes"
                );
                assert!(
                    s.iter()
                        .zip(scales.iter().cycle())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "M={m} K={k} scales"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn row_scales_and_irregular_m() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        for m in [1, 17, 128, 17, 1] {
            let x = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); m * 32], (m, 32), &dev)?;
            let w = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); 16 * 32], (16, 32), &dev)?;
            let sx = Tensor::from_vec(
                (0..m).map(|i| (i + 1) as f32 / 32.).collect::<Vec<_>>(),
                m,
                &dev,
            )?;
            let sw = Tensor::from_vec(
                (0..16).map(|i| (i + 1) as f32 / 16.).collect::<Vec<_>>(),
                16,
                &dev,
            )?;
            let y = executor
                .scaled_mm(&x, &w, &sx, &sw)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            for i in 0..m {
                for j in 0..16 {
                    assert_eq!(y[i][j], (i + 1) as f32 * (j + 1) as f32 / 16.);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn signed_values_and_layout_guards() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let (m, n, k) = (17, 32, 64);
        let xv = (0..m * k)
            .map(|i| ((i * 7 % 9) as f32 - 4.) / 4.)
            .collect::<Vec<_>>();
        let wv = (0..n * k)
            .map(|i| ((i * 11 % 13) as f32 - 6.) / 4.)
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(
            xv.iter()
                .copied()
                .map(float8::F8E4M3::from_f32)
                .collect::<Vec<_>>(),
            (m, k),
            &dev,
        )?;
        let w = Tensor::from_vec(
            wv.iter()
                .copied()
                .map(float8::F8E4M3::from_f32)
                .collect::<Vec<_>>(),
            (n, k),
            &dev,
        )?;
        let sx = Tensor::full(0.5f32, m, &dev)?;
        let sw = Tensor::full(0.25f32, n, &dev)?;
        let y = scaled_mm(&x, &w, &sx, &sw)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        for i in 0..m {
            for j in 0..n {
                let reference = (0..k).map(|q| xv[i * k + q] * wv[j * k + q]).sum::<f32>() * 0.125;
                assert_eq!(y[i][j], reference);
            }
        }
        assert!(scaled_mm(&x, &w, &sx.narrow(0, 0, m - 1)?, &sw).is_err());
        assert!(scaled_mm(&x.t()?, &w, &sx, &sw).is_err());
        assert!(scaled_mm(&x.narrow(0, 1, m - 1)?, &w, &sx.narrow(0, 1, m - 1)?, &sw).is_err());
        assert!(scaled_mm(&Tensor::ones((m, k), DType::F16, &dev)?, &w, &sx, &sw).is_err());
        Ok(())
    }
    #[test]
    fn cache_refreshes_scales_and_is_bounded() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let mut executor = Fp8Matmul::new(&dev)?;
        let w = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); 16 * 32], (16, 32), &dev)?;
        let sw = Tensor::full(0.25f32, 16, &dev)?;
        let mut pending = Vec::new();
        for m in 1..=17 {
            let x = Tensor::from_vec(vec![float8::F8E4M3::from_f32(1.); m * 32], (m, 32), &dev)?;
            let sa = Tensor::full(0.5f32, m, &dev)?;
            let sb = Tensor::full(0.25f32, m, &dev)?;
            for (scale, expected) in [(&sa, 4f32), (&sb, 2f32), (&sa, 4f32)] {
                pending.push((executor.scaled_mm(&x, &w, scale, &sw)?, expected));
            }
            assert!(executor.plans.len() <= 16);
        }
        assert_eq!(executor.plans.len(), 1);
        // Read only after cache eviction and after temporary scale/input owners
        // have been dropped, exercising asynchronous lifetime tracking.
        for (result, expected) in pending {
            let result = result
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(result.iter().all(|v| *v == expected));
        }
        Ok(())
    }
    #[test]
    fn native_conversion_matches_finite_reference() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        for k in [17, 768, 777, 1024, 1152, 3072, 4096, 8192, 12288] {
            let mut values = vec![f16::ZERO; 3 * k];
            for i in k..3 * k {
                values[i] = f16::from_f32(((i * 37 % 513) as f32 - 256.) / 32.);
            }
            values[k] = f16::MAX;
            values[2 * k] = f16::from_bits(1);
            let x = Tensor::from_vec(values.clone(), (3, k), &dev)?;
            let (q, s) = quantize_rows(&x)?;
            let q = q.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
            let s = s.to_vec1::<f32>()?;
            for row in 0..3 {
                let amax = values[row * k..(row + 1) * k]
                    .iter()
                    .map(|v| v.to_f32().abs())
                    .fold(1e-12f32, f32::max);
                let scale = amax * (1f32 / 448f32);
                assert_eq!(s[row].to_bits(), scale.to_bits());
                for col in 0..k {
                    let reference = float8::F8E4M3::from_f32(
                        (values[row * k + col].to_f32() / scale).clamp(-448., 448.),
                    );
                    assert_eq!(
                        q[row * k + col].to_bits(),
                        reference.to_bits(),
                        "K={k} row={row} col={col}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires INFINITY_FP8_FIXTURE from the independent Torch/Triton reference"]
    fn native_conversion_matches_external_fixture() -> Result<()> {
        let dev = candle::Device::new_cuda(0)?;
        let path = std::env::var("INFINITY_FP8_FIXTURE").map_err(candle::Error::wrap)?;
        let tensors = candle::safetensors::load(path, &dev)?;
        let mut checked = 0;
        for (name, x) in &tensors {
            if !name.ends_with(".input") {
                continue;
            }
            checked += 1;
            let prefix = name.trim_end_matches(".input");
            if let Some(weight) = tensors.get(&format!("{prefix}.weight")) {
                let layer = Fp8Linear::new(weight)?;
                let mut executor = Fp8Matmul::new(&dev)?;
                let output = layer
                    .forward(x, &mut executor)?
                    .flatten_all()?
                    .to_vec1::<f16>()?;
                let (m, k) = x.dims2()?;
                let n = weight.dim(0)?;
                let suffix = if executor.plans[&(m, n, k, DType::F16)].fast_accum {
                    "expected"
                } else {
                    "expected_full_accum"
                };
                let expected = tensors[&format!("{prefix}.{suffix}")]
                    .flatten_all()?
                    .to_vec1::<f16>()?;
                assert_eq!(output.len(), expected.len());
                assert_eq!(
                    output
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count(),
                    0,
                    "linear {prefix}"
                );
            }

            let (q, s) = quantize_rows(x)?;
            let actual = q.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
            let expected = tensors[&format!("{prefix}.fp8")]
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{prefix}"
            );
            let actual = s.flatten_all()?.to_vec1::<f32>()?;
            let expected = tensors[&format!("{prefix}.scale")]
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{prefix}"
            );
        }
        assert!(checked > 0, "fixture has no inputs");
        Ok(())
    }
}
