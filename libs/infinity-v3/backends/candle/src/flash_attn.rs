use candle::Tensor;
#[cfg(feature = "cuda")]
thread_local! {
    static COMPUTE_CAPS: std::cell::RefCell<std::collections::HashMap<
        candle::cuda_backend::DeviceId, usize
    >> = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(feature = "cuda")]
pub(crate) fn runtime_compute_cap(device: &candle::Device) -> candle::Result<usize> {
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("Flash attention requires a CUDA tensor");
    };
    COMPUTE_CAPS.with(|caps| {
        let mut caps = caps.borrow_mut();
        if let Some(&cap) = caps.get(&cuda.id()) {
            return Ok(cap);
        }
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
        };
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let major = context.attribute(MAJOR).map_err(candle::Error::wrap)?;
        let minor = context.attribute(MINOR).map_err(candle::Error::wrap)?;
        let cap = (major * 10 + minor) as usize;
        caps.insert(cuda.id(), cap);
        Ok(cap)
    })
}

#[allow(clippy::too_many_arguments, unused)]
pub(crate) fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    if q.device().is_cpu() {
        // infinity_v3/GPU callers supply cumulative offsets; Candle's CPU API takes lengths.
        let lengths = |offsets: &Tensor| -> candle::Result<Tensor> {
            let offsets = offsets.to_vec1::<u32>()?;
            if offsets.first() != Some(&0) || offsets.len() < 2 {
                candle::bail!("packed attention offsets must start at zero");
            }
            let lengths: candle::Result<Vec<u32>> = offsets
                .windows(2)
                .map(|w| {
                    w[1].checked_sub(w[0]).ok_or_else(|| {
                        candle::Error::Msg("packed attention offsets must be nondecreasing".into())
                    })
                })
                .collect();
            Tensor::new(lengths?, q.device())
        };
        // Candle treats a left-only window as causal. CUDA treats the missing
        // right bound as unbounded, so make that bound explicit on CPU.
        let window_size_right = if !causal && window_size_left.is_some() {
            Some(window_size_right.unwrap_or(max_seqlen_k))
        } else {
            window_size_right
        };
        return candle_nn::attention::flash_attn_varlen_cpu(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            alibi_slopes,
            &lengths(seqlens_q)?,
            &lengths(seqlens_k)?,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
        );
    }
    #[cfg(not(feature = "cuda"))]
    candle::bail!("packed attention requires CPU or a CUDA build");
    #[cfg(feature = "cuda")]
    {
        let runtime_compute_cap = runtime_compute_cap(q.device())?;

        #[cfg(feature = "fa4")]
        if runtime_compute_cap == candle_flash_attn_v4::compiled_compute_capability() as usize
            && alibi_slopes.is_none()
        {
            if let Some(output) = crate::fa4_native::try_forward(
                q,
                k,
                v,
                seqlens_q,
                seqlens_k,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
            )? {
                return Ok(output);
            }
        }

        if (80..90).contains(&runtime_compute_cap)
            || runtime_compute_cap == 90
            || runtime_compute_cap == 120
            || runtime_compute_cap == 100
        {
            #[cfg(feature = "flash-attn")]
            {
                use candle_flash_attn::{
                    flash_attn_varlen_alibi_windowed, flash_attn_varlen_windowed,
                };

                let window_size_right = if causal {
                    Some(0)
                } else if window_size_right.is_some() {
                    window_size_right
                } else {
                    None
                };

                let attention = if let Some(alibi_slopes) = alibi_slopes {
                    flash_attn_varlen_alibi_windowed(
                        q,
                        k,
                        v,
                        alibi_slopes,
                        seqlens_q,
                        seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        softmax_scale,
                        window_size_left,
                        window_size_right,
                    )
                } else {
                    flash_attn_varlen_windowed(
                        q,
                        k,
                        v,
                        seqlens_q,
                        seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        softmax_scale,
                        window_size_left,
                        window_size_right,
                    )
                };

                return attention;
            }
            #[cfg(not(feature = "flash-attn"))]
            candle::bail!("Flash attention is not installed. Use `flash-attn` feature.")
        }
        candle::bail!(
            "GPU with CUDA capability {} is not supported",
            runtime_compute_cap
        );
    }
}

/// Precision/device combinations supported by the packed model implementations.
pub(crate) fn validate_packed_device(vb: &candle_nn::VarBuilder) -> candle::Result<()> {
    match (vb.device(), vb.dtype()) {
        (candle::Device::Cpu, candle::DType::F32 | candle::DType::F16) => Ok(()),
        (candle::Device::Cuda(_), candle::DType::F16 | candle::DType::BF16) => Ok(()),
        _ => candle::bail!("packed inference requires CPU fp32/fp16 or CUDA fp16/bf16"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Result};

    #[test]
    fn cpu_packed_attention_matches_unfused_reference() -> Result<()> {
        let device = Device::Cpu;
        let offsets = Tensor::new(&[0u32, 1, 6, 9, 16], &device)?;
        let lengths = Tensor::new(&[1u32, 5, 3, 7], &device)?;
        for dtype in [DType::F32, DType::F16] {
            let packed = Tensor::randn(0f32, 0.3f32, (16, 8, 16), &device)?.to_dtype(dtype)?;
            // Deliberately non-contiguous head slices, as in merged QKV projections.
            let q = packed.narrow(1, 0, 4)?;
            let k = packed.narrow(1, 4, 2)?;
            let v = packed.narrow(1, 6, 2)?;
            let slopes = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &device)?;
            for (causal, window, right, alibi) in [
                (false, None, None, None),
                (true, None, None, None),
                (false, Some(2), Some(2), None),
                (false, Some(2), None, None),
                (true, Some(2), Some(0), Some(&slopes)),
            ] {
                let actual = flash_attn_varlen(
                    &q, &k, &v, alibi, &offsets, &offsets, 7, 7, 0.25, causal, window, right,
                )?;
                let reference_right = if !causal && window.is_some() {
                    Some(right.unwrap_or(7))
                } else {
                    right
                };
                let expected = candle_nn::attention::flash_attn_varlen_unfused(
                    &q,
                    &k,
                    &v,
                    alibi,
                    &lengths,
                    &lengths,
                    7,
                    7,
                    0.25,
                    causal,
                    window,
                    reference_right,
                )?;
                let error = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(
                    error < if dtype == DType::F32 { 1e-5 } else { 0.002 },
                    "{dtype:?} causal={causal} window={window:?}: {error}"
                );
            }
        }
        Ok(())
    }
}
