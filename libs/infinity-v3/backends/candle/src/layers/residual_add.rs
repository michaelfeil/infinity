use candle::{Result, Tensor};

/// Add a residual while preserving the model dtype's rounding.
pub fn residual_add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(lhs.device(), candle::Device::Cuda(_))
        && matches!(lhs.dtype(), candle::DType::F16 | candle::DType::BF16)
        && lhs.dtype() == rhs.dtype()
        && lhs.device().same_device(rhs.device())
        && lhs.shape() == rhs.shape()
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && lhs.layout().start_offset().is_multiple_of(4)
        && rhs.layout().start_offset().is_multiple_of(4)
        && lhs.elem_count() > 0
        && lhs.elem_count().is_multiple_of(4)
        && (lhs.elem_count() / 4).div_ceil(256) <= i32::MAX as usize
    {
        return lhs.apply_op2_no_bwd(rhs, &cuda::PackedAdd);
    }
    lhs.add(rhs)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::CudaDType;
    use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape};
    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/residual_ptx.rs"));
    }
    pub struct PackedAdd;

    fn launch<T: CudaDType + DeviceRepr>(
        lhs: &CudaStorage,
        ll: &Layout,
        rhs: &CudaStorage,
        rl: &Layout,
        name: &str,
    ) -> Result<(CudaStorage, Shape)> {
        let n = ll.shape().elem_count();
        let device = lhs.device();
        let lhs = lhs
            .as_cuda_slice::<T>()?
            .slice(ll.start_offset()..ll.start_offset() + n);
        let rhs = rhs
            .as_cuda_slice::<T>()?
            .slice(rl.start_offset()..rl.start_offset() + n);
        // Each thread writes four distinct elements; the public entry point checks alignment and length.
        let mut output = unsafe { device.alloc::<T>(n)? };
        let vectors = (n / 4) as u64;
        let function =
            device.get_or_load_custom_func(name, "tei-vector-residual-add", ptx::RESIDUAL_ADD)?;
        let mut builder = function.builder();
        builder.arg(&lhs).arg(&rhs).arg(&mut output).arg(&vectors);
        // All input/output buffers are tracked by the launch builder on the tensor device's stream.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (vectors.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            ll.shape().clone(),
        ))
    }
    impl CustomOp2 for PackedAdd {
        fn name(&self) -> &'static str {
            "vector-residual-add"
        }
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle::bail!("vector-residual-add requires CUDA")
        }
        fn cuda_fwd(
            &self,
            lhs: &CudaStorage,
            ll: &Layout,
            rhs: &CudaStorage,
            rl: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            match lhs.dtype() {
                DType::F16 => launch::<half::f16>(lhs, ll, rhs, rl, "residual_add_vec4_f16"),
                DType::BF16 => launch::<half::bf16>(lhs, ll, rhs, rl, "residual_add_vec4_bf16"),
                dtype => candle::bail!("unsupported vector-residual-add dtype {dtype:?}"),
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn compare(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
        let expected = lhs
            .add(rhs)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = residual_add(lhs, rhs)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "element {i}: {a:?} != {b:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn vector_residual_add_matches_candle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            let lhs = match dtype {
                DType::F16 => Tensor::from_vec(
                    (0..=u16::MAX).map(half::f16::from_bits).collect::<Vec<_>>(),
                    (256, 256),
                    &device,
                )?,
                DType::BF16 => Tensor::from_vec(
                    (0..=u16::MAX)
                        .map(half::bf16::from_bits)
                        .collect::<Vec<_>>(),
                    (256, 256),
                    &device,
                )?,
                _ => unreachable!(),
            };
            for value in [0., -0., 1., -1., 0.001, 1000.] {
                let rhs = Tensor::full(value, lhs.shape(), &device)?.to_dtype(dtype)?;
                compare(&lhs, &rhs)?;
                // Unaligned and odd-sized views must retain the regular add path.
                compare(
                    &lhs.flatten_all()?.narrow(0, 1, 1023)?,
                    &rhs.flatten_all()?.narrow(0, 1, 1023)?,
                )?;
                // Matching strided tensors also use the regular implementation.
                compare(&lhs.t()?, &rhs.t()?)?;
            }
        }
        Ok(())
    }
}
