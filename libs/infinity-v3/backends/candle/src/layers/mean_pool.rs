use candle::{Result, Tensor};

/// Mean-pool selected sequences from packed token embeddings, in selection order.
pub fn mean_pool(outputs: &Tensor, cumulative: &[u32], indices: &[u32]) -> Result<Tensor> {
    let (rows, width) = outputs.dims2()?;
    let mut spans = Vec::with_capacity(indices.len() * 2);
    for &index in indices {
        let index = index as usize;
        let (Some(&start), Some(&end)) = (cumulative.get(index), cumulative.get(index + 1)) else {
            candle::bail!("mean pooling sequence index out of range");
        };
        if end <= start || end as usize > rows {
            candle::bail!("mean pooling requires nonempty, in-bounds sequences");
        }
        spans.extend([start, end - start]);
    }
    if indices.is_empty() {
        return Tensor::zeros((0, width), outputs.dtype(), outputs.device());
    }
    #[cfg(feature = "cuda")]
    if matches!(outputs.device(), candle::Device::Cuda(_))
        && matches!(outputs.dtype(), candle::DType::F16 | candle::DType::BF16)
        && outputs.is_contiguous()
        && width > 0
        && width <= u32::MAX as usize
        && indices.len() <= 65535
    {
        let spans = Tensor::from_vec(spans, (indices.len(), 2), outputs.device())?;
        return outputs.apply_op2_no_bwd(&spans, &cuda::MeanPool);
    }
    let results: Result<Vec<_>> = spans
        .chunks_exact(2)
        .map(|span| {
            outputs
                .narrow(0, span[0] as usize, span[1] as usize)?
                .sum_keepdim(0)?
                / (span[1] as f64)
        })
        .collect();
    Tensor::cat(&results?, 0)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::CudaDType;
    use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape};
    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/pooling_ptx.rs"));
    }
    pub struct MeanPool;
    fn launch<T: CudaDType + DeviceRepr>(
        storage: &CudaStorage,
        layout: &Layout,
        spans: &CudaStorage,
        spans_layout: &Layout,
        name: &str,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = layout.shape().dims2()?;
        let count = spans_layout.shape().dim(0)?;
        let device = storage.device();
        let input = storage.as_cuda_slice::<T>()?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + rows * width);
        let spans = spans.as_cuda_slice::<u32>()?;
        let spans =
            spans.slice(spans_layout.start_offset()..spans_layout.start_offset() + count * 2);
        // Each valid (sequence, feature) output has exactly one writer.
        let mut output = unsafe { device.alloc::<T>(count * width)? };
        let function =
            device.get_or_load_custom_func(name, "tei-packed-mean-pool", ptx::MEAN_POOL)?;
        let mut builder = function.builder();
        let hidden = width as u32;
        builder
            .arg(&input)
            .arg(&spans)
            .arg(&mut output)
            .arg(&hidden);
        // The public entry point validates spans, shape, dtype and launch bounds.
        // Input/output lifetimes and stream dependencies remain tracked.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (width.div_ceil(8) as u32, count as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (count, width).into(),
        ))
    }
    impl CustomOp2 for MeanPool {
        fn name(&self) -> &'static str {
            "packed-mean-pool"
        }
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle::bail!("packed-mean-pool requires CUDA")
        }
        fn cuda_fwd(
            &self,
            s: &CudaStorage,
            l: &Layout,
            spans: &CudaStorage,
            sl: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            match s.dtype() {
                DType::F16 => launch::<half::f16>(s, l, spans, sl, "packed_mean_pool_f16"),
                DType::BF16 => launch::<half::bf16>(s, l, spans, sl, "packed_mean_pool_bf16"),
                dtype => candle::bail!("unsupported mean pooling dtype {dtype:?}"),
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn reference(xs: &Tensor, cumulative: &[u32], indices: &[u32]) -> Result<Tensor> {
        let values: Result<Vec<_>> = indices
            .iter()
            .map(|&i| {
                let start = cumulative[i as usize];
                let len = cumulative[i as usize + 1] - start;
                xs.narrow(0, start as usize, len as usize)?.sum_keepdim(0)? / len as f64
            })
            .collect();
        Tensor::cat(&values?, 0)
    }
    fn exact(a: &Tensor, b: &Tensor) -> Result<()> {
        assert_eq!(a.shape(), b.shape());
        match a.dtype() {
            DType::F16 => {
                let a = a.flatten_all()?.to_vec1::<half::f16>()?;
                let b = b.flatten_all()?.to_vec1::<half::f16>()?;
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            DType::BF16 => {
                let a = a.flatten_all()?.to_vec1::<half::bf16>()?;
                let b = b.flatten_all()?.to_vec1::<half::bf16>()?;
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            _ => unreachable!(),
        }
        Ok(())
    }
    #[test]
    fn packed_mean_matches_candle_reduction_tree() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let lengths = [
            1u32, 2, 3, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2048, 8192,
        ];
        let mut cumulative = vec![0];
        for length in lengths {
            cumulative.push(cumulative.last().unwrap() + length);
        }
        let rows = *cumulative.last().unwrap() as usize;
        for dtype in [DType::F16, DType::BF16] {
            for width in [1usize, 7, 8, 31, 384, 1024] {
                let data: Vec<_> = (0..(rows + 3) * width)
                    .map(|i| {
                        let mut x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
                        x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
                        ((x ^ (x >> 22)) % 10001) as f32 / 2500.0 - 2.0
                    })
                    .collect();
                let xs = Tensor::from_vec(data, (rows + 3, width), &device)?
                    .to_dtype(dtype)?
                    .narrow(0, 3, rows)?;
                for indices in [
                    (0..lengths.len() as u32).collect::<Vec<_>>(),
                    vec![16, 2, 0, 16, 4],
                    vec![7],
                ] {
                    exact(
                        &mean_pool(&xs, &cumulative, &indices)?,
                        &reference(&xs, &cumulative, &indices)?,
                    )?;
                }
            }
        }
        Ok(())
    }
    #[test]
    fn mean_pool_validates_spans_and_handles_noncontiguous_views() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let xs = Tensor::arange(0f32, 120f32, &device)?
            .reshape((12, 10))?
            .to_dtype(DType::F16)?
            .narrow(1, 1, 7)?;
        let cumulative = [0, 3, 12];
        exact(
            &mean_pool(&xs, &cumulative, &[1, 0])?,
            &reference(&xs, &cumulative, &[1, 0])?,
        )?;
        assert_eq!(mean_pool(&xs, &cumulative, &[])?.dims(), &[0, 7]);
        assert!(mean_pool(&xs, &cumulative, &[2]).is_err());
        assert!(mean_pool(&xs, &[0, 0], &[0]).is_err());
        assert!(mean_pool(&xs, &[0, 13], &[0]).is_err());
        assert!(mean_pool(&xs, &[4, 2], &[0]).is_err());
        Ok(())
    }
}
