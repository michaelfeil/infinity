#[cfg(feature = "cuda")]
use candle::{DType, Device};
use candle::{Result, Tensor};

use super::HiddenAct;

/// Apply an activation to the first half of a packed projection and multiply
/// by the second half. The CUDA kernel keeps Candle's intermediate rounding.
pub fn gated_activation(xs: &Tensor, activation: Option<&HiddenAct>) -> Result<Tensor> {
    let packed_width = xs.dim(candle::D::Minus1)?;
    if !packed_width.is_multiple_of(2) {
        candle::bail!("gated activation requires an even last dimension");
    }
    #[cfg(feature = "cuda")]
    if matches!(activation, Some(HiddenAct::Gelu | HiddenAct::Silu))
        && matches!(xs.device(), Device::Cuda(_))
        && matches!(xs.dtype(), DType::F16 | DType::BF16)
        && xs.rank() == 2
        && xs.is_contiguous()
        && packed_width > 0
        && xs.dim(0)? > 0
        && xs.dim(0)? <= i32::MAX as usize
        && packed_width / 2 <= u32::MAX as usize
    {
        return xs.apply_op1_no_bwd(&cuda::PackedGlu {
            gelu: matches!(activation, Some(HiddenAct::Gelu)),
        });
    }

    let width = packed_width / 2;
    let value = xs.narrow(candle::D::Minus1, 0, width)?;
    let gate = xs.narrow(candle::D::Minus1, width, width)?;
    let value = match activation {
        Some(activation) => activation.forward(&value)?,
        None => value,
    };
    value.mul(&gate)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::CudaDType;
    use candle::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/activation_ptx.rs"));
    }

    pub struct PackedGlu {
        pub gelu: bool,
    }

    fn launch<T: CudaDType + DeviceRepr>(
        storage: &CudaStorage,
        layout: &Layout,
        names: [&str; 2],
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, packed_width) = layout.shape().dims2()?;
        let width = packed_width / 2;
        let device = storage.device();
        let input = storage.as_cuda_slice::<T>()?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + rows * packed_width);
        // Every output element is written by precisely one thread in the kernel.
        let mut output = unsafe { device.alloc::<T>(rows * width)? };
        if rows != 0 {
            let output_vecs = rows * width / 4;
            let vectorized = width.is_multiple_of(4)
                && layout.start_offset().is_multiple_of(4)
                && output_vecs.div_ceil(256) <= i32::MAX as usize;
            let function = device.get_or_load_custom_func(
                names[usize::from(vectorized)],
                "tei-packed-gated-activation",
                ptx::GATED_ACTIVATION,
            )?;
            let width = if vectorized { width / 4 } else { width } as u32;
            let output_vecs = output_vecs as u64;
            let mut builder = function.builder();
            builder.arg(&input).arg(&mut output).arg(&width);
            let config = if vectorized {
                builder.arg(&output_vecs);
                LaunchConfig {
                    grid_dim: (output_vecs.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (if rows <= 256 { 1024 } else { 256 }, 1, 1),
                    shared_mem_bytes: 0,
                }
            };
            // The public entry point checks layout, dtype, and representable sizes.
            // The launch builder tracks all buffers on the tensor's CUDA stream.
            unsafe { builder.launch(config) }.map_err(candle::Error::wrap)?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (rows, width).into(),
        ))
    }

    impl CustomOp1 for PackedGlu {
        fn name(&self) -> &'static str {
            "packed-gated-activation"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle::bail!("packed-gated-activation requires CUDA")
        }

        fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
            match storage.dtype() {
                DType::F16 => launch::<half::f16>(
                    storage,
                    layout,
                    if self.gelu {
                        ["packed_geglu_f16", "packed_geglu_vec4_f16"]
                    } else {
                        ["packed_swiglu_f16", "packed_swiglu_vec4_f16"]
                    },
                ),
                DType::BF16 => launch::<half::bf16>(
                    storage,
                    layout,
                    if self.gelu {
                        ["packed_geglu_bf16", "packed_geglu_vec4_bf16"]
                    } else {
                        ["packed_swiglu_bf16", "packed_swiglu_vec4_bf16"]
                    },
                ),
                dtype => candle::bail!("unsupported packed-gated-activation dtype {dtype:?}"),
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn compare_one(input: &Tensor, gelu: bool) -> Result<()> {
        let reference = if gelu {
            let chunks = input.chunk(2, candle::D::Minus1)?;
            chunks[0].gelu()?.mul(&chunks[1])?
        } else {
            candle_nn::ops::swiglu(input)?
        };
        let expected = reference
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let activation = if gelu {
            HiddenAct::Gelu
        } else {
            HiddenAct::Silu
        };
        let actual = gated_activation(input, Some(&activation))?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits(),
                "element {index}: actual={actual:?}, expected={expected:?}"
            );
        }
        Ok(())
    }

    fn compare(input: &Tensor) -> Result<()> {
        compare_one(input, false)?;
        compare_one(input, true)
    }

    #[test]
    fn packed_glu_all_half_bit_patterns() -> Result<()> {
        let device = Device::new_cuda(0)?;
        // Every possible gate encoding, including signed zero, subnormals,
        // infinities and NaNs. The up operand uses a permuted bit pattern.
        let width = 128;
        let bits: Vec<u16> = (0..512)
            .flat_map(|row| {
                (0..2 * width).map(move |col| {
                    let gate = (row * width + col % width) as u16;
                    if col < width {
                        gate
                    } else {
                        gate.wrapping_mul(251).wrapping_add(1237)
                    }
                })
            })
            .collect();
        let f16: Vec<_> = bits.iter().copied().map(half::f16::from_bits).collect();
        let bf16: Vec<_> = bits.iter().copied().map(half::bf16::from_bits).collect();
        compare(&Tensor::from_vec(f16, (512, 256), &device)?)?;
        compare(&Tensor::from_vec(bf16, (512, 256), &device)?)?;
        Ok(())
    }

    #[test]
    fn packed_glu_layouts_and_widths() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for width in [1, 7, 127, 128, 255, 257, 1024, 1152, 12288] {
                let data: Vec<_> = (0..5 * 2 * width)
                    .map(|i| ((i * 73 % 4093) as f32 - 2046.0) / 137.0)
                    .collect();
                let input = Tensor::from_vec(data, (5, 2 * width), &device)?.to_dtype(dtype)?;
                compare(&input)?;
                // Contiguous view with nonzero storage offset.
                compare(&input.narrow(0, 1, 3)?)?;
                // A packed tensor can be contiguous without 8-byte alignment.
                let padded = Tensor::zeros(17, dtype, &device)?;
                let flat = Tensor::cat(&[&padded, &input.flatten_all()?], 0)?;
                compare(&flat.narrow(0, 17, 10 * width)?.reshape((5, 2 * width))?)?;
                if width > 1 {
                    // Non-contiguous inputs use the original implementation.
                    compare(&input.narrow(1, 0, 2 * (width - 1))?)?;
                }
            }
        }
        Ok(())
    }
}
