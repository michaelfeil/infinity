use super::Linear;
use candle::{Result, Tensor};

/// Only MLP projections opt into quantization. Attention and output heads retain
/// the existing precision and implementation.
pub(crate) enum MlpLinear {
    Dense(Linear),
    #[cfg(feature = "experimental-fp8")]
    Fp8 {
        linear: candle_cublaslt::fp8::Fp8Linear,
        bias: Option<Tensor>,
        act: Option<super::HiddenAct>,
    },
}

#[cfg(feature = "experimental-fp8")]
thread_local! {
    // Model weights can move from the loader to a backend worker. cuBLASLt
    // descriptors and mutable workspace stay on the executing thread instead.
    // One executor per thread bounds workspace usage independently of layer count.
    static FP8_EXECUTOR: std::cell::RefCell<Option<(candle::Device, candle_cublaslt::fp8::Fp8Matmul)>> = const { std::cell::RefCell::new(None) };
}

impl MlpLinear {
    pub(crate) fn new(weight: Tensor, enable_fp8_dynamic: bool) -> Result<Self> {
        Self::with_bias_activation(weight, None, None, enable_fp8_dynamic)
    }

    pub(crate) fn with_bias_activation(
        weight: Tensor,
        bias: Option<Tensor>,
        act: Option<super::HiddenAct>,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        if enable_fp8_dynamic {
            #[cfg(feature = "experimental-fp8")]
            return Ok(Self::Fp8 {
                linear: candle_cublaslt::fp8::Fp8Linear::new(&weight)?,
                bias,
                act,
            });
            #[cfg(not(feature = "experimental-fp8"))]
            candle::bail!("Dynamic FP8 requires an experimental-fp8 build");
        }
        Ok(Self::Dense(Linear::new(weight, bias, act)))
    }

    pub(crate) fn forward_gated(&self, x: &Tensor, act: &super::HiddenAct) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(&super::gated_activation(x, Some(act))?),
            #[cfg(feature = "experimental-fp8")]
            Self::Fp8 {
                linear,
                bias,
                act: output_act,
            } => {
                if matches!(act, super::HiddenAct::Silu) && bias.is_none() && output_act.is_none() {
                    if let Some(y) =
                        with_fp8_executor(x, |executor| linear.forward_packed_swiglu(x, executor))?
                    {
                        return Ok(y);
                    }
                }
                let activated = super::gated_activation(x, Some(act))?;
                self.forward(&activated)
            }
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            #[cfg(feature = "experimental-fp8")]
            Self::Fp8 { linear, bias, act } => {
                let y = with_fp8_executor(x, |executor| linear.forward(x, executor))?;
                let y = match bias {
                    Some(bias) => y.broadcast_add(bias)?,
                    None => y,
                };
                match act {
                    Some(act) => act.forward(&y),
                    None => Ok(y),
                }
            }
        }
    }
}

#[cfg(feature = "experimental-fp8")]
fn with_fp8_executor<T>(
    x: &Tensor,
    f: impl FnOnce(&mut candle_cublaslt::fp8::Fp8Matmul) -> Result<T>,
) -> Result<T> {
    FP8_EXECUTOR.with(|slot| {
        let mut slot = slot.try_borrow_mut().map_err(candle::Error::wrap)?;
        if slot
            .as_ref()
            .is_none_or(|(device, _)| !device.same_device(x.device()))
        {
            *slot = Some((
                x.device().clone(),
                candle_cublaslt::fp8::Fp8Matmul::new(x.device())?,
            ));
        }
        let (_, executor) = slot.as_mut().expect("executor initialized above");
        f(executor)
    })
}

#[cfg(all(test, feature = "experimental-fp8"))]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    #[ignore = "requires a Hopper GPU"]
    fn fp8_bias_activation_matches_dense_mlp() -> Result<()> {
        use super::super::HiddenAct;
        let device = Device::new_cuda(0)?;
        // Rectangular projections and odd token counts exercise encoder MLP
        // expansion/contraction and the dynamic activation scaling tail.
        for dtype in [DType::F16, DType::BF16] {
            for (input, output) in [(128, 256), (256, 128)] {
                // Deterministic broad-spectrum inputs avoid measuring relative
                // error against an almost-zero, cancelling sinusoidal product.
                let mut seed = 42u32;
                let mut samples = |count: usize, scale: f32| -> Vec<f32> {
                    (0..count)
                        .map(|_| {
                            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                            ((seed >> 8) as f32 / 16777216. * 2. - 1.) * scale
                        })
                        .collect()
                };
                let weight =
                    Tensor::from_vec(samples(input * output, 0.05), (output, input), &device)?
                        .to_dtype(dtype)?;
                let bias = Tensor::arange(0f32, output as f32, &device)?
                    .cos()?
                    .affine(0.1, 0.)?
                    .to_dtype(dtype)?;
                for act in [None, Some(HiddenAct::Gelu), Some(HiddenAct::Relu)] {
                    let dense = Linear::new(weight.clone(), Some(bias.clone()), act.clone());
                    let disabled = MlpLinear::with_bias_activation(
                        weight.clone(),
                        Some(bias.clone()),
                        act.clone(),
                        false,
                    )?;
                    let fp8 = MlpLinear::with_bias_activation(
                        weight.clone(),
                        Some(bias.clone()),
                        act.clone(),
                        true,
                    )?;
                    for rows in [1, 13, 129] {
                        let x =
                            Tensor::from_vec(samples(rows * input, 1.), (rows, input), &device)?
                                .to_dtype(dtype)?;
                        let values =
                            |t: Tensor| t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>();
                        let expected = values(dense.forward(&x)?)?;
                        assert_eq!(values(disabled.forward(&x)?)?, expected);
                        let actual = values(fp8.forward(&x)?)?;
                        assert!(actual.iter().all(|v| v.is_finite()));
                        let error: f32 = actual
                            .iter()
                            .zip(&expected)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        let energy: f32 = expected.iter().map(|v| v * v).sum();
                        assert!(
                            (error / energy.max(1e-12)).sqrt() < 0.08,
                            "FP8 MLP error: {dtype:?}, {input}->{output}, rows={rows}, act={act:?}, relative_rmse={}", (error/energy.max(1e-12)).sqrt()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn disabled_fp8_preserves_dense_outputs_without_executor() -> Result<()> {
        // A fresh thread isolates the lazy executor from other GPU tests.
        std::thread::spawn(|| -> Result<()> {
            let device = Device::new_cuda(0)?;
            let weight = Tensor::arange(0f32, 256f32, &device)?
                .reshape((16, 16))?
                .affine(0.001, -0.1)?
                .to_dtype(DType::F16)?;
            let dense = Linear::new(weight.clone(), None, None);
            let disabled = MlpLinear::new(weight, false)?;
            assert!(matches!(&disabled, MlpLinear::Dense(_)));
            let x = Tensor::arange(0f32, 64f32, &device)?
                .reshape((4, 16))?
                .affine(0.01, -0.2)?
                .to_dtype(DType::F16)?;
            let values = |t: Tensor| t.to_dtype(DType::F32)?.to_vec2::<f32>();
            assert_eq!(values(disabled.forward(&x)?)?, values(dense.forward(&x)?)?);
            let packed = Tensor::cat(&[&x, &x], 1)?;
            let act = super::super::HiddenAct::Silu;
            assert_eq!(
                values(disabled.forward_gated(&packed, &act)?)?,
                values(dense.forward(&super::super::gated_activation(&packed, Some(&act))?)?)?
            );
            FP8_EXECUTOR.with(|slot| assert!(slot.borrow().is_none()));
            Ok(())
        })
        .join()
        .expect("disabled FP8 test thread panicked")
    }
}
