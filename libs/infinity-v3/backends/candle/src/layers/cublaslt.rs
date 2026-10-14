use crate::layers::HiddenAct;
use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_cublaslt::{fused_batch_matmul, fused_matmul, Activation, CublasLt};

#[cfg(feature = "cuda")]
thread_local! {
    // A handle belongs to the tensor's device/stream and its native worker.
    // Never share GPU 0's handle across replicas.
    static CUBLASLT: std::cell::RefCell<std::collections::HashMap<
        candle::cuda_backend::DeviceId, CublasLtWrapper
    >> = std::cell::RefCell::new(std::collections::HashMap::new());
}

pub fn get_cublas_lt_wrapper(device: &Device) -> Result<Option<CublasLtWrapper>> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        return CUBLASLT.with(|handles| {
            let mut handles = handles.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(entry) = handles.entry(cuda.id()) {
                entry.insert(CublasLtWrapper {
                    cublaslt: CublasLt::new(device)?,
                });
            }
            Ok(handles.get(&cuda.id()).cloned())
        });
    }
    let _ = device;
    Ok(None)
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = match act {
                Some(HiddenAct::Gelu) => Some(Activation::Gelu),
                Some(HiddenAct::Relu) => Some(Activation::Relu),
                _ => None,
            };

            let mut result = fused_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(HiddenAct::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = match act {
                Some(HiddenAct::Gelu) => Some(Activation::Gelu),
                Some(HiddenAct::Relu) => Some(Activation::Relu),
                _ => None,
            };

            let mut result = fused_batch_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(HiddenAct::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }
}
