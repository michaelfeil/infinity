//! H100 development benchmark: native FP16 versus dynamic-row FP8 linear.
//! Wall time includes Candle submission/allocation overhead; no CUDA graphs.
use candle::{DType, Device, Result, Tensor};
use candle_cublaslt::{
    fp8::{Fp8Linear, Fp8Matmul},
    fused_matmul, CublasLt,
};
use std::time::Instant;
fn measure(dev: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    let stream = dev.as_cuda_device()?.cuda_stream();
    for _ in 0..10 {
        let _ = f()?;
    }
    stream.synchronize().map_err(candle::Error::wrap)?;
    let start = Instant::now();
    for _ in 0..50 {
        let _ = f()?;
    }
    stream.synchronize().map_err(candle::Error::wrap)?;
    Ok(start.elapsed().as_secs_f64() * 1000. / 50.)
}
fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let blas = CublasLt::new(&dev)?;
    let mut executor = Fp8Matmul::new(&dev)?;
    for (m, n, k) in [
        (128, 3072, 1024),
        (4096, 3072, 1024),
        (4096, 1024, 3072),
        (128, 12288, 4096),
        (4096, 12288, 4096),
        (4096, 4096, 12288),
    ] {
        let x = Tensor::randn(0f32, 0.5f32, (m, k), &dev)?.to_dtype(DType::F16)?;
        let w = Tensor::randn(0f32, 0.02f32, (n, k), &dev)?.to_dtype(DType::F16)?;
        let layer = Fp8Linear::new(&w)?;
        let mut fp16 = Vec::new();
        let mut fp8 = Vec::new();
        for round in 0..3 {
            if round % 2 == 0 {
                fp16.push(measure(&dev, || {
                    fused_matmul(&w, &x, None, None, None, None, None, blas.clone())
                })?);
                fp8.push(measure(&dev, || layer.forward(&x, &mut executor))?);
            } else {
                fp8.push(measure(&dev, || layer.forward(&x, &mut executor))?);
                fp16.push(measure(&dev, || {
                    fused_matmul(&w, &x, None, None, None, None, None, blas.clone())
                })?);
            }
        }
        fp16.sort_by(f64::total_cmp);
        fp8.sort_by(f64::total_cmp);
        println!("{{\"m\":{m},\"n\":{n},\"k\":{k},\"fp16_ms\":{},\"fp8_ms\":{},\"throughput_ratio\":{}}}",fp16[1],fp8[1],fp16[1]/fp8[1]);
    }
    Ok(())
}
