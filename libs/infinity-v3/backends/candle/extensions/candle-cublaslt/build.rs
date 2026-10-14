use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=kernels/fp8_quant.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_EXPERIMENTAL_FP8").is_none() {
        return;
    }
    let nvcc = env::var_os("NVCC").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env::var_os("CUDA_ROOT").unwrap_or_else(|| "/usr/local/cuda".into()))
            .join("bin/nvcc")
    });
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("fp8_quant.ptx");
    let status = Command::new(nvcc)
        .args([
            "--ptx",
            "-arch=compute_89",
            "-std=c++17",
            "-O3",
            "kernels/fp8_quant.cu",
            "-o",
        ])
        .arg(output)
        .status()
        .expect("experimental-fp8 requires nvcc");
    assert!(status.success(), "FP8 conversion PTX compilation failed");
}
