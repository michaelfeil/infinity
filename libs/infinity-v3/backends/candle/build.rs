use anyhow::{bail, Context, Result};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(gemma4_moe_cuda)");
    println!("cargo:rustc-check-cfg=cfg(gemma4_moe_hopper)");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if let Ok(compute_cap) = set_compute_cap() {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}");
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/kernels/gemma4_moe.cu");
        println!("cargo:rerun-if-changed=src/kernels/gemma4_moe_kernels.cuh");
        println!(
            "cargo:rerun-if-changed=extensions/candle-gemma4-moe/kernels/grouped_gemm_hopper.cu"
        );
        println!("cargo:rerun-if-changed=extensions/candle-gemma4-moe/kernels/grouped_gemm.cu");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // Routed BF16 experts need Ampere Tensor Cores. Keep older CUDA
        // targets buildable for the models they already support.
        if set_compute_cap().expect("CUDA compute capability") >= 80 {
            cudaforge::KernelBuilder::new()
                .source_files(["extensions/candle-gemma4-moe/kernels/grouped_gemm.cu"])
                .with_cutlass(Some("e406c186f510a15091cce01f782020ceb7ba8eb5"))
                .arg("-std=c++17")
                .arg("-O3")
                .arg("--expt-relaxed-constexpr")
                .build_lib(out.join("libgemma4_moe.a"))
                .expect("compile Gemma4 grouped MoE kernels");
            println!("cargo:rustc-link-search=native={}", out.display());
            println!("cargo:rustc-link-lib=static=gemma4_moe");
            // SM90a instructions are compiled only into Hopper-targeted builds.
            // Other targets retain the portable grouped GEMM implementation.
            if set_compute_cap().expect("CUDA compute capability") == 90 {
                cudaforge::KernelBuilder::new()
                    .source_files(["extensions/candle-gemma4-moe/kernels/grouped_gemm_hopper.cu"])
                    .with_compute_override_arch("grouped_gemm_hopper.cu", "90a")
                    // CUTLASS device assertions otherwise serialize WGMMA instructions.
                    .arg("-DNDEBUG")
                    .with_cutlass(Some("e406c186f510a15091cce01f782020ceb7ba8eb5"))
                    .arg("-std=c++17")
                    .arg("-O3")
                    .arg("--expt-relaxed-constexpr")
                    .build_lib(out.join("libgemma4_moe_hopper.a"))
                    .expect("compile Hopper Gemma4 grouped MoE kernels");
                println!("cargo:rustc-link-lib=static=gemma4_moe_hopper");
                println!("cargo:rustc-cfg=gemma4_moe_hopper");
            }
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-cfg=gemma4_moe_cuda");
        }
        println!("cargo:rerun-if-changed=src/pooling_kernels/mean_pool.cu");
        cudaforge::KernelBuilder::new()
            .source_dir("src/pooling_kernels")
            .arg("-std=c++17")
            .arg("-O3")
            .build_ptx()
            .expect("compile pooling kernels")
            .write(
                std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("pooling_ptx.rs"),
            )
            .expect("write pooling PTX bindings");
        println!("cargo:rerun-if-changed=src/kernels/gated_activation.cu");
        let bindings = cudaforge::KernelBuilder::new()
            .source_files(["src/kernels/gated_activation.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--expt-relaxed-constexpr")
            .build_ptx()
            .expect("compile activation kernels");
        bindings
            .write(
                std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                    .join("activation_ptx.rs"),
            )
            .expect("write activation PTX bindings");
        println!("cargo:rerun-if-changed=src/kernels/qk_norm_rope.cu");
        println!("cargo:rerun-if-changed=extensions/candle-layer-norm/kernels");
        cudaforge::KernelBuilder::new()
            .source_files(["src/kernels/qk_norm_rope.cu"])
            .include_path("extensions/candle-layer-norm/kernels")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .build_ptx()
            .expect("compile Q/K normalization and RoPE kernel")
            .write(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("qk_ptx.rs"))
            .expect("write Q/K PTX bindings");
        println!("cargo:rerun-if-changed=src/kernels/qwen35_gdn.cu");
        cudaforge::KernelBuilder::new()
            .source_files(["src/kernels/qwen35_gdn.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--fmad=false")
            .build_ptx()
            .expect("compile Qwen3.5 Gated DeltaNet")
            .write(out.join("qwen35_gdn_ptx.rs"))
            .expect("write Gated DeltaNet PTX");
        println!("cargo:rerun-if-changed=src/kernels/gemma_rms_norm.cu");
        cudaforge::KernelBuilder::new()
            .source_files(["src/kernels/gemma_rms_norm.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--fmad=false")
            .build_ptx()
            .expect("compile Gemma RMSNorm")
            .write(
                std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                    .join("gemma_norm_ptx.rs"),
            )
            .expect("write Gemma RMSNorm PTX bindings");
        println!("cargo:rerun-if-changed=src/kernels/residual_add.cu");
        cudaforge::KernelBuilder::new()
            .source_files(["src/kernels/residual_add.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--expt-relaxed-constexpr")
            .build_ptx()
            .expect("compile residual addition kernel")
            .write(
                std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("residual_ptx.rs"),
            )
            .expect("write residual addition PTX bindings");
    }
}

fn set_compute_cap() -> Result<usize> {
    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        compute_cap_str
            .parse::<usize>()
            .context("Could not parse code")?
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .context("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not a utf8 string")?;
        let mut lines = out.lines();
        if lines.next().context("missing line in stdout")? != "compute_cap" {
            bail!("First line should be `compute_cap`");
        }
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        cap.parse::<usize>().context("cannot parse as int")?
    };
    Ok(compute_cap)
}
