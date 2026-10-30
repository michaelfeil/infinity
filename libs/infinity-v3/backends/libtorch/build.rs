use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=INFINITY_FA4_ROOT");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=cpp/vendor/torch_flash_attention");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma.cpp");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma_64.cu");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma_72.cu");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma_128.cu");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma_256.cu");
    println!("cargo:rerun-if-changed=cpp/kernels/torch_flash_fma_512.cu");
    println!("cargo:rerun-if-env-changed=CUTLASS_ROOT");
    println!("cargo:rerun-if-env-changed=ENV_VAR_BASE");
    let torch = PathBuf::from(env::var_os("LIBTORCH").expect(
        "Set LIBTORCH to the root of a LibTorch 2.14.1 C++ distribution (include/, lib/, share/)",
    ));
    let mut config = cmake::Config::new("cpp");
    config.define("CMAKE_PREFIX_PATH", &torch);
    config.define("ENV_VAR_BASE", infinity_v3_config::ENV_VAR_BASE);
    // Distribution builds must not silently target only the build host's GPU.
    let architectures =
        env::var_os("LIBTORCH_CUDA_ARCHITECTURES").unwrap_or_else(|| "80;86;89;90;100;120".into());
    config.define("CMAKE_CUDA_ARCHITECTURES", architectures);
    let fa4_root = env::var_os("INFINITY_FA4_ROOT").unwrap_or_default();
    if !fa4_root.is_empty() {
        println!(
            "cargo:rerun-if-changed={}",
            PathBuf::from(&fa4_root).display()
        );
    }
    // Clear a previous CMake cache entry when the caller removes the opt-in.
    config.define("INFINITY_FA4_ROOT", fa4_root);
    let dst = config.build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=tei_torch");
    println!("cargo:lib_dir={}/lib", dst.display());
    // The bridge carries an rpath to LibTorch. Executables also need to find the bridge.
    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", dst.display());
    }
}
