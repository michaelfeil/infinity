use std::error::Error;
use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "libtorch")]
    {
        let bridge = std::env::var("DEP_INFINITY_BACKEND_TORCH_LIB_DIR")?;
        if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{bridge}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
    }
    // Try to get the git sha from the local git repository
    if EmitBuilder::builder()
        .fail_on_error()
        .git_sha(false)
        .emit()
        .is_err()
    {
        // Unable to get the git sha
        if let Ok(sha) = std::env::var("GIT_SHA") {
            // Set it from an env var
            println!("cargo:rustc-env=VERGEN_GIT_SHA={sha}");
        }
    }

    // Set docker label if present
    if let Ok(label) = std::env::var("DOCKER_LABEL") {
        // Set it from an env var
        println!("cargo:rustc-env=DOCKER_LABEL={label}");
    }

    Ok(())
}
