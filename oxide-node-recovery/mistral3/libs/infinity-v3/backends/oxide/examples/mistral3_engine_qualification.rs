// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root-only qualification example for the full Mistral3/Pixtral engine.
//! It refuses to run without `INFINITY_MISTRAL3_RUN_GPU=1`. The CPU author
//! never runs this example; it exercises real prepared weights, the shared text
//! decoder and the optional score head instead of a mock sink.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("INFINITY_MISTRAL3_RUN_GPU").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!("set INFINITY_MISTRAL3_RUN_GPU=1 (Root GPU only)");
        return Ok(());
    }
    let model = std::env::var_os("INFINITY_MISTRAL3_MODEL")
        .ok_or("set INFINITY_MISTRAL3_MODEL to the cached checkpoint")?;
    let ordinal: usize = std::env::var("INFINITY_MISTRAL3_DEVICE")
        .unwrap_or_else(|_| "0".to_owned())
        .parse()?;
    let fa4 = std::env::var_os("INFINITY_FA4_ROOT")
        .ok_or("set INFINITY_FA4_ROOT to the compiled ABI7 provider")?;
    let max_tokens: usize = std::env::var("INFINITY_MISTRAL3_MAX_TOKENS")
        .unwrap_or_else(|_| "4096".to_owned())
        .parse()?;
    let max_sequences: usize = std::env::var("INFINITY_MISTRAL3_MAX_SEQUENCES")
        .unwrap_or_else(|_| "8".to_owned())
        .parse()?;
    let max_images: usize = std::env::var("INFINITY_MISTRAL3_MAX_IMAGES")
        .unwrap_or_else(|_| "4".to_owned())
        .parse()?;
    let mut engine = infinity_backend_oxide::Mistral3Qualification::open(
        std::path::Path::new(&model),
        ordinal,
        std::path::Path::new(&fa4),
        max_tokens,
        max_sequences,
        max_images,
    )?;
    let run = engine.run_text_lengths(&[3, 5, 2])?;
    println!(
        "mistral3-engine pooled_floats={} raw_floats={} logits={}",
        run.pooled.len(),
        run.raw.len(),
        run.logits.len()
    );
    Ok(())
}
