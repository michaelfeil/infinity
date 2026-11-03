//! CPU-only startup profiling with a local tokenizer, without downloading weights.
//! Run with `cargo run --release -p infinity_v3-core --no-default-features
//! --example tokenizer_startup -- /path/to/tokenizer.json`.
use std::{error::Error, time::Instant};
use tokenizers::Tokenizer;

fn report(stage: &str, started: Instant) {
    println!("{stage}_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: tokenizer_startup /path/to/tokenizer.json")?;
    let started = Instant::now();
    let mut hf = Tokenizer::from_file(path)?;
    hf.with_padding(None).with_truncation(None)?;
    report("hf_load", started);

    let started = Instant::now();
    let mut json = serde_json::to_value(&hf)?;
    // Match the server: HF applies request-specific post-processing separately.
    json["post_processor"] = serde_json::Value::Null;
    report("serialize", started);

    let started = Instant::now();
    let fast = basetenkenizer::Tokenizer::from_json(json)?;
    report("baseten_build", started);

    let long = "hello world café 東京 e\u{301} 👩🏽‍💻\n".repeat(2048);
    for text in ["hello world", "café 東京 e\u{301} 👩🏽‍💻", long.as_str()] {
        let actual = fast.encode(text)?;
        let expected = hf.encode(text, false)?;
        if actual != expected.get_ids() {
            return Err("Basetenkenizer IDs differ from Hugging Face IDs".into());
        }
    }
    println!("id_parity=ok");

    let started = Instant::now();
    let clones: Vec<_> = (0..16).map(|_| hf.clone()).collect();
    report("hf_clones_16", started);
    std::hint::black_box(clones);
    Ok(())
}
