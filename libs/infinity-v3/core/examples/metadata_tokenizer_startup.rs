//! Compare eager and lazy metadata workers with a local tokenizer and no weights.
use infinity_v3_core::tokenization::{EncodingInput, Tokenization};
use std::{
    error::Error,
    num::NonZeroUsize,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

fn status() -> Result<String, std::io::Error> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    Ok(status
        .lines()
        .filter(|line| line.starts_with("VmRSS:") || line.starts_with("Threads:"))
        .collect::<Vec<_>>()
        .join(" "))
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: metadata_tokenizer_startup TOKENIZER_JSON eager|lazy")?;
    let mode = args.next().ok_or("expected eager or lazy")?;
    let tokenizer = Tokenizer::from_file(path)?;
    let text = "hello café 東京";
    let expected = tokenizer.encode(text, true)?.get_ids().to_vec();
    println!("before_workers {}", status()?);
    let capacity = NonZeroUsize::new(1024).ok_or("invalid queue capacity")?;
    let started = Instant::now();
    let workers = match mode.as_str() {
        "eager" => Tokenization::with_queue_capacity_and_pooled_text_acceleration(
            16, capacity, tokenizer, 16000, 0, None, None, None, false,
        ),
        "lazy" => Tokenization::with_lazy_queue_capacity(
            16, capacity, tokenizer, 16000, 0, None, None, None,
        ),
        _ => return Err("expected eager or lazy".into()),
    };
    println!(
        "constructor_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        // Model an idle deployed server before its first metadata request.
        tokio::time::sleep(Duration::from_secs(1)).await;
        println!("idle {}", status()?);
        let started = Instant::now();
        let (_, actual) = workers
            .tokenize(EncodingInput::Single(text.into()), true, None)
            .await?;
        println!(
            "first_metadata_ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        if actual.get_ids() != expected {
            return Err("metadata token IDs differ from Hugging Face".into());
        }
        workers.shutdown().await;
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    })
}
