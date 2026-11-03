//! Time first-party tokenization worker startup with a local tokenizer JSON.
//!
//! Run with:
//! `cargo run --release -p infinity_v3-core --no-default-features
//! --example tokenizer_startup -- /path/to/tokenizer.json [embedding|classification] [steady]`
use infinity_v3_core::tokenization::{EncodingInput, Tokenization};
use std::{
    error::Error,
    num::NonZeroUsize,
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, TruncationDirection};

type ExampleError = Box<dyn Error + Send + Sync>;

async fn encode(
    tokenization: &Tokenization,
    mode: &str,
    single: &str,
    pair: (&str, &str),
    paired: bool,
) -> Result<Vec<u32>, ExampleError> {
    let input = || {
        if paired {
            EncodingInput::Dual(pair.0.into(), pair.1.into())
        } else {
            EncodingInput::Single(single.into())
        }
    };
    let encoding = match mode {
        "embedding" => {
            tokenization
                .encode_embedding(input(), false, TruncationDirection::Right, None)
                .await?
        }
        "classification" => {
            tokenization
                .encode_classification(input(), false, TruncationDirection::Right)
                .await?
        }
        _ => return Err("mode must be embedding or classification".into()),
    };
    Ok(encoding.input_ids)
}

fn expected_ids(
    tokenizer: &Tokenizer,
    single: &str,
    pair: (&str, &str),
    paired: bool,
) -> Result<Vec<u32>, ExampleError> {
    let encoding = if paired {
        tokenizer.encode(pair, true)?
    } else {
        tokenizer.encode(single, true)?
    };
    Ok(encoding.get_ids().to_vec())
}

fn check_parity(actual: &[u32], expected: &[u32]) -> Result<(), ExampleError> {
    if actual != expected {
        return Err("token IDs differ from Hugging Face".into());
    }
    Ok(())
}

fn main() -> Result<(), ExampleError> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: tokenizer_startup TOKENIZER_JSON [embedding|classification] [steady]")?;
    let mode = args.next().unwrap_or_else(|| "embedding".to_owned());
    if !matches!(mode.as_str(), "embedding" | "classification") {
        return Err("mode must be embedding or classification".into());
    }
    let steady = match args.next().as_deref() {
        None => false,
        Some("steady") => true,
        Some(_) => return Err("expected embedding, classification, or steady".into()),
    };
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let tokenizer = Tokenizer::from_file(path)?;
    let single = "hello world café 東京 e\u{301} 👩🏽‍💻";
    let pair = ("query 東京", "a short document 👩🏽‍💻");
    let workers = Tokenization::with_queue_capacity(
        1,
        NonZeroUsize::new(8).ok_or("queue capacity must be non-zero")?,
        tokenizer.clone(),
        4096,
        0,
        None,
        None,
        None,
    );

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let started = Instant::now();
        let first = encode(&workers, &mode, single, pair, false).await?;
        println!(
            "first_request_ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        check_parity(&first, &expected_ids(&tokenizer, single, pair, false)?)?;

        let first_pair = encode(&workers, &mode, single, pair, true).await?;
        check_parity(&first_pair, &expected_ids(&tokenizer, single, pair, true)?)?;
        println!("hf_id_parity=ok");

        if steady {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let requests = 1_000;
            let started = Instant::now();
            for index in 0..requests {
                let ids = encode(&workers, &mode, single, pair, index % 2 == 0).await?;
                std::hint::black_box(ids);
            }
            let elapsed = started.elapsed();
            println!(
                "steady_requests={requests} steady_total_ms={:.3} steady_avg_us={:.3}",
                elapsed.as_secs_f64() * 1000.0,
                elapsed.as_secs_f64() * 1_000_000.0 / requests as f64
            );
        }

        workers.shutdown().await;
        println!("shutdown=ok");
        Ok::<(), ExampleError>(())
    })?;
    Ok(())
}
