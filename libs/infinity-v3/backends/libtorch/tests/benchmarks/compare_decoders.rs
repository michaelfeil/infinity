#![cfg(feature = "benchmark-cuda")]
//! Compare native backends on the same packed decoder inputs; tokenization/HTTP are excluded.
use infinity_backend_libtorch::LibtorchBackend;
use std::{
    hint::black_box,
    path::PathBuf,
    sync::{Barrier, Mutex},
    time::Instant,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

fn batch(lengths: &[usize], vocab_size: u32, bos: Option<u32>, eos: Option<u32>) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative = vec![0];
    for (row, &length) in lengths.iter().enumerate() {
        input_ids.extend((0..length).map(|i| {
            (if i + 1 == length { eos } else { None })
                .or(if i == 0 { bos } else { None })
                .unwrap_or_else(|| {
                    // Synthetic, valid vocabulary IDs; this measures backend parity/latency,
                    // not retrieval quality or tokenizer throughput.
                    (1000 + (i * 17 + row * 31) as u32) % vocab_size
                })
        }));
        position_ids.extend((0..length).map(|i| i as u32));
        cumulative.push(input_ids.len() as u32);
    }
    Batch {
        token_type_ids: vec![0; input_ids.len()],
        input_ids,
        position_ids,
        cumulative_seq_lengths: cumulative,
        max_length: *lengths.iter().max().unwrap() as u32,
        pooled_indices: (0..lengths.len() as u32).collect(),
        raw_indices: vec![],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn measure(backend: &dyn Backend, input: &Batch, iterations: usize) -> Vec<f64> {
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let batch = input.clone(); // cloning excluded equally for both runtimes
        let start = Instant::now();
        black_box(backend.embed(batch).unwrap()); // returns host outputs: CUDA work is synchronized
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times
}
fn stats(mut times: Vec<f64>, tokens: usize) -> serde_json::Value {
    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    serde_json::json!({"p50_ms":p50,"p95_ms":times[(times.len() * 95 / 100).min(times.len()-1)],
        "min_ms":times[0],"tokens_per_second_at_p50":tokens as f64 * 1000.0 / p50})
}
fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect(
        "compare_decoders MODEL_PATH [ITERATIONS] [TORCH_GPU] [CANDLE_GPU] [DTYPE] [POOL]",
    ));
    let iterations = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(100);
    assert!(iterations >= 20);
    let torch_gpu = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(6);
    let candle_gpu = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(7);
    assert_ne!(
        torch_gpu, candle_gpu,
        "Parallel comparisons require separate GPUs"
    );
    let root_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json")).expect("checkpoint config.json"),
    )
    .unwrap();
    // Multimodal checkpoints retain their authoritative composite config;
    // text-only packed inputs use the nested text vocabulary and boundaries.
    let config = root_config
        .get("text_config")
        .filter(|value| value.is_object())
        .unwrap_or(&root_config);
    let vocab_size = config["vocab_size"].as_u64().expect("vocab_size") as u32;
    let token_id = |key: &str| {
        config[key].as_u64().map(|id| {
            assert!(id < vocab_size as u64, "{key} outside vocabulary");
            id as u32
        })
    };
    let bos = token_id("bos_token_id");
    let eos = token_id("eos_token_id");
    let pad = token_id("pad_token_id");
    let dtype = args.next().unwrap_or_else(|| {
        config["torch_dtype"]
            .as_str()
            .or_else(|| config["dtype"].as_str())
            .or_else(|| root_config["dtype"].as_str())
            .unwrap_or("float16")
            .to_owned()
    });
    assert!(matches!(dtype.as_str(), "float16" | "bfloat16" | "float32"));
    let pooling = args
        .next()
        .unwrap_or_else(|| match config["pooling"].as_str() {
            Some("avg" | "mean") => "mean".into(),
            _ => "last_token".into(),
        });
    let pool = match pooling.as_str() {
        "last_token" => Pool::LastToken,
        "mean" => Pool::Mean,
        "cls" => Pool::Cls,
        _ => panic!("POOL must be last_token, mean, or cls"),
    };
    let checkpoint_revision = std::fs::read(path.join("revision.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|meta| meta["sha"].as_str().map(str::to_owned));
    let fp8_dynamic = std::env::var("INFINITY_BENCHMARK_FP8")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    let started = Instant::now();
    let torch = LibtorchBackend::new_with_dense_paths_and_fp8(
        &path,
        &dtype,
        ModelType::Embedding(pool.clone()),
        &format!("cuda:{torch_gpu}"),
        None,
        fp8_dynamic,
    )
    .unwrap();
    let torch_load = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let candle = CandleBackend::new_with_fp8(
        &path,
        dtype.clone(),
        ModelType::Embedding(pool.clone()),
        None,
        candle_gpu,
        fp8_dynamic,
    )
    .unwrap();
    let candle_load = started.elapsed().as_secs_f64();
    let torch = Mutex::new(torch);
    let candle = Mutex::new(candle);
    let long_context = std::env::var("INFINITY_BENCHMARK_LONG_CONTEXT").unwrap_or_default();
    let long_requested = matches!(long_context.as_str(), "1" | "true" | "only");
    let context_limit = config["max_position_embeddings"].as_u64();
    if long_requested {
        assert!(context_limit.is_some_and(|limit| limit >= 2048),
            "Long-context workloads require a checkpoint declaring at least 2048 positions; the checkpoint is never modified");
    }
    let mut workloads = vec![
        ("1x32", vec![32]),
        ("1x128", vec![128]),
        ("8x128", vec![128; 8]),
        ("32x128", vec![128; 32]),
        (
            "32 ragged 32..256",
            (0..32).map(|i| 32 + i % 8 * 32).collect(),
        ),
        ("8x512", vec![512; 8]),
        ("32x512", vec![512; 32]),
    ];
    if long_requested {
        if long_context == "only" {
            workloads.clear();
        }
        workloads.extend([
            ("1x1024", vec![1024]),
            ("1x2048", vec![2048]),
            ("8x1024", vec![1024; 8]),
            ("8x2048", vec![2048; 8]),
            ("ragged 1024/1536/2048/1280", vec![1024, 1536, 2048, 1280]),
        ]);
    }
    let mut rows = Vec::new();
    for (label, lengths) in workloads {
        let input = batch(&lengths, vocab_size, bos, eos);
        let t = torch.lock().unwrap().embed(input.clone()).unwrap();
        let c = candle.lock().unwrap().embed(input.clone()).unwrap();
        let mut min_cosine = 1.0_f64;
        let mut max_abs = 0.0_f32;
        for index in 0..lengths.len() {
            let (Embedding::Pooled(t), Embedding::Pooled(c)) = (&t[&index], &c[&index]) else {
                panic!("pooled output required")
            };
            assert_eq!(t.len(), c.len());
            let mut dot = 0.;
            let mut tn = 0.;
            let mut cn = 0.;
            for (&a, &b) in t.iter().zip(c) {
                assert!(a.is_finite() && b.is_finite());
                dot += a as f64 * b as f64;
                tn += (a as f64).powi(2);
                cn += (b as f64).powi(2);
                max_abs = max_abs.max((a - b).abs());
            }
            min_cosine = min_cosine.min(dot / (tn * cn).sqrt());
        }
        assert!(min_cosine > 0.999, "{label} output cosine {min_cosine}");
        let barrier = Barrier::new(2);
        let (torch_times, candle_times) = std::thread::scope(|scope| {
            let torch_thread = scope.spawn(|| {
                let model = torch.lock().unwrap();
                for _ in 0..20 {
                    black_box(model.embed(input.clone()).unwrap());
                }
                barrier.wait();
                measure(&*model, &input, iterations)
            });
            let candle_thread = scope.spawn(|| {
                let model = candle.lock().unwrap();
                for _ in 0..20 {
                    black_box(model.embed(input.clone()).unwrap());
                }
                barrier.wait();
                measure(&*model, &input, iterations)
            });
            (torch_thread.join().unwrap(), candle_thread.join().unwrap())
        });
        let torch_stats = stats(torch_times, input.input_ids.len());
        let candle_stats = stats(candle_times, input.input_ids.len());
        let ratio =
            candle_stats["p50_ms"].as_f64().unwrap() / torch_stats["p50_ms"].as_f64().unwrap();
        eprintln!(
            "{label}: torch {:.3}ms, candle {:.3}ms, torch speedup {:.2}x, cosine {:.6}",
            torch_stats["p50_ms"].as_f64().unwrap(),
            candle_stats["p50_ms"].as_f64().unwrap(),
            ratio,
            min_cosine
        );
        rows.push(
            serde_json::json!({"workload":label,"lengths":lengths,"tokens":input.input_ids.len(),
            "torch":torch_stats,"candle":candle_stats,"torch_speedup":ratio,
            "min_output_cosine":min_cosine,"max_output_absolute_difference":max_abs}),
        );
    }
    let mut report = serde_json::json!({"model_path":path,"dtype":dtype,"checkpoint_revision":checkpoint_revision,"vocab_size":vocab_size,"bos_token_id":bos,"eos_token_id":eos,"pad_token_id":pad,"input_pattern":"synthetic valid vocabulary IDs with configured BOS/EOS; no padding",
        "pooling":pooling,"iterations_per_backend":iterations,"torch_gpu":torch_gpu,"candle_gpu":candle_gpu,"torch_cuda_graph_count":torch.lock().unwrap().cuda_graph_count(),"torch_cuda_graph_max_tokens":std::env::var("INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS").unwrap_or_else(|_| "4096".into()), "torch_cudnn_varlen_requested":std::env::var("INFINITY_TORCH_CUDNN_VARLEN").is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),"torch_cuda_graphs_requested":std::env::var("INFINITY_TORCH_CUDA_GRAPHS").is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),"measurement_mode":"parallel separate GPUs","warmup_per_backend":20,
        "torch_load_seconds":torch_load,"candle_load_seconds":candle_load,"workloads":rows});
    if long_requested {
        report["long_context_mode"] = long_context.into();
        report["checkpoint_max_position_embeddings"] = serde_json::json!(context_limit);
    }
    if fp8_dynamic {
        report["dynamic_fp8_requested"] = true.into();
        report["dynamic_fp8_recipe"] = serde_json::json!({
            "weights": "MLP-only E4M3, per-output-channel FP32 scales, quantized once",
            "activations": "per-actual-token E4M3; FP32 absmax floored at 1e-12 times FP32 reciprocal of 448; existing model-dtype activation before quantization",
            "gemm": "FP32 accumulation, model-dtype output; original recipe fast-accumulation where supported, full-accumulation fallback",
            "unchanged_precision": "attention, normalization, residuals, embeddings and output heads",
            "token_padding": false,
            "checkpoint_supplied_quantization_scales": false,
            "comparison": "both backends use dynamic FP8; the existing finite-output and strict pooled cosine gates are unchanged"
        });
    }
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
