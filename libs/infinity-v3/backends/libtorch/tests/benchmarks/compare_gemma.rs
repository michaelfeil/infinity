//! Compare native backends on identical packed Gemma embeddings; HTTP/tokenization excluded.
use infinity_backend_libtorch::LibtorchBackend;
use std::{
    hint::black_box,
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
    time::Instant,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{
    AudioFeatures, Backend, Batch, Embedding, ImagePatches, ModelType, MultimodalEncoding, Pool,
};

fn batch(lengths: &[usize]) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative = vec![0];
    for (row, &length) in lengths.iter().enumerate() {
        input_ids.extend((0..length).map(|i| {
            if i == 0 {
                101
            } else if i + 1 == length {
                102
            } else {
                (1000 + (i * 17 + row * 31) % 20000) as u32
            }
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
    let path = PathBuf::from(
        args.next()
            .expect("compare_gemma MODEL_PATH [ITERATIONS] [TORCH_GPU] [CANDLE_GPU]"),
    );
    let iterations = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(100);
    assert!(iterations >= 20);
    let torch_gpu = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(0);
    let candle_gpu = args
        .next()
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(1);
    assert_ne!(
        torch_gpu, candle_gpu,
        "Parallel comparisons require separate GPUs"
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let qwen_vl = config["model_type"] == "qwen3_vl";
    let qwen_media =
        qwen_vl || config["model_type"] == "qwen3_5" || config["model_type"] == "qwen3_5_moe";
    let override_dtype = std::env::var("INFINITY_COMPARE_DTYPE").ok();
    let dtype = override_dtype
        .as_deref()
        .unwrap_or(if qwen_vl { "float16" } else { "bfloat16" });
    assert!(
        matches!(dtype, "float16" | "bfloat16"),
        "INFINITY_COMPARE_DTYPE must be float16 or bfloat16"
    );
    let dense_paths = std::fs::read(path.join("modules.json")).ok().map(|bytes| {
        serde_json::from_slice::<Vec<serde_json::Value>>(&bytes)
            .unwrap()
            .into_iter()
            .filter(|module| {
                module["type"]
                    .as_str()
                    .is_some_and(|name| name.rsplit('.').next() == Some("Dense"))
            })
            .map(|module| module["path"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    });
    let pool = if qwen_vl { Pool::LastToken } else { Pool::Mean };
    let started = Instant::now();
    let torch = LibtorchBackend::new(
        &path,
        dtype,
        ModelType::Embedding(pool.clone()),
        &format!("cuda:{torch_gpu}"),
    )
    .unwrap();
    let torch_load = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let candle = CandleBackend::new(
        &path,
        dtype.into(),
        ModelType::Embedding(pool),
        dense_paths.clone(),
        candle_gpu,
    )
    .unwrap();
    let candle_load = started.elapsed().as_secs_f64();
    let torch = Mutex::new(torch);
    let candle = Mutex::new(candle);
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
    if std::env::var_os("INFINITY_GEMMA_TEXT_ONLY").is_none() && config["vision_config"].is_object()
    {
        workloads.push(("image1x32", vec![32]));
        if qwen_media {
            workloads.push(("images2ragged32_48", vec![32, 48]));
        }
    }
    if std::env::var_os("INFINITY_GEMMA_TEXT_ONLY").is_none() && config["audio_config"].is_object()
    {
        workloads.push(("audio1x32", vec![32]));
    }
    if let Ok(selected) = std::env::var("INFINITY_GEMMA_CASE") {
        workloads.retain(|(label, _)| *label == selected);
    }
    let mut rows = Vec::new();
    for (label, lengths) in workloads {
        let mut input = batch(&lengths);
        if label.starts_with("image") || label.starts_with("audio") {
            for (sequence, &length) in lengths.iter().enumerate() {
                let offset = input.cumulative_seq_lengths[sequence] as usize;
                let mut media = MultimodalEncoding {
                    images: vec![],
                    audios: vec![],
                    position_ids: std::array::from_fn(|_| (0..length as u32).collect()),
                    memory: None,
                };
                if label.starts_with("image") {
                    let (grid, patch_dim, merge_size) = if qwen_media {
                        let vision = &config["vision_config"];
                        let patch = vision["patch_size"].as_u64().unwrap() as usize;
                        let temporal = vision["temporal_patch_size"].as_u64().unwrap() as usize;
                        let channels = vision["in_channels"].as_u64().unwrap() as usize;
                        (
                            [1, if sequence == 0 { 4 } else { 2 }, 4],
                            channels * temporal * patch * patch,
                            2,
                        )
                    } else {
                        ([1, 3, 3], 768, 3)
                    };
                    media.images.push((
                        2,
                        Arc::new(ImagePatches {
                            pixels: (0..grid.iter().product::<usize>() * patch_dim)
                                .map(|i| ((i + sequence * 19) % 251) as f32 / 250.)
                                .collect(),
                            grid_thw: grid,
                            patch_dim,
                            merge_size,
                        }),
                    ));
                    if qwen_media {
                        let visual_tokens =
                            grid.iter().product::<usize>() / (merge_size * merge_size);
                        for i in 0..visual_tokens {
                            input.input_ids[offset + 2 + i] =
                                config["image_token_id"].as_u64().unwrap() as u32;
                            media.position_ids[0][2 + i] = 2;
                            media.position_ids[1][2 + i] = (2 + i / 2) as u32;
                            media.position_ids[2][2 + i] = (2 + i % 2) as u32;
                        }
                        for positions in &mut media.position_ids {
                            for (i, position) in
                                positions.iter_mut().enumerate().skip(2 + visual_tokens)
                            {
                                *position = (i - visual_tokens + 2) as u32;
                            }
                        }
                    }
                } else {
                    media.audios.push((
                        2,
                        Arc::new(AudioFeatures {
                            values: (0..64 * 128)
                                .map(|i| ((i % 211) as f32 * 0.017).sin())
                                .collect(),
                            mask: vec![1; 64],
                            feature_size: 128,
                        }),
                    ));
                }
                input.multimodal.push(Some(Arc::new(media)));
            }
        }
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
        eprintln!("{label} parity: cosine={min_cosine:.8}, maxabs={max_abs:.6}");
        assert!(min_cosine > 0.99, "{label} output cosine {min_cosine}");
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
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"model_path":path,"dtype":dtype,"checkpoint_dtype":config["dtype"],"model_type":config["model_type"],
        "pooling":if qwen_vl {"last_token"} else {"mean"},"dense_paths":dense_paths,"iterations_per_backend":iterations,"torch_gpu":torch_gpu,"candle_gpu":candle_gpu,"measurement_mode":"parallel separate GPUs","warmup_per_backend":20,
        "torch_cuda_graph_count":torch.lock().unwrap().cuda_graph_count(),
        "torch_cuda_graph_max_tokens":std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_MAX_TOKENS")).unwrap_or_else(|_| "4096".into()),
        "torch_load_seconds":torch_load,"candle_load_seconds":candle_load,"workloads":rows}))
        .unwrap()
    );
}
