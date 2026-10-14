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
use text_embeddings_backend_core::{Backend, Batch, Embedding, Embeddings, ModelType, Pool};

// Distinct suffixes branch off a real shared token prefix. This is a trie over
// actual token/position/type triples, with no unused rows or token padding.
fn batch(
    lengths: &[usize],
    prefix: usize,
    vocab_size: u32,
    bos: Option<u32>,
    eos: Option<u32>,
    seed: u32,
) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative = vec![0];
    for (row, &length) in lengths.iter().enumerate() {
        assert!(prefix < length - 1, "Need independent suffixes before EOS");
        input_ids.extend((0..length).map(|i| {
            (if i + 1 == length { eos } else { None })
                .or(if i == 0 { bos } else { None })
                .unwrap_or_else(|| {
                    (1000
                        + seed * 43
                        + i as u32 * 17
                        + if i < prefix { 0 } else { (row as u32 + 1) * 31 })
                        % vocab_size
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
fn folded(input: &Batch, reverse: bool) -> Batch {
    use std::collections::HashMap;
    let mut trie = HashMap::new();
    let (mut ids, mut positions, mut scatter, mut gather) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for span in input.cumulative_seq_lengths.windows(2) {
        let mut parent = u32::MAX;
        for row in span[0]..span[1] {
            let i = row as usize;
            let key = (
                parent,
                input.input_ids[i],
                input.position_ids[i],
                input.token_type_ids[i],
            );
            let node = *trie.entry(key).or_insert_with(|| {
                let node = ids.len() as u32;
                ids.push(input.input_ids[i]);
                positions.push(input.position_ids[i]);
                gather.push(row);
                node
            });
            scatter.push(node);
            parent = node;
        }
    }
    assert!(
        ids.len() < input.input_ids.len(),
        "Workload must share actual prefixes"
    );
    if reverse {
        let last = ids.len() as u32 - 1;
        ids.reverse();
        positions.reverse();
        gather.reverse();
        for index in &mut scatter {
            *index = last - *index;
        }
    }
    let mut result = input.clone();
    result.compact_input_ids = Some(ids);
    result.compact_position_ids = Some(positions);
    result.scatter_unfold = Some(scatter);
    result.fold_gather = Some(gather);
    result
}
fn raw(input: &Batch) -> Batch {
    let mut result = input.clone();
    result.pooled_indices.clear();
    result.raw_indices = (0..input.cumulative_seq_lengths.len() as u32 - 1).collect();
    result
}
fn close(a: &Embeddings, b: &Embeddings, label: &str) -> serde_json::Value {
    metrics(a, b, label, true)
}
fn metrics(a: &Embeddings, b: &Embeddings, label: &str, enforce_gate: bool) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    let mut min_cosine = 1.0_f64;
    let mut max_abs = 0.0_f32;
    let mut vectors = 0;
    let (mut min_row, mut min_token) = (0, 0);
    for (&row, a) in a {
        let b = &b[&row];
        let pairs: Vec<(&[f32], &[f32])> = match (a, b) {
            (Embedding::Pooled(a), Embedding::Pooled(b)) => vec![(a, b)],
            (Embedding::All(a), Embedding::All(b)) => {
                assert_eq!(a.len(), b.len());
                a.iter()
                    .zip(b)
                    .map(|(a, b)| (a.as_slice(), b.as_slice()))
                    .collect()
            }
            _ => panic!("Mismatched output kind"),
        };
        for (token, (a, b)) in pairs.into_iter().enumerate() {
            assert_eq!(a.len(), b.len());
            let (mut dot, mut an, mut bn) = (0.0, 0.0, 0.0);
            for (&x, &y) in a.iter().zip(b) {
                assert!(x.is_finite() && y.is_finite());
                dot += f64::from(x) * f64::from(y);
                an += f64::from(x).powi(2);
                bn += f64::from(y).powi(2);
                max_abs = max_abs.max((x - y).abs());
            }
            let cosine = if an == 0.0 && bn == 0.0 {
                1.0
            } else {
                dot / (an * bn).sqrt()
            };
            assert!(
                cosine.is_finite(),
                "{label} nonfinite cosine row{row} token{token}"
            );
            if cosine < min_cosine {
                min_cosine = cosine;
                min_row = row;
                min_token = token;
            }
            vectors += 1;
        }
    }
    if enforce_gate {
        assert!(
            min_cosine > 0.999,
            "{label} minimum cosine {min_cosine} row{min_row} token{min_token}"
        );
    }
    serde_json::json!({"min_cosine":min_cosine,"min_sequence_index":min_row,"min_token_index_within_sequence":min_token,"max_absolute_difference":max_abs,"vectors":vectors})
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
    let path =
        PathBuf::from(args.next().expect(
            "compare_radix MODEL_PATH [ITERATIONS] [TORCH_GPU] [CANDLE_GPU] [DTYPE] [POOL]",
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
    let workloads = vec![
        ("8x128 prefix96", vec![128; 8], 96),
        ("32x128 prefix64", vec![128; 32], 64),
        (
            "32 ragged64..256 prefix32",
            (0..32).map(|i| 64 + i % 7 * 32).collect(),
            32,
        ),
        ("8x512 prefix384", vec![512; 8], 384),
        ("32x512 prefix256", vec![512; 32], 256),
    ];
    let graph_requested = std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPHS"))
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let graph_max: usize = std::env::var(infinity_v3_config::infinity_env!(
        "TORCH_CUDA_GRAPH_MAX_TOKENS"
    ))
    .unwrap_or_else(|_| "4096".into())
    .parse()
    .unwrap();
    let mut rows = Vec::new();
    for (workload_index, (label, lengths, prefix)) in workloads.into_iter().enumerate() {
        let full = batch(&lengths, prefix, vocab_size, bos, eos, 0);
        let input = folded(&full, false);
        let raw_t_full = torch.lock().unwrap().embed(raw(&full)).unwrap();
        let raw_c_full = candle.lock().unwrap().embed(raw(&full)).unwrap();
        let raw_full_cross = metrics(
            &raw_t_full,
            &raw_c_full,
            "native/Candle full raw baseline",
            false,
        );
        eprintln!("{label}: full native/Candle raw diagnostic {raw_full_cross}");
        let raw_t_fold = torch.lock().unwrap().embed(raw(&input)).unwrap();
        let raw_c_fold = candle.lock().unwrap().embed(raw(&input)).unwrap();
        let raw_native = close(&raw_t_full, &raw_t_fold, "native full/fold raw");
        let raw_candle = close(&raw_c_full, &raw_c_fold, "Candle full/fold raw");
        let raw_cross = close(&raw_t_fold, &raw_c_fold, "native/Candle folded raw");
        let t_full = torch.lock().unwrap().embed(full.clone()).unwrap();
        let c_full = candle.lock().unwrap().embed(full.clone()).unwrap();
        let t_fold = torch.lock().unwrap().embed(input.clone()).unwrap();
        let c_fold = candle.lock().unwrap().embed(input.clone()).unwrap();
        let pooled_native = close(&t_full, &t_fold, "native full/fold pooled");
        let pooled_candle = close(&c_full, &c_fold, "Candle full/fold pooled");
        let pooled_cross = close(&t_fold, &c_fold, "native/Candle folded pooled");
        let count_before = torch.lock().unwrap().cuda_graph_count();
        // Same shape, different compact row order and map contents must replay.
        let reordered = folded(&full, true);
        close(
            &raw_t_full,
            &torch.lock().unwrap().embed(raw(&reordered)).unwrap(),
            "changed maps replay",
        );
        assert_eq!(
            count_before,
            torch.lock().unwrap().cuda_graph_count(),
            "Map values unnecessarily recaptured graph"
        );
        // Change shared prefix and suffix IDs while retaining all graph shapes.
        let changed = batch(&lengths, prefix, vocab_size, bos, eos, 1);
        let changed_fold = folded(&changed, true);
        let changed_expected = torch.lock().unwrap().embed(raw(&changed)).unwrap();
        close(
            &changed_expected,
            &torch.lock().unwrap().embed(raw(&changed_fold)).unwrap(),
            "changed IDs replay",
        );
        if graph_requested && full.input_ids.len() <= graph_max {
            assert!(
                torch.lock().unwrap().cuda_graph_count() > 0,
                "No actual radix CUDA capture"
            );
            if workload_index == 0 {
                let shorter_prefix = batch(&lengths, prefix / 2, vocab_size, bos, eos, 0);
                let differently_compact = folded(&shorter_prefix, false);
                let expected = torch.lock().unwrap().embed(raw(&shorter_prefix)).unwrap();
                let before = torch.lock().unwrap().cuda_graph_count();
                close(
                    &expected,
                    &torch
                        .lock()
                        .unwrap()
                        .embed(raw(&differently_compact))
                        .unwrap(),
                    "changed compact count",
                );
                assert!(
                    torch.lock().unwrap().cuda_graph_count() > before,
                    "Different compact shape did not recapture"
                );
            }
        }
        // Perturb only the final sequence's last nonspecial suffix token.
        let mut isolation = full.clone();
        let index = isolation.input_ids.len() - 2;
        isolation.input_ids[index] = (isolation.input_ids[index] + 7) % vocab_size;
        let isolated = torch
            .lock()
            .unwrap()
            .embed(raw(&folded(&isolation, false)))
            .unwrap();
        for row in 0..lengths.len() - 1 {
            let (Embedding::All(a), Embedding::All(b)) = (&raw_t_fold[&row], &isolated[&row])
            else {
                panic!("raw required")
            };
            assert_eq!(a, b, "Suffix perturbation leaked into sequence{row}");
        }
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
            pooled_cross["min_cosine"].as_f64().unwrap()
        );
        rows.push(
            serde_json::json!({"workload":label,"lengths":lengths,"tokens":input.input_ids.len(),
            "torch":torch_stats,"candle":candle_stats,"torch_speedup":ratio,
            "shared_prefix_tokens":prefix,"compact_tokens":input.compact_input_ids.as_ref().unwrap().len(),
            "raw_full_cross_backend_diagnostic":raw_full_cross,"raw_native_full_vs_fold":raw_native,"raw_candle_full_vs_fold":raw_candle,"raw_cross_backend":raw_cross,
            "pooled_native_full_vs_fold":pooled_native,"pooled_candle_full_vs_fold":pooled_candle,"pooled_cross_backend":pooled_cross,
            "changed_maps_replay_passed":true,"changed_ids_replay_passed":true,"suffix_sequence_isolation_passed":true}),
        );
    }
    let mut report = serde_json::json!({"model_path":path,"dtype":dtype,"checkpoint_revision":checkpoint_revision,"vocab_size":vocab_size,"bos_token_id":bos,"eos_token_id":eos,"pad_token_id":pad,"input_pattern":"synthetic valid vocabulary IDs with configured BOS/EOS; real shared prefixes and distinct suffixes; no padding",
        "pooling":pooling,"iterations_per_backend":iterations,"torch_gpu":torch_gpu,"candle_gpu":candle_gpu,"torch_cuda_graph_count":torch.lock().unwrap().cuda_graph_count(),"torch_cuda_graph_max_tokens":std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_MAX_TOKENS")).unwrap_or_else(|_| "4096".into()), "torch_cudnn_varlen_requested":std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDNN_VARLEN")).is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),"torch_cuda_graphs_requested":std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPHS")).is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),"measurement_mode":"parallel separate GPUs; both backends radix-folded; pooled timings; raw parity outside timings","warmup_per_backend":20,
        "torch_load_seconds":torch_load,"candle_load_seconds":candle_load,"workloads":rows});
    if fp8_dynamic {
        report["dynamic_fp8_requested"] = true.into();
        report["dynamic_fp8_recipe"] = serde_json::json!({
            "weights": "MLP-only E4M3, per-output-channel FP32 scales, quantized once",
            "activations": "per-actual-token E4M3; absmax floor1e-12 divided by448; model-dtype gated activation before quantization",
            "gemm": "FP32 accumulation, model-dtype output; original recipe fast-accumulation where supported, full-accumulation fallback",
            "unchanged_precision": "attention, normalization, residuals, embeddings and output heads",
            "token_padding": false,
            "checkpoint_supplied_quantization_scales": false,
            "comparison": "both backends use dynamic FP8; all existing raw/full/fold/replay/isolation gates are unchanged"
        });
    }
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trie_keeps_equal_suffixes_with_different_pasts_separate() {
        let mut full = batch(&[3, 3, 3], 1, 32, None, None, 0);
        full.input_ids = vec![3, 4, 5, 8, 4, 5, 3, 4, 6];
        let input = folded(&full, false);
        assert_eq!(input.compact_input_ids.as_ref().unwrap().len(), 7);
        assert_eq!(
            input.scatter_unfold.as_ref().unwrap(),
            &vec![0, 1, 2, 3, 4, 5, 0, 1, 6]
        );
        for reverse in [false, true] {
            let input = folded(&full, reverse);
            let ids = input.compact_input_ids.as_ref().unwrap();
            let positions = input.compact_position_ids.as_ref().unwrap();
            let gather = input.fold_gather.as_ref().unwrap();
            for (row, &node) in input.scatter_unfold.as_ref().unwrap().iter().enumerate() {
                assert_eq!(ids[node as usize], full.input_ids[row]);
                assert_eq!(positions[node as usize], full.position_ids[row]);
                assert_eq!(
                    input.scatter_unfold.as_ref().unwrap()[gather[node as usize] as usize],
                    node
                );
            }
        }
    }
}
