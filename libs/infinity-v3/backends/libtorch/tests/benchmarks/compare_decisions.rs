//! Paired trained typed-decision comparison on exact packed token sequences.
use infinity_backend_libtorch::LibtorchBackend;
use std::{
    hint::black_box,
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
    time::Instant,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{
    AudioFeatures, Backend, Batch, ClefField, DecisionInput, DecisionOutput, ImagePatches,
    ModelType, MultimodalEncoding,
};
fn input(sequences: &[serde_json::Value]) -> (Batch, Vec<DecisionInput>) {
    let mut b = Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let mut requests = vec![];
    for s in sequences {
        let ids: Vec<u32> = serde_json::from_value(s["ids"].clone()).unwrap();
        let length = ids.len();
        let images = s["images"].as_array();
        let audios = s["audios"].as_array();
        let media = if images.is_some() || audios.is_some() {
            let mut media = MultimodalEncoding {
                images: vec![],
                audios: vec![],
                position_ids: s.get("multimodal_position_ids").map_or_else(
                    || std::array::from_fn(|_| (0..length as u32).collect()),
                    |positions| serde_json::from_value(positions.clone()).unwrap(),
                ),
                memory: None,
            };
            for image in images.into_iter().flatten() {
                let grid_thw: [usize; 3] =
                    serde_json::from_value(image["grid_thw"].clone()).unwrap();
                let patch_dim = image["patch_dim"].as_u64().unwrap() as usize;
                let pixels = image.get("pixels").map_or_else(
                    || {
                        (0..grid_thw.iter().product::<usize>() * patch_dim)
                            .map(|i| (i % 251) as f32 / 250.)
                            .collect()
                    },
                    |pixels| serde_json::from_value(pixels.clone()).unwrap(),
                );
                media.images.push((
                    image["token_start"].as_u64().unwrap() as usize,
                    Arc::new(ImagePatches {
                        pixels,
                        grid_thw,
                        patch_dim,
                        merge_size: image["merge_size"].as_u64().unwrap() as usize,
                    }),
                ));
            }
            for audio in audios.into_iter().flatten() {
                media.audios.push((
                    audio["token_start"].as_u64().unwrap() as usize,
                    Arc::new(AudioFeatures {
                        values: serde_json::from_value(audio["values"].clone()).unwrap(),
                        mask: serde_json::from_value(audio["mask"].clone()).unwrap(),
                        feature_size: audio["feature_size"].as_u64().unwrap() as usize,
                    }),
                ));
            }
            Some(Arc::new(media))
        } else {
            None
        };
        b.multimodal.push(media);
        b.max_length = b.max_length.max(ids.len() as u32);
        b.position_ids.extend(0..ids.len() as u32);
        b.token_type_ids.extend(vec![0; ids.len()]);
        b.input_ids.extend(ids);
        b.cumulative_seq_lengths.push(b.input_ids.len() as u32);
        requests.push(match s["kind"].as_str().unwrap() {
            "laya" => DecisionInput::Laya {
                question_type: s["question_type"].as_u64().unwrap() as usize,
                markers: serde_json::from_value(s["markers"].clone()).unwrap(),
            },
            "clef" => DecisionInput::Clef {
                fields: s["fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|f| ClefField {
                        kind: f["kind"].as_u64().unwrap() as usize,
                        question: serde_json::from_value(f["question"].clone()).unwrap(),
                        options: serde_json::from_value(f["options"].clone()).unwrap(),
                    })
                    .collect(),
            },
            "option_tokens" | "pplx" => DecisionInput::OptionTokens {
                token_ids: serde_json::from_value(s["option_token_ids"].clone()).unwrap(),
            },
            k => panic!("Unknown metadata {k}"),
        });
    }
    (b, requests)
}
#[cfg(test)]
mod tests {
    #[test]
    fn media_fixture_preserves_true_shapes_and_sequence_alignment() {
        let sequences = serde_json::json!([
            {"ids":[2,3,4,5,6,7,1],"kind":"option_tokens","option_token_ids":[8,9],
             "images":[{"token_start":2,"grid_thw":[1,3,6],"patch_dim":2,"merge_size":3}]},
            {"ids":[2,8,9,1],"kind":"option_tokens","option_token_ids":[8,9]},
            {"ids":[2,3,4,1],"kind":"option_tokens","option_token_ids":[8,9],
             "audios":[{"token_start":1,"feature_size":2,"values":[0.1,0.2,0.3,0.4],"mask":[1,0]}]}
        ]);
        let (batch, requests) = super::input(sequences.as_array().unwrap());
        assert_eq!(batch.cumulative_seq_lengths, [0, 7, 11, 15]);
        assert_eq!(requests.len(), 3);
        assert!(batch.multimodal[1].is_none());
        let image = &batch.multimodal[0].as_ref().unwrap().images[0];
        assert_eq!(image.0, 2);
        assert_eq!(image.1.grid_thw, [1, 3, 6]);
        assert_eq!(image.1.pixels.len(), 36);
        assert_eq!(image.1.token_count(), 2);
        let audio = &batch.multimodal[2].as_ref().unwrap().audios[0];
        assert_eq!(audio.0, 1);
        assert_eq!(audio.1.values.len(), 4);
        assert_eq!(audio.1.mask, [1, 0]);
    }
}
fn stats(mut t: Vec<f64>, tokens: usize) -> serde_json::Value {
    t.sort_by(f64::total_cmp);
    let p = t[t.len() / 2];
    serde_json::json!({"p50_ms":p,"p95_ms":t[(t.len()*95/100).min(t.len()-1)],"tokens_per_second_at_p50":tokens as f64*1000./p})
}
fn accuracy(
    a: &[DecisionOutput],
    b: &[DecisionOutput],
    requests: &[DecisionInput],
    dtype: &str,
) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), requests.len());
    let mut cosine = 1f64;
    let mut maximum_absolute_error = 0f32;
    let mut action_error = 0f32;
    let mut probability_error = 0f64;
    let mut argmax_matches = 0;
    let mut fields = 0;
    let mut scalar_fields = 0;
    let mut scalar_gate_pass = true;
    let mut maximum_scalar_error = 0f64;
    let precision_epsilon = match dtype {
        "bfloat16" => 1. / 128.,
        "float16" => 1. / 1024.,
        "float32" => f64::from(f32::EPSILON),
        _ => panic!("Unsupported decision comparison dtype"),
    };
    for ((a, b), request) in a.iter().zip(b).zip(requests) {
        assert_eq!(a.logits.len(), b.logits.len());
        assert!(
            a.action_probability.is_finite() && b.action_probability.is_finite(),
            "Non-finite action probability: native={}, Candle={}",
            a.action_probability,
            b.action_probability
        );
        action_error = action_error.max((a.action_probability - b.action_probability).abs());
        let widths = match request {
            DecisionInput::Clef { fields } => {
                fields.iter().map(|f| f.options.len()).collect::<Vec<_>>()
            }
            _ => vec![a.logits.len()],
        };
        assert_eq!(widths.iter().sum::<usize>(), a.logits.len());
        let mut offset = 0;
        for width in widths {
            let actual = &a.logits[offset..offset + width];
            let expected = &b.logits[offset..offset + width];
            offset += width;
            fields += 1;
            let (mut dot, mut an, mut bn) = (0., 0., 0.);
            for (&x, &y) in actual.iter().zip(expected) {
                assert!(x.is_finite() && y.is_finite());
                dot += f64::from(x) * f64::from(y);
                an += f64::from(x).powi(2);
                bn += f64::from(y).powi(2);
                maximum_absolute_error = maximum_absolute_error.max((x - y).abs());
            }
            if width == 1 {
                scalar_fields += 1;
                let error = f64::from((actual[0] - expected[0]).abs());
                // Scalar cosine cannot measure magnitude. The existing typed
                // CUDA contract supplies the 0.006 absolute floor; two model-
                // dtype epsilons bound the relative term, fixed before testing.
                let tolerance = 0.006 + 2. * precision_epsilon * f64::from(expected[0]).abs();
                scalar_gate_pass &= error <= tolerance;
                maximum_scalar_error = maximum_scalar_error.max(error);
            } else {
                cosine = cosine.min(if an * bn == 0. {
                    if actual == expected {
                        1.
                    } else {
                        0.
                    }
                } else {
                    dot / (an * bn).sqrt()
                });
            }
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .unwrap()
                    .0
            };
            argmax_matches += usize::from(argmax(actual) == argmax(expected));
            let softmax = |values: &[f32]| {
                let largest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponents = values
                    .iter()
                    .map(|v| (f64::from(*v) - f64::from(largest)).exp())
                    .collect::<Vec<_>>();
                let sum = exponents.iter().sum::<f64>();
                exponents.into_iter().map(|v| v / sum).collect::<Vec<_>>()
            };
            for (x, y) in softmax(actual).into_iter().zip(softmax(expected)) {
                probability_error = probability_error.max((x - y).abs());
            }
        }
    }
    serde_json::json!({"minimum_logit_cosine":cosine,"maximum_logit_absolute_error":maximum_absolute_error,"maximum_action_probability_error":action_error,"maximum_uncalibrated_option_probability_error":probability_error,"argmax_agreement":argmax_matches,"fields":fields,"sequences":a.len(),"cosine_gate_pass":cosine>=0.999,"action_gate_pass":action_error<=0.001,"argmax_gate_pass":argmax_matches==fields,"scalar_fields":scalar_fields,"scalar_gate_pass":scalar_gate_pass,"maximum_scalar_logit_absolute_error":maximum_scalar_error,"scalar_absolute_tolerance":0.006,"scalar_relative_tolerance":2.*precision_epsilon})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 6,
        "compare_decisions MODEL FIXTURE ITERATIONS TORCH_GPU CANDLE_GPU [DTYPE]"
    );
    let path = PathBuf::from(&args[1]);
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let iterations: usize = args[3].parse().unwrap();
    assert!(iterations >= 20);
    let tg: usize = args[4].parse().unwrap();
    let cg: usize = args[5].parse().unwrap();
    assert_ne!(tg, cg);
    let dtype = args.get(6).map(String::as_str).unwrap_or("bfloat16");
    let torch = Mutex::new(
        LibtorchBackend::new(&path, dtype, ModelType::Decision, &format!("cuda:{tg}")).unwrap(),
    );
    let candle =
        Mutex::new(CandleBackend::new(&path, dtype.into(), ModelType::Decision, None, cg).unwrap());
    let mut rows = vec![];
    for case in fixture["cases"].as_array().unwrap() {
        let (batch, requests) = input(case["sequences"].as_array().unwrap());
        let a = torch
            .lock()
            .unwrap()
            .decide(batch.clone(), requests.clone())
            .unwrap();
        let b = candle
            .lock()
            .unwrap()
            .decide(batch.clone(), requests.clone())
            .unwrap();
        let comparison = accuracy(&a, &b, &requests, dtype);
        let barrier = Barrier::new(2);
        let (ts, cs) = std::thread::scope(|scope| {
            let run = |backend: &Mutex<_>| {
                let backend = backend.lock().unwrap();
                let backend: &dyn Backend = &*backend;
                for _ in 0..20 {
                    black_box(backend.decide(batch.clone(), requests.clone()).unwrap());
                }
                barrier.wait();
                (0..iterations)
                    .map(|_| {
                        let batch = batch.clone();
                        let requests = requests.clone();
                        let start = Instant::now();
                        black_box(backend.decide(batch, requests).unwrap());
                        start.elapsed().as_secs_f64() * 1000.
                    })
                    .collect::<Vec<_>>()
            };
            let torch_ref = &torch;
            let t = scope.spawn(move || run(torch_ref));
            let c = scope.spawn(|| {
                let backend = candle.lock().unwrap();
                for _ in 0..20 {
                    black_box(backend.decide(batch.clone(), requests.clone()).unwrap());
                }
                barrier.wait();
                (0..iterations)
                    .map(|_| {
                        let batch = batch.clone();
                        let requests = requests.clone();
                        let start = Instant::now();
                        black_box(backend.decide(batch, requests).unwrap());
                        start.elapsed().as_secs_f64() * 1000.
                    })
                    .collect::<Vec<_>>()
            });
            (t.join().unwrap(), c.join().unwrap())
        });
        let ts = stats(ts, batch.input_ids.len());
        let cs = stats(cs, batch.input_ids.len());
        rows.push(serde_json::json!({"case":case["name"],"actual_tokens":batch.input_ids.len(),"cuda_graph_count":torch.lock().unwrap().cuda_graph_count(),"lengths":batch.cumulative_seq_lengths.windows(2).map(|p|p[1]-p[0]).collect::<Vec<_>>(),"accuracy":comparison,"torch":ts,"candle":cs,"latency_delta_percent":(ts["p50_ms"].as_f64().unwrap()/cs["p50_ms"].as_f64().unwrap()-1.)*100.,"torch_outputs":a.iter().map(|o|serde_json::json!({"logits":o.logits,"action_probability":o.action_probability})).collect::<Vec<_>>(),"candle_outputs":b.iter().map(|o|serde_json::json!({"logits":o.logits,"action_probability":o.action_probability})).collect::<Vec<_>>() }));
        eprintln!("{} complete", case["name"]);
    }
    let identity = std::fs::read(path.join("checkpoint_metadata.json"))
        .ok()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).unwrap());
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"model_path":path,"checkpoint_metadata":identity,"dtype":dtype,"torch_gpu":tg,"candle_gpu":cg,"iterations":iterations,"native_library_path":std::env::var("LD_LIBRARY_PATH").ok(),"no_token_padding":true,"cases":rows})).unwrap());
}
