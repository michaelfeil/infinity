//! Full packed pretrained classifier/SPLADE pipelines; preprocessing is excluded.
use infinity_backend_libtorch::LibtorchBackend;
use std::{
    hint::black_box,
    path::PathBuf,
    sync::{Barrier, Mutex},
    time::Instant,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

fn input(sequences: &[serde_json::Value]) -> Batch {
    let mut ids = Vec::new();
    let mut types = Vec::new();
    let mut positions = Vec::new();
    let mut ends = vec![0];
    let mut maximum = 0;
    for sequence in sequences {
        let row = sequence["ids"].as_array().unwrap();
        maximum = maximum.max(row.len());
        assert!(!row.is_empty());
        ids.extend(
            row.iter()
                .map(|n| u32::try_from(n.as_u64().unwrap()).unwrap()),
        );
        positions.extend((0..row.len()).map(|n| n as u32));
        let row_types = sequence["types"].as_array().unwrap();
        assert_eq!(row.len(), row_types.len());
        types.extend(
            row_types
                .iter()
                .map(|n| u32::try_from(n.as_u64().unwrap()).unwrap()),
        );
        ends.push(ids.len() as u32);
    }
    Batch {
        input_ids: ids,
        token_type_ids: types,
        position_ids: positions,
        cumulative_seq_lengths: ends,
        max_length: maximum as u32,
        pooled_indices: (0..sequences.len() as u32).collect(),
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
fn forward(backend: &dyn Backend, batch: Batch, splade: bool) -> Vec<Vec<f32>> {
    let rows = batch.len();
    if splade {
        let mut outputs = backend.embed(batch).unwrap();
        (0..rows)
            .map(|n| match outputs.remove(&n).unwrap() {
                Embedding::Pooled(v) => v,
                Embedding::All(_) => panic!("Expected pooled SPLADE output"),
            })
            .collect()
    } else {
        let mut outputs = backend.predict(batch).unwrap();
        (0..rows).map(|n| outputs.remove(&n).unwrap()).collect()
    }
}
fn stats(mut samples: Vec<f64>, tokens: usize) -> serde_json::Value {
    samples.sort_by(f64::total_cmp);
    let p50 = samples[samples.len() / 2];
    serde_json::json!({"p50_ms":p50,"p95_ms":samples[(samples.len()*95/100).min(samples.len()-1)],"tokens_per_second_at_p50":tokens as f64*1000./p50})
}
fn accuracy(
    native: &[Vec<f32>],
    reference: &[Vec<f32>],
    splade: bool,
    epsilon: f64,
) -> serde_json::Value {
    assert_eq!(native.len(), reference.len());
    let mut finite = true;
    let mut nonnegative = true;
    let mut nonempty_support = true;
    let mut cosine = 1f64;
    let mut absolute = 0f64;
    let mut relative_l2 = 0f64;
    let mut support_jaccard = 1f64;
    let mut scalar_pass = true;
    let mut native_active = Vec::new();
    let mut reference_active = Vec::new();
    for (a, b) in native.iter().zip(reference) {
        assert_eq!(a.len(), b.len());
        assert!(!a.is_empty());
        let (mut dot, mut an, mut bn, mut distance) = (0., 0., 0., 0.);
        let (mut intersection, mut union, mut na, mut nb) = (0usize, 0usize, 0usize, 0usize);
        for (&x, &y) in a.iter().zip(b) {
            finite &= x.is_finite() && y.is_finite();
            nonnegative &= x >= 0. && y >= 0.;
            let (x, y) = (f64::from(x), f64::from(y));
            dot += x * y;
            an += x * x;
            bn += y * y;
            distance += (x - y).powi(2);
            absolute = absolute.max((x - y).abs());
            na += usize::from(x > 0.);
            nb += usize::from(y > 0.);
            intersection += usize::from(x > 0. && y > 0.);
            union += usize::from(x > 0. || y > 0.);
            if a.len() == 1 {
                scalar_pass &= (x - y).abs() <= 0.006 + 2. * epsilon * y.abs();
            }
        }
        if splade {
            nonempty_support &= an > 0. && bn > 0.;
            cosine = cosine.min(if an * bn > 0. {
                dot / (an * bn).sqrt()
            } else {
                0.
            });
            relative_l2 = relative_l2.max(if bn > 0. {
                (distance / bn).sqrt()
            } else if distance == 0. {
                0.
            } else {
                f64::INFINITY
            });
            support_jaccard = support_jaccard.min(if union > 0 {
                intersection as f64 / union as f64
            } else {
                1.
            });
        } else {
            assert_eq!(
                a.len(),
                1,
                "This trained classifier contract is scalar; use a vector gate for other heads"
            );
        }
        native_active.push(na);
        reference_active.push(nb);
    }
    let pass = finite
        && if splade {
            nonnegative
                && nonempty_support
                && cosine >= 0.999
                && relative_l2 <= 0.05
                && support_jaccard >= 0.95
        } else {
            scalar_pass
        };
    serde_json::json!({"pass":pass,"finite":finite,"nonnegative":nonnegative,"nonempty_support":nonempty_support,"minimum_vector_cosine":if splade {Some(cosine)}else{None},"maximum_absolute_error":absolute,"maximum_relative_l2_error":if splade {Some(relative_l2)}else{None},"minimum_nonzero_support_jaccard":if splade {Some(support_jaccard)}else{None},"native_nonzero_counts":native_active,"reference_nonzero_counts":reference_active,"scalar_magnitude_pass":scalar_pass,"scalar_absolute_tolerance":0.006,"scalar_relative_tolerance":2.*epsilon})
}
fn saved_outputs(values: &[Vec<f32>], splade: bool) -> serde_json::Value {
    if splade {
        serde_json::json!(values.iter().map(|v|serde_json::json!({"width":v.len(),"nonzero":v.iter().enumerate().filter(|(_,n)|**n!=0.).map(|(i,n)|(i,*n)).collect::<Vec<_>>()})).collect::<Vec<_>>())
    } else {
        serde_json::json!(values)
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 7,
        "compare_heads MODEL FIXTURE classifier|splade ITERATIONS TORCH_GPU CANDLE_GPU [DTYPE]"
    );
    let path = PathBuf::from(&args[1]);
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let splade = match args[3].as_str() {
        "classifier" => false,
        "splade" => true,
        _ => panic!("Unknown head contract"),
    };
    let iterations: usize = args[4].parse().unwrap();
    assert!(iterations >= 20);
    let tg: usize = args[5].parse().unwrap();
    let cg: usize = args[6].parse().unwrap();
    assert_ne!(tg, cg);
    let dtype = args.get(7).map(String::as_str).unwrap_or("float16");
    let epsilon = match dtype {
        "float16" => 1. / 1024.,
        "bfloat16" => 1. / 128.,
        "float32" => f64::from(f32::EPSILON),
        _ => panic!("Unknown dtype"),
    };
    let kind = if splade {
        ModelType::Embedding(Pool::Splade)
    } else {
        ModelType::Classifier
    };
    let torch = Mutex::new(
        LibtorchBackend::new(&path, dtype, kind.clone(), &format!("cuda:{tg}")).unwrap(),
    );
    let candle = Mutex::new(CandleBackend::new(&path, dtype.into(), kind, None, cg).unwrap());
    let mut cases = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let batch = input(case["sequences"].as_array().unwrap());
        let a = forward(&*torch.lock().unwrap(), batch.clone(), splade);
        let b = forward(&*candle.lock().unwrap(), batch.clone(), splade);
        let comparison = accuracy(&a, &b, splade, epsilon);
        let barrier = Barrier::new(2);
        let (tt, ct) = std::thread::scope(|scope| {
            let run = |backend: &dyn Backend| {
                for _ in 0..20 {
                    black_box(forward(backend, batch.clone(), splade));
                }
                barrier.wait();
                (0..iterations)
                    .map(|_| {
                        let input = batch.clone();
                        let start = Instant::now();
                        black_box(forward(backend, input, splade));
                        start.elapsed().as_secs_f64() * 1000.
                    })
                    .collect::<Vec<_>>()
            };
            let torch_model = &torch;
            let candle_model = &candle;
            let t = scope.spawn(move || run(&*torch_model.lock().unwrap()));
            let c = scope.spawn(move || run(&*candle_model.lock().unwrap()));
            (t.join().unwrap(), c.join().unwrap())
        });
        let ts = stats(tt, batch.input_ids.len());
        let cs = stats(ct, batch.input_ids.len());
        let delta = (ts["p50_ms"].as_f64().unwrap() / cs["p50_ms"].as_f64().unwrap() - 1.) * 100.;
        cases.push(serde_json::json!({"case":case["name"],"actual_tokens":batch.input_ids.len(),"lengths":case["sequences"].as_array().unwrap().iter().map(|s|s["ids"].as_array().unwrap().len()).collect::<Vec<_>>(),"accuracy":comparison,"torch_outputs":saved_outputs(&a,splade),"candle_outputs":saved_outputs(&b,splade),"torch":ts,"candle":cs,"latency_delta_percent":delta,"cuda_graph_count":torch.lock().unwrap().cuda_graph_count()}));
        eprintln!("{} complete", case["name"]);
    }
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("checkpoint_metadata.json")).unwrap())
            .unwrap();
    let serialization = std::fs::read(path.join("serialization_metadata.json"))
        .ok()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).unwrap());
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"model_path":path,"head":args[3],"dtype":dtype,"iterations":iterations,"warmup_iterations":20,"torch_gpu":tg,"candle_gpu":cg,"checkpoint_metadata":metadata,"serialization_metadata":serialization,"gate_contract":"Fixed before trained runs: finite outputs; scalar abs error <= .006 + 2*dtype epsilon*abs(reference), no scalar cosine gate; SPLADE nonnegative, cosine>=.999, relativeL2<=.05, exact nonzero supportJaccard>=.95. All gates required.","fixture_path":args[2],"cases":cases})).unwrap());
}

#[cfg(test)]
mod tests {
    use super::accuracy;

    #[test]
    fn scalar_magnitude_cannot_pass_by_matching_sign() {
        assert!(
            !accuracy(&[vec![2.]], &[vec![1.]], false, 1. / 1024.)["pass"]
                .as_bool()
                .unwrap()
        );
        assert!(
            !accuracy(&[vec![f32::NAN]], &[vec![1.]], false, 1. / 1024.)["pass"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn splade_support_and_zero_vectors_cannot_hide_behind_cosine() {
        let result = accuracy(
            &[vec![1., 0.0001, 0.0001]],
            &[vec![1., 0., 0.]],
            true,
            1. / 1024.,
        );
        assert!(result["minimum_vector_cosine"].as_f64().unwrap() > 0.999);
        assert!(!result["pass"].as_bool().unwrap());
        assert!(
            !accuracy(&[vec![0., 0.]], &[vec![0., 0.]], true, 1. / 1024.)["pass"]
                .as_bool()
                .unwrap()
        );
    }
}
