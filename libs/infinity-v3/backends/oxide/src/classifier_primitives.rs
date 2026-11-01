// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Local ignored first-layer controls against exact actual-CAPI-preserving stages.
use super::*;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn source_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_CLASSIFIER_CASE_ROOT")
            .expect("set exact local classifier case root"),
    )
}
fn native_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_CLASSIFIER_TRACE_ROOT")
            .expect("set trusted native stage trace root"),
    )
}
fn stage(key: &str, name: &str) -> (Vec<Bf16>, usize, usize) {
    let root = native_root().join(key).join("native");
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["trusted_native_capi_final_bits"], true);
    let meta = &m["stage_manifest"][name];
    assert_eq!(meta["dtype"], "BF16");
    let rows = meta["shape"][0].as_u64().unwrap() as usize;
    let width = meta["shape"][1].as_u64().unwrap() as usize;
    let bytes = fs::read(root.join(meta["file"].as_str().unwrap())).unwrap();
    assert_eq!(
        bytes.len(),
        rows.checked_mul(width).unwrap().checked_mul(2).unwrap()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        meta["sha256"].as_str().unwrap()
    );
    (
        bytes
            .chunks_exact(2)
            .map(|x| Bf16::from_bits(u16::from_le_bytes([x[0], x[1]])))
            .collect(),
        rows,
        width,
    )
}
fn same(got: &[Bf16], gold: &[Bf16], label: &str) {
    assert_eq!(got.len(), gold.len(), "{label} length");
    let bad = got
        .iter()
        .zip(gold)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert!(
        bad.is_empty(),
        "{label}: {} of {} BF16 values differ; first {:?}",
        bad.len(),
        gold.len(),
        bad.first()
    );
}
fn model(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(if key == "minilm" {
            "INFINITY_OXIDE_CLASSIFIER_MINILM_MODEL"
        } else {
            "INFINITY_OXIDE_CLASSIFIER_NER_MODEL"
        })
        .expect("set actual trained classifier"),
    )
}

#[test]
#[ignore = "Root GPU: additive55 kernels, actual MiniLM model and trusted native stage files"]
fn trained_classifier_h384_embedding_residual_and_row_tails_match_native_bits() {
    let path = model("minilm");
    let (host, _) = BertWeights::load_classifier(&path).unwrap();
    let config = host.config.clone();
    assert_eq!(config.hidden_size, 384);
    let (gold, total, width) = stage("minilm", "embedding");
    let (residual, _, _) = stage("minilm", "embedding");
    let (projected, _, _) = stage("minilm", "layer_00_attention_projection");
    let (norm_gold, _, _) = stage("minilm", "layer_00_attention_norm");
    assert_eq!((total, width), (128, 384));
    let b: serde_json::Value =
        serde_json::from_slice(&fs::read(source_root().join("minilm/batch.json")).unwrap())
            .unwrap();
    let ids: Vec<u32> = serde_json::from_value(b["input_ids"].clone()).unwrap();
    let pos: Vec<u32> = serde_json::from_value(b["position_ids"].clone()).unwrap();
    let types: Vec<u32> = serde_json::from_value(b["token_type_ids"].clone()).unwrap();
    assert_eq!(ids.len(), total);
    assert!(
        Context::device_count().unwrap() > 0,
        "GPU gate must execute"
    );
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let kernels = Kernels::load(&context, &config, true).unwrap();
    assert!(kernels.embedding_384.is_none());
    assert!(kernels.norm_384.is_none());
    assert_eq!(kernels.embedding_geometry.rows_per_cta, 4);
    assert_eq!(kernels.embedding_geometry.threads, 128);
    let mut upload = WeightUploader::new(stream.clone()).unwrap();
    let word = upload.upload(&[&host.word]).unwrap();
    let position = upload.upload(&[&host.position]).unwrap();
    let token_type = upload.upload(&[&host.token_type]).unwrap();
    let gamma = upload.upload(&[&host.gamma]).unwrap();
    let beta = upload.upload(&[&host.beta]).unwrap();
    let ag = upload.upload(&[&host.layers[0].attention_gamma]).unwrap();
    let ab = upload.upload(&[&host.layers[0].attention_beta]).unwrap();
    context.bind_to_thread().unwrap();
    for rows in [1usize, 3, 5, 127, 128] {
        let di = DeviceBuffer::from_host(&stream, &ids[..rows]).unwrap();
        let dp = DeviceBuffer::from_host(&stream, &pos[..rows]).unwrap();
        let dt = DeviceBuffer::from_host(&stream, &types[..rows]).unwrap();
        let out = DeviceBuffer::<Bf16>::zeroed(&stream, rows * width).unwrap();
        let (mut ip, mut pp, mut tp) = (di.cu_deviceptr(), dp.cu_deviceptr(), dt.cu_deviceptr());
        let (mut w, mut p, mut t) = (
            word.cu_deviceptr(),
            position.cu_deviceptr(),
            token_type.cu_deviceptr(),
        );
        let (mut g, mut b, mut o) = (
            gamma.cu_deviceptr(),
            beta.cu_deviceptr(),
            out.cu_deviceptr(),
        );
        let (mut n, mut h, mut eps) = (
            u32_size(rows).unwrap(),
            384u32,
            config.layer_norm_eps as f32,
        );
        launch_threads(
            &kernels.embedding,
            &stream,
            (n.div_ceil(4), 1, 1),
            128,
            0,
            &mut [
                arg(&mut ip),
                arg(&mut pp),
                arg(&mut tp),
                arg(&mut w),
                arg(&mut p),
                arg(&mut t),
                arg(&mut g),
                arg(&mut b),
                arg(&mut o),
                arg(&mut n),
                arg(&mut h),
                arg(&mut eps),
            ],
        )
        .unwrap();
        same(
            &out.to_host_vec(&stream).unwrap(),
            &gold[..rows * width],
            "H384 embedding/tail",
        );
        let input = DeviceBuffer::from_host(&stream, &projected[..rows * width]).unwrap();
        let skip = DeviceBuffer::from_host(&stream, &residual[..rows * width]).unwrap();
        norm(
            &kernels,
            &stream,
            input.cu_deviceptr(),
            skip.cu_deviceptr(),
            0,
            &ag,
            &ab,
            out.cu_deviceptr(),
            n,
            h,
            eps,
        )
        .unwrap();
        same(
            &out.to_host_vec(&stream).unwrap(),
            &norm_gold[..rows * width],
            "H384 residual/tail",
        );
        norm(
            &kernels,
            &stream,
            input.cu_deviceptr(),
            skip.cu_deviceptr(),
            0,
            &ag,
            &ab,
            input.cu_deviceptr(),
            n,
            h,
            eps,
        )
        .unwrap();
        same(
            &input.to_host_vec(&stream).unwrap(),
            &norm_gold[..rows * width],
            "H384 residual input/output alias",
        );
    }
}

#[test]
#[ignore = "Root GPU: actual both classifier checkpoints and trusted native layer0 stages"]
fn trained_classifier_down_native_default_matches_both_actual_stage_bits() {
    // Read every real reference before any CUDA operation.
    let data = ["minilm", "ner"].map(|key| {
        let (x, rows, k) = stage(key, "layer_00_gelu");
        let (gold, gr, n) = stage(key, "layer_00_output_projection");
        assert_eq!(rows, gr);
        (key, x, rows, k, gold, n)
    });
    assert!(
        Context::device_count().unwrap() > 0,
        "GPU gate must execute"
    );
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let provider = LtProvider::new_native_default(context.clone(), stream.clone()).unwrap();
    let blas = Blas::new(context.clone(), stream.clone()).unwrap();
    for (key, input, rows, k, gold, n) in data {
        let (host, _) = BertWeights::load_classifier(&model(key)).unwrap();
        assert_eq!(
            (host.config.intermediate_size, host.config.hidden_size),
            (k, n)
        );
        assert_eq!(
            down_policy(&host.config),
            DownPolicy::ClassifierNativeDefault
        );
        let mut upload = WeightUploader::new(stream.clone()).unwrap();
        let weight = upload.upload(&[&host.layers[0].output.weight]).unwrap();
        let bias = upload.upload(&[&host.layers[0].output.bias]).unwrap();
        let x = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out = DeviceBuffer::<Bf16>::zeroed(&stream, rows * n).unwrap();
        let mut prepared = PreparedDown::new(&provider, &host.config).unwrap();
        context.bind_to_thread().unwrap();
        prepared.begin_prepare();
        unsafe {
            prepared.prepare_layer(
                &provider,
                Some(&blas),
                rows,
                &x,
                &weight,
                Some(&bias),
                &mut out,
                &stream,
                n,
                0,
            )
        }
        .unwrap();
        prepared.end_prepare();
        unsafe {
            enqueue_down(
                &mut prepared,
                &provider,
                Some(&blas),
                rows,
                0,
                &x,
                &weight,
                Some(&bias),
                &mut out,
                &stream,
            )
        }
        .unwrap();
        same(
            &out.to_host_vec(&stream).unwrap(),
            &gold,
            "classifier actual trained layer0 down",
        );
    }
}
