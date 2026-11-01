// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only native primitive fixture contract.
use super::*;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

pub(super) const WIDTH: usize = 768;
pub(super) const INTERMEDIATE: usize = 1152;
pub(super) const SENTINEL: u16 = 0x4242;
#[derive(Deserialize)]
pub(super) struct Word {
    pub width: usize,
    pub eps: f32,
    pub bias: bool,
    pub word_bits: Vec<u16>,
    pub gamma_bits: Vec<u16>,
    pub cases: Vec<WordCase>,
}
#[derive(Deserialize)]
pub(super) struct WordCase {
    pub rows: usize,
    pub ids: Vec<u32>,
    pub expected_bits: Vec<u16>,
}
#[derive(Deserialize)]
pub(super) struct NormCase {
    pub rows: usize,
    pub eps: f32,
    pub mode: String,
    pub input_bits: Vec<u16>,
    pub gamma_bits: Vec<u16>,
    #[serde(default)]
    pub residual_bits: Vec<u16>,
    pub expected_bits: Vec<u16>,
    #[serde(default)]
    pub expected_sum_bits: Vec<u16>,
}
#[derive(Deserialize)]
pub(super) struct RopeCase {
    pub theta: f64,
    pub capacity: usize,
    pub rows: usize,
    pub positions: Vec<u32>,
    pub qkv_input_bits: Vec<u16>,
    pub expected_qkv_bits: Vec<u16>,
    pub cosine_bits: Vec<u16>,
    pub sine_bits: Vec<u16>,
}
#[derive(Deserialize)]
pub(super) struct GemmCase {
    pub rows: usize,
    pub input_bits: Vec<u16>,
    pub weight_recipe: String,
    pub weight_shape: [usize; 2],
    pub expected_packed_bits: Vec<u16>,
    pub expected_gate_bits: Vec<u16>,
    pub used_ordinary: bool,
}
#[derive(Deserialize)]
pub(super) struct Oracle {
    pub word: Word,
    pub norm_cases: Vec<NormCase>,
    pub rope_cases: Vec<RopeCase>,
    pub gemm_gate_cases: Vec<GemmCase>,
}
fn validate_metadata(v: &Value) -> Result<(), String> {
    let sdk = &v["sdk_provenance"];
    if v["schema"] != "infinity-modern-native-primitives/v1"
        || v["native_library_sha256"]
            != "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
        || sdk["torch_version"] != "2.14.1+cu130"
        || sdk["torch_git_version"] != "5c4886908584029761b579af026dcfb627c84070"
        || sdk["cuda_runtime"] != "13.0"
        || sdk["cublaslt_version"] != 130200
        || sdk["all_blas_relocations_host_vendor"] != true
        || sdk["binding_verification_cuda_initialized"] != false
        || v["device_capability"]["visible_gpu_count"] != 1
        || v["device_capability"]["device"] != 0
    {
        return Err("native oracle SDK/library/pre-CUDA binding identity mismatch".into());
    }
    let is_hash = |value: &Value| {
        value
            .as_str()
            .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    if !is_hash(&v["helper_sha256"])
        || !v["sourceprovenance"]
            .as_object()
            .is_some_and(|m| m.len() >= 9 && m.values().all(is_hash))
    {
        return Err("native oracle source/helper hashes missing".into());
    }
    Ok(())
}
fn read_checked() -> (Oracle, Value) {
    let path = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_MODERN_REFERENCE")
            .expect("Root must generate the matched actual-native Modern reference"),
    );
    assert!(std::fs::metadata(&path).unwrap().len() <= 64 * 1024 * 1024);
    let value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    validate_metadata(&value).unwrap();
    let cap = value["device_capability"]["capability"].clone();
    let oracle: Oracle = serde_json::from_value(value).unwrap();
    let finite = |bits: &[u16]| bits.iter().all(|b| b & 0x7f80 != 0x7f80);
    assert_eq!(oracle.word.width, WIDTH);
    assert_eq!(oracle.word.eps, 1e-5);
    assert!(!oracle.word.bias);
    assert_eq!(oracle.word.word_bits.len(), 8 * WIDTH);
    assert_eq!(oracle.word.gamma_bits.len(), WIDTH);
    assert!(finite(&oracle.word.word_bits) && finite(&oracle.word.gamma_bits));
    assert_eq!(
        oracle.word.cases.iter().map(|c| c.rows).collect::<Vec<_>>(),
        [1, 3, 5, 128]
    );
    for c in &oracle.word.cases {
        assert_eq!(c.ids.len(), c.rows);
        assert!(c.ids.iter().all(|&id| id < 8));
        assert_eq!(c.expected_bits.len(), c.rows * WIDTH);
        assert!(finite(&c.expected_bits));
    }
    assert_eq!(oracle.norm_cases.len(), 24);
    for rows in [1, 3, 5, 128] {
        for eps in [1e-5f32, 1e-12] {
            for mode in ["plain", "add_store", "final"] {
                assert_eq!(
                    oracle
                        .norm_cases
                        .iter()
                        .filter(|c| c.rows == rows
                            && c.eps.to_bits() == eps.to_bits()
                            && c.mode == mode)
                        .count(),
                    1
                );
            }
        }
    }
    for c in &oracle.norm_cases {
        assert_eq!(c.input_bits.len(), c.rows * WIDTH);
        assert_eq!(c.gamma_bits.len(), WIDTH);
        assert_eq!(c.expected_bits.len(), c.rows * WIDTH);
        assert_eq!(
            c.residual_bits.len(),
            if c.mode == "plain" { 0 } else { c.rows * WIDTH }
        );
        assert_eq!(
            c.expected_sum_bits.len(),
            if c.mode == "add_store" {
                c.rows * WIDTH
            } else {
                0
            }
        );
        for v in [
            &c.input_bits,
            &c.gamma_bits,
            &c.residual_bits,
            &c.expected_bits,
            &c.expected_sum_bits,
        ] {
            assert!(finite(v));
        }
    }
    assert_eq!(oracle.rope_cases.len(), 2);
    for (c, theta) in oracle.rope_cases.iter().zip([160000.0, 10000.0]) {
        assert_eq!(c.theta, theta);
        assert_eq!(c.capacity, 8192);
        assert_eq!(c.rows, 5);
        assert_eq!(c.positions, [0, 1, 63, 64, 8191]);
        assert_eq!(c.qkv_input_bits.len(), 5 * 2304);
        assert_eq!(c.expected_qkv_bits.len(), 5 * 2304);
        assert_eq!(c.cosine_bits.len(), 8192 * 32);
        assert_eq!(c.sine_bits.len(), 8192 * 32);
        for v in [
            &c.qkv_input_bits,
            &c.expected_qkv_bits,
            &c.cosine_bits,
            &c.sine_bits,
        ] {
            assert!(finite(v));
        }
    }
    assert_eq!(
        oracle
            .gemm_gate_cases
            .iter()
            .map(|c| c.rows)
            .collect::<Vec<_>>(),
        [1, 3, 5, 128]
    );
    for c in &oracle.gemm_gate_cases {
        assert_eq!(
            c.weight_recipe,
            "f32_base127_over1024_plus_jitter5_over262144_v1"
        );
        assert_eq!(c.weight_shape, [2304, 768]);
        assert_eq!(c.input_bits.len(), c.rows * WIDTH);
        assert_eq!(c.expected_packed_bits.len(), c.rows * 2304);
        assert_eq!(c.expected_gate_bits.len(), c.rows * INTERMEDIATE);
        for v in [
            &c.input_bits,
            &c.expected_packed_bits,
            &c.expected_gate_bits,
        ] {
            assert!(finite(v));
        }
    }
    (oracle, cap)
}
pub(super) fn setup() -> (Oracle, Arc<Context>, Arc<Stream>, Kernels) {
    let (oracle, cap) = read_checked();
    let context = Context::new(0).unwrap();
    context.bind_to_thread().unwrap();
    let (major, minor) = context.compute_capability().unwrap();
    assert_eq!(cap, serde_json::json!([major, minor]));
    let stream = context.new_stream().unwrap();
    let kernels = Kernels::load(&context).unwrap();
    (oracle, context, stream, kernels)
}
#[test]
#[ignore = "CPU-only schema check requires Root generated native reference fixture"]
fn native_fixture_shapes_cpu() {
    let _ = read_checked();
}
pub(super) fn guarded(bits: &[u16], padding: usize) -> Vec<Bf16> {
    let mut v = vec![Bf16::from_bits(SENTINEL); padding];
    v.extend(bits.iter().copied().map(Bf16::from_bits));
    v.extend(vec![Bf16::from_bits(SENTINEL); 16]);
    v
}
pub(super) fn download(buffer: &DeviceBuffer<Bf16>, stream: &Stream) -> Vec<u16> {
    let mut data = vec![Bf16::from_bits(0); buffer.len()];
    buffer.copy_to_host(stream, &mut data).unwrap();
    data.into_iter().map(Bf16::to_bits).collect()
}
pub(super) fn assert_guard(data: &[u16], padding: usize, elements: usize) {
    assert!(data[..padding].iter().all(|&b| b == SENTINEL));
    assert!(data[padding + elements..].iter().all(|&b| b == SENTINEL));
}
#[test]
fn native_oracle_rejects_wrong_sdk_and_initialized_binding_proof_without_cuda() {
    let mut v = serde_json::json!({
        "schema":"infinity-modern-native-primitives/v1",
        "native_library_sha256":"b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e",
        "helper_sha256":"a".repeat(64),
        "sourceprovenance":(0..9).map(|i|(i.to_string(),Value::String("b".repeat(64)))).collect::<serde_json::Map<_,_>>(),
        "sdk_provenance":{"torch_version":"2.14.1+cu130","torch_git_version":"5c4886908584029761b579af026dcfb627c84070",
            "cuda_runtime":"13.0","cublaslt_version":130200,"all_blas_relocations_host_vendor":true,
            "binding_verification_cuda_initialized":false},
        "device_capability":{"device":0,"visible_gpu_count":1}
    });
    assert!(validate_metadata(&v).is_ok());
    v["sdk_provenance"]["binding_verification_cuda_initialized"] = true.into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["binding_verification_cuda_initialized"] = false.into();
    v["sdk_provenance"]["torch_git_version"] = "different".into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["torch_git_version"] = "5c4886908584029761b579af026dcfb627c84070".into();
    v["native_library_sha256"] = "different".into();
    assert!(validate_metadata(&v).is_err());
}
