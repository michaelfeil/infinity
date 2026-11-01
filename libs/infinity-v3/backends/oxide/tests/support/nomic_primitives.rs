// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only actual-native Nomic primitive fixture contract.
use crate::{start, tensor::Bf16};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) const WIDTH: usize = 768;
pub(crate) const INTERMEDIATE: usize = 3072;
pub(crate) const PACKED_QKV: usize = 2304;
pub(crate) const PACKED_MLP: usize = 2 * INTERMEDIATE;
pub(crate) const HEAD_DIM: usize = 64;
pub(crate) const ROPE_CAPACITY: usize = 8192;
pub(crate) const SENTINEL: u16 = 0x4242;
pub(crate) const ROWS: [usize; 4] = [1, 3, 5, 128];
pub(crate) const EPS_VALUES: [f32; 2] = [1e-12, 1e-5];
pub(crate) const VARIANTS: [&str; 3] = ["alternating", "type0_only", "type1_only"];
pub(crate) const ROPE_POSITIONS: [u32; 5] = [0, 1, 63, 64, 8191];
pub(crate) const SCHEMA: &str = "infinity-nomic-native-primitives/v1";
pub(crate) const NATIVE_SHA: &str =
    "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e";
pub(crate) const TORCH_GIT: &str = "5c4886908584029761b579af026dcfb627c84070";
pub(crate) const WEIGHT_RECIPE: &str = "f32_base127_over1024_plus_jitter5_over262144_v1";
const FIXTURE_ENV: &str = "INFINITY_OXIDE_NOMIC_REFERENCE";

#[derive(Clone, Deserialize)]
pub(crate) struct WordCase {
    pub(crate) rows: usize,
    pub(crate) eps: f32,
    pub(crate) variant: String,
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Vec<u32>,
    pub(crate) expected_bits: Vec<u16>,
}
#[derive(Clone, Deserialize)]
pub(crate) struct Word {
    pub(crate) width: usize,
    pub(crate) eps_values: Vec<f32>,
    pub(crate) bias: bool,
    pub(crate) word_bits: Vec<u16>,
    pub(crate) type_bits: Vec<Vec<u16>>,
    pub(crate) gamma_bits: Vec<u16>,
    pub(crate) beta_bits: Vec<u16>,
    pub(crate) cases: Vec<WordCase>,
}
#[derive(Clone, Deserialize)]
pub(crate) struct RopeCase {
    pub(crate) theta: f64,
    pub(crate) capacity: usize,
    pub(crate) rows: usize,
    pub(crate) positions: Vec<u32>,
    pub(crate) table_shape: [usize; 2],
    pub(crate) source_packed_bits: Vec<u16>,
    pub(crate) cosine_bits: Vec<u16>,
    pub(crate) sine_bits: Vec<u16>,
    pub(crate) expected_rotated_bits: Vec<u16>,
    pub(crate) input_unchanged: bool,
    pub(crate) v_chunks_unchanged: bool,
}
#[derive(Clone, Deserialize)]
pub(crate) struct GemmCase {
    pub(crate) rows: usize,
    pub(crate) input_bits: Vec<u16>,
    pub(crate) weight_recipe: String,
    pub(crate) weight_shape: [usize; 2],
    pub(crate) gate_chunk_shape: [usize; 2],
    pub(crate) up_chunk_shape: [usize; 2],
    pub(crate) expected_packed_bits: Vec<u16>,
    pub(crate) expected_gate_bits: Vec<u16>,
    pub(crate) used_ordinary: bool,
}
#[derive(Clone, Deserialize)]
pub(crate) struct Oracle {
    pub(crate) word: Word,
    pub(crate) rope: RopeCase,
    pub(crate) gemm_gate_cases: Vec<GemmCase>,
}
fn finite(bits: &[u16]) -> bool {
    bits.iter().all(|b| b & 0x7f80 != 0x7f80)
}
fn ensure(ok: bool, message: &str) -> Result<(), String> {
    ok.then_some(()).ok_or_else(|| message.into())
}
pub(crate) fn validate_metadata(v: &Value) -> Result<(), String> {
    let sdk = &v["sdk_provenance"];
    let recipes = &v["recipes"];
    if v["schema"] != SCHEMA
        || v["native_library_sha256"] != NATIVE_SHA
        || sdk["torch_version"] != "2.14.1+cu130"
        || sdk["torch_git_version"] != TORCH_GIT
        || sdk["cuda_runtime"] != "13.0"
        || sdk["cublaslt_version"] != 130200
        || sdk["all_blas_relocations_host_vendor"] != true
        || sdk["binding_verification_cuda_initialized"] != false
        || v["device_capability"]["visible_gpu_count"] != 1
        || v["device_capability"]["device"] != 0
    {
        return Err("native oracle SDK/library/pre-CUDA binding identity mismatch".into());
    }
    if recipes["weight"] != WEIGHT_RECIPE
        || recipes["activation"] != 5
        || recipes["gate_first"] != true
        || recipes["rope_theta"].as_f64() != Some(1000.0)
        || recipes["rope_capacity"] != 8192
    {
        return Err("native gate-first recipe mismatch".into());
    }
    let is_hash = |value: &Value| {
        value
            .as_str()
            .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    if !is_hash(&v["helper_sha256"])
        || !v["sourceprovenance"]
            .as_object()
            .is_some_and(|m| m.len() >= 8 && m.values().all(is_hash))
    {
        return Err("native oracle source/helper hashes missing".into());
    }
    Ok(())
}
pub(crate) fn validate_oracle(o: &Oracle) -> Result<(), String> {
    let w = &o.word;
    ensure(w.width == WIDTH, "word width mismatch")?;
    ensure(
        w.eps_values.len() == 2
            && w.eps_values[0].to_bits() == EPS_VALUES[0].to_bits()
            && w.eps_values[1].to_bits() == EPS_VALUES[1].to_bits(),
        "word eps values mismatch",
    )?;
    ensure(w.bias, "word.bias must be true")?;
    ensure(
        w.word_bits.len() == 8 * WIDTH && finite(&w.word_bits),
        "word table invalid",
    )?;
    ensure(
        w.type_bits.len() == 2 && w.type_bits.iter().all(|r| r.len() == WIDTH && finite(r)),
        "type rows invalid",
    )?;
    ensure(
        w.type_bits[0] != w.type_bits[1],
        "type rows must be distinct",
    )?;
    ensure(
        w.gamma_bits.len() == WIDTH && finite(&w.gamma_bits),
        "gamma invalid",
    )?;
    ensure(
        w.beta_bits.len() == WIDTH && finite(&w.beta_bits),
        "beta invalid",
    )?;
    ensure(w.gamma_bits != w.beta_bits, "gamma and beta must differ")?;
    ensure(w.cases.len() == 24, "word case count mismatch")?;
    for eps in EPS_VALUES {
        for rows in ROWS {
            for variant in VARIANTS {
                let count = w
                    .cases
                    .iter()
                    .filter(|c| {
                        c.rows == rows && c.eps.to_bits() == eps.to_bits() && c.variant == variant
                    })
                    .count();
                ensure(count == 1, "word case multiplicity mismatch")?;
            }
        }
    }
    for c in &w.cases {
        ensure(
            ROWS.contains(&c.rows)
                && EPS_VALUES.iter().any(|e| e.to_bits() == c.eps.to_bits())
                && VARIANTS.contains(&c.variant.as_str()),
            "word case selection invalid",
        )?;
        ensure(
            c.ids.len() == c.rows && c.type_ids.len() == c.rows,
            "word case id length mismatch",
        )?;
        ensure(c.ids.iter().all(|&id| id < 8), "word case id out of bounds")?;
        ensure(
            c.type_ids.iter().all(|&id| id < 2),
            "word case type id out of bounds",
        )?;
        ensure(
            c.expected_bits.len() == c.rows * WIDTH && finite(&c.expected_bits),
            "word case output invalid",
        )?;
    }
    let r = &o.rope;
    ensure(
        r.theta.to_bits() == 1000.0f64.to_bits(),
        "rope theta mismatch",
    )?;
    ensure(r.capacity == ROPE_CAPACITY, "rope capacity mismatch")?;
    ensure(
        r.rows == 5 && r.positions.as_slice() == ROPE_POSITIONS.as_slice(),
        "rope positions mismatch",
    )?;
    ensure(
        r.positions.iter().all(|&p| (p as usize) < r.capacity),
        "rope position out of bounds",
    )?;
    let table = r.capacity * (HEAD_DIM / 2);
    ensure(
        r.table_shape == [r.capacity, HEAD_DIM / 2],
        "rope table shape mismatch",
    )?;
    ensure(
        r.cosine_bits.len() == table
            && r.sine_bits.len() == table
            && finite(&r.cosine_bits)
            && finite(&r.sine_bits),
        "rope table invalid",
    )?;
    ensure(
        r.source_packed_bits.len() == r.rows * PACKED_QKV
            && r.expected_rotated_bits.len() == r.rows * PACKED_QKV,
        "rope packed length mismatch",
    )?;
    ensure(
        finite(&r.source_packed_bits) && finite(&r.expected_rotated_bits),
        "rope packed output invalid",
    )?;
    ensure(
        r.input_unchanged && r.v_chunks_unchanged,
        "rope invariants not proven",
    )?;
    ensure(o.gemm_gate_cases.len() == 4, "gemm case count mismatch")?;
    ensure(
        o.gemm_gate_cases.iter().map(|c| c.rows).collect::<Vec<_>>() == ROWS.to_vec(),
        "gemm rows mismatch",
    )?;
    for c in &o.gemm_gate_cases {
        ensure(
            c.weight_recipe == WEIGHT_RECIPE
                && c.weight_shape == [PACKED_MLP, WIDTH]
                && c.gate_chunk_shape == [INTERMEDIATE, WIDTH]
                && c.up_chunk_shape == [INTERMEDIATE, WIDTH],
            "gemm recipe/shape mismatch",
        )?;
        ensure(
            c.input_bits.len() == c.rows * WIDTH && finite(&c.input_bits),
            "gemm input invalid",
        )?;
        ensure(
            c.expected_packed_bits.len() == c.rows * PACKED_MLP && finite(&c.expected_packed_bits),
            "gemm packed output invalid",
        )?;
        ensure(
            c.expected_gate_bits.len() == c.rows * INTERMEDIATE && finite(&c.expected_gate_bits),
            "gemm gate output invalid",
        )?;
    }
    Ok(())
}
pub(crate) fn read_checked() -> (Oracle, Value) {
    let path = PathBuf::from(
        std::env::var_os(FIXTURE_ENV)
            .expect("Root must generate the matched actual-native Nomic reference"),
    );
    assert!(
        std::fs::metadata(&path).unwrap().len() <= 64 * 1024 * 1024,
        "native fixture exceeds 64MiB"
    );
    let value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    validate_metadata(&value).unwrap();
    let capability = value["device_capability"]["capability"].clone();
    let oracle: Oracle = serde_json::from_value(value).unwrap();
    validate_oracle(&oracle).unwrap();
    (oracle, capability)
}
pub(crate) struct PrimitiveKernels {
    pub(crate) table_init: Arc<Function>,
    pub(crate) word_type_norm: Function,
    pub(crate) rope: Function,
    pub(crate) silu_mul: Function,
}
impl PrimitiveKernels {
    fn load(context: &Arc<Context>) -> Result<Self, infinity_v3_backend_core::BackendError> {
        let path = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
            .ok_or_else(|| start("Nomic requires INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = Path::new(&path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&path).map_err(start)?;
        let module = context.load_module_from_image(&bytes).map_err(start)?;
        Ok(Self {
            table_init: Arc::new(
                module
                    .load_function("infinity_base_rope_tables_64_bf16")
                    .map_err(start)?,
            ),
            word_type_norm: module
                .load_function("infinity_embedding_word_type_norm_768_exact")
                .map_err(start)?,
            rope: module
                .load_function("infinity_rope_neox_hfma")
                .map_err(start)?,
            silu_mul: module
                .load_function("infinity_silu_mul_bf16_vector8_native")
                .map_err(start)?,
        })
    }
}
pub(crate) fn setup() -> (Oracle, Arc<Context>, Arc<Stream>, PrimitiveKernels) {
    let (oracle, capability) = read_checked();
    let context = Context::new(0).unwrap();
    context.bind_to_thread().unwrap();
    let (major, minor) = context.compute_capability().unwrap();
    assert_eq!(capability, serde_json::json!([major, minor]));
    let stream = context.new_stream().unwrap();
    let kernels = PrimitiveKernels::load(&context).unwrap();
    (oracle, context, stream, kernels)
}
pub(crate) fn guarded(bits: &[u16], padding: usize) -> Vec<Bf16> {
    let mut v = vec![Bf16::from_bits(SENTINEL); padding];
    v.extend(bits.iter().copied().map(Bf16::from_bits));
    v.extend(vec![Bf16::from_bits(SENTINEL); 16]);
    v
}
pub(crate) fn download(buffer: &DeviceBuffer<Bf16>, stream: &Stream) -> Vec<u16> {
    let mut data = vec![Bf16::from_bits(0); buffer.len()];
    buffer.copy_to_host(stream, &mut data).unwrap();
    data.into_iter().map(Bf16::to_bits).collect()
}
pub(crate) fn assert_guard(data: &[u16], padding: usize, elements: usize) {
    assert!(data[..padding].iter().all(|&b| b == SENTINEL));
    assert!(data[padding + elements..].iter().all(|&b| b == SENTINEL));
}
fn baseline_metadata() -> Value {
    serde_json::json!({
        "schema": SCHEMA,
        "native_library_sha256": NATIVE_SHA,
        "helper_sha256": "a".repeat(64),
        "sourceprovenance": (0..8)
            .map(|i| (i.to_string(), Value::String("b".repeat(64))))
            .collect::<serde_json::Map<_, _>>(),
        "sdk_provenance": {
            "torch_version": "2.14.1+cu130",
            "torch_git_version": TORCH_GIT,
            "cuda_runtime": "13.0",
            "cublaslt_version": 130200,
            "all_blas_relocations_host_vendor": true,
            "binding_verification_cuda_initialized": false
        },
        "device_capability": {"device": 0, "visible_gpu_count": 1, "capability": [9, 0]},
        "recipes": {
            "weight": WEIGHT_RECIPE,
            "activation": 5,
            "gate_first": true,
            "rope_theta": 1000.0,
            "rope_capacity": 8192
        }
    })
}
fn baseline_oracle() -> Oracle {
    let mut cases = Vec::new();
    for eps in EPS_VALUES {
        for rows in ROWS {
            for variant in VARIANTS {
                let type_ids = match variant {
                    "alternating" => (0..rows).map(|i| (i % 2) as u32).collect(),
                    "type0_only" => vec![0; rows],
                    _ => vec![1; rows],
                };
                cases.push(WordCase {
                    rows,
                    eps,
                    variant: variant.into(),
                    ids: (0..rows).map(|i| (i % 8) as u32).collect(),
                    type_ids,
                    expected_bits: vec![0; rows * WIDTH],
                });
            }
        }
    }
    let table = ROPE_CAPACITY * (HEAD_DIM / 2);
    Oracle {
        word: Word {
            width: WIDTH,
            eps_values: EPS_VALUES.to_vec(),
            bias: true,
            word_bits: vec![0; 8 * WIDTH],
            type_bits: vec![vec![0; WIDTH], vec![0x3f80; WIDTH]],
            gamma_bits: vec![0x3f80; WIDTH],
            beta_bits: vec![0; WIDTH],
            cases,
        },
        rope: RopeCase {
            theta: 1000.0,
            capacity: ROPE_CAPACITY,
            rows: 5,
            positions: ROPE_POSITIONS.to_vec(),
            table_shape: [ROPE_CAPACITY, HEAD_DIM / 2],
            source_packed_bits: vec![0; 5 * PACKED_QKV],
            cosine_bits: vec![0; table],
            sine_bits: vec![0; table],
            expected_rotated_bits: vec![0; 5 * PACKED_QKV],
            input_unchanged: true,
            v_chunks_unchanged: true,
        },
        gemm_gate_cases: ROWS
            .iter()
            .map(|&rows| GemmCase {
                rows,
                input_bits: vec![0; rows * WIDTH],
                weight_recipe: WEIGHT_RECIPE.into(),
                weight_shape: [PACKED_MLP, WIDTH],
                gate_chunk_shape: [INTERMEDIATE, WIDTH],
                up_chunk_shape: [INTERMEDIATE, WIDTH],
                expected_packed_bits: vec![0; rows * PACKED_MLP],
                expected_gate_bits: vec![0; rows * INTERMEDIATE],
                used_ordinary: false,
            })
            .collect(),
    }
}
#[test]
#[ignore = "CPU-only schema check requires Root generated actual matched Nomic reference fixture"]
fn native_fixture_shapes_cpu() {
    let _ = read_checked();
}
#[test]
fn native_oracle_rejects_wrong_sdk_and_initialized_binding_proof_without_cuda() {
    let mut v = baseline_metadata();
    assert!(validate_metadata(&v).is_ok());
    v["sdk_provenance"]["binding_verification_cuda_initialized"] = true.into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["binding_verification_cuda_initialized"] = false.into();
    v["sdk_provenance"]["torch_git_version"] = "different".into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["torch_git_version"] = TORCH_GIT.into();
    v["sdk_provenance"]["cublaslt_version"] = 130201.into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["cublaslt_version"] = 130200.into();
    v["native_library_sha256"] = "different".into();
    assert!(validate_metadata(&v).is_err());
    v["native_library_sha256"] = NATIVE_SHA.into();
    v["recipes"]["gate_first"] = false.into();
    assert!(validate_metadata(&v).is_err());
    v["recipes"]["gate_first"] = true.into();
    let provenance = v["sourceprovenance"].as_object().unwrap().clone();
    let mut short = provenance.clone();
    short.remove("7");
    v["sourceprovenance"] = Value::Object(short);
    assert!(validate_metadata(&v).is_err());
    v["sourceprovenance"] = Value::Object(provenance);
    assert!(validate_metadata(&v).is_ok());
}
#[test]
fn native_oracle_rejects_wrong_type_ids_count_theta_and_table_without_cuda() {
    let baseline = baseline_oracle();
    assert!(validate_oracle(&baseline).is_ok());
    let mut o = baseline.clone();
    o.word.cases[0].type_ids[0] = 2;
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.word.cases.pop();
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.rope.theta = 10000.0;
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.rope.cosine_bits.pop();
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.word.cases[0].ids[0] = 8;
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.word.type_bits[1] = o.word.type_bits[0].clone();
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.word.bias = false;
    assert!(validate_oracle(&o).is_err());
    let mut o = baseline.clone();
    o.gemm_gate_cases[0].weight_recipe = "wrong".into();
    assert!(validate_oracle(&o).is_err());
    assert!(validate_oracle(&baseline).is_ok());
}
