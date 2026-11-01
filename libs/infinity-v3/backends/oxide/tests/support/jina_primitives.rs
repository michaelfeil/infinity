// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only actual-native Jina primitive fixture contract.
use crate::{start, tensor::Bf16};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) const WIDTH: usize = 512;
pub(crate) const INTERMEDIATE: usize = 2048;
pub(crate) const PACKED_MLP: usize = 2 * INTERMEDIATE;
pub(crate) const SENTINEL: u16 = 0x4242;
pub(crate) const ROWS: [usize; 4] = [1, 3, 5, 128];
pub(crate) const EPS_VALUES: [f32; 2] = [1e-12, 1e-5];
pub(crate) const VARIANTS: [&str; 3] = ["alternating", "type0_only", "type1_only"];
pub(crate) const SCHEMA: &str = "infinity-jina-native-primitives/v1";
pub(crate) const NATIVE_SHA: &str =
    "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e";
pub(crate) const TORCH_VERSION: &str = "2.14.1+cu130";
pub(crate) const TORCH_GIT: &str = "5c4886908584029761b579af026dcfb627c84070";
pub(crate) const CUDA_VERSION: &str = "13.0";
pub(crate) const WEIGHT_RECIPE: &str =
    "f32_base127_over1024_plus_jitter5_over262144_round_f16_once_v1";
pub(crate) const WEIGHT_FORMULA: &str =
    "round_f16_once(((i*17+11)%127-63)/1024 + ((i*7)%5-2)/262144)";
pub(crate) const INPUT_RECIPE: &str = "bf16_((i*13+7)%97-48)/16";
pub(crate) const WORD_RECIPE: &str =
    "bf16_eight_rows_half_offset_nearconstant_cancellation_subnormal_v1";
pub(crate) const TYPE_RECIPE: &str = "bf16_two_rows_half_subnormal_cancellation_offsets_v1";
pub(crate) const GAMMA_RECIPE: &str = "bf16_1+((i*7)%17-8)/32";
pub(crate) const BETA_RECIPE: &str = "bf16_((i*11)%19-9)/64";
pub(crate) const RESIDUAL_INPUT_RECIPE: &str =
    "bf16_nearconstant_bases_plus_halfoffset_cancellation_v1";
pub(crate) const RESIDUAL_RECIPE: &str = "bf16_cancelling_residuals_halfoffset_subnormal_v1";
pub(crate) const SOURCE_FILES: [&str; 14] = [
    "jina_native_reference.cpp",
    "build_jina_reference.py",
    "encoder_norm.h",
    "fast_kernels.h",
    "fp8_linear.h",
    "attention.h",
    "packed_input.h",
    "checked_size.h",
    "admission_error.h",
    "infinity_fa4.h",
    "block_paged.h",
    "verify_bindings.py",
    "generate_jina_reference.py",
    "test_jina_reference_cpu.py",
];
const FIXTURE_ENV: &str = "INFINITY_OXIDE_JINA_REFERENCE";

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
pub(crate) struct ResidualCase {
    pub(crate) rows: usize,
    pub(crate) eps: f32,
    pub(crate) input_bits: Vec<u16>,
    pub(crate) residual_bits: Vec<u16>,
    pub(crate) gamma_bits: Vec<u16>,
    pub(crate) beta_bits: Vec<u16>,
    pub(crate) expected_bits: Vec<u16>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct Residual {
    pub(crate) width: usize,
    pub(crate) eps_values: Vec<f32>,
    pub(crate) gamma_bits: Vec<u16>,
    pub(crate) beta_bits: Vec<u16>,
    pub(crate) cases: Vec<ResidualCase>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct GemmCase {
    pub(crate) rows: usize,
    pub(crate) input_bits: Vec<u16>,
    pub(crate) weight_recipe: String,
    #[serde(default)]
    pub(crate) weight_formula: String,
    #[serde(default)]
    pub(crate) weight_dtype: String,
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
    pub(crate) residual: Residual,
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
        || sdk["torch_version"] != TORCH_VERSION
        || sdk["torch_git_version"] != TORCH_GIT
        || sdk["cuda_runtime"] != CUDA_VERSION
        || sdk["cublaslt_version"] != 130200
        || sdk["all_blas_relocations_host_vendor"] != true
        || sdk["binding_verification_cuda_initialized"] != false
        || v["device_capability"]["visible_gpu_count"] != 1
        || v["device_capability"]["device"] != 0
    {
        return Err("native oracle SDK/library/pre-CUDA binding identity mismatch".into());
    }
    if recipes["weight"] != WEIGHT_RECIPE
        || recipes["weight_dtype"] != "float16"
        || recipes["gemm_input"] != INPUT_RECIPE
        || recipes["word"] != WORD_RECIPE
        || recipes["type"] != TYPE_RECIPE
        || recipes["gamma"] != GAMMA_RECIPE
        || recipes["beta"] != BETA_RECIPE
        || recipes["residual_input"] != RESIDUAL_INPUT_RECIPE
        || recipes["residual"] != RESIDUAL_RECIPE
        || recipes["activation"] != 4
        || recipes["activation_name"] != "Candle GELU tanh"
        || recipes["gate_first"] != true
    {
        return Err("native Jina gate-first/rounded-residual recipe mismatch".into());
    }
    let is_hash = |value: &Value| {
        value
            .as_str()
            .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    let Some(source_hashes) = v["sourceprovenance"].as_object() else {
        return Err("native oracle sourceprovenance missing".into());
    };
    if source_hashes.len() < SOURCE_FILES.len()
        || !source_hashes.values().all(is_hash)
        || !SOURCE_FILES
            .iter()
            .all(|name| is_hash(&v["sourceprovenance"][*name]))
    {
        return Err("native oracle source hashes incomplete".into());
    }
    if !is_hash(&v["helper_sha256"]) {
        return Err("native oracle helper hash missing".into());
    }
    Ok(())
}

pub(crate) fn validate_oracle(o: &Oracle) -> Result<(), String> {
    let w = &o.word;
    ensure(w.width == WIDTH, "word width mismatch")?;
    ensure(
        w.eps_values.len() == EPS_VALUES.len()
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
        "word gamma invalid",
    )?;
    ensure(
        w.beta_bits.len() == WIDTH && finite(&w.beta_bits),
        "word beta invalid",
    )?;
    ensure(
        w.gamma_bits != w.beta_bits,
        "word gamma and beta must differ",
    )?;
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
            "word case selection length mismatch",
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

    let r = &o.residual;
    ensure(r.width == WIDTH, "residual width mismatch")?;
    ensure(
        r.eps_values.len() == EPS_VALUES.len()
            && r.eps_values[0].to_bits() == EPS_VALUES[0].to_bits()
            && r.eps_values[1].to_bits() == EPS_VALUES[1].to_bits(),
        "residual eps values mismatch",
    )?;
    ensure(
        r.gamma_bits.len() == WIDTH && finite(&r.gamma_bits),
        "residual gamma invalid",
    )?;
    ensure(
        r.beta_bits.len() == WIDTH && finite(&r.beta_bits),
        "residual beta invalid",
    )?;
    ensure(
        r.gamma_bits != r.beta_bits,
        "residual gamma and beta must differ",
    )?;
    ensure(
        r.cases.len() == EPS_VALUES.len() * ROWS.len(),
        "residual case count mismatch",
    )?;
    for eps in EPS_VALUES {
        for rows in ROWS {
            let count = r
                .cases
                .iter()
                .filter(|c| c.rows == rows && c.eps.to_bits() == eps.to_bits())
                .count();
            ensure(count == 1, "residual case multiplicity mismatch")?;
        }
    }
    for c in &r.cases {
        ensure(
            ROWS.contains(&c.rows) && EPS_VALUES.iter().any(|e| e.to_bits() == c.eps.to_bits()),
            "residual case selection invalid",
        )?;
        for bits in [&c.input_bits, &c.residual_bits, &c.expected_bits] {
            ensure(
                bits.len() == c.rows * WIDTH && finite(bits),
                "residual case output invalid",
            )?;
        }
        ensure(
            c.gamma_bits.len() == WIDTH && finite(&c.gamma_bits),
            "residual case gamma invalid",
        )?;
        ensure(
            c.beta_bits.len() == WIDTH && finite(&c.beta_bits),
            "residual case beta invalid",
        )?;
    }

    ensure(
        o.gemm_gate_cases.len() == ROWS.len(),
        "gemm case count mismatch",
    )?;
    ensure(
        o.gemm_gate_cases.iter().map(|c| c.rows).collect::<Vec<_>>() == ROWS.to_vec(),
        "gemm rows mismatch",
    )?;
    for c in &o.gemm_gate_cases {
        ensure(
            c.weight_recipe == WEIGHT_RECIPE
                && (c.weight_formula.is_empty() || c.weight_formula == WEIGHT_FORMULA)
                && (c.weight_dtype.is_empty() || c.weight_dtype == "float16")
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
            .expect("Root must generate the matched actual-native Jina reference"),
    );
    assert!(
        std::fs::metadata(&path).unwrap().len() <= 64 * 1024 * 1024,
        "native Jina fixture exceeds 64MiB"
    );
    let value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    validate_metadata(&value).unwrap();
    let capability = value["device_capability"]["capability"].clone();
    let oracle: Oracle = serde_json::from_value(value).unwrap();
    validate_oracle(&oracle).unwrap();
    (oracle, capability)
}

pub(crate) struct PrimitiveKernels {
    pub(crate) word_type_norm: Function,
    pub(crate) rounded_residual: Function,
    pub(crate) gated_gelu: Function,
}

impl PrimitiveKernels {
    fn load(context: &Arc<Context>) -> Result<Self, infinity_v3_backend_core::BackendError> {
        let path = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
            .ok_or_else(|| start("Jina requires INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = Path::new(&path);
        let manifest_path = if path.is_dir() {
            path.join("manifest.json")
        } else {
            path.parent()
                .and_then(Path::parent)
                .ok_or_else(|| start("Jina primitive module needs its bundle manifest"))?
                .join("manifest.json")
        };
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(manifest_path).map_err(start)?).map_err(start)?;
        let exports = manifest
            .get("kernels")
            .and_then(Value::as_array)
            .ok_or_else(|| start("Jina primitive manifest lacks kernel exports"))?;
        for name in [
            "infinity_embedding_word_type_norm_512_exact",
            "infinity_layer_norm_512_rounded_native",
            "infinity_gated_gelu_up_first",
        ] {
            if !exports.iter().any(|entry| entry.as_str() == Some(name)) {
                return Err(start(format!(
                    "Jina primitive manifest lacks exact export {name}"
                )));
            }
        }
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
            word_type_norm: module
                .load_function("infinity_embedding_word_type_norm_512_exact")
                .map_err(start)?,
            rounded_residual: module
                .load_function("infinity_layer_norm_512_rounded_native")
                .map_err(start)?,
            gated_gelu: module
                .load_function("infinity_gated_gelu_up_first")
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

#[test]
#[ignore = "CPU-only schema check requires Root generated actual matched Jina reference fixture"]
fn jina_native_fixture_shapes_cpu() {
    let _ = read_checked();
}

fn baseline_metadata() -> Value {
    serde_json::json!({
        "schema": SCHEMA,
        "native_library_sha256": NATIVE_SHA,
        "helper_sha256": "a".repeat(64),
        "sourceprovenance": SOURCE_FILES
            .iter()
            .map(|name| ((*name).to_string(), Value::String("b".repeat(64))))
            .collect::<serde_json::Map<_, _>>(),
        "sdk_provenance": {
            "torch_version": TORCH_VERSION,
            "torch_git_version": TORCH_GIT,
            "cuda_runtime": CUDA_VERSION,
            "cublaslt_version": 130200,
            "all_blas_relocations_host_vendor": true,
            "binding_verification_cuda_initialized": false
        },
        "device_capability": {"device": 0, "visible_gpu_count": 1, "capability": [9, 0]},
        "recipes": {
            "weight": WEIGHT_RECIPE,
            "weight_dtype": "float16",
            "gemm_input": INPUT_RECIPE,
            "word": WORD_RECIPE,
            "type": TYPE_RECIPE,
            "gamma": GAMMA_RECIPE,
            "beta": BETA_RECIPE,
            "residual_input": RESIDUAL_INPUT_RECIPE,
            "residual": RESIDUAL_RECIPE,
            "activation": 4,
            "activation_name": "Candle GELU tanh",
            "gate_first": true
        }
    })
}

#[test]
fn native_jina_schema_rejects_wrong_sdk_hashes_and_recipe_without_cuda() {
    let mut v = baseline_metadata();
    assert!(validate_metadata(&v).is_ok());

    v["schema"] = "infinity-jina-native-primitives/v0".into();
    assert!(validate_metadata(&v).is_err());
    v["schema"] = SCHEMA.into();

    v["native_library_sha256"] = "different".into();
    assert!(validate_metadata(&v).is_err());
    v["native_library_sha256"] = NATIVE_SHA.into();

    v["sdk_provenance"]["torch_git_version"] = "different".into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["torch_git_version"] = TORCH_GIT.into();

    v["sdk_provenance"]["cublaslt_version"] = 130201.into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["cublaslt_version"] = 130200.into();

    v["sdk_provenance"]["binding_verification_cuda_initialized"] = true.into();
    assert!(validate_metadata(&v).is_err());
    v["sdk_provenance"]["binding_verification_cuda_initialized"] = false.into();

    let provenance = v["sourceprovenance"].as_object().unwrap().clone();
    let mut incomplete = provenance.clone();
    incomplete.remove(SOURCE_FILES[0]);
    v["sourceprovenance"] = Value::Object(incomplete);
    assert!(validate_metadata(&v).is_err());
    v["sourceprovenance"] = Value::Object(provenance);

    v["recipes"]["activation"] = 5.into();
    assert!(validate_metadata(&v).is_err());
    v["recipes"]["activation"] = 4.into();

    v["recipes"]["gate_first"] = false.into();
    assert!(validate_metadata(&v).is_err());
    v["recipes"]["gate_first"] = true.into();

    v["recipes"]["weight"] = "wrong".into();
    assert!(validate_metadata(&v).is_err());
    v["recipes"]["weight"] = WEIGHT_RECIPE.into();

    assert!(validate_metadata(&v).is_ok());
}
