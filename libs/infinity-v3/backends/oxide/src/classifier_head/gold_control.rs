// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Test-only binding to Root's local, native-default trained-head oracle.
use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Read, path::Path};

const STAGES: [&str; 5] = [
    "input",
    "gold_token_score",
    "gold_pooler",
    "gold_tanh",
    "gold_score",
];
const MODELS: [(&str, usize, usize, &str); 2] = [
    ("minilm", 384, 1, "INFINITY_OXIDE_CLASSIFIER_MINILM_MODEL"),
    ("ner", 768, 9, "INFINITY_OXIDE_CLASSIFIER_NER_MODEL"),
];

fn assert_host_vendor() {
    use std::ffi::{c_char, c_void, CStr};
    #[repr(C)]
    struct DlInfo {
        filename: *const c_char,
        base: *mut c_void,
        symbol: *const c_char,
        address: *mut c_void,
    }
    type Dladdr = unsafe extern "C" fn(*const c_void, *mut DlInfo) -> i32;
    // SAFETY: querying the current process resolves a stable libc dladdr ABI.
    let process = libloading::os::unix::Library::this();
    let dladdr = unsafe { process.get::<Dladdr>(b"dladdr\0") }.unwrap();
    for (name, env, symbols) in [
        (
            "libcublas.so.13",
            "INFINITY_OXIDE_CLASSIFIER_HOST_CUBLAS_SHA256",
            &[b"cublasGemmEx\0".as_slice()][..],
        ),
        (
            "libcublasLt.so.13",
            "INFINITY_OXIDE_CLASSIFIER_HOST_CUBLASLT_SHA256",
            &[
                b"cublasLtMatmul\0".as_slice(),
                b"cublasLtGetVersion\0".as_slice(),
            ][..],
        ),
    ] {
        let expected = Path::new("/usr/local/cuda-13.1/lib64")
            .join(name)
            .canonicalize()
            .unwrap();
        assert_eq!(
            checksum(&expected).unwrap(),
            std::env::var(env).expect("set frozen host vendor SHA256")
        );
        // SAFETY: these pinned vendor DSOs remain open for every symbol query.
        let library = unsafe { libloading::Library::new(name) }.unwrap();
        for symbol in symbols {
            // SAFETY: only resolve the symbol address; no untyped call occurs.
            let address = *unsafe { library.get::<*const c_void>(symbol) }.unwrap();
            let mut info = DlInfo {
                filename: std::ptr::null(),
                base: std::ptr::null_mut(),
                symbol: std::ptr::null(),
                address: std::ptr::null_mut(),
            };
            // SAFETY: address belongs to a live DSO and info is a writable ABI object.
            assert_ne!(unsafe { dladdr(address, &mut info) }, 0);
            assert!(!info.filename.is_null());
            // SAFETY: successful dladdr returns a live NUL-terminated filename.
            let actual = PathBuf::from(unsafe { CStr::from_ptr(info.filename) }.to_str().unwrap())
                .canonicalize()
                .unwrap();
            assert_eq!(
                actual, expected,
                "Rust BLAS function bound outside the frozen host pair"
            );
        }
        if name == "libcublasLt.so.13" {
            // SAFETY: pinned public version-query ABI takes no arguments.
            let version =
                unsafe { library.get::<unsafe extern "C" fn() -> usize>(b"cublasLtGetVersion\0") }
                    .unwrap();
            assert_eq!(unsafe { version() }, 130200);
        }
    }
}

fn number(v: &Value) -> Result<usize, String> {
    v.as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| "invalid unsigned size".into())
}
fn checksum(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn plain_name(v: &str) -> bool {
    !v.is_empty() && v.bytes().all(|x| x.is_ascii_alphanumeric() || x == b'_')
}
fn validate_manifest(d: &Value) -> Result<(), String> {
    if d["schema"] != "infinity-oxide-trained-head-oracle-v1"
        || d["mode"] != "run"
        || d["status"] != "ok"
        || d["case_count"] != 14
        || d["flags"]["preferred_blas_library"] != "cublas"
        || d["flags"]["allow_bf16_reduced_precision_reduction"] != true
        || d["scope"]["backbone_replay_claimed"] != false
        || d["scope"]["performance_claimed"] != false
        || d["vendor"]["proof"]["native_sdk_sha256"]
            != "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
        || d["vendor"]["proof"]["relocation_proof"]["all_blas_relocations_host_vendor"] != true
        || d["vendor"]["proof"]["relocation_proof"]["probe_dependency_cublaslt_version"] != 130200
        || number(&d["bundle_bytes"])? > 16 * 1024 * 1024
    {
        return Err("wrong native-default oracle identity/policy".into());
    }
    for (key, h, c, _) in MODELS {
        let cases = d["models"][key]["cases"]
            .as_array()
            .ok_or("missing model cases")?;
        let mut names = BTreeSet::new();
        for case in cases {
            let n = number(&case["rows"])?;
            let kind = case["kind"].as_str().ok_or("missing case kind")?;
            if !((kind == "finite" && [1, 2, 3, 8, 32].contains(&n))
                || (n == 1 && ["zero", "signed_zero"].contains(&kind)))
            {
                return Err("wrong case matrix".into());
            }
            let name = case["name"].as_str().ok_or("missing case name")?;
            if name != format!("{key}_h{h}_m{n}_{kind}") || !plain_name(name) || !names.insert(name)
            {
                return Err("duplicate or unsafe case name".into());
            }
            let stages = case["stages"].as_object().ok_or("missing stages")?;
            if stages.len() != 5 {
                return Err("missing oracle stages".into());
            }
            for (stage, width) in STAGES.into_iter().zip([h, c, h, h, c]) {
                let m = stages.get(stage).ok_or("missing oracle stage")?;
                let bytes = n
                    .checked_mul(width)
                    .and_then(|x| x.checked_mul(2))
                    .ok_or("stage overflow")?;
                if m["dtype"] != "BF16"
                    || m["little_endian"] != true
                    || m["shape"] != serde_json::json!([n, width])
                    || m["strides"] != serde_json::json!([width, 1])
                    || number(&m["bytes"])? != bytes
                    || number(&m["elements"])? != bytes / 2
                    || m["file"] != format!("{stage}.bf16")
                {
                    return Err("oracle stage shape/type/byte ordering mismatch".into());
                }
            }
        }
        if names.len() != 7 {
            return Err("incomplete model matrix".into());
        }
    }
    Ok(())
}
fn read_stage(path: &Path, meta: &Value) -> Result<Vec<u16>, String> {
    let size = number(&meta["bytes"])?;
    if size > 16 * 1024 * 1024
        || size % 2 != 0
        || fs::metadata(path).map_err(|e| e.to_string())?.len() != size as u64
    {
        return Err("truncated or oversized BF16 stage".into());
    }
    if meta["dtype"] != "BF16"
        || meta["little_endian"] != true
        || checksum(path)? != meta["sha256"].as_str().ok_or("missing stage checksum")?
    {
        return Err("stage dtype/byte order/checksum mismatch".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let bits = bytes
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect::<Vec<_>>();
    if bits.iter().any(|x| x & 0x7f80 == 0x7f80) {
        return Err("non-finite BF16 oracle".into());
    }
    Ok(bits)
}
fn assert_bf16(actual: &[Bf16], expected: &[u16]) {
    let bits = actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mismatches = bits
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert_eq!(
        bits.len(),
        expected.len(),
        "native BF16 stage length differs"
    );
    assert!(
        mismatches.is_empty(),
        "full native BF16 stage bits differ: {} of {}, first {:?}",
        mismatches.len(),
        expected.len(),
        mismatches.first().map(|&i| (i, bits[i], expected[i]))
    );
}
fn assert_f32(actual: &[f32], expected: &[u16]) {
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected
            .iter()
            .map(|x| u32::from(*x) << 16)
            .collect::<Vec<_>>(),
        "full native F32 readback bits differ, including signed zero"
    );
}
fn batch(rows: usize, mode: ReadoutMode) -> Batch {
    let indices = (0..rows)
        .map(|x| u32::try_from(x).unwrap())
        .collect::<Vec<_>>();
    Batch {
        input_ids: vec![0; rows],
        position_ids: vec![0; rows],
        token_type_ids: vec![0; rows],
        cumulative_seq_lengths: (0..=rows).map(|x| u32::try_from(x).unwrap()).collect(),
        max_length: 1,
        pooled_indices: if mode == ReadoutMode::Sequence {
            indices.clone()
        } else {
            vec![]
        },
        raw_indices: if mode == ReadoutMode::Tokens {
            indices
        } else {
            vec![]
        },
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
#[allow(clippy::too_many_arguments)]
fn project(
    head: &mut ClassifierHead,
    pooler: bool,
    provider: &LtProvider,
    blas: &Blas,
    stream: &Arc<Stream>,
    input: &DeviceBuffer<Bf16>,
    rows: usize,
    output: &mut DeviceBuffer<Bf16>,
) {
    let linear = if pooler {
        head.pooler.as_mut().expect("stored trained pooler")
    } else {
        &mut head.classifier
    };
    let prepared = unsafe {
        linear.plan.prepare(
            provider,
            blas,
            rows,
            stream,
            input,
            &linear.device.weight,
            &linear.device.bias,
            output,
        )
    }
    .unwrap();
    unsafe {
        linear.plan.launch(
            &prepared,
            provider,
            blas,
            &head.kernels.bias_seed,
            stream,
            input,
            &linear.device.weight,
            &linear.device.bias,
            output,
        )
    }
    .unwrap();
}
#[allow(clippy::too_many_arguments)]
fn full(
    head: &mut ClassifierHead,
    provider: &LtProvider,
    blas: &Blas,
    stream: &Arc<Stream>,
    input: &DeviceBuffer<Bf16>,
    bounds: u64,
    rows: usize,
    mode: ReadoutMode,
) -> Vec<f32> {
    let layout = head.scratch_layout((rows, rows, rows)).unwrap();
    let mut scratch = ClassifierScratchBuffers::new(stream.clone(), layout).unwrap();
    let b = batch(rows, mode);
    unsafe { head.prepare(provider, blas, &b, mode, stream, input, &mut scratch) }.unwrap();
    unsafe { head.enqueue(provider, blas, stream, input, bounds, &mut scratch) }.unwrap();
    let mut out = scratch.logits().to_host_vec(stream).unwrap();
    out.truncate(rows * head.readout.width);
    out
}

// Generic public contract fixture: no native output, checkpoint tensor,
// benchmark report or local qualification file is required by ordinary CI.
fn contract_manifest() -> Value {
    let mut manifest = serde_json::json!({
        "schema":"infinity-oxide-trained-head-oracle-v1", "mode":"run", "status":"ok", "case_count":14,
        "bundle_bytes":0,
        "flags":{"preferred_blas_library":"cublas","allow_bf16_reduced_precision_reduction":true},
        "scope":{"backbone_replay_claimed":false,"performance_claimed":false},
        "vendor":{"proof":{
            "native_sdk_sha256":"b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e",
            "relocation_proof":{"all_blas_relocations_host_vendor":true,"probe_dependency_cublaslt_version":130200}
        }},
        "models":{}
    });
    let mut total = 0usize;
    for (key, h, c, _) in MODELS {
        let cases=[(1,"finite"),(2,"finite"),(3,"finite"),(8,"finite"),(32,"finite"),(1,"zero"),(1,"signed_zero")]
            .map(|(rows,kind)| {
                let mut stages=serde_json::Map::new();
                for (stage,width) in STAGES.into_iter().zip([h,c,h,h,c]) {
                    let bytes=rows*width*2;total+=bytes;
                    stages.insert(stage.into(),serde_json::json!({"dtype":"BF16","little_endian":true,"shape":[rows,width],"strides":[width,1],"bytes":bytes,"elements":bytes/2,"file":format!("{stage}.bf16")}));
                }
                serde_json::json!({"rows":rows,"kind":kind,"name":format!("{key}_h{h}_m{rows}_{kind}"),"stages":stages})
            });
        manifest["models"][key] = serde_json::json!({"cases":cases});
    }
    manifest["bundle_bytes"] = serde_json::json!(total);
    manifest
}

#[test]
fn manifest_contract_rejects_each_shape_policy_and_missing_case_error() {
    let valid = contract_manifest();
    validate_manifest(&valid).unwrap();
    let mut bad = valid.clone();
    bad["models"]["ner"]["cases"][0]["stages"]["gold_pooler"]["shape"] =
        serde_json::json!([1, 769]);
    assert!(validate_manifest(&bad).is_err());
    let mut bad = valid.clone();
    bad["flags"]["preferred_blas_library"] = serde_json::json!("cublaslt");
    assert!(validate_manifest(&bad).is_err());
    let mut bad = valid.clone();
    bad["models"]["minilm"]["cases"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(validate_manifest(&bad).is_err());
    let mut bad = valid;
    bad["models"]["ner"]["cases"][1]["name"] = serde_json::json!("../outside");
    assert!(validate_manifest(&bad).is_err());
}

#[test]
fn stage_loader_rejects_corruption_truncation_and_wrong_endianness() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "oxide-head-gold-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let path = dir.join("test.bf16");
    fs::write(&path, [0u8, 128, 128, 63]).unwrap();
    let mut meta = serde_json::json!({"bytes":4,"dtype":"BF16","little_endian":true,"sha256":checksum(&path).unwrap()});
    assert_eq!(read_stage(&path, &meta).unwrap(), vec![0x8000, 0x3f80]);
    fs::write(&path, [1u8, 128, 128, 63]).unwrap();
    assert!(read_stage(&path, &meta).is_err());
    fs::write(&path, [0u8, 128]).unwrap();
    assert!(read_stage(&path, &meta).is_err());
    fs::write(&path, [0u8, 128, 128, 63]).unwrap();
    meta["little_endian"] = serde_json::json!(false);
    assert!(read_stage(&path, &meta).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "Root GPU: actual trained head oracle, both model paths and kernels53 required"]
fn trained_heads_match_full_native_default_bf16_stages() {
    let root = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_CLASSIFIER_GOLD_DIR")
            .expect("set actual Root gold directory"),
    );
    let d: Value = serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    validate_manifest(&d).unwrap();
    let mut assets = Vec::new();
    let mut bytes = 0usize;
    for (key, h, c, env) in MODELS {
        let model = PathBuf::from(std::env::var_os(env).expect("set actual trained model path"));
        let pre = &d["models"][key]["preflight"];
        assert_eq!(
            checksum(&model.join("config.json")).unwrap(),
            pre["config_sha256"].as_str().unwrap()
        );
        assert_eq!(
            checksum(&model.join("model.safetensors")).unwrap(),
            pre["checkpoint_sha256"].as_str().unwrap()
        );
        for case in d["models"][key]["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let n = number(&case["rows"]).unwrap();
            let data = STAGES.map(|s| {
                let m = &case["stages"][s];
                let v = read_stage(
                    &root
                        .join("cases")
                        .join(name)
                        .join(m["file"].as_str().unwrap()),
                    m,
                )
                .unwrap();
                bytes += v.len() * 2;
                v
            });
            assets.push((key, h, c, model.clone(), n, data, name.to_owned()));
        }
    }
    assert_eq!(assets.len(), 14);
    assert_eq!(bytes, number(&d["bundle_bytes"]).unwrap());
    assert_host_vendor();
    assert!(
        Context::device_count().unwrap() > 0,
        "GPU gate must execute, never silently skip"
    );
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let provider = LtProvider::new_native_default(context.clone(), stream.clone()).unwrap();
    let blas = Blas::new(context.clone(), stream.clone()).unwrap();
    let mut checked = 0;
    for (key, h, c, _) in MODELS {
        let model = assets.iter().find(|x| x.0 == key).unwrap().3.clone();
        let config = crate::model::BertConfig::load_classifier(&model).unwrap();
        let checkpoint = Checkpoint::open(&model).unwrap();
        let host = ClassifierHeadHost::load(&checkpoint, &config).unwrap();
        assert!(host.pooler.is_some());
        let mut head = ClassifierHead::new(host, &context, &stream, &provider).unwrap();
        context.bind_to_thread().unwrap();
        for (_, _, _, _, n, data, name) in assets.iter().filter(|x| x.0 == key) {
            let rows = *n;
            let input_values = data[0]
                .iter()
                .copied()
                .map(Bf16::from_bits)
                .collect::<Vec<_>>();
            let input = DeviceBuffer::from_host(&stream, &input_values).unwrap();
            eprintln!("native trained-head stage: {name}/input");
            assert_bf16(&input.to_host_vec(&stream).unwrap(), &data[0]);
            checked += 1;
            let bounds = DeviceBuffer::from_host(
                &stream,
                &(0..=rows)
                    .map(|x| u32::try_from(x).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let mut direct = DeviceBuffer::zeroed(&stream, rows * c).unwrap();
            project(
                &mut head,
                false,
                &provider,
                &blas,
                &stream,
                &input,
                rows,
                &mut direct,
            );
            let direct_bits = direct.to_host_vec(&stream).unwrap();
            eprintln!("native trained-head stage: {name}/gold_token_score");
            assert_bf16(&direct_bits, &data[1]);
            checked += 1;
            let mut corrupted = data[1].clone();
            corrupted[0] ^= 1;
            assert_ne!(
                direct_bits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                corrupted
            );
            let mut pooled = DeviceBuffer::zeroed(&stream, rows * h).unwrap();
            project(
                &mut head,
                true,
                &provider,
                &blas,
                &stream,
                &input,
                rows,
                &mut pooled,
            );
            eprintln!("native trained-head stage: {name}/gold_pooler");
            assert_bf16(&pooled.to_host_vec(&stream).unwrap(), &data[2]);
            checked += 1;
            let mut ptr = pooled.cu_deviceptr();
            let mut count = u32_size(rows * h).unwrap();
            launch(
                &head.kernels.tanh,
                &stream,
                blocks(rows * h).unwrap(),
                0,
                &mut [arg(&mut ptr), arg(&mut count)],
            )
            .unwrap();
            eprintln!("native trained-head stage: {name}/gold_tanh");
            assert_bf16(&pooled.to_host_vec(&stream).unwrap(), &data[3]);
            checked += 1;
            eprintln!("native trained-head stage: {name}/full_readout");
            assert_f32(
                &full(
                    &mut head,
                    &provider,
                    &blas,
                    &stream,
                    &input,
                    bounds.cu_deviceptr(),
                    rows,
                    ReadoutMode::Tokens,
                ),
                &data[1],
            );
            assert_f32(
                &full(
                    &mut head,
                    &provider,
                    &blas,
                    &stream,
                    &input,
                    bounds.cu_deviceptr(),
                    rows,
                    ReadoutMode::Sequence,
                ),
                &data[4],
            );
            checked += 1;
            let queries = (
                head.classifier.plan.native_lt_queries(&provider),
                head.pooler
                    .as_ref()
                    .unwrap()
                    .plan
                    .native_lt_queries(&provider),
            );
            let entries = (
                head.classifier.plan.native_lt_cache_len(&provider),
                head.pooler
                    .as_ref()
                    .unwrap()
                    .plan
                    .native_lt_cache_len(&provider),
            );
            assert_f32(
                &full(
                    &mut head,
                    &provider,
                    &blas,
                    &stream,
                    &input,
                    bounds.cu_deviceptr(),
                    rows,
                    ReadoutMode::Sequence,
                ),
                &data[4],
            );
            assert_eq!(
                queries,
                (
                    head.classifier.plan.native_lt_queries(&provider),
                    head.pooler
                        .as_ref()
                        .unwrap()
                        .plan
                        .native_lt_queries(&provider)
                ),
                "repeated shapes cannot query native heuristics"
            );
            assert_eq!(
                entries,
                (
                    head.classifier.plan.native_lt_cache_len(&provider),
                    head.pooler
                        .as_ref()
                        .unwrap()
                        .plan
                        .native_lt_cache_len(&provider)
                )
            );
        }
    }
    assert_eq!(checked, 70, "all full native stage tensors must be checked");
}
