// Native b66 CAPI control packet and Rust consumer.
//
// The packet preserves the original checkpoint config, every weight binding and
// the bounded CPU preflight. It never loads the Native library, creates a CUDA
// context, imports Torch or reads tensor payloads. The consumer below is the
// actual b66 call sequence a Root GPU job can execute with a loaded library.

use crate::checkpoint::{start, CheckpointMetadata};
use crate::qwen3_vl_config::Qwen3VlConfig;
use crate::qwen3_vl_inventory::{InventoryReport, InventoryScope};
use crate::qwen3_vl_vision::{frame_plan, ImageGrid};
use crate::qwen3_vl_weights::{NativeWeight, WeightPlan};
use infinity_v3_backend_core::BackendError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::Path;

pub const PACKET_SCHEMA: &str = "oxide-qwen3-vl-native-b66-control/v1";
pub const BASE_COMMIT: &str = "ba466347edffc76574ea59d44d4b50ff443697ea";
pub const SDK_PREFIX: &str = "b66";
pub const CUBLASLT_PIN: u64 = 130_200;
pub const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub struct TeiConfig {
    pub hidden: i64,
    pub heads: i64,
    pub layers: i64,
    pub intermediate: i64,
    pub vocab: i64,
    pub positions: i64,
    pub types: i64,
    pub epsilon: f64,
    pub activation: i32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TeiImage {
    pub pixels: *const f32,
    pub rows: i64,
    pub patch_dim: i64,
    pub grid: [i64; 3],
    pub merge_size: i64,
    pub token_start: i64,
    pub token_count: i64,
    pub sequence_start: i64,
    pub feature_key: [u8; 32],
    pub has_feature_key: u8,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TeiWeight {
    pub name: *const c_char,
    pub shape: *const i64,
    pub rank: usize,
    pub dtype: i32,
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CaseImage {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub merge_size: usize,
    pub token_start: usize,
    pub token_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CaseSpec {
    pub name: String,
    pub scope: String,
    pub tokens: usize,
    pub images: Vec<CaseImage>,
    pub expected_merged_rows: usize,
    pub expected_frames: usize,
    pub expected_deepstack_rows: usize,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ShardSummary {
    pub shard: String,
    pub file_bytes: u64,
    pub data_start: u64,
    pub data_length: u64,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct InventorySummary {
    pub tensor_count: usize,
    pub vision_count: usize,
    pub deepstack_count: usize,
    pub text_count: usize,
    pub total_elements: u64,
    pub total_bytes: u64,
    pub shards: Vec<ShardSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SdkRecord {
    pub expected_prefix: String,
    pub candidate_paths: Vec<String>,
    pub sha256: Option<String>,
    pub matches_expected_prefix: bool,
    pub cublaslt_expected: u64,
    pub library_loaded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct IsolationRecord {
    pub cpu_affinity: String,
    pub job_limit: usize,
    pub cuda_visible_devices: String,
    pub cuda_hidden: bool,
    pub gpu_context: bool,
    pub torch_loaded: bool,
    pub docker_used: bool,
    pub nvidia_queries: u64,
    pub network_used: bool,
    pub tensor_payload_read: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RootControlPacket {
    pub schema: String,
    pub base_commit: String,
    pub checkpoint: String,
    pub original_config: Value,
    pub config_sha256: String,
    pub text_config_sha256: String,
    pub config_options: BTreeMap<String, String>,
    pub runtime_profile: String,
    pub runtime_options: BTreeMap<String, String>,
    pub tei_config: TeiConfig,
    pub model_type: String,
    pub dtype: String,
    pub scope: String,
    pub inventory: InventorySummary,
    pub weights: Vec<NativeWeight>,
    pub cases: Vec<CaseSpec>,
    pub sdk: SdkRecord,
    pub isolation: IsolationRecord,
    pub root_gpu_followup: Value,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn flatten(prefix: &str, value: &Value, out: &mut BTreeMap<String, String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let name = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&name, child, out);
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                flatten(&format!("{prefix}.{index}"), child, out);
            }
        }
        Value::String(value) => {
            out.insert(prefix.to_owned(), value.clone());
        }
        Value::Bool(value) => {
            out.insert(prefix.to_owned(), value.to_string());
        }
        Value::Number(value) => {
            out.insert(prefix.to_owned(), value.to_string());
        }
        Value::Null => {}
    }
}

pub fn config_options(config: &Qwen3VlConfig) -> BTreeMap<String, String> {
    let mut options = BTreeMap::new();
    flatten("", &config.raw, &mut options);
    options
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeProfile {
    NativeProduction,
    NativeGraphEvidence,
    OxideLowOverhead,
}

impl RuntimeProfile {
    pub const fn name(self) -> &'static str {
        match self {
            Self::NativeProduction => "native_production",
            Self::NativeGraphEvidence => "native_graph_evidence",
            Self::OxideLowOverhead => "oxide_low_overhead",
        }
    }
}

pub fn source_required_options() -> BTreeMap<String, String> {
    let mut options = BTreeMap::new();
    for (key, value) in [
        ("_multi_vector", "false"),
        ("_transformers_math", "false"),
        ("_skip_final_norm", "false"),
        ("_fp8_dynamic", "false"),
        ("_fp8_scope", "mlp"),
        ("_pooling", "embedding"),
        ("_embedding_pool", "last_token"),
        ("_dense_count", "0"),
        ("_classifier", "false"),
    ] {
        options.insert(key.to_owned(), value.to_owned());
    }
    options
}

pub fn runtime_options(profile: RuntimeProfile) -> BTreeMap<String, String> {
    let mut options = source_required_options();
    match profile {
        RuntimeProfile::NativeProduction => {
            for (key, value) in [
                ("_cuda_graphs", "true"),
                ("_media_cuda_graphs", "false"),
                ("_cuda_graph_entries", "4"),
                ("_cuda_graph_reuse", "3"),
                ("_cuda_graph_max_bytes", "536870912"),
                ("_cuda_graph_max_captures", "8"),
                ("_cuda_graph_max_tokens", "16384"),
                ("_prefix_cache_max_bytes", "0"),
                ("_prefix_cache_max_tokens", "8192"),
                ("_prefix_cache_entries", "64"),
            ] {
                options.insert(key.to_owned(), value.to_owned());
            }
        }
        RuntimeProfile::NativeGraphEvidence => {
            options = runtime_options(RuntimeProfile::NativeProduction);
            options.insert("_media_cuda_graphs".to_owned(), "true".to_owned());
        }
        RuntimeProfile::OxideLowOverhead => {
            options.insert("_cuda_graphs".to_owned(), "false".to_owned());
            options.insert("_media_cuda_graphs".to_owned(), "false".to_owned());
            options.insert("_prefix_cache_max_bytes".to_owned(), "0".to_owned());
        }
    }
    options
}

/// Native adapter activation codes (`backends/libtorch/src/lib.rs`):
/// gelu=0, tanh/erf GELU aliases=1, relu=2, silu=3.
pub fn activation_code(name: &str) -> Result<i32, BackendError> {
    match name {
        "gelu" => Ok(0),
        "gelu_new" | "gelu_pytorch_tanh" => Ok(1),
        "gelu_fast" => Ok(0),
        "relu" => Ok(2),
        "silu" | "swish" => Ok(3),
        other => Err(start(format!("unsupported decoder activation {other}"))),
    }
}

fn tei_config(config: &Qwen3VlConfig) -> Result<TeiConfig, BackendError> {
    Ok(TeiConfig {
        hidden: i64::try_from(config.text.hidden_size).map_err(start)?,
        heads: i64::try_from(config.text.num_attention_heads).map_err(start)?,
        layers: i64::try_from(config.text.num_hidden_layers).map_err(start)?,
        intermediate: i64::try_from(config.text.intermediate_size).map_err(start)?,
        vocab: i64::try_from(config.text.vocab_size).map_err(start)?,
        positions: i64::try_from(config.text.max_position_embeddings).map_err(start)?,
        types: 0,
        epsilon: config.text.rms_norm_eps,
        activation: activation_code(&config.text.hidden_act)?,
    })
}

pub fn sdk_record() -> SdkRecord {
    let candidates = vec![
        "/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build/libtei_torch.so".to_owned(),
        "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so".to_owned(),
    ];
    SdkRecord {
        expected_prefix: SDK_PREFIX.to_owned(),
        candidate_paths: candidates,
        sha256: None,
        matches_expected_prefix: false,
        cublaslt_expected: CUBLASLT_PIN,
        library_loaded: false,
    }
}

impl RootControlPacket {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        checkpoint: &Path,
        config: &Qwen3VlConfig,
        metadata: &CheckpointMetadata,
        report: &InventoryReport,
        plan: &WeightPlan,
        cases: Vec<CaseSpec>,
        profile: RuntimeProfile,
    ) -> Result<Self, BackendError> {
        plan.verify(report)?;
        let mut weights = Vec::with_capacity(plan.tensor_count);
        for shard in &plan.shards {
            weights.extend(shard.weights.iter().cloned());
        }
        if weights.len() != report.bound.len() {
            return Err(start("control packet weight set does not match inventory"));
        }
        let config_file = std::fs::read(checkpoint.join("config.json")).map_err(start)?;
        let text_value = config
            .raw
            .get("text_config")
            .ok_or_else(|| start("missing text_config for packet hash"))?;
        let text_bytes = serde_json::to_vec(text_value).map_err(start)?;
        let shards = plan
            .shards
            .iter()
            .map(|shard| ShardSummary {
                shard: shard.shard.clone(),
                file_bytes: shard.file_bytes,
                data_start: shard.data_start,
                data_length: shard.data_length,
                tensor_count: shard.tensor_count,
            })
            .collect();
        let packet = Self {
            schema: PACKET_SCHEMA.to_owned(),
            base_commit: BASE_COMMIT.to_owned(),
            checkpoint: checkpoint.display().to_string(),
            original_config: config.raw.clone(),
            config_sha256: sha256_hex(&config_file),
            text_config_sha256: sha256_hex(&text_bytes),
            config_options: config_options(config),
            runtime_profile: profile.name().to_owned(),
            runtime_options: runtime_options(profile),
            tei_config: tei_config(config)?,
            model_type: config.model_type.clone(),
            dtype: config.vision.dtype.name().to_owned(),
            scope: match report.scope {
                InventoryScope::TowerOnly => "tower_only".to_owned(),
                InventoryScope::Full => "full".to_owned(),
            },
            inventory: InventorySummary {
                tensor_count: report.bound.len(),
                vision_count: report.vision_count,
                deepstack_count: report.deepstack_count,
                text_count: report.text_count,
                total_elements: report.total_elements,
                total_bytes: report.total_bytes,
                shards,
            },
            weights,
            cases,
            sdk: sdk_record(),
            isolation: IsolationRecord {
                cpu_affinity: "72-79".to_owned(),
                job_limit: 4,
                cuda_visible_devices: String::new(),
                cuda_hidden: true,
                gpu_context: false,
                torch_loaded: false,
                docker_used: false,
                nvidia_queries: 0,
                network_used: false,
                tensor_payload_read: false,
            },
            root_gpu_followup: serde_json::json!({
                "run_now": false,
                "requires": ["Root-controlled real GPU", "Native b66 CAPI library", "original checkpoint config", "all weights"],
                "steps": [
                    "tei_device_preflight('cuda') under the Root device policy",
                    "tei_create with TeiConfig and dtype=2 (BF16)",
                    "apply every config option exactly, then low-overhead runtime options",
                    "tei_load_safetensors for every shard with all packet weights (no drops)",
                    "tei_ready",
                    "run the RealFixture cases; capture pooled/raw and image_features quality",
                    "record lifetime and performance only after paired Native gold exists"
                ]
            }),
        };
        packet.verify(metadata)?;
        Ok(packet)
    }

    pub fn verify(&self, metadata: &CheckpointMetadata) -> Result<(), BackendError> {
        if self.schema != PACKET_SCHEMA {
            return Err(start("unexpected control packet schema"));
        }
        if self.base_commit != BASE_COMMIT {
            return Err(start("control packet base commit mismatch"));
        }
        let parsed = Qwen3VlConfig::from_value(self.original_config.clone())?;
        if parsed.raw != self.original_config {
            return Err(start("packet config was mutated while parsing"));
        }
        let config_file = std::fs::read(metadata.root.join("config.json")).map_err(start)?;
        if sha256_hex(&config_file) != self.config_sha256 {
            return Err(start("packet config file hash mismatch"));
        }
        let text_value = self
            .original_config
            .get("text_config")
            .ok_or_else(|| start("packet is missing text_config"))?;
        if sha256_hex(&serde_json::to_vec(text_value).map_err(start)?) != self.text_config_sha256 {
            return Err(start("packet text_config hash mismatch"));
        }
        let expected_options = config_options(&parsed);
        if expected_options != self.config_options {
            return Err(start(
                "packet config options do not preserve original config",
            ));
        }
        for key in self.runtime_options.keys() {
            if !key.starts_with('_') {
                return Err(start(format!(
                    "runtime option {key} is not a private runtime key"
                )));
            }
        }
        for case in &self.cases {
            let images = case
                .images
                .iter()
                .map(|image| {
                    ImageGrid::new(
                        image.frames,
                        image.height,
                        image.width,
                        image.merge_size,
                        image.token_start,
                        image.token_count,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let plan = frame_plan(&images)?;
            let merged: usize = images.iter().map(|image| image.token_count).sum();
            let deepstack = merged
                .checked_mul(parsed.vision.deepstack_visual_indexes.len())
                .ok_or_else(|| start("case deepstack row overflow"))?;
            let max_end = images
                .iter()
                .map(|image| image.token_start + image.token_count)
                .max()
                .unwrap_or(0);
            if case.tokens < max_end {
                return Err(start(format!(
                    "case {} declares {} tokens but an image ends at {max_end}",
                    case.name, case.tokens
                )));
            }
            for (image, case_image) in images.iter().zip(case.images.iter()) {
                if image.merge_size != parsed.vision.spatial_merge_size {
                    return Err(start(format!(
                        "case {} image merge {} does not match config merge {}",
                        case.name, case_image.merge_size, parsed.vision.spatial_merge_size
                    )));
                }
            }
            if case.expected_merged_rows != merged {
                return Err(start(format!(
                    "case {} merged rows {} != {merged}",
                    case.name, case.expected_merged_rows
                )));
            }
            if case.expected_frames != plan.frame_lengths.len() {
                return Err(start(format!(
                    "case {} frames {} != {}",
                    case.name,
                    case.expected_frames,
                    plan.frame_lengths.len()
                )));
            }
            if case.expected_deepstack_rows != deepstack {
                return Err(start(format!(
                    "case {} deepstack rows {} != {deepstack}",
                    case.name, case.expected_deepstack_rows
                )));
            }
        }
        let graphs = self.runtime_options.get("_cuda_graphs").map(String::as_str);
        let media = self
            .runtime_options
            .get("_media_cuda_graphs")
            .map(String::as_str);
        match self.runtime_profile.as_str() {
            "native_production" => {
                if graphs != Some("true") || media != Some("false") {
                    return Err(start(
                        "native production profile must keep CUDA graphs on and media graphs at the source default",
                    ));
                }
            }
            "native_graph_evidence" => {
                if graphs != Some("true") || media != Some("true") {
                    return Err(start(
                        "native graph-evidence profile must enable both CUDA and media graphs",
                    ));
                }
            }
            "oxide_low_overhead" => {
                if graphs != Some("false") || media != Some("false") {
                    return Err(start(
                        "Oxide low-overhead profile must disable CUDA and media graphs",
                    ));
                }
            }
            other => return Err(start(format!("unknown runtime profile {other}"))),
        }
        if !self.isolation.cuda_hidden || self.isolation.cpu_affinity != "72-79" {
            return Err(start(
                "packet isolation record is not the offline CPU policy",
            ));
        }
        if self.inventory.tensor_count != self.weights.len() {
            return Err(start(
                "packet inventory count does not match weight entries",
            ));
        }
        if self.scope == "full" && self.weights.len() != metadata.tensors.len() {
            return Err(start("full packet dropped or invented a checkpoint tensor"));
        }
        let mut names = BTreeSet::new();
        let mut source_ends: BTreeMap<String, u64> = BTreeMap::new();
        let mut total_bytes = 0u64;
        for weight in &self.weights {
            if !names.insert(weight.name.clone()) {
                return Err(start(format!("duplicate packet weight {}", weight.name)));
            }
            let Some(meta) = metadata.tensors.get(&weight.original_name) else {
                return Err(start(format!(
                    "packet weight {} is not in the checkpoint",
                    weight.original_name
                )));
            };
            if meta.data_offsets.0 != weight.source_offset
                || meta.data_offsets.1 - meta.data_offsets.0 != weight.source_length
            {
                return Err(start(format!(
                    "packet weight {} source range drifted",
                    weight.name
                )));
            }
            if weight.dtype != meta.dtype.name() {
                return Err(start(format!(
                    "packet weight {} dtype drifted",
                    weight.name
                )));
            }
            if weight.shape != meta.shape || weight.rank != meta.shape.len() {
                return Err(start(format!(
                    "packet weight {} shape/rank drifted",
                    weight.name
                )));
            }
            let Some(shard) = metadata.shards.get(&weight.shard) else {
                return Err(start(format!(
                    "packet weight {} names an unknown shard",
                    weight.name
                )));
            };
            let end = weight
                .source_offset
                .checked_add(weight.source_length)
                .ok_or_else(|| start("packet source range overflow"))?;
            if end > shard.data_length {
                return Err(start(format!(
                    "packet weight {} exceeds shard data",
                    weight.name
                )));
            }
            if let Some(previous_source_end) = source_ends.get(&weight.shard) {
                if weight.source_offset < *previous_source_end {
                    return Err(start(format!(
                        "packet weight {} is not in increasing source order",
                        weight.name
                    )));
                }
            }
            source_ends.insert(weight.shard.clone(), end);
            total_bytes = total_bytes
                .checked_add(weight.source_length)
                .ok_or_else(|| start("packet byte total overflow"))?;
        }
        if total_bytes == 0 || total_bytes != self.inventory.total_bytes {
            return Err(start("packet byte total does not match inventory"));
        }
        if self.sdk.expected_prefix != SDK_PREFIX
            || self.sdk.cublaslt_expected != CUBLASLT_PIN
            || self.sdk.library_loaded
        {
            return Err(start("packet SDK record is not the b66 CPU control record"));
        }
        if self.isolation.tensor_payload_read
            || self.isolation.gpu_context
            || self.isolation.torch_loaded
            || self.isolation.docker_used
            || self.isolation.nvidia_queries != 0
            || self.isolation.network_used
        {
            return Err(start("packet isolation record claims forbidden work"));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, BackendError> {
        let text = serde_json::to_string_pretty(self).map_err(start)?;
        if text.len() > MAX_PACKET_BYTES {
            return Err(start("control packet exceeds 64 MiB"));
        }
        Ok(text)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.len() > MAX_PACKET_BYTES {
            return Err(start("control packet exceeds 64 MiB"));
        }
        serde_json::from_slice(bytes).map_err(start)
    }

    pub fn preflight(&self, metadata: &CheckpointMetadata) -> Result<CpuPreflight, BackendError> {
        self.verify(metadata)?;
        let text = self.to_json()?;
        Ok(CpuPreflight {
            packet_sha256: sha256_hex(text.as_bytes()),
            packet_bytes: text.len(),
            tensor_count: self.weights.len(),
            total_bytes: self.inventory.total_bytes,
            config_preserved: true,
            library_loaded: false,
            gpu_context: false,
            payload_read: false,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CpuPreflight {
    pub packet_sha256: String,
    pub packet_bytes: usize,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub config_preserved: bool,
    pub library_loaded: bool,
    pub gpu_context: bool,
    pub payload_read: bool,
}

pub type TeiCreate = unsafe extern "C" fn(*const TeiConfig, *const c_char, i32) -> *mut c_void;
pub type TeiDevicePreflight = unsafe extern "C" fn(*const c_char) -> i32;
pub type TeiOption = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> i32;
pub type TeiLoadSafetensors = unsafe extern "C" fn(
    *mut c_void,
    i32,
    u64,
    u64,
    *const TeiWeight,
    usize,
    usize,
    usize,
    i32,
) -> i32;
pub type TeiReady = unsafe extern "C" fn(*mut c_void) -> i32;
pub type TeiForward = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i32,
    i64,
    i64,
    i32,
    *const i64,
    usize,
    *const i64,
    usize,
    *mut f32,
    usize,
    *const i64,
    *const TeiImage,
    usize,
    *const c_void,
    usize,
) -> i32;
pub type TeiDestroy = unsafe extern "C" fn(*mut c_void);
pub type TeiError = unsafe extern "C" fn() -> *const c_char;

#[derive(Clone, Copy)]
pub struct NativeB66Api {
    pub create: TeiCreate,
    pub device_preflight: TeiDevicePreflight,
    pub option: TeiOption,
    pub load_safetensors: TeiLoadSafetensors,
    pub ready: TeiReady,
    pub forward: TeiForward,
    pub destroy: TeiDestroy,
    pub error: TeiError,
}

#[derive(Clone, Debug)]
pub struct ForwardFixture {
    pub ids: Vec<i64>,
    pub types: Vec<i64>,
    pub positions: Vec<i64>,
    pub cumulative: Vec<i32>,
    pub media_positions: Option<Vec<i64>>,
    pub images: Vec<FixtureImage>,
    pub pooled: Vec<i64>,
    pub raw: Vec<i64>,
    pub capacity: usize,
}

#[derive(Clone, Debug)]
pub struct FixtureImage {
    pub pixels: Vec<f32>,
    pub rows: usize,
    pub patch_dim: usize,
    pub grid: [usize; 3],
    pub merge_size: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub sequence_start: usize,
}

pub struct ConsumerShard<'a> {
    pub path: &'a str,
    pub data_start: u64,
    pub data_length: u64,
    pub weights: Vec<(&'a str, Vec<i64>, i32, u64, u64)>,
}

pub struct ConsumerPlan<'a> {
    pub config: TeiConfig,
    pub device: &'a str,
    pub dtype: i32,
    pub options: Vec<(&'a str, &'a str)>,
    pub shards: Vec<ConsumerShard<'a>>,
    pub fixtures: Vec<ForwardFixture>,
}

impl<'a> ConsumerPlan<'a> {
    pub fn from_packet(packet: &'a RootControlPacket) -> Self {
        let mut options: Vec<(&str, &str)> = packet
            .config_options
            .iter()
            .chain(packet.runtime_options.iter())
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        options.sort_unstable();
        let mut shards = Vec::new();
        for summary in &packet.inventory.shards {
            let weights = packet
                .weights
                .iter()
                .filter(|weight| weight.shard == summary.shard)
                .map(|weight| {
                    (
                        weight.original_name.as_str(),
                        weight.shape.iter().map(|dim| *dim as i64).collect(),
                        match weight.dtype.as_str() {
                            "F32" => 0,
                            "F16" => 1,
                            _ => 2,
                        },
                        weight.source_offset,
                        weight.source_length,
                    )
                })
                .collect();
            shards.push(ConsumerShard {
                path: summary.shard.as_str(),
                data_start: summary.data_start,
                data_length: summary.data_length,
                weights,
            });
        }
        Self {
            config: packet.tei_config,
            device: "cuda",
            dtype: 2,
            options,
            shards,
            fixtures: Vec::new(),
        }
    }

    /// Actual b66 CAPI sequence. Only a Root GPU process with a loaded b66
    /// library and real fixtures should call this. The CPU evidence never does.
    ///
    /// # Safety
    ///
    /// Every function pointer in `api` must come from a b66-compatible
    /// `libtei_torch.so`. `root` must contain the shards named by the packet,
    /// and `fixture` buffers must outlive the call. The caller must own the GPU
    /// and stream lifetime.
    pub unsafe fn run(
        &self,
        api: &NativeB66Api,
        root: &Path,
        fixture: &ForwardFixture,
    ) -> Result<Vec<f32>, String> {
        unsafe {
            let device = CString::new(self.device).map_err(|error| error.to_string())?;
            let status = (api.device_preflight)(device.as_ptr());
            if status != 0 {
                return Err(error_text(api));
            }
            let handle = (api.create)(&self.config, device.as_ptr(), self.dtype);
            if handle.is_null() {
                return Err(error_text(api));
            }
            let guard = HandleGuard {
                handle,
                destroy: api.destroy,
            };
            for (key, value) in &self.options {
                let key = CString::new(*key).map_err(|error| error.to_string())?;
                let value = CString::new(*value).map_err(|error| error.to_string())?;
                if (api.option)(guard.handle, key.as_ptr(), value.as_ptr()) != 0 {
                    return Err(error_text(api));
                }
            }
            for shard in &self.shards {
                let path = root.join(shard.path);
                let file = std::fs::File::open(&path).map_err(|error| error.to_string())?;
                let mut names = Vec::with_capacity(shard.weights.len());
                let mut shapes = Vec::with_capacity(shard.weights.len());
                let mut native = Vec::with_capacity(shard.weights.len());
                for (name, shape, _dtype, _offset, _length) in &shard.weights {
                    names.push(CString::new(*name).map_err(|error| error.to_string())?);
                    shapes.push(shape.clone());
                    native.push(null_weight());
                }
                for (index, weight) in native.iter_mut().enumerate() {
                    weight.name = names[index].as_ptr();
                    weight.shape = shapes[index].as_ptr();
                    weight.rank = shapes[index].len();
                    weight.dtype = shard.weights[index].2;
                    weight.offset = shard.weights[index].3;
                    weight.length = shard.weights[index].4;
                }
                use std::os::fd::AsRawFd;
                let status = (api.load_safetensors)(
                    guard.handle,
                    file.as_raw_fd(),
                    shard.data_start,
                    shard.data_length,
                    native.as_ptr(),
                    native.len(),
                    4,
                    64 * 1024 * 1024,
                    0,
                );
                if status != 0 {
                    return Err(error_text(api));
                }
            }
            if (api.ready)(guard.handle) != 0 {
                return Err(error_text(api));
            }
            let mut output = vec![0f32; fixture.capacity.max(1)];
            let mut images = Vec::with_capacity(fixture.images.len());
            for image in &fixture.images {
                if image.pixels.len()
                    != image
                        .rows
                        .checked_mul(image.patch_dim)
                        .ok_or_else(|| "image pixel size overflow".to_owned())?
                {
                    return Err("fixture image pixel length mismatch".to_owned());
                }
                images.push(TeiImage {
                    pixels: image.pixels.as_ptr(),
                    rows: image.rows as i64,
                    patch_dim: image.patch_dim as i64,
                    grid: [
                        image.grid[0] as i64,
                        image.grid[1] as i64,
                        image.grid[2] as i64,
                    ],
                    merge_size: image.merge_size as i64,
                    token_start: image.token_start as i64,
                    token_count: image.token_count as i64,
                    sequence_start: image.sequence_start as i64,
                    feature_key: [0u8; 32],
                    has_feature_key: 0,
                });
            }
            let pooled: Vec<i64> = fixture.pooled.clone();
            let tokens = fixture.cumulative.last().copied().unwrap_or(0).max(0) as usize;
            if fixture.ids.len() != tokens
                || fixture.types.len() != tokens
                || fixture.positions.len() != tokens
            {
                return Err("fixture token buffers do not match cumulative token count".to_owned());
            }
            if let Some(media) = fixture.media_positions.as_ref() {
                if media.len() != tokens.saturating_mul(3) {
                    return Err("fixture media_positions length is not 3*tokens".to_owned());
                }
            }
            let mut max_sequence = 0i64;
            for pair in fixture.cumulative.windows(2) {
                if pair[0] < 0 || pair[1] < pair[0] {
                    return Err("fixture cumulative offsets are not monotonic".to_owned());
                }
                max_sequence = max_sequence.max(i64::from(pair[1] - pair[0]));
            }
            if max_sequence <= 0 || max_sequence as usize > tokens {
                return Err("fixture max_sequence is invalid".to_owned());
            }
            if fixture
                .raw
                .iter()
                .any(|index| *index < 0 || *index >= fixture.cumulative.len() as i64 - 1)
            {
                return Err("fixture raw sequence index is out of range".to_owned());
            }
            let media_positions = fixture.media_positions.as_deref();
            let media_ptr = media_positions.map_or(std::ptr::null(), |values| values.as_ptr());
            let status = (api.forward)(
                guard.handle,
                fixture.ids.as_ptr(),
                fixture.types.as_ptr(),
                fixture.positions.as_ptr(),
                fixture.cumulative.as_ptr(),
                fixture.cumulative.len() as i64 - 1,
                max_sequence,
                0,
                if pooled.is_empty() {
                    std::ptr::null()
                } else {
                    pooled.as_ptr()
                },
                pooled.len(),
                if fixture.raw.is_empty() {
                    std::ptr::null()
                } else {
                    fixture.raw.as_ptr()
                },
                fixture.raw.len(),
                output.as_mut_ptr(),
                output.len(),
                media_ptr,
                if images.is_empty() {
                    std::ptr::null()
                } else {
                    images.as_ptr()
                },
                images.len(),
                std::ptr::null(),
                0,
            );
            if status != 0 {
                return Err(error_text(api));
            }
            Ok(output)
        }
    }
}

struct HandleGuard {
    handle: *mut c_void,
    destroy: TeiDestroy,
}

impl Drop for HandleGuard {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) };
    }
}

fn null_weight() -> TeiWeight {
    TeiWeight {
        name: std::ptr::null(),
        shape: std::ptr::null(),
        rank: 0,
        dtype: 0,
        offset: 0,
        length: 0,
    }
}

unsafe fn error_text(api: &NativeB66Api) -> String {
    unsafe {
        let pointer = (api.error)();
        if pointer.is_null() {
            return "Native b66 call failed without an error string".to_owned();
        }
        std::ffi::CStr::from_ptr(pointer)
            .to_string_lossy()
            .into_owned()
    }
}
