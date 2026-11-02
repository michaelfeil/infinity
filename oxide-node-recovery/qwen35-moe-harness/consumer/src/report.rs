use infinity_v3_backend_core::BackendError;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PRODUCT_SOURCE: &str =
    "/tmp/oxide-all-model-feature-matrix-v1/qwen35-moe-factory-service-blocking-v2";
pub const PRODUCT_FREEZE_SHA256: &str = "UNBOUND";
pub const NORMAL_FACTORY_SOURCE: &str = "/tmp/oxide-all-model-feature-matrix-v1/qwen35-moe-factory-service-blocking-v2/libs/infinity-v3/backends/src/lib.rs";
pub const CANONICAL31_PROVIDER: &str =
    "/tmp/oxide-all-model-feature-matrix-v1/qwen35-moe-v6/aot/canonical31-provider";
pub const ROOT_CHECKPOINT: &str =
    "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b";

#[derive(Debug, Clone, Serialize)]
pub struct Phase {
    pub name: String,
    pub status: String,
    pub detail: Value,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub mode: String,
    pub status: String,
    pub started_unix_seconds: u64,
    pub model: Option<String>,
    pub pool: Option<String>,
    pub product_source: &'static str,
    pub product_freeze_sha256: &'static str,
    pub normal_factory_source: &'static str,
    pub canonical31_provider: &'static str,
    pub root_checkpoint: &'static str,
    pub engine_count: usize,
    pub normal_factory_route: bool,
    pub submission_type: Option<String>,
    pub phases: Vec<Phase>,
    pub quality: Option<Value>,
    pub life: Option<Value>,
    pub metadata: Option<Value>,
    pub errors: Vec<String>,
    pub typed_errors: Vec<Value>,
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

impl Report {
    pub fn new(mode: &str) -> Self {
        Self {
            schema: "oxide-q35moe-unified-runtime-report/v1",
            mode: mode.into(),
            status: "running".into(),
            started_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: None,
            pool: None,
            product_source: PRODUCT_SOURCE,
            product_freeze_sha256: PRODUCT_FREEZE_SHA256,
            normal_factory_source: NORMAL_FACTORY_SOURCE,
            canonical31_provider: CANONICAL31_PROVIDER,
            root_checkpoint: ROOT_CHECKPOINT,
            engine_count: 0,
            normal_factory_route: false,
            submission_type: None,
            phases: Vec::new(),
            quality: None,
            life: None,
            metadata: None,
            errors: Vec::new(),
            typed_errors: Vec::new(),
            path: None,
        }
    }

    pub fn set_path(&mut self, path: PathBuf) {
        self.path = Some(path);
    }

    pub fn checkpoint(&self) {
        if let Some(path) = self.path.as_ref() {
            let _ = write_report(path, self);
        }
    }

    pub fn phase(&mut self, name: impl Into<String>, status: impl Into<String>, detail: Value) {
        self.phases.push(Phase {
            name: name.into(),
            status: status.into(),
            detail,
        });
        self.checkpoint();
    }

    pub fn finish(&mut self, status: &str) {
        self.status = status.into();
    }

    pub fn record_backend_error(&mut self, stage: &str, error: &BackendError) -> String {
        let variant = variant_name(error).to_string();
        self.typed_errors.push(serde_json::json!({
            "stage": stage,
            "variant": variant,
            "message": error.to_string(),
            "runtime_failed_fence_qualification": false,
            "source_cpu_failed_fence_proof": "separate product source proof; not claimed here",
        }));
        self.checkpoint();
        format!("{stage}: {} ({variant})", error)
    }
}

pub fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create report dir: {e}"))?;
    }
    let data = serde_json::to_vec_pretty(report).map_err(|e| format!("serialize report: {e}"))?;
    let temporary = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    fs::write(&temporary, data).map_err(|e| format!("write report temp: {e}"))?;
    fs::rename(&temporary, path).map_err(|e| format!("rename report: {e}"))?;
    Ok(())
}

pub fn variant_name(error: &BackendError) -> &'static str {
    match error {
        BackendError::Overloaded(_) => "Overloaded",
        BackendError::NoBackend => "NoBackend",
        BackendError::Start(_) => "Start",
        BackendError::Inference(_) => "Inference",
        BackendError::Unhealthy => "Unhealthy",
        BackendError::WeightsNotFound(_) => "WeightsNotFound",
    }
}

pub fn variant_matches(error: &BackendError, expected: &str) -> bool {
    variant_name(error) == expected
}

pub fn output_json(canonical: crate::outputs::CanonicalOutputs) -> Value {
    let outputs: Vec<Value> = canonical
        .outputs
        .into_iter()
        .map(|entry| {
            let bits: Vec<u32> = entry.values.iter().map(|value| value.to_bits()).collect();
            serde_json::json!({
                "index": entry.index,
                "kind": entry.kind,
                "shape": entry.shape,
                "values": entry.values,
                "bits_u32": bits,
            })
        })
        .collect();
    serde_json::json!({
        "outputs": outputs,
        "facts": canonical.facts,
    })
}
