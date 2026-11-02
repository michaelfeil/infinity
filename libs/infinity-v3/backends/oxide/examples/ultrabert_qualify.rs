// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

//! CPU-only UltraBERT qualification report and GPU requirement plan.
//!
//! This opens config.json and the safetensors header, validates the exact
//! tensor names/shapes/dtypes, plans per-layer RoPE/window/FA4 needs, and prints
//! the typed preparation requirements. It creates no CUDA context and executes
//! no model. The real executor is `UltraBertExecutor`; `prepare_gpu` refuses at
//! the missing local128/128 FA4 profile before creating CUDA objects.
//!
//!   CUDA_VISIBLE_DEVICES='' taskset -c 120-127 \
//!     cargo run -p infinity-backend-oxide --example ultrabert_qualify --offline \
//!       -- /path/to/ultrabert-embed-150m
//!
//! Optional task override: embedding|sequence_classification|token_classification.

use infinity_backend_oxide::ultrabert_qualification::{UltraBertQualification, UltraBertTask};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let Some(path) = arguments.next() else {
        eprintln!("usage: ultrabert_qualify <checkpoint-dir> [embedding|sequence_classification|token_classification]");
        std::process::exit(2);
    };
    let task = match arguments.next().as_deref() {
        None | Some("embedding") => None,
        Some("sequence_classification") => Some(UltraBertTask::SequenceClassification),
        Some("token_classification") => Some(UltraBertTask::TokenClassification),
        Some(other) => {
            eprintln!("unsupported task {other}");
            std::process::exit(2);
        }
    };
    let qualification = UltraBertQualification::open(Path::new(&path), "bfloat16", task)
        .map_err(|error| format!("{}: {error}", error.code()))?;
    let requirements = qualification.gpu_requirements();
    let readiness = match qualification.require_gpu_ready() {
        Ok(()) => serde_json::json!({"ready": true}),
        Err(error) => serde_json::json!({
            "ready": false,
            "code": error.code(),
            "message": error.to_string(),
        }),
    };
    let output = serde_json::json!({
        "report": qualification.report(),
        "gpu_requirements": requirements,
        "gpu_readiness": readiness,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
