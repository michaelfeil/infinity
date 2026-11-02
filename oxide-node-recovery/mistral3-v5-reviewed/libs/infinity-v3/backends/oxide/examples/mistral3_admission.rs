// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU CLI for the Mistral3/Pixtral admission path.
//!
//! Usage: mistral3_admission <model-dir>
//! Exits 2 on a typed admission error. No CUDA context is created.

use std::{path::Path, process::ExitCode};

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: mistral3_admission <model-dir>");
        return ExitCode::from(64);
    };
    match infinity_backend_oxide::admit_mistral3(Path::new(&path)) {
        Ok(admission) => {
            println!(
                "vision_tensors={} projector_tensors={} text_tensors={} score_tensors={} classifier={} text_layers={} vision_bytes={} projector_bytes={} cold_tensors={} cold_bytes={}",
                admission.vision_tensors,
                admission.projector_tensors,
                admission.text_tensors,
                admission.score_tensors,
                admission.classifier,
                admission.text_layers,
                admission.vision_bytes,
                admission.projector_bytes,
                admission.cold_plan_tensors,
                admission.cold_plan_bytes
            );
            match admission.vision_profile.as_deref() {
                Some(profile) => println!("vision_profile={profile}"),
                None => println!(
                    "vision_pending={}",
                    admission.vision_pending.as_deref().unwrap_or("")
                ),
            }
            match admission.text_pending.as_deref() {
                Some(pending) => println!("text_pending={pending}"),
                None => println!("text_pending=none"),
            }
            match admission.projector_pending.as_deref() {
                Some(pending) => println!("projector_pending={pending}"),
                None => println!("projector_pending=none"),
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
