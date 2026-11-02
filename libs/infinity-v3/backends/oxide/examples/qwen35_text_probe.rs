// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! First real-backbone control. This is not a Decision or performance claim.
use infinity_backend_oxide::{OxideBackend, Qwen35TextOptions};
use infinity_v3_backend_core::{Backend, Batch, Embedding, Pool};
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE:&str="qwen35_text_probe MODEL FA4_ROOT GDN_RAW GDN_PROVIDER GDN_AUX OUTPUT [TOKENS7] [causal|bidirectional] [pooled|raw]";
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|a| a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let a = |i| {
        args.get(i)
            .map(String::as_str)
            .ok_or_else(|| USAGE.to_owned())
    };
    let path = Path::new(a(0)?);
    let output = PathBuf::from(a(5)?);
    if output.exists() {
        return Err("Fresh output required".into());
    }
    let tokens = args
        .get(6)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(7);
    if tokens == 0 || tokens > 262144 {
        return Err("Invalid actual token count".into());
    }
    let causal =
        match args.get(7).map(String::as_str).unwrap_or("bidirectional") {
            "causal" => true,
            "bidirectional" => false,
            _ => return Err(
                "Mask must be explicit causal or bidirectional; block unsupported by this control"
                    .into(),
            ),
        };
    let raw = match args.get(8).map(String::as_str).unwrap_or("pooled") {
        "raw" => true,
        "pooled" => false,
        _ => return Err("Readout must be pooled or raw".into()),
    };
    let options = Qwen35TextOptions {
        causal,
        fa4_root: PathBuf::from(a(1)?),
        gdn_raw_kernels: PathBuf::from(a(2)?),
        gdn_provider: PathBuf::from(a(3)?),
        gdn_aux_kernels: PathBuf::from(a(4)?),
    };
    let backend = OxideBackend::new_qwen35_text_candidate(path, Pool::Mean, 0, options)?;
    let batch = Batch {
        input_ids: (0..tokens).map(|i| 20 + (i * 13 % 20000) as u32).collect(),
        position_ids: (0..tokens)
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?,
        token_type_ids: vec![0; tokens],
        cumulative_seq_lengths: vec![0, tokens.try_into()?],
        max_length: tokens.try_into()?,
        pooled_indices: if raw { vec![] } else { vec![0] },
        raw_indices: if raw { vec![0] } else { vec![] },
        multimodal: vec![],
        tokens: vec![],
        offsets: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
    };
    let result = backend.embed(batch)?;
    let values = match result.get(&0) {
        Some(Embedding::Pooled(v)) => vec![v.clone()],
        Some(Embedding::All(v)) => v.clone(),
        None => return Err("Missing actual backbone output".into()),
    };
    if values.is_empty()
        || values
            .iter()
            .any(|row| row.is_empty() || row.iter().any(|v| !v.is_finite()))
    {
        return Err("Actual backbone output is empty or nonfinite".into());
    }
    backend.health()?;
    let report = serde_json::json!({"schema":"infinity-qwen35-text-probe/v1","status":"passed-finite-only",
        "model":path,"tokens":tokens,"mask":if causal{"causal"}else{"bidirectional"},"readout":if raw{"raw"}else{"mean"},
        "dtype":"bfloat16","dynamic_fp8":false,"CUDA_graphs":false,"values":values,
        "scope":"Experimental real text backbone; no Native model quality, Decision, prefix/cache/media or performance claim"});
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("{{\"status\":\"passed-finite-only\",\"tokens\":{tokens}}}");
    Ok(())
}
