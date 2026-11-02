// Host-side trained readout composition for Qwen3-VL wrappers.
//
// The Native loader selects wrappers from the original `model_type`:
// * `colqwen3`         -> modules.json Dense token projection + row normalize
// * `qwen3_vl_nemotron_embed` -> pre-final-norm token rows + row normalize
// * `HF_ColBERT`       -> main-checkpoint `linear.weight` + row normalize
// Plain `qwen3_vl` embeddings use last-token pooling and no transform.
//
// This module reads only a small trained projection/head. It never rewrites the
// checkpoint or config, and it never invents a projection when one is absent.

use crate::checkpoint::start;
use crate::model::{read_json, Checkpoint};
use crate::qwen3_vl_config::{ReadoutConfig, ReadoutKind};
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::path::{Component, Path};

#[derive(Clone, Debug)]
pub struct LinearReadout {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub input_width: usize,
    pub output_width: usize,
    pub tanh: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct DenseConfig {
    in_features: usize,
    out_features: usize,
    #[serde(default)]
    bias: bool,
    #[serde(default = "identity_activation")]
    activation_function: String,
}

fn identity_activation() -> String {
    "torch.nn.modules.linear.Identity".to_owned()
}

#[derive(Clone, Debug)]
pub struct Qwen3VlReadout {
    pub config: ReadoutConfig,
    pub hidden: usize,
    chain: Vec<LinearReadout>,
    score: Option<LinearReadout>,
    norm_eps: f32,
}

impl Qwen3VlReadout {
    pub fn load(root: &Path, config: &ReadoutConfig, hidden: usize) -> Result<Self, BackendError> {
        if hidden == 0 {
            return Err(start("readout hidden width is zero"));
        }
        let mut chain = Vec::new();
        let mut score = None;
        match config.kind {
            ReadoutKind::Embedding => {
                if config.multi_vector {
                    // HF_ColBERT stores the token projection in the main
                    // checkpoint as `linear.weight` (Native decoder path).
                    let checkpoint = Checkpoint::open(root)?;
                    let (weight, bias, rows) =
                        read_linear(&checkpoint, "linear.weight", "linear.bias")?;
                    if weight.len() != rows * hidden {
                        return Err(start(format!(
                            "ColBERT token projection is {}x{}, expected {}x{}",
                            rows,
                            weight.len() / rows.max(1),
                            rows,
                            hidden
                        )));
                    }
                    chain.push(LinearReadout {
                        weight,
                        bias,
                        input_width: hidden,
                        output_width: rows,
                        tanh: false,
                    });
                }
            }
            ReadoutKind::ColQwen3 => {
                // Native bridge order: `embedding_proj_layer` (after stripping
                // `vlm.`), then `linear.weight`, then a Dense module.
                let checkpoint = Checkpoint::open(root)?;
                if checkpoint.contains("embedding_proj_layer.weight")
                    || checkpoint.contains("vlm.embedding_proj_layer.weight")
                {
                    let (weight, bias, rows) = read_linear_alias(
                        &checkpoint,
                        "embedding_proj_layer.weight",
                        "embedding_proj_layer.bias",
                    )?;
                    if weight.len() != rows * hidden {
                        return Err(start(
                            "ColQwen3 embedding_proj_layer has incompatible geometry",
                        ));
                    }
                    chain.push(LinearReadout {
                        weight,
                        bias,
                        input_width: hidden,
                        output_width: rows,
                        tanh: false,
                    });
                } else if checkpoint.contains("linear.weight")
                    || checkpoint.contains("vlm.linear.weight")
                {
                    let (weight, bias, rows) =
                        read_linear_alias(&checkpoint, "linear.weight", "linear.bias")?;
                    if weight.len() != rows * hidden {
                        return Err(start(
                            "ColQwen3 linear projection has incompatible geometry",
                        ));
                    }
                    chain.push(LinearReadout {
                        weight,
                        bias,
                        input_width: hidden,
                        output_width: rows,
                        tanh: false,
                    });
                } else {
                    chain = load_dense_chain(root, hidden)?;
                }
                let expected = config.token_projection_width.unwrap_or(0);
                let actual = chain.last().map_or(hidden, |module| module.output_width);
                if expected != 0 && actual != expected {
                    return Err(start(format!(
                        "ColQwen3 token projection width {actual} does not match expected {expected}"
                    )));
                }
            }
            ReadoutKind::NemotronColEmbed => {
                // The Native wrapper applies no token projection: it exports
                // the pre-final-norm residual states and normalizes rows.
            }
            ReadoutKind::Rerank => {
                let checkpoint = Checkpoint::open(root)?;
                let (weight, bias, rows) = read_linear(&checkpoint, "score.weight", "score.bias")?;
                if weight.len() != rows * hidden {
                    return Err(start("rerank score head has incompatible geometry"));
                }
                score = Some(LinearReadout {
                    weight,
                    bias,
                    input_width: hidden,
                    output_width: rows,
                    tanh: false,
                });
            }
        }
        if config.skip_final_norm && config.kind != ReadoutKind::NemotronColEmbed {
            return Err(start(
                "skip_final_norm is admitted only for the Nemotron ColEmbed wrapper",
            ));
        }
        Ok(Self {
            config: config.clone(),
            hidden,
            chain,
            score,
            norm_eps: config.norm_eps as f32,
        })
    }

    pub fn output_width(&self) -> usize {
        self.chain
            .last()
            .map_or(self.hidden, |module| module.output_width)
    }

    pub fn requires_token_rows(&self) -> bool {
        self.config.multi_vector
    }

    /// Apply the trained chain to row-major `[tokens, hidden]` rows and then
    /// L2-normalize when the wrapper requires it. Returns `[tokens, width]`.
    pub fn apply_rows(&self, rows: &[f32], tokens: usize) -> Result<Vec<f32>, BackendError> {
        let expected = tokens
            .checked_mul(self.hidden)
            .ok_or_else(|| start("readout token row count overflow"))?;
        if rows.len() != expected {
            return Err(start(format!(
                "readout received {} values, expected {expected}",
                rows.len()
            )));
        }
        if self.chain.is_empty() {
            let mut output = rows.to_vec();
            if self.config.normalize {
                l2_normalize(&mut output, tokens, self.hidden, self.norm_eps)?;
            }
            return Ok(output);
        }
        let mut current = rows.to_vec();
        let mut width = self.hidden;
        for module in &self.chain {
            if module.input_width != width {
                return Err(start("trained readout chain width mismatch"));
            }
            let mut next = vec![0f32; tokens * module.output_width];
            for (row, source) in current.chunks_exact(width).enumerate() {
                for (column, weight) in module.weight.chunks_exact(module.input_width).enumerate() {
                    let mut value = 0f32;
                    for (a, b) in source.iter().zip(weight) {
                        value += a * b;
                    }
                    if let Some(bias) = module.bias.as_ref() {
                        value += bias[column];
                    }
                    if module.tanh {
                        value = value.tanh();
                    }
                    next[row * module.output_width + column] = value;
                }
            }
            current = next;
            width = module.output_width;
        }
        if self.config.normalize {
            l2_normalize(&mut current, tokens, width, self.norm_eps)?;
        }
        Ok(current)
    }

    /// Apply a trained scalar/label score head to pooled rows. This is not
    /// selected by a Native Qwen3-VL model_type; callers must opt in.
    pub fn apply_scores(&self, pooled: &[f32], sequences: usize) -> Result<Vec<f32>, BackendError> {
        let score = self
            .score
            .as_ref()
            .ok_or_else(|| start("checkpoint has no trained rerank score head"))?;
        let expected = sequences
            .checked_mul(self.hidden)
            .ok_or_else(|| start("rerank row count overflow"))?;
        if pooled.len() != expected {
            return Err(start("rerank input row count mismatch"));
        }
        let mut output = vec![0f32; sequences * score.output_width];
        for (row, source) in pooled.chunks_exact(self.hidden).enumerate() {
            for (column, weight) in score.weight.chunks_exact(self.hidden).enumerate() {
                let mut value = 0f32;
                for (a, b) in source.iter().zip(weight) {
                    value += a * b;
                }
                if let Some(bias) = score.bias.as_ref() {
                    value += bias[column];
                }
                output[row * score.output_width + column] = value;
            }
        }
        Ok(output)
    }
}

fn l2_normalize(
    rows: &mut [f32],
    count: usize,
    width: usize,
    eps: f32,
) -> Result<(), BackendError> {
    if width == 0 {
        return Err(start("cannot normalize zero-width rows"));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(start("token normalization epsilon must be positive"));
    }
    for row in rows.chunks_exact_mut(width).take(count) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return Err(start("token row has a non-finite norm"));
        }
        // Native MultiVectorModel: states / clamp_min(||states||, eps).
        let denominator = norm.max(eps);
        for value in row {
            *value /= denominator;
        }
    }
    Ok(())
}

type LinearValues = (Vec<f32>, Option<Vec<f32>>, usize);

fn read_linear(
    checkpoint: &Checkpoint,
    weight: &str,
    bias: &str,
) -> Result<LinearValues, BackendError> {
    read_linear_alias(checkpoint, weight, bias)
}

fn read_linear_alias(
    checkpoint: &Checkpoint,
    weight: &str,
    bias: &str,
) -> Result<LinearValues, BackendError> {
    let weight_name = if checkpoint.contains(weight) {
        weight.to_owned()
    } else {
        format!("vlm.{weight}")
    };
    let bias_name = if checkpoint.contains(bias) {
        Some(bias.to_owned())
    } else if checkpoint.contains(&format!("vlm.{bias}")) {
        Some(format!("vlm.{bias}"))
    } else {
        None
    };
    read_linear_named(checkpoint, &weight_name, bias_name.as_deref())
}

fn read_linear_named(
    checkpoint: &Checkpoint,
    weight: &str,
    bias: Option<&str>,
) -> Result<LinearValues, BackendError> {
    let shape = checkpoint
        .shape(weight)
        .ok_or_else(|| start(format!("missing readout tensor {weight}")))?;
    if shape.len() != 2 {
        return Err(start(format!("readout tensor {weight} must be rank 2")));
    }
    let (rows, columns) = (shape[0], shape[1]);
    let source = checkpoint.source(weight, shape)?;
    let mut weight_values = vec![0f32; rows * columns];
    source.read_f32(0, &mut weight_values)?;
    let bias_values = if let Some(bias) = bias {
        let bias_shape = checkpoint
            .shape(bias)
            .ok_or_else(|| start("missing bias shape"))?;
        if bias_shape != [rows] {
            return Err(start(format!("readout bias {bias} has the wrong shape")));
        }
        let source = checkpoint.source(bias, bias_shape)?;
        let mut values = vec![0f32; rows];
        source.read_f32(0, &mut values)?;
        Some(values)
    } else {
        None
    };
    Ok((weight_values, bias_values, rows))
}

fn load_dense_chain(root: &Path, hidden: usize) -> Result<Vec<LinearReadout>, BackendError> {
    let modules_path = root.join("modules.json");
    if !modules_path.is_file() {
        return Err(start(
            "ColQwen3 requires modules.json with the trained Dense token projection",
        ));
    }
    let modules: serde_json::Value = read_json(&modules_path)?;
    let modules = modules
        .as_array()
        .ok_or_else(|| start("modules.json must contain an array"))?;
    let mut chain = Vec::new();
    let mut width = hidden;
    for module in modules {
        let kind = module
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| start("module is missing its type"))?;
        if kind.rsplit('.').next() != Some("Dense") {
            continue;
        }
        let path = module
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| start("Dense module is missing its path"))?;
        let relative = Path::new(path);
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(start(
                "Dense module path must stay inside the model directory",
            ));
        }
        let module_root = root.join(relative);
        let config: DenseConfig = read_json(&module_root.join("config.json"))?;
        if config.in_features != width {
            return Err(start(format!(
                "Dense module input {} does not match current width {width}",
                config.in_features
            )));
        }
        let checkpoint = Checkpoint::open_module(root, &module_root)?;
        let (weight, bias, rows) = read_linear(&checkpoint, "linear.weight", "linear.bias")?;
        if rows != config.out_features || weight.len() != config.out_features * config.in_features {
            return Err(start("Dense module weight does not match config.json"));
        }
        if config.bias != bias.is_some() {
            return Err(start(
                "Dense module bias presence does not match config.json",
            ));
        }
        let tanh = match config.activation_function.as_str() {
            "torch.nn.modules.linear.Identity" => false,
            "torch.nn.modules.activation.Tanh" => true,
            other => {
                return Err(start(format!(
                    "unsupported Dense activation {other} in the VL token readout"
                )))
            }
        };
        chain.push(LinearReadout {
            weight,
            bias,
            input_width: config.in_features,
            output_width: config.out_features,
            tanh,
        });
        width = config.out_features;
    }
    if chain.is_empty() {
        return Err(start(
            "ColQwen3 has no trained Dense token projection in modules.json",
        ));
    }
    Ok(chain)
}
