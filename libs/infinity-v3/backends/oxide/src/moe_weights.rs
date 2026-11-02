// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0

use crate::model::{start, Checkpoint, WeightSource};
use infinity_v3_backend_core::BackendError;

use super::RoutedMlpConfig;

// Keep checkpoint views inline: boxing would add another allocation per expert view.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum GateUpSources {
    Packed(WeightSource),
    Separate {
        gate: WeightSource,
        up: WeightSource,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertSources {
    pub gate_up: GateUpSources,
    pub down: WeightSource,
}

#[derive(Clone, Debug)]
struct SeparateExpert {
    gate: WeightSource,
    up: WeightSource,
    down: WeightSource,
}

// The fused branch retains two source views without allocating an expert table.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum Storage {
    Fused {
        gate_up: WeightSource,
        down: WeightSource,
    },
    Separate(Vec<SeparateExpert>),
}

#[derive(Clone, Debug)]
pub(crate) struct RoutedLayerWeights {
    pub router: WeightSource,
    storage: Storage,
    hidden: usize,
    intermediate: usize,
    experts: usize,
}

fn error(message: &str) -> BackendError {
    start(message)
}

fn fail<T>(message: &str) -> Result<T, BackendError> {
    Err(error(message))
}

fn mul(left: usize, right: usize, message: &str) -> Result<usize, BackendError> {
    left.checked_mul(right).ok_or_else(|| error(message))
}

fn build(router: WeightSource, storage: Storage, config: &RoutedMlpConfig) -> RoutedLayerWeights {
    RoutedLayerWeights {
        router,
        storage,
        hidden: config.hidden_size,
        intermediate: config.moe_intermediate_size,
        experts: config.num_experts,
    }
}

pub(crate) fn load_routed_layer(
    checkpoint: &Checkpoint,
    prefix: &str,
    config: &RoutedMlpConfig,
    index: usize,
) -> Result<Option<RoutedLayerWeights>, BackendError> {
    if !config.is_routed_layer(index)? {
        return Ok(None);
    }
    let stem = format!("{prefix}mlp");
    let router = checkpoint.source(
        &format!("{stem}.gate.weight"),
        &[config.num_experts, config.hidden_size],
    )?;
    let fused_name = format!("{stem}.experts.gate_up_proj");
    if checkpoint.contains(&fused_name) {
        let two_i = mul(
            config.moe_intermediate_size,
            2,
            "fused gate/up dimension overflow",
        )?;
        let gate_up = checkpoint.source(
            &fused_name,
            &[config.num_experts, two_i, config.hidden_size],
        )?;
        let down = checkpoint.source(
            &format!("{stem}.experts.down_proj"),
            &[
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ],
        )?;
        return Ok(Some(build(
            router,
            Storage::Fused { gate_up, down },
            config,
        )));
    }

    let required = config
        .num_experts
        .checked_mul(3)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| error("expert count overflow"))?;
    if checkpoint.names().take(required).count() < required {
        return fail("checkpoint has too few tensors for routed experts");
    }
    let mut experts = Vec::new();
    experts
        .try_reserve_exact(config.num_experts)
        .map_err(|_| error("expert allocation failed"))?;
    for expert in 0..config.num_experts {
        let gate = checkpoint.source(
            &format!("{stem}.experts.{expert}.gate_proj.weight"),
            &[config.moe_intermediate_size, config.hidden_size],
        )?;
        let up = checkpoint.source(
            &format!("{stem}.experts.{expert}.up_proj.weight"),
            &[config.moe_intermediate_size, config.hidden_size],
        )?;
        let down = checkpoint.source(
            &format!("{stem}.experts.{expert}.down_proj.weight"),
            &[config.hidden_size, config.moe_intermediate_size],
        )?;
        experts.push(SeparateExpert { gate, up, down });
    }
    Ok(Some(build(router, Storage::Separate(experts), config)))
}

impl RoutedLayerWeights {
    pub(crate) fn expert(&self, index: usize) -> Result<ExpertSources, BackendError> {
        if index >= self.experts {
            return fail("expert index out of range");
        }
        match &self.storage {
            Storage::Fused { gate_up, down } => {
                let two_i = mul(self.intermediate, 2, "gate/up dimension overflow")?;
                let gate_up_elements = mul(two_i, self.hidden, "gate/up size overflow")?;
                let down_elements = mul(self.hidden, self.intermediate, "down size overflow")?;
                let gate_up_offset = mul(index, gate_up_elements, "gate/up offset overflow")?;
                let down_offset = mul(index, down_elements, "down offset overflow")?;
                Ok(ExpertSources {
                    gate_up: GateUpSources::Packed(
                        gate_up.slice(gate_up_offset, vec![two_i, self.hidden])?,
                    ),
                    down: down.slice(down_offset, vec![self.hidden, self.intermediate])?,
                })
            }
            Storage::Separate(experts) => {
                let source = experts
                    .get(index)
                    .ok_or_else(|| error("expert index out of range"))?;
                let gate_up = GateUpSources::Separate {
                    gate: source.gate.slice(0, source.gate.shape().to_vec())?,
                    up: source.up.slice(0, source.up.shape().to_vec())?,
                };
                Ok(ExpertSources {
                    gate_up,
                    down: source.down.slice(0, source.down.shape().to_vec())?,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    const PREFIX: &str = "model.layers.0.";
    fn temp() -> Temp {
        let path = std::env::temp_dir().join(format!(
            "oxide-moe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Temp(path)
    }
    fn config() -> RoutedMlpConfig {
        RoutedMlpConfig::from_json(&json!({
            "model_type":"qwen3_moe", "intermediate_size":4, "num_experts_per_tok":1,
            "hidden_size": 2, "moe_intermediate_size": 2, "num_experts": 2,
            "num_hidden_layers": 2, "decoder_sparse_step": 1, "mlp_only_layers": [],
        }))
        .unwrap()
    }
    fn save(dir: &Path, tensors: Vec<(String, Vec<usize>, Vec<f32>)>) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                name,
                json!({
                    "dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]
                }),
            );
        }
        let mut raw = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !raw.len().is_multiple_of(8) {
            raw.push(b' ');
        }
        let path = dir.join("model.safetensors");
        let mut file = File::create(&path).unwrap();
        file.write_all(&(raw.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&raw).unwrap();
        file.write_all(&data).unwrap();
        path
    }
    fn values(count: usize, base: f32) -> Vec<f32> {
        (0..count).map(|i| base + i as f32).collect()
    }
    fn read(source: &WeightSource, count: usize) -> Vec<u16> {
        let mut out = vec![0u16; count];
        source.read_bf16(0, &mut out).unwrap();
        out
    }
    fn bf16(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }
    fn ready(path: &Path) -> (Checkpoint, RoutedLayerWeights) {
        let checkpoint = Checkpoint::open(path.parent().unwrap()).unwrap();
        let layer = load_routed_layer(&checkpoint, PREFIX, &config(), 0)
            .unwrap()
            .unwrap();
        (checkpoint, layer)
    }
    fn fused() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        vec![
            (
                format!("{PREFIX}mlp.gate.weight"),
                vec![2, 2],
                values(4, 100.0),
            ),
            (
                format!("{PREFIX}mlp.experts.gate_up_proj"),
                vec![2, 4, 2],
                values(16, 0.0),
            ),
            (
                format!("{PREFIX}mlp.experts.down_proj"),
                vec![2, 2, 2],
                values(8, 50.0),
            ),
        ]
    }
    fn separate() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut t = vec![(
            format!("{PREFIX}mlp.gate.weight"),
            vec![2, 2],
            values(4, 100.0),
        )];
        for e in 0..2 {
            let b = (e * 100) as f32;
            t.push((
                format!("{PREFIX}mlp.experts.{e}.gate_proj.weight"),
                vec![2, 2],
                values(4, b + 10.0),
            ));
            t.push((
                format!("{PREFIX}mlp.experts.{e}.up_proj.weight"),
                vec![2, 2],
                values(4, b + 20.0),
            ));
            t.push((
                format!("{PREFIX}mlp.experts.{e}.down_proj.weight"),
                vec![2, 2],
                values(4, b + 30.0),
            ));
        }
        t
    }
    #[test]
    fn fused_expert_offset_and_retained_view() {
        let dir = temp();
        let path = save(&dir.0, fused());
        let (checkpoint, layer) = ready(&path);
        let expert = layer.expert(1).unwrap();
        let packed = match expert.gate_up {
            GateUpSources::Packed(p) => p,
            _ => panic!(),
        };
        assert_eq!(packed.shape(), &[4, 2]);
        assert_eq!(
            read(&packed, 8),
            (8..16).map(|i| bf16(i as f32)).collect::<Vec<_>>()
        );
        assert_eq!(
            read(&expert.down, 4),
            (0..4).map(|i| bf16(54.0 + i as f32)).collect::<Vec<_>>()
        );
        drop(layer);
        drop(checkpoint);
        fs::remove_file(&path).unwrap();
        assert_eq!(read(&packed, 1)[0], bf16(8.0));
    }

    #[test]
    fn separate_order_and_bounds() {
        let dir = temp();
        let path = save(&dir.0, separate());
        let (checkpoint, layer) = ready(&path);
        let expert = layer.expert(1).unwrap();
        match expert.gate_up {
            GateUpSources::Separate { gate, up } => {
                assert_eq!(read(&gate, 1)[0], bf16(110.0));
                assert_eq!(read(&up, 1)[0], bf16(120.0));
            }
            _ => panic!(),
        }
        assert_eq!(read(&expert.down, 1)[0], bf16(130.0));
        assert!(layer.expert(2).is_err());
        assert!(load_routed_layer(&checkpoint, PREFIX, &config(), 2).is_err());
    }

    #[test]
    fn separate_expert_views_outlive_layer_and_checkpoint() {
        let dir = temp();
        let path = save(&dir.0, separate());
        let (checkpoint, layer) = ready(&path);
        let expert = layer.expert(1).unwrap();
        drop(layer);
        drop(checkpoint);
        fs::remove_file(&path).unwrap();
        if let GateUpSources::Separate { gate, up } = expert.gate_up {
            assert_eq!(read(&gate, 1)[0], bf16(110.0));
            assert_eq!(read(&up, 1)[0], bf16(120.0));
        } else {
            panic!("expected separate views");
        }
        assert_eq!(read(&expert.down, 1)[0], bf16(130.0));
    }

    #[test]
    fn dense_override_skips_missing_router_and_experts() {
        let dir = temp();
        save(&dir.0, vec![]);
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        let mut cfg = config();
        cfg.mlp_only_layers.insert(0);
        assert!(load_routed_layer(&checkpoint, PREFIX, &cfg, 0)
            .unwrap()
            .is_none());
        assert!(load_routed_layer(&checkpoint, PREFIX, &cfg, 1).is_err());
    }

    #[test]
    fn enough_unrelated_names_do_not_hide_a_missing_expert_weight() {
        let dir = temp();
        let mut tensors = separate();
        tensors.retain(|(name, _, _)| !name.ends_with("experts.1.up_proj.weight"));
        tensors.push(("unrelated".into(), vec![1], vec![1.0]));
        save(&dir.0, tensors);
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        assert!(load_routed_layer(&checkpoint, PREFIX, &config(), 0).is_err());
    }

    #[test]
    fn fused_wins_and_malformed_never_falls_back() {
        let dir = temp();
        let mut t = separate();
        t.push((
            format!("{PREFIX}mlp.experts.gate_up_proj"),
            vec![2, 4, 2],
            values(16, 0.0),
        ));
        t.push((
            format!("{PREFIX}mlp.experts.down_proj"),
            vec![2, 2, 2],
            values(8, 50.0),
        ));
        let path = save(&dir.0, t);
        let (checkpoint, layer) = ready(&path);
        assert!(matches!(
            layer.expert(0).unwrap().gate_up,
            GateUpSources::Packed(_)
        ));
        drop(checkpoint);

        for variant in 0..3 {
            let dir = temp();
            let mut t = separate();
            match variant {
                0 => t.push((
                    format!("{PREFIX}mlp.experts.gate_up_proj"),
                    vec![2, 4],
                    values(8, 0.0),
                )),
                1 => t.push((
                    format!("{PREFIX}mlp.experts.gate_up_proj"),
                    vec![2, 4, 2],
                    values(16, 0.0),
                )),
                _ => {
                    t.push((
                        format!("{PREFIX}mlp.experts.gate_up_proj"),
                        vec![2, 4, 2],
                        values(16, 0.0),
                    ));
                    t.push((
                        format!("{PREFIX}mlp.experts.down_proj"),
                        vec![2, 2],
                        values(4, 0.0),
                    ));
                }
            }
            let path = save(&dir.0, t);
            let checkpoint = Checkpoint::open(path.parent().unwrap()).unwrap();
            assert!(load_routed_layer(&checkpoint, PREFIX, &config(), 0).is_err());
        }
    }
}
