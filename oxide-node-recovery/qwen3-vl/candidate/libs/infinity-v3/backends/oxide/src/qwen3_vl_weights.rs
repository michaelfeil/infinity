// Bounded weight-loading plan for the Native safetensors CAPI.
//
// The plan mirrors `TeiWeight` (name/rank/shape/dtype/source offset/length) and
// the per-shard `tei_load_safetensors(fd, data_start, data_length, ...)` call.
// Destination offsets are aligned so a later `WeightUploader` can copy straight
// into its staging slots. No payload is read here and no device work is created.

use crate::checkpoint::{start, CheckpointDtype, CheckpointMetadata};
use crate::qwen3_vl_inventory::{BoundTensor, InventoryReport};
use infinity_v3_backend_core::BackendError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DESTINATION_ALIGNMENT: u64 = 256;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct NativeWeight {
    pub name: String,
    pub original_name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub rank: usize,
    pub source_offset: u64,
    pub source_length: u64,
    pub destination_offset: u64,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ShardPlan {
    pub shard: String,
    pub file_bytes: u64,
    pub data_start: u64,
    pub data_length: u64,
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub destination_bytes: u64,
    pub weights: Vec<NativeWeight>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct WeightPlan {
    pub scope: String,
    pub tensor_count: usize,
    pub total_source_bytes: u64,
    pub total_destination_bytes: u64,
    pub shards: Vec<ShardPlan>,
}

fn align_up(value: u64, alignment: u64) -> Result<u64, BackendError> {
    if alignment == 0 {
        return Err(start("destination alignment must be positive"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| start("destination alignment overflow"))
}

fn checked_add(left: u64, right: u64, name: &str) -> Result<u64, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn dtype_name(dtype: CheckpointDtype) -> &'static str {
    match dtype {
        CheckpointDtype::F32 => "F32",
        CheckpointDtype::F16 => "F16",
        CheckpointDtype::Bf16 => "BF16",
    }
}

pub fn plan(
    report: &InventoryReport,
    metadata: &CheckpointMetadata,
) -> Result<WeightPlan, BackendError> {
    let mut by_shard: BTreeMap<String, Vec<&BoundTensor>> = BTreeMap::new();
    for tensor in &report.bound {
        by_shard
            .entry(tensor.actual.shard.clone())
            .or_default()
            .push(tensor);
    }
    let mut shards = Vec::with_capacity(by_shard.len());
    let mut total_source_bytes = 0u64;
    let mut total_destination_bytes = 0u64;
    let mut tensor_count = 0usize;
    for (shard, mut tensors) in by_shard {
        let shard_meta = metadata
            .shards
            .get(&shard)
            .ok_or_else(|| start(format!("missing shard metadata for {shard}")))?;
        // Native `tei_load_safetensors` requires strictly increasing source
        // offsets per shard (`weight.offset >= previous_end`).
        tensors.sort_unstable_by_key(|tensor| tensor.actual.data_offsets.0);
        let mut destination = 0u64;
        let mut source_bytes = 0u64;
        let mut weights = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            let source_length = tensor
                .actual
                .data_offsets
                .1
                .checked_sub(tensor.actual.data_offsets.0)
                .ok_or_else(|| start(format!("invalid source range for {}", tensor.actual.name)))?;
            if tensor.actual.data_offsets.1 > shard_meta.data_length {
                return Err(start(format!(
                    "{} exceeds shard {} data length {}",
                    tensor.actual.name, shard, shard_meta.data_length
                )));
            }
            destination = align_up(destination, DESTINATION_ALIGNMENT)?;
            let destination_offset = destination;
            destination = checked_add(destination, source_length, "destination size")?;
            source_bytes = checked_add(source_bytes, source_length, "source bytes")?;
            weights.push(NativeWeight {
                name: tensor.expected.name.clone(),
                original_name: tensor.actual.name.clone(),
                shard: shard.clone(),
                dtype: dtype_name(tensor.actual.dtype).to_owned(),
                shape: tensor.actual.shape.clone(),
                rank: tensor.actual.shape.len(),
                source_offset: tensor.actual.data_offsets.0,
                source_length,
                destination_offset,
                kind: tensor.expected.kind.label().to_owned(),
            });
            tensor_count += 1;
        }
        destination = align_up(destination, DESTINATION_ALIGNMENT)?;
        total_source_bytes = checked_add(total_source_bytes, source_bytes, "total source bytes")?;
        total_destination_bytes = checked_add(
            total_destination_bytes,
            destination,
            "total destination bytes",
        )?;
        shards.push(ShardPlan {
            shard,
            file_bytes: shard_meta.file_bytes,
            data_start: shard_meta.data_start,
            data_length: shard_meta.data_length,
            tensor_count: weights.len(),
            source_bytes,
            destination_bytes: destination,
            weights,
        });
    }
    if tensor_count != report.bound.len() {
        return Err(start("weight plan dropped a bound tensor"));
    }
    Ok(WeightPlan {
        scope: match report.scope {
            crate::qwen3_vl_inventory::InventoryScope::TowerOnly => "tower_only".to_owned(),
            crate::qwen3_vl_inventory::InventoryScope::Full => "full".to_owned(),
        },
        tensor_count,
        total_source_bytes,
        total_destination_bytes,
        shards,
    })
}

impl WeightPlan {
    pub fn verify(&self, report: &InventoryReport) -> Result<(), BackendError> {
        if self.tensor_count != report.bound.len() {
            return Err(start("weight plan tensor count does not match inventory"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for shard in &self.shards {
            let mut end = 0u64;
            let mut previous_source_end = 0u64;
            for weight in &shard.weights {
                if !seen.insert(weight.name.clone()) {
                    return Err(start(format!("duplicate planned weight {}", weight.name)));
                }
                if weight.rank != weight.shape.len() {
                    return Err(start(format!("rank mismatch for {}", weight.name)));
                }
                let elements = weight.shape.iter().try_fold(1u64, |count, dim| {
                    count
                        .checked_mul(u64::try_from(*dim).map_err(start)?)
                        .ok_or_else(|| start("weight elements overflow"))
                })?;
                let width = match weight.dtype.as_str() {
                    "F32" => 4,
                    "F16" | "BF16" => 2,
                    other => return Err(start(format!("unsupported planned dtype {other}"))),
                };
                let expected = elements
                    .checked_mul(width)
                    .ok_or_else(|| start("weight bytes overflow"))?;
                if expected != weight.source_length {
                    return Err(start(format!(
                        "planned weight {} declares {} bytes but shape requires {expected}",
                        weight.name, weight.source_length
                    )));
                }
                let source_end = weight
                    .source_offset
                    .checked_add(weight.source_length)
                    .ok_or_else(|| start("weight source end overflow"))?;
                if source_end > shard.data_length {
                    return Err(start(format!(
                        "planned weight {} exceeds shard data",
                        weight.name
                    )));
                }
                if weight.source_offset < previous_source_end {
                    return Err(start(format!(
                        "planned weight {} is not in strictly increasing source order",
                        weight.name
                    )));
                }
                previous_source_end = source_end;
                if weight.destination_offset % DESTINATION_ALIGNMENT != 0 {
                    return Err(start(format!(
                        "planned weight {} is not destination aligned",
                        weight.name
                    )));
                }
                if weight.destination_offset < end {
                    return Err(start(format!(
                        "planned weight {} overlaps a previous destination",
                        weight.name
                    )));
                }
                end = weight
                    .destination_offset
                    .checked_add(weight.source_length)
                    .ok_or_else(|| start("planned destination end overflow"))?;
                if end > shard.destination_bytes {
                    return Err(start(format!(
                        "planned weight {} exceeds destination slab",
                        weight.name
                    )));
                }
            }
        }
        Ok(())
    }
}
