// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Gemma3 BF16 rotary tables for the configured head width.
//!
//! The host derives only the native inverse frequencies. The shared GPU
//! initializer evaluates the phase, cos/sin, and BF16-precision coefficients.
//! Construction is synchronized before the tables are published and never runs
//! per request.
use crate::{arg, launch, product, start, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const THREADS: usize = 256;

fn validated_base(theta: f64, linear_factor: f32) -> Result<f32, BackendError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(start("Gemma3 rotary theta must be finite and positive"));
    }
    let base = theta as f32;
    if !base.is_finite() || base <= 0.0 {
        return Err(start(
            "Gemma3 rotary theta must be representable as a positive f32",
        ));
    }
    if !linear_factor.is_finite() || linear_factor <= 0.0 {
        return Err(start(
            "Gemma3 rotary linear factor must be finite and positive",
        ));
    }
    Ok(base)
}

/// Matches the native expression order exactly:
/// `1.f / powf(float(theta), float(2 * j) / dim) / linear_factor`.
fn inverse_from_base(base: f32, linear_factor: f32, dim: usize) -> Result<Vec<f32>, BackendError> {
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(start(
            "Gemma3 rotary head dimension must be positive and even",
        ));
    }
    let pairs = dim / 2;
    let mut result = Vec::new();
    result.try_reserve_exact(pairs).map_err(start)?;
    for index in 0..pairs {
        let exponent = (index * 2) as f32 / dim as f32;
        let inverse_base = 1.0f32 / base.powf(exponent);
        let frequency = inverse_base / linear_factor;
        if !frequency.is_finite() {
            return Err(start("Gemma3 rotary inverse frequency must be finite"));
        }
        result.push(frequency);
    }
    Ok(result)
}

pub(crate) fn inverse_frequencies(
    theta: f64,
    linear_factor: f32,
    dim: usize,
) -> Result<Vec<f32>, BackendError> {
    inverse_from_base(validated_base(theta, linear_factor)?, linear_factor, dim)
}

struct TablesPlan {
    positions: u32,
    pairs: u32,
    elements: usize,
    total_bytes: usize,
    inverse: Vec<f32>,
}

fn tables_plan(
    theta: f64,
    linear_factor: f32,
    positions: usize,
    dim: usize,
    byte_limit: usize,
) -> Result<TablesPlan, BackendError> {
    let inverse = inverse_frequencies(theta, linear_factor, dim)?;
    if positions == 0 {
        return Err(start("Gemma3 rotary positions must be positive"));
    }
    let positions = u32_size(positions)?;
    let pairs = u32_size(inverse.len())?;
    let elements = product(positions as usize, inverse.len())?;
    let table_bytes = product(elements, core::mem::size_of::<f32>())?;
    let tables_bytes = table_bytes
        .checked_mul(2)
        .ok_or_else(|| start("Gemma3 rotary dual-table byte size overflowed usize"))?;
    let inverse_bytes = product(inverse.len(), core::mem::size_of::<f32>())?;
    let total_bytes = tables_bytes
        .checked_add(inverse_bytes)
        .ok_or_else(|| start("Gemma3 rotary byte budget overflowed usize"))?;
    if total_bytes > byte_limit {
        return Err(start(
            "Gemma3 rotary tables exceed the configured byte budget",
        ));
    }
    Ok(TablesPlan {
        positions,
        pairs,
        elements,
        total_bytes,
        inverse,
    })
}

/// Pure cold planning for the model-wide dual-table budget, before CUDA.
pub(crate) fn required_bytes(
    theta: f64,
    linear_factor: f32,
    positions: usize,
    dim: usize,
) -> Result<usize, BackendError> {
    Ok(tables_plan(theta, linear_factor, positions, dim, usize::MAX)?.total_bytes)
}

fn launch_blocks(elements: usize) -> Result<u32, BackendError> {
    let rounded = elements
        .checked_add(THREADS - 1)
        .ok_or_else(|| start("Gemma3 rotary launch grid overflowed usize"))?;
    let blocks = rounded / THREADS;
    if blocks > i32::MAX as usize {
        return Err(start("Gemma3 rotary launch grid exceeds CUDA x dimension"));
    }
    u32::try_from(blocks).map_err(|_| start("Gemma3 rotary launch grid exceeds CUDA grid limits"))
}

pub(crate) struct Tables {
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) stride: u32,
    _inverse: DeviceBuffer<f32>,
    _initializer: Arc<Function>,
}

struct PendingTables {
    stream: Arc<Stream>,
    tables: Option<Tables>,
}

impl Drop for PendingTables {
    fn drop(&mut self) {
        if self.tables.is_some() && self.stream.synchronize().is_err() {
            if let Some(tables) = self.tables.take() {
                std::mem::forget(tables);
            }
        }
    }
}

impl Tables {
    pub(crate) fn new(
        theta: f64,
        linear_factor: f32,
        positions: usize,
        dim: usize,
        byte_limit: usize,
        stream: &Arc<Stream>,
        initializer: Arc<Function>,
    ) -> Result<Self, BackendError> {
        let plan = tables_plan(theta, linear_factor, positions, dim, byte_limit)?;
        debug_assert!(plan.total_bytes <= byte_limit);
        let grid = launch_blocks(plan.elements)?;
        let cosine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
        let sine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
        let inverse = DeviceBuffer::from_host(stream, &plan.inverse).map_err(start)?;
        let mut pending = PendingTables {
            stream: stream.clone(),
            tables: Some(Self {
                cosine,
                sine,
                stride: plan.pairs,
                _inverse: inverse,
                _initializer: initializer,
            }),
        };
        let table = pending
            .tables
            .as_ref()
            .ok_or_else(|| start("Missing Gemma3 table constructor owner"))?;
        let mut inv = table._inverse.cu_deviceptr();
        let mut cos = table.cosine.cu_deviceptr();
        let mut sin = table.sine.cu_deviceptr();
        let mut positions = plan.positions;
        let mut pairs = plan.pairs;
        launch(
            &table._initializer,
            stream,
            (grid, 1, 1),
            0,
            &mut [
                arg(&mut inv),
                arg(&mut cos),
                arg(&mut sin),
                arg(&mut positions),
                arg(&mut pairs),
            ],
        )
        .map_err(start)?;
        stream.synchronize().map_err(start)?;
        pending
            .tables
            .take()
            .ok_or_else(|| start("Missing initialized Gemma3 rotary tables"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_width_128_derives_64_native_pairs() {
        let inverse = inverse_frequencies(1_000_000.0, 1.0, 128).unwrap();
        assert_eq!(inverse.len(), 64);
        assert_eq!(inverse[0], 1.0);
        assert!(inverse.iter().all(|value| value.is_finite()));
        let plan = tables_plan(1_000_000.0, 8.0, 131_072, 128, usize::MAX).unwrap();
        assert_eq!(plan.pairs, 64);
        assert_eq!(plan.elements, 131_072 * 64);
    }

    #[test]
    fn native_frequency_theta_one_factor_two() {
        let inverse = inverse_frequencies(1.0, 2.0, 256).unwrap();
        assert_eq!(inverse.len(), 128);
        assert!(inverse.iter().all(|&value| value == 0.5));
    }

    #[test]
    fn native_exact_power_of_two_frequencies_preserve_division_order() {
        let inverse = inverse_frequencies(16.0, 2.0, 256).unwrap();
        assert_eq!(inverse[0].to_bits(), 0.5f32.to_bits());
        assert_eq!(inverse[32].to_bits(), 0.25f32.to_bits());
        assert_eq!(inverse[64].to_bits(), 0.125f32.to_bits());
        assert_eq!(inverse[96].to_bits(), 0.0625f32.to_bits());
    }

    #[test]
    fn launch_blocks_round_up_and_reject_invalid_cuda_grid() {
        assert_eq!(launch_blocks(128).unwrap(), 1);
        assert_eq!(launch_blocks(256).unwrap(), 1);
        assert_eq!(launch_blocks(257).unwrap(), 2);
        assert_eq!(launch_blocks(32768 * 128).unwrap(), 16384);
        assert_eq!(
            launch_blocks(i32::MAX as usize * THREADS).unwrap(),
            i32::MAX as u32
        );
        assert!(launch_blocks(i32::MAX as usize * THREADS + 1).is_err());
        assert!(launch_blocks(usize::MAX).is_err());
    }

    #[test]
    fn theta_at_or_below_one_is_native() {
        assert!(inverse_frequencies(1.0, 1.0, 256).is_ok());
        let inverse = inverse_frequencies(0.5, 1.0, 256).unwrap();
        assert_eq!(inverse[0], 1.0);
        assert!(inverse.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn invalid_scalars_are_rejected() {
        for theta in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -1.0,
            f64::MAX,
            1e-300,
        ] {
            assert!(
                inverse_frequencies(theta, 1.0, 256).is_err(),
                "theta {theta}"
            );
        }
        for factor in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            assert!(
                inverse_frequencies(1.0, factor, 256).is_err(),
                "factor {factor}"
            );
        }
    }

    #[test]
    fn nonfinite_frequency_is_rejected() {
        assert!(inverse_frequencies(1e-44, 1.0, 256).is_err());
    }

    #[test]
    fn byte_budget_is_inclusive() {
        let limit = tables_plan(1.0, 1.0, 1, 256, usize::MAX)
            .unwrap()
            .total_bytes;
        assert!(tables_plan(1.0, 1.0, 1, 256, limit).is_ok());
        assert!(tables_plan(1.0, 1.0, 1, 256, limit - 1).is_err());
    }

    #[test]
    fn positions_must_fit_u32() {
        assert!(tables_plan(1.0, 1.0, 0, 256, usize::MAX).is_err());
        if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(tables_plan(1.0, 1.0, too_many, 256, usize::MAX).is_err());
        }
        #[cfg(target_pointer_width = "64")]
        assert!(tables_plan(1.0, 1.0, u32::MAX as usize, 256, usize::MAX).is_ok());
    }

    #[test]
    fn launch_grid_rounds_up() {
        assert_eq!(launch_blocks(128).unwrap(), 1);
        assert_eq!(launch_blocks(THREADS).unwrap(), 1);
        assert_eq!(launch_blocks(THREADS + 1).unwrap(), 2);
        assert_eq!(launch_blocks(32768 * 128).unwrap(), 16_384);
    }

    #[test]
    fn actual_32768_dual_table_budget() {
        let expected =
            32768usize * 128 * 2 * core::mem::size_of::<f32>() + 128 * core::mem::size_of::<f32>();
        assert_eq!(expected, 33_554_944);
        let plan = tables_plan(1.0, 1.0, 32768, 256, expected).unwrap();
        assert_eq!(plan.total_bytes, expected);
        assert_eq!(plan.elements, 32768 * 128);
        assert!(tables_plan(1.0, 1.0, 32768, 256, expected - 1).is_err());
    }
}

#[cfg(test)]
mod actual_reference_tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs, path::PathBuf};

    struct Case {
        theta: f64,
        factor: f32,
        positions: usize,
        buffers: BTreeMap<String, Vec<u32>>,
    }

    fn read_cases(variable: &str, schema: &str, count: usize) -> Vec<Case> {
        let root = PathBuf::from(std::env::var_os(variable).unwrap());
        let manifest: Value =
            serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema"], schema);
        assert_eq!(manifest["status"], "passed");
        assert_eq!(
            manifest["native_sdk_sha256"],
            "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
        );
        let cases = manifest["cases"].as_array().unwrap();
        assert_eq!(cases.len(), count);
        cases
            .iter()
            .map(|case| {
                assert_eq!(case["head_dimension"], 256);
                let mut buffers = BTreeMap::new();
                for (name, info) in case["buffers"].as_object().unwrap() {
                    assert_eq!(info["dtype"], "F32");
                    let path = root
                        .join(case["name"].as_str().unwrap())
                        .join(info["file"].as_str().unwrap());
                    let bytes = fs::read(path).unwrap();
                    assert_eq!(bytes.len(), info["bytes"].as_u64().unwrap() as usize);
                    assert_eq!(
                        format!("{:x}", Sha256::digest(&bytes)),
                        info["sha256"].as_str().unwrap()
                    );
                    let elements = info["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .try_fold(1usize, |n, value| {
                            n.checked_mul(value.as_u64().unwrap() as usize)
                        })
                        .unwrap();
                    assert_eq!(elements.checked_mul(4).unwrap(), bytes.len());
                    buffers.insert(
                        name.clone(),
                        bytes
                            .chunks_exact(4)
                            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                            .collect(),
                    );
                }
                Case {
                    theta: case["theta"].as_f64().unwrap(),
                    factor: case["linear_factor"].as_f64().unwrap() as f32,
                    positions: case["positions"].as_u64().unwrap() as usize,
                    buffers,
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "Actual native CPU-only C++ inverse reference; set INFINITY_OXIDE_GEMMA3_ROPE_INVERSE"]
    fn actual_native_cpp_inverse_all384_bits_match() {
        let cases = read_cases(
            "INFINITY_OXIDE_GEMMA3_ROPE_INVERSE",
            "gemma3-native-rope-inverse/v1",
            3,
        );
        for case in cases {
            assert_eq!(case.positions, 0);
            let actual = inverse_frequencies(case.theta, case.factor, 256)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), 128);
            assert_eq!(
                &actual, &case.buffers["inverse"],
                "theta {} factor {}",
                case.theta, case.factor
            );
        }
    }

    #[test]
    #[ignore = "Root GPU: actual native Gemma3 long-position ATen table gold and additive77 module; plain/mem/init"]
    fn actual_native_gpu_tables_and_constructor_all_bits_match() {
        let cases = read_cases(
            "INFINITY_OXIDE_GEMMA3_ROPE_GOLD",
            "gemma3-native-rope-tables/v1",
            12,
        );
        // All actual reference bytes and geometry are checked before CUDA.
        for case in &cases {
            assert!([1, 7, 4097, 32768].contains(&case.positions));
            assert_eq!(case.buffers["inverse"].len(), 128);
            assert_eq!(case.buffers["cosine"].len(), case.positions * 128);
            assert_eq!(case.buffers["sine"].len(), case.positions * 128);
            assert_eq!(
                inverse_frequencies(case.theta, case.factor, 256)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                case.buffers["inverse"]
            );
        }
        if std::env::var("INFINITY_OXIDE_GEMMA3_ROPE_PREPARE_ONLY").as_deref() == Ok("1") {
            println!("GEMMA3_ROPE_ACTUAL_CPU: 12 long-position table cases validated before CUDA");
            return;
        }
        let context = cuda_core::Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let path = PathBuf::from(std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE").unwrap());
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().unwrap();
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let module = context
            .load_module_from_image(&fs::read(path).unwrap())
            .unwrap();
        let initializer = Arc::new(
            module
                .load_function("infinity_gemma2_rope_tables_bf16")
                .unwrap(),
        );
        for case in cases {
            assert!(Tables::new(
                case.theta,
                case.factor,
                case.positions,
                256,
                0,
                &stream,
                initializer.clone()
            )
            .is_err());
            let tables = Tables::new(
                case.theta,
                case.factor,
                case.positions,
                256,
                128 * 1024 * 1024,
                &stream,
                initializer.clone(),
            )
            .unwrap();
            assert_eq!(tables.stride, 128);
            for (label, actual) in [
                ("cosine", tables.cosine.to_host_vec(&stream).unwrap()),
                ("sine", tables.sine.to_host_vec(&stream).unwrap()),
            ] {
                let actual = actual.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                assert_eq!(
                    actual, case.buffers[label],
                    "{label} theta {} factor {} positions {}",
                    case.theta, case.factor, case.positions
                );
            }
        }
    }
}
