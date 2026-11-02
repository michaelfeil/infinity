// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared relative-position policy sharing the ordinary encoder executor.
use crate::{
    arg, blocks,
    cublas::Blas,
    deberta_encoder_model::{DebertaConfig, DebertaWeights, RelativeLayerWeights},
    fa4_deberta::{ColdScoreMode, DebertaBatch, Fa4Deberta},
    infer, launch, launch_threads,
    lt::LtProvider,
    product, start,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::{mem::ManuallyDrop, path::Path, sync::Arc};

const HEADS: usize = 12;
const DIM: usize = 64;
const HIDDEN: usize = HEADS * DIM;

/// Column-major C[2span,tokens] = R[2span,64] Q[64,tokens], per head.
/// Values can be contiguous Q rows or original K inside packed QKV rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeDot {
    pub(crate) tokens: i32,
    pub(crate) columns: i32,
    pub(crate) values_row: i32,
    pub(crate) projection_stride: i64,
    pub(crate) output_stride: i64,
    pub(crate) values_elements: usize,
    pub(crate) projection_elements: usize,
    pub(crate) output_elements: usize,
}
impl RelativeDot {
    pub(crate) fn new(tokens: usize, span: usize, values_row: usize) -> Result<Self, BackendError> {
        if tokens == 0 || span == 0 || span > 4096 || values_row < HIDDEN {
            return Err(infer("Invalid DeBERTa relative matrix geometry"));
        }
        let columns = product(span, 2)?;
        let values_elements = product(tokens - 1, values_row)?
            .checked_add(HIDDEN)
            .ok_or_else(|| infer("DeBERTa value extent overflow"))?;
        let projection_elements = product(columns, HIDDEN)?;
        let output_stride = product(tokens, columns)?;
        let output_elements = product(HEADS, output_stride)?;
        u32_size(output_elements)?;
        i32::try_from(output_elements).map_err(infer)?;
        Ok(Self {
            tokens: i32::try_from(tokens).map_err(infer)?,
            columns: i32::try_from(columns).map_err(infer)?,
            values_row: i32::try_from(values_row).map_err(infer)?,
            projection_stride: i64::try_from(product(DIM, columns)?).map_err(infer)?,
            output_stride: i64::try_from(output_stride).map_err(infer)?,
            values_elements,
            projection_elements,
            output_elements,
        })
    }
}

struct Kernels {
    unpack: Function,
    transpose: Function,
    divide: Function,
    buckets: Function,
    norm: Function,
}
impl Kernels {
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| start("DeBERTa requires INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = Path::new(&path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path.to_path_buf()
        };
        let module = context
            .load_module_from_image(&std::fs::read(path).map_err(start)?)
            .map_err(start)?;
        Ok(Self {
            unpack: module
                .load_function("infinity_deberta_unpack_scaled")
                .map_err(start)?,
            transpose: module
                .load_function("infinity_deberta_relative_transpose")
                .map_err(start)?,
            divide: module
                .load_function("infinity_deberta_divide")
                .map_err(start)?,
            buckets: module
                .load_function("infinity_deberta_signed_buckets")
                .map_err(start)?,
            norm: module
                .load_function("infinity_deberta_layer_norm_768")
                .map_err(start)?,
        })
    }
}

pub(crate) struct Scratch {
    pub(crate) c2p: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) p2c: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) buckets: ManuallyDrop<DeviceBuffer<i32>>,
    stream: ManuallyDrop<Arc<Stream>>,
}
impl Scratch {
    pub(crate) fn bytes(tokens: usize, span: usize) -> Result<usize, BackendError> {
        let dot = RelativeDot::new(tokens, span, HIDDEN)?;
        let buckets = tokens
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| infer("DeBERTa bucket capacity overflow"))?;
        product(dot.output_elements, 4)?
            .checked_add(product(buckets, 4)?)
            .ok_or_else(|| infer("DeBERTa arena overflow"))
    }
    pub(crate) fn new(
        stream: &Arc<Stream>,
        tokens: usize,
        span: usize,
    ) -> Result<Self, BackendError> {
        let dot = RelativeDot::new(tokens, span, HIDDEN)?;
        let bucket_count = tokens
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| infer("DeBERTa bucket capacity overflow"))?;
        // Startup initialization also covers disabled score terms and padded
        // library reads. Ordinary Work teardown drains these ordered uses.
        let c2p = DeviceBuffer::zeroed(stream, dot.output_elements).map_err(infer)?;
        let p2c = match DeviceBuffer::zeroed(stream, dot.output_elements) {
            Ok(value) => value,
            Err(error) => {
                if stream.synchronize().is_err() {
                    crate::record_uncharged_retained_bytes(c2p.num_bytes());
                    std::mem::forget(c2p);
                }
                return Err(infer(error));
            }
        };
        let buckets = match DeviceBuffer::zeroed(stream, bucket_count) {
            Ok(value) => value,
            Err(error) => {
                if stream.synchronize().is_err() {
                    crate::record_uncharged_retained_bytes(
                        c2p.num_bytes().saturating_add(p2c.num_bytes()),
                    );
                    std::mem::forget(c2p);
                    std::mem::forget(p2c);
                }
                return Err(infer(error));
            }
        };
        Ok(Self {
            c2p: ManuallyDrop::new(c2p),
            p2c: ManuallyDrop::new(p2c),
            buckets: ManuallyDrop::new(buckets),
            stream: ManuallyDrop::new(stream.clone()),
        })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: retain all owners if ordered use cannot be proven complete.
            unsafe {
                ManuallyDrop::drop(&mut self.c2p);
                ManuallyDrop::drop(&mut self.p2c);
                ManuallyDrop::drop(&mut self.buckets);
                ManuallyDrop::drop(&mut self.stream);
            }
        } else {
            crate::record_uncharged_retained_bytes(
                self.c2p
                    .num_bytes()
                    .saturating_add(self.p2c.num_bytes())
                    .saturating_add(self.buckets.num_bytes()),
            );
        }
    }
}

struct RelativeLayer {
    key: Option<DeviceBuffer<Bf16>>,
    query: Option<DeviceBuffer<Bf16>>,
}
fn validate_relative_layers(
    layers: &[RelativeLayerWeights],
    count: usize,
    c2p: bool,
    p2c: bool,
) -> Result<(), BackendError> {
    if layers.len() != count {
        return Err(start("DeBERTa relative layer count mismatch"));
    }
    for (index, layer) in layers.iter().enumerate() {
        if layer.key.is_some() != c2p {
            return Err(start(format!(
                "DeBERTa relative layer {index} key selection mismatch"
            )));
        }
        if layer.query.is_some() != p2c {
            return Err(start(format!(
                "DeBERTa relative layer {index} query selection mismatch"
            )));
        }
    }
    Ok(())
}
pub(crate) struct PreparedDeberta {
    config: DebertaConfig,
    scale: f32,
    score_mode: ColdScoreMode,
    logarithmic: bool,
    kernels: ManuallyDrop<Arc<Kernels>>,
    provider: ManuallyDrop<Fa4Deberta>,
    layers: ManuallyDrop<Vec<RelativeLayer>>,
    staging: ManuallyDrop<Vec<DeviceBuffer<Bf16>>>,
    temporary: ManuallyDrop<Option<DeviceBuffer<Bf16>>>,
    stream: ManuallyDrop<Arc<Stream>>,
}
impl PreparedDeberta {
    pub(crate) fn span(&self) -> usize {
        self.config.span
    }
    pub(crate) fn new(
        host: &DebertaWeights,
        context: Arc<Context>,
        stream: Arc<Stream>,
        root: &Path,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        let config = host.config.clone();
        validate_relative_layers(
            &host.relative_layers,
            host.layers.len(),
            config.c2p,
            config.p2c,
        )?;
        let logarithmic = config.logarithmic;
        let rows = product(config.span, 2)?;
        let elements = product(rows, HIDDEN)?;
        let kernels = Arc::new(Kernels::load(&context)?);
        let divisor = ((1 + usize::from(config.c2p) + usize::from(config.p2c)) * DIM) as f32;
        let scale = half::bf16::from_f32(divisor.sqrt()).to_f32();
        let (provider, score_mode) =
            Fa4Deberta::new(context, stream.clone(), root, config.span, scale)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(host.layers.len()).map_err(start)?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(
                host.layers
                    .len()
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(3))
                    .ok_or_else(|| start("DeBERTa upload count overflow"))?,
            )
            .map_err(start)?;
        let temporary = Some(DeviceBuffer::zeroed(&stream, elements).map_err(start)?);
        let mut result = Self {
            config,
            scale,
            score_mode,
            logarithmic,
            kernels: ManuallyDrop::new(kernels),
            provider: ManuallyDrop::new(provider),
            layers: ManuallyDrop::new(layers),
            staging: ManuallyDrop::new(staging),
            temporary: ManuallyDrop::new(temporary),
            stream: ManuallyDrop::new(stream),
        };
        // WeightUploader copies use this exact ordered stream; its own Drop
        // drains or retains pinned staging on failure. LtPlan does likewise for
        // descriptors. Consumers can follow each upload without a cold fence.
        let mut uploader = WeightUploader::new((*result.stream).clone())?;
        result.push_source(&mut uploader, &host.relative)?;
        if let (Some(gamma), Some(beta)) = (&host.relative_gamma, &host.relative_beta) {
            let gi = result.push_source(&mut uploader, gamma)?;
            let bi = result.push_source(&mut uploader, beta)?;
            let mut input = result.stage(0)?.cu_deviceptr();
            let (mut residual, mut bias) = (0u64, 0u64);
            let mut gamma = result.stage(gi)?.cu_deviceptr();
            let mut beta = result.stage(bi)?.cu_deviceptr();
            let mut output = input;
            let mut nrows = u32_size(rows)?;
            let mut width = HIDDEN as u32;
            let mut eps = host.config.geometry.layer_norm_eps as f32;
            launch_threads(
                &result.kernels.norm,
                &result.stream,
                (nrows.div_ceil(4), 1, 1),
                128,
                0,
                &mut [
                    arg(&mut input),
                    arg(&mut residual),
                    arg(&mut bias),
                    arg(&mut gamma),
                    arg(&mut beta),
                    arg(&mut output),
                    arg(&mut nrows),
                    arg(&mut width),
                    arg(&mut eps),
                ],
            )?;
        }
        let mut plan = lt.prepare_native_small_bias_shape::<Bf16>(HIDDEN, HIDDEN)?;
        lt.set_logical_rows(&mut plan, rows)
            .map_err(|error| start(format!("DeBERTa relative plan: {error:?}")))?;
        for (index, layer) in host.relative_layers.iter().enumerate() {
            result.layers.push(RelativeLayer {
                key: None,
                query: None,
            });
            let destination = result
                .layers
                .last_mut()
                .ok_or_else(|| start("DeBERTa relative destination missing"))?;
            if result.config.c2p {
                destination.key =
                    Some(DeviceBuffer::zeroed(&result.stream, elements).map_err(start)?);
            }
            if result.config.p2c {
                destination.query =
                    Some(DeviceBuffer::zeroed(&result.stream, elements).map_err(start)?);
            }
            for (enabled, source, key) in [
                (result.config.c2p, layer.key.as_ref(), true),
                (result.config.p2c, layer.query.as_ref(), false),
            ] {
                if !enabled {
                    continue;
                }
                let source =
                    source.ok_or_else(|| start("DeBERTa selected relative projection missing"))?;
                let wi = result.push_source(&mut uploader, &source.weight)?;
                let bi = result.push_source(&mut uploader, &source.bias)?;
                let relative = result
                    .staging
                    .first()
                    .ok_or_else(|| start("DeBERTa relative source missing"))?;
                let weight = result
                    .staging
                    .get(wi)
                    .ok_or_else(|| start("DeBERTa relative weight missing"))?;
                let bias = result
                    .staging
                    .get(bi)
                    .ok_or_else(|| start("DeBERTa relative bias missing"))?;
                let temporary = result
                    .temporary
                    .as_mut()
                    .ok_or_else(|| start("DeBERTa relative temporary missing"))?;
                // SAFETY: cold BF16 buffers share the prepared stream and remain
                // held in result through a successful drain or retained failure.
                unsafe {
                    lt.enqueue(&plan, relative, weight, bias, temporary)?;
                }
                let prepared = result
                    .layers
                    .get(index)
                    .ok_or_else(|| start("DeBERTa relative layer missing"))?;
                let output = if key {
                    prepared.key.as_ref()
                } else {
                    prepared.query.as_ref()
                }
                .ok_or_else(|| start("DeBERTa relative destination missing"))?;
                let mut input = temporary.cu_deviceptr();
                let mut output = output.cu_deviceptr();
                let mut count = u32_size(rows)?;
                let mut width = HIDDEN as u32;
                launch(
                    &result.kernels.transpose,
                    &result.stream,
                    blocks(elements)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut output),
                        arg(&mut count),
                        arg(&mut width),
                    ],
                )?;
            }
        }
        uploader.finish()?;
        result.stream.synchronize().map_err(start)?;
        result.staging.clear();
        drop(result.temporary.take());
        Ok(result)
    }
    fn stage(&self, index: usize) -> Result<&DeviceBuffer<Bf16>, BackendError> {
        self.staging
            .get(index)
            .ok_or_else(|| start("DeBERTa staging source missing"))
    }
    fn push_source(
        &mut self,
        uploader: &mut WeightUploader,
        source: &crate::model::WeightSource,
    ) -> Result<usize, BackendError> {
        self.staging.try_reserve(1).map_err(start)?;
        let index = self.staging.len();
        self.staging.push(uploader.upload(&[source])?);
        Ok(index)
    }
    pub(crate) fn buckets(&self, scratch: &Scratch, maximum: usize) -> Result<(), BackendError> {
        let count = maximum
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| infer("DeBERTa bucket count overflow"))?;
        if maximum == 0 || count > scratch.buckets.len() {
            return Err(infer("DeBERTa bucket scratch too small"));
        }
        let mut output = scratch.buckets.cu_deviceptr();
        let mut length = u32_size(maximum)?;
        let mut span = u32_size(self.config.span)?;
        let mut logarithmic = u32::from(self.logarithmic);
        let mid = self.config.span as f64 * 0.5;
        let mut denominator = if self.logarithmic {
            (((self.config.max_relative - 1) as f64) / mid).ln() as f32
        } else {
            1.0
        };
        launch(
            &self.kernels.buckets,
            &self.stream,
            blocks(count)?,
            0,
            &mut [
                arg(&mut output),
                arg(&mut length),
                arg(&mut span),
                arg(&mut logarithmic),
                arg(&mut denominator),
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &self,
        layer: usize,
        blas: &Blas,
        stream: &Stream,
        scratch: &Scratch,
        packed: u64,
        packed_elements: usize,
        query: u64,
        key: u64,
        value: u64,
        output: u64,
        offsets: u64,
        tokens: usize,
        batch: usize,
        maximum: usize,
    ) -> Result<(), BackendError> {
        self.provider.check_stream(stream)?;
        if tokens > self.provider.token_limit() {
            return Err(infer("DeBERTa native table token limit exceeded"));
        }
        if blas.stream().cu_stream() != stream.cu_stream() {
            return Err(infer("DeBERTa BLAS stream mismatch"));
        }
        let active = product(tokens, HIDDEN)?;
        if product(active, 3)? > packed_elements {
            return Err(infer("DeBERTa packed QKV scratch too small"));
        }
        let mut packed_pointer = packed;
        let mut q = query;
        let mut k = key;
        let mut v = value;
        let mut elements = u32_size(active)?;
        let mut width = HIDDEN as u32;
        let mut scale = self.scale;
        launch(
            &self.kernels.unpack,
            stream,
            blocks(active)?,
            0,
            &mut [
                arg(&mut packed_pointer),
                arg(&mut q),
                arg(&mut k),
                arg(&mut v),
                arg(&mut elements),
                arg(&mut width),
                arg(&mut scale),
            ],
        )?;
        let layer = self
            .layers
            .get(layer)
            .ok_or_else(|| infer("Missing DeBERTa relative layer"))?;
        for (projection, values, row_stride, destination) in [
            (layer.key.as_ref(), query, HIDDEN, &scratch.c2p),
            (
                layer.query.as_ref(),
                packed
                    .checked_add(u64::try_from(product(HIDDEN, 2)?).map_err(infer)?)
                    .ok_or_else(|| infer("DeBERTa raw K address overflow"))?,
                3 * HIDDEN,
                &scratch.p2c,
            ),
        ] {
            let Some(projection) = projection else {
                continue;
            };
            let dot = RelativeDot::new(tokens, self.config.span, row_stride)?;
            if projection.len() < dot.projection_elements || destination.len() < dot.output_elements
            {
                return Err(infer("DeBERTa relative score scratch too small"));
            }
            // SAFETY: all views are checked, owned by this model/Work on its
            // retained stream; outputs are separate from QKV/projection inputs.
            unsafe {
                blas.strided_batched_bf16(
                    projection.cu_deviceptr(),
                    values,
                    destination.cu_deviceptr(),
                    dot.columns,
                    dot.tokens,
                    DIM as i32,
                    dot.columns,
                    dot.values_row,
                    dot.columns,
                    dot.projection_stride,
                    DIM as i64,
                    dot.output_stride,
                    HEADS as i32,
                )?;
            }
            if self.score_mode == ColdScoreMode::Normalized {
                let mut pointer = destination.cu_deviceptr();
                let mut count = u32_size(dot.output_elements)?;
                launch(
                    &self.kernels.divide,
                    stream,
                    blocks(dot.output_elements)?,
                    0,
                    &mut [arg(&mut pointer), arg(&mut count), arg(&mut scale)],
                )?;
            }
        }
        let bucket_count = maximum
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| infer("DeBERTa bucket count overflow"))?;
        // SAFETY: native dedicated ABI consumes contiguous BF16 views and these
        // int32 boundaries/buckets; every owner survives ordered completion.
        unsafe {
            self.provider.launch_bound(
                stream,
                &DebertaBatch {
                    q: query,
                    k: key,
                    v: value,
                    out: output,
                    offsets,
                    c2p: scratch.c2p.cu_deviceptr(),
                    p2c: scratch.p2c.cu_deviceptr(),
                    buckets: scratch.buckets.cu_deviceptr(),
                    tokens,
                    batch,
                    max_sequence: maximum,
                    bucket_count,
                },
            )
        }
    }
}
impl Drop for PreparedDeberta {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: all pending cold/request operations have completed. An
            // uncertain drain deliberately retains code, buffers and contexts.
            unsafe {
                ManuallyDrop::drop(&mut self.layers);
                ManuallyDrop::drop(&mut self.staging);
                ManuallyDrop::drop(&mut self.temporary);
                ManuallyDrop::drop(&mut self.provider);
                ManuallyDrop::drop(&mut self.kernels);
                ManuallyDrop::drop(&mut self.stream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_views_preserve_native_head_and_packed_k_coordinates() {
        for tokens in [1, 7, 127, 128, 129, 1024] {
            let q = RelativeDot::new(tokens, 256, 768).unwrap();
            let k = RelativeDot::new(tokens, 256, 2304).unwrap();
            assert_eq!(q.output_elements, 12 * tokens * 512);
            assert_eq!(q.projection_elements, 12 * 64 * 512);
            for head in [0, 1, 11] {
                for token in [0, tokens - 1] {
                    for dim in [0, 63] {
                        assert_eq!(
                            head * 64 + token * q.values_row as usize + dim,
                            token * 768 + head * 64 + dim
                        );
                        assert_eq!(
                            768 + head * 64 + token * k.values_row as usize + dim,
                            token * 2304 + 768 + head * 64 + dim
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn relative_scratch_rejects_geometry_overflow_before_allocation() {
        for (tokens, span, row) in [
            (0, 256, 768),
            (1, 0, 768),
            (1, 4097, 768),
            (1, 256, 767),
            (usize::MAX, 256, 2304),
        ] {
            assert!(RelativeDot::new(tokens, span, row).is_err());
        }
        assert_eq!(
            Scratch::bytes(129, 256).unwrap(),
            2 * 12 * 129 * 512 * 2 + (2 * 129 - 1) * 4
        );
    }
}
