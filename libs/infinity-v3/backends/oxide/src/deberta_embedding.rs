// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared DeBERTa embedding projection and post-layer convolution policy.
//!
//! The ordinary H768 word-only relative path keeps the dedicated word kernel.
//! Variants with absolute position/type tables, an embedding projection, or
//! convolution use the source kernels in this module. Every addition follows
//! the Native BF16 boundary: each tensor add rounds once to BF16, a bias-free
//! projection accumulates in F32 and rounds once, and LayerNorm consumes the
//! resulting BF16 rows. The convolution rounds its Conv1d result to BF16 before
//! the configured activation, then the host rounds the residual sum before the
//! conv LayerNorm.

use crate::{
    arg,
    cublas::Blas,
    deberta_encoder_model::{DebertaConfig, DebertaWeights},
    encoder_linear::ProjectionPlan,
    infer, launch, launch_threads,
    lt::LtProvider,
    product, start, u32_size,
    upload::{UploadElement, WeightUploader},
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::{mem::size_of, path::Path, sync::Arc};

struct Kernels {
    compose: Function,
    conv: Option<Function>,
    norm: Function,
}

impl Kernels {
    fn load(context: &Arc<Context>, conv: bool, hidden: usize) -> Result<Self, BackendError> {
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
            compose: module
                .load_function("infinity_deberta_embedding_compose")
                .map_err(start)?,
            conv: if conv {
                Some(
                    module
                        .load_function("infinity_deberta_conv_768")
                        .map_err(start)?,
                )
            } else {
                None
            },
            norm: module
                .load_function(if hidden == 768 {
                    "infinity_deberta_layer_norm_768"
                } else {
                    "infinity_deberta_plain_norm_bf16_aten"
                })
                .map_err(start)?,
        })
    }
}

struct ConvWeights<E> {
    weight: DeviceBuffer<E>,
    bias: DeviceBuffer<E>,
    gamma: DeviceBuffer<E>,
    beta: DeviceBuffer<E>,
}

struct EmbeddingProjection<E: UploadElement> {
    weight: DeviceBuffer<E>,
    plan: ProjectionPlan<E>,
}

pub(crate) struct PreparedEmbedding<E: UploadElement> {
    config: DebertaConfig,
    kernels: Kernels,
    position: Option<DeviceBuffer<E>>,
    token_type: Option<DeviceBuffer<E>>,
    gamma: DeviceBuffer<E>,
    beta: DeviceBuffer<E>,
    projection: Option<EmbeddingProjection<E>>,
    conv: Option<ConvWeights<E>>,
}

impl<E: UploadElement> PreparedEmbedding<E> {
    pub(crate) fn new(
        host: &DebertaWeights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Option<Self>, BackendError> {
        if <E as crate::tensor::Element>::FORMAT != crate::tensor::ScalarFormat::Bf16 {
            return Err(start("DeBERTa is qualified only for BF16 storage"));
        }
        let config = host.config.clone();
        let hidden = config.geometry.hidden_size;
        let has_projection = host.embed_proj.is_some();
        let has_position = host.position.is_some();
        let has_type = host.token_type.is_some();
        let has_conv = host.conv.is_some();
        if hidden == 768 && !has_projection && !has_position && !has_type && !has_conv {
            return Ok(None);
        }
        if config.embedding_size != hidden && !has_projection {
            return Err(start(
                "DeBERTa embedding_size differs from hidden_size without embed_proj.weight",
            ));
        }
        if config.embedding_size == hidden && has_projection {
            return Err(start(
                "DeBERTa embed_proj.weight is only valid when embedding_size differs",
            ));
        }
        let kernels = Kernels::load(context, has_conv, hidden)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let position = host
            .position
            .as_ref()
            .map(|source| uploader.upload_typed::<E>(&[source]))
            .transpose()?;
        let token_type = host
            .token_type
            .as_ref()
            .map(|source| uploader.upload_typed::<E>(&[source]))
            .transpose()?;
        let gamma = uploader.upload_typed::<E>(&[&host.gamma])?;
        let beta = uploader.upload_typed::<E>(&[&host.beta])?;
        let projection = match &host.embed_proj {
            Some(source) => Some(EmbeddingProjection {
                weight: uploader.upload_typed::<E>(&[source])?,
                plan: ProjectionPlan::plain(config.embedding_size, hidden)?,
            }),
            None => None,
        };
        let conv = match &host.conv {
            Some(source) => Some(ConvWeights {
                weight: uploader.upload_typed::<E>(&[&source.weight])?,
                bias: uploader.upload_typed::<E>(&[&source.bias])?,
                gamma: uploader.upload_typed::<E>(&[&source.gamma])?,
                beta: uploader.upload_typed::<E>(&[&source.beta])?,
            }),
            None => None,
        };
        uploader.finish()?;
        Ok(Some(Self {
            config,
            kernels,
            position,
            token_type,
            gamma,
            beta,
            projection,
            conv,
        }))
    }

    pub(crate) fn scratch_width(&self) -> usize {
        if self.conv.is_some() {
            self.config
                .embedding_size
                .max(self.config.geometry.hidden_size)
        } else {
            self.config.embedding_size
        }
    }

    pub(crate) fn has_conv(&self) -> bool {
        self.conv.is_some()
    }

    pub(crate) fn set_rows(
        &mut self,
        provider: &LtProvider,
        rows: usize,
    ) -> Result<(), BackendError> {
        if let Some(projection) = self.projection.as_mut() {
            projection
                .plan
                .set_rows(provider, rows)
                .map_err(|error| start(format!("DeBERTa embedding projection: {error:?}")))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_embedding(
        &self,
        provider: &LtProvider,
        blas: Option<&Blas>,
        stream: &Stream,
        tokens: usize,
        word: &DeviceBuffer<E>,
        ids: u64,
        positions: u64,
        types: u64,
        scratch: &DeviceBuffer<E>,
        output: &mut DeviceBuffer<E>,
        eps: f32,
    ) -> Result<(), BackendError> {
        if tokens == 0 {
            return Err(infer("DeBERTa embedding requires at least one token"));
        }
        // The ordinary encoder owns its single uploaded word matrix through
        // deferred completion. Borrow that allocation rather than uploading a
        // second vocabulary table for optional composition or wider norms.
        if !Arc::ptr_eq(word.context(), stream.context())
            || word.len() < product(self.config.geometry.vocab_size, self.config.embedding_size)?
        {
            return Err(infer("DeBERTa shared word table context/extent mismatch"));
        }
        let active = product(tokens, self.config.embedding_size)?;
        let hidden_elements = product(tokens, self.config.geometry.hidden_size)?;
        if scratch.len() < active || output.len() < hidden_elements {
            return Err(infer("DeBERTa embedding scratch is too small"));
        }
        let mut compose_ids = ids;
        let mut compose_positions = positions;
        let mut compose_types = types;
        let mut compose_word = word.cu_deviceptr();
        let mut compose_position = self.position.as_ref().map_or(0, DeviceBuffer::cu_deviceptr);
        let mut compose_type = self
            .token_type
            .as_ref()
            .map_or(0, DeviceBuffer::cu_deviceptr);
        let mut compose_out = scratch.cu_deviceptr();
        let mut compose_tokens = u32_size(tokens)?;
        let mut width = u32_size(self.config.embedding_size)?;
        let mut position_biased = u32::from(self.config.position_biased_input);
        let mut token_types = u32::from(self.config.geometry.type_vocab_size > 0);
        launch(
            &self.kernels.compose,
            stream,
            (compose_tokens, 1, 1),
            0,
            &mut [
                arg(&mut compose_ids),
                arg(&mut compose_positions),
                arg(&mut compose_types),
                arg(&mut compose_word),
                arg(&mut compose_position),
                arg(&mut compose_type),
                arg(&mut compose_out),
                arg(&mut compose_tokens),
                arg(&mut width),
                arg(&mut position_biased),
                arg(&mut token_types),
            ],
        )?;
        let normalized_input = match &self.projection {
            Some(projection) => {
                let blas = blas.ok_or_else(|| {
                    infer("DeBERTa embedding projection requires the plain cuBLAS handle")
                })?;
                // SAFETY: caller retains the prepared projection, scratch,
                // weight and output on this ordered stream through completion.
                unsafe {
                    projection.plan.enqueue(
                        provider,
                        Some(blas),
                        scratch,
                        &projection.weight,
                        None,
                        output,
                    )?;
                }
                output.cu_deviceptr()
            }
            None => scratch.cu_deviceptr(),
        };
        self.normalize(
            stream,
            tokens,
            normalized_input,
            0,
            &self.gamma,
            &self.beta,
            output.cu_deviceptr(),
            eps,
        )?;
        if self.conv.is_some() {
            let bytes = product(
                product(tokens, self.config.geometry.hidden_size)?,
                size_of::<E>(),
            )?;
            // SAFETY: both allocations belong to this prepared embedding and
            // the copy stays ordered before the saved row is consumed.
            unsafe {
                cuda_core::memory::memcpy_dtod_async(
                    scratch.cu_deviceptr(),
                    output.cu_deviceptr(),
                    bytes,
                    stream.cu_stream(),
                )
            }
            .map_err(infer)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_conv(
        &self,
        stream: &Stream,
        tokens: usize,
        batch: usize,
        offsets: u64,
        scratch: &DeviceBuffer<E>,
        temporary: &mut DeviceBuffer<E>,
        residual: &mut DeviceBuffer<E>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let conv = self
            .conv
            .as_ref()
            .ok_or_else(|| infer("DeBERTa convolution is not prepared"))?;
        let conv_config = self
            .config
            .conv
            .ok_or_else(|| infer("DeBERTa convolution configuration is missing"))?;
        let active = product(tokens, self.config.geometry.hidden_size)?;
        if scratch.len() < active || temporary.len() < active || residual.len() < active {
            return Err(infer("DeBERTa convolution scratch is too small"));
        }
        let mut input = scratch.cu_deviceptr();
        let mut offsets = offsets;
        let mut temp = temporary.cu_deviceptr();
        let mut weight = conv.weight.cu_deviceptr();
        let mut bias = conv.bias.cu_deviceptr();
        let mut conv_tokens = u32_size(tokens)?;
        let mut conv_batch = u32_size(batch)?;
        let mut hidden = u32_size(self.config.geometry.hidden_size)?;
        let mut kernel = u32_size(conv_config.kernel)?;
        let mut groups = u32_size(conv_config.groups)?;
        let mut padding = u32_size((conv_config.kernel - 1) / 2)?;
        let mut activation = conv_config.activation.code();
        launch(
            self.kernels
                .conv
                .as_ref()
                .ok_or_else(|| infer("DeBERTa convolution kernel is not prepared"))?,
            stream,
            (conv_tokens, 1, 1),
            0,
            &mut [
                arg(&mut input),
                arg(&mut offsets),
                arg(&mut temp),
                arg(&mut weight),
                arg(&mut bias),
                arg(&mut conv_tokens),
                arg(&mut conv_batch),
                arg(&mut hidden),
                arg(&mut kernel),
                arg(&mut groups),
                arg(&mut padding),
                arg(&mut activation),
            ],
        )?;
        let residual_pointer = residual.cu_deviceptr();
        self.normalize(
            stream,
            tokens,
            temp,
            residual_pointer,
            &conv.gamma,
            &conv.beta,
            residual_pointer,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn normalize(
        &self,
        stream: &Stream,
        tokens: usize,
        input: u64,
        residual: u64,
        gamma: &DeviceBuffer<E>,
        beta: &DeviceBuffer<E>,
        output: u64,
        eps: f32,
    ) -> Result<(), BackendError> {
        let mut input = input;
        let mut residual = residual;
        let mut bias = 0u64;
        let mut gamma = gamma.cu_deviceptr();
        let mut beta = beta.cu_deviceptr();
        let mut output = output;
        let mut rows = u32_size(tokens)?;
        let mut width = u32_size(self.config.geometry.hidden_size)?;
        let mut eps = eps;
        launch_threads(
            &self.kernels.norm,
            stream,
            (
                if self.config.geometry.hidden_size == 768 {
                    rows.div_ceil(4)
                } else {
                    rows
                },
                1,
                1,
            ),
            128,
            0,
            &mut [
                arg(&mut input),
                arg(&mut residual),
                arg(&mut bias),
                arg(&mut gamma),
                arg(&mut beta),
                arg(&mut output),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut eps),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::deberta_encoder_model::DebertaConvActivation;

    fn bf16_round(value: f32) -> f32 {
        half::bf16::from_f32(value).to_f32()
    }

    fn bf16_bits(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }

    fn activate_reference(value: f32, activation: DebertaConvActivation) -> f32 {
        let inv_sqrt_two = std::f32::consts::FRAC_1_SQRT_2;
        match activation {
            DebertaConvActivation::Gelu => 0.5 * value * (1.0 + libm::erff(value * inv_sqrt_two)),
            DebertaConvActivation::Tanh => libm::tanhf(value),
            DebertaConvActivation::Relu => {
                if value.is_nan() || value >= 0.0 {
                    value
                } else {
                    0.0
                }
            }
            DebertaConvActivation::Silu => value / (1.0 + libm::expf(-value)),
            DebertaConvActivation::GeluTanh => {
                0.5 * value * (1.0 + libm::tanhf(0.797_884_6 * (value + 0.044_715 * value.powi(3))))
            }
        }
    }

    /// The corrected order: round the Conv1d bias sum to BF16, activate that
    /// BF16 value, then round the activation result to BF16.
    fn corrected_conv_activation(sum: f32, bias: f32, activation: DebertaConvActivation) -> u16 {
        let convolved = bf16_round(sum + bias);
        bf16_bits(activate_reference(convolved, activation))
    }

    /// The rejected C1 order: activate the unrounded F32 bias sum and round
    /// only once at the output store.
    fn unrounded_conv_activation(sum: f32, bias: f32, activation: DebertaConvActivation) -> u16 {
        bf16_bits(activate_reference(sum + bias, activation))
    }

    #[test]
    fn conv_bf16_boundary_precedes_each_native_activation() {
        // These tiny F32/BF16 counterexamples are derived only from IEEE
        // round-to-nearest-even and the documented ATen activation formulas.
        // They are not Native outputs or gold arrays. Each pair differs by one
        // BF16 ULP, so the test distinguishes the two rounding orders.
        for (activation, sum, bias, corrected, unrounded) in [
            (
                DebertaConvActivation::Gelu,
                -5.738_281_3,
                0.195_312_5,
                0xb431,
                0x8000,
            ),
            (
                DebertaConvActivation::GeluTanh,
                -5.363_281_3,
                0.195_312_5,
                0xb425,
                0x8000,
            ),
            (
                DebertaConvActivation::Tanh,
                -3.660_156_3,
                0.195_312_5,
                0xbf80,
                0xbf7f,
            ),
            (
                DebertaConvActivation::Silu,
                -31.25,
                -0.195_312_5,
                0xab39,
                0xab43,
            ),
        ] {
            assert_eq!(corrected_conv_activation(sum, bias, activation), corrected);
            assert_eq!(unrounded_conv_activation(sum, bias, activation), unrounded);
            assert_ne!(corrected, unrounded);
        }
    }

    #[test]
    fn compose_rounds_each_native_add_before_the_next_member() {
        let word = -3.906_25f32;
        let position = -0.121_093_75f32;
        let token_type = -0.113_281_25f32;
        let two_adds = bf16_bits(bf16_round(bf16_round(word + position) + token_type));
        let single_add = bf16_bits(bf16_round((word + position) + token_type));
        assert_eq!(two_adds, 0xc085);
        assert_eq!(single_add, 0xc084);
        assert_ne!(two_adds, single_add);
    }
}
