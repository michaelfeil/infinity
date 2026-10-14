mod sequence_classifier;
pub(crate) use sequence_classifier::SequenceClassifier;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle::{Result, Tensor};
use infinity_v3_backend_core::Batch;

mod bert;
mod clef;
mod dense;
mod distilbert;
#[cfg_attr(not(feature = "flash-attn"), allow(dead_code, unused_imports))]
mod embedding_gemma2;
mod gemma3;
#[cfg_attr(not(feature = "flash-attn"), allow(dead_code, unused_imports))]
mod gemma4;
mod gte;
mod jina;
mod laya;
mod llama;
mod mistral;
mod modernbert;
mod mpnet;
mod nomic;
mod ultrabert;
pub use ultrabert::{UltraBertConfig, UltraBertModel};
mod qwen2;
mod qwen3;
mod qwen3_moe;

mod flash_bert;

mod flash_distilbert;

mod flash_gte;

mod flash_jina;

mod flash_jina_code;

mod flash_mistral;

mod flash_modernbert;

mod flash_nomic;

mod flash_qwen2;

mod flash_qwen3;

pub use bert::{BertConfig, PositionEmbeddingType};
pub use dense::{Dense, DenseConfig, DenseLayer};
pub use distilbert::DistilBertConfig;
pub use embedding_gemma2::EmbeddingGemma2Config;
#[cfg(feature = "flash-attn")]
pub use embedding_gemma2::EmbeddingGemma2Model;
pub use gemma3::Gemma3Config;
#[cfg(feature = "flash-attn")]
pub use gemma3::Gemma3Model;
pub use gemma4::Gemma4Config;
#[cfg(feature = "flash-attn")]
pub use gemma4::Gemma4Model;
pub use gte::GTEConfig;

pub use laya::{LayaConfig, LayaModel, LayaOutput};
pub use llama::LLamaConfig;
pub use mistral::MistralConfig;
pub use modernbert::ModernBertConfig;
pub use mpnet::MPNetConfig;
pub use nomic::NomicConfig;
pub use qwen2::Qwen2Config;
pub use qwen3::Qwen3Config;

pub use flash_bert::FlashBertModel;

pub use flash_distilbert::FlashDistilBertModel;

pub use flash_gte::FlashGTEModel;

pub use flash_jina::FlashJinaBertModel;

pub use flash_jina_code::FlashJinaCodeBertModel;

pub use flash_mistral::FlashMistralModel;

pub use flash_modernbert::FlashModernBertModel;

pub use flash_nomic::FlashNomicBertModel;

pub use flash_qwen2::FlashQwen2Model;

pub use flash_qwen3::FlashQwen3Model;

pub(crate) trait Model {
    fn decide(
        &self,
        _batch: Batch,
        _inputs: Vec<infinity_v3_backend_core::DecisionInput>,
    ) -> Result<Vec<infinity_v3_backend_core::DecisionOutput>> {
        candle::bail!("Model does not support typed decisions")
    }

    fn supports_radix_mlp(&self) -> bool {
        false
    }

    fn embed(&self, _batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        candle::bail!("`embed` is not implemented for this model");
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict` is not implemented for this model");
    }

    fn predict_tokens(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict_tokens` is not implemented for this model");
    }
}

#[cfg(feature = "experimental-deberta")]
mod deberta;
#[cfg(feature = "experimental-deberta")]
pub use deberta::{DebertaConfig, DebertaModel};

mod qwen35_config;
pub use qwen35_config::Qwen35Config;
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
mod qwen35;
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
mod qwen35_vision;
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
pub use qwen35::Qwen35Model;

#[cfg(all(feature = "cuda", feature = "flash-attn"))]
mod qwen3_vl;
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
pub use qwen3_vl::Qwen3VlModel;

#[cfg(feature = "flash-attn")]
mod gemma4_vision;
