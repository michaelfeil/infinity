// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Typed pointwise encoder activation names.
//!
//! This leaf is deliberately smaller than V1: it models a BF16 pointwise
//! activation over an input that is already the output of the Native linear
//! (including any linear bias), plus an optional *explicit separate* BF16 bias
//! add. It does not model a fused GEMM epilogue and does not infer a family or
//! device policy from the spelling. The caller selects the bias contract at the
//! call site.

use std::fmt;

/// Pointwise activation arithmetic. `GeluTanh` is the tanh recipe used by
/// Native `at::gelu(x, "tanh")`; it is not the erf recipe retained by the
/// existing `GeluExact` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    GeluTanh,
    Relu,
    Silu,
    Tanh,
}

impl Activation {
    /// Parse a spelling for this plain pointwise operator. Every `gelu*`
    /// spelling here means `at::gelu(x, "tanh")` only. It does not represent
    /// family-specific Native mappings: MPNet/DeBERTa `gelu` is erf, and Nomic
    /// normalizes `gelu_pytorch_tanh` before its own dispatch. Those families
    /// must select their recipe explicitly and must not call this helper.
    /// `swiglu` is rejected because Native uses it only after splitting a
    /// packed gate/up row.
    pub fn from_pointwise_spelling(value: &str) -> Result<Self, ActivationError> {
        match value {
            "gelu" | "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" | "gelu_approx" => {
                Ok(Self::GeluTanh)
            }
            "relu" => Ok(Self::Relu),
            "silu" => Ok(Self::Silu),
            "tanh" => Ok(Self::Tanh),
            "swiglu" => Err(ActivationError::GatedNotPointwise("swiglu")),
            other => Err(ActivationError::Unsupported(other.to_owned())),
        }
    }

    /// Spellings covered by the pointwise actual-ATen controls. The first four
    /// are the common BERT/RoBERTa/DistilBERT recipes; `tanh` is the existing
    /// fourth export and must be controlled explicitly.
    pub const fn required_spellings() -> [&'static str; 5] {
        ["gelu_new", "gelu_pytorch_tanh", "relu", "silu", "tanh"]
    }

    /// Native AOT export name for the BF16 pointwise kernel.
    pub const fn export(self) -> &'static str {
        match self {
            Self::GeluTanh => "infinity_encoder_activation_gelu_tanh_bf16",
            Self::Relu => "infinity_encoder_activation_relu_bf16",
            Self::Silu => "infinity_encoder_activation_silu_bf16",
            Self::Tanh => "infinity_encoder_activation_tanh_bf16",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    Unsupported(String),
    GatedNotPointwise(&'static str),
}

impl fmt::Display for ActivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(value) => write!(f, "unsupported encoder activation: {value}"),
            Self::GatedNotPointwise(value) => {
                write!(
                    f,
                    "gated activation is not a plain pointwise operator: {value}"
                )
            }
        }
    }
}

impl std::error::Error for ActivationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pointwise_aliases_map_to_four_recipes() {
        for spelling in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
            "gelu_approx",
        ] {
            assert_eq!(
                Activation::from_pointwise_spelling(spelling),
                Ok(Activation::GeluTanh)
            );
        }
        assert_eq!(
            Activation::from_pointwise_spelling("relu"),
            Ok(Activation::Relu)
        );
        assert_eq!(
            Activation::from_pointwise_spelling("silu"),
            Ok(Activation::Silu)
        );
        assert_eq!(
            Activation::from_pointwise_spelling("tanh"),
            Ok(Activation::Tanh)
        );
        assert_eq!(
            Activation::from_pointwise_spelling("swiglu"),
            Err(ActivationError::GatedNotPointwise("swiglu"))
        );
        assert_eq!(
            Activation::from_pointwise_spelling("gelu_erf"),
            Err(ActivationError::Unsupported("gelu_erf".into()))
        );
    }

    #[test]
    fn gelu_spelling_is_tanh_pointwise_only() {
        assert_eq!(
            Activation::from_pointwise_spelling("gelu"),
            Ok(Activation::GeluTanh)
        );
        assert_eq!(
            Activation::from_pointwise_spelling("gelu_pytorch_tanh"),
            Ok(Activation::GeluTanh)
        );
        // Family-specific erf/normalization is intentionally not represented.
        assert!(Activation::from_pointwise_spelling("gelu_erf").is_err());
    }

    #[test]
    fn required_spellings_cover_every_export() {
        assert_eq!(
            Activation::required_spellings(),
            ["gelu_new", "gelu_pytorch_tanh", "relu", "silu", "tanh"]
        );
        let mut exports = std::collections::BTreeSet::new();
        for spelling in Activation::required_spellings() {
            exports.insert(
                Activation::from_pointwise_spelling(spelling)
                    .unwrap()
                    .export(),
            );
        }
        assert_eq!(exports.len(), 4);
    }

    #[test]
    fn exports_are_unique_and_native_named() {
        let exports = [
            Activation::GeluTanh.export(),
            Activation::Relu.export(),
            Activation::Silu.export(),
            Activation::Tanh.export(),
        ];
        let unique: std::collections::BTreeSet<_> = exports.into_iter().collect();
        assert_eq!(unique.len(), 4);
        assert!(exports.iter().all(|name| name.starts_with("infinity_")));
    }
}
