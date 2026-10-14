use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    Auto,
    // Float16 is not available on accelerate
    #[cfg(any(
        feature = "libtorch",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    #[cfg(any(feature = "candle", feature = "libtorch"))]
    Float32,
    #[cfg(any(feature = "candle", feature = "libtorch"))]
    Bfloat16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DType::Auto => write!(f, "auto"),
            // Float16 is not available on accelerate
            #[cfg(any(
                feature = "libtorch",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),
            #[cfg(any(feature = "candle", feature = "libtorch"))]
            DType::Float32 => write!(f, "float32"),
            #[cfg(any(feature = "candle", feature = "libtorch"))]
            DType::Bfloat16 => write!(f, "bfloat16"),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DType {
    fn default() -> Self {
        #[cfg(any(feature = "accelerate", feature = "mkl"))]
        {
            DType::Float32
        }
        #[cfg(all(feature = "candle", not(any(feature = "accelerate", feature = "mkl"))))]
        {
            DType::Float16
        }
        #[cfg(all(not(feature = "candle"), feature = "libtorch"))]
        {
            DType::Float32
        }
        #[cfg(not(any(feature = "candle", feature = "libtorch")))]
        {
            DType::Auto
        }
    }
}

#[cfg(all(test, feature = "clap", feature = "candle"))]
mod tests {
    use super::*;

    #[test]
    fn candle_accepts_bfloat16_cli_dtype() {
        let dtype = DType::from_str("bfloat16", false).unwrap();
        assert_eq!(dtype, DType::Bfloat16);
        assert_eq!(dtype.to_string(), "bfloat16");
    }
}
