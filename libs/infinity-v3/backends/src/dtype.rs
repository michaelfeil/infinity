use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    Auto,
    Float16,
    Float32,
    Bfloat16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DType::Auto => write!(f, "auto"),
            DType::Float16 => write!(f, "float16"),
            DType::Float32 => write!(f, "float32"),
            DType::Bfloat16 => write!(f, "bfloat16"),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DType {
    fn default() -> Self {
        DType::Float32
    }
}

#[cfg(all(test, feature = "clap", feature = "libtorch"))]
mod tests {
    use super::*;

    #[test]
    fn libtorch_accepts_bfloat16_cli_dtype() {
        let dtype = DType::from_str("bfloat16", false).unwrap();
        assert_eq!(dtype, DType::Bfloat16);
        assert_eq!(dtype.to_string(), "bfloat16");
    }
}
