//! Configuration shared by the router and inference backends.
use std::fmt;

include!(concat!(env!("OUT_DIR"), "/env_base.rs"));

/// Projection quantization requested for a LibTorch model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DynamicQuantization {
    #[default]
    None,
    Fp8,
    Mxfp8,
    Nvfp4,
}

impl DynamicQuantization {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fp8 => "fp8",
            Self::Mxfp8 => "mxfp8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    pub const fn requires_bfloat16(self) -> bool {
        matches!(self, Self::Mxfp8 | Self::Nvfp4)
    }
}

impl fmt::Display for DynamicQuantization {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scope of the requested projection quantization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DynamicQuantizationScope {
    #[default]
    Auto,
    Mlp,
    All,
}

impl DynamicQuantizationScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mlp => "mlp",
            Self::All => "all",
        }
    }
}

impl fmt::Display for DynamicQuantizationScope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

const RETIRED_DYNAMIC_QUANTIZATION_ENV: [(&str, &str, &str, &str); 4] = [
    (
        "ENABLE_FP8_DYNAMIC",
        "DYNAMIC_QUANTIZATION",
        "--dynamic-quantization",
        "none disables quantization; fp8 is rowwise FP8; mxfp8 makes the formerly automatic eligible MLP path explicit",
    ),
    (
        "TORCH_FP8_SCOPE",
        "DYNAMIC_QUANTIZATION_SCOPE",
        "--dynamic-quantization-scope",
        "expected auto, mlp, or all",
    ),
    (
        "TORCH_MXFP8_MLP",
        "DYNAMIC_QUANTIZATION",
        "--dynamic-quantization",
        "none disables quantization; fp8 is rowwise FP8; mxfp8 makes the formerly automatic eligible MLP path explicit",
    ),
    (
        "TORCH_NVFP4_MLP",
        "DYNAMIC_QUANTIZATION",
        "--dynamic-quantization",
        "none disables quantization; nvfp4 is an explicit MLP format",
    ),
];

/// Reject retired dynamic-quantization environment selectors.
///
/// The lookup is injectable so callers and tests do not depend on global
/// process state. Presence alone is rejected, including empty, `0`, or `false`
/// values; values are never printed.
pub fn validate_retired_dynamic_quantization_env(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<(), String> {
    validate_retired_dynamic_quantization_env_with_base(ENV_VAR_BASE, lookup)
}

fn validate_retired_dynamic_quantization_env_with_base(
    base: &str,
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<(), String> {
    for (suffix, replacement, flag, detail) in RETIRED_DYNAMIC_QUANTIZATION_ENV {
        let prefixed = format!("{base}_{suffix}");
        for name in [prefixed.as_str(), suffix] {
            if lookup(name).is_some() {
                let choices = if replacement == "DYNAMIC_QUANTIZATION_SCOPE" {
                    "auto, mlp, all"
                } else {
                    "none, fp8, mxfp8, nvfp4"
                };
                return Err(format!(
                    "Remove {name}; it is retired. Set {base}_{replacement} or {flag} to one of {choices} ({detail})."
                ));
            }
        }
    }
    Ok(())
}

/// Read the prefix-cache budget from TORCH_PREFIX_CACHE_SIZE in GB or GiB.
pub fn prefix_cache_size_bytes() -> Result<i64, String> {
    prefix_cache_size_with(|name| std::env::var_os(name))
}

fn prefix_cache_size_with(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<i64, String> {
    let name = infinity_env!("TORCH_PREFIX_CACHE_SIZE");
    lookup(name).map_or(Ok(0), |value| {
        value.to_str().and_then(parse_byte_size).ok_or_else(|| {
            format!("Invalid {name}: expected GB or GiB, e.g. 3.5GiB, or 0 to disable")
        })
    })
}

/// Parse GB (10^9 bytes), GiB (2^30 bytes), or the off switch 0.
/// Unit values support up to nine decimal places; fractional bytes round down.
fn parse_byte_size(value: &str) -> Option<i64> {
    let value = value.trim();
    if value == "0" {
        return Some(0);
    }
    let (amount, unit) = value
        .strip_suffix("GB")
        .map(|n| (n, 1_000_000_000_i64))
        .or_else(|| value.strip_suffix("GiB").map(|n| (n, 1_i64 << 30)))?;
    let amount = amount.trim();
    let (whole, fraction) = amount.split_once('.').unwrap_or((amount, ""));
    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || amount.ends_with('.')
        || fraction.len() > 9
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let fraction = format!("{fraction:0<9}").parse::<i64>().ok()?;
    whole
        .parse::<i64>()
        .ok()?
        .checked_mul(unit)?
        .checked_add(fraction.checked_mul(unit)? / 1_000_000_000)
}

#[cfg(test)]
mod tests {
    #[test]
    fn retired_dynamic_quantization_environment_is_actionable_even_when_disabled() {
        let retired = [
            (
                "ENABLE_FP8_DYNAMIC",
                "ACME_DYNAMIC_QUANTIZATION",
                "--dynamic-quantization",
                "rowwise",
            ),
            (
                "TORCH_FP8_SCOPE",
                "ACME_DYNAMIC_QUANTIZATION_SCOPE",
                "--dynamic-quantization-scope",
                "auto",
            ),
            (
                "TORCH_MXFP8_MLP",
                "ACME_DYNAMIC_QUANTIZATION",
                "--dynamic-quantization",
                "formerly automatic",
            ),
            (
                "TORCH_NVFP4_MLP",
                "ACME_DYNAMIC_QUANTIZATION",
                "--dynamic-quantization",
                "explicit MLP format",
            ),
        ];
        for (suffix, replacement, flag, detail) in retired {
            for name in [format!("ACME_{suffix}"), suffix.to_owned()] {
                for value in ["0", "false", ""] {
                    let found = name.clone();
                    let error = super::validate_retired_dynamic_quantization_env_with_base(
                        "ACME",
                        move |candidate| (candidate == found).then(|| value.into()),
                    )
                    .unwrap_err();
                    assert!(error.contains(&name), "{error}");
                    assert!(error.contains(replacement), "{error}");
                    assert!(error.contains(flag), "{error}");
                    assert!(error.contains(detail), "{error}");
                }
            }
        }
        assert!(
            super::validate_retired_dynamic_quantization_env_with_base("ACME", |_| None).is_ok()
        );
        let compiled = infinity_env!("ENABLE_FP8_DYNAMIC");
        let error = super::validate_retired_dynamic_quantization_env(|name| {
            (name == compiled).then(|| "0".into())
        })
        .unwrap_err();
        assert!(error.contains(compiled), "{error}");
    }

    #[test]
    fn prefix_cache_size_reads_only_size_and_defaults_to_disabled() {
        for (size, expected) in [
            (None, Some(0)),
            (Some("3.5GiB"), Some(3_758_096_384)),
            (Some("3.5GB"), Some(3_500_000_000)),
            (Some("0"), Some(0)),
            (Some("3500000000"), None),
            (Some("bad"), None),
            (Some(""), None),
        ] {
            let actual = super::prefix_cache_size_with(|name| {
                assert_eq!(name, infinity_env!("TORCH_PREFIX_CACHE_SIZE"));
                size.map(std::ffi::OsString::from)
            });
            assert_eq!(actual.ok(), expected, "size={size:?}");
        }
    }

    #[test]
    fn byte_sizes_require_gb_or_gib_except_for_zero() {
        for (input, expected) in [
            ("0", 0),
            ("0GB", 0),
            ("3.5GB", 3_500_000_000),
            ("80GB", 80_000_000_000),
            ("3.5GiB", 3_758_096_384),
            ("3.1GiB", 3_328_599_654),
            ("0GiB", 0),
            ("0.000000001GiB", 1),
            ("8589934591.999999999GiB", i64::MAX - 1),
            (" 3.5 GB ", 3_500_000_000),
            ("0.000000001GB", 1),
            ("9223372036.854775807GB", i64::MAX),
        ] {
            assert_eq!(super::parse_byte_size(input), Some(expected), "{input}");
        }
        for input in [
            "3500000000",
            "1",
            "9223372036854775807",
            "",
            "GB",
            "-1",
            "-1GB",
            "3.5",
            "1.GB",
            ".5GB",
            "1..5GB",
            "NaNGB",
            "infGB",
            "1e3GB",
            "GiB",
            "-1GiB",
            "1.GiB",
            "8589934592GiB",
            "0.0000000001GiB",
            "3.5MB",
            "10%",
            "3.5gb",
            "0.0000000001GB",
            "9223372036854775808",
            "9223372037GB",
            "9223372036.854775808GB",
        ] {
            assert_eq!(super::parse_byte_size(input), None, "{input}");
        }
    }

    #[test]
    fn names_are_static_and_follow_the_compiled_base() {
        assert_eq!(
            crate::ENV_VAR_BASE,
            option_env!("ENV_VAR_BASE").unwrap_or("INFINITY")
        );
        const NAME: &str = infinity_env!("MODEL_ID");
        assert_eq!(NAME, format!("{}_MODEL_ID", crate::ENV_VAR_BASE));
        assert_eq!(infinity_env!(""), format!("{}_", crate::ENV_VAR_BASE));
    }
}
