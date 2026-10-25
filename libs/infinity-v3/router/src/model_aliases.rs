use crate::DecisionProtocol;
use anyhow::Result;
use infinity_v3_backend::Pool;
use std::path::Path;

/// Resolve CLI shorthands before downloading model artifacts.
pub(crate) fn resolve(
    model_id: String,
    protocol: Option<DecisionProtocol>,
    pooling: Option<&Pool>,
) -> Result<(String, Option<DecisionProtocol>)> {
    // Local directories take precedence over registered shorthands.
    if Path::new(&model_id).is_dir() {
        return Ok((model_id, protocol));
    }
    let (target, expected) = match model_id.as_str() {
        "pplx-decider-v1.1-27b" => (
            "perplexity-ai/pplx-decider-v1.1-27b",
            DecisionProtocol::Pplx,
        ),
        "pplx-decider-flash-v1.2" | "perplexity-ai/pplx-decider-flash-v1.2" => (
            "perplexity-ai/pplx-decider-flash-v1.2",
            DecisionProtocol::Pplx,
        ),
        "d3" | "vllm-sr/d3" => ("vllm-sr/d3", DecisionProtocol::D3),
        "d3-mini" | "vllm-sr/d3-mini" => ("vllm-sr/d3-mini", DecisionProtocol::D3),
        _ => return Ok((model_id, protocol)),
    };
    anyhow::ensure!(
        pooling.is_none(),
        "Decision model aliases cannot be used with --pooling"
    );
    anyhow::ensure!(
        protocol.as_ref().is_none_or(|p| p == &expected),
        "Model alias requires its native decision protocol {expected:?}"
    );
    Ok((target.into(), Some(expected)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decider_alias_selects_checkpoint_and_protocol() -> Result<()> {
        for protocol in [None, Some(DecisionProtocol::Pplx)] {
            let (id, protocol) = resolve("pplx-decider-v1.1-27b".into(), protocol, None)?;
            assert_eq!(id, "perplexity-ai/pplx-decider-v1.1-27b");
            assert!(matches!(protocol, Some(DecisionProtocol::Pplx)));
        }
        Ok(())
    }

    #[test]
    fn flash_decider_alias_selects_the_joint_checkpoint() -> Result<()> {
        for id in [
            "pplx-decider-flash-v1.2",
            "perplexity-ai/pplx-decider-flash-v1.2",
        ] {
            let (target, protocol) = resolve(id.into(), None, None)?;
            assert_eq!(target, "perplexity-ai/pplx-decider-flash-v1.2");
            assert_eq!(protocol, Some(DecisionProtocol::Pplx));
            assert!(resolve(id.into(), Some(DecisionProtocol::D3), None).is_err());
        }
        Ok(())
    }

    #[test]
    fn decider_alias_rejects_conflicting_options() {
        for protocol in [
            DecisionProtocol::Rune,
            DecisionProtocol::Onejev,
            DecisionProtocol::Clef,
        ] {
            assert!(resolve("pplx-decider-v1.1-27b".into(), Some(protocol), None).is_err());
        }
        assert!(resolve("pplx-decider-v1.1-27b".into(), None, Some(&Pool::Mean)).is_err());
    }

    #[test]
    fn d3_aliases_select_the_native_protocol() -> Result<()> {
        for id in ["d3", "vllm-sr/d3", "d3-mini", "vllm-sr/d3-mini"] {
            let (target, protocol) = resolve(id.into(), None, None)?;
            assert_eq!(
                target,
                if id.ends_with("-mini") {
                    "vllm-sr/d3-mini"
                } else {
                    "vllm-sr/d3"
                }
            );
            assert!(matches!(protocol, Some(DecisionProtocol::D3)));
            assert!(resolve(id.into(), Some(DecisionProtocol::Pplx), None).is_err());
            assert!(resolve(id.into(), None, Some(&Pool::Mean)).is_err());
        }
        Ok(())
    }

    #[test]
    fn ordinary_model_ids_and_local_directories_are_preserved() -> Result<()> {
        for id in ["Qwen/Qwen3-Embedding-0.6B", "/tmp"] {
            let (actual, protocol) = resolve(id.into(), None, Some(&Pool::Mean))?;
            assert_eq!(actual, id);
            assert!(protocol.is_none());
        }
        Ok(())
    }
}
