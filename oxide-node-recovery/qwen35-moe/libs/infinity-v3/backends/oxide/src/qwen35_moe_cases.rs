// Fixed cold token/layout cases shared by the Native CPU control and the Rust
// executor qualification. These are metadata only: no tokenizer or payload is
// read here. The JSON copy lives in the v3 shared directory and is pinned by a
// test when QWEN35_MOE_SHARED_CASES is supplied.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadoutLayout {
    Pooled,
    Raw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ColdCase {
    pub(crate) name: &'static str,
    pub(crate) tokens: usize,
    pub(crate) layout: ReadoutLayout,
}

pub(crate) const SHORT_TOKENS: [usize; 3] = [1, 7, 128];
pub(crate) const LONG_TOKENS: usize = 2048;

pub(crate) const CASES: [ColdCase; 5] = [
    ColdCase {
        name: "short-t1-pooled",
        tokens: 1,
        layout: ReadoutLayout::Pooled,
    },
    ColdCase {
        name: "short-t7-pooled",
        tokens: 7,
        layout: ReadoutLayout::Pooled,
    },
    ColdCase {
        name: "short-t128-pooled",
        tokens: 128,
        layout: ReadoutLayout::Pooled,
    },
    ColdCase {
        name: "long-t2048-pooled",
        tokens: 2048,
        layout: ReadoutLayout::Pooled,
    },
    ColdCase {
        name: "long-t2048-raw",
        tokens: 2048,
        layout: ReadoutLayout::Raw,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::Path;

    #[test]
    fn fixed_cases_match_shared_json_when_supplied() {
        assert_eq!(SHORT_TOKENS, [1, 7, 128]);
        assert_eq!(LONG_TOKENS, 2048);
        let Some(path) = std::env::var_os("QWEN35_MOE_SHARED_CASES") else {
            if std::env::var_os("QWEN35_MOE_REQUIRE_SHARED_CASES").is_some() {
                panic!("QWEN35_MOE_SHARED_CASES is required");
            }
            return;
        };
        let value: Value =
            serde_json::from_slice(&std::fs::read(Path::new(&path)).unwrap()).unwrap();
        let cases = value["cases"].as_array().unwrap();
        assert_eq!(cases.len(), CASES.len());
        for (actual, expected) in CASES.iter().zip(cases) {
            assert_eq!(actual.name, expected["name"].as_str().unwrap());
            assert_eq!(actual.tokens as u64, expected["tokens"].as_u64().unwrap());
            assert_eq!(
                match actual.layout {
                    ReadoutLayout::Pooled => "pooled",
                    ReadoutLayout::Raw => "raw",
                },
                expected["layout"].as_str().unwrap()
            );
        }
    }
}
