use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ENV_VAR_BASE");
    let base = env::var("ENV_VAR_BASE").unwrap_or_else(|error| match error {
        env::VarError::NotPresent => "INFINITY".to_owned(),
        _ => panic!("ENV_VAR_BASE must be valid Unicode"),
    });
    assert!(
        !base.is_empty()
            && base
                .bytes()
                .enumerate()
                .all(|(i, c)| c.is_ascii_uppercase() || c == b'_' || (i > 0 && c.is_ascii_digit())),
        "ENV_VAR_BASE must match [A-Z_][A-Z0-9_]*"
    );
    let source = format!(
        r#"
/// Environment-variable base compiled into this build.
pub const ENV_VAR_BASE: &str = {base:?};
/// Construct a static application environment-variable name.
#[macro_export]
macro_rules! infinity_env {{
    ($suffix:literal) => {{ concat!({base:?}, "_", $suffix) }};
}}
"#
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("env_base.rs"),
        source,
    )
    .expect("write generated environment-variable names");
}
