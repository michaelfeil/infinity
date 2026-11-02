
/// Environment-variable base compiled into this build.
pub const ENV_VAR_BASE: &str = "INFINITY";
/// Construct a static application environment-variable name.
#[macro_export]
macro_rules! infinity_env {
    ($suffix:literal) => { concat!("INFINITY", "_", $suffix) };
}
