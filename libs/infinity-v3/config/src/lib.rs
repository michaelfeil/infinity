//! Compile-time configuration shared by the router and inference backends.
include!(concat!(env!("OUT_DIR"), "/env_base.rs"));

#[cfg(test)]
mod tests {
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
