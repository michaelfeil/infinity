# Compile-time environment-variable base

Infinity v3 defaults to the `INFINITY` environment-variable base. To build a
binary with a different base, set `ENV_VAR_BASE` when invoking Cargo:

```sh
cd libs/infinity-v3
ENV_VAR_BASE=ACME cargo build --locked --release --no-default-features --features candle,http
ACME_MODEL_ID=BAAI/bge-small-en-v1.5 ./target/release/infinity_v3
```

The resulting binary reads `ACME_MODEL_ID`, `ACME_PORT`,
`ACME_EARLY_DISPATCH_TOKENS`, `ACME_HTTP_INPUT_MAX_BYTES`, and the corresponding
`ACME_TORCH_*` settings. Its CLI help and validation messages use those names.
The compiled constant is `infinity_v3_config::ENV_VAR_BASE`; Rust callers use
`infinity_v3_config::infinity_env!("MODEL_ID")` to produce a static string.
Native LibTorch code uses the same base through its generated header.

The base must match `[A-Z_][A-Z0-9_]*`. An unset build variable selects
`INFINITY`; an empty or invalid value fails the build. Cargo tracks changes to
`ENV_VAR_BASE`, so changing it rebuilds the affected code without `cargo clean`.
Setting it when starting a compiled binary has no effect.

CLI flags keep their names. CLI values take precedence over prefixed variables,
which take precedence over existing unprefixed CLI aliases such as `MODEL_ID`
and `PORT`. A custom-base binary does not also read `INFINITY_*` settings.
Third-party variables such as `HF_TOKEN`, `RUST_LOG`, and CUDA settings retain
their existing names. Model fixture and benchmark-only variables are unchanged.

Automatic input truncation is enabled by default. Set `--auto-truncate=false`
or `INFINITY_AUTO_TRUNCATE=false` to reject oversized inputs on standard inference
routes. Individual requests can also set `truncate: false`. Model-specific
Perplexity and typed-decision adapters retain their own input-length validation.

## Docker builds

All Dockerfiles accept `--build-arg ENV_VAR_BASE=ACME`. From the Infinity repo root:

```sh
docker build -f libs/infinity-v3/Dockerfile-libtorch \
  --build-arg ENV_VAR_BASE=ACME \
  -t infinity-v3:acme libs/infinity-v3
docker run --rm --gpus all -p 8080:80 \
  -e ACME_MODEL_ID=BAAI/bge-small-en-v1.5 infinity-v3:acme
```

The NVIDIA image sets `BACKEND=libtorch` through the existing unprefixed CLI
alias, so the default works with any compiled base. A prefixed variable such as
`ACME_BACKEND` still takes precedence over that default. Published alpha images use the default `INFINITY`
base. This setting changes configuration names, not executable names, image
names, package names, or license notices.

For direct native CMake builds, pass `-DENV_VAR_BASE=ACME` (or set the build
process environment). Cargo supplies this setting automatically for LibTorch.
