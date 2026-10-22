# Modified by Infinity Core Contributors: license notices and monorepo build validation.
FROM lukemathwalker/cargo-chef:latest-rust-1.85-bookworm AS chef
WORKDIR /usr/src
COPY rust-toolchain.toml rust-toolchain.toml
RUN rustup show

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

# Donwload, configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY backends backends
COPY core core
COPY config config
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY LICENSE ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

ARG ENV_VAR_BASE=INFINITY

# sccache specific variables
ARG SCCACHE_GHA_ENABLED

# Refresh the base image's trust store before contacting the Intel repository.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential pkg-config libssl-dev protobuf-compiler python3 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel=2024.0.0-49656 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" > fakeintel.c && \
    gcc -shared -fPIC -o libfakeintel.so fakeintel.c

COPY --from=planner /usr/src/recipe.json recipe.json
COPY LICENSE ./

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo chef cook --release --features candle,mkl,static-linking --no-default-features --recipe-path recipe.json && sccache -s

COPY backends backends
COPY core core
COPY config config
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY LICENSE ./
COPY NOTICE LICENSING.md THIRD_PARTY_NOTICES.md ./
COPY LICENSES LICENSES
COPY scripts/collect-license-notices.sh scripts/collect-license-notices.sh
COPY scripts/collect-cargo-notices.py scripts/collect-cargo-notices.py
RUN sh scripts/collect-license-notices.sh /usr/src/distribution-licenses

FROM builder AS http-builder

# Per-commit values: declared here, not in builder, so they don't invalidate the cached dependency layers.
ARG GIT_SHA
ARG DOCKER_LABEL

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo build --release --bin infinity_v3 --features candle,mkl,static-linking,http --no-default-features \
      --message-format=json-render-diagnostics > /tmp/cargo-build.jsonl \
    && python3 scripts/collect-cargo-notices.py /tmp/cargo-build.jsonl /usr/src/distribution-licenses/cargo \
    && sccache -s

FROM debian:bookworm-slim AS base

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy a lot of the Intel shared objects because of the mkl_serv_intel_cpu_true patch...
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_def.so.2 /usr/local/lib/libmkl_vml_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2 /usr/local/lib/libmkl_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so.2 /usr/local/lib/libmkl_vml_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx512.so.2 /usr/local/lib/libmkl_vml_avx512.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2 /usr/local/lib/libmkl_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx512.so.2 /usr/local/lib/libmkl_avx512.so.2
COPY --from=builder /usr/src/libfakeintel.so /usr/local/libfakeintel.so

ENV LD_PRELOAD=/usr/local/libfakeintel.so

FROM base AS http

COPY --from=http-builder /usr/src/target/release/infinity_v3 /usr/local/bin/infinity_v3

COPY --from=http-builder /usr/src/distribution-licenses/ /usr/share/licenses/infinity_v3/

ENTRYPOINT ["infinity_v3"]
CMD ["--json-output"]
