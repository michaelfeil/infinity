# syntax=docker/dockerfile:1.7
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/opt/torch-sdk/bin:/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils build-essential ca-certificates cmake curl git libssl-dev pkg-config \
    protobuf-compiler python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none
WORKDIR /usr/src/infinity-v3
COPY rust-toolchain.toml ./
RUN rustup show
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/torch-sdk && \
    /opt/torch-sdk/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
ENV LIBTORCH=/opt/torch-sdk/lib/python3.12/site-packages/torch \
    LIBTORCH_CUDA_ARCHITECTURES="" \
    RUSTFLAGS="-C target-cpu=x86-64"
ARG BUILD_JOBS=4
ARG ENV_VAR_BASE=INFINITY
ARG INFINITY_LOG_PREFIX=infinity
ARG GIT_SHA
ARG DOCKER_LABEL
ENV CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS \
    CARGO_BUILD_JOBS=$BUILD_JOBS
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,id=infinity-v3-libtorch-cpu-target,target=/usr/src/infinity-v3/target \
    cargo build --locked --release -p infinity_v3-router --message-format=json-render-diagnostics \
      > /tmp/cargo-build.jsonl \
    && python3 scripts/package-libtorch-runtime.py \
      /tmp/cargo-build.jsonl /opt/torch-sdk/lib/python3.12/site-packages /opt/infinity \
    && sh scripts/collect-license-notices.sh /opt/infinity/licenses \
    && python3 scripts/collect-cargo-notices.py /tmp/cargo-build.jsonl /opt/infinity/licenses/cargo

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libssl3t64 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/infinity/bin/ /usr/local/bin/
COPY --from=builder /opt/infinity/lib/ /opt/infinity/lib/
COPY --from=builder /opt/infinity/licenses/ /usr/share/licenses/infinity_v3/
COPY --from=builder /opt/infinity/sdk-requirements.txt /usr/share/infinity_v3/sdk-requirements.txt
RUN echo /opt/infinity/lib > /etc/ld.so.conf.d/infinity-v3.conf && ldconfig \
    && infinity_v3 --help > /dev/null
ENV BACKEND=libtorch \
    HUGGINGFACE_HUB_CACHE=/data \
    PORT=80
EXPOSE 80
ENTRYPOINT ["infinity_v3"]
CMD ["--json-output"]
