# Infinity v3 — Rust

Infinity v3 is a planned Rust inference server based on the
[TEI fork](https://github.com/michaelfeil/text-embeddings-inference). It is a
separate project within the Infinity repository.

This directory reserves the project location. The Rust source, Cargo workspace,
Dockerfiles, and build/release workflows will arrive with the source import.
There is no installable v3 package or container image from this repository yet.

## Project boundaries

- Source and Cargo workspace: `libs/infinity_v3/`.
- Executable: `infinity_v3`.
- Container repository: `michaelf34/infinity-v3`.
- Git release tags: `v3/<version>`, for example `v3/0.1.0-alpha.1`.
- Package version: independent of the Python package and the v3 project name.
- Documentation: this README and project-specific documentation in this directory.
- CI and releases: dedicated workflows in the repository's root `.github/workflows/`.

The Cargo workspace will be rooted here, with the existing `router/`, `core/`,
and `backends/` structure from the TEI fork. Dockerfiles will live alongside
`Cargo.toml`, using `libs/infinity_v3` as their build context. Each release will
build from its own tag and publish to the v3 container repository.

## Source import

The import should preserve the fork's history. Include its toolchain
configuration, lockfile, tests, and Docker build
assets. Adapt GitHub workflows to the new paths in the root `.github/workflows/`;
GitHub will not discover workflows nested inside this directory.

Document supported models, API compatibility, and migration instructions as the
implementation is integrated. Compatibility with the Python API and generated
client must be verified before it is advertised.

The Python project continues to live in `libs/infinity_emb` and uses the
`infinity_emb v2` command. See [project boundaries and releases](../../docs/docs/projects.md)
for the repository-wide conventions.
