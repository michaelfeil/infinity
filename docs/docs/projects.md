# Projects and releases

Infinity hosts two projects with independent build and release lifecycles.

| | Python Infinity | Infinity v3 |
| --- | --- | --- |
| Status | Available | Planned Rust source import |
| Source | `libs/infinity_emb/` | `libs/infinity_v3/` |
| Build system | Poetry | Cargo workspace rooted in the project directory |
| Command | `infinity_emb v2` | `infinity_v3` (planned) |
| Container repository | `michaelf34/infinity` | `michaelf34/infinity-v3` (planned) |
| Release tags | `0.x.y` (also accepts `v0.x.y`) | `v3/<version>` |

The `v2` in the Python command names its CLI generation. Infinity v3 names the
Rust project; its package version can advance independently, starting with a
release such as `v3/0.1.0-alpha.1`.

## Python releases

Keep the existing Python release tag convention. Python publishing and release
CI jobs accept tags beginning with `0.` or `v0.`. Other tag namespaces, including
`v3/`, do not trigger Python package/image publishing, Modal deployment, or a
Python documentation release. A future Python major version will require an
explicit update to these release conditions.

The Docker and Modal workflows use the tag from the release event, rather than
discovering the newest tag in the repository. Stable Python releases update the
existing Docker `latest` aliases and documentation `latest` alias. GitHub
prereleases publish versioned images and documentation without updating those
aliases or deploying the Modal demo.

The Python package's PyPI description comes from `libs/infinity_emb/README.md`.
The root README introduces the repository's projects and retains the existing
Python usage guide.

## Infinity v3 releases

Reserve `v3/<version>` for v3 Git tags and releases. For example, a future
`v3/0.1.0-alpha.1` release will publish the image
`michaelf34/infinity-v3:0.1.0-alpha.1`. The `v3/` prefix is a Git namespace and
must be removed when constructing a container tag.

Dedicated Rust CI and release workflows will be added with the source import.
They must target `libs/infinity_v3`, use the release's exact tag, and publish only
v3 artifacts. Prereleases should use versioned tags or a v3 preview alias; the
v3 `latest` alias should be reserved for stable releases. Creating a v3 release
before those workflows exist does not publish any artifacts.

The Rust workspace, dependencies, tests, Dockerfiles, and documentation belong
to the v3 project. Its workflows must live in the root `.github/workflows/`.

## Shared repository assets

The root README links to both projects. The existing documentation site covers
Python Infinity and these repository conventions; v3 documentation will live
with the Rust project until its own publishing workflow is introduced.

Python CI continues to run for pull requests, pushes, scheduled checks, and
manual dispatches. Rust checks will be added separately. Client generation is
scoped to the Python server, client generator, and their workflow inputs; v3 API
compatibility will be assessed when the Rust source is imported.
