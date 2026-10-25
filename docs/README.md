# Documentation sites

The existing Python / v2 site uses `mkdocs.yml`. The v3 preview uses
`mkdocs-v3.yml` and publishes at <https://michaelfeil.eu/infinity/v3/>.

```bash
pip install -r docs/requirements-v3.txt
mkdocs serve --config-file docs/mkdocs-v3.yml
mkdocs build --strict --config-file docs/mkdocs-v3.yml
```

Edit v3's overview and reference pages in `libs/infinity-v3/`; `gen_v3_docs.py`
includes those files at build time and points source-code links back to GitHub.
Quickstart and API landing pages live in `docs/v3/`. No inference dependencies
or GPU are needed to build the site.

The v3 workflow validates pull requests and publishes `main` with `mike deploy v3`.
Both publishers share a concurrency group to serialize writes to `gh-pages`.
Publishing v3 does not change `latest`, the root redirect, or historical versions.
A later migration can move the default and `latest` alias to v3 while retaining
versioned Python docs and updating links to their archived version.
