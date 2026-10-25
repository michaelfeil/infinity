# Documentation

One MkDocs site contains both server generations: **Infinity v3** and
**Python / v2** in the sidebar. Existing Python page paths stay unchanged;
v3 pages live under `v3/` within each documentation version, for example
<https://michaelfeil.eu/infinity/main/v3/>.

```bash
pip install -r docs/requirements.txt
mkdocs serve --config-file docs/mkdocs.yml
mkdocs build --strict --config-file docs/mkdocs.yml
```

Edit v3's overview and reference pages in `libs/infinity-v3/`; `gen_v3_docs.py`
includes them at build time. Quickstart and API landing pages live in
`docs/docs/v3/`. The build needs no inference dependencies or GPU.

Pull requests build a downloadable preview. Main publishes the whole site as
`main`; Python releases retain the existing numbered versions and `latest`
alias behavior. V3 is a section, not a separate version in the version selector.
Changing the default and retiring v2 navigation are later migration steps.
