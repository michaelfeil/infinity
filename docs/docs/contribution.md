# Contribute and Develop

## Developer setup

Install via Poetry 1.8.4 and Python3.11 on Ubuntu 22.04
```bash
git clone https://github.com/michaelfeil/infinity
cd infinity
cd libs/infinity_emb
poetry install --extras all --with test
```

To ensure your contributions pass the Continuous Integration (CI), there are some useful local actions.
The `libs/infinity_emb/Makefile` is a useful entrypoint for this.
```bash
cd libs/infinity_emb
make format
make lint
make template-docker
poetry run pytest ./tests
```

As an alternative, you can also use the following command, which bundles a range of the above.
```bash
cd libs/infinity_emb
make precommit
```

## CI and releases

Python CI runs automatically for pushes and pull requests that change
`libs/infinity_emb/`, `libs/embed_package/`, or their CI workflows and shared
actions. Manual runs are also available. Client generation is scoped to changes
in the Python server, client generator, and their workflow inputs.

Python publishing jobs accept release tags beginning with `0.` or `v0.`.
Other tags do not publish Python packages, containers, or documentation, or
deploy the Modal demo. Update these conditions when changing the package's
major version.

Docker and Modal use the version from the triggering release. Stable releases
update the existing Docker and documentation `latest` aliases. GitHub
prereleases publish versioned images and documentation without changing those
aliases or deploying the Modal demo.

The PyPI description comes from `libs/infinity_emb/README.md`.

## CLA
Infinity is developed as open source project. 
All contributions must be made in a way to be compatible with the MIT License of this repo.
