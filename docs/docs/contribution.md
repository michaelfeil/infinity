# Contribute and Develop

## Developer setup

Install via Poetry 1.7.1 and Python3.11 on Ubuntu 22.04
```bash
git clone https://github.com/michaelfeil/infinity
cd infinity
cd libs/infinity_emb
poetry install --extras all --with test
```

To ensure your contributions pass the Continuous Integration (CI) checks:
```bash
cd libs/infinity_emb
make format
make lint
poetry run pytest ./tests
```
As an alternative, you can also use the following command:
```bash
cd libs/infinity_emb
make precommit
```

## CLA
All contributions must be made in a way to be compatible with the MIT License of this repo. 