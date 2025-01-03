.PHONY: all clean docs_build docs_clean docs_linkcheck api_docs_build api_docs_clean api_docs_linkcheck format lint test tests test_watch template_docker integration_tests docker_tests help extended_tests build-all-docker build-amd build-trt

# Default target executed when no arguments are given to make.
all: help

precommit : | format spell_fix spell_check lint poetry_check cli_v2_docs template_docker openapi test 

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
VERSION ?= DEV

# Run unit tests and generate a coverage report.
coverage:
	poetry run coverage run --source ./infinity_emb -m pytest 
	poetry run coverage report -m
	poetry run coverage xml

test tests:
	poetry run pytest 

openapi:
	poetry run ./../../docs/assets/create_openapi_with_server_hook.sh

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint: PYTHON_FILES=./infinity_emb
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/infinity_emb --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	poetry run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy $(PYTHON_FILES)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)

template_docker:
	poetry run jinja2 Dockerfile.jinja2 Docker.template.yaml --format=yaml -s amd > Dockerfile.amd_auto
	poetry run jinja2 Dockerfile.jinja2 Docker.template.yaml --format=yaml -s cpu > Dockerfile.cpu_auto
	poetry run jinja2 Dockerfile.jinja2 Docker.template.yaml --format=yaml -s nvidia > Dockerfile.nvidia_auto
	poetry run jinja2 Dockerfile.jinja2 Docker.template.yaml --format=yaml -s trt > Dockerfile.trt_onnx_auto

# Add new targets
build-amd:
	docker buildx build --platform linux/amd64 -t michaelf34/infinity:$(VERSION)-amd -f Dockerfile.amd_auto --push .

build-trt:
	docker buildx build --platform linux/amd64 -t michaelf34/infinity:$(VERSION)-trt-onnx -f Dockerfile.trt_onnx_auto --push .

build-cpu:
	docker buildx build --platform linux/amd64 -t michaelf34/infinity:$(VERSION)-cpu -f Dockerfile.cpu_auto --push .

build-nvidia:
	docker buildx build --platform linux/amd64 -t michaelf34/infinity:$(VERSION) -f Dockerfile.nvidia_auto --push .

# Combined target to build both
build-all-docker: build-nvidia build-cpu build-amd build-trt 

poetry_check:
	poetry check

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

benchmark_embed: tests/data/benchmark/benchmark_embed.json
	ab -n 50 -c 50 -l -s 480 \
	-T 'application/json' \
	-p $< \
	http://127.0.0.1:7997/embeddings
	# sudo apt-get install apache2-utils

benchmark_embed_vision: tests/data/benchmark/benchmark_embed_image.json
	ab -n 50 -c 50 -l -s 480 \
	-T 'application/json' \
	-p $< \
	http://127.0.0.1:7997/embeddings
	# sudo apt-get install apache2-utils

# Generate CLI v2 documentation
cli_v2_docs:
	poetry run ./../../docs/assets/create_cli_v2_docs.sh

######################
# HELP
######################

help:
	@echo '===================='
	@echo 'clean                        - run docs_clean and api_docs_clean'
	@echo 'docs_build                   - build the documentation'
	@echo 'docs_clean                   - clean the documentation build artifacts'
	@echo 'docs_linkcheck               - run linkchecker on the documentation'
	@echo 'api_docs_build               - build the API Reference documentation'
	@echo 'api_docs_clean               - clean the API Reference documentation build artifacts'
	@echo 'api_docs_linkcheck           - run linkchecker on the API Reference documentation'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'
	@echo 'poetry_check                 - run poetry check'
	@echo '-- TESTS --'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests (alias for "make test")'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'extended_tests               - run only extended unit tests'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'integration_tests            - run integration tests'
	@echo 'docker_tests                 - run unit tests in docker'
	@echo '-- DOCUMENTATION tasks are from the top-level Makefile --'
