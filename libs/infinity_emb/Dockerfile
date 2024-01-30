# Use the Python base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    EXTRAS="all"

RUN apt-get update && apt-get install gcc python3-dev python3.10 curl -y 
# python3.10-venv 
WORKDIR /app

FROM base as builder

# Set the working directory for the app

# Define the version of Poetry to install (default is 1.6.1)
# Define the directory to install Poetry to (default is /opt/poetry)
ARG POETRY_VERSION=1.6.1
ARG POETRY_HOME=/opt/poetry

# Create a Python virtual environment for Poetry and install it
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME POETRY_VERSION=$POETRY_VERSION python3.10 -

ENV PATH=$POETRY_HOME/bin:$PATH

# Test if Poetry is installed in the expected path
RUN echo "Poetry version:" && poetry --version

# Copy the rest of the app source code (this layer will be invalidated and rebuilt whenever the source code changes)
COPY poetry.lock poetry.toml pyproject.toml README.md /app/
# Install dependencies only
RUN poetry install --no-interaction --no-ansi --no-root --extras "${EXTRAS}" --without lint,test
COPY infinity_emb infinity_emb
# Install dependency with infinity_emb package
RUN poetry install --no-interaction --no-ansi --extras "${EXTRAS}"  --without lint,test
# remove cache
RUN poetry cache clear pypi --all

FROM builder as testing
# install lint and test dependencies
RUN poetry install --no-interaction --no-ansi --extras "${EXTRAS}"

# lint 
RUN poetry run ruff .
RUN poetry run black --check .
RUN poetry run mypy .
# pytest
COPY tests tests
# run end to end tests because of duration of build in github ci.
# Run tests/end_to_end on TARGETPLATFORM x86_64 otherwise run tests/end_to_end_gpu
# poetry run python -m pytest tests/end_to_end -x
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] ; then \
poetry run python -m pytest tests/end_to_end -x ; \
else \
poetry run python -m pytest tests/end_to_end/test_api_with_dummymodel.py -x ; \
fi
RUN echo "all tests passed" > "test_results.txt"

# Export with tensorrt, not recommended.
# docker buildx build --target=production-tensorrt -f Dockerfile .
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS production-tensorrt
RUN apt-get update && apt-get install gcc python3-dev python3-pip python3.10 curl -y 
COPY --from=builder /app /app
# force testing stage to run
COPY --from=testing /app/test_results.txt /app/test_results.txt
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch
ENV PATH=/app/.venv/bin:$PATH
RUN pip install --no-cache-dir "onnxruntime-gpu==1.16.3" "tensorrt==8.6.*"
ENV LD_LIBRARY_PATH /app/.venv/lib/python3.10/site-packages/tensorrt:/usr/lib/x86_64-linux-gnu:/app/.venv/lib/python3.10/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}
ENV PATH /app/.venv/lib/python3.10/site-packages/tensorrt/bin:${PATH}
ENTRYPOINT ["infinity_emb"]

# Use a multi-stage build -> production version
FROM base AS production

COPY --from=builder /app /app
# force testing stage to run
COPY --from=testing /app/test_results.txt /app/test_results.txt

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch
ENV PATH=/app/.venv/bin:$PATH

ENTRYPOINT ["infinity_emb"]