# Use the Python base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
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
    POETRY_NO_INTERACTION=1 

RUN apt-get update && apt-get install python3.10 curl -y 
# python3.10-venv 

FROM base as builder

# Set the working directory for the app
WORKDIR /app

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
COPY poetry.lock poetry.toml pyproject.toml README.md .
COPY infinity_emb infinity_emb

# Install dependencies and project
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --extras all
# remove cache
RUN poetry cache clear pypi --all

# Use a multi-stage build -> production version
from base AS production

COPY --from=builder /app /app
WORKDIR /app

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch

CMD . /app/.venv/bin/activate && infinity_emb