# Use the Python base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

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

RUN apt-get update && apt-get install python3.10 python3.10-venv -y 

# Define the version of Poetry to install (default is 1.4.2)
ARG POETRY_VERSION=1.6.1

# Define the directory to install Poetry to (default is /opt/poetry)
ARG POETRY_HOME=/opt/poetry

# Create a Python virtual environment for Poetry and install it
RUN python3.10 -m venv ${POETRY_HOME} && \
    $POETRY_HOME/bin/pip install --upgrade pip && \
    $POETRY_HOME/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Test if Poetry is installed in the expected path
RUN echo "Poetry version:" && $POETRY_HOME/bin/poetry --version

# Set the working directory for the app
WORKDIR /app

# Use a multi-stage build to install dependencies
FROM builder AS dependencies

ARG POETRY_HOME

# Copy only the dependency files for installation
COPY pyproject.toml poetry.lock poetry.toml ./

RUN $POETRY_HOME/bin/poetry config installer.max-workers 10

# Install the Poetry dependencies (this layer will be cached as long as the dependencies don't change)
RUN $POETRY_HOME/bin/poetry install --no-root --no-interaction --no-ansi --extras all

# Use a multi-stage build to run tests
FROM dependencies AS finisher

# Copy the rest of the app source code (this layer will be invalidated and rebuilt whenever the source code changes)
COPY . .

RUN $POETRY_HOME/bin/poetry install --no-interaction --no-ansi --extras all
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch

ENTRYPOINT ["/opt/poetry/bin/poetry","run","infinity_emb"]