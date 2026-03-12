# This is an example Dockerfile that builds a minimal container for running LK Agents
# For more information on the build process, see https://docs.livekit.io/agents/ops/deployment/builds/
# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.12 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
# This must match pyproject.toml and uv.lock, which are pinned to Python 3.12.
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# --- Build stage ---
FROM base AS build

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# g++: C++ compiler needed for building Python packages with C++ extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app

# Copy just the dependency files first, for more efficient layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using UV's lock file
# --locked ensures we use exact versions from uv.lock for reproducible builds
# This creates a virtual environment and installs all dependencies
# Ensure your uv.lock file is checked in for consistency across environments
RUN uv sync --locked

# Copy all remaining pplication files into the container
# This includes source code, configuration files, and dependency specifications
# (Excludes files specified in .dockerignore)
COPY . .

# Pre-download any ML models or files the agent needs
# This ensures the container is ready to run immediately without downloading
# dependencies at runtime, which improves startup time and reliability
RUN uv run "agent.py" download-files

# --- Production stage ---
FROM base

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/app" \
  --shell "/sbin/nologin" \
  --uid "${UID}" \
  appuser

WORKDIR /app

# Copy the prepared app and virtual environment with correct ownership.
COPY --from=build --chown=appuser:appuser /app /app

# Switch to the non-privileged user for all subsequent operations.
USER appuser

# Run the application using UV
# UV will activate the virtual environment and run the agent.
# The "start" command tells the worker to connect to LiveKit and begin waiting for jobs.
CMD ["uv", "run", "agent.py", "start"]
