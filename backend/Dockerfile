# The API's image (FastAPI). Two stages: `build` installs the API's libraries and the embedding
# model with uv and builds the document index; the final stage copies only the finished
# environment, so uv, its cache and the source tree stay out of the image that runs. Build from
# the repo root:
#
#   docker build -f backend/Dockerfile -t heavden-api .
#   docker run --rm -p 8000:8000 -v "<an exported snapshot>:/app/snapshot:ro" heavden-api
#
# The image holds code only, no data: on Container Apps the API downloads its snapshot from Azure
# Blob Storage when it starts (SNAPSHOT_URL), so a new snapshot needs no new image; locally a
# snapshot folder is mounted at /app/snapshot. Its ML libraries are the ones the model was trained
# with on Databricks (the `serving` group), so the image refuses a snapshot whose model was pickled
# elsewhere, such as a demo snapshot built locally (heavden_api.snapshot checks its check hours).
# The build context is an allowlist (backend/Dockerfile.dockerignore): the image is public.
# Base images are pinned to releases at least 3 weeks old (Python 3.12.14, 25 Sep 2026; uv 0.12.20,
# 28 Sep 2026), by digest so a rebuilt tag can't change them.

ARG PYTHON=python:3.12.14-slim-trixie@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

FROM ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 AS uv

FROM ${PYTHON} AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# The libraries first, a layer that changes only with uv.lock. Only the API's groups: none of
# the test, Spark, notebook or Agent Framework ones; `serving` = Databricks' ML library versions.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --group backend --group rag --group serving --no-editable
# The document search's embedding model, so a cold start never downloads it (readable by the
# API's user: some cache files are written owner-only)
RUN /app/.venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')" \
    && chmod -R a+rX /app/models
# Then our own code, installed as a package (no source tree needed at run time)
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group backend --group rag --group serving --no-editable
# The document index, from the documents in git (shipped in the package): it changes with them
RUN /app/.venv/bin/python -m heavden_api.documents /app/rag --cache-dir /app/models \
    && chmod -R a+rX /app/rag

FROM ${PYTHON}
# The repository: links the GHCR package to it, so its workflows' own token can push new images
LABEL org.opencontainers.image.source="https://github.com/kelvin-meyet/HeavDen-Nexus-Databricks" \
      org.opencontainers.image.description="HeavDen Nexus API: the website's backend (synthetic data)"
# LightGBM needs the OpenMP runtime. The API runs as its own user without a shell; it can't
# change its code, models or data (all owned by root).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin api
COPY --from=build /app /app
# HF_HUB_OFFLINE: the embedding model is in the image; don't ask Hugging Face at start-up
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    SNAPSHOT_DIR=/app/snapshot EMBED_CACHE_DIR=/app/models RAG_DIR=/app/rag
USER api
WORKDIR /app
EXPOSE 8000
# One worker: the chat's rate limits and daily cap are counted in this process
CMD ["uvicorn", "heavden_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
