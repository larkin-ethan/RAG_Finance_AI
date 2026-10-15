# One image for the API, the UI, and the daily job (each Cloud Run service/job
# just runs a different command). Build remotely with Cloud Build:
#   gcloud builds submit --tag <image> .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf \
    PORT=8080

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first (cached layer). On Linux the lock resolves torch to the
# CPU-only build, so no CUDA libraries end up in the image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY ui ./ui
COPY config ./config
COPY evals ./evals
COPY .streamlit ./.streamlit
RUN uv sync --frozen

# Bake the embedding model into the image: no 1.2 GB download on cold start.
RUN uv run --no-sync python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app /opt/hf
USER app

# Default: the API. The UI and the job override the command at deploy time.
CMD ["sh", "-c", "uv run --no-sync uvicorn finrag.api.main:app --host 0.0.0.0 --port ${PORT}"]
