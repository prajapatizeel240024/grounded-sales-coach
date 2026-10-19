FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIELDCOACH_ROOT=/app

WORKDIR /app

# Install dependencies and the package first so code edits don't bust the dependency layer.
COPY pyproject.toml README.md ./
COPY fieldcoach ./fieldcoach
RUN pip install .

# Synthetic transcripts, golden labels, accepted runs and eval results.
COPY data ./data
COPY evals ./evals

RUN useradd --create-home app && chown -R app /app
USER app

# Cloud Run sets $PORT; default to 8080 elsewhere.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn fieldcoach.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
