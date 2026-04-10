FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml .
RUN apt-get update && apt-get install -y g++ && \
    uv sync --no-dev --no-install-project

COPY scmodal/ ./scmodal/

ENV PATH=/app/.venv/bin:$PATH
ENV PYTHONPATH=/app

CMD ["python"]
