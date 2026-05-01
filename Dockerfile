FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY LICENSE README.md pyproject.toml ./
COPY scmodal/ ./scmodal/

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 && \
    python3 -m pip install . && \
    python3 -m pip install mygene

ENV PYTHONPATH=/app

CMD ["python3"]
