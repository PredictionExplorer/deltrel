FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.93.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /build
COPY . .
RUN python3 -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip \
    && mkdir /build/wheels \
    && /build/venv/bin/pip wheel --no-deps ./crates/deltrel-py -w /build/wheels \
    && /build/venv/bin/pip wheel --no-deps . -w /build/wheels

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 deltrelserve
COPY --from=builder /build/wheels /tmp/wheels
RUN python3 -m venv /opt/deltrelserve \
    && /opt/deltrelserve/bin/pip install --upgrade pip \
    && /opt/deltrelserve/bin/pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.34" \
    && /opt/deltrelserve/bin/pip install /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /root/.cache

ENV PATH="/opt/deltrelserve/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DELTRELSERVE_CONFIG=/config/deltrelserve.yaml
USER deltrelserve
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]
CMD ["sh", "-c", "exec deltrelserve --config \"$DELTRELSERVE_CONFIG\""]
