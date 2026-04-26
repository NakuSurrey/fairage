# multi-stage build keeps the runtime image small.
# stage 1 (builder) installs every dependency. stage 2 (runtime) copies
# only the installed site-packages and the application code. final image
# is roughly half the size of a single-stage build, faster to deploy
# and ship through Docker registries.

# ---------- stage 1: builder ----------
FROM python:3.11-slim AS builder

# system deps Pillow needs to decode JPEG/PNG. apt-get clean keeps the
# layer slim — these binaries are needed at install time only and the
# runtime stage copies the wheels, not the apt state.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# copy requirements first so the dependency layer is cached separately
# from the source code. small code edits skip re-installing pip packages.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ---------- stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# matching system libs for Pillow at runtime — no -dev packages needed,
# only the shared libs the wheels link against
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# non-root user — never serve as root in production. uid/gid pinned so
# bind-mounted volumes have predictable ownership on the Hetzner host.
RUN groupadd -r -g 1000 fairage && \
    useradd -r -u 1000 -g 1000 -m -d /home/fairage fairage

# copy installed packages from the builder stage. --user install above
# put them in /root/.local; move them to the fairage user's home so the
# non-root user can import them.
COPY --from=builder /root/.local /home/fairage/.local
ENV PATH=/home/fairage/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# copy only what the API needs to run — src/ for the code, artifacts/
# is bind-mounted at runtime by docker compose so it does not bloat
# the image with model files
COPY src/ /app/src/

# fix ownership before dropping privileges
RUN chown -R fairage:fairage /app /home/fairage

USER fairage

# port 8003 — same as src/config.py API_PORT, matches the Hetzner Nginx
# reverse-proxy config
EXPOSE 8003

# healthcheck hits /health every 30s. unhealthy after 3 consecutive
# failures; Docker reports the status to the load balancer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health')" || exit 1

# uvicorn is the ASGI server. one worker is fine on a 4-vCPU box because
# onnxruntime parallelises internally; multiple workers would each load
# their own copy of the models and waste memory.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8003", \
     "--workers", "1", \
     "--access-log"]
