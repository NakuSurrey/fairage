# Streamlit demo container.
#
# kept separate from the API Dockerfile because:
#   - Streamlit pulls ~250 MB of JS + theming. baking it into the API
#     image would bloat the production inference container for no reason.
#   - the demo and the API can be deployed independently. swap one,
#     leave the other running.
#
# the demo does not load any model itself — it only calls the API.
# every container in this project follows the single-responsibility
# pattern: API does inference, demo renders UI, no overlap.

FROM python:3.11-slim AS runtime

# minimal system libs — Pillow for image decoding, the rest comes
# from pip wheels. apt-get clean removes the package index after
# install so the layer stays small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# non-root user — never serve as root. matches the API container's
# uid/gid so bind-mounted volumes have predictable ownership.
RUN groupadd -r -g 1001 fairage && \
    useradd -r -u 1001 -g 1001 -m -d /home/fairage fairage

WORKDIR /app

# only the deps the demo needs — streamlit itself, requests for API
# calls, pillow + numpy for the saliency overlay. NO torch, NO onnxruntime.
# keeps the demo image about 1/3 the size of the API image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        streamlit==1.39.0 \
        requests==2.32.3 \
        pillow==10.4.0 \
        numpy==1.26.4

# copy only the Streamlit app code. src/ is not needed — the demo
# talks to the API over HTTP, not via Python imports.
COPY streamlit_app/ /app/streamlit_app/

RUN chown -R fairage:fairage /app /home/fairage

USER fairage

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# port 8503 — same as src/config.py STREAMLIT_PORT. nginx proxies
# /fairage-demo/ to this port, bound to 127.0.0.1 by docker-compose.
EXPOSE 8503

# Streamlit's healthcheck endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8503/fairage-demo/_stcore/health')" || exit 1

# server.address 0.0.0.0 lets the container accept traffic from the
# docker bridge. the port is still bound to 127.0.0.1 on the host
# (set in docker-compose.yml), so nothing on 0.0.0.0:8503 leaks.
# server.baseUrlPath matches the nginx /fairage-demo prefix so links
# inside the page resolve correctly through the reverse proxy.
CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8503", \
     "--server.baseUrlPath", "/fairage-demo", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
