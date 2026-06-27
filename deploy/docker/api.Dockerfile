FROM python:3.11-slim

# Keep the image lean: no .pyc cache, no pip cache, no apt cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is for the HEALTHCHECK below; nothing else needs build tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the layer cache survives source edits.
COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/api /app/apps/api

# Corpus + dense-embedding cache live under /app/data, bind-mounted at
# runtime by docker-compose. The container ships with an empty data/
# directory; the volume mount overlays the host's corpus.
RUN mkdir -p /app/data/corpus /app/data/cache

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "apps/api"]
