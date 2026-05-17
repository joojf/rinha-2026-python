FROM --platform=linux/amd64 python:3.12-slim AS index-downloader
RUN python3 -c "import urllib.request, os; os.makedirs('/data', exist_ok=True); urllib.request.urlretrieve('https://github.com/joojf/rinha-2026-python/releases/download/v2-index/index.faiss', '/data/index.faiss'); urllib.request.urlretrieve('https://github.com/joojf/rinha-2026-python/releases/download/v2-index/labels.npy', '/data/labels.npy')"

FROM --platform=linux/amd64 python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY --from=index-downloader /data/index.faiss /data/index.faiss
COPY --from=index-downloader /data/labels.npy /data/labels.npy
RUN test -f /data/index.faiss && test -f /data/labels.npy
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072
CMD ["sh", "-c", "umask 0000 && exec uvicorn main:app --loop uvloop --http httptools --workers 1 --uds ${LISTEN_UDS:-/sockets/api.sock} --no-access-log --limit-concurrency 256 --backlog 2048 --timeout-keep-alive 30"]
