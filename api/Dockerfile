FROM --platform=linux/amd64 python:3.12-slim AS index-downloader
RUN python3 -c "import urllib.request, os; os.makedirs('/data', exist_ok=True); urllib.request.urlretrieve('https://github.com/joojf/rinha-2026-python/releases/download/v1-index/index.faiss', '/data/index.faiss'); urllib.request.urlretrieve('https://github.com/joojf/rinha-2026-python/releases/download/v1-index/labels.npy', '/data/labels.npy')"

FROM --platform=linux/amd64 python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY --from=index-downloader /data/index.faiss /data/index.faiss
COPY --from=index-downloader /data/labels.npy /data/labels.npy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--loop", "uvloop", "--http", "httptools", "--workers", "1"]
