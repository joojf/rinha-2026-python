FROM --platform=linux/amd64 python:3.12-slim AS index-builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
RUN python -c "from main import _build_or_load_index; _build_or_load_index()"

FROM --platform=linux/amd64 python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY --from=index-builder /data/index.faiss /data/index.faiss
COPY --from=index-builder /data/labels.npy /data/labels.npy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--loop", "uvloop", "--http", "httptools", "--workers", "1"]
