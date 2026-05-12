import os
import gzip
import asyncio
import concurrent.futures
import numpy as np
import faiss
import ijson
import orjson
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

REFERENCES_PATH = "/data/references.json.gz"
REFERENCES_URL = "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz"
INDEX_CACHE_PATH = "/data/index.faiss"
LABELS_CACHE_PATH = "/data/labels.npy"

index: faiss.Index | None = None
labels: np.ndarray | None = None
ready: bool = False

DIM = 14
K = 5
NLIST = 1024
NPROBE = 16
THRESHOLD = 0.6
TRAIN_SIZE = 150_000
BATCH_SIZE = 300_000

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _stream_records(path: str):
    with gzip.open(path, "rb") as f:
        yield from ijson.items(f, "item")


def _ensure_dataset() -> None:
    if not os.path.exists(REFERENCES_PATH):
        import urllib.request
        print("[searcher] Dataset not found — downloading from GitHub...")
        os.makedirs(os.path.dirname(REFERENCES_PATH), exist_ok=True)
        urllib.request.urlretrieve(REFERENCES_URL, REFERENCES_PATH)
        print("[searcher] Download complete.")


def _build_or_load_index() -> None:
    global index, labels, ready

    if os.path.exists(INDEX_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        print("[searcher] mmap-loading cached FAISS index...")
        index = faiss.read_index(INDEX_CACHE_PATH, faiss.IO_FLAG_MMAP)
        index.nprobe = NPROBE
        labels = np.load(LABELS_CACHE_PATH, mmap_mode="r")
        warmup_vec = np.zeros((1, DIM), dtype=np.float32)
        for _ in range(8):
            index.search(warmup_vec, K)
        ready = True
        print(f"[searcher] Ready — {index.ntotal:,} vectors mmap'd.")
        return

    _ensure_dataset()

    print("[searcher] Building FAISS index from dataset...")

    MAX_RECORDS = 4_000_000
    labels_buf = bytearray(MAX_RECORDS)

    print("[searcher] Pass 1 — collecting training sample...")
    train_vecs = np.empty((TRAIN_SIZE, DIM), dtype=np.float32)
    train_count = 0

    for record in _stream_records(REFERENCES_PATH):
        if train_count >= TRAIN_SIZE:
            break
        train_vecs[train_count] = record["vector"]
        train_count += 1

    train_vecs = train_vecs[:train_count]
    print(f"[searcher] Collected {train_count:,} training vectors.")

    quantizer = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFScalarQuantizer(
        quantizer,
        DIM,
        NLIST,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_L2,
    )
    print("[searcher] Training IVF + scalar quantizer...")
    idx.train(train_vecs)
    del train_vecs

    print("[searcher] Pass 2 — adding all vectors to index...")
    batch_vecs = np.empty((BATCH_SIZE, DIM), dtype=np.float32)
    batch_len = 0
    total = 0

    def flush_batch():
        nonlocal batch_len, total
        idx.add(batch_vecs[:batch_len])
        total += batch_len
        batch_len = 0

    for record in _stream_records(REFERENCES_PATH):
        vec = record["vector"]
        label = 1 if record["label"] == "fraud" else 0
        batch_vecs[batch_len] = vec
        labels_buf[total + batch_len] = label
        batch_len += 1
        if batch_len == BATCH_SIZE:
            flush_batch()
            if total % 1_000_000 == 0:
                print(f"[searcher]   {total:,} vectors added...")

    if batch_len > 0:
        flush_batch()

    del batch_vecs
    print(f"[searcher] All {total:,} vectors added.")

    idx.nprobe = NPROBE
    faiss.write_index(idx, INDEX_CACHE_PATH)

    labels_arr = np.frombuffer(labels_buf[:total], dtype=np.uint8).copy()
    np.save(LABELS_CACHE_PATH, labels_arr)
    del labels_buf

    index = idx
    labels = labels_arr
    ready = True
    print("[searcher] Index built and cached. Ready.")


def _faiss_search(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return index.search(vec, K)


async def search(request: object) -> Response:
    body = await request.body()
    vec = np.array(orjson.loads(body), dtype=np.float32).reshape(1, DIM)

    loop = asyncio.get_running_loop()
    _, ids = await loop.run_in_executor(executor, _faiss_search, vec)

    valid_ids = ids[0][ids[0] >= 0]
    fraud_score_val = int(labels[valid_ids].sum()) / K if len(valid_ids) > 0 else 0.0

    return Response(
        orjson.dumps({"approved": fraud_score_val < THRESHOLD, "fraud_score": fraud_score_val}),
        media_type="application/json",
    )


async def health(request: object) -> Response:
    if ready:
        return Response("ok", status_code=200)
    return Response("not ready", status_code=503)


async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _build_or_load_index)


app = Starlette(
    routes=[
        Route("/search", search, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ],
    on_startup=[on_startup],
)
