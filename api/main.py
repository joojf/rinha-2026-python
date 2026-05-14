import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import asyncio

import numpy as np
import faiss
import orjson

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

INDEX_PATH = "/data/index.faiss"
LABELS_PATH = "/data/labels.npy"

K = 5
THRESHOLD = 0.6

ready = False
index = None
labels = None


def _load_index():
    global index, labels
    print("[startup] loading index...", flush=True)
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    index.nprobe = 16
    labels = np.load(LABELS_PATH, mmap_mode="r")
    print("[startup] index loaded.", flush=True)


async def on_startup():
    global ready
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _load_index)
        ready = True
        print("[startup] ready=True", flush=True)
    except Exception as exc:
        print(f"[startup] FAILED: {exc}", flush=True)
        raise


async def fraud_score(request: object) -> Response:
    vec = np.zeros((1, 14), dtype=np.float32)
    _, indices = index.search(vec, K)

    neighbor_labels = labels[indices[0]]
    score = float(np.sum(neighbor_labels == 1)) / K
    approved = score < THRESHOLD

    return Response(
        orjson.dumps({"approved": approved, "fraud_score": score}),
        media_type="application/json",
    )


async def ready_check(request: object) -> Response:
    if not ready:
        return Response("loading", status_code=503)
    return Response("ok", status_code=200)


app = Starlette(
    on_startup=[on_startup],
    routes=[
        Route("/fraud-score", fraud_score, methods=["POST"]),
        Route("/ready", ready_check, methods=["GET"]),
    ],
)
