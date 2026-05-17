import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import asyncio
import concurrent.futures
from datetime import datetime

import numpy as np
import faiss
import orjson

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

INDEX_PATH = "/data/index.faiss"
LABELS_PATH = "/data/labels.npy"

MAX_AMOUNT = 10_000.0
MAX_INSTALLMENTS = 12.0
AMOUNT_VS_AVG_RATIO = 10.0
MAX_MINUTES = 1_440.0
MAX_KM = 1_000.0
MAX_TX_COUNT_24H = 20.0
MAX_MERCHANT_AVG_AMOUNT = 10_000.0

MCC_RISK: dict[str, float] = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}

DIM = 14
K = 5
NPROBE_FAST = 12
NPROBE_FULL = 20

FRAUD_RESPONSES: list[bytes] = [
    b'{"approved":true,"fraud_score":0.0}',
    b'{"approved":true,"fraud_score":0.2}',
    b'{"approved":true,"fraud_score":0.4}',
    b'{"approved":false,"fraud_score":0.6}',
    b'{"approved":false,"fraud_score":0.8}',
    b'{"approved":false,"fraud_score":1.0}',
]

index: faiss.Index | None = None
labels: np.ndarray | None = None
ready: bool = False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize(data: dict) -> list[float]:
    tx = data["transaction"]
    customer = data["customer"]
    merchant = data["merchant"]
    terminal = data["terminal"]
    last_tx = data.get("last_transaction")

    amount: float = tx["amount"]
    installments: int = tx["installments"]
    requested_at: str = tx["requested_at"]
    avg_amount: float = customer["avg_amount"]
    tx_count_24h: int = customer["tx_count_24h"]
    known_merchants: list[str] = customer["known_merchants"]
    merchant_id: str = merchant["id"]
    mcc: str = merchant["mcc"]
    merchant_avg: float = merchant["avg_amount"]
    is_online: bool = terminal["is_online"]
    card_present: bool = terminal["card_present"]
    km_from_home: float = terminal["km_from_home"]

    dt = datetime.fromisoformat(requested_at.replace("Z", "+00:00"))
    hour = dt.hour
    dow = dt.weekday()

    v0 = _clamp(amount / MAX_AMOUNT)
    v1 = _clamp(installments / MAX_INSTALLMENTS)
    v2 = _clamp((amount / avg_amount) / AMOUNT_VS_AVG_RATIO) if avg_amount > 0 else 1.0
    v3 = hour / 23.0
    v4 = dow / 6.0

    if last_tx is None:
        v5 = -1.0
        v6 = -1.0
    else:
        last_ts = datetime.fromisoformat(last_tx["timestamp"].replace("Z", "+00:00"))
        minutes = (dt - last_ts).total_seconds() / 60.0
        v5 = _clamp(minutes / MAX_MINUTES)
        v6 = _clamp(last_tx["km_from_current"] / MAX_KM)

    v7 = _clamp(km_from_home / MAX_KM)
    v8 = _clamp(tx_count_24h / MAX_TX_COUNT_24H)
    v9 = 1.0 if is_online else 0.0
    v10 = 1.0 if card_present else 0.0
    v11 = 0.0 if merchant_id in known_merchants else 1.0
    v12 = MCC_RISK.get(mcc, 0.5)
    v13 = _clamp(merchant_avg / MAX_MERCHANT_AVG_AMOUNT)

    return [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]


def _load_index() -> None:
    global index, labels, ready
    print("[startup] loading index...", flush=True)
    try:
        index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
        labels = np.load(LABELS_PATH, mmap_mode="r")
        index.nprobe = NPROBE_FAST
        warmup_vec = np.zeros((1, DIM), dtype=np.float32)
        for _ in range(4):
            index.search(warmup_vec, K)
        ready = True
        print(f"[startup] index loaded — {index.ntotal:,} vectors. ready=True", flush=True)
    except Exception as exc:
        import traceback
        print(f"[startup] FAILED: {exc}", flush=True)
        traceback.print_exc()


def _faiss_search(vec: np.ndarray, nprobe: int) -> np.ndarray:
    index.nprobe = nprobe
    _, ids = index.search(vec, K)
    return ids[0]


def _count_fraud(ids_row: np.ndarray) -> int:
    valid = ids_row[ids_row >= 0]
    if len(valid) == 0:
        return 0
    return int(labels[valid].sum())


async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_index)


async def fraud_score(request) -> Response:
    body = await request.body()
    data = orjson.loads(body)
    vec = np.array(normalize(data), dtype=np.float32).reshape(1, DIM)

    loop = asyncio.get_running_loop()
    ids = await loop.run_in_executor(executor, _faiss_search, vec, NPROBE_FAST)
    fraud_count = _count_fraud(ids)

    if fraud_count == 2 or fraud_count == 3:
        ids = await loop.run_in_executor(executor, _faiss_search, vec, NPROBE_FULL)
        fraud_count = _count_fraud(ids)

    return Response(FRAUD_RESPONSES[fraud_count], media_type="application/json")


async def ready_check(request) -> Response:
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
