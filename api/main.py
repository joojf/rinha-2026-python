import os
import gzip
import asyncio
import concurrent.futures
from datetime import datetime
import numpy as np
import faiss
import ijson
import orjson
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

INDEX_CACHE_PATH = "/data/index.faiss"
LABELS_CACHE_PATH = "/data/labels.npy"
REFERENCES_PATH = "/data/references.json.gz"
REFERENCES_URL = "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz"

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
NLIST = 1024
NPROBE = 16
THRESHOLD = 0.6
TRAIN_SIZE = 150_000
BATCH_SIZE = 300_000

index: faiss.Index | None = None
labels: np.ndarray | None = None
ready: bool = False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


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


def _stream_records(path: str):
    with gzip.open(path, "rb") as f:
        yield from ijson.items(f, "item")


def _ensure_dataset() -> None:
    if not os.path.exists(REFERENCES_PATH):
        import urllib.request
        print("[api] Dataset not found — downloading from GitHub...")
        os.makedirs(os.path.dirname(REFERENCES_PATH), exist_ok=True)
        urllib.request.urlretrieve(REFERENCES_URL, REFERENCES_PATH)
        print("[api] Download complete.")


def _build_or_load_index() -> None:
    global index, labels, ready

    if os.path.exists(INDEX_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        print("[api] mmap-loading cached FAISS index...")
        index = faiss.read_index(INDEX_CACHE_PATH, faiss.IO_FLAG_MMAP)
        index.nprobe = NPROBE
        labels = np.load(LABELS_CACHE_PATH, mmap_mode="r")
        warmup_vec = np.zeros((1, DIM), dtype=np.float32)
        for _ in range(8):
            index.search(warmup_vec, K)
        ready = True
        print(f"[api] Ready — {index.ntotal:,} vectors mmap'd.")
        return

    _ensure_dataset()

    print("[api] Building FAISS index from dataset...")
    MAX_RECORDS = 4_000_000
    labels_buf = bytearray(MAX_RECORDS)

    print("[api] Pass 1 — collecting training sample...")
    train_vecs = np.empty((TRAIN_SIZE, DIM), dtype=np.float32)
    train_count = 0
    for record in _stream_records(REFERENCES_PATH):
        if train_count >= TRAIN_SIZE:
            break
        train_vecs[train_count] = record["vector"]
        train_count += 1
    train_vecs = train_vecs[:train_count]

    quantizer = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFScalarQuantizer(
        quantizer, DIM, NLIST, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2,
    )
    idx.train(train_vecs)
    del train_vecs

    print("[api] Pass 2 — adding all vectors to index...")
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

    if batch_len > 0:
        flush_batch()

    del batch_vecs
    idx.nprobe = NPROBE
    faiss.write_index(idx, INDEX_CACHE_PATH)
    labels_arr = np.frombuffer(labels_buf[:total], dtype=np.uint8).copy()
    np.save(LABELS_CACHE_PATH, labels_arr)
    del labels_buf

    index = idx
    labels = labels_arr
    ready = True
    print("[api] Index built and cached. Ready.")


def _faiss_search(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return index.search(vec, K)


async def fraud_score(request: object) -> Response:
    body = await request.body()
    data = orjson.loads(body)
    vec = np.array(normalize(data), dtype=np.float32).reshape(1, DIM)

    loop = asyncio.get_running_loop()
    _, ids = await loop.run_in_executor(executor, _faiss_search, vec)

    valid_ids = ids[0][ids[0] >= 0]
    fraud_score_val = int(labels[valid_ids].sum()) / K if len(valid_ids) > 0 else 0.0

    return Response(
        orjson.dumps({"approved": fraud_score_val < THRESHOLD, "fraud_score": fraud_score_val}),
        media_type="application/json",
    )


async def ready_check(request: object) -> Response:
    if ready:
        return Response("ok", status_code=200)
    return Response("not ready", status_code=503)


async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _build_or_load_index)


app = Starlette(
    routes=[
        Route("/fraud-score", fraud_score, methods=["POST"]),
        Route("/ready", ready_check, methods=["GET"]),
    ],
    on_startup=[on_startup],
)
