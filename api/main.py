import os
from datetime import datetime, timezone
import asyncio
import httpx
import orjson
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

SEARCHER_URL = os.environ.get("SEARCHER_URL", "http://searcher:8001")

# Normalization constants (from normalization.json — fixed for the competition)
MAX_AMOUNT = 10_000.0
MAX_INSTALLMENTS = 12.0
AMOUNT_VS_AVG_RATIO = 10.0
MAX_MINUTES = 1_440.0
MAX_KM = 1_000.0
MAX_TX_COUNT_24H = 20.0
MAX_MERCHANT_AVG_AMOUNT = 10_000.0

# MCC risk table (from mcc_risk.json — fixed for the competition)
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

# Reused across all requests — do NOT create a new client per request
http_client = httpx.AsyncClient(
    base_url=SEARCHER_URL,
    timeout=3.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
)


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
    hour = dt.hour          # 0-23 UTC
    dow = dt.weekday()      # Mon=0, Sun=6

    # dim 0 — amount
    v0 = _clamp(amount / MAX_AMOUNT)

    # dim 1 — installments
    v1 = _clamp(installments / MAX_INSTALLMENTS)

    # dim 2 — amount vs customer average (10× average → clamped to 1.0)
    if avg_amount > 0:
        v2 = _clamp((amount / avg_amount) / AMOUNT_VS_AVG_RATIO)
    else:
        v2 = 1.0

    # dim 3 — hour of day (0–23 UTC)
    v3 = hour / 23.0

    # dim 4 — day of week (Mon=0 … Sun=6)
    v4 = dow / 6.0

    # dims 5 & 6 — last transaction features (-1 sentinel when null)
    if last_tx is None:
        v5 = -1.0
        v6 = -1.0
    else:
        last_ts = datetime.fromisoformat(
            last_tx["timestamp"].replace("Z", "+00:00")
        )
        minutes = (dt - last_ts).total_seconds() / 60.0
        v5 = _clamp(minutes / MAX_MINUTES)
        v6 = _clamp(last_tx["km_from_current"] / MAX_KM)

    # dim 7 — km from home
    v7 = _clamp(km_from_home / MAX_KM)

    # dim 8 — tx count in last 24h
    v8 = _clamp(tx_count_24h / MAX_TX_COUNT_24H)

    # dim 9 — is_online flag
    v9 = 1.0 if is_online else 0.0

    # dim 10 — card present flag
    v10 = 1.0 if card_present else 0.0

    # dim 11 — unknown merchant (1 = unseen merchant, 0 = known)
    v11 = 0.0 if merchant_id in known_merchants else 1.0

    # dim 12 — MCC risk score (default 0.5 for unknown MCCs)
    v12 = MCC_RISK.get(mcc, 0.5)

    # dim 13 — merchant average ticket
    v13 = _clamp(merchant_avg / MAX_MERCHANT_AVG_AMOUNT)

    return [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]


async def fraud_score(request: object) -> Response:
    body = await request.body()
    data = orjson.loads(body)
    vector = normalize(data)
    resp = await http_client.post(
        "/search",
        content=orjson.dumps(vector),
        headers={"Content-Type": "application/json"},
    )
    return Response(resp.content, media_type="application/json")


async def ready(request: object) -> Response:
    try:
        resp = await http_client.get("/health", timeout=1.0)
        if resp.status_code == 200:
            return Response("ok", status_code=200)
    except Exception:
        pass
    return Response("not ready", status_code=503)


app = Starlette(
    routes=[
        Route("/fraud-score", fraud_score, methods=["POST"]),
        Route("/ready", ready, methods=["GET"]),
    ]
)
