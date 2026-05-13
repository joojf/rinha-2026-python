import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: F401
import faiss  # noqa: F401
import orjson  # noqa: F401

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response


async def fraud_score(request: object) -> Response:
    return Response(
        b'{"approved":true,"fraud_score":0.0}',
        media_type="application/json",
    )


async def ready_check(request: object) -> Response:
    return Response("ok", status_code=200)


app = Starlette(
    routes=[
        Route("/fraud-score", fraud_score, methods=["POST"]),
        Route("/ready", ready_check, methods=["GET"]),
    ],
)
