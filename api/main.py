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
