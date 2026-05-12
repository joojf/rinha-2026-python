#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD=0
SMOKE=0
READY_TIMEOUT=600

usage() {
    echo "Usage: $0 [--build] [--smoke]"
    echo "  --build   Rebuild Docker images before running"
    echo "  --smoke   Run smoke test only (5 requests, fast)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --build) BUILD=1 ;;
        --smoke) SMOKE=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

cd "$ROOT"

cleanup() {
    echo ""
    echo "==> Bringing down stack..."
    docker compose down
}
trap cleanup EXIT

if [[ $BUILD -eq 1 ]]; then
    echo "==> Building images..."
    docker compose build
fi

echo "==> Starting stack..."
docker compose up -d

echo "==> Waiting for stack to be ready (searcher loads FAISS index, may take a few minutes)..."
ELAPSED=0
until curl -sf http://localhost:9999/ready > /dev/null 2>&1; do
    if [[ $ELAPSED -ge $READY_TIMEOUT ]]; then
        echo "ERROR: Stack not ready after ${READY_TIMEOUT}s"
        docker compose logs
        exit 1
    fi
    printf "    waiting... %ds\r" "$ELAPSED"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "    Ready after ${ELAPSED}s.           "

echo ""
if [[ $SMOKE -eq 1 ]]; then
    echo "==> Running smoke test..."
    K6_NO_USAGE_REPORT=true k6 run rinha-ref/test/smoke.js
else
    echo "==> Running full load test..."
    K6_NO_USAGE_REPORT=true k6 run rinha-ref/test/test.js
    echo ""
    echo "==> Results:"
    cat test/results.json
fi
