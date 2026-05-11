#!/usr/bin/env bash
# Download the official Rinha de Backend 2026 dataset files into ./data/
set -euo pipefail

BASE="https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources"
mkdir -p data

echo "Downloading references.json.gz (~16 MB)..."
curl -L --progress-bar "$BASE/references.json.gz" -o data/references.json.gz

echo "Downloading mcc_risk.json..."
curl -L --silent "$BASE/mcc_risk.json" -o data/mcc_risk.json

echo "Downloading normalization.json..."
curl -L --silent "$BASE/normalization.json" -o data/normalization.json

echo "Done. Files in ./data/:"
ls -lh data/
