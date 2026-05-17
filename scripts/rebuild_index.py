"""Rebuild FAISS index with QT_fp16 (instead of QT_8bit) for better accuracy.

Usage:
    python3.12 scripts/rebuild_index.py

Outputs:
    data/index_fp16.faiss
    data/labels.npy  (unchanged, but re-written for consistency)
"""
import os
import gzip
import time

import numpy as np
import faiss
import ijson

REFERENCES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "references.json.gz")
INDEX_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "index_fp16.faiss")
LABELS_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "labels_fp16.npy")

DIM = 14
NLIST = 4096
TRAIN_SIZE = 200_000
BATCH_SIZE = 300_000
MAX_RECORDS = 4_000_000


def stream_records(path: str):
    with gzip.open(path, "rb") as f:
        for record in ijson.items(f, "item"):
            yield record


def main() -> None:
    t0 = time.perf_counter()

    print(f"[rebuild] References: {REFERENCES_PATH}")
    print(f"[rebuild] Output: {INDEX_OUT}")

    labels_buf = bytearray(MAX_RECORDS)

    # Pass 1: training sample
    print("[rebuild] Pass 1 — collecting training sample...", flush=True)
    train_vecs = np.empty((TRAIN_SIZE, DIM), dtype=np.float32)
    train_count = 0
    for record in stream_records(REFERENCES_PATH):
        if train_count >= TRAIN_SIZE:
            break
        train_vecs[train_count] = record["vector"]
        train_count += 1
    train_vecs = train_vecs[:train_count]
    print(f"[rebuild] {train_count:,} training vectors collected ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Build index with QT_fp16
    quantizer = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFScalarQuantizer(
        quantizer, DIM, NLIST,
        faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_L2,
    )
    print("[rebuild] Training IVF + QT_fp16...", flush=True)
    idx.train(train_vecs)
    del train_vecs
    print(f"[rebuild] Training done ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Pass 2: add all vectors in batches
    print("[rebuild] Pass 2 — adding all vectors...", flush=True)
    batch_vecs = np.empty((BATCH_SIZE, DIM), dtype=np.float32)
    batch_len = 0
    total = 0

    for record in stream_records(REFERENCES_PATH):
        vec = record["vector"]
        label = 1 if record["label"] == "fraud" else 0
        batch_vecs[batch_len] = vec
        labels_buf[total] = label
        batch_len += 1
        total += 1
        if batch_len == BATCH_SIZE:
            idx.add(batch_vecs[:batch_len])
            print(f"[rebuild]   added {total:,} vectors ({time.perf_counter()-t0:.1f}s)", flush=True)
            batch_len = 0

    if batch_len > 0:
        idx.add(batch_vecs[:batch_len])
        total += 0

    print(f"[rebuild] Total: {idx.ntotal:,} vectors ({time.perf_counter()-t0:.1f}s)", flush=True)

    labels_arr = np.frombuffer(labels_buf, dtype=np.uint8, count=total).copy()
    np.save(LABELS_OUT, labels_arr)
    print(f"[rebuild] Labels saved → {LABELS_OUT} (fraud_rate={labels_arr.mean():.4f})", flush=True)

    faiss.write_index(idx, INDEX_OUT)
    size_mb = os.path.getsize(INDEX_OUT) / 1024 / 1024
    print(f"[rebuild] Index saved → {INDEX_OUT} ({size_mb:.1f} MB)", flush=True)
    print(f"[rebuild] Done in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
