"""
Unit tests for the normalize() function.
Ground-truth vectors taken verbatim from REGRAS_DE_DETECCAO.md.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from main import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_vec(actual, expected, tol=0.0001):
    assert len(actual) == len(expected) == 14
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) <= tol, (
            f"dim {i}: got {a:.6f}, expected {e:.6f} (diff={abs(a-e):.6f})"
        )


# ---------------------------------------------------------------------------
# Example 1 — legítima (from REGRAS_DE_DETECCAO.md, "Visão geral do fluxo")
# ---------------------------------------------------------------------------

TX_LEGIT = {
    "id": "tx-1329056812",
    "transaction": {
        "amount": 41.12,
        "installments": 2,
        "requested_at": "2026-03-11T18:45:53Z",
    },
    "customer": {
        "avg_amount": 82.24,
        "tx_count_24h": 3,
        "known_merchants": ["MERC-003", "MERC-016"],
    },
    "merchant": {
        "id": "MERC-016",
        "mcc": "5411",
        "avg_amount": 60.25,
    },
    "terminal": {
        "is_online": False,
        "card_present": True,
        "km_from_home": 29.23,
    },
    "last_transaction": None,
}

EXPECTED_LEGIT = [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]


def test_normalize_legit_vector():
    vec = normalize(TX_LEGIT)
    approx_vec(vec, EXPECTED_LEGIT)


def test_normalize_legit_approved():
    vec = normalize(TX_LEGIT)
    fraud_count = 0  # all 5 neighbors are legit in the example
    fraud_score = fraud_count / 5
    assert fraud_score < 0.6  # would be approved


# ---------------------------------------------------------------------------
# Example 2 — fraudulenta (from "Exemplo de transação fraudulenta")
# ---------------------------------------------------------------------------

TX_FRAUD = {
    "id": "tx-3330991687",
    "transaction": {
        "amount": 9505.97,
        "installments": 10,
        "requested_at": "2026-03-14T05:15:12Z",
    },
    "customer": {
        "avg_amount": 81.28,
        "tx_count_24h": 20,
        "known_merchants": ["MERC-008", "MERC-007", "MERC-005"],
    },
    "merchant": {
        "id": "MERC-068",
        "mcc": "7802",
        "avg_amount": 54.86,
    },
    "terminal": {
        "is_online": False,
        "card_present": True,
        "km_from_home": 952.27,
    },
    "last_transaction": None,
}

EXPECTED_FRAUD = [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]


def test_normalize_fraud_vector():
    vec = normalize(TX_FRAUD)
    approx_vec(vec, EXPECTED_FRAUD)


def test_normalize_fraud_not_approved():
    vec = normalize(TX_FRAUD)
    fraud_count = 5  # all 5 neighbors are fraud in the example
    fraud_score = fraud_count / 5
    assert fraud_score >= 0.6  # would not be approved


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_normalize_last_tx_present():
    """When last_transaction is present, dims 5 and 6 must be in [0, 1]."""
    data = {
        "id": "tx-001",
        "transaction": {
            "amount": 100.0,
            "installments": 1,
            "requested_at": "2026-03-11T12:00:00Z",
        },
        "customer": {
            "avg_amount": 100.0,
            "tx_count_24h": 1,
            "known_merchants": [],
        },
        "merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 100.0},
        "terminal": {"is_online": True, "card_present": False, "km_from_home": 0.0},
        "last_transaction": {
            "timestamp": "2026-03-11T06:00:00Z",   # 6 hours = 360 minutes ago
            "km_from_current": 50.0,
        },
    }
    vec = normalize(data)
    assert 0.0 <= vec[5] <= 1.0, f"dim 5 (minutes_since_last_tx) out of range: {vec[5]}"
    assert 0.0 <= vec[6] <= 1.0, f"dim 6 (km_from_last_tx) out of range: {vec[6]}"
    # 360 min / 1440 max = 0.25
    assert abs(vec[5] - 0.25) < 0.0001
    # 50 km / 1000 max = 0.05
    assert abs(vec[6] - 0.05) < 0.0001


def test_normalize_clamp_amount():
    """Values over the max must be clamped to 1.0."""
    data = {
        "id": "tx-002",
        "transaction": {"amount": 99_999.0, "installments": 100, "requested_at": "2026-01-01T00:00:00Z"},
        "customer": {"avg_amount": 1.0, "tx_count_24h": 100, "known_merchants": []},
        "merchant": {"id": "MERC-X", "mcc": "9999", "avg_amount": 99_999.0},
        "terminal": {"is_online": True, "card_present": True, "km_from_home": 99_999.0},
        "last_transaction": None,
    }
    vec = normalize(data)
    assert vec[0] == 1.0   # amount clamped
    assert vec[1] == 1.0   # installments clamped
    assert vec[2] == 1.0   # amount_vs_avg clamped
    assert vec[7] == 1.0   # km_from_home clamped
    assert vec[8] == 1.0   # tx_count_24h clamped
    assert vec[13] == 1.0  # merchant_avg_amount clamped


def test_normalize_unknown_mcc_default():
    """Unknown MCCs must default to 0.5."""
    data = {
        "id": "tx-003",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": ["MERC-A"]},
        "merchant": {"id": "MERC-A", "mcc": "0000", "avg_amount": 50.0},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 5.0},
        "last_transaction": None,
    }
    vec = normalize(data)
    assert vec[12] == 0.5


def test_normalize_known_vs_unknown_merchant():
    base = {
        "id": "tx-004",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": ["MERC-A", "MERC-B"]},
        "merchant": {"id": "MERC-A", "mcc": "5411", "avg_amount": 50.0},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 5.0},
        "last_transaction": None,
    }
    vec_known = normalize(base)
    assert vec_known[11] == 0.0  # known merchant

    unknown = dict(base)
    unknown["merchant"] = dict(base["merchant"])
    unknown["merchant"]["id"] = "MERC-UNKNOWN"
    vec_unknown = normalize(unknown)
    assert vec_unknown[11] == 1.0  # unknown merchant


def test_normalize_is_online_and_card_present_flags():
    base = {
        "id": "tx-005",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": []},
        "merchant": {"id": "MERC-A", "mcc": "5411", "avg_amount": 50.0},
        "terminal": {"is_online": True, "card_present": False, "km_from_home": 0.0},
        "last_transaction": None,
    }
    vec = normalize(base)
    assert vec[9] == 1.0   # is_online=True
    assert vec[10] == 0.0  # card_present=False

    offline = dict(base)
    offline["terminal"] = {"is_online": False, "card_present": True, "km_from_home": 0.0}
    vec2 = normalize(offline)
    assert vec2[9] == 0.0  # is_online=False
    assert vec2[10] == 1.0  # card_present=True


def test_normalize_output_length():
    vec = normalize(TX_LEGIT)
    assert len(vec) == 14


def test_normalize_sentinel_values_only_dims_5_6():
    """Only dims 5 and 6 may be -1 (when last_transaction is null)."""
    vec = normalize(TX_LEGIT)
    assert vec[5] == -1.0
    assert vec[6] == -1.0
    for i, v in enumerate(vec):
        if i not in (5, 6):
            assert 0.0 <= v <= 1.0, f"dim {i} has unexpected value {v}"
