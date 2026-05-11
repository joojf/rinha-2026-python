import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from main import normalize


def approx_vec(actual, expected, tol=0.0001):
    assert len(actual) == len(expected) == 14
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) <= tol, (
            f"dim {i}: got {a:.6f}, expected {e:.6f} (diff={abs(a-e):.6f})"
        )


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


def test_normalize_legit_vector():
    approx_vec(normalize(TX_LEGIT), EXPECTED_LEGIT)


def test_normalize_fraud_vector():
    approx_vec(normalize(TX_FRAUD), EXPECTED_FRAUD)


def test_normalize_output_length():
    assert len(normalize(TX_LEGIT)) == 14


def test_normalize_sentinel_only_dims_5_6():
    vec = normalize(TX_LEGIT)
    assert vec[5] == -1.0
    assert vec[6] == -1.0
    for i, v in enumerate(vec):
        if i not in (5, 6):
            assert 0.0 <= v <= 1.0, f"dim {i} has unexpected value {v}"


def test_normalize_last_tx_present():
    data = {
        "id": "tx-001",
        "transaction": {
            "amount": 100.0,
            "installments": 1,
            "requested_at": "2026-03-11T12:00:00Z",
        },
        "customer": {"avg_amount": 100.0, "tx_count_24h": 1, "known_merchants": []},
        "merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 100.0},
        "terminal": {"is_online": True, "card_present": False, "km_from_home": 0.0},
        "last_transaction": {
            "timestamp": "2026-03-11T06:00:00Z",
            "km_from_current": 50.0,
        },
    }
    vec = normalize(data)
    assert 0.0 <= vec[5] <= 1.0
    assert 0.0 <= vec[6] <= 1.0
    assert abs(vec[5] - 0.25) < 0.0001
    assert abs(vec[6] - 0.05) < 0.0001


def test_normalize_clamp():
    data = {
        "id": "tx-002",
        "transaction": {"amount": 99_999.0, "installments": 100, "requested_at": "2026-01-01T00:00:00Z"},
        "customer": {"avg_amount": 1.0, "tx_count_24h": 100, "known_merchants": []},
        "merchant": {"id": "MERC-X", "mcc": "9999", "avg_amount": 99_999.0},
        "terminal": {"is_online": True, "card_present": True, "km_from_home": 99_999.0},
        "last_transaction": None,
    }
    vec = normalize(data)
    assert vec[0] == 1.0
    assert vec[1] == 1.0
    assert vec[2] == 1.0
    assert vec[7] == 1.0
    assert vec[8] == 1.0
    assert vec[13] == 1.0


def test_normalize_unknown_mcc_defaults_to_half():
    data = {
        "id": "tx-003",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": ["MERC-A"]},
        "merchant": {"id": "MERC-A", "mcc": "0000", "avg_amount": 50.0},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 5.0},
        "last_transaction": None,
    }
    assert normalize(data)[12] == 0.5


def test_normalize_known_vs_unknown_merchant():
    base = {
        "id": "tx-004",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": ["MERC-A", "MERC-B"]},
        "merchant": {"id": "MERC-A", "mcc": "5411", "avg_amount": 50.0},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 5.0},
        "last_transaction": None,
    }
    assert normalize(base)[11] == 0.0

    unknown = {**base, "merchant": {**base["merchant"], "id": "MERC-UNKNOWN"}}
    assert normalize(unknown)[11] == 1.0


def test_normalize_is_online_card_present_flags():
    base = {
        "id": "tx-005",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-01-01T08:00:00Z"},
        "customer": {"avg_amount": 50.0, "tx_count_24h": 1, "known_merchants": []},
        "merchant": {"id": "MERC-A", "mcc": "5411", "avg_amount": 50.0},
        "terminal": {"is_online": True, "card_present": False, "km_from_home": 0.0},
        "last_transaction": None,
    }
    vec = normalize(base)
    assert vec[9] == 1.0
    assert vec[10] == 0.0

    offline = {**base, "terminal": {"is_online": False, "card_present": True, "km_from_home": 0.0}}
    vec2 = normalize(offline)
    assert vec2[9] == 0.0
    assert vec2[10] == 1.0
