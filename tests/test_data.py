"""Tests for the dataset registry and loaders (use the cached data in data/raw/)."""
import pytest

from src import data


def test_registry_ids_are_allowed():
    for spec in data.SPECS.values():
        assert spec.uci_id in data.ALLOWED_UCI_IDS


def test_disallowed_id_is_rejected_without_network():
    # 891 (CDC diabetes) was dropped from the plan, so it must never load.
    with pytest.raises(AssertionError):
        data.assert_is_uci(891)
    with pytest.raises(AssertionError):
        data.assert_is_uci(770)  # NASA Flood, left out by choice


@pytest.mark.parametrize(
    "key,rows,positives",
    [
        ("maternal", 1014, 608),   # mid or high vs low
        ("ctg", 2126, 471),        # NSP 2 or 3 vs 1
        ("oral_tox", 8992, 741),
        ("androgen", 1687, 199),
        ("biodeg", 1055, 356),     # positive = RB
    ],
)
def test_binary_labels_and_sizes(key, rows, positives):
    X, y = data.split_xy(key)
    assert len(X) == len(y) == rows
    assert set(y.unique()) <= {0, 1}
    assert int(y.sum()) == positives


def test_targets_never_stay_in_features():
    X, _ = data.split_xy("ctg")
    assert "CLASS" not in X.columns and "NSP" not in X.columns
    X, _ = data.split_xy("maternal")
    assert "RiskLevel" not in X.columns


def test_dedup_removes_all_exact_duplicates():
    X, y = data.split_xy("maternal")
    Xd, yd = data.dedup(X, y)
    full = Xd.copy()
    full["__y__"] = yd.values
    assert not full.duplicated().any()
    assert len(Xd) < len(X)
