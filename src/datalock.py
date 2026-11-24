"""Data lock: download each core dataset and recompute every statistic from the raw file.

Writes results/data_cards/<id>.json and results/data_lock.md. Run:
    .venv/bin/python -m src.datalock
Nothing from an agent report is trusted; every number below is computed from the file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import data

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "results" / "data_cards"

# Plan rule: demote 863 if fewer than ~400 unique rows or fewer than 100 per class after dedup.
MIN_UNIQUE_ROWS = 400
MIN_PER_CLASS = 100


def card_for(key: str) -> dict:
    spec = data.SPECS[key]
    raw = data.load_raw(key)
    meta = json.loads((data.RAW / str(spec.uci_id) / "uci_metadata.json").read_text())
    X, y = data.split_xy(key, raw)
    n = len(X)
    pos = int(y.sum())
    maj = max(pos, n - pos) / n
    Xd, yd = data.dedup(X, y)
    nd = len(Xd)
    posd = int(yd.sum())
    card = {
        "uci_id": spec.uci_id,
        "key": key,
        "niche": spec.niche,
        "title": spec.title,
        "uci_metadata": {
            "num_instances": meta.get("num_instances"),
            "num_features": meta.get("num_features"),
            "tasks": meta.get("tasks"),
            "area": meta.get("area"),
            "doi": meta.get("dataset_doi"),
        },
        "binary_rule": spec.rule,
        "rows_in_file": n,
        "n_features_after_split": int(X.shape[1]),
        "positives": pos,
        "negatives": n - pos,
        "positive_rate": round(pos / n, 4),
        "majority_rate": round(maj, 4),
        "exact_duplicate_rows": n - nd,
        "unique_rows": nd,
        "conflicting_feature_groups_raw": data.conflicting_groups(X, y),
        "after_dedup": {
            "rows": nd,
            "positives": posd,
            "negatives": nd - posd,
            "majority_rate": round(max(posd, nd - posd) / nd, 4),
        },
        "missing_cells_in_features": int(X.isna().sum().sum()),
        "feature_columns": [str(c) for c in X.columns[:30]] + (["..."] if X.shape[1] > 30 else []),
    }
    return card


def main() -> None:
    CARDS.mkdir(parents=True, exist_ok=True)
    cards = []
    for key in data.SPECS:
        card = card_for(key)
        (CARDS / f"{card['uci_id']}.json").write_text(json.dumps(card, indent=2))
        cards.append(card)

    lines = ["# Data lock (computed from raw UCI files)", ""]
    lines.append("| UCI | Dataset | Niche | Rows (UCI meta) | Rows (file) | Features | Pos / Neg | Majority | Dup rows | Unique | Conflict groups | After dedup pos/neg |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cards:
        a = c["after_dedup"]
        lines.append(
            f"| {c['uci_id']} | {c['title']} | {c['niche']} | {c['uci_metadata']['num_instances']} | {c['rows_in_file']} | "
            f"{c['n_features_after_split']} | {c['positives']} / {c['negatives']} | {c['majority_rate']:.3f} | "
            f"{c['exact_duplicate_rows']} | {c['unique_rows']} | {c['conflicting_feature_groups_raw']} | "
            f"{a['positives']} / {a['negatives']} |"
        )
    lines.append("")
    m = next(c for c in cards if c["uci_id"] == 863)["after_dedup"]
    ok = m["rows"] >= MIN_UNIQUE_ROWS and min(m["positives"], m["negatives"]) >= MIN_PER_CLASS
    lines.append(
        f"863 demotion rule (>= {MIN_UNIQUE_ROWS} unique rows and >= {MIN_PER_CLASS} per class after dedup): "
        f"{'PASSES, stays core' if ok else 'FAILS, demote to optional'} "
        f"(rows {m['rows']}, positives {m['positives']}, negatives {m['negatives']})."
    )
    lines.append("")
    lines.append("579 (Myocardial infarction complications) is not loaded yet: its post-admission drop list still has to be built from the UCI variable table.")
    (ROOT / "results" / "data_lock.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
