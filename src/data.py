"""Dataset registry, download, and loading for the COGS 118A final project.

Every core dataset must be a UCI Machine Learning Repository dataset. `download()`
asserts this by checking the UCI metadata endpoint (HTTP 200 and matching uci_id)
before any file is fetched. Binary label rules are pre-registered here, before any
modeling, as the plan requires.
"""
from __future__ import annotations

import io
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UCI_API = "https://archive.ics.uci.edu/api/dataset?id={id}"
UCI_STATIC = "https://archive.ics.uci.edu/static/public/{id}/{name}"

# The only dataset IDs allowed anywhere in the core experiments.
ALLOWED_UCI_IDS = {863, 193, 579, 508, 509, 254, 995, 75, 880}


@dataclass(frozen=True)
class Spec:
    uci_id: int
    key: str
    niche: str
    title: str
    fetch: str  # file name under static/public/<id>/ (csv or zip)
    member: str | None  # file inside the zip, None for a plain csv
    sep: str
    header: bool
    encoding: str
    label_col: str | int  # column name, or integer position for headerless files
    positive: Callable[[pd.Series], pd.Series]  # maps raw label column -> bool (True = positive)
    drop_cols: tuple[str, ...] = ()  # extra non-feature columns (besides the label)
    rule: str = ""  # human-readable binary rule, copied into data cards


SPECS: dict[str, Spec] = {
    "maternal": Spec(
        863, "maternal", "D", "Maternal Health Risk (rural Bangladesh)",
        "data.csv", None, ",", True, "utf-8-sig", "RiskLevel",
        lambda s: s.isin(["mid risk", "high risk"]),
        rule="positive = mid or high risk; negative = low risk",
    ),
    "ctg": Spec(
        193, "ctg", "D", "Cardiotocography",
        "data.csv", None, ",", True, "utf-8-sig", "NSP",
        lambda s: s.astype(int).isin([2, 3]),
        drop_cols=("CLASS",),
        rule="positive = NSP 2 (suspect) or 3 (pathologic); negative = NSP 1 (normal); CLASS dropped (encodes the target)",
    ),
    "oral_tox": Spec(
        508, "oral_tox", "B", "QSAR oral toxicity",
        "qsar+oral+toxicity.zip", "qsar_oral_toxicity.csv", ";", False, "utf-8", -1,
        lambda s: s == "positive",
        rule="positive = very toxic ('positive'); negative = not very toxic",
    ),
    "androgen": Spec(
        509, "androgen", "B", "QSAR androgen receptor",
        "qsar+androgen+receptor.zip", "qsar_androgen_receptor.csv", ";", False, "utf-8", -1,
        lambda s: s == "positive",
        rule="positive = binder to androgen receptor ('positive'); negative = non-binder",
    ),
    "biodeg": Spec(
        254, "biodeg", "B", "QSAR biodegradation",
        "qsar+biodegradation.zip", "biodeg.csv", ";", False, "utf-8", -1,
        lambda s: s == "RB",
        rule="positive = RB (ready biodegradable); negative = NRB",
    ),
}


def _fetch(url: str) -> bytes:
    # Use certifi's CA bundle: the python.org macOS build ships without system certs.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        return r.read()


def assert_is_uci(uci_id: int) -> dict:
    """Fail loudly unless `uci_id` is an allowed ID and the UCI API confirms it."""
    if uci_id not in ALLOWED_UCI_IDS:
        raise AssertionError(f"UCI id {uci_id} is not in the allowed core list {sorted(ALLOWED_UCI_IDS)}")
    try:
        meta = json.loads(_fetch(UCI_API.format(id=uci_id)))
    except Exception as exc:  # network error, non-200, bad JSON
        raise AssertionError(f"UCI metadata endpoint failed for id {uci_id}: {exc}") from exc
    data = meta.get("data", {})
    if data.get("uci_id") != uci_id:
        raise AssertionError(f"UCI metadata id mismatch for {uci_id}: got {data.get('uci_id')}")
    return data


def download(key: str, force: bool = False) -> Path:
    """Download one dataset into data/raw/<id>/ and save its UCI metadata JSON."""
    spec = SPECS[key]
    meta = assert_is_uci(spec.uci_id)
    out_dir = RAW / str(spec.uci_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uci_metadata.json").write_text(json.dumps(meta, indent=2))
    dest = out_dir / (spec.member or spec.fetch)
    if dest.exists() and not force:
        return dest
    blob = _fetch(UCI_STATIC.format(id=spec.uci_id, name=spec.fetch))
    if spec.member is None:
        dest.write_bytes(blob)
    else:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            dest.write_bytes(z.read(spec.member))
    return dest


def load_raw(key: str) -> pd.DataFrame:
    """Load the raw table exactly as shipped (no dedup, no cleaning)."""
    spec = SPECS[key]
    path = download(key)
    df = pd.read_csv(
        path, sep=spec.sep, header=0 if spec.header else None,
        encoding=spec.encoding, keep_default_na=False, na_values=[""],
    )
    return df


def split_xy(key: str, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, binary label) with the pre-registered rule applied."""
    spec = SPECS[key]
    df = load_raw(key) if df is None else df
    label_col = df.columns[spec.label_col] if isinstance(spec.label_col, int) else spec.label_col
    y = spec.positive(df[label_col]).astype(int).rename("y")
    drop = [label_col, *spec.drop_cols]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    # Other raw targets (CTG's CLASS and NSP) must never stay in the features.
    return X.reset_index(drop=True), y.reset_index(drop=True)


def dedup(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop exact duplicate rows (features + label), keeping the first occurrence."""
    full = X.copy()
    full["__y__"] = y.values
    keep = ~full.duplicated(keep="first")
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def conflicting_groups(X: pd.DataFrame, y: pd.Series) -> int:
    """Count feature vectors that appear with more than one label."""
    tmp = X.copy()
    tmp["__y__"] = y.values
    nunique = tmp.groupby(list(X.columns), dropna=False)["__y__"].nunique()
    return int((nunique > 1).sum())
