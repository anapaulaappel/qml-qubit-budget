"""Grava kernels de hardware em docs/ (não em results/, que está no gitignore)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

HARDWARE_DIR = Path(__file__).resolve().parents[1] / "docs" / "hardware"
SUMMARY_CSV = HARDWARE_DIR / "summary.csv"
SUMMARY_FIELDS = (
    "tag",
    "dataset",
    "backend",
    "q",
    "n",
    "shots",
    "n_circuits",
    "elapsed_s",
    "mae",
    "mean_offdiag_hw",
    "mean_offdiag_sv",
    "utc",
)


def hardware_dir() -> Path:
    HARDWARE_DIR.mkdir(parents=True, exist_ok=True)
    return HARDWARE_DIR


def mean_offdiag(kernel: NDArray[np.floating]) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    n = k.shape[0]
    if n < 2:
        return 0.0
    return float(k[~np.eye(n, dtype=bool)].mean())


def save_run(
    *,
    tag: str,
    dataset: str,
    backend: str,
    q: int,
    n: int,
    shots: int,
    elapsed_s: float,
    X: NDArray[np.floating],
    k_hw: NDArray[np.floating],
    k_sv: NDArray[np.floating],
    extra: dict[str, Any] | None = None,
) -> Path:
    out = hardware_dir()
    k_hw = np.asarray(k_hw, dtype=np.float64)
    k_sv = np.asarray(k_sv, dtype=np.float64)
    mae = float(np.mean(np.abs(k_hw - k_sv)))
    record = {
        "tag": tag,
        "dataset": dataset,
        "backend": backend,
        "q": int(q),
        "n": int(n),
        "shots": int(shots),
        "n_circuits": int(n * (n - 1) // 2),
        "elapsed_s": round(float(elapsed_s), 3),
        "mae": mae,
        "mean_offdiag_hw": mean_offdiag(k_hw),
        "mean_offdiag_sv": mean_offdiag(k_sv),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        record["extra"] = extra
    np.savez_compressed(
        out / f"{tag}.npz",
        X=np.asarray(X, dtype=np.float64),
        k_hw=k_hw,
        k_sv=k_sv,
    )
    (out / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _append_summary(record)
    return out / f"{tag}.json"


def _append_summary(record: dict[str, Any]) -> None:
    path = SUMMARY_CSV
    write_header = not path.exists()
    row = {k: record[k] for k in SUMMARY_FIELDS}
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
