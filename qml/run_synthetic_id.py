#!/usr/bin/env python3
"""Sweep sintético: k conhecido (1..8), E=20 fixo, joelho do kernel ZZ vs D2."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE
from qml.data import load_intrinsic_k
from qml.qubit_budget import last_alive_q, run_dataset_sweep

DEFAULT_OUT = Path("d2-qubit-budget/docs/qubit-budget/synthetic_id.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/figures/qubit-budget/synthetic_id.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--n-data", type=int, default=400)
    parser.add_argument("--e", type=int, default=20)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário.")

    records: list[dict[str, object]] = []
    print(
        f"{'k':>3s} {'D2':>6s} {'⌈D2⌉':>5s} {'PCA95':>5s} {'knee':>5s}",
        flush=True,
    )
    for k in range(int(args.k_min), int(args.k_max) + 1):
        data = load_intrinsic_k(k, n=args.n_data, n_features=args.e, random_state=0)
        rows = run_dataset_sweep(
            data,
            n_kernel=args.n,
            q_max=args.q_max,
            views=("prefix",),
            families=("fidelity",),
            random_state=0,
            geometry_only=True,
            include_fdase=False,
        )
        knee = last_alive_q(rows, "prefix", "fidelity")
        rec = {
            "k": k,
            "d2": rows[0].d2,
            "d2_ceiling": rows[0].d2_ceiling,
            "pca95": rows[0].pca95,
            "knee": "" if knee is None else knee,
            "e": args.e,
            "n_kernel": args.n,
            "n_data": args.n_data,
        }
        records.append(rec)
        print(
            f"{k:3d} {rows[0].d2:6.2f} {rows[0].d2_ceiling:5d} {rows[0].pca95:5d} "
            f"{str(knee):>5s}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    ks = [int(r["k"]) for r in records]
    d2c = [int(r["d2_ceiling"]) for r in records]
    pca = [int(r["pca95"]) for r in records]
    knees = [int(r["knee"]) if r["knee"] != "" else 0 for r in records]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(ks, ks, "k:", label=r"true $k$")
    ax.plot(ks, d2c, "o-", label=r"$\lceil D_2\rceil$")
    ax.plot(ks, knees, "s-", label="ZZ knee (prefix view)")
    ax.plot(ks, pca, "^--", color="0.45", label="PCA-95%")
    ax.set_xlabel(r"true intrinsic width $k$")
    ax.set_ylabel(r"$q$")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf"$E={args.e}$, $n={args.n}$ kernel, prefix view, one-layer $ZZ$")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    plt.close(fig)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.fig}", flush=True)


if __name__ == "__main__":
    main()
