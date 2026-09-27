#!/usr/bin/env python3
"""Cubo k-D rodado: D2, FD-ASE nas colunas originais, joelho PCA do ZZ."""

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
from qml.data import load_intrinsic_k_oblique
from qml.qubit_budget import SweepRow, last_alive_q, run_dataset_sweep

DEFAULT_OUT = Path("d2-qubit-budget/docs/qubit-budget/oblique_id.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/figures/qubit-budget/oblique_id.png")


def _row_at(rows: list[SweepRow], view: str, q: int) -> SweepRow | None:
    hits = [r for r in rows if r.view == view and r.family == "fidelity" and r.q == q]
    return hits[0] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
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
        f"{'k':>3s} {'D2':>6s} {'ceil':>5s} {'PCA95':>5s} "
        f"{'FD':>4s} {'FDalive':>7s} {'knee':>5s} {'PCAat':>6s}",
        flush=True,
    )
    for k in range(int(args.k_min), int(args.k_max) + 1):
        data = load_intrinsic_k_oblique(
            k, n=args.n_data, n_features=args.e, random_state=0
        )
        rows = run_dataset_sweep(
            data,
            n_kernel=args.n,
            q_max=args.q_max,
            views=("pca",),
            families=("fidelity",),
            random_state=0,
            geometry_only=True,
            include_fdase=True,
        )
        head = rows[0]
        knee = last_alive_q(rows, "pca", "fidelity")
        fdase = [r for r in rows if r.view == "fdase" and r.family == "fidelity"]
        fdase_row = fdase[0] if fdase else None
        ceiling = int(head.d2_ceiling)
        pca_at = _row_at(rows, "pca", ceiling)
        rec: dict[str, object] = {
            "k": k,
            "d2": f"{head.d2:.4f}",
            "d2_ceiling": ceiling,
            "pca95": head.pca95,
            "fdase_q": "" if fdase_row is None else fdase_row.q,
            "fdase_alive": "" if fdase_row is None else int(fdase_row.kernel_alive),
            "pca_knee": "" if knee is None else knee,
            "pca_alive_at_ceiling": "" if pca_at is None else int(pca_at.kernel_alive),
            "e": args.e,
            "n_kernel": args.n,
            "n_data": args.n_data,
        }
        records.append(rec)
        print(
            f"{k:3d} {head.d2:6.2f} {ceiling:5d} {head.pca95:5d} "
            f"{str(rec['fdase_q']):>4s} {str(rec['fdase_alive']):>7s} "
            f"{str(rec['pca_knee']):>5s} {str(rec['pca_alive_at_ceiling']):>6s}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    ks = [int(r["k"]) for r in records]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(ks, ks, "k:", label=r"true $k$")
    ax.plot(ks, [int(r["d2_ceiling"]) for r in records], "o-", label=r"$\lceil D_2\rceil$")
    ax.plot(
        ks,
        [int(r["pca95"]) for r in records],
        "^--",
        color="0.45",
        label="PCA-95%",
    )
    fdase_q = [int(r["fdase_q"]) if r["fdase_q"] != "" else 0 for r in records]
    knees = [int(r["pca_knee"]) if r["pca_knee"] != "" else 0 for r in records]
    ax.plot(ks, fdase_q, "D-", label="FD-ASE width")
    ax.plot(ks, knees, "s-", label="ZZ knee (PCA view)")
    ax.set_xlabel(r"true latent width $k$")
    ax.set_ylabel(r"$q$")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf"oblique cube, $E={args.e}$, $n={args.n}$, one-layer $ZZ$")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    plt.close(fig)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.fig}", flush=True)


if __name__ == "__main__":
    main()
