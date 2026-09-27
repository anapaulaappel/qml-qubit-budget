#!/usr/bin/env python3
"""Rank efetivo de K contra o joelho alive, no breast e nos moons."""

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
from qml.data import load_qml_dataset
from qml.qubit_budget import SweepRow, last_alive_q, run_dataset_sweep

DEFAULT_OUT = Path("d2-qubit-budget/docs/qubit-budget/effective_rank.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/figures/qubit-budget/effective_rank.png")


def rank_knee(rows: list[SweepRow]) -> int | None:
    """Último q antes do maior salto do rank efetivo. Sem pisos."""
    ordered = sorted(
        (r for r in rows if r.view == "pca" and r.family == "fidelity"),
        key=lambda r: r.q,
    )
    if len(ordered) < 2:
        return None
    best_q = ordered[0].q
    best_delta = float("-inf")
    for previous, current in zip(ordered, ordered[1:]):
        delta = current.effective_rank - previous.effective_rank
        if delta > best_delta:
            best_delta = delta
            best_q = previous.q
    return best_q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["breast", "moons"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário.")

    records: list[dict[str, object]] = []
    curves: dict[str, list[SweepRow]] = {}
    print(
        f"{'set':<8s} {'q':>3s} {'alive':>5s} {'erank':>7s} {'stdK':>7s} {'knee':>5s} {'rknee':>5s}",
        flush=True,
    )
    for name in args.datasets:
        data = load_qml_dataset(name, random_state=0)
        rows = run_dataset_sweep(
            data,
            n_kernel=args.n,
            q_max=args.q_max,
            views=("pca",),
            families=("fidelity",),
            random_state=0,
            geometry_only=True,
            include_fdase=False,
        )
        curves[name] = rows
        alive = last_alive_q(rows, "pca", "fidelity")
        rknee = rank_knee(rows)
        for row in sorted(rows, key=lambda r: r.q):
            rec = {
                "dataset": name,
                "q": row.q,
                "d2": f"{row.d2:.4f}",
                "d2_ceiling": row.d2_ceiling,
                "kernel_alive": int(row.kernel_alive),
                "effective_rank": f"{row.effective_rank:.4f}",
                "std_offdiag": f"{row.std_offdiag:.4f}",
                "mean_offdiag": f"{row.mean_offdiag:.4f}",
                "alive_knee": "" if alive is None else alive,
                "rank_knee": "" if rknee is None else rknee,
                "n": row.n,
            }
            records.append(rec)
            print(
                f"{name:<8s} {row.q:3d} {int(row.kernel_alive):5d} "
                f"{row.effective_rank:7.2f} {row.std_offdiag:7.3f} "
                f"{str(rec['alive_knee']):>5s} {str(rec['rank_knee']):>5s}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, name in zip(axes, args.datasets):
        rows = sorted(curves[name], key=lambda r: r.q)
        qs = [r.q for r in rows]
        ranks = [r.effective_rank for r in rows]
        ax.plot(qs, ranks, "o-", color="C0", label="effective rank")
        knee = last_alive_q(rows, "pca", "fidelity")
        if knee is not None:
            ax.axvline(knee, color="0.35", ls="--", label=f"alive knee q={knee}")
        ax.set_title(name)
        ax.set_xlabel(r"$q$")
        ax.set_xticks(qs)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("effective rank of $K$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    plt.close(fig)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.fig}", flush=True)


if __name__ == "__main__":
    main()
