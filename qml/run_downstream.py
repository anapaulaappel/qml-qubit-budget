#!/usr/bin/env python3
"""QSVM e kNN de fidelidade em n=128: q=teto D2 contra um q morto."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from bbs.fdase import correlation_fractal_dimension
from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel
from qml.data import display_name, load_qml_dataset
from qml.features import d2_qubit_ceiling, project_to_q
from qml.kernel_tasks import score_kernel
from qml.qubit_budget import _majority_label

DEFAULT_OUT = Path("d2-qubit-budget/docs/qubit-budget/downstream_n128.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/figures/qubit-budget/downstream_n128.png")
DEAD_Q = 8


def _ceiling(data, n_levels: int) -> tuple[float, int]:
    rng = np.random.default_rng(0)
    n_all = data.X.shape[0]
    n_d2 = min(n_all, 4_000)
    d2_idx = rng.choice(n_all, size=n_d2, replace=False)
    d2 = float(correlation_fractal_dimension(np.asarray(data.X[d2_idx], dtype=np.float64), n_levels=n_levels))
    return d2, d2_qubit_ceiling(d2)


def _score_q(data, n_kernel: int, q: int, seed: int) -> dict[str, float | int | bool]:
    rng_kernel = np.random.default_rng(seed + 17)
    n_all = data.X.shape[0]
    take = min(int(n_kernel), n_all)
    idx = np.sort(rng_kernel.choice(n_all, size=take, replace=False))
    X = np.asarray(data.X[idx], dtype=np.float64)
    y = np.asarray(data.y[idx])
    local = np.arange(take)
    try:
        train_idx, test_idx = train_test_split(
            local, test_size=0.3, random_state=seed, stratify=y
        )
    except ValueError:
        train_idx, test_idx = train_test_split(local, test_size=0.3, random_state=seed)
    Xq = project_to_q(X, q, method="pca", random_state=seed)
    kernel = fidelity_kernel(Xq, reps=1, entanglement="linear", bandwidth=1.0)
    scores = score_kernel(
        kernel,
        Xq,
        y,
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        minority_label=data.minority_label,
        majority_label=_majority_label(y),
        regression_target=np.zeros(take, dtype=np.float64),
        geometry_only=False,
    )
    return {
        "q": int(Xq.shape[1]),
        "n": take,
        "kernel_alive": bool(scores.kernel_alive),
        "knn_acc": float(scores.knn_acc),
        "qsvm_f1": float(scores.qsvm_f1),
        "mean_offdiag": float(scores.mean_offdiag),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["breast", "moons"])
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--dead-q", type=int, default=DEAD_Q)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário.")

    records: list[dict[str, object]] = []
    print(
        f"{'set':<8s} {'seed':>4s} {'role':<6s} {'q':>3s} {'alive':>5s} {'knn':>6s} {'f1':>6s}",
        flush=True,
    )
    for name in args.datasets:
        data = load_qml_dataset(name, random_state=0)
        d2, ceiling = _ceiling(data, n_levels=8)
        widths = (("ceiling", ceiling), ("dead", args.dead_q))
        for seed in range(args.seeds):
            for role, q in widths:
                scored = _score_q(data, args.n, q, seed)
                rec = {
                    "dataset": display_name(data.name),
                    "seed": seed,
                    "role": role,
                    "q": scored["q"],
                    "d2": f"{d2:.4f}",
                    "d2_ceiling": ceiling,
                    "kernel_alive": int(bool(scored["kernel_alive"])),
                    "knn_acc": f"{float(scored['knn_acc']):.4f}",
                    "qsvm_f1": f"{float(scored['qsvm_f1']):.4f}",
                    "mean_offdiag": f"{float(scored['mean_offdiag']):.4f}",
                    "n": scored["n"],
                }
                records.append(rec)
                print(
                    f"{name:<8s} {seed:4d} {role:<6s} {int(scored['q']):3d} "
                    f"{int(bool(scored['kernel_alive'])):5d} "
                    f"{float(scored['knn_acc']):6.3f} {float(scored['qsvm_f1']):6.3f}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    names = []
    for rec in records:
        if rec["dataset"] not in names:
            names.append(rec["dataset"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    metrics = (("knn_acc", "fidelity kNN"), ("qsvm_f1", "QSVM F1"))
    for ax, (key, label) in zip(axes, metrics):
        positions = []
        means = []
        stds = []
        tick_labels = []
        cursor = 0
        for name in names:
            for role, color in (("ceiling", "C0"), ("dead", "C3")):
                vals = [
                    float(r[key])
                    for r in records
                    if r["dataset"] == name and r["role"] == role
                ]
                positions.append(cursor)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                q_show = next(int(r["q"]) for r in records if r["dataset"] == name and r["role"] == role)
                tick_labels.append(f"{name}\nq={q_show}")
                ax.bar(cursor, means[-1], yerr=stds[-1], color=color, capsize=3, width=0.8)
                cursor += 1
            cursor += 0.6
        ax.set_xticks(positions, tick_labels, fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].bar(0, 0, color="C0", label=r"$\lceil D_2\rceil$")
    axes[0].bar(0, 0, color="C3", label=r"dead $q=8$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    plt.close(fig)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.fig}", flush=True)


if __name__ == "__main__":
    main()
