#!/usr/bin/env python3
"""Sweep de bandwidth c do mapa ZZ (ângulos em [0, cπ]), vista PCA, n=32."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel
from qml.data import display_name, load_qml_dataset
from qml.features import d2_qubit_ceiling, project_to_q
from qml.kernel_tasks import score_geometry
from bbs.fdase import correlation_fractal_dimension

DEFAULT_DATASETS = ("breast", "moons", "pendigits")
DEFAULT_C = (0.25, 0.5, 1.0, 2.0)
DEFAULT_OUT = Path("d2-qubit-budget/docs/encodings/bandwidth.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/encodings/bandwidth.png")


def _subsample(X: np.ndarray, n: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state + 17)
    take = np.sort(rng.choice(X.shape[0], size=min(int(n), X.shape[0]), replace=False))
    return np.asarray(X[take], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--c", nargs="+", type=float, default=list(DEFAULT_C))
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário.")

    records: list[dict[str, object]] = []
    print(
        f"{'dataset':12s} {'c':>5s} {'⌈D2⌉':>5s} {'knee':>5s}",
        flush=True,
    )
    for name in args.datasets:
        data = load_qml_dataset(name, random_state=0)
        n_d2 = min(data.X.shape[0], 4_000)
        rng = np.random.default_rng(0)
        d2_idx = rng.choice(data.X.shape[0], size=n_d2, replace=False)
        d2 = float(correlation_fractal_dimension(data.X[d2_idx], n_levels=8))
        ceiling = d2_qubit_ceiling(d2)
        Xk = _subsample(data.X, args.n, random_state=0)
        q_hi = max(2, min(int(args.q_max), Xk.shape[1], Xk.shape[0] - 1))
        for c in args.c:
            alive_at: list[int] = []
            for q in range(2, q_hi + 1):
                Xq = project_to_q(Xk, q, method="pca", random_state=0)
                kernel = fidelity_kernel(
                    Xq, reps=1, entanglement="linear", bandwidth=float(c)
                )
                mean_off, std_off, near, far, ratio, alive = score_geometry(kernel, Xq)
                records.append(
                    {
                        "dataset": display_name(data.name),
                        "bandwidth": float(c),
                        "q": q,
                        "d2": d2,
                        "d2_ceiling": ceiling,
                        "mean_offdiag": mean_off,
                        "std_offdiag": std_off,
                        "fid_near": near,
                        "fid_far": far,
                        "near_far_ratio": ratio,
                        "kernel_alive": alive,
                        "n": args.n,
                    }
                )
                if alive:
                    alive_at.append(q)
            knee = max(alive_at) if alive_at else None
            print(
                f"{display_name(data.name):12s} {float(c):5.2f} {ceiling:5d} {str(knee):>5s}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    names = sorted({str(r["dataset"]) for r in records})
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), sharey=True)
    if len(names) == 1:
        axes = [axes]
    colours = {0.25: "#2a9d8f", 0.5: "#1d4e89", 1.0: "#9b2226", 2.0: "#c47b2b"}
    for ax, name in zip(axes, names, strict=True):
        chunk = [r for r in records if r["dataset"] == name]
        ceiling = int(chunk[0]["d2_ceiling"])
        ax.axvline(ceiling, color="k", linestyle=":", linewidth=1.2)
        for c in args.c:
            series = sorted(
                [r for r in chunk if float(r["bandwidth"]) == float(c)],
                key=lambda r: int(r["q"]),
            )
            qs = [int(r["q"]) for r in series]
            ys = [float(r["mean_offdiag"]) for r in series]
            alive = [bool(r["kernel_alive"]) for r in series]
            ax.plot(
                qs,
                ys,
                "-o",
                color=colours.get(float(c), "0.3"),
                label=rf"$c={c:g}$",
                markersize=4,
            )
            dead = [(q, y) for q, y, a in zip(qs, ys, alive, strict=True) if not a]
            if dead:
                ax.scatter(
                    [p[0] for p in dead],
                    [p[1] for p in dead],
                    facecolors="none",
                    edgecolors=colours.get(float(c), "0.3"),
                    s=36,
                    zorder=4,
                )
        ax.axhline(0.03, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_title(str(name))
        ax.set_xlabel(r"$q$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean off-diagonal $K$")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Bandwidth $c$: angles in $[0, c\pi]$; open markers fail alive")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    plt.close(fig)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.fig}", flush=True)


if __name__ == "__main__":
    main()
