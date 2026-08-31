#!/usr/bin/env python3
"""Robustez: limiares alive no CSV existente, seeds, e n=128 em wine/diabetes."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE
from qml.data import display_name, load_qml_dataset
from qml.kernel_tasks import kernel_is_alive
from qml.qubit_budget import last_alive_q, run_dataset_sweep
from qml.run_qubit_budget import _rows_from_csv

DEFAULT_CSV = Path("d2-qubit-budget/docs/qubit-budget/qubit_budget.csv")
DEFAULT_THRESH = Path("d2-qubit-budget/docs/qubit-budget/alive_thresholds.csv")
DEFAULT_SEEDS = Path("d2-qubit-budget/docs/qubit-budget/seeds.csv")
DEFAULT_N128 = Path("d2-qubit-budget/docs/qubit-budget/n128.csv")
DEFAULT_FIG = Path("d2-qubit-budget/docs/figures/qubit-budget/robustness_seeds.png")


def _threshold_sweep(src: Path, out: Path) -> None:
    rows = _rows_from_csv(src)
    pca = [r for r in rows if r.view == "pca" and r.family == "fidelity"]
    by_ds: dict[str, list] = defaultdict(list)
    for r in pca:
        by_ds[r.dataset].append(r)
    near_mult = (0.5, 1.0, 1.5)
    ratio_mult = (0.5, 1.0, 1.5)
    mean_mult = (0.5, 1.0, 1.5)
    records: list[dict[str, object]] = []
    print(
        f"{'dataset':12s} {'near':>5s} {'ratio':>5s} {'mean':>5s} {'knee':>5s} {'Δ':>4s}",
        flush=True,
    )
    for name, chunk in sorted(by_ds.items()):
        chunk.sort(key=lambda r: r.q)
        base = last_alive_q(chunk, "pca", "fidelity")
        for nm in near_mult:
            for rm in ratio_mult:
                for mm in mean_mult:
                    nf, rf, mf = 0.25 * nm, 2.0 * rm, 0.03 * mm
                    alive_qs = [
                        r.q
                        for r in chunk
                        if kernel_is_alive(
                            r.fid_near,
                            r.fid_far,
                            r.mean_offdiag,
                            near_floor=nf,
                            ratio_floor=rf,
                            mean_floor=mf,
                        )
                    ]
                    knee = max(alive_qs) if alive_qs else None
                    delta = "" if knee is None or base is None else str(int(knee) - int(base))
                    records.append(
                        {
                            "dataset": display_name(name),
                            "near_floor": nf,
                            "ratio_floor": rf,
                            "mean_floor": mf,
                            "knee": "" if knee is None else knee,
                            "base_knee": "" if base is None else base,
                            "delta": delta,
                        }
                    )
                    if nm == 1.0 and rm == 1.0 and mm == 1.0:
                        continue
                    print(
                        f"{display_name(name):12s} {nf:5.2f} {rf:5.1f} {mf:5.3f} "
                        f"{str(knee):>5s} {delta:>4s}",
                        flush=True,
                    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Limiares: {out}", flush=True)


def _seed_sweep(out: Path, fig: Path, n_seeds: int, n: int, q_max: int) -> None:
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário para o sweep de seeds.")
    records: list[dict[str, object]] = []
    print(f"{'dataset':12s} {'seed':>4s} {'knee':>5s} {'⌈D2⌉':>5s}", flush=True)
    for name in ("breast", "moons"):
        data = load_qml_dataset(name, random_state=0)
        knees: list[int] = []
        for seed in range(n_seeds):
            rows = run_dataset_sweep(
                data,
                n_kernel=n,
                q_max=q_max,
                views=("pca",),
                families=("fidelity",),
                random_state=seed,
                geometry_only=True,
                include_fdase=False,
            )
            knee = last_alive_q(rows, "pca", "fidelity")
            records.append(
                {
                    "dataset": display_name(data.name),
                    "seed": seed,
                    "d2": rows[0].d2,
                    "d2_ceiling": rows[0].d2_ceiling,
                    "knee": "" if knee is None else knee,
                    "n": n,
                }
            )
            if knee is not None:
                knees.append(int(knee))
            print(
                f"{display_name(data.name):12s} {seed:4d} {str(knee):>5s} "
                f"{rows[0].d2_ceiling:5d}",
                flush=True,
            )
        if knees:
            print(
                f"  {display_name(data.name)} knee median={np.median(knees):.1f} "
                f"range={min(knees)}–{max(knees)}",
                flush=True,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    fig.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({str(r["dataset"]) for r in records})
    fig_ax, ax = plt.subplots(figsize=(5.4, 3.6))
    data_plot = []
    labels = []
    for name in names:
        vals = [int(r["knee"]) for r in records if r["dataset"] == name and r["knee"] != ""]
        data_plot.append(vals)
        labels.append(name)
    ax.boxplot(data_plot)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("last alive $q$")
    ax.set_title(rf"PCA / one-layer $ZZ$, $n={n}$, {n_seeds} kernel samples")
    ax.grid(True, axis="y", alpha=0.3)
    fig_ax.tight_layout()
    fig_ax.savefig(fig, dpi=140)
    plt.close(fig_ax)
    print(f"Seeds: {out}", flush=True)
    print(f"Figura: {fig}", flush=True)


def _n128(out: Path) -> None:
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário para n=128.")
    records: list[dict[str, object]] = []
    print(f"{'dataset':12s} {'n':>4s} {'⌈D2⌉':>5s} {'knee':>5s} {'FD-ASE':>6s}", flush=True)
    for name in ("wine", "diabetes"):
        data = load_qml_dataset(name, random_state=0)
        for n in (32, 128):
            take = min(n, data.X.shape[0])
            rows = run_dataset_sweep(
                data,
                n_kernel=take,
                q_max=8,
                views=("pca",),
                families=("fidelity",),
                random_state=0,
                geometry_only=True,
                include_fdase=True,
            )
            knee = last_alive_q(rows, "pca", "fidelity")
            fdase = [r for r in rows if r.view == "fdase" and r.family == "fidelity"]
            rec = {
                "dataset": display_name(data.name),
                "n": take,
                "d2": rows[0].d2,
                "d2_ceiling": rows[0].d2_ceiling,
                "pca95": rows[0].pca95,
                "pca_knee": "" if knee is None else knee,
                "fdase_q": fdase[0].q if fdase else "",
                "fdase_alive": fdase[0].kernel_alive if fdase else "",
            }
            records.append(rec)
            fd = "n/a" if not fdase else ("yes" if fdase[0].kernel_alive else "no")
            print(
                f"{display_name(data.name):12s} {take:4d} {rows[0].d2_ceiling:5d} "
                f"{str(knee):>5s} {fd:>6s}",
                flush=True,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"n=128: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--thresholds-out", type=Path, default=DEFAULT_THRESH)
    parser.add_argument("--seeds-out", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--n128-out", type=Path, default=DEFAULT_N128)
    parser.add_argument("--seeds-fig", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument("--skip-seeds", action="store_true")
    parser.add_argument("--skip-n128", action="store_true")
    args = parser.parse_args()
    if not args.skip_thresholds:
        _threshold_sweep(args.from_csv, args.thresholds_out)
    if not args.skip_seeds:
        _seed_sweep(args.seeds_out, args.seeds_fig, args.n_seeds, args.n, args.q_max)
    if not args.skip_n128:
        _n128(args.n128_out)


if __name__ == "__main__":
    main()
