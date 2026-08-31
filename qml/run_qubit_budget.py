#!/usr/bin/env python3
"""Benchmark do teto de qubits: D2 vs joelho do kernel (fidelidade e RBF)."""

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
from qml.data import DATASET_NAMES, load_qml_dataset
from qml.features import SWEEP_VIEWS
from qml.kernel_tasks import KERNEL_FAMILIES
from qml.qubit_budget import SweepRow, last_alive_q, run_dataset_sweep

DEFAULT_DATASETS = (
    "intrinsic2",
    "moons",
    "iris",
    "breast",
    "diabetes",
    "wine",
    "digits",
    "pendigits",
    "onebig",
)


def _write_csv(path: Path, rows: list[SweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [r.__dict__ for r in rows]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _rows_from_csv(path: Path) -> list[SweepRow]:
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    rows: list[SweepRow] = []
    for rec in records:
        rows.append(
            SweepRow(
                dataset=rec["dataset"],
                view=rec["view"],
                family=rec["family"],
                q=int(rec["q"]),
                d2=float(rec["d2"]),
                d2_ceiling=int(rec["d2_ceiling"]),
                e_original=int(rec["e_original"]),
                n=int(rec["n"]),
                mean_offdiag=float(rec["mean_offdiag"]),
                std_offdiag=float(rec["std_offdiag"]),
                fid_near=float(rec["fid_near"]),
                fid_far=float(rec["fid_far"]),
                near_far_ratio=float(rec["near_far_ratio"]),
                kernel_alive=_parse_bool(rec["kernel_alive"]),
                alignment=float(rec["alignment"]),
                knn_acc=float(rec["knn_acc"]),
                cluster_ari=float(rec["cluster_ari"]),
                oneclass_auc=float(rec["oneclass_auc"]),
                krr_r2=float(rec["krr_r2"]),
                qsvm_f1=float(rec["qsvm_f1"]),
                last_alive=_parse_bool(rec["last_alive"]),
                twonn=float(rec["twonn"]),
                twonn_ceiling=int(rec["twonn_ceiling"]),
                pca95=int(rec["pca95"]),
            )
        )
    return rows


def _plot_dataset(rows: list[SweepRow], out: Path) -> None:
    names = sorted({r.dataset for r in rows})
    metrics = (
        ("near_far_ratio", "near/far ratio"),
        ("mean_offdiag", "mean off-diagonal"),
        ("knn_acc", "fidelity kNN (acc)"),
        ("krr_r2", "KRR $R^2$"),
        ("alignment", "kernel–label alignment"),
        ("qsvm_f1", "precomputed SVM F1"),
    )
    for name in names:
        subset = [r for r in rows if r.dataset == name]
        ceiling = subset[0].d2_ceiling
        d2 = subset[0].d2
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True)
        for ax, (field, title) in zip(axes.ravel(), metrics, strict=True):
            for family in KERNEL_FAMILIES:
                for view in ("pca", "random", "prefix"):
                    series = [r for r in subset if r.view == view and r.family == family]
                    if not series:
                        continue
                    series.sort(key=lambda r: r.q)
                    style = "-" if family == "fidelity" else "--"
                    ax.plot(
                        [r.q for r in series],
                        [getattr(r, field) for r in series],
                        style,
                        marker="o",
                        markersize=3.5,
                        label=f"{view}/{family}",
                    )
            fdase_pts = [r for r in subset if r.view == "fdase"]
            for r in fdase_pts:
                ax.scatter(
                    [r.q],
                    [getattr(r, field)],
                    marker="*",
                    s=90,
                    zorder=5,
                    label=f"fdase/{r.family}" if r is fdase_pts[0] else None,
                )
            ax.axvline(ceiling, color="k", linestyle=":", linewidth=1.2)
            ax.axvline(subset[0].pca95, color="0.5", linestyle="--", linewidth=0.9)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=7, loc="best")
        axes[1, 0].set_xlabel(r"$q$ (qubits / features)")
        axes[1, 1].set_xlabel(r"$q$ (qubits / features)")
        axes[1, 2].set_xlabel(r"$q$ (qubits / features)")
        fig.suptitle(
            rf"{name}: $D_2$={d2:.2f}, ceiling $\lceil D_2\rceil$={ceiling} (dotted), "
            f"PCA-95%={subset[0].pca95} (dashed)"
        )
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_name(f"qubit_budget_{name}.png"), dpi=140)
        plt.close(fig)


def _plot_ceiling_scatter(rows: list[SweepRow], out: Path) -> None:
    """last_alive (PCA/fidelity) against rival ceilings. PCA panel has its own x-scale."""
    by_ds: dict[str, list[SweepRow]] = defaultdict(list)
    for r in rows:
        by_ds[r.dataset].append(r)
    names: list[str] = []
    d2_c: list[int] = []
    twonn_c: list[int] = []
    pca95: list[int] = []
    alive: list[float] = []
    for name, chunk in sorted(by_ds.items()):
        last = last_alive_q(chunk, "pca", "fidelity")
        if last is None:
            last = last_alive_q(chunk, "pca", "rbf")
        if last is None:
            continue
        names.append(name)
        d2_c.append(chunk[0].d2_ceiling)
        twonn_c.append(chunk[0].twonn_ceiling)
        pca95.append(chunk[0].pca95)
        alive.append(float(last))
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), sharey=True)
    y_lo = 1
    y_hi = max(int(max(alive)), max(d2_c), 8)
    compact = (
        (d2_c, r"$\lceil D_2\rceil$", axes[0], True),
        (twonn_c, r"$\lceil\mathrm{TwoNN}\rceil$", axes[1], False),
        (pca95, "PCA 95%", axes[2], False),
    )
    for xs, xlabel, ax, equal in compact:
        x_hi = max(max(xs), y_hi) if equal else max(max(xs), 8)
        ax.plot([y_lo, x_hi], [y_lo, x_hi], "k:", linewidth=1.0)
        ax.scatter(xs, alive, s=42, zorder=3)
        for x, y, label in zip(xs, alive, names, strict=True):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_xlim(y_lo - 0.4, x_hi + 0.4)
        ax.set_ylim(y_lo - 0.4, y_hi + 0.4)
        if equal:
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("last alive $q$ (PCA view)")
    fig.suptitle("Proposed ceiling vs observed kernel knee")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _print_summary(rows: list[SweepRow]) -> None:
    by_ds: dict[str, list[SweepRow]] = defaultdict(list)
    for r in rows:
        by_ds[r.dataset].append(r)
    print(
        f"{'dataset':16s} {'E':>3s} {'D2':>6s} {'⌈D2⌉':>5s} {'TwoNN':>6s} "
        f"{'⌈TN⌉':>5s} {'PCA95':>5s} {'alive':>6s} {'ΔD2':>5s} {'Δ95':>5s}",
        flush=True,
    )
    for name, chunk in sorted(by_ds.items()):
        last = last_alive_q(chunk, "pca", "fidelity")
        if last is None:
            last = last_alive_q(chunk, "pca", "rbf")
        d2_c = chunk[0].d2_ceiling
        pca95 = chunk[0].pca95
        delta_d2 = "" if last is None else str(int(last) - d2_c)
        delta_95 = "" if last is None else str(int(last) - pca95)
        print(
            f"{name:16s} {chunk[0].e_original:3d} {chunk[0].d2:6.2f} {d2_c:5d} "
            f"{chunk[0].twonn:6.2f} {chunk[0].twonn_ceiling:5d} {pca95:5d} "
            f"{str(last):>6s} {delta_d2:>5s} {delta_95:>5s}",
            flush=True,
        )
    print(
        f"{'dataset':16s} {'view':8s} {'family':9s} {'D2':>6s} {'⌈D2⌉':>5s} "
        f"{'last_alive':>10s} {'Δ':>4s}",
        flush=True,
    )
    grouped: dict[tuple[str, str, str], list[SweepRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.view, r.family)].append(r)
    for key in sorted(grouped):
        chunk = grouped[key]
        ceiling = chunk[0].d2_ceiling
        d2 = chunk[0].d2
        alive = last_alive_q(chunk, key[1], key[2])
        delta = "" if alive is None else str(int(alive) - ceiling)
        print(
            f"{key[0]:16s} {key[1]:8s} {key[2]:9s} {d2:6.2f} {ceiling:5d} "
            f"{str(alive):>10s} {delta:>4s}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DATASET_NAMES,
    )
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--views", nargs="+", default=list(SWEEP_VIEWS), choices=SWEEP_VIEWS)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(KERNEL_FAMILIES),
        choices=KERNEL_FAMILIES,
    )
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("d2-qubit-budget/docs/qubit-budget/qubit_budget.csv"),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("d2-qubit-budget/docs/figures/qubit-budget"),
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Só regenera figuras a partir de um CSV já gravado.",
    )
    args = parser.parse_args()
    if args.from_csv is not None:
        all_rows = _rows_from_csv(args.from_csv)
        if not all_rows:
            raise SystemExit(f"CSV vazio: {args.from_csv}")
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_dataset(all_rows, args.fig_dir / "qubit_budget.png")
        _plot_ceiling_scatter(all_rows, args.fig_dir / "qubit_budget_ceilings.png")
        _print_summary(all_rows)
        print(f"Figuras: {args.fig_dir}", flush=True)
        return
    if args.fast:
        args.datasets = ["intrinsic2", "breast"]
        args.n = 16
        args.q_max = 5
    families = tuple(args.families)
    if args.classical_only or not QISKIT_AVAILABLE:
        families = ("rbf",)
        print("Qiskit ausente ou --classical-only: só kernel RBF clássico.", flush=True)

    all_rows: list[SweepRow] = []
    for name in args.datasets:
        try:
            data = load_qml_dataset(name, random_state=0)
        except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"↷ {name} ignorado: {exc}", flush=True)
            continue
        print(f"→ {data.name} N={data.X.shape[0]} E={data.X.shape[1]}", flush=True)
        rows = run_dataset_sweep(
            data,
            n_kernel=args.n,
            q_max=args.q_max,
            views=tuple(args.views),
            families=families,
            random_state=0,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("Nenhuma linha gerada.")
    _write_csv(args.out, all_rows)
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_dataset(all_rows, fig_dir / "qubit_budget.png")
    _plot_ceiling_scatter(all_rows, fig_dir / "qubit_budget_ceilings.png")
    _print_summary(all_rows)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figuras: {fig_dir}/qubit_budget_<dataset>.png", flush=True)


if __name__ == "__main__":
    main()
