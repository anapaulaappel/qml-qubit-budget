#!/usr/bin/env python3
"""Abla\u00e7\u00e3o de feature maps no mesmo kernel de fidelidade (PCA, n=32).

Compara Z (sem emaranhamento), ZZ com uma e duas camadas, e um mapa IQP
diagonal linear, em breast e moons, q=2..8. O crit\u00e9rio alive \u00e9 o mesmo do
teto D2. N\u00e3o empilha atributos: um coordenada PCA por qubit.
"""

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

from bbs.fdase import correlation_fractal_dimension
from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel, kernel_concentration, near_far_fidelity
from qml.data import load_qml_dataset
from qml.features import d2_qubit_ceiling, project_to_q
from qml.id_estimators import pca_variance_qubits
from qml.kernel_tasks import kernel_is_alive, near_far_ratio

DEFAULT_OUT = Path("d2-qubit-budget/docs/encodings/encoding_ablation.csv")

MAPS: tuple[tuple[str, int, str], ...] = (
    ("zz", 1, "ZZ, one layer"),
    ("zz", 2, "ZZ, two layers"),
    ("z", 1, "Z (product states)"),
    ("iqp", 1, "IQP, linear CP"),
)

MAP_STYLE: dict[tuple[str, int], tuple[str, str]] = {
    ("zz", 1): ("#1d4e89", "o"),
    ("zz", 2): ("#c47b2b", "s"),
    ("z", 1): ("#2a9d8f", "^"),
    ("iqp", 1): ("#9b2226", "D"),
}


def _subsample(X: np.ndarray, y: np.ndarray, n: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state + 17)
    take = rng.choice(X.shape[0], size=min(int(n), X.shape[0]), replace=False)
    take.sort()
    return np.asarray(X[take], dtype=np.float64), np.asarray(y[take])


def _label(encoding: str, reps: int) -> str:
    for kind, nrep, name in MAPS:
        if kind == encoding and nrep == reps:
            return name
    return f"{encoding} reps={reps}"


def last_alive_q(rows: list[dict[str, object]], dataset: str, encoding: str, reps: int) -> int | None:
    alive = [
        int(r["q"])
        for r in rows
        if str(r["dataset"]) == dataset
        and str(r["encoding"]) == encoding
        and int(r["reps"]) == reps
        and bool(r["kernel_alive"])
    ]
    return max(alive) if alive else None


def _score_row(
    Xq: np.ndarray,
    *,
    encoding: str,
    reps: int,
    q: int,
    dataset: str,
    d2: float,
    ceiling: int,
    pca95: int,
    note: str,
) -> dict[str, object]:
    kernel = fidelity_kernel(
        Xq,
        reps=reps,
        entanglement="linear",
        encoding=encoding,
        n_qubits=q,
    )
    conc = kernel_concentration(kernel)
    near, far = near_far_fidelity(Xq, kernel)
    ratio = near_far_ratio(near, far)
    alive = kernel_is_alive(near, far, conc["mean_offdiag"])
    return {
        "dataset": dataset,
        "encoding": encoding,
        "reps": reps,
        "q": q,
        "n_features": int(Xq.shape[1]),
        "d2": d2,
        "d2_ceiling": ceiling,
        "pca95": pca95,
        "mean_offdiag": conc["mean_offdiag"],
        "std_offdiag": conc["std_offdiag"],
        "fid_near": near,
        "fid_far": far,
        "near_far_ratio": ratio if np.isfinite(ratio) else "",
        "kernel_alive": alive,
        "note": note,
        "n": int(Xq.shape[0]),
    }


def _plot(rows: list[dict[str, object]], out: Path) -> None:
    names = sorted({str(r["dataset"]) for r in rows})
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * max(len(names), 1), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        chunk = [r for r in rows if r["dataset"] == name]
        ceiling = int(chunk[0]["d2_ceiling"])
        pca95 = int(chunk[0]["pca95"])
        for encoding, reps, title in MAPS:
            series = [r for r in chunk if str(r["encoding"]) == encoding and int(r["reps"]) == reps]
            if not series:
                continue
            series.sort(key=lambda r: int(r["q"]))
            color, marker = MAP_STYLE[(encoding, reps)]
            ax.plot(
                [int(r["q"]) for r in series],
                [float(r["mean_offdiag"]) for r in series],
                color=color,
                marker=marker,
                markersize=6,
                label=title,
            )
            dead = [r for r in series if not bool(r["kernel_alive"])]
            if dead:
                ax.scatter(
                    [int(r["q"]) for r in dead],
                    [float(r["mean_offdiag"]) for r in dead],
                    facecolors="white",
                    edgecolors=color,
                    marker=marker,
                    s=42,
                    zorder=4,
                )
        ax.axvline(ceiling, color="k", linestyle=":", linewidth=1.1)
        ax.axvline(pca95, color="0.55", linestyle="--", linewidth=0.9)
        ax.axhline(0.03, color="k", linestyle=":", linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel(r"$q$ (PCA coordinates / qubits)")
        ax.set_ylabel(r"mean off-diagonal $K$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.6, 8.4)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Feature-map ablation (PCA view). Open markers: not alive.")
    fig.tight_layout()
    fig_path = out.with_name("encoding_ablation.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _rows_from_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    rows: list[dict[str, object]] = []
    for rec in records:
        rec["q"] = int(rec["q"])
        rec["reps"] = int(rec["reps"])
        rec["n_features"] = int(rec["n_features"])
        rec["d2"] = float(rec["d2"])
        rec["d2_ceiling"] = int(rec["d2_ceiling"])
        rec["pca95"] = int(rec["pca95"])
        rec["mean_offdiag"] = float(rec["mean_offdiag"])
        rec["std_offdiag"] = float(rec["std_offdiag"])
        rec["fid_near"] = float(rec["fid_near"])
        rec["fid_far"] = float(rec["fid_far"])
        rec["n"] = int(rec["n"])
        rec["kernel_alive"] = str(rec["kernel_alive"]).strip().lower() in {"true", "1", "yes"}
        rows.append(rec)
    return rows


def _print_summary(rows: list[dict[str, object]]) -> None:
    datasets = sorted({str(r["dataset"]) for r in rows})
    print(f"{'dataset':16s} {'map':22s} {'\u2308D2\u2309':>5s} {'PCA95':>5s} {'alive':>6s}", flush=True)
    for name in datasets:
        chunk = [r for r in rows if r["dataset"] == name]
        ceiling = int(chunk[0]["d2_ceiling"])
        pca95 = int(chunk[0]["pca95"])
        for encoding, reps, title in MAPS:
            last = last_alive_q(rows, name, encoding, reps)
            print(
                f"{name:16s} {title:22s} {ceiling:5d} {pca95:5d} {str(last):>6s}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["breast", "moons"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="S\u00f3 regenera a figura a partir de um CSV j\u00e1 gravado.",
    )
    args = parser.parse_args()
    if args.from_csv is not None:
        rows = _rows_from_csv(args.from_csv)
        if not rows:
            raise SystemExit(f"CSV vazio: {args.from_csv}")
        _plot(rows, args.from_csv)
        _print_summary(rows)
        print(f"Figura: {args.from_csv.with_name('encoding_ablation.png')}", flush=True)
        return
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit ausente. pip install -e '.[qml]'")

    rows: list[dict[str, object]] = []
    for name in args.datasets:
        try:
            data = load_qml_dataset(name, random_state=0)
        except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"\u21b7 {name}: {exc}", flush=True)
            continue
        n_d2 = min(data.X.shape[0], 4_000)
        rng = np.random.default_rng(0)
        d2_idx = rng.choice(data.X.shape[0], size=n_d2, replace=False)
        d2 = float(correlation_fractal_dimension(data.X[d2_idx], n_levels=8))
        ceiling = d2_qubit_ceiling(d2)
        pca95 = int(pca_variance_qubits(data.X))
        Xk, _yk = _subsample(data.X, data.y, args.n, 0)
        q_hi = max(2, min(int(args.q_max), Xk.shape[1], Xk.shape[0] - 1))
        print(
            f"\u2192 {data.name} E={data.X.shape[1]} D2={d2:.2f} \u2308D2\u2309={ceiling} "
            f"PCA95={pca95} q=2..{q_hi}",
            flush=True,
        )
        for encoding, reps, note in MAPS:
            for q in range(2, q_hi + 1):
                Xp = project_to_q(Xk, q, method="pca", random_state=0)
                row = _score_row(
                    Xp,
                    encoding=encoding,
                    reps=reps,
                    q=q,
                    dataset=data.name,
                    d2=d2,
                    ceiling=ceiling,
                    pca95=pca95,
                    note=note,
                )
                rows.append(row)
                print(
                    f"  {note:22s} q={q} meanK={float(row['mean_offdiag']):.3f} "
                    f"near={float(row['fid_near']):.3f} far={float(row['fid_far']):.3f} "
                    f"alive={row['kernel_alive']}",
                    flush=True,
                )

    if not rows:
        raise SystemExit("Nenhuma linha gerada.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.out)
    _print_summary(rows)
    print(f"CSV: {args.out}", flush=True)
    print(f"Figura: {args.out.with_name('encoding_ablation.png')}", flush=True)


if __name__ == "__main__":
    main()
