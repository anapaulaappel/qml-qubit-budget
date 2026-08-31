#!/usr/bin/env python3
"""Empilhar atributos por qubit: ZZ 1:1 vs dense-angle vs re-uploading.

A tese: ⌈D2⌉ orça o Hilbert space (quantos qubits), não a regra «um atributo =
um qubit». Empilhar PCA-lixo em 3 qubits não recupera o kernel que o FD-ASE
obtém escolhendo 3 colunas originais.
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
from qml.circuits import (
    QISKIT_AVAILABLE,
    encode_circuit_depth,
    fidelity_kernel,
    kernel_concentration,
    near_far_fidelity,
    packed_layers,
)
from qml.data import load_qml_dataset
from qml.features import d2_qubit_ceiling, project_to_q
from qml.id_estimators import pca_variance_qubits
from qml.kernel_tasks import kernel_is_alive, near_far_ratio

DEFAULT_OUT = Path("d2-qubit-budget/docs/encodings/packed_encoding.csv")


def _subsample(X: np.ndarray, y: np.ndarray, n: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state + 17)
    take = rng.choice(X.shape[0], size=min(int(n), X.shape[0]), replace=False)
    take.sort()
    return np.asarray(X[take], dtype=np.float64), np.asarray(y[take])


def _cases(e: int, ceiling: int, pca95: int) -> list[dict[str, object]]:
    """Configurações para o artigo: 1:1 no teto, empilhar além do teto no mesmo q."""
    q_ceil = max(2, min(int(ceiling), e))
    q_wide = max(q_ceil, min(e, max(q_ceil + 3, 7), 8))
    pack_mid = min(e, 2 * q_ceil)
    pack_wide = min(e, max(pca95, 4 * q_ceil, 12), 16)
    return [
        {"encoding": "zz", "q": q_ceil, "n_features": q_ceil, "note": "1:1 at D2 ceiling"},
        {"encoding": "zz", "q": q_wide, "n_features": q_wide, "note": "1:1 beyond ceiling"},
        {
            "encoding": "dense_angle",
            "q": q_ceil,
            "n_features": q_ceil,
            "note": "dense, same features as ceiling ZZ",
        },
        {
            "encoding": "dense_angle",
            "q": q_ceil,
            "n_features": pack_mid,
            "note": "2 features/qubit at ceiling q",
        },
        {
            "encoding": "reuploading",
            "q": q_ceil,
            "n_features": pack_mid,
            "note": "re-upload extra PCs onto ceiling q",
        },
        {
            "encoding": "reuploading",
            "q": q_ceil,
            "n_features": pack_wide,
            "note": "re-upload toward PCA-95% onto ceiling q",
        },
    ]


def _score_row(
    Xq: np.ndarray,
    *,
    encoding: str,
    q: int,
    note: str,
    dataset: str,
    d2: float,
    ceiling: int,
    pca95: int,
) -> dict[str, object]:
    n_features = int(Xq.shape[1])
    layers = packed_layers(n_features, q, encoding, reps=1)
    depth = encode_circuit_depth(q, n_features, encoding=encoding, reps=1, entanglement="linear")
    kernel = fidelity_kernel(Xq, reps=1, entanglement="linear", encoding=encoding, n_qubits=q)
    conc = kernel_concentration(kernel)
    near, far = near_far_fidelity(Xq, kernel)
    ratio = near_far_ratio(near, far)
    alive = kernel_is_alive(near, far, conc["mean_offdiag"])
    return {
        "dataset": dataset,
        "encoding": encoding,
        "q": q,
        "n_features": n_features,
        "layers": layers,
        "depth": depth,
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
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        chunk = [r for r in rows if r["dataset"] == name]
        labels = [f"{r['encoding']}\nq={r['q']} E={r['n_features']}" for r in chunk]
        vals = [float(r["mean_offdiag"]) for r in chunk]
        colors = ["#2a9d8f" if r["kernel_alive"] else "#9aa0a6" for r in chunk]
        ax.bar(range(len(chunk)), vals, color=colors)
        ax.set_xticks(range(len(chunk)), labels, fontsize=7)
        ax.set_ylabel("mean off-diagonal $K$")
        ax.set_title(name)
        ax.axhline(0.03, color="k", linestyle=":", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Packed encodings: alive bars in teal (simulator)")
    fig.tight_layout()
    fig_path = out.with_name("packed_encoding.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["breast", "moons", "pendigits"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Só regenera a figura a partir de um CSV já gravado.",
    )
    args = parser.parse_args()
    if args.from_csv is not None:
        with args.from_csv.open(newline="") as fh:
            packed_rows = list(csv.DictReader(fh))
        if not packed_rows:
            raise SystemExit(f"CSV vazio: {args.from_csv}")
        for rec in packed_rows:
            rec["q"] = int(rec["q"])
            rec["n_features"] = int(rec["n_features"])
            rec["mean_offdiag"] = float(rec["mean_offdiag"])
            rec["kernel_alive"] = str(rec["kernel_alive"]).strip().lower() in {"true", "1", "yes"}
        _plot(packed_rows, args.from_csv)
        print(f"Figura: {args.from_csv.with_name('packed_encoding.png')}", flush=True)
        return
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit ausente. pip install -e '.[qml]'")

    rows: list[dict[str, object]] = []
    for name in args.datasets:
        try:
            data = load_qml_dataset(name, random_state=0)
        except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"↷ {name}: {exc}", flush=True)
            continue
        n_d2 = min(data.X.shape[0], 4_000)
        rng = np.random.default_rng(0)
        d2_idx = rng.choice(data.X.shape[0], size=n_d2, replace=False)
        d2 = float(correlation_fractal_dimension(data.X[d2_idx], n_levels=8))
        ceiling = d2_qubit_ceiling(d2)
        pca95 = int(pca_variance_qubits(data.X))
        Xk, _yk = _subsample(data.X, data.y, args.n, 0)
        print(
            f"→ {data.name} E={data.X.shape[1]} D2={d2:.2f} ⌈D2⌉={ceiling} PCA95={pca95}",
            flush=True,
        )
        for spec in _cases(data.X.shape[1], ceiling, pca95):
            encoding = str(spec["encoding"])
            q = int(spec["q"])
            n_features = int(spec["n_features"])
            Xp = project_to_q(Xk, n_features, method="pca", random_state=0)
            row = _score_row(
                Xp,
                encoding=encoding,
                q=q,
                note=str(spec["note"]),
                dataset=data.name,
                d2=d2,
                ceiling=ceiling,
                pca95=pca95,
            )
            rows.append(row)
            print(
                f"  {encoding:12s} q={q} feats={n_features} layers={row['layers']} "
                f"meanK={float(row['mean_offdiag']):.3f} alive={row['kernel_alive']}",
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
    print(f"CSV: {args.out}", flush=True)


if __name__ == "__main__":
    main()
