#!/usr/bin/env python3
"""Estudo de encoding: FD-ASE vs PCA vs eixos aleatórios vs E saturado.

Comparação quântico–quântico no mesmo kernel de fidelidade. Hipótese: Es
com q≈⌈D2⌉ mantém o kernel vivo face a PCA-95%, prefixo/full ou aleatório.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import (
    QISKIT_AVAILABLE,
    amplitude_prep_stats,
    feature_map_depth,
    fidelity_kernel,
    kernel_concentration,
    near_far_fidelity,
)
from qml.data import load_qml_dataset
from qml.features import FEATURE_VIEWS, select_feature_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("onebig", "breast", "credit"), default="breast")
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--kernel-n", type=int, default=32, help="Pontos no kernel (subamostra uniforme).")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--entanglement", choices=("linear", "full"), default="linear")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--views", nargs="+", choices=FEATURE_VIEWS, default=list(FEATURE_VIEWS))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("d2-qubit-budget/results/encode_compare.csv"),
    )
    args = parser.parse_args()
    use_quantum = QISKIT_AVAILABLE and not args.classical_only

    data = load_qml_dataset(args.dataset, random_state=0)
    n_all = data.X.shape[0]
    rng = np.random.default_rng(0)
    n_kernel = min(int(args.kernel_n), n_all)
    kernel_idx = np.sort(rng.choice(n_all, size=n_kernel, replace=False))
    print(
        f"{data.name}: N={n_all} E={data.X.shape[1]} kernel_n={n_kernel} qiskit={use_quantum}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    amp_depth: int | None = None
    if use_quantum:
        _, amp_depth = amplitude_prep_stats(data.X.shape[1])

    for view_name in args.views:
        view = select_feature_view(
            data.X, method=view_name, q_max=args.q_max, random_state=0
        )
        Xk = view.X[kernel_idx]
        depth: int | None = None
        conc: dict[str, float] = {
            "mean_offdiag": float("nan"),
            "std_offdiag": float("nan"),
            "collapse_to_half": float("nan"),
        }
        near = far = float("nan")
        if use_quantum and view.q >= 2:
            depth = feature_map_depth(view.q, reps=args.reps, entanglement=args.entanglement)
            kernel = fidelity_kernel(Xk, reps=args.reps, entanglement=args.entanglement)
            conc = kernel_concentration(kernel)
            near, far = near_far_fidelity(Xk, kernel)
        print(
            f"  {view.method:6s} q={view.q:2d} D2={view.d2:.3f} "
            f"q_amp={view.q_amp} depth={depth} "
            f"meanK={conc['mean_offdiag']:.3f} stdK={conc['std_offdiag']:.3f} "
            f"capped={view.capped}",
            flush=True,
        )
        rows.append(
            {
                "dataset": data.name,
                "view": view.method,
                "q": view.q,
                "d2": view.d2,
                "e_original": view.e_original,
                "q_amp": view.q_amp,
                "amp_prep_depth": amp_depth if amp_depth is not None else "",
                "feature_map_depth": depth if depth is not None else "",
                "mean_offdiag": conc["mean_offdiag"],
                "std_offdiag": conc["std_offdiag"],
                "collapse_to_half": conc["collapse_to_half"],
                "fid_near": near,
                "fid_far": far,
                "capped": view.capped,
                "kernel_n": n_kernel,
                "qiskit": use_quantum,
            }
        )

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Gravado {args.out}")


if __name__ == "__main__":
    main()
