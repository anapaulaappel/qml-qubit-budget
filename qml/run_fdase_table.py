#!/usr/bin/env python3
"""Kernel FD-ASE nos nove conjuntos: geometria clássica, n=32, mapa ZZ."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE
from qml.data import DATASET_NAMES, display_name, load_qml_dataset
from qml.qubit_budget import last_alive_q, run_dataset_sweep

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
DEFAULT_OUT = Path("d2-qubit-budget/docs/qubit-budget/fdase_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DATASET_NAMES)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--q-max", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit é necessário.")

    records: list[dict[str, object]] = []
    print(
        f"{'dataset':12s} {'⌈D2⌉':>5s} {'PCA95':>5s} {'PCA knee':>8s} "
        f"{'FD-ASE q':>8s} {'alive':>6s} {'near':>6s} {'far':>6s} {'ratio':>6s}",
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
            include_fdase=True,
        )
        pca_knee = last_alive_q(rows, "pca", "fidelity")
        fdase = [r for r in rows if r.view == "fdase" and r.family == "fidelity"]
        rec: dict[str, object] = {
            "dataset": display_name(data.name),
            "raw_name": data.name,
            "e": rows[0].e_original,
            "d2": rows[0].d2,
            "d2_ceiling": rows[0].d2_ceiling,
            "pca95": rows[0].pca95,
            "pca_knee": "" if pca_knee is None else pca_knee,
            "n": args.n,
        }
        if fdase:
            r = fdase[0]
            rec.update(
                {
                    "fdase_q": r.q,
                    "fdase_alive": r.kernel_alive,
                    "fid_near": r.fid_near,
                    "fid_far": r.fid_far,
                    "near_far_ratio": r.near_far_ratio,
                    "mean_offdiag": r.mean_offdiag,
                }
            )
            alive = "yes" if r.kernel_alive else "no"
            print(
                f"{display_name(data.name):12s} {r.d2_ceiling:5d} {r.pca95:5d} "
                f"{str(pca_knee):>8s} {r.q:8d} {alive:>6s} "
                f"{r.fid_near:6.3f} {r.fid_far:6.3f} {r.near_far_ratio:6.2f}",
                flush=True,
            )
        else:
            rec.update(
                {
                    "fdase_q": "",
                    "fdase_alive": "",
                    "fid_near": "",
                    "fid_far": "",
                    "near_far_ratio": "",
                    "mean_offdiag": "",
                }
            )
            print(
                f"{display_name(data.name):12s} {rows[0].d2_ceiling:5d} {rows[0].pca95:5d} "
                f"{str(pca_knee):>8s} {'—':>8s} {'n/a':>6s}",
                flush=True,
            )
        records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV: {args.out}", flush=True)


if __name__ == "__main__":
    main()
