#!/usr/bin/env python3
"""US vs BBS no mesmo n e Es: SVM clássico e QSVM, F1 da minoria.

Protocolo honesto: se BBS+SVM já recupera a minoria, o ganho não é quântico.
QSVM só entra como comparação no mesmo coreset. Simulador; sem hardware.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from bbs.sampler import BiasedBoxSampler
from qml.circuits import QISKIT_AVAILABLE
from qml.classify import classical_svm_predict, minority_scores, qsvm_predict
from qml.data import load_qml_dataset
from qml.features import select_feature_view

SAMPLE_METHODS = ("us", "bbs", "bbs_pca")


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _sample(
    method: str,
    X_train: np.ndarray,
    grid_sig: np.ndarray,
    grid_pca: np.ndarray,
    n: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "us":
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(X_train.shape[0], size=n, replace=False).astype(np.int64))
        return idx, np.ones(idx.size, dtype=np.float64)
    if method == "bbs":
        sampler = BiasedBoxSampler(ratio=n / X_train.shape[0], n_levels=5, random_state=random_state)
        return sampler.fit_sample_with_weights(X_train, grid_X=grid_sig, n_points=n)
    if method == "bbs_pca":
        sampler = BiasedBoxSampler(ratio=n / X_train.shape[0], n_levels=5, random_state=random_state)
        return sampler.fit_sample_with_weights(X_train, grid_X=grid_pca, n_points=n)
    raise ValueError(f"method deve ser um de {SAMPLE_METHODS}, recebido {method!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("onebig", "breast", "credit"), default="onebig")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--q-max", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--max-test",
        type=int,
        default=200,
        help="Teto de pontos no teste (kernel n_test × n). 0 = todos.",
    )
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--entanglement", choices=("linear", "full"), default="linear")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SAMPLE_METHODS,
        default=list(SAMPLE_METHODS),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("d2-qubit-budget/results/qsvm_minority.csv"),
    )
    args = parser.parse_args()
    use_quantum = QISKIT_AVAILABLE and not args.classical_only

    data = load_qml_dataset(args.dataset, random_state=0)
    print(
        f"{data.name}: N={data.X.shape[0]} E={data.X.shape[1]} "
        f"minoria={data.minority_label} qiskit={use_quantum}",
        flush=True,
    )
    idx_all = np.arange(data.X.shape[0])
    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=args.test_size,
        random_state=0,
        stratify=data.y,
    )
    X_train, y_train = data.X[train_idx], data.y[train_idx]
    X_test, y_test = data.X[test_idx], data.y[test_idx]
    if args.max_test and X_test.shape[0] > args.max_test:
        rng = np.random.default_rng(1)
        keep = rng.choice(X_test.shape[0], size=args.max_test, replace=False)
        X_test, y_test = X_test[keep], y_test[keep]
        test_idx = test_idx[keep]
    y_fine_train = data.y_fine[train_idx]

    view_sig = select_feature_view(X_train, method="fdase", q_max=args.q_max, random_state=0)
    view_pca = select_feature_view(X_train, method="pca", q_max=args.q_max, random_state=0)
    print(
        f"Es FD-ASE q={view_sig.q} D2={view_sig.d2:.3f}; PCA q={view_pca.q}",
        flush=True,
    )
    if view_sig.columns is None or view_sig.columns.size == 0:
        raise RuntimeError("FD-ASE deveria devolver colunas originais.")
    Xtr_es = X_train[:, view_sig.columns]
    Xte_es = X_test[:, view_sig.columns]

    n_keep = min(int(args.n), X_train.shape[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for method in args.methods:
        f1_svm: list[float] = []
        f1_svm_w: list[float] = []
        rec_svm: list[float] = []
        f1_q: list[float] = []
        f1_q_w: list[float] = []
        rec_q: list[float] = []
        noise_frac: list[float] = []
        for rep in range(args.repeats):
            local, weights = _sample(
                method,
                X_train,
                grid_sig=view_sig.X,
                grid_pca=view_pca.X,
                n=n_keep,
                random_state=rep,
            )
            Xs, ys, ws = Xtr_es[local], y_train[local], weights
            if data.noise_label is not None:
                noise_frac.append(float(np.mean(y_fine_train[local] == data.noise_label)))
            else:
                noise_frac.append(0.0)

            pred = classical_svm_predict(Xs, ys, Xte_es)
            f1, rec = minority_scores(y_test, pred, data.minority_label)
            pred_w = classical_svm_predict(Xs, ys, Xte_es, sample_weight=ws)
            f1w, _ = minority_scores(y_test, pred_w, data.minority_label)
            f1_svm.append(f1)
            rec_svm.append(rec)
            f1_svm_w.append(f1w)

            q_f1 = q_f1w = q_rec = float("nan")
            if use_quantum and Xs.shape[1] >= 2:
                qpred = qsvm_predict(
                    Xs, ys, Xte_es, reps=args.reps, entanglement=args.entanglement
                )
                q_f1, q_rec = minority_scores(y_test, qpred, data.minority_label)
                qpred_w = qsvm_predict(
                    Xs,
                    ys,
                    Xte_es,
                    sample_weight=ws,
                    reps=args.reps,
                    entanglement=args.entanglement,
                )
                q_f1w, _ = minority_scores(y_test, qpred_w, data.minority_label)
            f1_q.append(q_f1)
            f1_q_w.append(q_f1w)
            rec_q.append(q_rec)
            print(
                f"  {method:7s} rep={rep} n={len(local)} "
                f"F1-SVM={f1:.3f} F1-SVM-w={f1w:.3f} "
                f"F1-QSVM={q_f1:.3f} ruído={noise_frac[-1]:.3f}",
                flush=True,
            )
        rows.append(
            {
                "dataset": data.name,
                "sample": method,
                "n": n_keep,
                "q": view_sig.q,
                "d2": view_sig.d2,
                "f1_svm": _mean(f1_svm),
                "recall_svm": _mean(rec_svm),
                "f1_svm_weighted": _mean(f1_svm_w),
                "f1_qsvm": _mean(f1_q),
                "recall_qsvm": _mean(rec_q),
                "f1_qsvm_weighted": _mean(f1_q_w),
                "noise_frac": float(np.mean(noise_frac)),
                "qiskit": use_quantum,
            }
        )

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Gravado {args.out}")
    print(
        "Protocolo: se BBS+SVM clássico já recupera a minoria, o ganho não é quântico.",
        flush=True,
    )


if __name__ == "__main__":
    main()
