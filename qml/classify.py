"""Classificadores no mesmo n e Es: SVM clássico e QSVM (kernel de fidelidade)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, recall_score
from sklearn.svm import SVC

from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel


def classical_svm_predict(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_test: NDArray[np.floating],
    sample_weight: NDArray[np.floating] | None = None,
    kernel: str = "rbf",
) -> NDArray[np.int64]:
    y_train = np.asarray(y_train)
    n_test = np.asarray(X_test).shape[0]
    if np.unique(y_train).size < 2:
        fill = int(y_train[0]) if y_train.size else 0
        return np.full(n_test, fill, dtype=np.int64)
    clf = SVC(kernel=kernel, class_weight=None)
    fit_kw: dict[str, object] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    clf.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train), **fit_kw)
    return clf.predict(np.asarray(X_test, dtype=np.float64)).astype(np.int64)


def qsvm_predict(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_test: NDArray[np.floating],
    sample_weight: NDArray[np.floating] | None = None,
    reps: int = 1,
    entanglement: str = "linear",
) -> NDArray[np.int64]:
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit ausente; QSVM indisponível.")
    y_train = np.asarray(y_train)
    n_test = np.asarray(X_test).shape[0]
    if np.unique(y_train).size < 2:
        fill = int(y_train[0]) if y_train.size else 0
        return np.full(n_test, fill, dtype=np.int64)
    k_train = fidelity_kernel(X_train, reps=reps, entanglement=entanglement)
    k_test = fidelity_kernel(X_test, X_train, reps=reps, entanglement=entanglement)
    clf = SVC(kernel="precomputed")
    fit_kw: dict[str, object] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    clf.fit(k_train, np.asarray(y_train), **fit_kw)
    return clf.predict(k_test).astype(np.int64)


def minority_scores(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    minority_label: int,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=minority_label,
            average="binary",
            zero_division=0,
        )
    )
    rec = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=minority_label,
            average="binary",
            zero_division=0,
        )
    )
    return f1, rec
