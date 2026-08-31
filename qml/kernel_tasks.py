"""Tarefas QML (e um controle RBF) sobre um kernel pré-computado.

Família coberta: qualquer método que use similaridade de estados angle-encoded
(kernel SVM, kNN por fidelidade, clustering espectral, one-class, KRR).
O mesmo K alimenta todas; o que muda com q é a geometria do feature map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC, OneClassSVM

from qml.circuits import kernel_concentration, near_far_fidelity

KERNEL_FAMILIES = ("fidelity", "rbf")


@dataclass
class TaskScores:
    mean_offdiag: float
    std_offdiag: float
    fid_near: float
    fid_far: float
    near_far_ratio: float
    kernel_alive: bool
    alignment: float
    knn_acc: float
    cluster_ari: float
    oneclass_auc: float
    krr_r2: float
    qsvm_f1: float


def frobenius_cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(a * b) / denom)


def kernel_target_alignment(kernel: NDArray[np.floating], y: NDArray[np.integer]) -> float:
    """Alinhamento de kernel com a gram dos rótulos 1[yi=yj]."""
    kernel = np.asarray(kernel, dtype=np.float64)
    y = np.asarray(y)
    gram = np.equal.outer(y, y).astype(np.float64)
    return frobenius_cosine(kernel, gram)


def near_far_ratio(near: float, far: float) -> float:
    return float(near / far) if far > 1e-12 else (float("inf") if near > 0 else 0.0)


def kernel_is_alive(
    near: float,
    far: float,
    mean_offdiag: float,
    *,
    near_floor: float = 0.25,
    ratio_floor: float = 2.0,
    mean_floor: float = 0.03,
) -> bool:
    """Kernel ainda distingue geometria clássica: near≫far e off-diagonais não nulos."""
    ratio = near_far_ratio(near, far)
    return bool(near >= near_floor and ratio >= ratio_floor and mean_offdiag >= mean_floor)


def _split_kernel(
    kernel: NDArray[np.floating],
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
) -> NDArray[np.float64]:
    return np.asarray(kernel, dtype=np.float64)[np.ix_(np.asarray(rows), np.asarray(cols))]


def fidelity_knn_accuracy(
    kernel: NDArray[np.floating],
    y: NDArray[np.integer],
    train_idx: NDArray[np.integer],
    test_idx: NDArray[np.integer],
) -> float:
    """kNN de um vizinho médio: classe com maior fidelidade média no treino."""
    y = np.asarray(y)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    classes = np.unique(y[train_idx])
    if classes.size < 2 or test_idx.size == 0:
        return 0.0
    k_te = _split_kernel(kernel, test_idx, train_idx)
    pred = np.empty(test_idx.size, dtype=np.int64)
    y_tr = y[train_idx]
    for i in range(test_idx.size):
        best_c = int(classes[0])
        best_s = -1.0
        for c in classes:
            mask = y_tr == c
            if not np.any(mask):
                continue
            score = float(k_te[i, mask].mean())
            if score > best_s:
                best_s = score
                best_c = int(c)
        pred[i] = best_c
    return float(accuracy_score(y[test_idx], pred))


def spectral_ari(kernel: NDArray[np.floating], y: NDArray[np.integer]) -> float:
    y = np.asarray(y)
    valid = y >= 0
    labels = y[valid]
    n_clusters = int(np.unique(labels).size)
    if n_clusters < 2 or int(np.sum(valid)) < n_clusters:
        return 0.0
    k_use = np.clip(np.asarray(kernel, dtype=np.float64)[np.ix_(valid, valid)], 0.0, 1.0)
    off = k_use[~np.eye(k_use.shape[0], dtype=bool)]
    if off.size == 0 or float(off.mean()) < 0.03:
        return 0.0
    k_use = 0.5 * (k_use + k_use.T)
    k_use = k_use + 1e-4 * np.eye(k_use.shape[0])
    try:
        pred = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=0,
            assign_labels="discretize",
        ).fit_predict(k_use)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError):
        return 0.0
    return float(adjusted_rand_score(labels, pred))


def oneclass_auc(
    kernel: NDArray[np.floating],
    y: NDArray[np.integer],
    train_idx: NDArray[np.integer],
    test_idx: NDArray[np.integer],
    majority_label: int,
) -> float:
    y = np.asarray(y)
    inliers = train_idx[y[train_idx] == majority_label]
    if inliers.size < 5 or test_idx.size < 4:
        return float("nan")
    y_anom = (y[test_idx] != majority_label).astype(np.int64)
    if y_anom.min() == y_anom.max():
        return float("nan")
    try:
        oc = OneClassSVM(kernel="precomputed", nu=0.15)
        oc.fit(_split_kernel(kernel, inliers, inliers))
        scores = oc.decision_function(_split_kernel(kernel, test_idx, inliers))
        return float(roc_auc_score(y_anom, -np.asarray(scores)))
    except ValueError:
        return float("nan")


def krr_r2(
    kernel: NDArray[np.floating],
    target: NDArray[np.floating],
    train_idx: NDArray[np.integer],
    test_idx: NDArray[np.integer],
) -> float:
    target = np.asarray(target, dtype=np.float64)
    if train_idx.size < 3 or test_idx.size < 2:
        return float("nan")
    try:
        model = KernelRidge(alpha=1e-2, kernel="precomputed")
        model.fit(_split_kernel(kernel, train_idx, train_idx), target[train_idx])
        pred = model.predict(_split_kernel(kernel, test_idx, train_idx))
        return float(r2_score(target[test_idx], pred))
    except ValueError:
        return float("nan")


def precomputed_svm_f1(
    kernel: NDArray[np.floating],
    y: NDArray[np.integer],
    train_idx: NDArray[np.integer],
    test_idx: NDArray[np.integer],
    minority_label: int,
) -> float:
    y = np.asarray(y)
    y_tr = y[train_idx]
    if np.unique(y_tr).size < 2:
        return 0.0
    clf = SVC(kernel="precomputed")
    clf.fit(_split_kernel(kernel, train_idx, train_idx), y_tr)
    pred = clf.predict(_split_kernel(kernel, test_idx, train_idx))
    n_classes = int(np.unique(y).size)
    if n_classes == 2:
        return float(
            f1_score(
                y[test_idx],
                pred,
                pos_label=minority_label,
                average="binary",
                zero_division=0,
            )
        )
    return float(f1_score(y[test_idx], pred, average="macro", zero_division=0))


def rbf_kernel_matrix(X: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.asarray(rbf_kernel(np.asarray(X, dtype=np.float64)), dtype=np.float64)


def score_geometry(
    kernel: NDArray[np.floating],
    X_encoded: NDArray[np.floating],
    *,
    near_floor: float = 0.25,
    ratio_floor: float = 2.0,
    mean_floor: float = 0.03,
) -> tuple[float, float, float, float, float, bool]:
    """Devolve (mean_off, std_off, near, far, ratio, alive) sem tarefas a jusante."""
    conc = kernel_concentration(kernel)
    near, far = near_far_fidelity(X_encoded, kernel)
    ratio = near_far_ratio(near, far)
    finite_ratio = ratio if np.isfinite(ratio) else 0.0
    alive = kernel_is_alive(
        near,
        far,
        conc["mean_offdiag"],
        near_floor=near_floor,
        ratio_floor=ratio_floor,
        mean_floor=mean_floor,
    )
    return conc["mean_offdiag"], conc["std_offdiag"], near, far, finite_ratio, alive


def score_kernel(
    kernel: NDArray[np.floating],
    X_encoded: NDArray[np.floating],
    y: NDArray[np.integer],
    train_idx: NDArray[np.integer],
    test_idx: NDArray[np.integer],
    minority_label: int,
    majority_label: int,
    regression_target: NDArray[np.floating],
    *,
    geometry_only: bool = False,
    near_floor: float = 0.25,
    ratio_floor: float = 2.0,
    mean_floor: float = 0.03,
) -> TaskScores:
    mean_off, std_off, near, far, ratio, alive = score_geometry(
        kernel,
        X_encoded,
        near_floor=near_floor,
        ratio_floor=ratio_floor,
        mean_floor=mean_floor,
    )
    if geometry_only:
        return TaskScores(
            mean_offdiag=mean_off,
            std_offdiag=std_off,
            fid_near=near,
            fid_far=far,
            near_far_ratio=ratio,
            kernel_alive=alive,
            alignment=float("nan"),
            knn_acc=float("nan"),
            cluster_ari=float("nan"),
            oneclass_auc=float("nan"),
            krr_r2=float("nan"),
            qsvm_f1=float("nan"),
        )
    return TaskScores(
        mean_offdiag=mean_off,
        std_offdiag=std_off,
        fid_near=near,
        fid_far=far,
        near_far_ratio=ratio,
        kernel_alive=alive,
        alignment=kernel_target_alignment(kernel, y),
        knn_acc=fidelity_knn_accuracy(kernel, y, train_idx, test_idx),
        cluster_ari=spectral_ari(kernel, y),
        oneclass_auc=oneclass_auc(kernel, y, train_idx, test_idx, majority_label),
        krr_r2=krr_r2(kernel, regression_target, train_idx, test_idx),
        qsvm_f1=precomputed_svm_f1(kernel, y, train_idx, test_idx, minority_label),
    )
