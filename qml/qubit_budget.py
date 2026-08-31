"""Varredura q vs D2: o teto de qubits para kernels angle-encoded."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from bbs.fdase import FDASE, correlation_fractal_dimension
from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel
from qml.data import QmlDataset
from qml.features import SWEEP_VIEWS, d2_qubit_ceiling, project_to_q
from qml.id_estimators import pca_variance_qubits, two_nn_dimension
from qml.kernel_tasks import KERNEL_FAMILIES, TaskScores, rbf_kernel_matrix, score_kernel


@dataclass
class SweepRow:
    dataset: str
    view: str
    family: str
    q: int
    d2: float
    d2_ceiling: int
    e_original: int
    n: int
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
    last_alive: bool
    twonn: float
    twonn_ceiling: int
    pca95: int


def _majority_label(y: NDArray[np.integer]) -> int:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _regression_target(X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Alvo contínuo = 1º componente principal do X original (sem o feature map)."""
    n, e = X.shape
    k = 1
    pc = PCA(n_components=k, random_state=0).fit_transform(X)
    return np.asarray(pc[:, 0], dtype=np.float64)


def last_alive_q(rows: list[SweepRow], view: str, family: str) -> int | None:
    """Maior q em que o kernel ainda está vivo, na curva ``view``/``family``."""
    alive = [r.q for r in rows if r.view == view and r.family == family and r.kernel_alive]
    return max(alive) if alive else None


def run_dataset_sweep(
    data: QmlDataset,
    n_kernel: int = 32,
    q_max: int = 8,
    views: tuple[str, ...] = SWEEP_VIEWS,
    families: tuple[str, ...] = KERNEL_FAMILIES,
    n_levels: int = 8,
    random_state: int = 0,
    test_size: float = 0.3,
) -> list[SweepRow]:
    rng = np.random.default_rng(random_state)
    rng_kernel = np.random.default_rng(random_state + 17)
    n_all = data.X.shape[0]
    n_d2 = min(n_all, 4_000)
    d2_idx = rng.choice(n_all, size=n_d2, replace=False)
    X_id = np.asarray(data.X[d2_idx], dtype=np.float64)
    d2 = float(correlation_fractal_dimension(X_id, n_levels=n_levels))
    ceiling = d2_qubit_ceiling(d2)
    twonn = two_nn_dimension(X_id)
    twonn_ceiling = d2_qubit_ceiling(twonn)
    pca95 = pca_variance_qubits(X_id, threshold=0.95)
    take = min(int(n_kernel), n_all)
    idx = np.sort(rng_kernel.choice(n_all, size=take, replace=False))
    X = np.asarray(data.X[idx], dtype=np.float64)
    y = np.asarray(data.y[idx])
    local = np.arange(take)
    try:
        train_idx, test_idx = train_test_split(
            local, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            local, test_size=test_size, random_state=random_state
        )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    target = _regression_target(X)
    majority = _majority_label(y)
    q_hi = max(2, min(int(q_max), X.shape[1], take - 1))

    def _row(
        view: str,
        family: str,
        q: int,
        scores: TaskScores,
    ) -> SweepRow:
        return SweepRow(
            dataset=data.name,
            view=view,
            family=family,
            q=q,
            d2=d2,
            d2_ceiling=ceiling,
            e_original=int(data.X.shape[1]),
            n=take,
            mean_offdiag=scores.mean_offdiag,
            std_offdiag=scores.std_offdiag,
            fid_near=scores.fid_near,
            fid_far=scores.fid_far,
            near_far_ratio=scores.near_far_ratio,
            kernel_alive=scores.kernel_alive,
            alignment=scores.alignment,
            knn_acc=scores.knn_acc,
            cluster_ari=scores.cluster_ari,
            oneclass_auc=scores.oneclass_auc,
            krr_r2=scores.krr_r2,
            qsvm_f1=scores.qsvm_f1,
            last_alive=False,
            twonn=twonn,
            twonn_ceiling=twonn_ceiling,
            pca95=pca95,
        )

    rows: list[SweepRow] = []
    for view in views:
        if view not in SWEEP_VIEWS:
            raise ValueError(f"view deve ser um de {SWEEP_VIEWS}, recebido {view!r}.")
        for q in range(2, q_hi + 1):
            Xq = project_to_q(X, q, method=view, random_state=random_state)
            for family in families:
                if family not in KERNEL_FAMILIES:
                    raise ValueError(
                        f"family deve ser um de {KERNEL_FAMILIES}, recebido {family!r}."
                    )
                if family == "fidelity":
                    if not QISKIT_AVAILABLE:
                        continue
                    kernel = fidelity_kernel(Xq, reps=1, entanglement="linear")
                elif family == "rbf":
                    kernel = rbf_kernel_matrix(Xq)
                else:
                    raise ValueError(
                        f"family deve ser um de {KERNEL_FAMILIES}, recebido {family!r}."
                    )
                scores: TaskScores = score_kernel(
                    kernel,
                    Xq,
                    y,
                    train_idx,
                    test_idx,
                    minority_label=data.minority_label,
                    majority_label=majority,
                    regression_target=target,
                )
                rows.append(_row(view, family, q, scores))

    fdase = FDASE(eps=0.15, n_levels=min(n_levels, 7)).fit(X)
    if fdase.selected_ is not None and fdase.selected_.size >= 2:
        cols = fdase.selected_[: min(fdase.selected_.size, q_hi)]
        Xf = X[:, cols]
        q_f = int(Xf.shape[1])
        for family in families:
            if family == "fidelity" and not QISKIT_AVAILABLE:
                continue
            if family == "fidelity":
                kernel = fidelity_kernel(Xf, reps=1, entanglement="linear")
            elif family == "rbf":
                kernel = rbf_kernel_matrix(Xf)
            else:
                raise ValueError(
                    f"family deve ser um de {KERNEL_FAMILIES}, recebido {family!r}."
                )
            scores = score_kernel(
                kernel,
                Xf,
                y,
                train_idx,
                test_idx,
                minority_label=data.minority_label,
                majority_label=majority,
                regression_target=target,
            )
            rows.append(_row("fdase", family, q_f, scores))

    marked: list[SweepRow] = []
    last_q: dict[tuple[str, str], int] = {}
    for r in rows:
        if r.kernel_alive:
            key = (r.view, r.family)
            last_q[key] = max(last_q.get(key, 0), r.q)
    for r in rows:
        star = last_q.get((r.view, r.family))
        marked.append(replace(r, last_alive=star is not None and r.q == star))
    return marked
