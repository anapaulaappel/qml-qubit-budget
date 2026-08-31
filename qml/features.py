"""Vistas de atributos para encoding NISQ: full, PCA, FD-ASE e eixos aleatórios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from bbs.fdase import FDASE, correlation_fractal_dimension

FEATURE_VIEWS = ("full", "pca", "fdase", "random")
SWEEP_VIEWS = ("pca", "random", "prefix")


def project_to_q(
    X: NDArray[np.floating],
    q: int,
    method: str,
    random_state: int = 0,
) -> NDArray[np.float64]:
    """Projeta X para exatamente ``q`` colunas (PCA, eixos aleatórios ou prefixo)."""
    if method not in SWEEP_VIEWS:
        raise ValueError(f"method deve ser um de {SWEEP_VIEWS}, recebido {method!r}.")
    X = np.asarray(X, dtype=np.float64)
    n, e = X.shape
    q = max(2, min(int(q), e, max(1, n - 1)))
    if method == "pca":
        pca = PCA(n_components=q, random_state=random_state)
        return np.asarray(pca.fit_transform(X), dtype=np.float64)
    if method == "random":
        rng = np.random.default_rng(random_state)
        cols = np.sort(rng.choice(e, size=q, replace=False).astype(np.int64))
        return X[:, cols]
    if method == "prefix":
        return X[:, :q]
    raise ValueError(f"method deve ser um de {SWEEP_VIEWS}, recebido {method!r}.")


def d2_qubit_ceiling(d2: float, q_min: int = 2) -> int:
    """Teto de qubits proposto: max(q_min, ceil(D2))."""
    if not np.isfinite(d2) or d2 <= 0.0:
        return q_min
    return max(q_min, int(np.ceil(d2)))


@dataclass
class FeatureView:
    """Matriz ``N x q`` para o feature map, com metadados do recorte."""

    X: NDArray[np.float64]
    method: str
    d2: float
    q: int
    e_original: int
    q_amp: int
    capped: bool
    columns: NDArray[np.int64] | None


def intrinsic_k(d2: float, e: int, q_max: int, n_samples: int) -> int:
    k = int(np.ceil(d2)) if d2 > 0.0 else min(8, e)
    return max(2, min(k, q_max, e, max(1, n_samples - 1)))


def amplitude_qubits(n_features: int) -> int:
    """Qubits de amplitude encoding: ceil(log2 E). O BBS não reduz este circuito."""
    return int(np.ceil(np.log2(max(int(n_features), 2))))


def select_feature_view(
    X: NDArray[np.floating],
    method: str,
    q_max: int = 8,
    n_levels: int = 8,
    random_state: int = 0,
) -> FeatureView:
    """Escolhe ``q`` eixos/componentes. ``full`` usa todas as E se E <= q_max."""
    if method not in FEATURE_VIEWS:
        raise ValueError(f"method deve ser um de {FEATURE_VIEWS}, recebido {method!r}.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X deve ter forma (n_amostras, n_atributos).")
    n, e = X.shape
    d2 = float(correlation_fractal_dimension(X, n_levels=n_levels))
    k = intrinsic_k(d2, e, q_max, n)
    q_amp = amplitude_qubits(e)

    if method == "full":
        q = min(e, q_max)
        return FeatureView(
            X=X[:, :q],
            method="full",
            d2=d2,
            q=q,
            e_original=e,
            q_amp=q_amp,
            capped=e > q_max,
            columns=np.arange(q, dtype=np.int64),
        )

    if method == "pca":
        pca = PCA(n_components=k, random_state=random_state)
        xp = np.asarray(pca.fit_transform(X), dtype=np.float64)
        return FeatureView(
            X=xp,
            method="pca",
            d2=d2,
            q=int(xp.shape[1]),
            e_original=e,
            q_amp=q_amp,
            capped=False,
            columns=None,
        )

    if method == "fdase":
        fdase = FDASE(eps=0.15, n_levels=min(n_levels, 7)).fit(X)
        selected = fdase.selected_
        if selected is None or selected.size == 0:
            cols = np.arange(k, dtype=np.int64)
        else:
            cols = np.asarray(selected[: min(selected.size, q_max)], dtype=np.int64)
            if cols.size < 2:
                cols = np.arange(min(2, e), dtype=np.int64)
        return FeatureView(
            X=X[:, cols],
            method="fdase",
            d2=d2,
            q=int(cols.size),
            e_original=e,
            q_amp=q_amp,
            capped=bool(selected is not None and selected.size > q_max),
            columns=cols,
        )

    if method == "random":
        rng = np.random.default_rng(random_state)
        cols = np.sort(rng.choice(e, size=min(k, e), replace=False).astype(np.int64))
        return FeatureView(
            X=X[:, cols],
            method="random",
            d2=d2,
            q=int(cols.size),
            e_original=e,
            q_amp=q_amp,
            capped=False,
            columns=cols,
        )

    raise ValueError(f"method deve ser um de {FEATURE_VIEWS}, recebido {method!r}.")
