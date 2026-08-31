"""Estimadores clássicos de dimensão intrínseca, para comparar com D2 como teto de qubits.

TwoNN: Facco et al., Scientific Reports 2017.
PCA-95%: prática de facto em QML (reduzir até explicar 95% da variância, ou até caber no hardware).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def two_nn_dimension(X: NDArray[np.floating]) -> float:
    r"""Dimensão intrínseca TwoNN: \(d = (\mathrm{média}\,\log(r_2/r_1))^{-1}\)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 8 or X.shape[1] == 0:
        return 0.0
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    r1 = np.maximum(dist[:, 1], 1e-12)
    r2 = np.maximum(dist[:, 2], r1 * (1.0 + 1e-12))
    logs = np.log(r2 / r1)
    mean_log = float(np.mean(logs))
    if mean_log <= 1e-12:
        return 0.0
    return float(np.clip(1.0 / mean_log, 0.0, float(X.shape[1])))


def pca_variance_qubits(X: NDArray[np.floating], threshold: float = 0.95) -> int:
    """Menor q cuja PCA, após standardização, acumula ``threshold`` da variância.

    A standardização é a prática usual nos pipelines QML (escala heterogénea
    senão o primeiro PC captura só a unidade da feature mais larga).
    """
    X = np.asarray(X, dtype=np.float64)
    n, e = X.shape
    if n < 2 or e == 0:
        return 1
    Xs = StandardScaler().fit_transform(X)
    k = max(1, min(n - 1, e))
    pca = PCA(n_components=k)
    pca.fit(Xs)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    q = int(np.searchsorted(cumulative, threshold) + 1)
    return max(1, min(q, e))
