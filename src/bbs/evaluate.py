"""Avaliação das amostras com DBSCAN, no protocolo dos artigos de 2007."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ClusterEval:
    n_clusters: int
    n_noise: int
    error_rate: float
    ari: float
    n_sampled: int


def scale_unit(X: NDArray[np.floating]) -> NDArray[np.float64]:
    return MinMaxScaler().fit_transform(np.asarray(X, dtype=np.float64))


def run_dbscan(
    X: NDArray[np.floating],
    eps: float,
    min_samples: int,
) -> NDArray[np.int64]:
    model = DBSCAN(eps=eps, min_samples=max(1, min_samples))
    return model.fit_predict(scale_unit(X)).astype(np.int64)


def evaluate_sample(
    X: NDArray[np.floating],
    y_true: NDArray[np.integer],
    indices: NDArray[np.integer],
    eps: float,
    min_samples: int,
) -> ClusterEval:
    """Aplica DBSCAN à amostra e compara com os rótulos originais (não usados no clustering)."""
    idx = np.asarray(indices, dtype=np.int64)
    Xs = np.asarray(X, dtype=np.float64)[idx]
    ys = np.asarray(y_true, dtype=np.int64)[idx]
    pred = run_dbscan(Xs, eps=eps, min_samples=min_samples)
    n_clusters = int(len(set(pred.tolist()) - {-1}))
    n_noise = int(np.sum(pred == -1))
    true_noise = ys < 0
    pred_noise = pred < 0
    noise_mismatch = float(np.mean(true_noise != pred_noise))
    comparable = ~true_noise
    if np.any(comparable) and len(set(pred[comparable].tolist()) - {-1}) > 0:
        ari = float(adjusted_rand_score(ys[comparable], pred[comparable]))
    else:
        ari = 0.0
    error_rate = 0.5 * noise_mismatch + 0.5 * max(0.0, 1.0 - max(ari, 0.0))
    return ClusterEval(
        n_clusters=n_clusters,
        n_noise=n_noise,
        error_rate=error_rate,
        ari=ari,
        n_sampled=int(idx.size),
    )


def min_samples_for_onebig(ratio: float, smallest_cluster: int = 1_000) -> int:
    """n proporcional ao menor cluster na amostra, como no PKDD."""
    expected = max(3, int(round(smallest_cluster * ratio * 0.2)))
    return expected


def min_samples_for_pendigits(n_sample: int) -> int:
    """MinPts do DBSCAN na amostra.

    O SAC usa 10% do conjunto *original* no Pendigits completo; nas amostras
    isso impede qualquer cluster. Aqui escala com o tamanho da amostra
    (cerca de 2%, mínimo 5), mantendo eps=0.4.
    """
    return max(5, int(round(0.02 * n_sample)))
