"""Baselines de amostragem reimplementados de forma aproximada (US, GBS, DBS)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity

from bbs.grid import normalize_unit_cube


def _target_size(n: int, ratio: float) -> int:
    return max(1, min(n, int(round(n * ratio))))


def uniform_sample(
    X: NDArray[np.floating],
    ratio: float,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Amostragem uniforme sem reposição."""
    rng = np.random.default_rng(random_state)
    n = np.asarray(X).shape[0]
    return rng.choice(n, size=_target_size(n, ratio), replace=False).astype(np.int64)


def gbs_sample(
    X: NDArray[np.floating],
    ratio: float,
    exponent: float = 0.5,
    n_bins: int = 8,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """GBS aproximado (Palmer e Faloutsos, 2000).

    Células de lado fixo; a probabilidade por ponto é proporcional a
    ``n_i ** (exponent - 1)``, com ``exponent=0.5`` como no PKDD 2007.
    """
    rng = np.random.default_rng(random_state)
    Xn = normalize_unit_cube(np.asarray(X, dtype=np.float64))
    n = Xn.shape[0]
    bins = np.minimum((Xn * n_bins).astype(np.int64), n_bins - 1)
    cells: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, key in enumerate(map(tuple, bins.tolist())):
        cells[key].append(i)
    weights = np.empty(n, dtype=np.float64)
    for members in cells.values():
        n_i = len(members)
        w = float(n_i) ** (exponent - 1.0)
        for i in members:
            weights[i] = w
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return uniform_sample(X, ratio, random_state=rng)
    weights /= total
    k = _target_size(n, ratio)
    return rng.choice(n, size=k, replace=False, p=weights).astype(np.int64)


def dbs_sample(
    X: NDArray[np.floating],
    ratio: float,
    bias: float = -0.25,
    n_kernels: int = 1000,
    bandwidth: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """DBS aproximado (Kollios et al., 2003).

    KDE em até ``n_kernels`` centros e pesos ``f(x) ** bias``.
    ``bias=-0.25`` é o valor usado no PKDD 2007 para ruído e clusters pequenos.
    """
    rng = np.random.default_rng(random_state)
    Xn = normalize_unit_cube(np.asarray(X, dtype=np.float64))
    n, e = Xn.shape
    n_centers = min(n_kernels, n)
    centers = Xn[rng.choice(n, size=n_centers, replace=False)]
    if bandwidth is None:
        bandwidth = max(0.05, n ** (-1.0 / (e + 4)))
    kde = KernelDensity(bandwidth=bandwidth, kernel="epanechnikov")
    kde.fit(centers)
    log_density = kde.score_samples(Xn)
    density = np.exp(np.clip(log_density, -50.0, 50.0))
    density = np.maximum(density, 1e-15)
    weights = density**bias
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return uniform_sample(X, ratio, random_state=rng)
    weights /= total
    k = _target_size(n, ratio)
    return rng.choice(n, size=k, replace=False, p=weights).astype(np.int64)
