"""Geradores dos conjuntos do PKDD/SAC 2007 e loader do Pendigits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray


@dataclass
class LabeledDataset:
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    name: str

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.y[self.y >= 0]).size)


def make_onebig(random_state: int | np.random.Generator | None = 0) -> LabeledDataset:
    """OneBig: 20-D, 1 cluster com 50k, 8 com 1k, 10k de ruído uniforme (~15%)."""
    rng = np.random.default_rng(random_state)
    n_features = 20
    n_big = 50_000
    n_small = 1_000
    n_small_clusters = 8
    n_noise = 10_000
    centers = np.zeros((1 + n_small_clusters, n_features), dtype=np.float64)
    for i in range(1 + n_small_clusters):
        centers[i, i] = 6.0
    std_big = 0.35
    std_small = 0.25
    parts: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    parts.append(rng.normal(loc=centers[0], scale=std_big, size=(n_big, n_features)))
    labels.append(np.zeros(n_big, dtype=np.int64))
    for k in range(n_small_clusters):
        parts.append(
            rng.normal(loc=centers[k + 1], scale=std_small, size=(n_small, n_features))
        )
        labels.append(np.full(n_small, k + 1, dtype=np.int64))
    low = centers.min(axis=0) - 3.0
    high = centers.max(axis=0) + 3.0
    parts.append(rng.uniform(low, high, size=(n_noise, n_features)))
    labels.append(np.full(n_noise, -1, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    order = rng.permutation(X.shape[0])
    return LabeledDataset(X=X[order], y=y[order], name="OneBig")


def make_onebig_tiny(random_state: int | np.random.Generator | None = 0) -> LabeledDataset:
    """Versão reduzida do OneBig para testes unitários."""
    rng = np.random.default_rng(random_state)
    n_features = 8
    n_big = 2_000
    n_small = 80
    n_small_clusters = 4
    n_noise = 400
    centers = np.zeros((1 + n_small_clusters, n_features), dtype=np.float64)
    for i in range(1 + n_small_clusters):
        centers[i, i] = 5.0
    parts: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    parts.append(rng.normal(loc=centers[0], scale=0.3, size=(n_big, n_features)))
    labels.append(np.zeros(n_big, dtype=np.int64))
    for k in range(n_small_clusters):
        parts.append(rng.normal(loc=centers[k + 1], scale=0.25, size=(n_small, n_features)))
        labels.append(np.full(n_small, k + 1, dtype=np.int64))
    parts.append(rng.uniform(-2.0, 7.0, size=(n_noise, n_features)))
    labels.append(np.full(n_noise, -1, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    order = rng.permutation(X.shape[0])
    return LabeledDataset(X=X[order], y=y[order], name="OneBigTiny")


def _disk(
    rng: np.random.Generator,
    center: tuple[float, float],
    radius: float,
    n: int,
) -> NDArray[np.float64]:
    ang = rng.uniform(0, 2 * np.pi, size=n)
    rad = radius * np.sqrt(rng.uniform(0, 1, size=n))
    return np.column_stack((center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)))


def _ellipse(
    rng: np.random.Generator,
    center: tuple[float, float],
    rx: float,
    ry: float,
    n: int,
    angle: float = 0.0,
) -> NDArray[np.float64]:
    pts = _disk(rng, (0.0, 0.0), 1.0, n)
    pts[:, 0] *= rx
    pts[:, 1] *= ry
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.asarray(center)


def make_uniform_clusters(random_state: int | np.random.Generator | None = 0) -> LabeledDataset:
    """UniformClusters 2-D no estilo Kollios: um disco grande, dois pequenos,
    dois elipsóides ligados por uma cadeia de outliers, e ruído espalhado.
    """
    rng = np.random.default_rng(random_state)
    big = _disk(rng, (0.0, 0.0), 1.8, 3_000)
    small_a = _disk(rng, (4.2, 2.4), 0.45, 400)
    small_b = _disk(rng, (4.2, -2.4), 0.45, 400)
    ell_a = _ellipse(rng, (-3.8, 1.6), 1.1, 0.35, 500, angle=0.4)
    ell_b = _ellipse(rng, (-3.8, -1.6), 1.1, 0.35, 500, angle=-0.4)
    t = np.linspace(0, 1, 80)
    chain = np.column_stack((-3.8 + 0.05 * rng.normal(size=80), 1.6 + t * (-3.2) + 0.04 * rng.normal(size=80)))
    noise = rng.uniform(-6.0, 6.0, size=(350, 2))
    X = np.vstack([big, small_a, small_b, ell_a, ell_b, chain, noise])
    y = np.concatenate(
        [
            np.zeros(len(big), dtype=np.int64),
            np.full(len(small_a), 1, dtype=np.int64),
            np.full(len(small_b), 2, dtype=np.int64),
            np.full(len(ell_a), 3, dtype=np.int64),
            np.full(len(ell_b), 4, dtype=np.int64),
            np.full(len(chain), -1, dtype=np.int64),
            np.full(len(noise), -1, dtype=np.int64),
        ]
    )
    order = rng.permutation(X.shape[0])
    return LabeledDataset(X=X[order], y=y[order], name="UniformClusters")


def _pendigits_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _read_pendigits_file(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rows = np.loadtxt(path, delimiter=",")
    X = rows[:, :-1].astype(np.float64)
    y = rows[:, -1].astype(np.int64)
    return X, y


def _download_pendigits(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/"
    for name in ("pendigits.tra", "pendigits.tes"):
        url = base + name
        target = dest_dir / name
        if target.exists():
            continue
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())


def load_pendigits() -> LabeledDataset:
    """Pendigits UCI (10.992 × 16). Usa ``data/pendigits.{tra,tes}``; baixa se faltar."""
    data_dir = _pendigits_data_dir()
    tra = data_dir / "pendigits.tra"
    tes = data_dir / "pendigits.tes"
    if not (tra.exists() and tes.exists()):
        try:
            _download_pendigits(data_dir)
        except (OSError, URLError) as exc:
            raise FileNotFoundError(
                "Pendigits não encontrado em data/ e o download falhou. "
                "Coloque pendigits.tra e pendigits.tes nessa pasta."
            ) from exc
    X_tr, y_tr = _read_pendigits_file(tra)
    X_te, y_te = _read_pendigits_file(tes)
    X = np.vstack((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))
    return LabeledDataset(X=X, y=y, name="Pendigits")
