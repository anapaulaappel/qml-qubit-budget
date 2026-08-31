"""Conjuntos tabulares para os estudos QML (independentes da linha LLM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer as _sk_breast_cancer
from sklearn.datasets import load_diabetes as _sk_diabetes
from sklearn.datasets import load_digits as _sk_digits
from sklearn.datasets import load_iris as _sk_iris
from sklearn.datasets import load_wine as _sk_wine
from sklearn.datasets import make_moons

from bbs.datasets import load_pendigits, make_onebig_tiny

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    fetch_openml = None


@dataclass
class QmlDataset:
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    y_fine: NDArray[np.int64]
    name: str
    minority_label: int
    noise_label: int | None


def _minority_label(y: NDArray[np.integer]) -> int:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return int(values[int(np.argmin(counts))])


def load_onebig_binary(random_state: int = 0) -> QmlDataset:
    """OneBig reduzido: minoria = clusters pequenos (y>0); resto = grande + ruído."""
    data = make_onebig_tiny(random_state=random_state)
    y_bin = np.where(data.y > 0, 1, 0).astype(np.int64)
    return QmlDataset(
        X=data.X,
        y=y_bin,
        y_fine=data.y,
        name="onebig_tiny",
        minority_label=1,
        noise_label=-1,
    )


def load_breast() -> QmlDataset:
    """Wisconsin breast cancer (E=30, N pequeno)."""
    bunch = _sk_breast_cancer()
    y = np.asarray(bunch.target, dtype=np.int64)
    return QmlDataset(
        X=np.asarray(bunch.data, dtype=np.float64),
        y=y,
        y_fine=y,
        name="breast_cancer",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_credit(max_rows: int = 2_000, random_state: int = 0) -> QmlDataset | None:
    """Subamostra OpenML de crédito (opcional; falha de rede devolve None)."""
    if fetch_openml is None:
        return None
    try:
        bunch = fetch_openml("credit-g", version=1, as_frame=False, parser="auto")
    except Exception:
        return None
    X = np.asarray(bunch.data, dtype=np.float64)
    raw = np.asarray(bunch.target)
    y = (raw.astype(str) == "bad").astype(np.int64)
    n = X.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(random_state)
        take = rng.choice(n, size=max_rows, replace=False)
        X = X[take]
        y = y[take]
    return QmlDataset(
        X=X,
        y=y,
        y_fine=y,
        name="credit_g",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_wine() -> QmlDataset:
    bunch = _sk_wine()
    y = np.asarray(bunch.target, dtype=np.int64)
    return QmlDataset(
        X=np.asarray(bunch.data, dtype=np.float64),
        y=y,
        y_fine=y,
        name="wine",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_digits(max_rows: int = 1_797, random_state: int = 0) -> QmlDataset:
    bunch = _sk_digits()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.int64)
    if X.shape[0] > max_rows:
        rng = np.random.default_rng(random_state)
        take = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y = X[take], y[take]
    return QmlDataset(
        X=X,
        y=y,
        y_fine=y,
        name="digits",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_intrinsic2(
    n: int = 400,
    n_features: int = 20,
    random_state: int = 0,
) -> QmlDataset:
    """Círculo 2-D embebido em E dimensões (controle: D2 ≈ 2)."""
    rng = np.random.default_rng(random_state)
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    signal = np.column_stack((np.cos(t), np.sin(t)))
    signal += 0.05 * rng.normal(size=signal.shape)
    extra = max(0, int(n_features) - 2)
    noise = 0.05 * rng.normal(size=(n, extra))
    X = np.hstack((signal, noise)) if extra else signal
    y = (t > np.pi).astype(np.int64)
    return QmlDataset(
        X=X.astype(np.float64),
        y=y,
        y_fine=y,
        name="intrinsic2",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_iris() -> QmlDataset:
    bunch = _sk_iris()
    y = np.asarray(bunch.target, dtype=np.int64)
    return QmlDataset(
        X=np.asarray(bunch.data, dtype=np.float64),
        y=y,
        y_fine=y,
        name="iris",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_diabetes() -> QmlDataset:
    """Diabetes UCI: alvo contínuo binarizado na mediana (para as tarefas com rótulo)."""
    bunch = _sk_diabetes()
    X = np.asarray(bunch.data, dtype=np.float64)
    t = np.asarray(bunch.target, dtype=np.float64)
    y = (t > np.median(t)).astype(np.int64)
    return QmlDataset(
        X=X,
        y=y,
        y_fine=np.round(t).astype(np.int64),
        name="diabetes",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_pendigits_qml(max_rows: int = 4_000, random_state: int = 0) -> QmlDataset:
    data = load_pendigits()
    X, y = data.X, data.y
    if X.shape[0] > max_rows:
        rng = np.random.default_rng(random_state)
        take = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y = X[take], y[take]
    return QmlDataset(
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=np.int64),
        y_fine=np.asarray(y, dtype=np.int64),
        name="pendigits",
        minority_label=_minority_label(y),
        noise_label=None,
    )


def load_moons(
    n: int = 400,
    n_features: int = 20,
    random_state: int = 0,
) -> QmlDataset:
    """Two moons 2-D embebidos em E dimensões (variedade ~1 com dois ramos)."""
    X2, y = make_moons(n_samples=n, noise=0.08, random_state=random_state)
    rng = np.random.default_rng(random_state)
    extra = max(0, int(n_features) - 2)
    noise = 0.05 * rng.normal(size=(n, extra))
    X = np.hstack((X2, noise)) if extra else X2
    y = np.asarray(y, dtype=np.int64)
    return QmlDataset(
        X=np.asarray(X, dtype=np.float64),
        y=y,
        y_fine=y,
        name="moons",
        minority_label=_minority_label(y),
        noise_label=None,
    )


DATASET_NAMES = (
    "intrinsic2",
    "moons",
    "iris",
    "breast",
    "diabetes",
    "wine",
    "digits",
    "pendigits",
    "onebig",
    "credit",
)


def load_qml_dataset(name: str, random_state: int = 0) -> QmlDataset:
    if name == "onebig":
        return load_onebig_binary(random_state=random_state)
    if name == "breast":
        return load_breast()
    if name == "credit":
        data = load_credit(random_state=random_state)
        if data is None:
            raise RuntimeError("Não foi possível carregar credit-g (OpenML).")
        return data
    if name == "wine":
        return load_wine()
    if name == "digits":
        return load_digits(random_state=random_state)
    if name == "intrinsic2":
        return load_intrinsic2(random_state=random_state)
    if name == "iris":
        return load_iris()
    if name == "diabetes":
        return load_diabetes()
    if name == "pendigits":
        return load_pendigits_qml(random_state=random_state)
    if name == "moons":
        return load_moons(random_state=random_state)
    raise ValueError(f"name deve ser um de {DATASET_NAMES}, recebido {name!r}.")
