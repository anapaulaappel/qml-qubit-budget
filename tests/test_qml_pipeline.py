"""Pipeline QML: pesos, vistas de encoding e F1 da minoria (Qiskit opcional)."""

from __future__ import annotations

import numpy as np
import pytest

from bbs.sampler import BiasedBoxSampler
from qml.circuits import QISKIT_AVAILABLE, kernel_concentration
from qml.classify import classical_svm_predict, minority_scores
from qml.features import amplitude_qubits, select_feature_view


def _imbalanced(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    major = rng.normal(loc=0.0, scale=0.25, size=(280, 6))
    minor = rng.normal(loc=(5.0, 5.0, 0.0, 0.0, 0.0, 0.0), scale=0.2, size=(36, 6))
    junk = rng.normal(size=(major.shape[0] + minor.shape[0], 2))
    X = np.hstack((np.vstack((major, minor)), junk))
    y = np.concatenate((np.zeros(len(major), dtype=np.int64), np.ones(len(minor), dtype=np.int64)))
    return X, y


def test_amplitude_qubits_is_log() -> None:
    assert amplitude_qubits(8) == 3
    assert amplitude_qubits(30) == 5
    assert amplitude_qubits(1) == 1


def test_feature_views_q_and_rows() -> None:
    X, _ = _imbalanced()
    d2_ref = None
    for method in ("full", "pca", "fdase", "random"):
        view = select_feature_view(X, method=method, q_max=4, n_levels=6, random_state=0)
        assert view.X.shape[0] == X.shape[0]
        assert 2 <= view.q <= 4
        assert view.q == view.X.shape[1]
        assert view.e_original == X.shape[1]
        assert view.q_amp == amplitude_qubits(X.shape[1])
        if d2_ref is None:
            d2_ref = view.d2
        else:
            assert view.d2 == pytest.approx(d2_ref)
    full = select_feature_view(X, method="full", q_max=4, n_levels=6)
    assert full.capped is True
    with pytest.raises(ValueError):
        select_feature_view(X, method="nope")


def test_kernel_concentration_on_matrix() -> None:
    kernel = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )
    stats = kernel_concentration(kernel)
    assert stats["mean_offdiag"] == pytest.approx(0.5)
    assert stats["std_offdiag"] == pytest.approx(0.0)
    assert stats["collapse_to_half"] == pytest.approx(0.0)


def test_bbs_weights_aligned_with_sample_indices() -> None:
    X, _ = _imbalanced()
    view = select_feature_view(X, method="fdase", q_max=4, n_levels=6, random_state=0)
    sampler = BiasedBoxSampler(ratio=0.1, n_levels=5, random_state=1)
    sampler.fit(X, grid_X=view.X)
    idx = sampler.sample_indices(n_points=40)
    idx_w, weights = sampler.sample_indices_with_weights(n_points=40)
    np.testing.assert_array_equal(idx, idx_w)
    assert weights.shape == (40,)
    np.testing.assert_allclose(float(weights.mean()), 1.0)


def test_bbs_beats_us_minority_f1_classical() -> None:
    X, y = _imbalanced()
    grid = X[:, :2]
    n = X.shape[0]
    rng = np.random.default_rng(0)
    test = rng.choice(n, size=80, replace=False)
    train = np.array([i for i in range(n) if i not in set(test.tolist())], dtype=np.int64)
    n_us_min: list[int] = []
    n_bbs_min: list[int] = []
    f1_us: list[float] = []
    f1_bbs: list[float] = []
    for seed in range(6):
        us = rng.choice(train.size, size=40, replace=False)
        sampler = BiasedBoxSampler(ratio=40 / train.size, n_levels=5, random_state=seed)
        bbs, _ = sampler.fit_sample_with_weights(X[train], grid_X=grid[train], n_points=40)
        n_us_min.append(int(np.sum(y[train][us] == 1)))
        n_bbs_min.append(int(np.sum(y[train][bbs] == 1)))
        pred_us = classical_svm_predict(grid[train][us], y[train][us], grid[test])
        pred_bbs = classical_svm_predict(grid[train][bbs], y[train][bbs], grid[test])
        f1_us.append(minority_scores(y[test], pred_us, 1)[0])
        f1_bbs.append(minority_scores(y[test], pred_bbs, 1)[0])
    assert float(np.mean(n_bbs_min)) >= float(np.mean(n_us_min))
    assert float(np.mean(f1_bbs)) >= float(np.mean(f1_us)) - 1e-9
    assert max(f1_bbs) > 0.0


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="extra qml (qiskit) não instalado")
def test_fidelity_kernel_when_qiskit_present() -> None:
    from qml.circuits import feature_map_depth, fidelity_kernel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    kernel = fidelity_kernel(X, reps=1, entanglement="linear")
    assert kernel.shape == (8, 8)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-6)
    assert np.all(kernel >= -1e-9)
    assert feature_map_depth(2, reps=1) >= 1
