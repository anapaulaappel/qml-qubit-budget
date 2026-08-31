"""Benchmark do teto de qubits (D2) — tarefas sobre kernel, Qiskit opcional."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qml.data import (
    display_name,
    load_diabetes,
    load_intrinsic2,
    load_intrinsic_k,
    load_iris,
    load_moons,
    load_pendigits_qml,
)
from qml.features import d2_qubit_ceiling, project_to_q
from qml.id_estimators import pca_variance_qubits, two_nn_dimension
from qml.kernel_tasks import (
    kernel_is_alive,
    kernel_target_alignment,
    near_far_ratio,
    rbf_kernel_matrix,
    score_kernel,
)
from qml.qubit_budget import last_alive_q, run_dataset_sweep


def test_d2_ceiling() -> None:
    assert d2_qubit_ceiling(2.26) == 3
    assert d2_qubit_ceiling(0.0) == 2
    assert d2_qubit_ceiling(float("nan")) == 2


def test_project_to_q_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 10))
    for method in ("pca", "random", "prefix"):
        Xq = project_to_q(X, 3, method=method, random_state=0)
        assert Xq.shape == (40, 3)
    with pytest.raises(ValueError):
        project_to_q(X, 3, method="nope")


def test_alive_and_alignment_on_toy_kernels() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    k_id = np.eye(6)
    assert kernel_target_alignment(k_id, y) >= 0.0
    assert near_far_ratio(0.9, 0.1) == pytest.approx(9.0)
    assert kernel_is_alive(0.8, 0.1, 0.4) is True
    assert kernel_is_alive(0.02, 0.01, 0.01) is False
    ones = np.ones((6, 6))
    np.fill_diagonal(ones, 1.0)
    assert kernel_is_alive(0.99, 0.99, 0.99) is False
    assert kernel_is_alive(0.20, 0.05, 0.10, near_floor=0.15) is True
    assert kernel_is_alive(0.20, 0.05, 0.10, near_floor=0.25) is False


def test_score_kernel_rbf_without_qiskit() -> None:
    data = load_intrinsic2(n=60, n_features=8, random_state=0)
    Xq = project_to_q(data.X, 2, method="pca", random_state=0)
    kernel = rbf_kernel_matrix(Xq)
    idx = np.arange(data.X.shape[0])
    train, test = idx[:42], idx[42:]
    target = Xq[:, 0]
    scores = score_kernel(
        kernel,
        Xq,
        data.y,
        train,
        test,
        minority_label=data.minority_label,
        majority_label=0,
        regression_target=target,
    )
    assert 0.0 <= scores.knn_acc <= 1.0
    assert scores.near_far_ratio >= 0.0


def test_rbf_sweep_marks_ceiling() -> None:
    data = load_intrinsic2(n=80, n_features=12, random_state=1)
    rows = run_dataset_sweep(
        data,
        n_kernel=24,
        q_max=4,
        views=("pca",),
        families=("rbf",),
        n_levels=6,
        random_state=0,
    )
    assert rows
    assert rows[0].d2_ceiling >= 2
    assert rows[0].pca95 >= 1
    assert rows[0].twonn >= 0.0
    assert rows[0].twonn_ceiling >= 2
    qs = [r.q for r in rows if r.view == "pca" and r.family == "rbf"]
    assert qs == sorted(set(qs))
    alive = last_alive_q(rows, "pca", "rbf")
    assert alive is None or alive >= 2


def test_two_nn_recovers_plane_and_line() -> None:
    rng = np.random.default_rng(1)
    plane = rng.normal(size=(400, 2))
    dim_plane = two_nn_dimension(plane)
    assert 1.4 < dim_plane < 2.7
    line = rng.uniform(0.0, 10.0, size=(400, 1))
    dim_line = two_nn_dimension(line)
    assert 0.6 < dim_line < 1.6
    assert dim_line < dim_plane


def test_pca95_on_low_rank() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(250, 2))
    mix = rng.normal(size=(2, 12))
    X = latent @ mix
    q = pca_variance_qubits(X, threshold=0.95)
    assert 1 <= q <= 4


def test_display_name_and_intrinsic_k() -> None:
    assert display_name("onebig_tiny") == "onebig"
    assert display_name("breast_cancer") == "breast"
    assert display_name("moons") == "moons"
    data = load_intrinsic_k(3, n=80, n_features=12, random_state=0)
    assert data.X.shape == (80, 12)
    assert data.name == "intrinsic_k3"
    assert np.allclose(data.X[:, 3:], 0.0)


def test_scale_angles_bandwidth() -> None:
    from qml.circuits import scale_angles

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    s1 = scale_angles(X, bandwidth=1.0)
    s05 = scale_angles(X, bandwidth=0.5)
    assert s1.max() == pytest.approx(np.pi, rel=1e-5)
    assert s05.max() == pytest.approx(0.5 * np.pi, rel=1e-5)
    np.testing.assert_allclose(s05, 0.5 * s1, atol=1e-10)
    with pytest.raises(ValueError):
        scale_angles(X, bandwidth=0.0)


def test_new_tabular_loaders() -> None:
    iris = load_iris()
    assert iris.X.shape == (150, 4)
    assert iris.y.min() >= 0
    diabetes = load_diabetes()
    assert diabetes.X.shape[1] == 10
    assert set(np.unique(diabetes.y)) <= {0, 1}
    moons = load_moons(n=80, n_features=12, random_state=0)
    assert moons.X.shape == (80, 12)


def test_pendigits_loader_if_present() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    if not ((data_dir / "pendigits.tra").exists() and (data_dir / "pendigits.tes").exists()):
        pytest.skip("Pendigits não está em data/")
    data = load_pendigits_qml(max_rows=120, random_state=0)
    assert data.X.shape[0] == 120
    assert data.X.shape[1] == 16


def test_compute_uncompute_circuit_without_ibm() -> None:
    from qml.circuits import QISKIT_AVAILABLE

    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit ausente")
    from qml.hardware import compute_uncompute_circuit, kernel_from_pair_probs, zero_probability

    qc = compute_uncompute_circuit([0.2, 0.3], [0.2, 0.3])
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    qc_z = compute_uncompute_circuit([0.2, 0.3], [0.4, 0.1], encoding="z")
    qc_iqp = compute_uncompute_circuit([0.2, 0.3], [0.4, 0.1], encoding="iqp")
    qc_zz2 = compute_uncompute_circuit([0.2, 0.3], [0.4, 0.1], encoding="zz", reps=2)
    assert qc_z.num_qubits == qc_iqp.num_qubits == qc_zz2.num_qubits == 2
    assert zero_probability({"00": 90, "01": 10}, 100) == pytest.approx(0.9)
    k = kernel_from_pair_probs(3, [0.8, 0.1, 0.2])
    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(1.0)
    assert k[0, 1] == pytest.approx(0.8)


def test_packed_encodings_kernel_shape() -> None:
    from qml.circuits import QISKIT_AVAILABLE, encode_circuit, fidelity_kernel, packed_layers

    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit ausente")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 6))
    assert packed_layers(6, 3, "dense_angle", 1) == 1
    assert packed_layers(6, 3, "reuploading", 1) == 2
    qc = encode_circuit(X[0], 3, encoding="dense_angle")
    assert qc.num_qubits == 3
    k_dense = fidelity_kernel(X, encoding="dense_angle", n_qubits=3)
    k_reup = fidelity_kernel(X, encoding="reuploading", n_qubits=3)
    assert k_dense.shape == (6, 6)
    assert k_reup.shape == (6, 6)
    np.testing.assert_allclose(np.diag(k_dense), 1.0, atol=1e-8)
    np.testing.assert_allclose(np.diag(k_reup), 1.0, atol=1e-8)
    k_z = fidelity_kernel(X[:, :3], encoding="z")
    k_iqp = fidelity_kernel(X[:, :3], encoding="iqp")
    k_zz2 = fidelity_kernel(X[:, :3], encoding="zz", reps=2)
    assert k_z.shape == k_iqp.shape == k_zz2.shape == (6, 6)
    np.testing.assert_allclose(np.diag(k_z), 1.0, atol=1e-8)
    np.testing.assert_allclose(np.diag(k_iqp), 1.0, atol=1e-8)
    qc_z = encode_circuit(X[0, :3], 3, encoding="z")
    qc_iqp = encode_circuit(X[0, :3], 3, encoding="iqp")
    assert qc_z.num_qubits == 3
    assert qc_iqp.num_qubits == 3
    with pytest.raises(ValueError):
        fidelity_kernel(X, encoding="nope", n_qubits=3)
