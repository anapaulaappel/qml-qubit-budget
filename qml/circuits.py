"""Feature maps e kernel de fidelidade no simulador de estado (Qiskit opcional)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Never

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, state_fidelity

    try:
        from qiskit.circuit.library import zz_feature_map as _make_zz_feature_map
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap as _ZZFeatureMap

        def _make_zz_feature_map(
            feature_dimension: int,
            reps: int = 1,
            entanglement: str = "linear",
            **kwargs: object,
        ) -> Any:
            return _ZZFeatureMap(
                feature_dimension=feature_dimension,
                reps=reps,
                entanglement=entanglement,
            )

    try:
        from qiskit.circuit.library import z_feature_map as _make_z_feature_map
    except ImportError:
        from qiskit.circuit.library import ZFeatureMap as _ZFeatureMap

        def _make_z_feature_map(
            feature_dimension: int,
            reps: int = 1,
            **kwargs: object,
        ) -> Any:
            return _ZFeatureMap(feature_dimension=feature_dimension, reps=reps)

    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    Statevector = None
    state_fidelity = None

    def _make_zz_feature_map(*args: object, **kwargs: object) -> Any:
        raise ImportError("Qiskit não está instalado.")

    def _make_z_feature_map(*args: object, **kwargs: object) -> Any:
        raise ImportError("Qiskit não está instalado.")

    QISKIT_AVAILABLE = False

EncodingKind = Literal["zz", "z", "iqp", "dense_angle", "reuploading"]
ENTANGLEMENT_KINDS = ("linear", "full", "none")
ENCODING_KINDS: tuple[EncodingKind, ...] = ("zz", "z", "iqp", "dense_angle", "reuploading")


def require_qiskit() -> None:
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit não está instalado. "
            'Use: pip install -e ".[qml]" ou passe --classical-only.'
        )


def scale_angles(X: NDArray[np.floating], bandwidth: float = 1.0) -> NDArray[np.float64]:
    """Escala cada atributo para [0, cπ], com c o bandwidth do mapa de ângulo.

    O valor c=1 é a faixa usual do ZZFeatureMap. c<1 encolhe as rotações
    (retarda concentração); c>1 alarga-as. Cada coluna é equalizada
    independentemente, por isso um eixo de ruído entra tão largo quanto um
    eixo de sinal.
    """
    X = np.asarray(X, dtype=np.float64)
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth deve ser um real positivo finito.")
    if X.shape[0] == 0:
        return X
    hi = float(bandwidth) * float(np.pi)
    scaler = MinMaxScaler(feature_range=(0.0, hi))
    return np.asarray(scaler.fit_transform(X), dtype=np.float64)


def zz_feature_map(q: int, reps: int = 1, entanglement: str = "linear") -> Any:
    require_qiskit()
    if q < 2:
        raise ValueError("ZZFeatureMap exige pelo menos 2 qubits.")
    return _make_zz_feature_map(
        feature_dimension=int(q),
        reps=int(reps),
        entanglement=entanglement,
    )


def z_feature_map(q: int, reps: int = 1) -> Any:
    require_qiskit()
    if q < 2:
        raise ValueError("ZFeatureMap exige pelo menos 2 qubits.")
    return _make_z_feature_map(feature_dimension=int(q), reps=int(reps))


def feature_map_depth(q: int, reps: int = 1, entanglement: str = "linear") -> int:
    """Profundidade do feature map de ângulo após ``decompose``."""
    fm = zz_feature_map(q, reps=reps, entanglement=entanglement)
    decomposed = fm.decompose()
    return int(decomposed.depth())


def amplitude_prep_stats(n_features: int) -> tuple[int, int]:
    """Qubits e profundidade de um ``initialize`` denso (apêndice, não o caminho principal)."""
    require_qiskit()
    q = int(np.ceil(np.log2(max(int(n_features), 2))))
    dim = 1 << q
    vec = np.ones(dim, dtype=np.float64)
    vec /= np.linalg.norm(vec)
    qc = QuantumCircuit(q)
    qc.initialize(vec.tolist(), list(range(q)))
    return q, int(qc.decompose().depth())


def _as_encoding(kind: str) -> EncodingKind:
    if kind not in ENCODING_KINDS:
        raise ValueError(f"encoding deve ser um de {ENCODING_KINDS}, recebido {kind!r}.")
    return kind  # type: ignore[return-value]


def _pad_features(x: Sequence[float] | NDArray[np.floating], width: int) -> NDArray[np.float64]:
    vec = np.asarray(x, dtype=np.float64).ravel()
    if width <= 0:
        raise ValueError("width deve ser positivo.")
    if vec.size >= width:
        return vec[:width]
    out = np.zeros(width, dtype=np.float64)
    out[: vec.size] = vec
    return out


def _apply_entanglement(qc: Any, n_qubits: int, entanglement: str) -> None:
    if entanglement == "none":
        return
    if entanglement == "linear":
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)
        return
    if entanglement == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.cz(i, j)
        return
    raise ValueError(f"entanglement deve ser um de {ENTANGLEMENT_KINDS}, recebido {entanglement!r}.")


def _apply_entanglement_iqp(
    qc: Any,
    n_qubits: int,
    features: NDArray[np.float64],
    entanglement: str,
) -> None:
    """Fases controladas dos produtos x_i x_j (IQP diagonal)."""
    if entanglement == "none":
        return
    if entanglement == "linear":
        for i in range(n_qubits - 1):
            qc.cp(float(features[i] * features[i + 1]), i, i + 1)
        return
    if entanglement == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.cp(float(features[i] * features[j]), i, j)
        return
    raise ValueError(f"entanglement deve ser um de {ENTANGLEMENT_KINDS}, recebido {entanglement!r}.")


def packed_layers(n_features: int, n_qubits: int, encoding: str, reps: int) -> int:
    """Quantas camadas o mapa usa para empilhar ``n_features`` em ``n_qubits``."""
    kind = _as_encoding(encoding)
    q = max(int(n_qubits), 1)
    feats = max(int(n_features), 1)
    if kind == "zz":
        return max(int(reps), 1)
    if kind == "z":
        return max(int(reps), 1)
    if kind == "iqp":
        return max(int(reps), 1)
    if kind == "dense_angle":
        need = int(np.ceil(feats / (2 * q)))
        return max(int(reps), need, 1)
    if kind == "reuploading":
        need = int(np.ceil(feats / q))
        return max(int(reps), need, 1)
    unused: Never = kind
    raise ValueError(f"encoding desconhecido: {unused!r}.")


def encode_circuit(
    x: Sequence[float] | NDArray[np.floating],
    n_qubits: int,
    *,
    encoding: str = "zz",
    reps: int = 1,
    entanglement: str = "linear",
) -> Any:
    """Prepara |ψ(x)⟩ em ``n_qubits``. Dense/re-uploading empilham vários atributos por qubit."""
    require_qiskit()
    kind = _as_encoding(encoding)
    q = int(n_qubits)
    if q < 2:
        raise ValueError("O feature map exige pelo menos 2 qubits.")
    vec = np.asarray(x, dtype=np.float64).ravel()
    if kind == "zz":
        feats = _pad_features(vec, q)
        fm = zz_feature_map(q, reps=reps, entanglement=entanglement)
        return fm.assign_parameters(feats.tolist())
    if kind == "z":
        feats = _pad_features(vec, q)
        fm = z_feature_map(q, reps=reps)
        return fm.assign_parameters(feats.tolist())
    qc = QuantumCircuit(q)
    if kind == "iqp":
        padded = _pad_features(vec, q)
        for i in range(q):
            qc.h(i)
        for _ in range(max(int(reps), 1)):
            for i in range(q):
                qc.rz(float(padded[i]), i)
            _apply_entanglement_iqp(qc, q, padded, entanglement)
        return qc
    if kind == "dense_angle":
        layers = packed_layers(int(vec.size), q, kind, reps)
        width = 2 * q * layers
        padded = _pad_features(vec, width)
        cursor = 0
        for _ in range(layers):
            for i in range(q):
                qc.ry(float(padded[cursor]), i)
                qc.rz(float(padded[cursor + 1]), i)
                cursor += 2
            _apply_entanglement(qc, q, entanglement)
        return qc
    if kind == "reuploading":
        layers = packed_layers(int(vec.size), q, kind, reps)
        padded = _pad_features(vec, q * layers)
        cursor = 0
        for _ in range(layers):
            for i in range(q):
                qc.ry(float(padded[cursor + i]), i)
            cursor += q
            _apply_entanglement(qc, q, entanglement)
        return qc
    unused: Never = kind
    raise ValueError(f"encoding desconhecido: {unused!r}.")


def encode_circuit_depth(
    n_qubits: int,
    n_features: int,
    *,
    encoding: str = "zz",
    reps: int = 1,
    entanglement: str = "linear",
) -> int:
    dummy = np.zeros(max(int(n_features), 1), dtype=np.float64)
    qc = encode_circuit(
        dummy,
        n_qubits,
        encoding=encoding,
        reps=reps,
        entanglement=entanglement,
    )
    decomposed = qc.decompose() if hasattr(qc, "decompose") else qc
    return int(decomposed.depth())


def _states(
    X: NDArray[np.floating],
    reps: int,
    entanglement: str,
    encoding: str = "zz",
    n_qubits: int | None = None,
    bandwidth: float = 1.0,
) -> list[Any]:
    require_qiskit()
    Xs = scale_angles(X, bandwidth=bandwidth)
    q = int(n_qubits) if n_qubits is not None else int(Xs.shape[1])
    kind = _as_encoding(encoding)
    if kind == "zz" and n_qubits is None:
        fm = zz_feature_map(q, reps=reps, entanglement=entanglement)
        return [Statevector.from_instruction(fm.assign_parameters(row.tolist())) for row in Xs]
    if kind == "z" and n_qubits is None:
        fm = z_feature_map(q, reps=reps)
        return [Statevector.from_instruction(fm.assign_parameters(row.tolist())) for row in Xs]
    return [
        Statevector.from_instruction(
            encode_circuit(row, q, encoding=kind, reps=reps, entanglement=entanglement)
        )
        for row in Xs
    ]


def fidelity_kernel(
    X: NDArray[np.floating],
    Y: NDArray[np.floating] | None = None,
    reps: int = 1,
    entanglement: str = "linear",
    encoding: str = "zz",
    n_qubits: int | None = None,
    bandwidth: float = 1.0,
) -> NDArray[np.float64]:
    """Kernel de fidelidade |⟨ψ(x)|ψ(y)⟩|² via statevector (simulador exato)."""
    require_qiskit()
    X = np.asarray(X, dtype=np.float64)
    kind = _as_encoding(encoding)
    states_x = _states(
        X,
        reps=reps,
        entanglement=entanglement,
        encoding=kind,
        n_qubits=n_qubits,
        bandwidth=bandwidth,
    )
    if Y is None:
        n = len(states_x)
        kernel = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                value = float(state_fidelity(states_x[i], states_x[j]))
                kernel[i, j] = value
                kernel[j, i] = value
        return kernel
    Y = np.asarray(Y, dtype=np.float64)
    states_y = _states(
        Y,
        reps=reps,
        entanglement=entanglement,
        encoding=kind,
        n_qubits=n_qubits,
        bandwidth=bandwidth,
    )
    kernel = np.empty((len(states_x), len(states_y)), dtype=np.float64)
    for i, sx in enumerate(states_x):
        for j, sy in enumerate(states_y):
            kernel[i, j] = float(state_fidelity(sx, sy))
    return kernel


def kernel_concentration(kernel: NDArray[np.floating]) -> dict[str, float]:
    """Média e dispersão dos off-diagonais; média → 0.5 sugere colapso em similaridade."""
    kernel = np.asarray(kernel, dtype=np.float64)
    n = kernel.shape[0]
    if n < 2:
        return {"mean_offdiag": 0.0, "std_offdiag": 0.0, "collapse_to_half": 0.5}
    off = kernel[~np.eye(n, dtype=bool)]
    mean = float(off.mean())
    std = float(off.std())
    return {
        "mean_offdiag": mean,
        "std_offdiag": std,
        "collapse_to_half": float(abs(mean - 0.5)),
    }


def near_far_fidelity(
    X: NDArray[np.floating],
    kernel: NDArray[np.floating],
    n_pairs: int = 20,
) -> tuple[float, float]:
    """Fidelidade média de pares próximos vs. distantes em ``X`` (euclidiano)."""
    X = np.asarray(X, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    n = X.shape[0]
    if n < 4:
        return 0.0, 0.0
    dist = euclidean_distances(X)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu])
    take = max(1, min(int(n_pairs), order.size // 2))
    near = order[:take]
    far = order[-take:]
    k_vals = kernel[iu]
    return float(k_vals[near].mean()), float(k_vals[far].mean())
