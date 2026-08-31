"""Kernel de fidelidade por compute-uncompute, pronto para hardware IBM.

A chave **não** entra em ficheiros nem na linha de comandos. Só variáveis de ambiente:

    QISKIT_IBM_TOKEN     API key (IBM Cloud / Quantum Platform)
    QISKIT_IBM_INSTANCE  CRN da instância (recomendado)
    QISKIT_IBM_CHANNEL   default: ibm_quantum_platform
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qml.circuits import QISKIT_AVAILABLE, encode_circuit, require_qiskit, scale_angles

try:
    from qiskit import QuantumCircuit
except ImportError:
    QuantumCircuit = None  # type: ignore[misc, assignment]


def token_is_configured() -> bool:
    if os.environ.get("QISKIT_IBM_TOKEN", "").strip():
        return True
    saved = Path.home() / ".qiskit" / "qiskit-ibm.json"
    return saved.is_file()


def compute_uncompute_circuit(
    x: Sequence[float],
    y: Sequence[float],
    reps: int = 1,
    entanglement: str = "linear",
    encoding: str = "zz",
) -> Any:
    """Circuito U(y)† U(x)|0⟩ com medição: P(0…0) ≈ |⟨ψ(y)|ψ(x)⟩|²."""
    require_qiskit()
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim != 1 or xa.size < 2:
        raise ValueError("x e y devem ser vetores 1-D com pelo menos 2 entradas.")
    q = int(xa.size)
    ux = encode_circuit(xa, q, encoding=encoding, reps=reps, entanglement=entanglement)
    uy = encode_circuit(ya, q, encoding=encoding, reps=reps, entanglement=entanglement)
    qc = QuantumCircuit(q, q)
    qc.compose(ux, inplace=True)
    qc.compose(uy.inverse(), inplace=True)
    qc.measure(range(q), range(q))
    return qc


def pair_circuits(
    X: NDArray[np.floating],
    reps: int = 1,
    entanglement: str = "linear",
    encoding: str = "zz",
) -> list[Any]:
    """Um circuito por par i<j (mais a diagonal, omitida: fidelidade 1)."""
    Xs = scale_angles(X)
    n = Xs.shape[0]
    circuits: list[Any] = []
    for i in range(n):
        for j in range(i + 1, n):
            circuits.append(
                compute_uncompute_circuit(
                    Xs[i],
                    Xs[j],
                    reps=reps,
                    entanglement=entanglement,
                    encoding=encoding,
                )
            )
    return circuits


def zero_probability(counts: dict[str, int], shots: int) -> float:
    if shots <= 0:
        return 0.0
    zeros = 0
    for bitstring, freq in counts.items():
        bits = bitstring.replace(" ", "")
        if bits and set(bits) <= {"0"}:
            zeros += int(freq)
    return float(zeros / shots)


def kernel_from_pair_probs(n: int, offdiag: Sequence[float]) -> NDArray[np.float64]:
    kernel = np.eye(n, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            value = float(np.clip(offdiag[k], 0.0, 1.0))
            kernel[i, j] = value
            kernel[j, i] = value
            k += 1
    if k != len(offdiag):
        raise ValueError("Número de pares não coincide com n(n-1)/2.")
    return kernel


def connect_service() -> Any:
    """Liga ao IBM Quantum Platform. Não imprime nem grava a chave."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError(
            "qiskit-ibm-runtime ausente. pip install 'qiskit-ibm-runtime'"
        ) from exc
    token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    if token:
        kwargs: dict[str, str] = {
            "channel": os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform").strip()
            or "ibm_quantum_platform",
            "token": token,
        }
        instance = os.environ.get("QISKIT_IBM_INSTANCE", "").strip()
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs)
    try:
        return QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "Sem QISKIT_IBM_TOKEN e sem conta gravada em ~/.qiskit. "
            "export QISKIT_IBM_TOKEN='…' no terminal (não no chat)."
        ) from exc


def backend_queue(service: Any, min_qubits: int = 2) -> list[dict[str, object]]:
    """Fila e tamanho de cada QPU operacional. Não gasta tempo de hardware."""
    rows: list[dict[str, object]] = []
    backends = service.backends(min_num_qubits=min_qubits, simulator=False)
    for backend in backends:
        try:
            status = backend.status()
        except Exception:
            continue
        operational = bool(getattr(status, "operational", False))
        pending = int(getattr(status, "pending_jobs", -1))
        msg = str(getattr(status, "status_msg", ""))
        qubits = int(getattr(backend, "num_qubits", 0) or 0)
        rows.append(
            {
                "name": str(backend.name),
                "qubits": qubits,
                "operational": operational,
                "pending_jobs": pending,
                "status": msg,
            }
        )
    rows.sort(key=lambda r: (not r["operational"], int(r["pending_jobs"]), -int(r["qubits"])))
    return rows


def list_backends(service: Any, min_qubits: int = 2) -> list[str]:
    return [
        str(row["name"])
        for row in backend_queue(service, min_qubits=min_qubits)
        if row["operational"]
    ]


def _counts_from_pub(pub_result: Any) -> dict[str, int]:
    data = pub_result.data
    for attr in ("c", "meas", "measure"):
        if hasattr(data, attr):
            container = getattr(data, attr)
            if hasattr(container, "get_counts"):
                return dict(container.get_counts())
    if hasattr(data, "get_counts"):
        return dict(data.get_counts())
    raise RuntimeError("Não foi possível ler counts do resultado do Sampler.")


def run_sampler_job(
    circuits: list[Any],
    *,
    backend: Any,
    shots: int,
) -> list[dict[str, int]]:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = [pm.run(qc) for qc in circuits]
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa, shots=int(shots))
    result = job.result()
    return [_counts_from_pub(result[i]) for i in range(len(circuits))]
