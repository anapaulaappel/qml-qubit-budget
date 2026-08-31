#!/usr/bin/env python3
"""Smoke test de kernel de fidelidade em hardware IBM.

    python3 d2-qubit-budget/qml/run_hardware.py --probe
    python3 d2-qubit-budget/qml/run_hardware.py --q 5 --n 8 --shots 256 --backend NAME

--probe lista fila e qubits (sem gastar QPU). --q/--n definem o tamanho do job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel
from qml.hardware import (
    backend_queue,
    connect_service,
    kernel_from_pair_probs,
    pair_circuits,
    run_sampler_job,
    token_is_configured,
    zero_probability,
)

SIZE_HINTS = (
    (3, 6),
    (5, 8),
    (7, 8),
    (7, 12),
)


def _points(n: int, q: int, random_state: int = 0) -> np.ndarray:
    """Dois blobs em q dimensões (teste um pouco maior que 2 qubits)."""
    rng = np.random.default_rng(random_state)
    n = max(4, int(n))
    q = max(2, int(q))
    n_a = n // 2
    n_b = n - n_a
    a = rng.normal(loc=0.15, scale=0.04, size=(n_a, q))
    b = rng.normal(loc=0.85, scale=0.04, size=(n_b, q))
    return np.clip(np.vstack((a, b)), 0.0, 1.0)


def _pair_count(n: int) -> int:
    return n * (n - 1) // 2


def _print_queue(rows: list[dict[str, object]]) -> None:
    print(
        f"{'backend':22s} {'qubits':>6s} {'fila':>6s} {'ok':>4s}  status",
        flush=True,
    )
    for row in rows:
        ok = "yes" if row["operational"] else "no"
        print(
            f"{str(row['name']):22s} {int(row['qubits']):6d} "
            f"{int(row['pending_jobs']):6d} {ok:>4s}  {row['status']}",
            flush=True,
        )


def _print_size_hints() -> None:
    print("\nTamanho do teste (circuitos = n(n-1)/2; um job Sampler com todos os pares):", flush=True)
    print(f"{'q':>4s} {'n':>4s} {'circuitos':>10s}", flush=True)
    for q, n in SIZE_HINTS:
        print(f"{q:4d} {n:4d} {_pair_count(n):10d}", flush=True)
    print(
        "Com 10 min de crédito: q=5 n=8 (28 circuitos) é o compromisso usual. "
        "q=7 n=12 (66) só se a fila estiver vazia.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Mostra fila e qubits, sem submeter.")
    parser.add_argument("--backend", default="", help="Nome do backend; vazio = menor fila.")
    parser.add_argument("--q", type=int, default=2, help="Qubits (= atributos no ZZ map).")
    parser.add_argument("--n", type=int, default=4, help="Pontos no kernel.")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit ausente. pip install -e '.[qml]'")

    n_circ = _pair_count(max(4, args.n))
    print(
        f"Pedido: q={args.q} n={args.n} → {n_circ} circuitos compute-uncompute, shots={args.shots}",
        flush=True,
    )

    if args.dry_run:
        X = _points(args.n, args.q)
        circuits = pair_circuits(X, reps=1, entanglement="linear")
        print(circuits[0])
        print(f"dry-run: {len(circuits)} circuitos, nada submetido.", flush=True)
        return

    if not token_is_configured():
        raise SystemExit(
            "Sem token no ambiente e sem ~/.qiskit/qiskit-ibm.json. "
            "export QISKIT_IBM_TOKEN no terminal (não no chat)."
        )

    print("A ligar ao IBM Quantum Platform…", flush=True)
    service = connect_service()
    rows = backend_queue(service, min_qubits=2)
    _print_queue(rows)
    _print_size_hints()
    live = [r for r in rows if r["operational"]]
    if args.probe:
        if live:
            best = live[0]
            print(
                f"\nMenor fila: {best['name']}  ({best['qubits']} qubits, "
                f"{best['pending_jobs']} jobs). "
                f"Exemplo: --backend {best['name']} --q 5 --n 8 --shots 256",
                flush=True,
            )
        print("probe: sem job submetido.", flush=True)
        return
    if not live:
        raise SystemExit("Nenhuma QPU operacional.")

    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=max(2, args.q))
    print(f"Backend: {backend.name}  qubits={backend.num_qubits}", flush=True)
    X = _points(args.n, args.q)
    circuits = pair_circuits(X, reps=1, entanglement="linear")
    print(f"A submeter {len(circuits)} circuitos (pode esperar na fila)…", flush=True)
    counts_list = run_sampler_job(circuits, backend=backend, shots=args.shots)
    probs = [zero_probability(c, args.shots) for c in counts_list]
    k_hw = kernel_from_pair_probs(X.shape[0], probs)
    k_sv = fidelity_kernel(X, reps=1, entanglement="linear")
    mae = float(np.mean(np.abs(k_hw - k_sv)))
    print("Kernel hardware (P(0…0)):", np.round(k_hw, 3), sep="\n", flush=True)
    print("Kernel statevector:", np.round(k_sv, 3), sep="\n", flush=True)
    print(f"MAE vs statevector: {mae:.3f}", flush=True)


if __name__ == "__main__":
    main()
