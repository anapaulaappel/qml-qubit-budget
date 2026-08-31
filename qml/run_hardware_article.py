#!/usr/bin/env python3
"""Suite de hardware para o artigo: blobs q=7 e breast PCA q=3 vs q=7.

Grava kernels e metadados em d2-qubit-budget/docs/hardware/ (versionável).
O run q=5 n=8 já feito é arquivado a partir da matriz medida.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "d2-qubit-budget"))

from qml.circuits import QISKIT_AVAILABLE, fidelity_kernel, kernel_concentration
from qml.data import load_breast
from qml.features import project_to_q, select_feature_view
from qml.hardware import (
    backend_queue,
    connect_service,
    kernel_from_pair_probs,
    pair_circuits,
    run_sampler_job,
    token_is_configured,
    zero_probability,
)
from qml.hardware_io import hardware_dir, save_run
from qml.run_hardware import _points, _print_queue


def _archive_blobs_q5() -> None:
    """Primeiro job (ibm_fez, 2026-08-30): kernels já medidos, X reproduzível."""
    X = _points(8, 5, random_state=0)
    k_sv = fidelity_kernel(X, reps=1, entanglement="linear")
    k_hw = np.array(
        [
            [1.0, 0.086, 0.004, 0.125, 0.02, 0.016, 0.027, 0.031],
            [0.086, 1.0, 0.004, 0.0, 0.008, 0.012, 0.008, 0.008],
            [0.004, 0.004, 1.0, 0.004, 0.652, 0.367, 0.512, 0.5],
            [0.125, 0.0, 0.004, 1.0, 0.023, 0.02, 0.02, 0.027],
            [0.02, 0.008, 0.652, 0.023, 1.0, 0.676, 0.711, 0.789],
            [0.016, 0.012, 0.367, 0.02, 0.676, 1.0, 0.637, 0.695],
            [0.027, 0.008, 0.512, 0.02, 0.711, 0.637, 1.0, 0.57],
            [0.031, 0.008, 0.5, 0.027, 0.789, 0.695, 0.57, 1.0],
        ],
        dtype=np.float64,
    )
    save_run(
        tag="blobs_q5_n8",
        dataset="blobs",
        backend="ibm_fez",
        q=5,
        n=8,
        shots=256,
        elapsed_s=48.0,
        X=X,
        k_hw=k_hw,
        k_sv=k_sv,
        extra={"note": "First live job; wall-clock ~48s; archived from measured kernel."},
    )
    print("Arquivado blobs_q5_n8", flush=True)


def _breast_take(n: int, random_state: int = 0) -> np.ndarray:
    """Os mesmos 8 pontos do job PCA, para comparar vistas no dispositivo."""
    data = load_breast()
    rng = np.random.default_rng(random_state + 17)
    take = rng.choice(data.X.shape[0], size=int(n), replace=False)
    take.sort()
    return take


def _breast_pca(n: int, q: int, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = load_breast()
    xp = project_to_q(data.X, q, method="pca", random_state=random_state)
    take = _breast_take(n, random_state)
    return np.asarray(xp[take], dtype=np.float64), np.asarray(data.y[take])


def _breast_columns(n: int, method: str, random_state: int = 0) -> tuple[np.ndarray, int, list[int]]:
    data = load_breast()
    view = select_feature_view(data.X, method=method, q_max=8, random_state=random_state)
    take = _breast_take(n, random_state)
    cols = [] if view.columns is None else [int(c) for c in view.columns.tolist()]
    return np.asarray(view.X[take], dtype=np.float64), int(view.q), cols


def _ablation_last_alive(dataset: str, encoding: str, reps: int) -> int | None:
    """Último q vivo na ablação PCA (simulador). None se o CSV ainda não existe."""
    path = ROOT / "d2-qubit-budget" / "docs" / "encodings" / "encoding_ablation.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    alive = [
        int(rec["q"])
        for rec in records
        if rec["dataset"] == dataset
        and rec["encoding"] == encoding
        and int(rec["reps"]) == reps
        and str(rec["kernel_alive"]).strip().lower() in {"true", "1", "yes"}
    ]
    return max(alive) if alive else None


def _knee_follows_ceiling(last_alive: int | None, ceiling: int, q_max: int = 8) -> bool:
    """O mapa morre perto do teto fractal, não vive até q_max."""
    if last_alive is None:
        return False
    return last_alive <= ceiling + 1 and last_alive < q_max


def _submit(
    backend: object,
    X: np.ndarray,
    *,
    tag: str,
    dataset: str,
    q: int,
    shots: int,
    encoding: str = "zz",
    reps: int = 1,
    extra: dict[str, object] | None = None,
) -> None:
    n = int(X.shape[0])
    print(
        f"→ {tag}: q={q} n={n} {encoding} reps={reps} circuitos={n * (n - 1) // 2}",
        flush=True,
    )
    t0 = time.perf_counter()
    circuits = pair_circuits(X, reps=reps, entanglement="linear", encoding=encoding)
    counts_list = run_sampler_job(circuits, backend=backend, shots=shots)
    probs = [zero_probability(c, shots) for c in counts_list]
    k_hw = kernel_from_pair_probs(n, probs)
    k_sv = fidelity_kernel(X, reps=reps, entanglement="linear", encoding=encoding)
    elapsed = time.perf_counter() - t0
    path = save_run(
        tag=tag,
        dataset=dataset,
        backend=str(getattr(backend, "name", backend)),
        q=q,
        n=n,
        shots=shots,
        elapsed_s=elapsed,
        X=X,
        k_hw=k_hw,
        k_sv=k_sv,
        extra=extra,
    )
    conc_hw = kernel_concentration(k_hw)
    conc_sv = kernel_concentration(k_sv)
    mae = float(np.mean(np.abs(k_hw - k_sv)))
    print(
        f"  MAE={mae:.3f}  offdiag hw={conc_hw['mean_offdiag']:.3f} "
        f"sv={conc_sv['mean_offdiag']:.3f}  {elapsed:.1f}s  {path.name}",
        flush=True,
    )


def main() -> None:
    if not QISKIT_AVAILABLE:
        raise SystemExit("Qiskit ausente.")
    if (hardware_dir() / "blobs_q5_n8.npz").is_file():
        print("blobs_q5_n8 já em docs/hardware/, a saltar arquivo.", flush=True)
    else:
        _archive_blobs_q5()
    x_fdase, q_fdase, cols_fdase = _breast_columns(8, "fdase")
    x_rand, q_rand, cols_rand = _breast_columns(8, "random")
    print(
        f"FD-ASE q={q_fdase} cols={cols_fdase}; random q={q_rand} cols={cols_rand}",
        flush=True,
    )
    jobs: list[tuple[str, str, np.ndarray, int, dict[str, object] | None]] = [
        ("blobs_q7_n8", "blobs", _points(8, 7), 7, None),
        ("breast_pca_q3_n8", "breast_pca", _breast_pca(8, 3)[0], 3, None),
        ("breast_pca_q7_n8", "breast_pca", _breast_pca(8, 7)[0], 7, None),
        (
            f"breast_fdase_q{q_fdase}_n8",
            "breast_fdase",
            x_fdase,
            q_fdase,
            {"columns": cols_fdase, "note": "FD-ASE original columns; same 8 rows as PCA jobs."},
        ),
        (
            f"breast_random_q{q_rand}_n8",
            "breast_random",
            x_rand,
            q_rand,
            {"columns": cols_rand, "note": "Random original columns; same 8 rows as PCA jobs."},
        ),
    ]
    x_pca3 = _breast_pca(8, 3)[0]
    x_pca7 = _breast_pca(8, 7)[0]
    map_candidates = (
        (
            "z",
            1,
            "breast_z_q3_n8",
            "breast_z_q7_n8",
            "Z feature map (product states); same 8 PCA rows as ZZ jobs.",
        ),
        (
            "zz",
            2,
            "breast_zz2_q3_n8",
            "breast_zz2_q7_n8",
            "ZZ map, two layers; same 8 PCA rows as ZZ reps=1 jobs.",
        ),
        (
            "iqp",
            1,
            "breast_iqp_q3_n8",
            "breast_iqp_q7_n8",
            "IQP linear CP; same 8 PCA rows. Only if the simulator knee tracks D2.",
        ),
    )
    for encoding, reps, tag_lo, tag_hi, note in map_candidates:
        last = _ablation_last_alive("breast_cancer", encoding, reps)
        follows = _knee_follows_ceiling(last, ceiling=3)
        print(
            f"Abla\u00e7\u00e3o {encoding} reps={reps}: last_alive={last} "
            f"{'joelho ok \u2192 IBM' if follows else 'sem joelho no teto; skip IBM'}",
            flush=True,
        )
        if not follows:
            continue
        jobs.append(
            (
                tag_lo,
                f"breast_{encoding}",
                x_pca3,
                3,
                {"encoding": encoding, "reps": reps, "note": note},
            )
        )
        jobs.append(
            (
                tag_hi,
                f"breast_{encoding}",
                x_pca7,
                7,
                {"encoding": encoding, "reps": reps, "note": note},
            )
        )
    n16 = 16
    take16 = _breast_take(n16, random_state=0)
    x_pca3_n16, _ = _breast_pca(n16, 3)
    x_fdase_n16, q_fdase_n16, cols_fdase_n16 = _breast_columns(n16, "fdase")
    x_rand_n16, q_rand_n16, cols_rand_n16 = _breast_columns(n16, "random")
    print(
        f"n=16 views: PCA q=3, FD-ASE q={q_fdase_n16} cols={cols_fdase_n16}, "
        f"random q={q_rand_n16} cols={cols_rand_n16}",
        flush=True,
    )
    jobs.extend([
        (
            "breast_pca_q3_n16",
            "breast_pca",
            x_pca3_n16,
            3,
            {"note": "PCA q=3, n=16. Near/far feasible at this sample size."},
        ),
        (
            f"breast_fdase_q{q_fdase_n16}_n16",
            "breast_fdase",
            x_fdase_n16,
            q_fdase_n16,
            {
                "columns": cols_fdase_n16,
                "note": "FD-ASE original columns, n=16. Key test for selector claim.",
            },
        ),
        (
            f"breast_random_q{q_rand_n16}_n16",
            "breast_random",
            x_rand_n16,
            q_rand_n16,
            {
                "columns": cols_rand_n16,
                "note": "Random original columns, n=16. Baseline for selector comparison.",
            },
        ),
    ])
    pending = [job for job in jobs if not (hardware_dir() / f"{job[0]}.npz").is_file()]
    if not pending:
        print("Suite já completa. Ficheiros em d2-qubit-budget/docs/hardware/", flush=True)
        return
    if not token_is_configured():
        raise SystemExit("Sem credenciais IBM.")
    print("A ligar…", flush=True)
    service = connect_service()
    _print_queue(backend_queue(service, min_qubits=2))
    backend = service.backend("ibm_fez")
    print(f"Backend: {backend.name}", flush=True)
    shots = 256
    for tag, dataset, X, q, extra in pending:
        enc = str((extra or {}).get("encoding", "zz"))
        rep = int((extra or {}).get("reps", 1))
        _submit(
            backend, X,
            tag=tag, dataset=dataset, q=q, shots=shots,
            encoding=enc, reps=rep, extra=extra,
        )
    print("Suite concluída. Ficheiros em d2-qubit-budget/docs/hardware/", flush=True)


if __name__ == "__main__":
    main()
