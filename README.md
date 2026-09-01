# Fractal dimension as a qubit budget for quantum kernels

This is **exploratory research**: a working hypothesis, the code that tests it, and the figures from those runs. It is not a finished method and not a production library.

Angle encoding maps one coordinate onto one qubit. On a table with $E$ recorded columns that usually means a circuit of width $E$, or of the PCA-95% width after a linear cut. Neither number is the dimension of the data: a circle in 20-D still has intrinsic dimension near 1, and a Swiss roll is a 2-D sheet even though PCA needs three components.

This repository asks a narrower question than “does a quantum kernel beat an SVM?”. For a **fixed** angle-encoded fidelity kernel, how many qubits should the feature map actually use, and which original attributes should those qubits see?

The answer used here is the **correlation fractal dimension** $D_2$.

## Why fractal dimension

A point cloud in $\mathbb{R}^E$ almost never fills the recorded cube. Intrinsic dimension (ID) is the number of degrees of freedom of the support. $D_2$ estimates that number from pair counts on a multi-scale grid: if occupancy scales as $S(r) \propto r^{D_2}$, the slope of $\log S$ versus $\log r$ is the dimension of the set (Grassberger & Procaccia; Belussi & Faloutsos).

Properties that matter for encoding:

- $D_2 \le E$, with equality only when the cloud fills ambient space.
- Redundant or tightly correlated columns barely raise $D_2$.
- The estimate can be non-integer (fractal supports).
- It is classical, unsupervised, and computed on thousands of rows — not on the tiny sample used to build a NISQ kernel.

$D_2$ says **how many** degrees of freedom the support has. It does not name coordinates. **FD-ASE** (fractal-dimension attribute-subset evaluation) grows a subset of the original columns until the partial dimension stops rising, and returns a set whose size is about $D_2$.

That is the opposite of PCA at encoding time: PCA returns linear mixtures of every axis; FD-ASE returns a subset of the recorded attributes. PCA is a variance cut, not an ID estimator.

## What this code is for

The working hypothesis is that an angle-encoded kernel **collapses** when the circuit is wider than the data: off-diagonal fidelities flatten, near/far structure disappears, and downstream tasks (QSVM, fidelity $k$NN, spectral clustering, one-class SVM, kernel ridge) inherit a near-diagonal Gram matrix.

The proposed budget is

$$q^\ast = \lceil D_2 \rceil$$

with a small floor (two qubits). Comparisons are **quantum vs quantum**: the same fidelity kernel, different feature views (PCA, random columns, prefix axes, FD-ASE), not QSVM vs SVM.

The default map is a one-layer linear $ZZ$ circuit with angles in $[0, c\pi]$. Bandwidth $c$ is part of the claim: at $c = 1$ the kernel is tight on low-ID tables; shrinking $c$ moves the collapse to larger $q$. A kernel is called **alive** when near-neighbour fidelity, the near/far ratio, and the mean off-diagonal all stay above fixed floors. The **knee** is the last alive $q$.

Also included: a synthetic $k$-sweep (signal on $k$ axes, null padding), seed and sample-size checks, a dense-angle / re-uploading ablation, and optional IBM hardware kernels already measured (no credentials in the tree).

## Layout

| Path | Role |
|------|------|
| `qml/` | Feature maps, fidelity kernel, $q$-sweeps, FD-ASE views, CLIs |
| `src/` | Correlation fractal dimension $D_2$ and FD-ASE |
| `tests/` | Unit tests (Qiskit optional for most checks) |
| `docs/` | Notes (PT/EN), CSVs, figures, hardware Gram matrices |
| `results/` | Local run output (gitignored) |

The LaTeX article is **not** in this repository.

## Install and run

From the repository root:

```bash
python3 -m pip install -e ".[dev,qml]"
python3 qml/run_qubit_budget.py --fast
python3 qml/run_fdase_table.py
python3 qml/run_bandwidth.py
python3 qml/run_synthetic_id.py
python3 qml/run_robustness.py
python3 -m pytest -q tests
```

Optional IBM jobs read `QISKIT_IBM_TOKEN` (and `QISKIT_IBM_INSTANCE` if the platform asks for a CRN) from the **environment**. Do not put tokens in files or in this README. Recorded kernels live under `docs/hardware/`.

```bash
python3 -m pip install -e ".[qml,ibm]"
python3 qml/run_hardware.py --probe
```

## Recorded results

| Path | Role |
|------|------|
| `docs/qubit-budget/` | Simulator $q$-sweep, FD-ASE table, synthetic $k$, seeds, $n=128$ |
| `docs/figures/qubit-budget/` | English plots |
| `docs/encodings/` | Bandwidth and packed-encoding ablations |
| `docs/hardware/` | IBM kernels (`ibm_fez`, 256 shots) |

Longer notes: [docs/d2-qubit-budget.en.md](docs/d2-qubit-budget.en.md) (English), [docs/d2-qubit-budget.md](docs/d2-qubit-budget.md) (Portuguese).
