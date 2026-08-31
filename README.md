# D2 as a qubit budget / fractal features for QML

\(D_2\) answers *how many* qubits; FD-ASE answers *which* original attributes. Comparisons are **quantum vs quantum** (same fidelity kernel, different feature views) — not QSVM vs SVM.

Notes: [docs/d2-qubit-budget.md](docs/d2-qubit-budget.md) (Portuguese), [docs/d2-qubit-budget.en.md](docs/d2-qubit-budget.en.md) (English). The idea of density-biased *row* sampling for QML is parked in [BBS-em-quantum.md](../BBS-em-quantum.md) at the repo root; it is not evaluated here.

## Layout

| Path | Role |
|------|------|
| `qml/` | Feature maps, kernels, tasks, loaders, CLI |
| `tests/` | Ceiling benchmark and encoding pipeline (Qiskit optional) |
| `docs/` | Article (PT/EN) and figures |
| `results/` | CSV and plots from the last run (gitignored) |

## Run

From the repository root:

```bash
python3 -m pip install -e ".[dev,qml]"
python3 d2-qubit-budget/qml/run_qubit_budget.py
python3 d2-qubit-budget/qml/run_qubit_budget.py --fast
python3 d2-qubit-budget/qml/encode_compare.py --dataset breast
python3 d2-qubit-budget/qml/run_packed_encoding.py
python3 -m pytest -q d2-qubit-budget/tests
```

Article artefacts (versioned, not gitignored):

| Path | Role |
|------|------|
| `docs/qubit-budget/qubit_budget.csv` | Simulator $q$-sweep |
| `docs/figures/qubit-budget/` | English plots |
| `docs/encodings/` | Dense-angle / re-uploading |
| `docs/hardware/` | IBM kernels (`*.npz`, CSV, figures) |
| `docs/paper/d2-qubit-budget.tex` | LNCS-style draft |

Hardware IBM (suite for the paper; credentials **only** in the environment, never in the repo):

```bash
python3 -m pip install -e ".[qml,ibm]"
export QISKIT_IBM_TOKEN='…'          # no teu terminal, não no chat
export QISKIT_IBM_INSTANCE='crn:…'  # se a plataforma pedir CRN
python3 d2-qubit-budget/qml/run_hardware.py --probe
python3 d2-qubit-budget/qml/run_hardware.py --backend ibm_fez --q 5 --n 8 --shots 256
python3 d2-qubit-budget/qml/run_hardware_article.py
python3 d2-qubit-budget/qml/plot_hardware.py
```
