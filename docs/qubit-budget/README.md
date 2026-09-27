# Simulator sweep (versioned)

`qubit_budget.csv` is the full $q$-sweep used in the article (not gitignored,
unlike `d2-qubit-budget/results/`).

```bash
python3 d2-qubit-budget/qml/run_qubit_budget.py
python3 d2-qubit-budget/qml/run_qubit_budget.py --from-csv \
  d2-qubit-budget/docs/qubit-budget/qubit_budget.csv
```

Figures: `docs/figures/qubit-budget/` (English labels).

Also recorded from the same simulator stack:

- `oblique_id.csv` — Haar-rotated $k$-cube (`qml/run_oblique.py`)
- `effective_rank.csv` — effective rank of $K$ on breast and moons (`qml/run_effective_rank.py`)
- `downstream_n128.csv` — fidelity $k$NN and precomputed SVM F1 at $n=128$ (`qml/run_downstream.py`)
