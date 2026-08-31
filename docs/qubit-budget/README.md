# Simulator sweep (versioned)

`qubit_budget.csv` is the full $q$-sweep used in the article (not gitignored,
unlike `d2-qubit-budget/results/`).

```bash
python3 d2-qubit-budget/qml/run_qubit_budget.py
python3 d2-qubit-budget/qml/run_qubit_budget.py --from-csv \
  d2-qubit-budget/docs/qubit-budget/qubit_budget.csv
```

Figures: `docs/figures/qubit-budget/` (English labels).
