# Packed encodings (simulator, $n=32$)

Same fidelity kernel; features are PCA coordinates packed onto $q=\lceil D_2\rceil$ qubits unless noted. Teal in `packed_encoding.png` marks the operational *alive* rule.

| data set | encoding | $q$ | features | layers | mean $K$ | alive? |
|---|---|---:|---:|---:|---:|---|
| breast | ZZ 1:1 | 3 | 3 | 1 | 0.139 | yes |
| breast | ZZ 1:1 | 7 | 7 | 1 | 0.008 | no |
| breast | dense-angle | 3 | 3 | 1 | 0.585 | yes |
| breast | dense-angle | 3 | 6 | 1 | 0.360 | yes |
| breast | re-uploading | 3 | 6 | 2 | 0.305 | yes |
| breast | re-uploading | 3 | 12 | 4 | 0.165 | no |
| moons | ZZ 1:1 | 2 | 2 | 1 | 0.246 | yes |
| moons | ZZ 1:1 | 7 | 7 | 1 | 0.009 | no |
| moons | dense-angle | 2 | 4 | 1 | 0.373 | yes |
| moons | re-uploading | 2 | 16 | 8 | 0.266 | no (near $<$ far) |
| pendigits | ZZ 1:1 | 6 | 6 | 1 | 0.023 | no (near $0.249$) |
| pendigits | dense-angle | 6 | 12 | 1 | 0.073 | yes |
| pendigits | re-uploading | 6 | 16 | 3 | 0.028 | no |

## Reading for the paper

- $\lceil D_2\rceil$ budgets **qubits** (Hilbert-space width), not a 1-attribute-per-qubit law. Dense-angle and data re-uploading keep a live kernel at the fractal $q$ while packing two coordinates per qubit.
- Packing toward PCA-95% onto that same $q$ (breast 12 PCs on 3 qubits; moons 16 PCs on 2) **kills or inverts** the near/far geometry. Extra ambient axes are still junk, even when they share a qubit.
- CLI: `python3 d2-qubit-budget/qml/run_packed_encoding.py`
