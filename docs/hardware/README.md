# IBM hardware kernels (ibm_fez, 256 shots, 2026-08-30)

Compute–uncompute fidelity kernel $P(0\ldots0)\approx|\langle\psi_y|\psi_x\rangle|^2$.
ZZ feature map, linear entanglement, `reps=1`. Matrices: `*_k_hw.csv`, `*_k_sv.csv`, `*.npz`.

| tag | dataset | $q$ | $n$ | circuits | wall (s) | MAE vs SV | mean $K$ HW | mean $K$ SV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `blobs_q5_n8` | blobs | 5 | 8 | 28 | 48.0 | 0.077 | 0.234 | 0.315 |
| `blobs_q7_n8` | blobs | 7 | 8 | 28 | 37.7 | 0.054 | 0.142 | 0.202 |
| `breast_pca_q3_n8` | breast_pca | 3 | 8 | 28 | 37.4 | 0.021 | 0.082 | 0.081 |
| `breast_pca_q7_n8` | breast_pca | 7 | 8 | 28 | 36.3 | 0.003 | 0.004 | 0.003 |
| `breast_fdase_q4_n8` | breast_fdase | 4 | 8 | 28 | 40.8 | 0.012 | 0.065 | 0.068 |
| `breast_random_q3_n8` | breast_random | 3 | 8 | 28 | 36.4 | 0.030 | 0.164 | 0.166 |
| `breast_pca_q3_n16` | breast_pca | 3 | 16 | 120 | 137.3 | 0.027 | 0.118 | 0.115 |
| `breast_fdase_q4_n16` | breast_fdase | 4 | 16 | 120 | 130.1 | 0.023 | 0.063 | 0.061 |
| `breast_random_q3_n16` | breast_random | 3 | 16 | 120 | 143.8 | 0.034 | 0.133 | 0.131 |

## Reading for the paper

- **Breast $q=3=\lceil D_2\rceil$ (PCA):** hardware tracks the exact kernel (MAE $0.021$).
- **Breast $q=7$ (PCA):** both hardware and statevector have collapsed.
- **Breast FD-ASE $q=4$ (columns 27, 21, 9, 8):** same 8 rows; MAE $0.012$ (closest to SV).
- **Breast random $q=3$ (columns 15, 18, 23):** same 8 rows; MAE $0.030$. Mean $K$ is higher, not collapsed; $n=8$ is too small for the near/far *alive* rule (PCA $q=3$ also fails it on this subsample).
- **Blobs $q=5\to 7$:** off-diagonals drop as $q$ grows, with MAE $0.077$ then $0.054$.

Figures: `figures/`.
