#!/usr/bin/env python3
"""Figuras e CSV dos kernels IBM para o artigo (lê docs/hardware/*.npz)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "docs" / "hardware"
FIG_DIR = HARDWARE / "figures"
SUMMARY = HARDWARE / "summary.csv"

TITLES = {
    "blobs_q5_n8": "Blobs, $q=5$, $n=8$",
    "blobs_q7_n8": "Blobs, $q=7$, $n=8$",
    "breast_pca_q3_n8": r"Breast PCA, $q=3=\lceil D_2\rceil$, $n=8$",
    "breast_pca_q7_n8": "Breast PCA, $q=7$ (beyond ceiling), $n=8$",
    "breast_fdase_q4_n8": "Breast FD-ASE, $q=4$, $n=8$",
    "breast_random_q3_n8": "Breast random axes, $q=3$, $n=8$",
    "breast_z_q3_n8": r"Breast PCA, Z map, $q=3$, $n=8$",
    "breast_z_q7_n8": r"Breast PCA, Z map, $q=7$, $n=8$",
    "breast_zz2_q3_n8": r"Breast PCA, ZZ two layers, $q=3$, $n=8$",
    "breast_zz2_q7_n8": r"Breast PCA, ZZ two layers, $q=7$, $n=8$",
    "breast_iqp_q3_n8": r"Breast PCA, IQP, $q=3$, $n=8$",
    "breast_iqp_q7_n8": r"Breast PCA, IQP, $q=7$, $n=8$",
    "breast_pca_q3_n16": r"Breast PCA, $q=3=\lceil D_2\rceil$, $n=16$",
    "breast_fdase_q4_n16": "Breast FD-ASE, $q=4$, $n=16$",
    "breast_random_q3_n16": "Breast random axes, $q=3$, $n=16$",
}


def _load_json(tag: str) -> dict[str, object]:
    return json.loads((HARDWARE / f"{tag}.json").read_text(encoding="utf-8"))


def _export_csv_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


def _plot_kernels(tag: str) -> None:
    packed = np.load(HARDWARE / f"{tag}.npz")
    k_hw = np.asarray(packed["k_hw"], dtype=np.float64)
    k_sv = np.asarray(packed["k_sv"], dtype=np.float64)
    x = np.asarray(packed["X"], dtype=np.float64)
    meta = _load_json(tag)
    diff = np.abs(k_hw - k_sv)
    title = TITLES.get(tag, tag)
    mae = float(meta["mae"])
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.35))
    panels = (
        (k_hw, "IBM hardware $K$", 0.0, 1.0, "viridis"),
        (k_sv, "Statevector $K$", 0.0, 1.0, "viridis"),
        (diff, rf"$|K_{{\mathrm{{hw}}}}-K_{{\mathrm{{sv}}}}|$  MAE={mae:.3f}", 0.0, max(float(diff.max()), 0.05), "magma"),
    )
    for ax, (mat, panel_title, vmin, vmax, cmap) in zip(axes, panels, strict=True):
        image = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(panel_title, fontsize=10)
        ax.set_xticks(range(mat.shape[0]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xlabel("point $j$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("point $i$")
    fig.suptitle(
        f"{title}  ·  {meta['backend']}  ·  {meta['shots']} shots  ·  "
        f"{meta['n_circuits']} circuits",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{tag}_kernels.png", dpi=160)
    plt.close(fig)
    _export_csv_matrix(HARDWARE / f"{tag}_k_hw.csv", k_hw)
    _export_csv_matrix(HARDWARE / f"{tag}_k_sv.csv", k_sv)
    _export_csv_matrix(HARDWARE / f"{tag}_X.csv", x)


def _plot_mae_bars(tags: list[str]) -> None:
    labels: list[str] = []
    mae: list[float] = []
    off_hw: list[float] = []
    off_sv: list[float] = []
    for tag in tags:
        meta = _load_json(tag)
        raw = str(meta["tag"]).replace("_n8", "").replace("_n16", " n16").replace("_", "\n")
        labels.append(raw)
        mae.append(float(meta["mae"]))
        off_hw.append(float(meta["mean_offdiag_hw"]))
        off_sv.append(float(meta["mean_offdiag_sv"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    x = np.arange(len(labels))
    axes[0].bar(x, mae, color="#3b6ea5")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("MAE (hardware vs statevector)")
    axes[0].set_title("Kernel reconstruction error")
    axes[0].grid(True, axis="y", alpha=0.3)
    width = 0.38
    axes[1].bar(x - width / 2, off_hw, width, label="hardware", color="#3b6ea5")
    axes[1].bar(x + width / 2, off_sv, width, label="statevector", color="#c47b2b")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylabel("mean off-diagonal $K$")
    axes[1].set_title("Concentration (lower = more collapsed)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "hardware_mae_concentration.png", dpi=160)
    plt.close(fig)


def _write_markdown_table() -> None:
    if not SUMMARY.is_file():
        return
    with SUMMARY.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    lines = [
        "# IBM hardware kernels (ibm_fez, 256 shots, 2026-08-30)",
        "",
        "Compute–uncompute fidelity kernel $P(0\\ldots0)\\approx|\\langle\\psi_y|\\psi_x\\rangle|^2$.",
        "ZZ feature map, linear entanglement, `reps=1`. Matrices: `*_k_hw.csv`, `*_k_sv.csv`, `*.npz`.",
        "",
        "| tag | dataset | $q$ | $n$ | circuits | wall (s) | MAE vs SV | mean $K$ HW | mean $K$ SV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['tag']}` | {row['dataset']} | {row['q']} | {row['n']} | "
            f"{row['n_circuits']} | {float(row['elapsed_s']):.1f} | "
            f"{float(row['mae']):.3f} | {float(row['mean_offdiag_hw']):.3f} | "
            f"{float(row['mean_offdiag_sv']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading for the paper",
            "",
            "- **Breast $q=3=\\lceil D_2\\rceil$ (PCA):** hardware tracks the exact kernel (MAE $0.021$).",
            "- **Breast $q=7$ (PCA):** both hardware and statevector have collapsed.",
            "- **Breast FD-ASE $q=4$ (columns 27, 21, 9, 8):** same 8 rows; MAE $0.012$ (closest to SV).",
            "- **Breast random $q=3$ (columns 15, 18, 23):** same 8 rows; MAE $0.030$. Mean $K$ is higher, not collapsed; $n=8$ is too small for the near/far *alive* rule (PCA $q=3$ also fails it on this subsample).",
            "- **Blobs $q=5\\to 7$:** off-diagonals drop as $q$ grows, with MAE $0.077$ then $0.054$.",
            "",
            "Figures: `figures/`.",
            "",
        ]
    )
    (HARDWARE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tags = sorted(p.stem for p in HARDWARE.glob("*.npz"))
    if not tags:
        raise SystemExit(f"Nenhum npz em {HARDWARE}")
    for tag in tags:
        _plot_kernels(tag)
        print(f"figura {tag}_kernels.png", flush=True)
    _plot_mae_bars(tags)
    _write_markdown_table()
    print(f"Figuras em {FIG_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
