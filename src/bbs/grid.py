"""Grade multi-resolução esparsa (LiBOC) e estimador da dimensão fractal D2.

A estrutura só materializa células ocupadas, para não alocar 2^E filhos.
Cada nível j corresponde a células de lado r = 1/2^j.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def normalize_unit_cube(X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Normaliza cada atributo para o cubo unitário [0, 1]."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X deve ter forma (n_amostras, n_atributos).")
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi - lo < 1e-15, 1.0, hi - lo)
    scaled = (X - lo) / span
    return np.clip(scaled, 0.0, 1.0 - 1e-15)


def cell_bins(x: NDArray[np.floating], level: int) -> tuple[int, ...]:
    """Índice da célula no nível ``level`` (``2**level`` faixas por dimensão)."""
    n_bins = 1 << level
    bins = np.minimum(np.floor(x * n_bins).astype(np.int64), n_bins - 1)
    return tuple(int(b) for b in bins)


def child_code(x: NDArray[np.floating], level: int) -> int:
    """Código do octante relativo ao pai: bit d vale 1 se x[d] está na metade alta.

    No nível ``level`` o espaço já está dividido em ``2**(level-1)`` faixas;
    o bit extra é o bit menos significativo do índice nesse nível.
    """
    bins = np.minimum(np.floor(x * (1 << level)).astype(np.int64), (1 << level) - 1)
    bits = bins & 1
    code = 0
    for d, bit in enumerate(bits):
        if bit:
            code |= 1 << d
    return int(code)


@dataclass
class GridNode:
    """Nó da MG-tree: ocupância, filhos esparsos e, se folha, índices dos pontos."""

    level: int
    occupancy: int = 0
    children: dict[int, GridNode] = field(default_factory=dict)
    indices: list[int] | None = None

    def is_leaf(self) -> bool:
        return not self.children and self.indices is not None


class MultiResolutionGrid:
    """Árvore de grade multi-resolução usada pelo LiBOC e pelo BBSCT."""

    def __init__(self, n_levels: int = 5) -> None:
        if n_levels < 2:
            raise ValueError("n_levels deve ser pelo menos 2.")
        self.n_levels = n_levels
        self.root = GridNode(level=0)
        self.n_points = 0
        self.n_features = 0

    def insert_all(self, X: NDArray[np.floating], store_indices: bool = True) -> None:
        """Insere todos os pontos. Índices só nas folhas do nível ``n_levels - 1``."""
        X = np.asarray(X, dtype=np.float64)
        n, e = X.shape
        self.n_points = n
        self.n_features = e
        max_level = self.n_levels - 1
        self.root.occupancy = n
        for i in range(n):
            node = self.root
            point = X[i]
            for level in range(1, self.n_levels):
                code = child_code(point, level)
                child = node.children.get(code)
                if child is None:
                    child = GridNode(level=level)
                    node.children[code] = child
                child.occupancy += 1
                if store_indices and level == max_level:
                    if child.indices is None:
                        child.indices = []
                    child.indices.append(i)
                node = child

    def occupancy_squares_by_level(self) -> list[float]:
        """S(r) = soma dos C_i^2 em cada nível 0 .. n_levels-1."""
        totals = [0.0] * self.n_levels

        def walk(node: GridNode) -> None:
            totals[node.level] += float(node.occupancy) ** 2
            for child in node.children.values():
                walk(child)

        walk(self.root)
        return totals

    def condense(self, delta: float) -> None:
        """BBSJL: se C < delta, concatena listas dos descendentes e o nó vira folha."""

        def collect_indices(node: GridNode) -> list[int]:
            if node.indices is not None:
                return list(node.indices)
            collected: list[int] = []
            for child in node.children.values():
                collected.extend(collect_indices(child))
            return collected

        def walk(node: GridNode) -> None:
            for child in list(node.children.values()):
                walk(child)
            if node.level == 0:
                return
            if node.children and node.occupancy < delta:
                node.indices = collect_indices(node)
                node.children.clear()

        walk(self.root)

    def iter_leaves(self) -> list[GridNode]:
        """Folhas da árvore condensada (nível > 0 com lista de índices)."""
        leaves: list[GridNode] = []

        def walk(node: GridNode) -> None:
            if node.indices is not None:
                leaves.append(node)
                return
            for child in node.children.values():
                walk(child)

        walk(self.root)
        return leaves


def correlation_fractal_dimension(
    X: NDArray[np.floating],
    n_levels: int = 10,
    min_level: int = 0,
    occupancy_floor: float = 1.5,
) -> float:
    """Estima D2 como a inclinação de log S(r) versus log r (box-count LiBOC).

    S(r) = soma das ocupâncias ao quadrado nas células de lado r = 1/2^j.
    Escalas em que cada ponto cai numa célula só (S ≈ N) são descartadas;
    o ajuste usa a faixa auto-similar, como no gráfico log-log do LiBOC.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] < 2 or X.shape[1] == 0:
        return 0.0

    grid = MultiResolutionGrid(n_levels=n_levels)
    grid.insert_all(normalize_unit_cube(X), store_indices=False)
    s_values = grid.occupancy_squares_by_level()
    n = float(X.shape[0])
    floor = occupancy_floor * n

    log_r: list[float] = []
    log_s: list[float] = []
    last = min(n_levels, len(s_values))
    for level in range(min_level, last):
        s = s_values[level]
        if s <= 0:
            continue
        if level > 0 and s <= floor:
            continue
        r = 0.5**level
        log_r.append(float(np.log(r)) if r > 0 else 0.0)
        log_s.append(float(np.log(s)))

    if len(log_r) < 2:
        log_r = []
        log_s = []
        for level in range(min_level, last):
            s = s_values[level]
            if s <= 0:
                continue
            r = 0.5**level
            log_r.append(float(np.log(r)) if r > 0 else 0.0)
            log_s.append(float(np.log(s)))
    if len(log_r) < 2:
        return 0.0
    slope = float(np.polyfit(np.asarray(log_r), np.asarray(log_s), 1)[0])
    return float(np.clip(slope, 0.0, float(X.shape[1])))
