"""Biased Box Sampling (BBS): BBSCT, BBSJL, BBSES e recorte uniforme."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bbs.fdase import FDASE
from bbs.grid import MultiResolutionGrid, normalize_unit_cube


class BiasedBoxSampler:
    """Amostragem viesada por densidade via grade multi-resolução (PKDD 2007).

    Parameters
    ----------
    ratio
        Fração alvo da amostra, em (0, 1]. O artigo usa *Ratio* em percentual
        (1 significa 1%); aqui ``ratio=0.01`` equivale a 1%.
    n_levels
        Número de resoluções R (incluindo o cubo raiz). O artigo usa R = 5.
    random_state
        Semente ou ``np.random.Generator``.
    """

    def __init__(
        self,
        ratio: float = 0.01,
        n_levels: int = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio deve estar em (0, 1].")
        if n_levels < 2:
            raise ValueError("n_levels deve ser pelo menos 2.")
        self.ratio = float(ratio)
        self.n_levels = int(n_levels)
        self.random_state = random_state
        self.n_features_in_: int | None = None
        self.delta_: float | None = None
        self.grid_: MultiResolutionGrid | None = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _percent(self) -> float:
        return self.ratio * 100.0

    def fit(self, X: NDArray[np.floating], grid_X: NDArray[np.floating] | None = None) -> BiasedBoxSampler:
        """BBSCT + BBSJL: constrói e condensa a MG-tree.

        Se ``grid_X`` for dado (BBS-Significant), a árvore usa esses atributos
        e os índices ainda se referem às linhas de ``X``.
        """
        X = np.asarray(X, dtype=np.float64)
        coords = X if grid_X is None else np.asarray(grid_X, dtype=np.float64)
        if coords.shape[0] != X.shape[0]:
            raise ValueError("grid_X deve ter o mesmo número de linhas que X.")
        e_grid = int(coords.shape[1])
        self.n_features_in_ = e_grid
        percent = self._percent()
        self.delta_ = e_grid * 100.0 / (2.0 * percent)

        grid = MultiResolutionGrid(n_levels=self.n_levels)
        grid.insert_all(normalize_unit_cube(coords), store_indices=True)
        grid.condense(self.delta_)
        self.grid_ = grid
        return self

    def sample_indices(self, n_points: int | None = None) -> NDArray[np.int64]:
        """BBSES + recorte para ``n_points`` (padrão: N * ratio).

        Cada folha pede ``round(ratio * C)`` pontos, com o fator
        ``2 * (R - nivel)`` em folhas rasas (PKDD). Garante pelo menos um
        ponto por folha enquanto couber no tamanho alvo, para o viés não
        desaparecer no arredondamento nem no recorte.
        """
        if self.grid_ is None:
            raise RuntimeError("Chame fit() antes de sample_indices().")
        rng = self._rng()
        percent = self._percent()
        finest = self.n_levels - 1
        n_total = self.grid_.n_points
        target = int(round(n_total * self.ratio)) if n_points is None else int(n_points)
        target = max(1, min(target, n_total))

        leaves = [
            leaf
            for leaf in self.grid_.iter_leaves()
            if leaf.occupancy > 0 and leaf.indices
        ]
        if not leaves:
            return rng.choice(n_total, size=target, replace=False).astype(np.int64)

        weights: list[float] = []
        members: list[list[int]] = []
        for leaf in leaves:
            sample_size = percent * leaf.occupancy / 100.0
            if leaf.level != finest:
                sample_size *= 2.0 * (finest - leaf.level)
            weights.append(max(sample_size, 1.0))
            members.append(list(leaf.indices))

        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        picked: set[int] = set()

        n_leaves = len(members)
        if n_leaves >= target:
            chosen_leaves = rng.choice(n_leaves, size=target, replace=False, p=w)
            for li in chosen_leaves:
                idx = int(rng.choice(members[int(li)]))
                picked.add(idx)
        else:
            for li, pts in enumerate(members):
                picked.add(int(rng.choice(pts)))
            remaining_slots = target - len(picked)
            leftover: list[int] = []
            leftover_w: list[float] = []
            for li, pts in enumerate(members):
                unused = [p for p in pts if p not in picked]
                leftover.extend(unused)
                leftover_w.extend([weights[li]] * len(unused))
            if remaining_slots > 0 and leftover:
                lw = np.asarray(leftover_w, dtype=np.float64)
                lw = lw / lw.sum()
                take = min(remaining_slots, len(leftover))
                extra = rng.choice(len(leftover), size=take, replace=False, p=lw)
                for j in extra:
                    picked.add(int(leftover[int(j)]))

        if len(picked) < target:
            remaining = [i for i in range(n_total) if i not in picked]
            need = min(target - len(picked), len(remaining))
            if need:
                picked.update(int(i) for i in rng.choice(remaining, size=need, replace=False))

        return np.sort(np.asarray(list(picked)[:target], dtype=np.int64))

    def _point_leaf_stats(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Ocupância e nível da folha condensada de cada ponto (após ``fit``)."""
        if self.grid_ is None:
            raise RuntimeError("Chame fit() antes de _point_leaf_stats().")
        n_total = self.grid_.n_points
        occupancy = np.ones(n_total, dtype=np.float64)
        level = np.full(n_total, self.n_levels - 1, dtype=np.int64)
        for leaf in self.grid_.iter_leaves():
            if not leaf.indices:
                continue
            occ = float(leaf.occupancy)
            lvl = int(leaf.level)
            for i in leaf.indices:
                occupancy[i] = occ
                level[i] = lvl
        return occupancy, level

    def sample_indices_with_weights(
        self,
        n_points: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Mesmo recorte PKDD de ``sample_indices``, com peso por ponto.

        O peso é ocupação da folha dividida pelo fator de profundidade
        ``2*(R-ℓ)`` nas folhas rasas (1 no nível mais fino). Folhas ``C=1``
        (ruído) ficam com peso baixo, para o SVM/QSVM não tratar outlier como
        minoria. A amostragem em si **não** muda: é ``sample_indices``.
        Os pesos são reescalados para média 1.
        """
        idx = self.sample_indices(n_points=n_points)
        occupancy, level = self._point_leaf_stats()
        finest = self.n_levels - 1
        depth = np.ones(idx.size, dtype=np.float64)
        shallow = level[idx] != finest
        if np.any(shallow):
            depth[shallow] = 2.0 * (finest - level[idx][shallow])
        depth = np.maximum(depth, 1.0)
        weights = occupancy[idx] / depth
        mean_w = float(weights.mean()) if weights.size else 1.0
        if mean_w > 0.0 and np.isfinite(mean_w):
            weights = weights / mean_w
        return idx, weights.astype(np.float64)

    def fit_sample_with_weights(
        self,
        X: NDArray[np.floating],
        grid_X: NDArray[np.floating] | None = None,
        n_points: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """``fit`` + ``sample_indices_with_weights``."""
        self.fit(X, grid_X=grid_X)
        return self.sample_indices_with_weights(n_points=n_points)

    def fit_sample_significant_with_weights(
        self,
        X: NDArray[np.floating],
        fdase: FDASE | None = None,
        n_points: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], FDASE]:
        """BBS-Significant com pesos de folha; índices relativos a ``X``."""
        selector = fdase if fdase is not None else FDASE()
        if selector.selected_ is None:
            selector.fit(X)
        if selector.selected_ is None or selector.selected_.size == 0:
            idx, weights = self.fit_sample_with_weights(X, n_points=n_points)
            return idx, weights, selector
        idx, weights = self.fit_sample_with_weights(
            X, grid_X=X[:, selector.selected_], n_points=n_points
        )
        return idx, weights, selector

    def fit_sample_significant(
        self,
        X: NDArray[np.floating],
        fdase: FDASE | None = None,
    ) -> tuple[NDArray[np.int64], FDASE]:
        """BBS-Significant (SAC 2007): grade em Es, índices relativos a X completo."""
        selector = fdase if fdase is not None else FDASE()
        if selector.selected_ is None:
            selector.fit(X)
        if selector.selected_ is None or selector.selected_.size == 0:
            idx = self.fit_sample(X)
            return idx, selector
        idx = self.fit_sample(X, grid_X=X[:, selector.selected_])
        return idx, selector

    def fit_sample(
        self,
        X: NDArray[np.floating],
        grid_X: NDArray[np.floating] | None = None,
    ) -> NDArray[np.int64]:
        """Ajusta a árvore e devolve os índices da amostra."""
        self.fit(X, grid_X=grid_X)
        return self.sample_indices()
