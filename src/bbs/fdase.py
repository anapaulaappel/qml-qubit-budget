"""FD-ASE: seleção de atributos pela dimensão fractal de correlação (Sousa et al., 2002)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from bbs.grid import correlation_fractal_dimension


class FDASE:
    """Estimador de significância de atributos baseado em dimensão fractal.

    Remove constantes (pD ≈ 0), inclui atributos enquanto a dimensão fractal
    parcial cresce, identifica grupos de correlação e devolve a união das
    bases de correlação — o conjunto Es usado no BBS-Significant.

    Parameters
    ----------
    eps
        Variação de pD considerada nula (atributo constante ou correlacionado).
    n_levels
        Níveis de box-count para estimar D2.
    method
        ``forward_greedy`` escolhe a cada passo o atributo que mais aumenta pD
        (adequado a Es sem ordem canônica). ``original`` percorre os atributos
        na ordem das colunas, como no artigo de Sousa et al.
    """

    def __init__(
        self,
        eps: float = 0.15,
        n_levels: int = 8,
        method: str = "forward_greedy",
    ) -> None:
        if method not in {"forward_greedy", "original"}:
            raise ValueError("method deve ser 'forward_greedy' ou 'original'.")
        self.eps = float(eps)
        self.n_levels = int(n_levels)
        self.method = method
        self.embedding_dimension_: int | None = None
        self.intrinsic_dimension_: float | None = None
        self.selected_: NDArray[np.int64] | None = None
        self.correlation_groups_: list[list[int]] | None = None
        self.correlation_bases_: list[list[int]] | None = None

    def _pD(self, X: NDArray[np.floating], cols: Sequence[int]) -> float:
        if not cols:
            return 0.0
        return correlation_fractal_dimension(X[:, list(cols)], n_levels=self.n_levels)

    def _drop_constants(self, X: NDArray[np.floating], attrs: list[int]) -> list[int]:
        kept: list[int] = []
        for a in attrs:
            if self._pD(X, [a]) > self.eps:
                kept.append(a)
        return kept

    def _correlation_base(self, X: NDArray[np.floating], group: list[int]) -> list[int]:
        base = list(group)
        if len(base) <= 1:
            return base
        d_group = self._pD(X, base)
        changed = True
        while changed and len(base) > 1:
            changed = False
            for attr in list(base):
                trial = [x for x in base if x != attr]
                if abs(self._pD(X, trial) - d_group) <= self.eps:
                    base = trial
                    changed = True
                    break
        return base

    def _fit_greedy(self, X: NDArray[np.floating], remaining: list[int], d_full: float) -> None:
        selected: list[int] = []
        groups: list[list[int]] = []
        bases: list[list[int]] = []
        pD = 0.0
        leftover = list(remaining)
        while leftover and pD < d_full - self.eps:
            best_attr = leftover[0]
            best_pD = -1.0
            for attr in leftover:
                trial_pD = self._pD(X, selected + [attr])
                if trial_pD > best_pD + 1e-12:
                    best_pD = trial_pD
                    best_attr = attr
            gain = best_pD - pD
            if gain <= self.eps:
                correlated = [best_attr]
                leftover.remove(best_attr)
                still = list(leftover)
                for attr in still:
                    extra = self._pD(X, selected + correlated + [attr])
                    if extra - best_pD <= self.eps:
                        correlated.append(attr)
                        leftover.remove(attr)
                group = selected[-1:] + correlated if selected else correlated
                groups.append(group)
                base = self._correlation_base(X, group)
                bases.append(base)
                for attr in group:
                    if attr not in selected:
                        selected.append(attr)
                pD = self._pD(X, selected)
                continue
            leftover.remove(best_attr)
            selected.append(best_attr)
            pD = best_pD
        if not groups and selected:
            groups = [list(selected)]
            bases = [self._correlation_base(X, selected)]
            selected = list(dict.fromkeys(a for b in bases for a in b))
        else:
            selected = list(dict.fromkeys(a for b in bases for a in b)) if bases else selected
        self.selected_ = np.asarray(selected, dtype=np.int64)
        self.correlation_groups_ = groups
        self.correlation_bases_ = bases

    def _fit_original(self, X: NDArray[np.floating], remaining: list[int], d_full: float) -> None:
        groups: list[list[int]] = []
        bases: list[list[int]] = []
        leftover = list(remaining)
        while leftover:
            current = [leftover[0]]
            pD_prev = self._pD(X, current)
            found_correlation = False
            for attr in leftover[1:]:
                pD = self._pD(X, current + [attr])
                current.append(attr)
                if pD - pD_prev <= self.eps:
                    found_correlation = True
                    break
                pD_prev = pD
                if pD >= d_full - self.eps:
                    break
            group = list(current)
            if found_correlation and len(group) >= 2:
                last = group[-1]
                for attr in list(group[:-1]):
                    without = [x for x in group if x != attr]
                    rest = [x for x in without if x != last]
                    if not rest:
                        continue
                    p_rest = self._pD(X, rest)
                    p_without = self._pD(X, without)
                    p_group = self._pD(X, group)
                    if abs(p_without - p_rest) <= self.eps < (p_group - p_rest):
                        group.remove(attr)
                p_group = self._pD(X, group)
                for attr in leftover:
                    if attr in group:
                        continue
                    extra = self._pD(X, group + [attr])
                    if extra - p_group <= self.eps:
                        group.append(attr)
                        p_group = extra
            base = self._correlation_base(X, group)
            groups.append(group)
            bases.append(base)
            redundant = set(group) - set(base)
            leftover = [a for a in leftover if a not in group]
            leftover = [a for a in leftover if a not in redundant]
            selected_so_far = [i for b in bases for i in b]
            if selected_so_far and self._pD(X, selected_so_far) >= d_full - self.eps:
                break
        selected = list(dict.fromkeys(a for b in bases for a in b))
        self.selected_ = np.asarray(selected, dtype=np.int64)
        self.correlation_groups_ = groups
        self.correlation_bases_ = bases

    def fit(self, X: NDArray[np.floating]) -> FDASE:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X deve ter forma (n_amostras, n_atributos).")
        e = X.shape[1]
        self.embedding_dimension_ = e
        d_full = correlation_fractal_dimension(X, n_levels=self.n_levels)
        self.intrinsic_dimension_ = d_full
        remaining = self._drop_constants(X, list(range(e)))
        if not remaining:
            self.selected_ = np.zeros(0, dtype=np.int64)
            self.correlation_groups_ = []
            self.correlation_bases_ = []
            return self
        if self.method == "forward_greedy":
            self._fit_greedy(X, remaining, d_full)
        else:
            self._fit_original(X, remaining, d_full)
        if self.selected_ is None or self.selected_.size == 0:
            self.selected_ = np.asarray(remaining[: max(1, int(np.ceil(d_full)))], dtype=np.int64)
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.selected_ is None:
            raise RuntimeError("Chame fit() antes de transform().")
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.selected_]

    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.fit(X).transform(X)
