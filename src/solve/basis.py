"""Базисные B-сплайны на логарифмически эквидистантной сетке узлов
и составные квадратуры Гаусса–Лежандра для интегрирования ядер.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline

from .kernels import dvol_dlnr_factor


def clamped_knots(r0: float, r1: float, degree: int, n_basis: int) -> np.ndarray:
    """Вектор узлов зажатого (clamped) B-сплайнового базиса.

    Базис степени ``degree`` из ``n_basis`` функций на [r0, r1];
    внутренние узлы расположены логарифмически эквидистантно.
    Количество различных внутренних узлов: n_basis - degree - 1.
    При p = 0 (например, степень 4 при M = 5 функциях) базис вырождается
    в полиномы Бернштейна степени ``degree`` на всем [r0, r1] — это
    корректный предельный случай.
    """
    p = n_basis - degree - 1
    if p < 0:
        raise ValueError("нужно n_basis >= degree + 1")
    interior = np.geomspace(r0, r1, p + 2)[1:-1]
    return np.concatenate(
        (np.full(degree + 1, r0), interior, np.full(degree + 1, r1))
    )


def bspline_basis(r0: float, r1: float, degree: int, n_basis: int,
                  interior_only: bool = False):
    """Список из ``n_basis`` B-сплайнов (объекты scipy BSpline, степени
    ``degree``; образуют разбиение единицы на [r0, r1]) и вектор узлов.

    При ``interior_only=True`` из базиса исключаются первая и последняя
    функции (B_0 и B_{n-1}): все оставшиеся функции равны нулю на обоих
    концах [r0, r1], т.е. любое разложение по ним автоматически
    удовлетворяет однородным граничным условиям n(r0) = n(r1) = 0.
    Вектор узлов при этом не меняется.
    """
    knots = clamped_knots(r0, r1, degree, n_basis)
    basis = [BSpline(knots, e, degree, extrapolate=False) for e in np.eye(n_basis)]
    if interior_only:
        if n_basis < 3:
            raise ValueError("граничные условия n(r0)=n(r1)=0 требуют n_basis >= 3")
        basis = basis[1:-1]
    return basis, knots


@lru_cache(maxsize=None)
def _leggauss(order: int):
    """Узлы и веса Гаусса–Лежандра порядка ``order`` на [-1, 1] (кэш).

    ``np.polynomial.legendre.leggauss`` решает задачу на собственные значения,
    поэтому результат кэшируется: при переборе окон порядок повторяется.
    """
    return np.polynomial.legendre.leggauss(order)


def _gl_nodes(a: float, b: float, order: int):
    """Узлы и веса Гаусса–Лежандра порядка ``order`` на отрезке [a, b]."""
    x, w = _leggauss(order)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def quadrature_grid(breaks, order: int = 24, max_log_span: float = 0.1):
    """Составная квадратура Гаусса–Лежандра на [breaks[0], breaks[-1]].

    Отрезок разбивается в точках ``breaks`` (узлы сплайнового базиса),
    а каждый полученный сегмент дополнительно дробится на логарифмически
    равномерные подсегменты шириной не больше ``max_log_span``, чтобы
    разрешить осциллирующие ядра Mie.

    Результат кэшируется по тройке (сетка ``breaks``, ``order``,
    ``max_log_span``): одни и те же узлы повторяются для всех показателей
    преломления ``m`` и всех ядер, поэтому строить их заново нет смысла.

    Возвращает (узлы, веса) — свежие массивы (копии кэшированных).
    """
    key = tuple(np.asarray(breaks, dtype=float).tolist())
    nodes, weights = _quadrature_grid(key, order, max_log_span)
    return nodes.copy(), weights.copy()


@lru_cache(maxsize=None)
def _quadrature_grid(breaks: tuple, order: int, max_log_span: float):
    """Ядро ``quadrature_grid``: узлы и веса для конкретного разбиения."""
    breaks_arr = np.asarray(breaks, dtype=float)
    nodes, weights = [], []
    for a, b in zip(breaks_arr[:-1], breaks_arr[1:]):
        n_seg = max(1, int(np.ceil(np.log(b / a) / max_log_span)))
        log_bounds = np.linspace(np.log(a), np.log(b), n_seg + 1)
        for u, v in zip(log_bounds[:-1], log_bounds[1:]):
            x, w = _gl_nodes(np.exp(u), np.exp(v), order)
            nodes.append(x)
            weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


@dataclass(frozen=True, eq=False)
class WindowGeometry:
    """Кэшированная геометрия окна [r0, r1]: квадратура, базис и его значения.

    Не зависит ни от показателя преломления ``m``, ни от измеренных
    коэффициентов, поэтому строится один раз на окно и переиспользуется
    для всех ``m`` и всех высот (в пакетном режиме). Поля массивов
    доступны только для чтения — не мутируйте их.

    ``values[i, k]`` — значение ``k``-й базисной функции в ``i``-м узле
    квадратуры (форма (n_nodes, n_unknown)), ``dvol`` — пересчётный
    множитель q(r) = 3/(4π r⁴) в тех же узлах.
    """

    degree: int
    r0: float
    r1: float
    n_basis: int
    interior_only: bool
    knots: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray
    basis: tuple
    values: np.ndarray
    dvol: np.ndarray

    @property
    def n_unknown(self) -> int:
        return int(self.values.shape[1])


@lru_cache(maxsize=None)
def window_geometry(r0: float, r1: float, degree: int, n_basis: int,
                    interior_only: bool = False, order: int = 24,
                    max_log_span: float = 0.1) -> WindowGeometry:
    """Геометрия окна [r0, r1] для базиса из ``n_basis`` B-сплайнов степени
    ``degree`` (кэш по параметрам окна).

    Возвращает узлы/веса квадратуры Гаусса–Лежандра, значения базисных
    функций в этих узлах (одним векторным вызовом ``BSpline`` — быстрее,
    чем покомпонентный цикл), сами функции ``BSpline`` (разделяются между
    всеми решениями этого окна) и пересчётный множитель dV/dln r.

    При ``interior_only=True`` крайние базисные функции исключаются
    (см. ``bspline_basis``) и ``n_unknown = n_basis - 2``.
    """
    if interior_only and n_basis < 3:
        raise ValueError("граничные условия n(r0)=n(r1)=0 требуют n_basis >= 3")
    knots = clamped_knots(r0, r1, degree, n_basis)
    full = [BSpline(knots, e, degree, extrapolate=False) for e in np.eye(n_basis)]
    nodes, weights = quadrature_grid(np.unique(knots), order, max_log_span)
    # один векторный вызов вместо n_basis отдельных: (n_nodes, n_basis)
    values = BSpline(knots, np.eye(n_basis), degree, extrapolate=False)(nodes)
    if interior_only:
        basis = tuple(full[1:-1])
        values = values[:, 1:-1]
    else:
        basis = tuple(full)
    dvol = dvol_dlnr_factor(nodes)
    for arr in (knots, nodes, weights, values, dvol):
        arr.flags.writeable = False
    return WindowGeometry(
        degree=int(degree), r0=float(r0), r1=float(r1), n_basis=int(n_basis),
        interior_only=bool(interior_only), knots=knots,
        nodes=nodes, weights=weights, basis=basis, values=values, dvol=dvol,
    )


def design_matrix(channels, kernels, basis, knots: np.ndarray,
                  order: int = 24, max_log_span: float = 0.1,
                  volume: bool = True) -> np.ndarray:
    """Матрица A: A[j, i] = int_{r0}^{r1} K_j(r) B_i(r) dr.

    ``kernels`` — экземпляр ``kernels.KernelTable``.

    При ``volume=True`` (по умолчанию) искомое распределение —
    объёмное F(r) = dV/dln r (n(r) = q(r)·F(r)), поэтому ядра
    умножаются на пересчётный фактор q(r) = 3/(4π r⁴).
    """
    r_q, w_q = quadrature_grid(np.unique(knots), order, max_log_span)
    j_count = len(channels)
    n_basis = len(basis)
    k_vals = np.empty((j_count, r_q.size))
    for j in range(j_count):
        k_vals[j] = kernels.kernel(j, r_q)
    if volume:
        k_vals *= dvol_dlnr_factor(r_q)[None, :]
    b_vals = np.empty((n_basis, r_q.size))
    for i, b in enumerate(basis):
        b_vals[i] = b(r_q)
    return (k_vals * w_q) @ b_vals.T