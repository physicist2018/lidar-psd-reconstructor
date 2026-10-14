"""Базисные B-сплайны на логарифмически эквидистантной сетке узлов
и составные квадратуры Гаусса–Лежандра для интегрирования ядер.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


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


def bspline_basis(r0: float, r1: float, degree: int, n_basis: int):
    """Список из ``n_basis`` B-сплайнов (объекты scipy BSpline, степени
    ``degree``; образуют разбиение единицы на [r0, r1]) и вектор узлов."""
    knots = clamped_knots(r0, r1, degree, n_basis)
    basis = [BSpline(knots, e, degree, extrapolate=False) for e in np.eye(n_basis)]
    return basis, knots


def _gl_nodes(a: float, b: float, order: int):
    """Узлы и веса Гаусса–Лежандра порядка ``order`` на отрезке [a, b]."""
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def quadrature_grid(breaks: np.ndarray, order: int = 24, max_log_span: float = 0.1):
    """Составная квадратура Гаусса–Лежандра на [breaks[0], breaks[-1]].

    Отрезок разбивается в точках ``breaks`` (узлы сплайнового базиса),
    а каждый полученный сегмент дополнительно дробится на логарифмически
    равномерные подсегменты шириной не больше ``max_log_span``, чтобы
    разрешить осциллирующие ядра Mie.

    Возвращает (узлы, веса).
    """
    nodes, weights = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        n_seg = max(1, int(np.ceil(np.log(b / a) / max_log_span)))
        log_bounds = np.linspace(np.log(a), np.log(b), n_seg + 1)
        for u, v in zip(log_bounds[:-1], log_bounds[1:]):
            x, w = _gl_nodes(np.exp(u), np.exp(v), order)
            nodes.append(x)
            weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


def design_matrix(channels, kernels, basis, knots: np.ndarray,
                  order: int = 24, max_log_span: float = 0.1) -> np.ndarray:
    """Матрица A: A[j, i] = int_{r0}^{r1} K_j(r) B_i(r) dr.

    ``kernels`` — экземпляр ``kernels.KernelTable``.
    """
    r_q, w_q = quadrature_grid(np.unique(knots), order, max_log_span)
    j_count = len(channels)
    n_basis = len(basis)
    k_vals = np.empty((j_count, r_q.size))
    for j in range(j_count):
        k_vals[j] = kernels.kernel(j, r_q)
    b_vals = np.empty((n_basis, r_q.size))
    for i, b in enumerate(basis):
        b_vals[i] = b(r_q)
    return (k_vals * w_q) @ b_vals.T