"""Обратная задача: восстановление распределения частиц по размерам.

Ищем n(r) = sum_i C_i B_i(r) на базисе B-сплайнов (логарифмическая сетка).
Коэффициенты C находятся методом наименьших квадратов с регуляризацией
по кривизне (вторая разность коэффициентов):

    min ||W (A C - g1)||^2 + gamma^2 ||L C||^2,   L — матрица вторых разностей,

невязка Err = sum_j |g_j - g1_j| / g1_j.

Перебор (m, r0, r1, степень) даёт ансамбль решений; берутся G решений с
наименьшей невязкой, по каждому восстанавливается n(r), результат
усредняется; показатель преломления также усредняется по G лучшим.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import bspline_basis, design_matrix


@dataclass
class Solution:
    """Одно решение для конкретного (m, r0, r1, степень сплайна)."""

    m: complex
    degree: int
    r0: float
    r1: float
    gamma: float
    C: np.ndarray          # коэффициенты (в абсолютных единицах)
    basis: list            # B-сплайны этого окна
    err: float             # Err = sum_j |g_j - g1_j| / g1_j
    g_fit: np.ndarray      # рассчитанные коэффициенты g = A C

    def distribution(self, r) -> np.ndarray:
        """Восстановленная функция распределения n(r) = sum C_i B_i(r).

        Базисные сплайны поддерживаются на [r0, r1]; вне окна значения
        обнуляются (BSpline с extrapolate=False даёт NaN вне области).
        """
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        inside = (r >= self.r0) & (r <= self.r1)
        r_in = r[inside]
        if r_in.size:
            for c, b in zip(self.C, self.basis):
                out[inside] += c * b(r_in)
        return out


def second_diff_matrix(n: int) -> np.ndarray:
    """Матрица вторых разностей размера (n-2) x n (регуляризация по кривизне)."""
    m = max(n - 2, 0)
    L = np.zeros((m, n))
    for i in range(m):
        L[i, i:i + 3] = (1.0, -2.0, 1.0)
    return L


def solve_regularized(A: np.ndarray, g1: np.ndarray, gamma: float,
                      weights: np.ndarray | None = None, nonneg: bool = False) -> np.ndarray:
    """МНК с регуляризацией по кривизне.

    min ||W(A C - g1)||^2 + gamma^2 ||L C||^2 через расширенную систему.
    При ``nonneg=True`` накладывается ограничение C >= 0 (тогда и n(r) >= 0).
    """
    if weights is None:
        weights = np.ones_like(g1)
    aw = A * weights[:, None]
    bw = g1 * weights
    L = second_diff_matrix(A.shape[1])
    a_aug = np.vstack([aw, gamma * L])
    b_aug = np.concatenate([bw, np.zeros(L.shape[0])])
    if nonneg:
        from scipy.optimize import lsq_linear
        return lsq_linear(a_aug, b_aug, bounds=(0.0, np.inf)).x
    c, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    return c


def residual_err(A: np.ndarray, C: np.ndarray, g1: np.ndarray) -> float:
    """Невязка Err = sum_j |g_j - g1_j| / g1_j (g1 > 0)."""
    g1 = np.asarray(g1, dtype=float)
    return float(np.sum(np.abs(A @ C - g1) / g1))


def _solve_with_mask(A, b, gamma, weights, mask, nonneg):
    """Регуляризованный МНК по подмножеству строк (для кросс-валидации)."""
    L = second_diff_matrix(A.shape[1])
    a_tr = np.vstack([A[mask] * weights[mask, None], gamma * L])
    b_tr = np.concatenate([b[mask] * weights[mask], np.zeros(L.shape[0])])
    if nonneg:
        from scipy.optimize import lsq_linear
        return lsq_linear(a_tr, b_tr, bounds=(0.0, np.inf)).x
    c, *_ = np.linalg.lstsq(a_tr, b_tr, rcond=None)
    return c


def cv_error(A: np.ndarray, g1: np.ndarray, gamma: float,
             weights: np.ndarray | None = None, nonneg: bool = False) -> float:
    """Leave-one-out кросс-валидация в метрике относительной невязки.

    Для каждого канала j решение ищется без него и сравнивается
    предсказанное g_j с измеренным g1_j. Такое правило выбора gamma
    защищает от переобучения (точной интерполяции с 5 неизвестными
    по 5 измерениям) и не требует знания уровня шума.
    """
    j_count = A.shape[0]
    if weights is None:
        weights = np.ones(j_count)
    tot = 0.0
    for j in range(j_count):
        mask = np.ones(j_count, dtype=bool)
        mask[j] = False
        c = _solve_with_mask(A, g1, gamma, weights, mask, nonneg)
        pred = float(A[j] @ c)
        tot += abs(pred - g1[j]) / g1[j]
    return tot


def scan_variants(channels, kernels_map: dict, g1: np.ndarray, degrees,
                  r0_grid, r1_grid, *,
                  min_ratio: float = 4.0,
                  gamma_exp_min: float = -7.0, gamma_exp_max: float = 5.0,
                  n_gamma: int = 41,
                  weights: np.ndarray | None = None,
                  nonneg: bool = False,
                  relative: bool = True,
                  gamma_select: str = "fixed",
                  gamma_rel_fixed: float | None = None,
                  zero_boundary: bool = True,
                  n_basis: int | None = None,
                  order: int = 24, max_log_span: float = 0.1):
    """Перебор (показатель преломления m, r0, r1, степень) с выбором gamma.

    ``kernels_map`` — словарь {m: KernelTable} (таблицы ядер для каждого m
    из сетки перебора).

    ``n_basis`` — число базисных функций на окно; при ``None`` берется
    число каналов ``len(channels)``.

    ``zero_boundary`` — при True из разложения исключаются крайние базисные
    функции: каждое решение удовлетворяет n(r0) = n(r1) = 0 (однородные
    граничные условия).

    ``gamma_select``:
      "fixed" — фиксированный gamma_rel (относительно ||A||_2).
      "cv"    — leave-one-out кросс-валидация по относительной невязке;
      "err"   — минимум невязки Err (как указано в задании; при 5 уравнениях
                и 5 неизвестных вырождается в точную интерполяцию).

    Возвращает (отсортированные по Err решения, scale), где scale —
    нормировочный множитель измеренных коэффициентов (max |g1|).
    """
    if gamma_select not in ("fixed", "cv", "err"):
        raise ValueError("gamma_select должен быть 'fixed', 'cv' или 'err'")
    if gamma_select == "fixed" and gamma_rel_fixed is None:
        gamma_rel_fixed = 0.1  # значение по умолчанию
    g1 = np.asarray(g1, dtype=float)
    if np.any(g1 <= 0):
        raise ValueError("все измеренные коэффициенты g1 должны быть положительными")
    scale = float(g1.max())
    g1n = g1 / scale
    if weights is None:
        # веса 1/g1 — относительная невязка в согласии с Err
        weights = 1.0 / g1n if relative else np.ones_like(g1n)
    gamma_rel = 10.0 ** np.linspace(gamma_exp_min, gamma_exp_max, n_gamma)

    if not degrees:
        raise ValueError("список степеней пуст")
    if n_basis is None:
        n_basis = len(channels)
    degree_max = max(degrees)
    if n_basis < degree_max + 1:
        raise ValueError(
            f"n_basis ({n_basis}) < max(степень)+1 ({degree_max + 1}): "
            f"уменьшите --degree или увеличьте --n-basis")
    if zero_boundary and n_basis < 3:
        raise ValueError("при нулевых граничных условиях нужно n_basis >= 3")
    n_unknown = n_basis - (2 if zero_boundary else 0)
    if n_unknown > len(channels):
        import warnings
        warnings.warn(
            f"неизвестных на окно ({n_unknown}) больше измерений ({len(channels)}): "
            f"решение опирается главным образом на регуляризацию "
            f"(система недоопределена)", stacklevel=2)

    solutions: list[Solution] = []
    for m, kernels in kernels_map.items():
        for degree in degrees:
            for r0 in r0_grid:
                for r1 in r1_grid:
                    if r1 < min_ratio * r0:
                        continue
                    basis, knots = bspline_basis(r0, r1, degree, n_basis,
                                                  interior_only=zero_boundary)
                    A = design_matrix(channels, kernels, basis, knots,
                                      order=order, max_log_span=max_log_span)
                    s0 = np.linalg.norm(A, 2)
                    if gamma_select == "fixed":
                        gamma = (gamma_rel_fixed if gamma_rel_fixed is not None else 0.1) * s0
                    else:
                        best = None
                        for gamma_rel_i in gamma_rel:
                            gamma_i = gamma_rel_i * s0
                            if gamma_select == "cv":
                                score = cv_error(A, g1n, gamma_i, weights=weights,
                                                 nonneg=nonneg)
                            else:
                                c = solve_regularized(A, g1n, gamma_i,
                                                      weights=weights, nonneg=nonneg)
                                score = residual_err(A, c, g1n)
                            if best is None or score < best[0]:
                                best = (score, gamma_i)
                        assert best is not None, "сетка gamma пуста"
                        _score, gamma = best
                    c_norm = solve_regularized(A, g1n, gamma, weights=weights,
                                               nonneg=nonneg)
                    err = residual_err(A, c_norm, g1n)
                    c_abs = c_norm * scale
                    solutions.append(Solution(
                        m=m, degree=degree, r0=r0, r1=r1, gamma=gamma,
                        C=c_abs, basis=basis, err=err, g_fit=A @ c_abs,
                    ))

    solutions.sort(key=lambda s: s.err)
    return solutions, scale


def smooth_distribution(r_grid: np.ndarray, n_avg: np.ndarray,
                        w: np.ndarray | None = None,
                        lam_rel: float | None = None) -> np.ndarray:
    """Сглаживание среднего по лучшим решениям сглаживающим сплайном.

    Кубический сглаживающий сплайн строится в нормированных координатах
    (аргумент — логарифм радиуса, приведённый к [0, 1]; значения —
    n/max|n|), поэтому параметр ``lam_rel`` безразмерен и переносится
    между наборами данных. По умолчанию ``lam_rel = 1e-4`` (подобран по
    демо: убирает изломы на границах окон без искажения формы).

    Веса ``w`` — обычно кратность покрытия точек; крайние точки покрытия
    дополнительно притягиваются к своим сырым значениям (веса краёв
    усилены), чтобы сглаживание не давало артефактов на границах.

    Точки вне покрытия (NaN в ``n_avg``) в аппроксимации не участвуют и
    в результате остаются NaN. Возвращает массив той же формы, что n_avg.
    """
    from scipy.interpolate import make_smoothing_spline

    r_grid = np.asarray(r_grid, dtype=float)
    n_avg = np.asarray(n_avg, dtype=float)
    ok = np.isfinite(n_avg)
    if ok.sum() < 4:  # сглаживающему сплайну нужно достаточно точек
        return n_avg.copy()
    if lam_rel is None:
        lam_rel = 1e-4

    t = np.log(r_grid[ok])
    t = (t - t[0]) / (t[-1] - t[0])          # -> [0, 1]
    y = n_avg[ok].astype(float)
    scale = np.max(np.abs(y))
    if scale <= 0 or not np.isfinite(scale):
        return n_avg.copy()
    ys = y / scale

    if w is None:
        wts = np.ones_like(ys)
    else:
        wts = np.asarray(w, dtype=float)[ok]
    wmax = float(np.max(wts)) if wts.size else 1.0
    wts = wts / max(wmax, 1e-300)
    wts[0] = wts[-1] = 1e3   # привязка краёв покрытия к сырым значениям

    spl = make_smoothing_spline(t, ys, w=wts, lam=lam_rel)
    out = np.full_like(n_avg, np.nan)
    out[ok] = spl(t) * scale
    return out


def ensemble_average(solutions: list[Solution], n_best: int, r_grid: np.ndarray,
                     smooth: bool = True, lam_rel: float | None = None):
    """Среднее по ``n_best`` лучшим решениям на общей сетке r_grid.

    В каждой точке r усредняются только те решения, чьё окно [r0, r1]
    покрывает эту точку (нормировка по покрытию): если окно не описывает
    данный диапазон размеров, его нельзя интерпретировать как n(r) = 0.
    Точки, не покрытые ни одним решением, получают NaN.

    При ``smooth=True`` к среднему применяется сглаживание
    (``smooth_distribution``, относительный параметр ``lam_rel``) —
    поточечное среднее решений с разными узлами сплайнов и границами
    окон изломается на границах покрытия, сглаженная кривая убирает
    эти изломы.

    Показатель преломления усредняется по тем же ``n_best`` решениям
    (простое среднее комплексных значений).

    Возвращает (n_avg, n_best_matrix, m_avg), где n_avg — сглаженное
    (или исходное, если ``smooth=False``) среднее.
    """
    n_best = max(1, min(n_best, len(solutions)))
    r_grid = np.asarray(r_grid, dtype=float)
    n = r_grid.size
    ns = np.empty((n_best, n))
    total = np.zeros(n)
    count = np.zeros(n)
    for t, s in enumerate(solutions[:n_best]):
        ns[t] = s.distribution(r_grid)
        cov = (r_grid >= s.r0) & (r_grid <= s.r1)
        total += np.where(cov, ns[t], 0.0)
        count += cov
    n_avg = np.full(n, np.nan)
    ok = count > 0
    n_avg[ok] = total[ok] / count[ok]
    if smooth:
        n_avg = smooth_distribution(r_grid, n_avg, w=count, lam_rel=lam_rel)
    best = solutions[:n_best]
    m_avg = complex(sum(s.m for s in best) / n_best)
    return n_avg, ns, m_avg