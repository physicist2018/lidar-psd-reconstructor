"""Обратная задача: восстановление распределения частиц по размерам.

Ищем n(r) = sum_i C_i B_i(r) на базисе B-сплайнов (логарифмическая сетка).
Коэффициенты C находятся методом наименьших квадратов с регуляризацией
по кривизне (вторая разность коэффициентов):

    min ||W (A C - g1)||^2 + gamma^2 ||L C||^2,   L — матрица вторых разностей,

невязка Err = (1/J) sum_j |g_j - g1_j| / g1_j — среднее по каналам.

Перебор (m, r0, r1, степень) даёт ансамбль решений; берутся G решений с
наименьшей невязкой, по каждому восстанавливается n(r), результат
усредняется; показатель преломления также усредняется по G лучшим.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import window_geometry


@dataclass(slots=True)
class Solution:
    """Одно решение для конкретного (m, r0, r1, степень сплайна)."""

    m: complex
    degree: int
    r0: float
    r1: float
    gamma: float
    C: np.ndarray          # коэффициенты (в абсолютных единицах)
    basis: list            # B-сплайны этого окна
    err: float             # Err = mean_j |g_j - g1_j| / g1_j
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
    """Невязка Err = mean_j |g_j - g1_j| / g1_j (g1 > 0)."""
    g1 = np.asarray(g1, dtype=float)
    return float(np.mean(np.abs(A @ C - g1) / g1))


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


@dataclass(frozen=True)
class Window:
    """Окно [r0, r1] и степень базиса — идентификатор варианта перебора."""

    degree: int
    r0: float
    r1: float


class DesignSet:
    """Предвычисленные матрицы ``A[m, окно]`` и их спектральные нормы.

    Элемент ``A[j, i] = int K_j(r; m) q(r) B_i(r) dr`` не зависит от
    измеренных коэффициентов, поэтому в пакетном режиме строится ОДИН раз
    и переиспользуется для всех высот. Форма ``A`` —
    ``(n_m, n_windows, n_channels, n_unknown)``; ``s0`` — те же первые две
    оси со значением ‖A‖₂ (нужна для масштаба регуляризации).
    """

    __slots__ = ("channels", "m_values", "windows", "geometry", "n_basis",
                 "zero_boundary", "order", "max_log_span", "A", "s0")

    channels: list
    m_values: list
    windows: list
    geometry: list
    n_basis: int
    zero_boundary: bool
    order: int
    max_log_span: float
    A: np.ndarray
    s0: np.ndarray

    def __init__(self, channels, m_values, windows, geometry, A, s0, *,
                 n_basis, zero_boundary, order, max_log_span):
        self.channels = list(channels)
        self.m_values = list(m_values)
        self.windows = list(windows)
        self.geometry = list(geometry)
        self.A = A
        self.s0 = s0
        self.n_basis = int(n_basis)
        self.zero_boundary = bool(zero_boundary)
        self.order = int(order)
        self.max_log_span = float(max_log_span)

    @property
    def n_unknown(self) -> int:
        return int(self.A.shape[-1])

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def build_design(channels, kernels_map: dict, degrees, r0_grid, r1_grid, *,
                 min_ratio: float = 4.0,
                 n_basis: int | None = None,
                 zero_boundary: bool = True,
                 order: int = 24, max_log_span: float = 0.1) -> DesignSet:
    """Построить матрицы A для всех ``(m, degree, r0, r1)`` за один проход.

    Геометрия окон (узлы/веса квадратуры, значения базиса) кэшируется и не
    зависит от ``m``, поэтому базис строится один раз на окно, а не на
    каждую пару ``(m, окно)``. Результат пригоден для многократного
    решения по разным ``g1`` (высотам) через ``scan_variants(design=...)``.
    """
    channels = list(channels)
    degrees = list(degrees)
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

    windows: list[Window] = []
    for degree in degrees:
        for r0 in r0_grid:
            for r1 in r1_grid:
                if r1 < min_ratio * r0:
                    continue
                windows.append(Window(int(degree), float(r0), float(r1)))
    geometry = [
        window_geometry(w.r0, w.r1, w.degree, n_basis,
                        interior_only=zero_boundary,
                        order=order, max_log_span=max_log_span)
        for w in windows
    ]

    m_values = list(kernels_map)
    n_unknown = n_basis - (2 if zero_boundary else 0)
    A = np.empty((len(m_values), len(windows), len(channels), n_unknown))
    for i_m, mv in enumerate(m_values):
        table = kernels_map[mv]
        for i_w, g in enumerate(geometry):
            k = table.kernels(g.nodes)                # (n_channels, n_q)
            A[i_m, i_w] = (k * g.dvol * g.weights) @ g.values
    if windows:
        s0 = np.linalg.svd(A, compute_uv=False)[..., 0]
    else:
        s0 = np.zeros(A.shape[:2])
    return DesignSet(channels, m_values, windows, geometry, A, s0,
                     n_basis=n_basis, zero_boundary=zero_boundary,
                     order=order, max_log_span=max_log_span)


def _batched_lstsq(aug: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Батчевый МНК через SVD — эквивалент ``np.linalg.lstsq(rcond=None)``.

    ``aug`` — (..., m, n), ``rhs`` — (..., m); результат — (..., n).
    Отсечка сингулярных чисел повторяет правило numpy (rcond=\u2009max(m,n)·eps),
    поэтому решение совпадает с циклом ``np.linalg.lstsq`` до машинной точности.
    """
    u, s, vh = np.linalg.svd(aug, full_matrices=False)
    cutoff = np.finfo(aug.dtype).eps * max(aug.shape[-2], aug.shape[-1])
    keep = s > cutoff * s[..., :1]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    ut_b = np.einsum("...ji,...j->...i", u, rhs)
    return np.einsum("...ji,...j->...i", vh, s_inv * ut_b)


def _weights_vector(g1n: np.ndarray, weights, relative: bool) -> np.ndarray:
    """Веса строк МНК: 1/g1 (относительная невязка) или единицы."""
    if weights is None:
        return 1.0 / g1n if relative else np.ones_like(g1n)
    return np.asarray(weights, dtype=float)


def _augmented_batched(A: np.ndarray, g1n: np.ndarray, gamma: np.ndarray,
                       weights: np.ndarray, L: np.ndarray):
    """Расширенная система [W A; γL] c = [W g1; 0] для батча (n_m, n_w)."""
    aw = A * weights[None, None, :, None]
    aug = np.concatenate([aw, gamma[..., None, None] * L], axis=-2)
    rhs = np.empty(aug.shape[:-1])
    rhs[...] = np.concatenate([g1n * weights, np.zeros(L.shape[0])])
    return aug, rhs


def _solve_fixed_batched(design: DesignSet, g1n: np.ndarray, gamma: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    """Батчевое решение расширенной системы для готовых ``gamma`` (n_m, n_w)."""
    L = second_diff_matrix(design.n_unknown)
    aug, rhs = _augmented_batched(design.A, g1n, gamma, weights, L)
    return _batched_lstsq(aug, rhs)


def _residual_batched(design: DesignSet, c_norm: np.ndarray,
                      g1n: np.ndarray) -> np.ndarray:
    """Относительная невязка Err (среднее по каналам) для батча (n_m, n_w)."""
    g_fit = np.einsum("...jn,...n->...j", design.A, c_norm)
    return np.mean(np.abs(g_fit - g1n) / g1n, axis=-1)


def _cv_score_batched(design: DesignSet, g1n: np.ndarray, gamma: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    """LOO-кросс-валидация закрытой формулой (батчево по окнам).

    Пусть ``M = X_wᵀ X_w + γ² Lᵀ L`` (это ровно ``augᵀ aug``), ``c`` — МНК
    по всем строкам, ``p_i = a_i·c``, ``k_i = w_i²`` и
    ``s_i = a_iᵀ M⁻¹ a_i`` (leverage строки в весовой постановке). Тогда
    решение без i-й строки даёт

        y_i − pred_i^{(−i)} = (y_i − p_i) / (1 − k_i·s_i),

    а ``s_i`` считается из уже имеющегося SVD: ``M⁻¹ = V S⁻² Vᵀ``, т.е.
    ``s_i = ‖S⁻¹ Vᵀ a_i‖²``. Возвращает ``Σ_i |y_i − pred_i|/y_i`` (та же
    метрика, что и у брутфорс ``cv_error``).

    Если leverage близок к 1 (вырожденный случай), соответствующие окна
    досчитываются брутфорсом.
    """
    A = design.A
    L = second_diff_matrix(design.n_unknown)
    aug, rhs = _augmented_batched(A, g1n, gamma, weights, L)
    u, s, vh = np.linalg.svd(aug, full_matrices=False)
    cutoff = np.finfo(aug.dtype).eps * max(aug.shape[-2], aug.shape[-1])
    keep = s > cutoff * s[..., :1]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    ut_b = np.einsum("...ji,...j->...i", u, rhs)
    c = np.einsum("...ji,...j->...i", vh, s_inv * ut_b)
    # M⁻¹ = V S⁻² Vᵀ = Zᵀ Z, где Z = S⁻¹ Vᵀ
    z = s_inv[..., :, None] * vh                       # (..., n, n)
    za = np.einsum("...ki,...ji->...jk", z, A)         # (..., j, n)
    s_ii = np.einsum("...jk,...jk->...j", za, za)      # a_iᵀ M⁻¹ a_i
    p = np.einsum("...ji,...i->...j", A, c)            # a_i · c
    k = weights * weights
    denom = 1.0 - k * s_ii                             # (..., j)
    ok = denom > 0.0
    resid = (g1n - p) / np.where(ok, denom, 1.0)
    score = np.sum(np.abs(resid) / g1n, axis=-1)
    if not np.all(ok):
        for i_m, i_w in np.argwhere(~ok.all(axis=-1)):
            score[i_m, i_w] = cv_error(A[i_m, i_w], g1n,
                                       float(gamma[i_m, i_w]),
                                       weights=weights)
    return score


def _select_gamma_batched(design: DesignSet, g1n: np.ndarray,
                          gamma_rel: np.ndarray,
                          weights: np.ndarray, gamma_select: str) -> np.ndarray:
    """Выбор gamma по каждому (m, окну) без ограничения C >= 0.

    Оба правила батчевые: "err" — по невязке Err, "cv" — по LOO-оценке
    закрытой формулой (см. ``_cv_score_batched``)."""
    s0 = design.s0
    best_score = np.full(s0.shape, np.inf)
    best_gamma = np.zeros_like(s0)
    for gr in gamma_rel:
        gamma = gr * s0
        if gamma_select == "cv":
            score = _cv_score_batched(design, g1n, gamma, weights)
        else:
            c = _solve_fixed_batched(design, g1n, gamma, weights)
            score = _residual_batched(design, c, g1n)
        better = score < best_score
        best_score = np.where(better, score, best_score)
        best_gamma = np.where(better, gamma, best_gamma)
    return best_gamma


def _scan_batched(design: DesignSet, g1: np.ndarray, g1n: np.ndarray,
                  scale: float, gamma_select: str, gamma_rel_fixed: float,
                  gamma_exp_min: float, gamma_exp_max: float, n_gamma: int,
                  weights: np.ndarray) -> list[Solution]:
    """Перебор без ограничения C >= 0: решение для всех окон сразу."""
    gamma: np.ndarray
    if gamma_select == "fixed":
        gamma = float(gamma_rel_fixed) * design.s0
    else:
        gamma_rel = 10.0 ** np.linspace(gamma_exp_min, gamma_exp_max, n_gamma)
        gamma = _select_gamma_batched(design, g1n, gamma_rel, weights, gamma_select)
    c_norm = _solve_fixed_batched(design, g1n, gamma, weights)
    c_abs = c_norm * scale
    g_fit = np.einsum("...jn,...n->...j", design.A, c_abs)
    err = np.mean(np.abs(g_fit - g1) / g1, axis=-1)
    out: list[Solution] = []
    for i_m, m in enumerate(design.m_values):
        for i_w, w in enumerate(design.windows):
            out.append(Solution(
                m=m, degree=w.degree, r0=w.r0, r1=w.r1,
                gamma=float(gamma[i_m, i_w]),
                C=c_abs[i_m, i_w],
                basis=design.geometry[i_w].basis,
                err=float(err[i_m, i_w]),
                g_fit=g_fit[i_m, i_w],
            ))
    return out


def _scan_nonneg(design: DesignSet, g1n: np.ndarray, scale: float,
                 gamma_select: str, gamma_rel_fixed: float,
                 gamma_exp_min: float, gamma_exp_max: float, n_gamma: int,
                 weights: np.ndarray) -> list[Solution]:
    """Перебор с ограничением C >= 0 (bounded МНК не батчируется)."""
    gamma_rel = 10.0 ** np.linspace(gamma_exp_min, gamma_exp_max, n_gamma)
    out: list[Solution] = []
    for i_m, m in enumerate(design.m_values):
        for i_w, w in enumerate(design.windows):
            A = design.A[i_m, i_w]
            s0 = float(design.s0[i_m, i_w])
            if gamma_select == "fixed":
                gamma = float(gamma_rel_fixed) * s0
            else:
                best = None
                for gr in gamma_rel:
                    gi = gr * s0
                    if gamma_select == "cv":
                        score = cv_error(A, g1n, gi, weights=weights, nonneg=True)
                    else:
                        c = solve_regularized(A, g1n, gi, weights=weights,
                                              nonneg=True)
                        score = residual_err(A, c, g1n)
                    if best is None or score < best[0]:
                        best = (score, gi)
                assert best is not None, "сетка gamma пуста"
                _score, gamma = best
            c_norm = solve_regularized(A, g1n, gamma, weights=weights, nonneg=True)
            err = residual_err(A, c_norm, g1n)
            c_abs = c_norm * scale
            out.append(Solution(
                m=m, degree=w.degree, r0=w.r0, r1=w.r1, gamma=float(gamma),
                C=c_abs, basis=design.geometry[i_w].basis, err=err,
                g_fit=A @ c_abs,
            ))
    return out


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
                  order: int = 24, max_log_span: float = 0.1,
                  design: DesignSet | None = None):
    """Перебор (показатель преломления m, r0, r1, степень) с выбором gamma.

    ``kernels_map`` — словарь {m: KernelTable} (таблицы ядер для каждого m
    из сетки перебора).

    ``n_basis`` — число базисных функций на окно; при ``None`` берется
    число каналов ``len(channels)``.

    ``zero_boundary`` — при True из разложения исключаются крайние базисные
    функции: каждое решение удовлетворяет n(r0) = n(r1) = 0 (однородные
    граничные условия).

    ``design`` — готовый ``DesignSet`` (матрицы A для всех ``(m, окно)``).
    Если он задан, аргументы ``degrees``/``r0_grid``/``r1_grid``,
    ``min_ratio``, ``n_basis``, ``zero_boundary``, ``order``,
    ``max_log_span`` не используются: окна и матрицы берутся из ``design``.
    Это позволяет в пакетном режиме строить A один раз на все высоты.

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
    gamma_rel_fixed_eff = 0.1 if gamma_rel_fixed is None else float(gamma_rel_fixed)
    g1 = np.asarray(g1, dtype=float)
    if np.any(g1 <= 0):
        raise ValueError("все измеренные коэффициенты g1 должны быть положительными")
    scale = float(g1.max())
    g1n = g1 / scale
    # веса 1/g1 — относительная невязка в согласии с Err
    weights = _weights_vector(g1n, weights, relative)

    if design is None:
        design = build_design(channels, kernels_map, degrees, r0_grid, r1_grid,
                              min_ratio=min_ratio, n_basis=n_basis,
                              zero_boundary=zero_boundary,
                              order=order, max_log_span=max_log_span)
    n_unknown = design.n_unknown
    if n_unknown > len(channels):
        import warnings
        warnings.warn(
            f"неизвестных на окно ({n_unknown}) больше измерений ({len(channels)}): "
            f"решение опирается главным образом на регуляризацию "
            f"(система недоопределена)", stacklevel=2)

    if design.n_windows == 0:
        return [], scale

    if nonneg:
        solutions = _scan_nonneg(design, g1n, scale, gamma_select,
                                 gamma_rel_fixed_eff, gamma_exp_min,
                                 gamma_exp_max, n_gamma, weights)
    else:
        solutions = _scan_batched(design, g1, g1n, scale, gamma_select,
                                  gamma_rel_fixed_eff, gamma_exp_min,
                                  gamma_exp_max, n_gamma, weights)

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