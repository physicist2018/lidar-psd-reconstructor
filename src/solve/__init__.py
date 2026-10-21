"""solve — восстановление объёмного распределения частиц по размерам
из лидарных коэффициентов (обратное рассеяние и ослабление).

Единицы входных данных: β — 1/(м·ср), α — 1/м (ядра приводятся к м²/м²·ср).
Восстанавливается F(r) = dV/dln r (объём частиц на логарифмический
интервал радиуса); в выводе объёмные величины приводятся к мкм³/см³,
числовая концентрация N — к см⁻³. Дополнительно выводятся интегральные
характеристики (N, C_v, ⟨V⟩, r_eff) и параметры логнормальных мод
(Vf, rm_f, σf, Vc, rm_c, σc и общие V, rm, σ).

Метод:
  * F(r) ищется в виде разложения по базисным B-сплайнам степени 1..4
    на логарифмически эквидистантной сетке узлов интервала [r0, r1];
    число базисных функций равно числу измеренных коэффициентов M;
    крайние функции базиса исключаются — каждое элементарное решение
    обращается в ноль на границах: F(r0) = F(r1) = 0;
  * связь с числовым распределением: n(r) = dN/dr = F(r)·3/(4π r⁴),
    поэтому ядра в матрице A умножаются на пересчётный фактор
    q(r) = 3/(4π r⁴): A[j, i] = int K_j(r)·q(r)·B_i(r) dr (ядра Mie —
    через miepython, с кэширующей таблицей и кубической интерполяцией);
  * система решается МНК с регуляризацией по кривизне (вторая разность)
    ||W(A C - g1)||^2 + gamma^2 ||L C||^2; параметр gamma выбирается
    фиксированным (gamma_rel·||A||_2, по умолчанию), по LOO-кросс-валидации
    или по минимуму невязки Err = sum_j |g_j - g1_j| / g1_j;
  * перебираются окна [r0, r1] (и степени сплайнов), а также сетка
    показателей преломления m; берутся G решений с наименьшей невязкой,
    по каждому восстанавливается F(r), результат усредняется и
    сглаживается; среднее по G решениям значение m и есть итоговый
    показатель преломления;
  * по усреднённому F(r) оцениваются параметры логнормальных мод:
    fine (r < r_split), coarse (r > r_split) и общая логнормаль.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .basis import quadrature_grid
from .inverse import ensemble_average, scan_variants
from .kernels import (
    Channel,
    KernelTable,
    dvol_dlnr_factor,
    load_table_channels,
    mie_jit_enabled,
    set_mie_jit,
)

__all__ = [
    "Channel", "KernelTable", "scan_variants", "ensemble_average",
    "channels_from_data", "forward_measurements",
    "lognormal_volume", "demo_true_distribution", "mode_analysis",
    "size_moments", "main",
]

DEMO_BETA_WL_NM = (355, 532, 1064)
DEMO_ALPHA_WL_NM = (355, 532)
# Примечание: в исходном задании указаны обратное рассеяние 533/532/1064 нм
# и ослабление 355/532 нм.  "533" здесь трактуется как опечатка для 355 нм
# (стандартный набор трёхволнового лидара 355/532/1064).
# Набор длин волн полностью определяется данными (--data), код их не
# зашивает.


# ----------------------------------------------------------------------
# форматы данных
# ----------------------------------------------------------------------

def parse_grid_spec(tokens, flag: str) -> list[float]:
    """Разбор спецификации сетки для --m-reals / --m-imags.

    Каждый токен может быть:
      1.45                — одиночное значение;
      1.3:1.6:7           — линейная сетка: np.linspace(1.3, 1.6, 7);
      0.001:0.1:5:log     — логарифмическая сетка: np.geomspace(0.001, 0.1, 5)
                            (границы должны быть положительными).
    Несколько токенов объединяются. Возвращает отсортированный список
    уникальных значений.
    """
    values: list[float] = []
    for tok in tokens:
        parts = tok.split(":")
        try:
            if len(parts) == 1:
                values.append(float(parts[0]))
            elif len(parts) == 3:
                a, b, n = float(parts[0]), float(parts[1]), int(parts[2])
                if n < 1:
                    raise ValueError("число точек должно быть >= 1")
                values.extend(np.linspace(a, b, n).tolist())
            elif len(parts) == 4 and parts[3] == "log":
                a, b, n = float(parts[0]), float(parts[1]), int(parts[2])
                if a <= 0 or b <= 0:
                    raise ValueError("логарифмическая сетка требует положительные границы")
                if n < 1:
                    raise ValueError("число точек должно быть >= 1")
                values.extend(np.geomspace(a, b, n).tolist())
            else:
                raise ValueError("ожидается значение или диапазон a:b:n[:log]")
        except ValueError as exc:
            raise SystemExit(f"ошибка в {flag}: токен {tok!r}: {exc}") from None
    seen: set[float] = set()
    out: list[float] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def channels_from_data(data: dict):
    """Каналы и измеренные коэффициенты из словаря (JSON).

    Формат:
    {
      "m": {"real": 1.45, "imag": -0.005},
      "backscatter":  {"wavelengths_nm": [...], "coefficients": [...]},
      "extinction":   {"wavelengths_nm": [...], "coefficients": [...]}
    }
    """
    m = complex(data["m"]["real"], data["m"]["imag"])
    channels = []
    g1 = []
    for kind, key in (("beta", "backscatter"), ("alpha", "extinction")):
        ch = data.get(key) or {}
        for wl_nm, val in zip(ch.get("wavelengths_nm", []),
                              ch.get("coefficients", [])):
            channels.append(Channel(kind, float(wl_nm) * 1e-3))
            g1.append(float(val))
    if not channels:
        raise ValueError("в данных нет ни одного канала")
    return m, channels, np.asarray(g1, float)


def forward_measurements(channels, kernels: KernelTable, n_func, r_min, r_max,
                         order: int = 24, max_log_span: float = 0.05,
                         volume: bool = True) -> np.ndarray:
    """Прямое моделирование: g1_j = int_{r_min}^{r_max} K_j(r) n(r) dr

    с плотной составной квадратурой Гаусса–Лежандра.  При ядрах в м²
    коэффициенты beta получаются в 1/(м·ср), alpha — в 1/м.

    При ``volume=True`` (по умолчанию) ``n_func`` задаёт объёмное
    распределение F(r) = dV/dln r, а интегрируется K_j(r)·q(r)·F(r)
    (n(r) = q(r)·F(r), q = 3/(4π r⁴)).
    """
    r_q, w_q = quadrature_grid(np.array([r_min, r_max]), order, max_log_span)
    n_q = np.asarray(n_func(r_q), float)
    if volume:
        n_q = n_q * dvol_dlnr_factor(r_q)
    out = np.empty(len(channels))
    for j in range(len(channels)):
        out[j] = np.sum(w_q * kernels.kernel(j, r_q) * n_q)
    return out


# ----------------------------------------------------------------------
# интегральные характеристики восстановленного распределения
# ----------------------------------------------------------------------

UM3_TO_M3 = 1e-18  # (1 мкм)^3 = 1e-18 м^3
CM3_PER_M3 = 1e6   # 1 м³ = 1e6 см³


def _per_cm3(v):
    """Привести объёмную/числовую величину «на м³» к «на см³» (v / 1e6).

    Работает и со скалярами, и с массивами numpy. Внутренние вычисления
    ведутся в единицах «на м³», к см³ приводится только вывод.
    """
    return v / CM3_PER_M3


def size_moments(n_avg, r_grid) -> dict:
    """Интегральные характеристики восстановленного объёмного распределения
    F(r) = dV/dln r (внутри — в мкм³/м³), r в мкм.

    Точки вне покрытия лучших решений (F = NaN) дают нулевой вклад;
    диапазон покрытия и его доля возвращаются в ``r_cov_min``,
    ``r_cov_max``, ``cov_frac``. Отрицательные артефакты восстановления
    отбрасываются (по физическому смыслу F ≥ 0).

    Оценки (n(r) = q(r)·F(r), q = 3/(4π r⁴)); объёмные и числовые величины
    возвращаются в расчёте на см³:
      N         — числовая концентрация, см⁻³;
      vol_conc  — объёмная концентрация V = ∫F dln r, мкм³/см³;
      vol_mean  — средний объём частицы, мкм³;
      r_eff     — эффективный радиус, мкм.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    n_arr = np.asarray(n_avg, dtype=float)
    cov = np.isfinite(n_arr)
    f_arr = np.where(cov, n_arr, 0.0)
    f_arr = np.maximum(f_arr, 0.0)  # физическое распределение неотрицательно
    ln_r = np.log(r_grid)

    v_conc = np.trapezoid(f_arr, ln_r)                       # ∫F dln r    [мкм³/м³]
    m_ln1 = np.trapezoid(f_arr / r_grid, ln_r)               # ∫(F/r) dln r
    n_num = np.trapezoid(f_arr * dvol_dlnr_factor(r_grid),
                         r_grid)                             # ∫ n dr      [м⁻³]

    vol_mean = v_conc / n_num if n_num > 0 else np.nan
    r_eff = v_conc / m_ln1 if m_ln1 > 0 else np.nan          # ∫r³n dr/∫r²n dr

    r_cov = r_grid[cov]
    r_cov_min = float(r_cov.min()) if r_cov.size else np.nan
    r_cov_max = float(r_cov.max()) if r_cov.size else np.nan

    return {
        "N_cm-3": float(_per_cm3(n_num)),
        "vol_mean_um3": float(vol_mean),
        "vol_conc_um3_per_cm3": float(_per_cm3(v_conc)),
        "r_eff_um": float(r_eff),
        "r_cov_min_um": r_cov_min,
        "r_cov_max_um": r_cov_max,
        "cov_frac": float(cov.mean()) if r_grid.size else 0.0,
    }


def _print_moments(mom: dict) -> None:
    print("\n=== Интегральные характеристики (оценки по покрытой области) ===")
    print(f"  покрытый диапазон радиусов: {mom['r_cov_min_um']:.4g}–"
          f"{mom['r_cov_max_um']:.4g} мкм (доля сетки: {mom['cov_frac'] * 100:.0f}%)")
    v_m3 = mom["vol_mean_um3"] * UM3_TO_M3
    print(f"  числовая концентрация N      = {mom['N_cm-3']:.4g} см⁻³")
    print(f"  объёмная концентрация C_v    = {mom['vol_conc_um3_per_cm3']:.4g} мкм³/см³")
    print(f"  средний объём частиц ⟨V⟩    = {mom['vol_mean_um3']:.4g} мкм³ "
          f"({v_m3:.4g} м³)")
    print(f"  эффективный радиус r_eff      = {mom['r_eff_um']:.4g} мкм")


# ----------------------------------------------------------------------
# логнормальный анализ мод объёмного распределения
# ----------------------------------------------------------------------

def _fit_lognormal_ln(x, y, w=None):
    """Фиѭ логнормали к точкам (x = ln r, y = ln F): y = a + b·x + c·x².

    Для логнормали y = ln V − ln(√(2π)·s) − (x−μ)²/(2s²):
      μ = −b/(2c),  s² = −1/(2c),  V = exp(a − b²/(4c))·√(2π)·s.
    Возвращает (V, rm, sigma) или (nan, nan, nan), если фит не удался
    (менее трёх точек или c ≥ 0 — не мода, а «впадина»).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    design = np.column_stack([np.ones_like(x), x, x * x])
    if w is None:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        w = np.asarray(w, dtype=float)
        wd = design * np.sqrt(w)[:, None]
        wy = y * np.sqrt(w)
        coeffs, *_ = np.linalg.lstsq(wd, wy, rcond=None)
    a, b, c = coeffs
    if c >= 0 or not np.isfinite(c):
        return float("nan"), float("nan"), float("nan")
    mu = -b / (2.0 * c)
    s2 = -1.0 / (2.0 * c)
    s = np.sqrt(s2)
    v = np.exp(a - b * b / (4.0 * c)) * np.sqrt(2.0 * np.pi) * s
    if not (v > 0 and np.isfinite(v) and np.isfinite(mu)):
        return float("nan"), float("nan"), float("nan")
    return float(v), float(np.exp(mu)), float(np.exp(s))


def mode_analysis(r_grid, f_avg, r_split: float = 0.5,
                  w=None, method: str = "auto") -> dict:
    """Логнормальный анализ мод объёмного распределения F = dV/dln r.

    fine — по покрытой области r ∈ [r_cov_min, r_split];
    coarse — r ∈ [r_split, r_cov_max];
    overall — весь покрытый диапазон.

    Для каждого диапазона:
      V — интеграл F по ln r (внутренние единицы мкм³/м³; при выводе
          переводится в мкм³/см³);
      rm, sigma — либо из взвешенного МНК-фита логнормали в координатах
      (ln r, ln F) (точки с F ≤ 0 исключены, веса — покрытие), либо из
      статистической (моментной) оценки rm = exp⟨ln r⟩_V,
      sigma = exp(СКО ln r)_V.

    method — способ расчёта rm и sigma:
      "auto" — фит логнормалью, при вырождении фита — моменты (по умолчанию);
      "fit"  — только фит (при неудаче — моменты);
      "stat" — только статистическая (моментная) оценка, без фита.

    Возвращает словарь с ключами fine/coarse/overall; каждая запись —
    (V, rm, sigma, ok, method), где ok=False означает, что данных
    недостаточно либо объём неположителен; method — "fit", "moments"
    или описание причины неудачи.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    f = np.asarray(f_avg, dtype=float)
    cov = np.isfinite(f)
    r = r_grid[cov]
    fv = f[cov]
    if r.size < 4:
        nan = (float("nan"), float("nan"), float("nan"), False,
               "недостаточно точек")
        return {"fine": nan, "coarse": nan, "overall": nan}
    wv = None if w is None else np.asarray(w, dtype=float)[cov]
    r0, r1 = float(r.min()), float(r.max())

    def analyze(mask, r_lo, r_hi):
        r_sel = r[mask]
        f_sel = fv[mask]
        if r_sel.size < 3:
            return (float("nan"), float("nan"), float("nan"), False,
                    "недостаточно точек")
        v = float(np.trapezoid(np.maximum(f_sel, 0.0), np.log(r_sel)))
        if not (v > 0 and np.isfinite(v)):
            return (float("nan"), float("nan"), float("nan"), False,
                    "нет положительного объёма")
        pos = f_sel > 0
        if pos.sum() < 3:
            return v, float("nan"), float("nan"), False, "мало точек с F > 0"
        x = np.log(r_sel[pos])
        if method != "stat":
            wts = None if wv is None else wv[mask][pos]
            v_f, rm, sg = _fit_lognormal_ln(x, np.log(f_sel[pos]), wts)
            if np.isfinite(v_f) and v_f > 0 and np.isfinite(rm):
                return v, rm, sg, True, "fit"
        # статистическая (моментная) оценка: эквивалентная логнормаль
        # по объёмным моментам: rm = exp⟨ln r⟩_V, sigma = exp(СКО ln r)_V
        wts_f = np.asarray(f_sel[pos], dtype=float)
        wsum = float(np.sum(wts_f))
        mu = float(np.sum(wts_f * x) / wsum)
        s2 = float(np.sum(wts_f * (x - mu) ** 2) / wsum)
        s2 = s2 if (s2 > 0 and np.isfinite(s2)) else 0.0
        return v, float(np.exp(mu)), float(np.exp(np.sqrt(s2))), True, "moments"

    fine = (r <= r_split) & (r >= r0)
    coarse = (r >= r_split) & (r <= r1)
    overall = np.ones(r.size, dtype=bool)
    return {
        "fine": analyze(fine, r0, r_split),
        "coarse": analyze(coarse, r_split, r1),
        "overall": analyze(overall, r0, r1),
    }


def _print_modes(modes: dict, r_split: float, method: str = "auto") -> None:
    print("\n=== Параметры логнормальных мод (dV/dln r) ===")
    print(f"  граница раздела мод: {r_split:g} мкм (--r-split)")
    print(f"  способ расчёта параметров: {method} (--modes)")
    print(f"  {'мода':<8} {'V, мкм³/см³':>12} {'rm, мкм':>9} {'sigma':>7}   примечание")
    for name in ("fine", "coarse", "overall"):
        v, rm, sg, ok, method = modes[name]
        if ok:
            note = "" if method == "fit" else f"({method})"
            print(f"  {name:<8} {float(_per_cm3(v)):>12.4g} {rm:>9.4g} {sg:>7.4g}   {note}")
        else:
            print(f"  {name:<8} {'—':>12} {'—':>9} {'—':>7}   {method}")


# ----------------------------------------------------------------------
# демонстрационные данные
# ----------------------------------------------------------------------

def lognormal_volume(r, V: float, r_g: float, sigma: float) -> np.ndarray:
    """Логнормальная мода в объёмном представлении dV/dln r:

    F(r) = V exp(-(ln(r/r_g))^2 / (2 ln^2 sigma)) / (sqrt(2 pi) ln sigma),

    V — интеграл по ln r (объёмная концентрация моды; внутренние единицы
    построения — мкм³/м³, в выводе — мкм³/см³),
    r_g — медианный (модальный) радиус, sigma — геометрическое СКО.
    """
    r = np.asarray(r, float)
    s = np.log(sigma)
    return V / (np.sqrt(2.0 * np.pi) * s) * np.exp(-(np.log(r / r_g) ** 2) / (2.0 * s * s))


DEMO_FINE = (4.0e7, 0.15, 1.5)     # Vf, rm_f, sigma_f (dV/dln r)
DEMO_COARSE = (6.0e7, 1.2, 1.7)    # Vc, rm_c, sigma_c


def demo_true_distribution(r, fine: tuple = DEMO_FINE,
                           coarse: tuple = DEMO_COARSE) -> np.ndarray:
    """Тестовая «истинная» функция распределения — сумма двух логнормальных
    мод в представлении dV/dln r (fine и coarse).

    Параметры (V в мкм³/м³, rm в мкм, sigma безразмерно) подобраны так,
    чтобы при физических ядрах (м²) модельные коэффициенты были
    реалистичны: beta ~ 10⁻⁵ 1/(м·ср), alpha ~ 10⁻⁴ 1/м.
    """
    r = np.asarray(r, float)
    return lognormal_volume(r, *fine) + lognormal_volume(r, *coarse)


def _demo_measurements(args, channels, kernels: KernelTable, r1_max: float):
    """Измеренные коэффициенты для демонстрации (с опциональным шумом)."""
    g1 = forward_measurements(channels, kernels, demo_true_distribution,
                              args.r0_min, r1_max)
    if args.noise > 0:
        rng = np.random.default_rng(args.seed)
        g1 = g1 * (1.0 + rng.normal(0.0, args.noise, size=g1.size))
    return g1


# ----------------------------------------------------------------------
# построение графиков
# ----------------------------------------------------------------------

def plot_results(out_path: str, *, r_grid, n_true, n_avg, n_avg_raw, n_best,
                 solutions, names, g1, g_fit, m_avg) -> bool:
    """Два панно: n(r) (log-log) и сравнение измеренных g1 с расчётными g."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[предупреждение] matplotlib недоступен: {exc}")
        return False

    # объёмное распределение выводится в мкм³/см³
    n_avg = _per_cm3(n_avg)
    n_best = _per_cm3(n_best)
    n_true = None if n_true is None else _per_cm3(n_true)
    n_avg_raw = None if n_avg_raw is None else _per_cm3(n_avg_raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    if n_true is not None:
        ax1.semilogx(r_grid, n_true, "k--", lw=1.6, label="истинное n(r)")
    for n_s in n_best:
        ax1.semilogx(r_grid, n_s, color="0.75", lw=1.0)
    if n_avg_raw is not None and not np.array_equal(np.nan_to_num(n_avg), np.nan_to_num(n_avg_raw)):
        ax1.semilogx(r_grid, n_avg_raw, color="darkorange", lw=1.0, ls=":",
                     label="среднее (несглаженное)")
    ax1.semilogx(r_grid, n_avg, "C0-", lw=2.6, label="среднее по лучшим")
    ax1.set_xlabel("r, мкм")
    ax1.set_ylabel("dV/dln r, мкм³/см³")
    ax1.set_title(f"Объёмное распределение частиц по размерам, "
                  f"⟨m⟩ = {m_avg.real:.3f}{m_avg.imag:+.3f}j")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    x = np.arange(len(names))
    w = 0.36
    ax2.bar(x - w / 2, np.asarray(g1), w, label="измеренные g1", color="C1")
    ax2.bar(x + w / 2, np.asarray(g_fit), w, label="расчётные g (усредн.)", color="C2")
    ax2.set_yscale("linear")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("коэффициент, отн. ед.")
    ax2.set_title("Согласие измеренных и расчётных коэффициентов")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"график сохранён: {out_path}")
    return True


# ----------------------------------------------------------------------
# вывод результатов
# ----------------------------------------------------------------------

def _json_payload(r_grid, n_avg, n_best, solutions, channels, g1, g_fit,
                  m_avg, moments, n_avg_raw=None, modes=None) -> dict:
    payload = {
        "m_avg": {"real": m_avg.real, "imag": m_avg.imag},
        "moments": moments,
        "units_input": {"beta": "1/(м*ср)", "alpha": "1/м"},
        "channels": [
            {"kind": ch.kind, "wl_nm": ch.wl_um * 1e3,
             "g1": float(g1[j]), "g_fit": float(g_fit[j])}
            for j, ch in enumerate(channels)
        ],
        "r_um": r_grid.tolist(),
        "n_avg": [None if not np.isfinite(v) else float(_per_cm3(v)) for v in n_avg],
        "best_solutions": [
            {"rank": t + 1, "m_real": s.m.real, "m_imag": s.m.imag,
             "degree": s.degree, "r0": s.r0, "r1": s.r1,
             "gamma": s.gamma, "err": s.err, "C": s.C.tolist()}
            for t, s in enumerate(solutions)
        ],
    }
    if n_avg_raw is not None:
        payload["n_avg_raw"] = [None if not np.isfinite(v) else float(_per_cm3(v))
                                for v in n_avg_raw]
    if modes is not None:
        payload["modes"] = {
            k: {
                "V_um3_per_cm3": None if not np.isfinite(v[0]) else float(_per_cm3(v[0])),
                "rm_um": None if not np.isfinite(v[1]) else float(v[1]),
                "sigma": None if not np.isfinite(v[2]) else float(v[2]),
                "ok": bool(v[3]),
                "method": v[4],
            }
            for k, v in modes.items()
        }
    return payload


def _print_summary(args, solutions, g1, channels, n_avg, g_fit, r_grid,
                   m_avg) -> None:
    print("\n=== Ансамбль решений (перебор m, r0, r1) ===")
    print(f"вариантов перебрано: {len(solutions)}; усредняем: G = {min(args.best, len(solutions))}")
    print(f"{'ранг':>4} {'m':>13} {'степень':>7} {'r0, мкм':>9} {'r1, мкм':>9} "
          f"{'gamma':>10} {'Err':>9}")
    for t, s in enumerate(solutions):
        mark = " <--" if t == 0 else ""
        print(f"{t + 1:>4} {s.m.real:6.3f}{s.m.imag:+7.3f}j {s.degree:>7} "
              f"{s.r0:>9.4g} {s.r1:>9.4g} "
              f"{s.gamma:>10.3g} {s.err:>9.5f}{mark}")
        if t + 1 >= min(args.best, len(solutions)):
            break

    print(f"\nсредний показатель преломления (по {min(args.best, len(solutions))} лучшим): "
          f"m = {m_avg.real:.3f}{m_avg.imag:+.3f}j")

    print("\n=== Средняя функция распределения (dV/dln r) ===")
    print("ключевые точки (полная сетка — в --json, при задании):")
    idx = np.linspace(0, r_grid.size - 1, min(7, r_grid.size)).astype(int)
    for i in idx:
        r = r_grid[i]
        if not np.isfinite(n_avg[i]):
            print(f"  r = {r:9.4g} мкм,  dV/dln r = нет покрытия (вне окон лучших решений)")
            continue
        print(f"  r = {r:9.4g} мкм,  dV/dln r = {float(_per_cm3(n_avg[i])):.4g} мкм³/см³")

    print("\n=== Согласие по каналам (среднее решение) ===")
    for j, ch in enumerate(channels):
        rel = abs(g_fit[j] - g1[j]) / g1[j] * 100.0
        print(f"  {ch.name:<14} g1 = {g1[j]:.6g}   g = {g_fit[j]:.6g}   "
              f"|g-g1|/g1 = {rel:.2f}%")
    err_avg = float(np.sum(np.abs(np.asarray(g_fit) - g1) / g1))
    print(f"  Err (среднее решение) = {err_avg:.5f}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="solve",
        description="Восстановление функции распределения частиц по размерам "
                    "из лидарных коэффициентов (B-сплайны + МНК с регуляризацией "
                    "по кривизне, ансамбль по m и окнам [r0, r1]).",
    )
    p.add_argument("--data", metavar="JSON", default=None,
                   help="файл с измеренными коэффициентами (см. README); "
                        "без него запускается демонстрация на синтетических данных")
    p.add_argument("--tables", metavar="DIR", default=None,
                   help="табличный режим ядер: корень с каталогами "
                        "m<real>_i<imag>/, внутри — файлы Cext.txt / Cback.txt "
                        "(сечения C(x) в мкм², снятые при λ_ref = 2π мкм); "
                        "по умолчанию — ядра Mie через miepython")
    p.add_argument("--table-extrapolate", action="store_true",
                   help="разрешить экстраполяцию сечения за пределы диапазона x "
                        "в табличном режиме (по умолчанию — ошибка)")
    p.add_argument("--table-ref-lambda-um", type=float, default=2.0 * np.pi,
                   help="длина волны (мкм), при которой посчитаны внешние "
                        "таблицы сечений; сечение пересчитывается на длину волны "
                        "канала как (λ/λ_ref)² [2π]")
    p.add_argument("--batch", metavar="TXT", default=None,
                   help="пакетная обработка: txt-файл со столбцами "
                        "Altitude b<нм>... a<нм>... (по строке на высоту); "
                        "таблица ядер строится один раз на весь пакет")
    p.add_argument("--batch-out-dir", metavar="DIR", default=None,
                   help="каталог для результатов пакетной обработки (.txt, .png); "
                        "по умолчанию — рядом с входным txt (без расширения)")
    p.add_argument("--degree", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="степени B-сплайнов (1..4) [по умолчанию: 1 2 3 4]")
    p.add_argument("--n-basis", metavar="N", type=int, default=None,
                   help="число базисных функций на окно; по умолчанию — число каналов [авто]")
    p.add_argument("--r0-min", type=float, default=0.02, help="нижняя граница окна, мкм [0.02]")
    p.add_argument("--r0-max", type=float, default=0.5, help="верхняя граница нижних границ окна, мкм [0.5]")
    p.add_argument("--nr0", type=int, default=6, help="число значений r0 (лог. сетка) [6]")
    p.add_argument("--r1-min", type=float, default=2.0, help="нижняя граница верхних границ окна, мкм [2]")
    p.add_argument("--r1-max", type=float, default=10.0, help="верхняя граница окна, мкм [10]")
    p.add_argument("--nr1", type=int, default=4, help="число значений r1 (лог. сетка) [4]")
    p.add_argument("--min-ratio", type=float, default=4.0,
                   help="отбрасывать окна с r1 < min_ratio * r0 [4]")
    p.add_argument("--best", type=int, default=5, help="число лучших решений для усреднения G [5]")
    p.add_argument("--gamma-min", type=float, default=-7.0,
                   help="мин. показатель gamma (log10, отн. ||A||_2) [-7]")
    p.add_argument("--gamma-max", type=float, default=5.0,
                   help="макс. показатель gamma [5]")
    p.add_argument("--ngamma", type=int, default=41, help="число значений gamma [41]")
    p.add_argument("--gamma-select", choices=["fixed", "cv", "err"], default="fixed",
                   help="правило выбора gamma в окне: fixed — фиксированный gamma_rel "
                        "(рекомендуется), cv — LOO-кросс-валидация, err — минимум невязки "
                        "Err [fixed]")
    p.add_argument("--gamma-rel", type=float, default=0.1,
                   help="фиксированный gamma относительно ||A||_2 (для --gamma-select fixed) [0.1]")
    p.add_argument("--nonneg", action="store_true",
                   help="накладывать ограничение C >= 0 (тогда n(r) >= 0)")
    p.add_argument("--no-zero-boundary", action="store_true",
                   help="не накладывать граничные условия n(r0)=n(r1)=0 "
                        "(по умолчанию крайние базисные функции исключаются)")
    p.add_argument("--no-smooth", action="store_true",
                   help="не сглаживать среднее по лучшим решениям "
                        "(по умолчанию — сглаживающий сплайн)")
    p.add_argument("--smooth-lam", type=float, default=1e-4,
                   help="относительный параметр сглаживания среднего "
                        "(нормированная шкала, безразмерный) [1e-4]")
    p.add_argument("--r-split", type=float, default=0.5,
                   help="граница разделения fine/coarse мод для анализа, мкм [0.5]")
    p.add_argument("--modes", choices=["fit", "stat", "auto"], default="auto",
                   help="способ расчёта параметров мод (fine/coarse/overall): "
                        "fit — фит логнормалью, stat — статистическая "
                        "(моментная) оценка, auto — фит, при неудаче моменты [auto]")
    p.add_argument("--absolute", action="store_true",
                   help="МНК по абсолютным невязкам (по умолчанию веса 1/g1)")
    p.add_argument("--dx-max", type=float, default=0.1,
                   help="шаг параметра размера x в таблице ядер [0.1]")
    p.add_argument("--no-jit", action="store_true",
                   help="отключить numba-бэкенд miepython: таблицы Mie-ядер "
                        "считаются чистым Python и работают заметно медленнее "
                        "(по умолчанию JIT включён, если доступен numba)")
    p.add_argument("--quad-order", type=int, default=24,
                   help="порядок Гаусса–Лежандра на подсегменте [24]")
    p.add_argument("--log-span", type=float, default=0.1,
                   help="макс. ширина подсегмента в ln r [0.1]")
    p.add_argument("--m-real", type=float, default=1.45,
                   help="Re(m) истинного распределения для демо [1.45]")
    p.add_argument("--m-imag", type=float, default=-0.005,
                   help="Im(m) истинного распределения для демо [-0.005]")
    p.add_argument("--lidar-ratio-cons", action="store_true",
                   help="ограничить плавность лидарного отношения S = α/β с высотой: "
                        "переранжирование решений по отклонению от сглаженного по "
                        "высотам опорного профиля S (только в --batch) [выкл.]")
    p.add_argument("--lidar-ratio-w", type=float, default=0.5,
                   help="вес штрафа за отклонение лидарного отношения от опорного "
                        "профиля в ранге решений (score = Err + w·Δln S) [0.5]")
    p.add_argument("--lidar-ratio-iter", type=int, default=3,
                   help="макс. число итераций уточнения опорного профиля лидарного "
                        "отношения [3]")
    p.add_argument("--m-reals", nargs="+", type=float, default=None,
                   help="сетка Re(m) для перебора [демо: 1.40 1.45 1.50; данные: сетка вокруг m из файла]")
    p.add_argument("--m-imags", nargs="+", type=float, default=None,
                   help="сетка Im(m) для перебора [демо: -0.01 -0.005 0; данные: вокруг m из файла]")
    p.add_argument("--noise", type=float, default=0.0,
                   help="относительный шум на демонстрационные g1 [0]")
    p.add_argument("--seed", type=int, default=1, help="зерно генератора шума [1]")
    p.add_argument("--out", metavar="PNG", default="result.png",
                   help="файл графика [result.png]")
    p.add_argument("--json", metavar="JSON", default=None,
                   help="сохранить результаты в JSON")
    p.add_argument("--no-plot", action="store_true", help="не строить график")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.lidar_ratio_w < 0:
        raise SystemExit("ошибка: --lidar-ratio-w должен быть >= 0")
    if args.lidar_ratio_iter < 1:
        raise SystemExit("ошибка: --lidar-ratio-iter должен быть >= 1")
    if args.lidar_ratio_cons and args.batch is None:
        print("[предупреждение] --lidar-ratio-cons действует только в пакетном "
              "режиме (--batch); игнорируется")

    if args.no_jit:
        set_mie_jit(False)

    if args.batch is not None:
        from .batch import run_batch
        try:
            run_batch(args)
        except FileNotFoundError as exc:
            raise SystemExit(f"ошибка таблиц: {exc}") from None
        return

    degrees = sorted(set(args.degree))
    if any(k < 1 or k > 4 for k in degrees):
        raise ValueError("степень сплайна должна быть в диапазоне 1..4")

    g1: np.ndarray
    if args.data is not None:
        with open(args.data, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        m, channels, g1 = channels_from_data(data)
        n_true = None
        print(f"данные: {args.data}; каналов: {len(channels)}")
    else:
        m = complex(args.m_real, args.m_imag)
        channels = (
            [Channel("beta", wl_nm * 1e-3) for wl_nm in DEMO_BETA_WL_NM]
            + [Channel("alpha", wl_nm * 1e-3) for wl_nm in DEMO_ALPHA_WL_NM]
        )
        n_true = demo_true_distribution
        g1 = np.zeros(0)  # будет вычислено ниже (демо)
        print("демонстрация на синтетических данных, m =", m)
    for ch in channels:
        print(f"  канал: {ch.name}")
    print("входные единицы: β [1/(м·ср)], α [1/м]")
    bc = "выключено (--no-zero-boundary)" if args.no_zero_boundary else "включено"
    print(f"граничные условия n(r0)=n(r1)=0 (каждое элементарное решение): {bc}")
    sm = "выключено (--no-smooth)" if args.no_smooth else \
        f"включено (rel-lam={args.smooth_lam:g})"
    print(f"сглаживание среднего по лучшим решениям: {sm}")
    nb = args.n_basis if args.n_basis is not None else len(channels)
    if not args.no_zero_boundary:
        print(f"базисных функций на окно: {nb} (с ГУ неизвестных: {nb - 2})")
    else:
        print(f"базисных функций на окно: {nb}")
    print(f"показатель преломления m (исходный/истинный) = {m.real:g} {m.imag:+g}j")

    # ---- сетка показателя преломления для перебора ----
    if args.m_reals is None or args.m_imags is None:
        if args.data is not None:
            mi = -abs(m.imag)
            m_reals = [m.real * f for f in (0.95, 1.0, 1.05)]
            m_imags = [mi, mi / 2.0, 0.0]
        else:
            m_reals = [1.40, 1.45, 1.50]
            m_imags = [-0.01, -0.005, 0.0]
        m_reals = sorted(set(m_reals))
        m_imags = sorted(set(m_imags))
    else:
        m_reals = sorted(set(args.m_reals))
        m_imags = sorted(set(args.m_imags))
    m_grid = [complex(re_, im_) for re_ in m_reals for im_ in m_imags]
    if not m_grid:
        raise SystemExit("сетка показателя преломления пуста")
    print("сетка показателя преломления (перебор):")
    for mv in m_grid:
        print(f"  m = {mv.real:.3f}{mv.imag:+.3f}j")

    try:
        print(f"строим таблицы ядер ({len(m_grid)} значений m)...")
        if args.tables is None:
            r1_max_eff = args.r1_max
            print(f"  Mie-бэкенд: {'numba (JIT)' if mie_jit_enabled() else 'чистый Python'}")
        else:
            # табличный режим: покрытие по x ограничивает верхнюю границу окон
            caps = []
            for mv in m_grid:
                _chs, cap = load_table_channels(args.tables, mv, channels,
                                                args.table_extrapolate,
                                                ref_wl_um=args.table_ref_lambda_um)
                caps.append(cap)
            r1_cap = float(min(caps))
            r1_max_eff = min(args.r1_max, r1_cap)
            if r1_max_eff < args.r1_min:
                raise SystemExit(
                    f"покрытие таблиц по x ограничивает r1_max = {r1_max_eff:g} мкм < "
                    f"r1_min = {args.r1_min:g} — расширьте таблицы или уменьшите --r1-min")
            print(f"табличный режим ядер (внешние сечения), корень: {args.tables}")
            print(f"  λ_ref таблиц = {args.table_ref_lambda_um:g} мкм "
                  f"(сечения пересчитываются как (λ/λ_ref)²)")
            _chs0, _ = load_table_channels(args.tables, m_grid[0], channels,
                                           args.table_extrapolate,
                                           ref_wl_um=args.table_ref_lambda_um)
            for ch in _chs0:
                print(f"  {ch.name:<14} x_max = {ch.table.x_max:g}, "
                      f"r_max = {ch.table.x_max * ch.wl_um / (2 * np.pi):.3g} мкм")
            print(f"  эффективный r1_max = {r1_max_eff:g} мкм "
                  f"(задан: {args.r1_max:g}) — окна и сетка r ограничены")
        tables: dict = {}
        for mv in m_grid:
            if args.tables is not None:
                chs_tbl, _ = load_table_channels(args.tables, mv, channels,
                                                 args.table_extrapolate,
                                                 ref_wl_um=args.table_ref_lambda_um)
                tables[mv] = KernelTable(mv, chs_tbl, args.r0_min, r1_max_eff,
                                         dx_max=args.dx_max)
            else:
                tables[mv] = KernelTable(mv, channels, args.r0_min, r1_max_eff,
                                         dx_max=args.dx_max)
        n_pts = tables[m_grid[0]].r.size
        print(f"  таблица: {n_pts} точек по r на каждое m")
    except FileNotFoundError as exc:
        raise SystemExit(f"ошибка таблиц: {exc}") from None

    if args.data is None:
        # таблица для «истинного» m демо (переиспользуем, если есть в сетке)
        true_kernels = tables.get(m)
        if true_kernels is None:
            print(f"  строим таблицу для истинного m демо: m = {m.real:g}{m.imag:+g}j")
            if args.tables is not None:
                chs_tbl, _ = load_table_channels(args.tables, m, channels,
                                                 args.table_extrapolate,
                                                 ref_wl_um=args.table_ref_lambda_um)
                true_kernels = KernelTable(m, chs_tbl, args.r0_min, r1_max_eff,
                                           dx_max=args.dx_max)
            else:
                true_kernels = KernelTable(m, channels, args.r0_min, r1_max_eff,
                                           dx_max=args.dx_max)
        g1 = _demo_measurements(args, channels, true_kernels, r1_max_eff)
        print("измеренные коэффициенты g1:", np.array2string(g1, precision=5))

    r0_grid = np.geomspace(args.r0_min, args.r0_max, args.nr0)
    r1_grid = np.geomspace(args.r1_min, r1_max_eff, args.nr1)

    print("перебор (m, r0, r1, степень)...")
    try:
        solutions, _scale = scan_variants(
            channels, tables, g1, degrees, r0_grid, r1_grid,
            min_ratio=args.min_ratio,
            gamma_exp_min=args.gamma_min, gamma_exp_max=args.gamma_max,
            n_gamma=args.ngamma,
            weights=None,
            nonneg=args.nonneg,
            relative=not args.absolute,
            gamma_select=args.gamma_select,
            gamma_rel_fixed=args.gamma_rel if args.gamma_select == "fixed" else None,
            zero_boundary=not args.no_zero_boundary,
            n_basis=args.n_basis,
            order=args.quad_order, max_log_span=args.log_span,
        )
    except ValueError as exc:
        raise SystemExit(f"ошибка: {exc}") from None
    if not solutions:
        raise SystemExit("нет ни одного допустимого окна [r0, r1] — "
                         "проверьте границы и min-ratio")

    r_grid = np.geomspace(args.r0_min, r1_max_eff, 256)
    n_avg, n_best, m_avg = ensemble_average(solutions, args.best, r_grid,
                                            smooth=not args.no_smooth,
                                            lam_rel=args.smooth_lam)
    # сырое (несглаженное) среднее — для сравнения на графике и в JSON
    n_avg_raw = ensemble_average(solutions, args.best, r_grid, smooth=False)[0]

    # расчётные g для усреднённых решения и показателя преломления:
    # ядра усредняются по G лучшим решениям (по их m), распределение —
    # объёмное F = dV/dln r, поэтому ядра умножаются на q(r) = 3/(4π r⁴)
    r_q, w_q = quadrature_grid(np.array([args.r0_min, r1_max_eff]),
                               args.quad_order, args.log_span)
    n_q = np.interp(r_q, r_grid, np.nan_to_num(n_avg, nan=0.0))
    g_best = solutions[: max(1, min(args.best, len(solutions)))]
    kern_sum = np.zeros((len(channels), r_q.size))
    for s in g_best:
        tbl = tables[s.m]
        for j in range(len(channels)):
            kern_sum[j] += tbl.kernel(j, r_q)
    q_vals = dvol_dlnr_factor(r_q)
    g_fit = np.empty(len(channels))
    for j in range(len(channels)):
        g_fit[j] = np.sum(w_q * (kern_sum[j] / len(g_best)) * q_vals * n_q)

    names = [ch.name for ch in channels]
    _print_summary(args, solutions, g1, channels, n_avg, g_fit, r_grid, m_avg)

    moments = size_moments(n_avg, r_grid)
    _print_moments(moments)

    # кратность покрытия точек сетки лучшими решениями — веса анализа мод
    count = np.zeros(r_grid.size)
    for s in g_best:
        count += (r_grid >= s.r0) & (r_grid <= s.r1)
    modes = mode_analysis(r_grid, n_avg, r_split=args.r_split, w=count,
                          method=args.modes)
    _print_modes(modes, args.r_split, method=args.modes)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_json_payload(r_grid, n_avg, n_best, solutions,
                                    channels, g1, g_fit, m_avg, moments,
                                    n_avg_raw=n_avg_raw, modes=modes),
                      fh, indent=2)
        print(f"результаты сохранены: {args.json}")

    if not args.no_plot:
        plot_results(args.out, r_grid=r_grid, n_true=n_true(r_grid) if n_true else None,
                     n_avg=n_avg, n_avg_raw=n_avg_raw, n_best=n_best,
                     solutions=solutions[: args.best],
                     names=names, g1=g1, g_fit=np.asarray(g_fit), m_avg=m_avg)
