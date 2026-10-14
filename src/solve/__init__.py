"""solve — восстановление функции распределения частиц по размерам
из лидарных коэффициентов (обратное рассеяние и ослабление).

Единицы входных данных: β — 1/(м·ср), α — 1/м (ядра приводятся к м²/м²·ср).
Восстановленное n(r) — в м⁻³·мкм⁻¹; дополнительно выводятся интегральные
характеристики: средний объём частиц, объёмная концентрация,
эффективный радиус.

Метод:
  * n(r) ищется в виде разложения по базисным B-сплайнам степени 1..4
    на логарифмически эквидистантной сетке узлов интервала [r0, r1];
    число базисных функций равно числу измеренных коэффициентов M;
  * строится матрица A: A[j, i] = int K_j(r) B_i(r) dr (ядра Mie —
    через miepython, с кэширующей таблицей и кубической интерполяцией);
  * система решается МНК с регуляризацией по кривизне (вторая разность)
    ||W(A C - g1)||^2 + gamma^2 ||L C||^2; параметр gamma выбирается
    фиксированным (gamma_rel·||A||_2, по умолчанию), по LOO-кросс-валидации
    или по минимуму невязки Err = sum_j |g_j - g1_j| / g1_j;
  * перебираются окна [r0, r1] (и степени сплайнов), а также сетка
    показателей преломления m; берутся G решений с наименьшей невязкой,
    по каждому восстанавливается n(r), результат усредняется; среднее
    по G решениям значение m и есть итоговый показатель преломления.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .basis import quadrature_grid
from .inverse import ensemble_average, scan_variants
from .kernels import Channel, KernelTable

__all__ = [
    "Channel", "KernelTable", "scan_variants", "ensemble_average",
    "channels_from_data", "forward_measurements",
    "lognormal", "demo_true_distribution", "main",
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
                         order: int = 24, max_log_span: float = 0.05) -> np.ndarray:
    """Прямое моделирование: g1_j = int_{r_min}^{r_max} K_j(r) n(r) dr

    с плотной составной квадратурой Гаусса–Лежандра.  При ядрах в м²
    и n(r) в м⁻³·мкм⁻¹ коэффициенты beta получаются в 1/(м·ср),
    alpha — в 1/м.
    """
    r_q, w_q = quadrature_grid(np.array([r_min, r_max]), order, max_log_span)
    n_q = np.asarray(n_func(r_q), float)
    out = np.empty(len(channels))
    for j in range(len(channels)):
        out[j] = np.sum(w_q * kernels.kernel(j, r_q) * n_q)
    return out


# ----------------------------------------------------------------------
# интегральные характеристики восстановленного распределения
# ----------------------------------------------------------------------

UM3_TO_M3 = 1e-18  # (1 мкм)^3 = 1e-18 м^3


def size_moments(n_avg, r_grid) -> dict:
    """Интегральные характеристики восстановленного распределения n(r).

    n(r) в м⁻³·мкм⁻¹, r в мкм.  Точки вне покрытия лучших решений
    (n_avg = NaN) дают нулевой вклад; диапазон покрытия и его доля
    возвращаются в ``r_cov_min``, ``r_cov_max``, ``cov_frac``.
    Отрицательные артефакты восстановления отбрасываются (по физическому
    смыслу n(r) ≥ 0).

    Оценки:
      N         — числовая концентрация, м⁻³;
      vol_mean  — средний объём частицы, мкм³;
      vol_conc  — объёмная концентрация, мкм³/м³;
      r_eff     — эффективный радиус, мкм.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    n_arr = np.asarray(n_avg, dtype=float)
    cov = np.isfinite(n_arr)
    n = np.where(cov, n_arr, 0.0)
    n = np.maximum(n, 0.0)  # физическое распределение неотрицательно

    m0 = np.trapezoid(n, r_grid)                  # ∫ n dr            [м⁻³]
    m2 = np.trapezoid(n * r_grid ** 2, r_grid)    # ∫ r² n dr
    m3 = np.trapezoid(n * r_grid ** 3, r_grid)    # ∫ r³ n dr

    vol_conc = (4.0 / 3.0) * np.pi * m3           # мкм³/м³
    n_num = m0 if m0 > 0 else np.nan
    vol_mean = vol_conc / n_num if n_num and np.isfinite(n_num) else np.nan
    r_eff = m3 / m2 if m2 > 0 else np.nan

    r_cov = r_grid[cov]
    r_cov_min = float(r_cov.min()) if r_cov.size else np.nan
    r_cov_max = float(r_cov.max()) if r_cov.size else np.nan

    return {
        "N_m-3": float(n_num),
        "vol_mean_um3": float(vol_mean),
        "vol_conc_um3_per_m3": float(vol_conc),
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
    print(f"  числовая концентрация N      = {mom['N_m-3']:.4g} м⁻³")
    print(f"  средний объём частиц ⟨V⟩    = {mom['vol_mean_um3']:.4g} мкм³ "
          f"({v_m3:.4g} м³)")
    print(f"  объёмная концентрация C_v    = {mom['vol_conc_um3_per_m3']:.4g} мкм³/м³")
    print(f"  эффективный радиус r_eff      = {mom['r_eff_um']:.4g} мкм")


# ----------------------------------------------------------------------
# демонстрационные данные
# ----------------------------------------------------------------------

def lognormal(r, r_g: float, sigma: float) -> np.ndarray:
    """Логнормальное распределение (плотность по радиусу):
    n(r) = exp(-(ln(r/r_g))^2 / (2 ln^2 sigma)) / (sqrt(2 pi) r ln sigma).
    """
    r = np.asarray(r, float)
    s = np.log(sigma)
    return np.exp(-(np.log(r / r_g) ** 2) / (2.0 * s * s)) / (np.sqrt(2.0 * np.pi) * r * s)


def demo_true_distribution(r, amps=(2.4e8, 1.2e8), modes=(0.08, 0.45),
                           sigmas=(1.6, 1.6)) -> np.ndarray:
    """Тестовая «истинная» функция распределения — бимодальное
    логнормальное распределение (Айткеновская и аккумулятивная моды).

    Амплитуды подобраны так, чтобы при физических ядрах (м²) модельные
    коэффициенты были реалистичны: beta ~ 10⁻⁵ 1/(м·ср), alpha ~ 10⁻⁴ 1/м,
    числовая концентрация N ~ 10⁸–10⁹ м⁻³.
    """
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    for a, rg, sg in zip(amps, modes, sigmas):
        out += a * lognormal(r, rg, sg)
    return out


def _demo_measurements(args, channels, kernels: KernelTable):
    """Измеренные коэффициенты для демонстрации (с опциональным шумом)."""
    g1 = forward_measurements(channels, kernels, demo_true_distribution,
                              args.r0_min, args.r1_max)
    if args.noise > 0:
        rng = np.random.default_rng(args.seed)
        g1 = g1 * (1.0 + rng.normal(0.0, args.noise, size=g1.size))
    return g1


# ----------------------------------------------------------------------
# построение графиков
# ----------------------------------------------------------------------

def plot_results(out_path: str, *, r_grid, n_true, n_avg, n_best,
                 solutions, names, g1, g_fit, m_avg) -> bool:
    """Два панно: n(r) (log-log) и сравнение измеренных g1 с расчётными g."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[предупреждение] matplotlib недоступен: {exc}")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    if n_true is not None:
        ax1.semilogx(r_grid, n_true, "k--", lw=1.6, label="истинное n(r)")
    for n_s in n_best:
        ax1.semilogx(r_grid, n_s, color="0.75", lw=1.0)
    ax1.semilogx(r_grid, n_avg, "C0-", lw=2.6, label="среднее по лучшим")
    ax1.set_xlabel("r, мкм")
    ax1.set_ylabel("n(r), отн. ед.")
    ax1.set_title(f"Функция распределения частиц по размерам, "
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
                  m_avg, moments) -> dict:
    return {
        "m_avg": {"real": m_avg.real, "imag": m_avg.imag},
        "moments": moments,
        "units_input": {"beta": "1/(м*ср)", "alpha": "1/м"},
        "channels": [
            {"kind": ch.kind, "wl_nm": ch.wl_um * 1e3,
             "g1": float(g1[j]), "g_fit": float(g_fit[j])}
            for j, ch in enumerate(channels)
        ],
        "r_um": r_grid.tolist(),
        "n_avg": [None if not np.isfinite(v) else float(v) for v in n_avg],
        "best_solutions": [
            {"rank": t + 1, "m_real": s.m.real, "m_imag": s.m.imag,
             "degree": s.degree, "r0": s.r0, "r1": s.r1,
             "gamma": s.gamma, "err": s.err, "C": s.C.tolist()}
            for t, s in enumerate(solutions)
        ],
    }


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

    print("\n=== Средняя функция распределения ===")
    print("ключевые точки n(r) (полная сетка — в --json, при задании):")
    idx = np.linspace(0, r_grid.size - 1, min(7, r_grid.size)).astype(int)
    for i in idx:
        r = r_grid[i]
        if not np.isfinite(n_avg[i]):
            print(f"  r = {r:9.4g} мкм,  n(r) = нет покрытия (вне окон лучших решений)")
            continue
        n1 = r * n_avg[i]
        print(f"  r = {r:9.4g} мкм,  n(r) = {n_avg[i]:.4g},  dN/dln r = {n1:.4g}")

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
    p.add_argument("--degree", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="степени B-сплайнов (1..4) [по умолчанию: 1 2 3 4]")
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
    p.add_argument("--absolute", action="store_true",
                   help="МНК по абсолютным невязкам (по умолчанию веса 1/g1)")
    p.add_argument("--dx-max", type=float, default=0.1,
                   help="шаг параметра размера x в таблице ядер [0.1]")
    p.add_argument("--quad-order", type=int, default=24,
                   help="порядок Гаусса–Лежандра на подсегменте [24]")
    p.add_argument("--log-span", type=float, default=0.1,
                   help="макс. ширина подсегмента в ln r [0.1]")
    p.add_argument("--m-real", type=float, default=1.45,
                   help="Re(m) истинного распределения для демо [1.45]")
    p.add_argument("--m-imag", type=float, default=-0.005,
                   help="Im(m) истинного распределения для демо [-0.005]")
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

    print(f"строим таблицы ядер Mie ({len(m_grid)} значений m, miepython)...")
    tables: dict = {}
    for mv in m_grid:
        tables[mv] = KernelTable(mv, channels, args.r0_min, args.r1_max,
                                 dx_max=args.dx_max)
    n_pts = tables[m_grid[0]].r.size
    print(f"  таблица: {n_pts} точек по r на каждое m")

    if args.data is None:
        # таблица для «истинного» m демо (переиспользуем, если есть в сетке)
        true_kernels = tables.get(m)
        if true_kernels is None:
            print(f"  строим таблицу для истинного m демо: m = {m.real:g}{m.imag:+g}j")
            true_kernels = KernelTable(m, channels, args.r0_min, args.r1_max,
                                       dx_max=args.dx_max)
        g1 = _demo_measurements(args, channels, true_kernels)
        print("измеренные коэффициенты g1:", np.array2string(g1, precision=5))

    r0_grid = np.geomspace(args.r0_min, args.r0_max, args.nr0)
    r1_grid = np.geomspace(args.r1_min, args.r1_max, args.nr1)

    print("перебор (m, r0, r1, степень)...")
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
        order=args.quad_order, max_log_span=args.log_span,
    )
    if not solutions:
        raise SystemExit("нет ни одного допустимого окна [r0, r1] — "
                         "проверьте границы и min-ratio")

    r_grid = np.geomspace(args.r0_min, args.r1_max, 256)
    n_avg, n_best, m_avg = ensemble_average(solutions, args.best, r_grid)

    # расчётные g для усреднённых решения и показателя преломления:
    # ядра усредняются по G лучшим решениям (по их m)
    r_q, w_q = quadrature_grid(np.array([args.r0_min, args.r1_max]),
                               args.quad_order, args.log_span)
    n_q = np.interp(r_q, r_grid, np.nan_to_num(n_avg, nan=0.0))
    g_best = solutions[: max(1, min(args.best, len(solutions)))]
    kern_sum = np.zeros((len(channels), r_q.size))
    for s in g_best:
        tbl = tables[s.m]
        for j in range(len(channels)):
            kern_sum[j] += tbl.kernel(j, r_q)
    g_fit = np.empty(len(channels))
    for j in range(len(channels)):
        g_fit[j] = np.sum(w_q * (kern_sum[j] / len(g_best)) * n_q)

    names = [ch.name for ch in channels]
    _print_summary(args, solutions, g1, channels, n_avg, g_fit, r_grid, m_avg)

    moments = size_moments(n_avg, r_grid)
    _print_moments(moments)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_json_payload(r_grid, n_avg, n_best, solutions,
                                    channels, g1, g_fit, m_avg, moments),
                      fh, indent=2)
        print(f"результаты сохранены: {args.json}")

    if not args.no_plot:
        plot_results(args.out, r_grid=r_grid, n_true=n_true(r_grid) if n_true else None,
                     n_avg=n_avg, n_best=n_best, solutions=solutions[: args.best],
                     names=names, g1=g1, g_fit=np.asarray(g_fit), m_avg=m_avg)
