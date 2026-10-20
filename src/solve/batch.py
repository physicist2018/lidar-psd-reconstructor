"""Пакетная обработка: восстановление распределений частиц по размерам
для нескольких высот из одного txt-файла измерений.

Формат входного txt (первая строка — заголовок, далее по строке на высоту):

    Altitude  b355  b532  b1064 a355  a532
    100       9.879211342818602e-06 1.06715705204743e-05 7.002700027432126e-06 0.000302238873845872 0.000327611920803267
    ...

Имена каналов:
    b<Нм>  → обратное рассеяние β (backscatter), 1/(м·ср);
    a<Нм>  → ослабление α (extinction), 1/м.
Единицы совпадают с JSON-данными (числа в примере прямо соответствуют
``input_example.json``).

Основная идея (согласно заданию):
  1. таблица ядер (Mie) строится ОДИН раз на весь пакет — она зависит только
     от показателя преломления m, набора каналов и диапазона r, но не от
     измеренных коэффициентов, поэтому переиспользуется для всех высот;
  2. для каждой высоты по своим g1 выполняется поиск решения (перебор
     (m, r0, r1, степень) + ансамбль лучших) — весь переиспользуемый
     механизм из ``solve.scan_variants`` / ``solve.ensemble_average``;
  3. на выходе создаются файлы (в ``--batch-out-dir``):
       dist_<H>.txt     — функция распределения по высоте (шапка с высотой
                          и входной строкой);
       moments.txt      — интегральные характеристики по каждой высоте;
       refractive.txt   — показатель преломления по каждой высоте;
       errors.txt       — относительные ошибки восстановления по каналам;
       plot_<H>.png     — график распределения и согласия по каждому каналу.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .basis import quadrature_grid
from .inverse import ensemble_average, scan_variants
from .kernels import Channel, KernelTable, dvol_dlnr_factor

CM3_PER_M3 = 1e6


@dataclass
class BatchChannel:
    """Канал из заголовка txt: kind ('beta'/'alpha') и длина волны в мкм."""

    kind: str
    wl_um: float
    name: str


def parse_batch_txt(path: str):
    """Разбор txt-файла пакетной обработки.

    Возвращает (channels, altitudes, data), где ``channels`` — список
    ``BatchChannel`` (каналы пакета: они одинаковы для всех высот),
    ``altitudes`` — массив высот (м), ``data`` — матрица (n_altitudes,
    n_channels) измеренных коэффициентов (β и α в единицах входного файла).
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"файл {path} пуст")

    header = lines[0].split()
    if not header or header[0].lower() not in ("altitude", "alt", "h"):
        raise ValueError(
            f"первое поле заголовка должно быть 'Altitude', получено {header[:1]!r}")

    channels: list[BatchChannel] = []
    for name in header[1:]:
        kind = None
        wl_nm = 0
        if name[:1].lower() == "b" and name[1:].isdigit():
            kind, wl_nm = "beta", int(name[1:])
        elif name[:1].lower() == "a" and name[1:].isdigit():
            kind, wl_nm = "alpha", int(name[1:])
        if kind is None:
            raise ValueError(f"неизвестное имя канала {name!r} "
                             f"(ожидается b<нм> или a<нм>)")
        channels.append(BatchChannel(kind, wl_nm * 1e-3, name))

    altitudes, rows = [], []
    for ln in lines[1:]:
        toks = ln.split()
        try:
            altitudes.append(float(toks[0]))
            rows.append([float(t) for t in toks[1:]])
        except (ValueError, IndexError):
            raise ValueError(f"не удалось разобрать строку {ln!r}") from None
    A = np.asarray(rows, dtype=float)
    if not channels:
        raise ValueError("в заголовке нет ни одного канала (ожидается b/a<Нм>)")
    n_cols = A.shape[-1]
    if A.ndim != 2 or n_cols != len(channels):
        raise ValueError(
            f"ожидалось {len(channels)} коэффициентов на строку, "
            f"получено {n_cols if A.ndim == 2 else '—'}")
    return channels, np.asarray(altitudes, dtype=float), A


def _m_grid_from_args(args):
    """Декартово произведение сетки Re(m)×Im(m) (едино для всех высот)."""
    def sort_vals(vals):
        return sorted(set(float(v) for v in vals))

    if args.m_reals is not None:
        m_reals = sort_vals(args.m_reals)
    else:
        m_reals = sort_vals(args.m_real * f for f in (0.95, 1.0, 1.05))
    if args.m_imags is not None:
        m_imags = sort_vals(args.m_imags)
    else:
        mi = -abs(args.m_imag)
        m_imags = sort_vals([mi, mi / 2.0, 0.0])
    return [complex(re_, im_) for re_ in m_reals for im_ in m_imags]


def _build_kernel_tables(args, channels, m_grid):
    """Таблица ядер Mie: строится ОДИН раз на весь пакет.

    Зависит только от (m, каналы, диапазон r) — измеренные g1 не участвуют,
    поэтому таблицы переиспользуются для всех высот.
    """
    solve_channels = [Channel(ch.kind, ch.wl_um) for ch in channels]
    tables = {}
    print(f"строим таблицы ядер Mie ({len(m_grid)} значений m, один раз на пакет)...")
    for mv in m_grid:
        tables[mv] = KernelTable(mv, solve_channels, args.r0_min, args.r1_max,
                                 dx_max=args.dx_max)
    print(f"  таблица: {tables[m_grid[0]].r.size} точек по r на каждое m")
    return solve_channels, tables


def _solve_profile(args, channels, tables, g1):
    """Весь цикл решения для одной высоты: перебор → ансамбль → моменты.

    Использует уже построенные таблицы ядер (``tables`` переиспользуются).
    Возвращает dict со всеми результатами профиля (см. ``_write_profile``).
    """
    from . import mode_analysis, size_moments  # избегаем циклического импорта

    degrees = sorted(set(args.degree))
    r0_grid = np.geomspace(args.r0_min, args.r0_max, args.nr0)
    r1_grid = np.geomspace(args.r1_min, args.r1_max, args.nr1)

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
        raise SystemExit("нет ни одного допустимого окна [r0, r1]")

    r_grid = np.geomspace(args.r0_min, args.r1_max, 256)
    n_avg, n_best, m_avg = ensemble_average(solutions, args.best, r_grid,
                                            smooth=not args.no_smooth,
                                            lam_rel=args.smooth_lam)
    n_avg_raw = ensemble_average(solutions, args.best, r_grid, smooth=False)[0]

    # расчётные g для среднего решения (ядра усредняются по G лучшим)
    r_q, w_q = quadrature_grid(np.array([args.r0_min, args.r1_max]),
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

    moments = size_moments(n_avg, r_grid)
    # кратность покрытия точек сетки лучшими решениями — веса анализа мод
    count = np.zeros(r_grid.size)
    for s in g_best:
        count += (r_grid >= s.r0) & (r_grid <= s.r1)
    modes = mode_analysis(r_grid, n_avg, r_split=args.r_split, w=count)

    return {
        "r_grid": r_grid, "n_avg": n_avg, "n_avg_raw": n_avg_raw,
        "n_best": n_best, "m_avg": m_avg, "g_fit": g_fit, "g1": g1,
        "moments": moments, "modes": modes, "solutions": solutions,
    }


def _per_cm3(v):
    """Объёмная/числовая величина «на м³» → «на см³» (v / 1e6)."""
    return np.asarray(v, dtype=float) / CM3_PER_M3


def _write_profile(out_dir, h: float, channels, prof, g1) -> None:
    """Запись файлов для одной высоты: dist_<H>.txt + строки в общие файлы."""
    header_labels = [ch.name for ch in channels]
    # 1) функция распределения на высоту
    dist_path = os.path.join(out_dir, f"dist_{h:g}.txt")
    with open(dist_path, "w", encoding="utf-8") as fd:
        fd.write(f"# высота H = {h:g} м\n")
        fd.write("# входные данные (β [1/(м·ср)], α [1/м]): " +
                 " ".join(f"{header_labels[j]}={g1[j]:g}"
                          for j in range(len(channels))) + "\n")
        fd.write("# r, мкм  dV/dln r, мкм³/см³\n")
        n_c = _per_cm3(prof["n_avg"])
        for rv, nv in zip(prof["r_grid"], n_c):
            if np.isfinite(nv):
                fd.write(f"{rv:g}  {nv:g}\n")

    # 2) moments.txt — интегральные характеристики (Vf, Vc, V, rm, reff, N, σ)
    moments = prof["moments"]
    modes = prof["modes"]
    Vf, rmf, sgf, _okf, _mf = modes["fine"]
    Vc, rmc, sgc, _okc, _mc = modes["coarse"]
    _Vo, rmo, sgo, _oko, _mo = modes["overall"]
    V_c = moments["vol_conc_um3_per_cm3"]   # уже в мкм³/см³
    N_c = moments["N_cm-3"]
    reff = moments["r_eff_um"]
    # Vf/Vc/V — объёмные концентрации мод (мкм³/см³); rm — медианные радиусы (мкм)
    with open(os.path.join(out_dir, "moments.txt"), "a", encoding="utf-8") as fm:
        fm.write((f"{h:g}  {_per_cm3(Vf):g}  {_per_cm3(Vc):g}  {V_c:g}  "
                  f"{rmf:g}  {rmc:g}  {rmo:g}  {reff:g}  {N_c:g}  "
                  f"{sgf:g}  {sgc:g}  {sgo:g}\n"))

    # 3) refractive.txt — показатель преломления (действительная/мнимая)
    with open(os.path.join(out_dir, "refractive.txt"), "a", encoding="utf-8") as fr:
        fr.write(f"{h:g}  {prof['m_avg'].real:g}  {prof['m_avg'].imag:g}\n")

    # 4) errors.txt — относительные ошибки восстановления по каналам + суммарная
    g1a = np.asarray(g1, dtype=float)
    rel = np.abs(prof["g_fit"] - g1a) / g1a
    err = float(np.sum(rel))
    with open(os.path.join(out_dir, "errors.txt"), "a", encoding="utf-8") as fe:
        fe.write(f"{h:g}  " + "  ".join(f"{v:g}" for v in rel) + f"  {err:g}\n")


def _plot_profile(out_path, channels, g1, prof) -> bool:
    """Компактный график профиля: распределение + согласие по каналам."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[предупреждение] matplotlib недоступен: {exc}")
        return False

    r = prof["r_grid"]
    n_c = _per_cm3(prof["n_avg"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.semilogx(r, np.maximum(n_c, 1e-30), "C0-", lw=2)
    ax1.set_xlabel("r, мкм")
    ax1.set_ylabel("dV/dln r, мкм³/см³")
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    names = [ch.name for ch in channels]
    x = np.arange(len(channels))
    w = 0.36
    g1a = np.asarray(g1, dtype=float)
    ax2.bar(x - w / 2, g1a, w, label="измеренные g1", color="C1")
    ax2.bar(x + w / 2, prof["g_fit"], w, label="расчётные g", color="C2")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_yscale("linear")
    ax2.set_ylabel("коэффициент")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def run_batch(args) -> None:
    """Пакетный режим: читает txt, решает каждую высоту, пишет выходы."""
    channels, altitudes, data = parse_batch_txt(args.batch)
    m_grid = _m_grid_from_args(args)

    print(f"пакетная обработка: {args.batch}")
    print(f"  высот: {len(altitudes)}; каналов: {len(channels)}")
    for ch in channels:
        print(f"  канал: {ch.name} ({ch.kind}, {ch.wl_um * 1e3:g} нм)")
    print("  единицы β [1/(м·ср)], α [1/м]")

    # 1) таблица ядер строится один раз и переиспользуется для всех высот
    solve_channels, tables = _build_kernel_tables(args, channels, m_grid)

    out_dir = args.batch_out_dir or os.path.dirname(os.path.abspath(args.batch))
    os.makedirs(out_dir, exist_ok=True)

    # шапки общих файлов (единицы в комментариях)
    with open(os.path.join(out_dir, "moments.txt"), "w", encoding="utf-8") as fm:
        fm.write("# H  Vf  Vc  V  rmf  rmc  rm  reff  N  "
                 "sigma_f  sigma_c  sigma\n"
                 "# V — мкм³/см³; rm/reff — мкм; N — см⁻³; sigma — безразм.\n")
    with open(os.path.join(out_dir, "refractive.txt"), "w", encoding="utf-8") as fr:
        fr.write("# H  Re(m)  Im(m)\n")
    with open(os.path.join(out_dir, "errors.txt"), "w", encoding="utf-8") as fe:
        fe.write("# H  " + "  ".join(ch.name for ch in channels) + "  Err\n" +
                 "# относительные ошибки |g_fit-g1|/g1 (Err = сумма по каналам)\n")

    # 2) для каждой высоты — поиск решения и запись выходов
    for h, g1 in zip(altitudes, data):
        prof = _solve_profile(args, solve_channels, tables, g1)
        _write_profile(out_dir, h, channels, prof, g1)
        _plot_profile(os.path.join(out_dir, f"plot_{h:g}.png"), channels, g1, prof)
        print(f"  высота {h:g} м: ⟨m⟩ = {prof['m_avg'].real:.3f}"
              f"{prof['m_avg'].imag:+.3f}j, "
              f"N = {prof['moments']['N_cm-3']:.3g} см⁻³")

    print(f"пакетная обработка завершена; результаты: {out_dir}")
