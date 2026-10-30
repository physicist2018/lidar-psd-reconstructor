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
       errors.txt       — относительные ошибки восстановления по каналам
                          и средняя по каналам Err;
       plot_<H>.png     — график распределения и согласия по каждому каналу.

Опциональное ограничение (--lidar-ratio-cons): восстановленное лидарное
отношение S = α/β по общим длинам волн каналов не должно сильно варьировать
с высотой. Ранжирование решений каждой высоты дополняется штрафом за
отклонение S от опорного профиля — сглаженной по высотам оценки S
(при --lidar-ratio-cons создаётся файл lidar_ratios.txt). По умолчанию
ограничение выключено.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .basis import quadrature_grid
from .inverse import build_design, ensemble_average, scan_variants
from .kernels import (
    CBACK_SCALES,
    Channel,
    KernelTable,
    dvol_dlnr_factor,
    load_table_channels,
    mie_jit_enabled,
    table_m_grid,
)

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
    """Сетка m для перебора (единая для всех высот).

    В табличном режиме (``args.tables``) — только значения, для которых есть
    таблицы; ``--m-reals`` / ``--m-imags`` (если заданы) сужают этот набор.
    """
    if args.tables is not None:
        return table_m_grid(args.tables, args.m_reals, args.m_imags)

    def sort_vals(vals):
        return sorted(set(float(v) for v in vals))

    if args.m_reals is not None:
        m_reals = sort_vals(args.m_reals)
    else:
        m_real = 1.45 if args.m_real is None else args.m_real
        m_reals = sort_vals(m_real * f for f in (0.95, 1.0, 1.05))
    if args.m_imags is not None:
        m_imags = sort_vals(args.m_imags)
    else:
        m_imag = -0.005 if args.m_imag is None else args.m_imag
        mi = -abs(m_imag)
        m_imags = sort_vals([mi, mi / 2.0, 0.0])
    return [complex(re_, im_) for re_ in m_reals for im_ in m_imags]


def _build_kernel_tables(args, channels, m_grid):
    """Таблица ядер: строится ОДИН раз на весь пакет.

    Зависит только от (m, каналы, диапазон r) — измеренные g1 не участвуют,
    поэтому таблицы переиспользуются для всех высот. В табличном режиме
    (``args.tables``) ядра берутся из внешних таблиц сечений, а верхняя
    граница окон ограничивается покрытием таблиц по x (см. ``r1_max_eff``).

    Возвращает (solve_channels, tables, r1_max_eff).
    """
    solve_channels = [Channel(ch.kind, ch.wl_um) for ch in channels]
    if args.tables is not None:
        caps = []
        for mv in m_grid:
            _chs, cap = load_table_channels(args.tables, mv, solve_channels,
                                            args.table_extrapolate,
                                            ref_wl_um=args.table_ref_lambda_um,
                                            cback_scale=CBACK_SCALES[args.table_cback])
            caps.append(cap)
        r1_max_eff = min(args.r1_max, float(min(caps)))
        if r1_max_eff < args.r1_min:
            raise SystemExit(
                f"покрытие таблиц по x ограничивает r1_max = {r1_max_eff:g} мкм < "
                f"r1_min = {args.r1_min:g} — расширьте таблицы или уменьшите --r1-min")
        print(f"табличный режим ядер (внешние сечения), корень: {args.tables}")
        print(f"  λ_ref таблиц = {args.table_ref_lambda_um:g} мкм "
              f"(сечения пересчитываются как (λ/λ_ref)²)")
        print(f"  конвенция Cback = {args.table_cback}")
        _chs0, _ = load_table_channels(args.tables, m_grid[0], solve_channels,
                                       args.table_extrapolate,
                                       ref_wl_um=args.table_ref_lambda_um,
                                       cback_scale=CBACK_SCALES[args.table_cback])
        for ch in _chs0:
            print(f"  {ch.name:<14} x_max = {ch.table.x_max:g}, "
                  f"r_max = {ch.table.x_max * ch.wl_um / (2 * np.pi):.3g} мкм")
        print(f"  эффективный r1_max = {r1_max_eff:g} мкм "
              f"(задан: {args.r1_max:g}) — окна и сетка r ограничены")
    else:
        r1_max_eff = args.r1_max

    print(f"строим таблицы ядер ({len(m_grid)} значений m, один раз на пакет)...")
    if args.tables is None:
        print(f"  Mie-бэкенд: {'numba (JIT)' if mie_jit_enabled() else 'чистый Python'}")
    tables = {}
    for mv in m_grid:
        if args.tables is not None:
            chs_tbl, _ = load_table_channels(args.tables, mv, solve_channels,
                                             args.table_extrapolate,
                                             ref_wl_um=args.table_ref_lambda_um,
                                             cback_scale=CBACK_SCALES[args.table_cback])
            tables[mv] = KernelTable(mv, chs_tbl, args.r0_min, r1_max_eff,
                                     dx_max=args.dx_max)
        else:
            tables[mv] = KernelTable(mv, solve_channels, args.r0_min, r1_max_eff,
                                     dx_max=args.dx_max)
    print(f"  таблица: {tables[m_grid[0]].r.size} точек по r на каждое m")
    return solve_channels, tables, r1_max_eff


def _common_lidar_wavelengths(channels):
    """Общие длины волн каналов β и α.

    Возвращает список ``(wl_um, idx_beta, idx_alpha)`` — индексы каналов,
    у которых есть и обратное рассеяние, и ослабление на одной длине волны
    (по ним определяется лидарное отношение S = α/β). Индексы соответствуют
    порядку каналов пакета (совпадает с порядком ``g_fit`` решений).
    """
    beta = {ch.wl_um: j for j, ch in enumerate(channels) if ch.kind == "beta"}
    alpha = {ch.wl_um: j for j, ch in enumerate(channels) if ch.kind == "alpha"}
    return [(wl, beta[wl], alpha[wl]) for wl in sorted(set(beta) & set(alpha))]


def _lidar_ratio(g_fit, common) -> np.ndarray:
    """Лидарное отношение S = α/β восстановленных коэффициентов по общим λ.

    ``g_fit`` — рассчитанные коэффициенты (``Solution.g_fit`` или
    ``prof['g_fit']``) в порядке каналов пакета. Если α ≤ 0 или β ≤ 0 —
    отношение NaN (недопустимое).
    """
    out = np.empty(len(common))
    for j, (_wl, ib, ia) in enumerate(common):
        b, a = float(g_fit[ib]), float(g_fit[ia])
        out[j] = a / b if (b > 0 and a > 0) else np.nan
    return out


LIDAR_REF_LAM = 1.0  # сглаживание опорного профиля S (нормированная шкала высоты)


def _windowed_median_ref(S: np.ndarray) -> np.ndarray:
    """Скользящая медиана матрицы S (n_alt × n_λ) по высоте.

    Окно — соседняя высота слева и справа (ширина 3, у краёв сужается).
    NaN-устойчива (np.nanmedian по столбцам окна).
    """
    S = np.asarray(S, dtype=float)
    n_alt = S.shape[0]
    out = np.full_like(S, np.nan)
    for h in range(n_alt):
        lo, hi = max(0, h - 1), min(n_alt, h + 2)
        out[h] = np.nanmedian(S[lo:hi], axis=0)
    return out


def _smooth_ref_profile(altitudes, S: np.ndarray) -> np.ndarray:
    """Опорный профиль S_ref(h, λ): сглаживание оценки S по высоте.

    При >= 5 высот — кубический сглаживающий сплайн по нормированной высоте
    (NaN-точки исключаются, при неудаче — скользящая медиана); при меньшем
    числе высот или неположительной шкале — скользящая медиана
    (``_windowed_median_ref``).
    """
    altitudes = np.asarray(altitudes, dtype=float)
    S = np.asarray(S, dtype=float)
    n_alt, n_l = S.shape

    def fallback():
        return _windowed_median_ref(S)

    if n_alt < 5:
        return fallback()
    t = altitudes - altitudes[0]
    span = t[-1] - t[0]
    if span <= 0:
        return fallback()
    tn = t / span

    from scipy.interpolate import make_smoothing_spline

    out = np.full_like(S, np.nan)
    for li in range(n_l):
        y = S[:, li]
        ok = np.isfinite(y)
        if ok.sum() < 5:
            out[:, li] = _windowed_median_ref(S[:, [li]])[:, 0]
            continue
        scale = float(np.max(np.abs(y[ok])))
        if not (scale > 0 and np.isfinite(scale)):
            out[:, li] = _windowed_median_ref(S[:, [li]])[:, 0]
            continue
        try:
            spl = make_smoothing_spline(tn[ok], y[ok] / scale, lam=LIDAR_REF_LAM)
            out[ok, li] = spl(tn[ok]) * scale
        except Exception:
            out[:, li] = _windowed_median_ref(S[:, [li]])[:, 0]
    return out


def _apply_lidar_ratio_constraint(solutions_per_alt, altitudes, common, args) -> None:
    """Ограничение плавности лидарного отношения S = α/β с высотой.

    Переранжирование решений каждой высоты на месте (списки
    ``solutions_per_alt[h]`` сортируются по новому рангу):

        score = Err + w · mean_λ |ln S_сол(λ) − ln S_ref(h, λ)|,

    где S_ref — опорный профиль, сглаженная по высотам оценка S. Оценка
    S для высоты — медиана S по top-K решениям-кандидатам (K = max(3·G, 9));
    итерации обновляют top-K отобранных и опорный профиль (до
    ``args.lidar_ratio_iter`` раундов, досрочно — при стабилизации выбора
    лучших G).
    """
    n_alt = len(altitudes)
    n_l = len(common)
    g_eff = [max(1, min(args.best, len(sols))) for sols in solutions_per_alt]
    k_eff = [min(len(sols), max(3 * g, 9)) for sols, g in zip(solutions_per_alt, g_eff)]

    # S каждого решения каждой высоты ((n_alt, n_sol) списки массивов n_λ)
    s_table = [np.array([_lidar_ratio(s.g_fit, common) for s in sols])
               for sols in solutions_per_alt]

    def estimate_top_k(h: int) -> np.ndarray:
        """Медиана S по top-K решениям высоты (отсортированы по текущему рангу)."""
        pool = s_table[h][:k_eff[h]]
        est = np.full(n_l, np.nan)
        for li in range(n_l):
            v = pool[:, li]
            if np.isfinite(v).any():
                est[li] = float(np.nanmedian(v))
        return est

    def smooth_ref():
        est_mat = np.array([estimate_top_k(h) for h in range(n_alt)])
        ref = _smooth_ref_profile(altitudes, est_mat)
        # остаточные NaN (нет данных для λ) — медиана по столбцу профиля
        for li in range(n_l):
            col = ref[:, li]
            bad = ~np.isfinite(col)
            if bad.any():
                fill = float(np.nanmedian(col))
                if np.isfinite(fill):
                    ref[bad, li] = fill
        return ref

    prev_tops = None
    for it in range(max(1, args.lidar_ratio_iter)):
        ref = smooth_ref()
        for h, sols in enumerate(solutions_per_alt):
            log_ref = np.log(np.maximum(ref[h], 1e-300))
            scores = []
            for s, s_r in zip(sols, s_table[h]):
                log_s = np.log(np.maximum(s_r, 1e-300))
                dev = float(np.nanmean(np.abs(log_s - log_ref)))
                if not np.isfinite(dev):
                    dev = 0.0
                scores.append(s.err + args.lidar_ratio_w * dev)
            order = np.argsort(scores, kind="stable")
            sols[:] = [sols[i] for i in order]
            s_table[h] = s_table[h][order]
        tops = [tuple(id(s) for s in sols[:g])
                for sols, g in zip(solutions_per_alt, g_eff)]
        if tops == prev_tops:
            break
        prev_tops = tops


def _scan_grids(args, r1_max):
    """Сетки перебора (степени, r0, r1), общие для DesignSet и scan_variants."""
    degrees = sorted(set(args.degree))
    r0_grid = np.geomspace(args.r0_min, args.r0_max, args.nr0)
    r1_grid = np.geomspace(args.r1_min, r1_max, args.nr1)
    return degrees, r0_grid, r1_grid


def _build_design(args, channels, tables, r1_max):
    """Предвычислить матрицы A один раз на весь пакет (все m и окна).

    В отличие от перебора по высотам, матрицы A не зависят от измеренных
    коэффициентов, поэтому решение каждой высоты сводится к МНК по готовым
    A — это основной резерв ускорения пакетного режима.
    """
    degrees, r0_grid, r1_grid = _scan_grids(args, r1_max)
    try:
        return build_design(
            channels, tables, degrees, r0_grid, r1_grid,
            min_ratio=args.min_ratio, n_basis=args.n_basis,
            zero_boundary=not args.no_zero_boundary,
            order=args.quad_order, max_log_span=args.log_span,
        )
    except ValueError as exc:
        raise SystemExit(f"ошибка: {exc}") from None


def _scan_profile(args, channels, tables, g1, r1_max, design=None):
    """Фаза перебора: полный список решений для одной высоты.

    Возвращает (solutions, scale): solutions уже отсортированы по Err.
    Список решений сохраняется между высотами — он нужен для ограничения
    лидарного отношения (переранжирование) и для финализации.

    ``design`` — общий для всех высот DesignSet (строится один раз в
    ``run_batch``); если не задан, строится здесь.
    """
    degrees, r0_grid, r1_grid = _scan_grids(args, r1_max)

    print("перебор (m, r0, r1, степень)...")
    try:
        solutions, scale = scan_variants(
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
            design=design,
        )
    except ValueError as exc:
        raise SystemExit(f"ошибка: {exc}") from None
    if not solutions:
        raise SystemExit("нет ни одного допустимого окна [r0, r1]")
    return solutions, scale


def _finalize_profile(args, channels, tables, g1, solutions, r1_max):
    """Фаза финализации одной высоты: ансамбль, моменты, моды.

    Принимает уже отобранные (отсортированные) решения ``solutions`` —
    начиная с них (``solutions[:best]``) строится усреднённое распределение.
    Возвращает dict со всеми результатами профиля (см. ``_write_profile``).
    """
    from . import mode_analysis, size_moments  # избегаем циклического импорта

    r_grid = np.geomspace(args.r0_min, r1_max, args.nr_out)
    n_avg, n_best, m_avg = ensemble_average(solutions, args.best, r_grid,
                                            smooth=not args.no_smooth,
                                            lam_rel=args.smooth_lam)
    n_avg_raw = ensemble_average(solutions, args.best, r_grid, smooth=False)[0]

    # расчётные g для среднего решения (ядра усредняются по G лучшим)
    r_q, w_q = quadrature_grid(np.array([args.r0_min, r1_max]),
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
    modes = mode_analysis(r_grid, n_avg, r_split=args.r_split, w=count,
                          method=args.modes)

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

    # 4) errors.txt — относительные ошибки по каналам + средняя по каналам
    g1a = np.asarray(g1, dtype=float)
    rel = np.abs(prof["g_fit"] - g1a) / g1a
    err = float(np.mean(rel))
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


def _emit_profile(args, out_dir, h, channels, g1, solutions, tables, r1_max,
                  common, write_lidar) -> None:
    """Финализация одной высоты: распределение, график, строки общих файлов."""
    prof = _finalize_profile(args, channels, tables, g1, solutions, r1_max)
    _write_profile(out_dir, h, channels, prof, g1)
    _plot_profile(os.path.join(out_dir, f"plot_{h:g}.png"), channels, g1, prof)
    s_str = ""
    if write_lidar:
        s_vals = _lidar_ratio(prof["g_fit"], common)
        with open(os.path.join(out_dir, "lidar_ratios.txt"), "a",
                  encoding="utf-8") as fl:
            fl.write(f"{h:g}  " + "  ".join(
                f"{v:g}" if np.isfinite(v) else "nan" for v in s_vals) + "\n")
        s_str = "  S = " + ", ".join(
            f"{v:g}" if np.isfinite(v) else "nan" for v in s_vals)
    print(f"  высота {h:g} м: ⟨m⟩ = {prof['m_avg'].real:.3f}"
          f"{prof['m_avg'].imag:+.3f}j, "
          f"N = {prof['moments']['N_cm-3']:.3g} см⁻³{s_str}")


def run_batch(args) -> None:
    """Пакетный режим: читает txt, решает каждую высоту, пишет выходы."""
    channels, altitudes, data = parse_batch_txt(args.batch)
    m_grid = _m_grid_from_args(args)

    print(f"пакетная обработка: {args.batch}")
    print(f"  высот: {len(altitudes)}; каналов: {len(channels)}")
    for ch in channels:
        print(f"  канал: {ch.name} ({ch.kind}, {ch.wl_um * 1e3:g} нм)")
    print("  единицы β [1/(м·ср)], α [1/м]")
    lr_flag = args.lidar_ratio_cons
    print("ограничение плавности лидарного отношения S=α/β с высотой: "
          + (f"включено (вес {args.lidar_ratio_w:g}, итераций <= {args.lidar_ratio_iter})"
             if lr_flag else "выключено"))

    # 1) таблица ядер строится один раз и переиспользуется для всех высот
    solve_channels, tables, r1_max_eff = _build_kernel_tables(args, channels, m_grid)
    # матрицы A (все m и окна) — тоже один раз на пакет
    design = _build_design(args, solve_channels, tables, r1_max_eff)
    print(f"  матрицы A предвычислены: {design.n_windows} окон × {len(design.m_values)} m")

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
                 "# относительные ошибки |g_fit-g1|/g1 (Err = среднее по каналам)\n")

    common = _common_lidar_wavelengths(channels)
    write_lidar = lr_flag and bool(common)
    # переранжирование по лидарному отношению требует решений всех высот,
    # поэтому в этом случае их приходится держать в памяти целиком
    retain = lr_flag and len(altitudes) >= 2 and bool(common)

    if write_lidar:
        with open(os.path.join(out_dir, "lidar_ratios.txt"), "w", encoding="utf-8") as fl:
            fl.write("# H  " + "  ".join(f"S_{wl * 1e3:g}" for wl, _, _ in common) + "\n")
            fl.write("# восстановленное лидарное отношение S = α/β по общим длинам волн\n")

    if retain:
        # 2) перебор всех высот: решения нужны для ограничения лидарного отношения
        solutions_per_alt = []
        for h, g1 in zip(altitudes, data):
            solutions, _scale = _scan_profile(args, solve_channels, tables, g1,
                                              r1_max_eff, design=design)
            solutions_per_alt.append(solutions)
            print(f"  высота {h:g} м: вариантов {len(solutions)}")

        # 3) ограничение лидарного отношения
        print("  применяем ограничение: общие длины волн "
              + ", ".join(f"{wl * 1e3:g} нм" for wl, _, _ in common))
        _apply_lidar_ratio_constraint(solutions_per_alt, altitudes, common, args)

        # 4) финализация по каждой высоте и запись выходов
        for i, (h, g1) in enumerate(zip(altitudes, data)):
            _emit_profile(args, out_dir, h, channels, g1, solutions_per_alt[i],
                          tables, r1_max_eff, common, write_lidar)
            # списки решений больше не нужны — освобождаем память
            solutions_per_alt[i] = None
    else:
        if lr_flag:
            if len(altitudes) < 2:
                print("  [предупреждение] --lidar-ratio-cons: нужно >= 2 высот; "
                      "ограничение не применяется")
            else:
                print("  [предупреждение] --lidar-ratio-cons: нет общих длин волн "
                      "β∩α для лидарного отношения; ограничение не применяется")
        # высоты независимы — обрабатываем потоково, не копя решения в памяти
        for h, g1 in zip(altitudes, data):
            solutions, _scale = _scan_profile(args, solve_channels, tables, g1,
                                              r1_max_eff, design=design)
            print(f"  высота {h:g} м: вариантов {len(solutions)}")
            _emit_profile(args, out_dir, h, channels, g1, solutions,
                          tables, r1_max_eff, common, write_lidar)

    print(f"пакетная обработка завершена; результаты: {out_dir}")
