"""Mie-ядра для прямого моделирования лидарных коэффициентов.

Радиусы r и длины волн задаются в микрометрах, параметр размера
x = 2*pi*r/lambda. Ядра возвращаются в физических единицах СИ:

  K_ext(r, lambda, m) = pi * R^2 * qext            [м^2]      — ослабление
  K_back(r, lambda, m) = (R^2 / 4) * qback          [м^2/ср]   — обратное
                                                                 рассеяние

где R — радиус в метрах (R = r_мкм * 1e-6). Благодаря этому измеренные
коэффициенты в принятых единицах (бета [1/(м*ср)], альфа [1/м]) дают
восстановленное распределение n(r) в единицах [м^-3 * мкм^-1]
(концентрация частиц на кубометр на единицу радиуса в микрометрах).

Соотношение для K_back следует из определения qback в miepython:
qback = |sum_n (2n+1) (-1)^n (a_n - b_n)|^2 / x^2 = 4 |S(pi)|^2 / x^2,
а C_back = |S(pi)|^2 / k^2, отсюда C_back = qback * r^2 / 4.
Проверено численно на рэлеевском пределе.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
from scipy.interpolate import CubicSpline

# numba-бэкенд miepython ускоряет расчёт таблиц Mie-ядер примерно в 45 раз
# (замеры: ~26 с → ~0.55 с на демонстрационном наборе). В miepython 3.x он
# выключен по умолчанию и включается переменной MIEPYTHON_USE_JIT, которая
# читается при импорте miepython, — поэтому выставляем её ДО `import miepython`.
# Если numba недоступен (например, Emscripten), остаётся чистый Python.
if importlib.util.find_spec("numba") is not None:
    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")

import miepython as mie

# (1 мкм)^2 = 1e-12 м^2 — перевод сечения из мкм^2 в м^2
UM2_TO_M2 = 1e-12

# Эталонная длина волны внешних таблиц сечений (мкм). Таблица задаёт C(x) при
# некоторой фиксированной λ_ref; поскольку сечение ∝ λ², при использовании на
# длине волны канала λ оно пересчитывается как C·(λ/λ_ref)². По умолчанию
# λ_ref = 2π мкм: тогда абсцисса таблицы x = 2πr/λ_ref численно равна радиусу r
# (в мкм). Переопределяется флагом --table-ref-lambda-um.
TABLE_REF_WL_UM = 2.0 * np.pi

# Конвенции записи столбца Cback во внешних таблицах:
#   "dsigma" — Cback = dσ/dΩ(180°) в мкм²/ср: так пишет mkbase из ADDA
#              (s11(θ) = dσ/dΩ, проверено по ∫s11 dΩ = Csca) и так считает
#              miepython (Cback = r²·qback/4), поэтому это единица по умолчанию;
#   "4pi"    — Cback = 4π·dσ/dΩ («backscattering cross section» в конвенции
#              Борена–Хаффмана): так пишут таблицы сфер mkbase_spheres/miedir
#              (Cback = πx²·qback). Множитель ниже переводит запись к dσ/dΩ.
CBACK_SCALES = {"dsigma": 1.0, "4pi": 1.0 / (4.0 * np.pi)}


def set_mie_jit(use_jit: bool) -> bool:
    """Переключить Mie-бэкенд miepython в рантайме.

    ``use_jit=True`` — numba-бэкенд (быстрый, используется по умолчанию,
    если numba доступен); ``use_jit=False`` — чистый Python (флаг ``--no-jit``).

    Возвращает True, если запрошенный режим применён (False — numba недоступен
    при ``use_jit=True``). Переключается ``single_sphere`` в ``miepython.core``:
    именно её вызывает ``efficiencies_mx``, а через неё — ``Channel.kernel``.
    """
    from miepython import core as mie_core
    if use_jit:
        try:
            from miepython.mie_jit import _single_sphere_nb as impl
        except ImportError:  # numba недоступен
            return False
    else:
        from miepython.mie_nojit import _single_sphere_py as impl
    mie_core.single_sphere = impl
    return True


def mie_jit_enabled() -> bool:
    """Включён ли в данный момент numba-бэкенд miepython."""
    from miepython import core as mie_core
    return getattr(mie_core.single_sphere, "__module__", "").endswith("mie_jit")


def dvol_dlnr_factor(r) -> np.ndarray:
    """Переход от dN/dr к dV/dln r:  n(r) = q(r) * F(r).

    Здесь F = dV/dln r, r в мкм, v(r) = (4π/3) r³ — объём частицы:
      dV/dln r = r * dV/dr = r * (4π/3) r³ * dN/dr = (4π/3) r⁴ n(r),
    откуда n(r) = 3 F(r) / (4π r⁴) = q(r) F(r).
    """
    r = np.asarray(r, dtype=float)
    return 3.0 / (4.0 * np.pi * r ** 4)


class Channel:
    """Один измеряемый канал: ослабление ('alpha') или обратное рассеяние ('beta').

    При ``table`` не None ядро берётся из внешней таблицы сечений
    (``TabularCrossSection``) вместо расчёта Mie: таблица содержит сечения
    в мкм² как функцию безразмерного параметра размера x = 2πr/λ, снятые при
    эталонной длине волны ``table_ref_wl_um`` (по умолчанию 2π мкм), поэтому
    показатель преломления m в расчёте ядра не участвует. Поскольку сечение
    ∝ λ², при использовании на длине волны канала оно домножается на
    (λ/λ_ref)².
    """

    __slots__ = ("kind", "wl_um", "table", "table_ref_wl_um", "cback_scale")

    def __init__(self, kind: str, wl_um: float,
                 table: "TabularCrossSection | None" = None,
                 table_ref_wl_um: float = TABLE_REF_WL_UM,
                 cback_scale: float = 1.0) -> None:
        if kind not in ("alpha", "beta"):
            raise ValueError(f"kind должен быть 'alpha' или 'beta', получено {kind!r}")
        self.kind = kind
        self.wl_um = float(wl_um)
        self.table = table
        self.table_ref_wl_um = float(table_ref_wl_um)
        # только для beta: перевод конвенции Cback таблицы к dσ/dΩ
        self.cback_scale = float(cback_scale)

    @property
    def name(self) -> str:
        prefix = "beta" if self.kind == "beta" else "alpha"
        return f"{prefix}({self.wl_um * 1e3:g} nm)"

    def kernel(self, m: complex, r) -> np.ndarray:
        """Физическое ядро на частицу: alpha — [м^2], beta — [м^2/ср].

        Скорость определяется бэкендом miepython: с numba (по умолчанию —
        см. ``set_mie_jit``) примерно в 45 раз быстрее, чем на чистом Python.
        При ``self.table`` не None используется табличное сечение, снятое при
        эталонной λ_ref: C(x)·(λ/λ_ref)² (мкм² → м²).
        """
        r = np.asarray(r, dtype=float)
        x = 2.0 * np.pi * r / self.wl_um
        if self.table is not None:
            C = self.table.cross_section(x)
            K = UM2_TO_M2 * C * (self.wl_um / self.table_ref_wl_um) ** 2
            if self.kind == "beta":
                K = K * self.cback_scale
            return K
        qext, _qsca, qback, _g = mie.efficiencies_mx(m, x)
        if self.kind == "alpha":
            return UM2_TO_M2 * np.pi * r * r * qext
        return UM2_TO_M2 * r * r * qback / 4.0


class TabularCrossSection:
    """Внешняя таблица сечения: C(x) в мкм² на безразмерный параметр размера x.

    Источник — файл ``data.txt``, создаваемый ``mkbase`` / ``mkbase_spheres``:
    заголовок ``# x Cext Cabs Cback Qext Qabs Qback [n]`` и строки с числовыми
    столбцами. Сечение выбирается по имени колонки из заголовка (``Cext`` для
    ослабления, ``Cback`` для обратного рассеяния), абсцисса — колонка ``x``;
    остальные столбцы не используются. Комментарии ``#`` и пустые строки
    игнорируются, x строго возрастает, C > 0. Интерполяция в логарифмических координатах
    (ln x, ln C) кубическим сплайном — как в ``KernelTable``; на осциллирующих
    факторах Mie кубический сплайн отслеживает рипл лучше shape-preserving
    методов (PCHIP "сглаживает" осцилляции между узлами). При выходе x за
    пределы таблицы по умолчанию выбрасывается ошибка; при
    ``extrapolate=True`` — экстраполяция граничной кривой с предупреждением.

    Единицы: C в мкм². Перевод в м² и пересчёт с эталонной длины волны таблицы
    ``table_ref_wl_um`` на длину волны канала выполняются в ``Channel.kernel``
    (``UM2_TO_M2 = 1e-12``, множитель ``(λ/λ_ref)²``).
    """

    __slots__ = ("x", "C", "x_min", "x_max", "_interp", "extrapolate", "_path")

    def __init__(self, x: np.ndarray, C: np.ndarray, extrapolate: bool = False,
                 path: str | None = None) -> None:
        x = np.asarray(x, dtype=float)
        C = np.asarray(C, dtype=float)
        if x.ndim != 1 or C.ndim != 1 or x.size != C.size or x.size < 3:
            raise ValueError("таблица сечений: нужны одномерные массивы x и C "
                             "одинаковой длины (>= 3 точек)")
        if np.any(np.diff(x) <= 0):
            raise ValueError("таблица сечений: x должен строго возрастать")
        if np.any(C <= 0) or not np.all(np.isfinite(C)):
            raise ValueError("таблица сечений: сечения C должны быть > 0 и конечными")
        # крупный шаг по x не разрешает осцилляции (рипл) факторов Mie
        dx_max = float(np.max(np.diff(x)))
        if dx_max > 1.0:
            warnings.warn(f"таблица {path or '?'}: крупный шаг по x "
                          f"(max Δx = {dx_max:g}); рипл Mie может быть не разрешён",
                          UserWarning, stacklevel=2)
        self.x = x
        self.C = C
        self.x_min = float(x[0])
        self.x_max = float(x[-1])
        self.extrapolate = bool(extrapolate)
        self._path = path
        # Интерполяция в лог-координатах кубическим сплайном (как в ``KernelTable``):
        # для осциллирующих факторов Mie кубический сплайн отслеживает рипл
        # лучше shape-preserving методов (PCHIP зануляет осцилляции между узлами).
        # Границы проверяются отдельно в ``cross_section`` (ошибка по умолчанию).
        self._interp = CubicSpline(np.log(x), np.log(C))

    @classmethod
    def from_data_file(cls, path: str, column: str = "Cext",
                       extrapolate: bool = False) -> "TabularCrossSection":
        """Загрузка одного сечения из ``data.txt`` формата mkbase.

        ``column`` — имя колонки в заголовке (``"Cext"`` или ``"Cback"``).
        Заголовок — строка-комментарий, у которой первый токен после ``#``
        равен ``x``, а нужное имя колонки присутствует как отдельный токен;
        строки вроде ``# Units: ...`` (их пишет ``mkbase_spheres``) на
        заголовок не претендуют, так как начинаются с ``Units:``. Столбцы
        выбираются по имени, а не по позиции, поэтому лишние колонки
        (``n`` в ``average/``) и их порядок значения не имеют.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"файл таблицы сечений не найден: {path}")
        header: list[str] | None = None
        rows: list[list[str]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    toks = s.lstrip("#").split()
                    if header is None and toks and toks[0] == "x" and column in toks:
                        header = toks
                    continue
                rows.append(s.split())
        if header is None:
            raise ValueError(
                f"{path}: не найден заголовок формата mkbase с колонками "
                f"'x' и {column!r} (ожидается, например, "
                f"'# x Cext Cabs Cback Qext Qabs Qback')")
        ix = header.index("x")
        ic = header.index(column)
        xs, cs = [], []
        for toks in rows:
            if len(toks) <= max(ix, ic):
                raise ValueError(
                    f"{path}: в строке нет колонки {column!r}: {' '.join(toks)!r}")
            xs.append(float(toks[ix]))
            cs.append(float(toks[ic]))
        if len(xs) < 3:
            raise ValueError(f"{path}: недостаточно точек (>= 3), найдено {len(xs)}")
        return cls(np.asarray(xs, float), np.asarray(cs, float),
                   extrapolate=extrapolate, path=path)

    def cross_section(self, x) -> np.ndarray:
        """Сечение C(x) в мкм² (интерполяция в log-log)."""
        x = np.asarray(x, dtype=float)
        if self.extrapolate:
            out = np.exp(self._interp(np.log(np.asarray(x, dtype=float))))
        else:
            if np.any(x < self.x_min) or np.any(x > self.x_max):
                raise ValueError(
                    f"параметр размера x вне диапазона таблицы "
                    f"[{self.x_min:g}, {self.x_max:g}] — увеличьте диапазон таблиц "
                    f"или включите --table-extrapolate")
            out = np.exp(self._interp(np.log(x)))
        return out


def m_table_dirname(m: complex) -> str:
    """Имя каталога таблиц для ``m`` в формате mkbase.

    ``mre_<real:.2f>_mim_<imag:.5f>`` — как пишут ``mkbase`` и
    ``mkbase_spheres``. Отрицательный ноль нормализуется, иначе ``mim_0.00000``
    превратился бы в ``mim_-0.00000``.
    """
    re = float(m.real) + 0.0
    im = float(m.imag) + 0.0
    return f"mre_{re:.2f}_mim_{im:.5f}"


def _parse_m_dirname(name: str) -> "complex | None":
    """Разбор имени ``mre_<real>_mim_<imag>``; None, если формат другой."""
    if not name.startswith("mre_") or "_mim_" not in name:
        return None
    re_s, _, im_s = name[len("mre_"):].partition("_mim_")
    try:
        return complex(float(re_s), float(im_s))
    except ValueError:
        return None


def _m_values_str(values, limit: int = 10) -> str:
    """Список m в виде ``1.45+0.0168j, ...`` для сообщений."""
    parts = [f"{z.real:g}{z.imag:+g}j" for z in values[:limit]]
    if len(values) > limit:
        parts.append("...")
    return ", ".join(parts)


def _layout_hint(root: str) -> str:
    """Подсказка, если вместо каталога с m указан корень базы (``base/``)."""
    if not os.path.isdir(root):
        return f"; каталог {root} отсутствует"
    nested = sorted(
        e for e in os.listdir(root)
        if os.path.isdir(os.path.join(root, e))
        and (e == "average" or (e.startswith("shape") and e[5:].isdigit())))
    if not nested:
        return ""
    return (f"; {root} содержит {', '.join(nested)} — укажите "
            f"--tables {os.path.join(root, 'average')} или "
            f"--tables <root>/<shapeNN>")


def available_m_values(root: str) -> "list[complex]":
    """Значения m, для которых в таблицах есть готовый ``data.txt``.

    Сканирует подкаталоги ``mre_<real:.2f>_mim_<imag:.5f>`` в ``root``
    (например ``base/average`` или ``base/shapeNN``) и возвращает их m,
    отсортированные по (Re, Im). Каталоги без ``data.txt`` игнорируются.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"каталог таблиц не найден: {root}")
    values = []
    for name in os.listdir(root):
        mv = _parse_m_dirname(name)
        if mv is None:
            continue
        if os.path.isfile(os.path.join(root, name, "data.txt")):
            values.append(mv)
    return sorted(set(values), key=lambda z: (z.real, z.imag))


def table_m_grid(root: str, m_reals=None, m_imags=None) -> "list[complex]":
    """Сетка m для перебора — только значения, для которых есть таблицы.

    Базовый набор — все ``m`` из ``available_m_values(root)``. Если заданы
    ``m_reals`` / ``m_imags``, набор сужается по Re / Im; сравнение идёт по
    тому же округлению, с которым формируются имена каталогов (2 и 5 знаков),
    т.е. согласовано с ``m_table_dirname``. Пустой результат — ошибка с
    перечнем доступных m.
    """
    available = available_m_values(root)
    if not available:
        raise FileNotFoundError(
            f"в каталоге таблиц {root} нет ни одного подкаталога "
            f"mre_<real>_mim_<imag>/data.txt" + _layout_hint(root))
    grid = available
    if m_reals is not None:
        keys = {f"{float(v):.2f}" for v in m_reals}
        grid = [z for z in grid if f"{z.real:.2f}" in keys]
    if m_imags is not None:
        keys = {f"{float(v):.5f}" for v in m_imags}
        grid = [z for z in grid if f"{z.imag:.5f}" in keys]
    if not grid:
        raise FileNotFoundError(
            f"в таблицах {root} нет m, удовлетворяющих --m-reals/--m-imags; "
            f"доступные m: {_m_values_str(available)}")
    return sorted(set(grid), key=lambda z: (z.real, z.imag))


def _missing_table_error(root: str, mdir: str, m: complex) -> FileNotFoundError:
    """Ошибка с диагностикой: какие m есть и куда указывать ``--tables``."""
    msg = (f"каталог таблиц не найден: {mdir} (ожидается "
           f"{m_table_dirname(m)}/data.txt — формат mkbase/mkbase_spheres)")
    if not os.path.isdir(root):
        return FileNotFoundError(f"{msg}{_layout_hint(root)}")
    available = available_m_values(root)
    if available:
        msg += f"; доступные m в {root}: {_m_values_str(available)}"
    else:
        msg += _layout_hint(root)
    return FileNotFoundError(msg)


def load_table_channels(root: str, m: complex, channels,
                        extrapolate: bool = False,
                        ref_wl_um: float = TABLE_REF_WL_UM,
                        cback_scale: float = 1.0):
    """Загрузка таблиц сечений для показателя преломления ``m``.

    ``root`` — каталог с подкаталогами ``mre_<real:.2f>_mim_<imag:.5f>/``
    (например ``base/average`` или ``base/shapeNN``), в каждом — ``data.txt``
    формата ``mkbase`` / ``mkbase_spheres`` с колонками
    ``x Cext Cabs Cback Qext Qabs Qback [n]``. Из него берутся ``Cext``
    (ослабление) и ``Cback`` (обратное рассеяние) — по одной таблице на
    ``(m, вид)``, общая для всех длин волн каналов. Таблица универсальна по
    ``x`` и снята при эталонной длине волны ``ref_wl_um``, поэтому сечение
    пересчитывается как ``(λ/λ_ref)²``. ``cback_scale`` переводит конвенцию
    столбца ``Cback`` к ``dσ/dΩ`` (см. ``CBACK_SCALES``); для каналов alpha
    не применяется. Возвращает (channels_tbl, r_max_cap) —
    каналы с табличными ядрами и ограничение сверху по радиусу (мкм),
    определяемое покрытием таблиц по x (минимум по каналам:
    r_max = x_max·λ/(2π)).
    """
    mdir = os.path.join(root, m_table_dirname(m))
    path = os.path.join(mdir, "data.txt")
    if not os.path.isfile(path):
        raise _missing_table_error(root, mdir, m)
    out = []
    caps = []
    cache: dict[str, TabularCrossSection] = {}
    for ch in channels:
        column = "Cext" if ch.kind == "alpha" else "Cback"
        if column not in cache:
            cache[column] = TabularCrossSection.from_data_file(
                path, column, extrapolate=extrapolate)
        tbl = cache[column]
        out.append(Channel(ch.kind, ch.wl_um, table=tbl,
                           table_ref_wl_um=ref_wl_um,
                           cback_scale=cback_scale if ch.kind == "beta" else 1.0))
        caps.append(tbl.x_max * ch.wl_um / (2.0 * np.pi))
    return out, float(min(caps))


class KernelTable:
    """Таблица ядер с кубической интерполяцией на плотной сетке по r.

    Шаг сетки выбирается так, чтобы шаг параметра размера x для самой
    короткой длины волны не превосходил ``dx_max`` — этого достаточно,
    чтобы разрешить осцилляции (рипл) факторов эффективности Mie.
    Таким образом miepython вызывается один раз на канал, а квадратуры
    для всех окон [r0, r1] используют быструю интерполяцию.

    Интерполяция выполняется в логарифмических координатах (ln r, ln K):
    ядра меняются на порядки по r, но в ln-координатах гладкие.
    """

    def __init__(self, m: complex, channels, r_min: float, r_max: float,
                 dx_max: float = 0.05) -> None:
        lam_min = min(ch.wl_um for ch in channels)
        dr = dx_max * lam_min / (2.0 * np.pi)
        r = np.arange(r_min, r_max, dr)
        if r.size == 0 or r[-1] < r_max - 0.5 * dr:
            r = np.append(r, r_max)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.r = r
        logr = np.log(r)
        self._splines = []
        for ch in channels:
            K = ch.kernel(m, r)
            if np.any(K <= 0):
                raise ValueError("ядро K(r) должно быть строго положительным на сетке")
            self._splines.append(CubicSpline(logr, np.log(K)))

    def kernel(self, index: int, r) -> np.ndarray:
        """Значение ядра канала ``index`` в точках r (интерполяция)."""
        logr = np.log(np.clip(np.asarray(r, dtype=float), self.r_min, self.r_max))
        return np.exp(self._splines[index](logr))