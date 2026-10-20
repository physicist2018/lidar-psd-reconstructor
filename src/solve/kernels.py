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
    в мкм² как функцию безразмерного параметра размера x = 2πr/λ, поэтому
    показатель преломления m в расчёте ядра не участвует.
    """

    __slots__ = ("kind", "wl_um", "table")

    def __init__(self, kind: str, wl_um: float, table: "TabularCrossSection | None" = None) -> None:
        if kind not in ("alpha", "beta"):
            raise ValueError(f"kind должен быть 'alpha' или 'beta', получено {kind!r}")
        self.kind = kind
        self.wl_um = float(wl_um)
        self.table = table

    @property
    def name(self) -> str:
        prefix = "beta" if self.kind == "beta" else "alpha"
        return f"{prefix}({self.wl_um * 1e3:g} nm)"

    def kernel(self, m: complex, r) -> np.ndarray:
        """Физическое ядро на частицу: alpha — [м^2], beta — [м^2/ср].

        Скорость определяется бэкендом miepython: с numba (по умолчанию —
        см. ``set_mie_jit``) примерно в 45 раз быстрее, чем на чистом Python.
        При ``self.table`` не None используется табличное сечение (мкм² → м²).
        """
        r = np.asarray(r, dtype=float)
        x = 2.0 * np.pi * r / self.wl_um
        if self.table is not None:
            return UM2_TO_M2 * self.table.cross_section(x)
        qext, _qsca, qback, _g = mie.efficiencies_mx(m, x)
        if self.kind == "alpha":
            return UM2_TO_M2 * np.pi * r * r * qext
        return UM2_TO_M2 * r * r * qback / 4.0


class TabularCrossSection:
    """Внешняя таблица сечения: C(x) в мкм² на безразмерный параметр размера x.

    Файл — два столбца ``x  C`` (комментарии ``#`` и пустые строки игнорируются),
    x строго возрастает, C > 0. Интерполяция в логарифмических координатах
    (ln x, ln C) кубическим сплайном — как в ``KernelTable``; на осциллирующих
    факторах Mie кубический сплайн отслеживает рипл лучше shape-preserving
    методов (PCHIP "сглаживает" осцилляции между узлами). При выходе x за
    пределы таблицы по умолчанию выбрасывается ошибка; при
    ``extrapolate=True`` — экстраполяция граничной кривой с предупреждением.

    Единицы: C в мкм²; перевод в м² (ядра) выполняется умножением на
    ``UM2_TO_M2 = 1e-12`` (см. ``Channel.kernel``).
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
    def from_file(cls, path: str, extrapolate: bool = False) -> "TabularCrossSection":
        """Загрузка из текстового файла ``x C`` (комментарии ``#`` разрешены)."""
        xs, cs = [], []
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"файл таблицы сечений не найден: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                if len(toks) < 2:
                    raise ValueError(f"{path}: строка без двух чисел: {ln.rstrip()!r}")
                xs.append(float(toks[0]))
                cs.append(float(toks[1]))
        if len(xs) < 3:
            raise ValueError(f"{path}: недостаточно точек (>= 3), найдено {len(xs)}")
        obj = cls(np.asarray(xs, float), np.asarray(cs, float),
                  extrapolate=extrapolate, path=path)
        return obj

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


def load_table_channels(root: str, m: complex, channels, extrapolate: bool = False):
    """Загрузка таблиц сечений для показателя преломления ``m``.

    Каталог: ``<root>/m{real:g}_i{imag:g}/``, файлы ``Cext_<нм>.txt``
    (ослабление) и ``Cback_<нм>.txt`` (обратное рассеяние), где ``<нм>`` —
    длина волны канала в нанометрах. Возвращает (channels_tbl, r_max_cap)
    — каналы с табличными ядрами и ограничение сверху по радиусу (мкм),
    определяемое покрытием таблиц по x (минимум по каналам:
    r_max = x_max·λ/(2π)).
    """
    mdir = os.path.join(root, f"m{m.real:g}_i{m.imag:g}")
    if not os.path.isdir(mdir):
        raise FileNotFoundError(
            f"каталог таблиц не найден: {mdir} (создайте его с файлами "
            f"Cext_<нм>.txt / Cback_<нм>.txt)")
    out = []
    caps = []
    for ch in channels:
        prefix = "Cext" if ch.kind == "alpha" else "Cback"
        wl_nm = int(round(ch.wl_um * 1e3))
        path = os.path.join(mdir, f"{prefix}_{wl_nm}.txt")
        tbl = TabularCrossSection.from_file(path, extrapolate=extrapolate)
        new_ch = Channel(ch.kind, ch.wl_um, table=tbl)
        out.append(new_ch)
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