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
    """Один измеряемый канал: ослабление ('alpha') или обратное рассеяние ('beta')."""

    __slots__ = ("kind", "wl_um")

    def __init__(self, kind: str, wl_um: float) -> None:
        if kind not in ("alpha", "beta"):
            raise ValueError(f"kind должен быть 'alpha' или 'beta', получено {kind!r}")
        self.kind = kind
        self.wl_um = float(wl_um)

    @property
    def name(self) -> str:
        prefix = "beta" if self.kind == "beta" else "alpha"
        return f"{prefix}({self.wl_um * 1e3:g} nm)"

    def kernel(self, m: complex, r) -> np.ndarray:
        """Физическое ядро на частицу: alpha — [м^2], beta — [м^2/ср].

        Скорость определяется бэкендом miepython: с numba (по умолчанию —
        см. ``set_mie_jit``) примерно в 45 раз быстрее, чем на чистом Python.
        """
        r = np.asarray(r, dtype=float)
        x = 2.0 * np.pi * r / self.wl_um
        qext, _qsca, qback, _g = mie.efficiencies_mx(m, x)
        if self.kind == "alpha":
            return UM2_TO_M2 * np.pi * r * r * qext
        return UM2_TO_M2 * r * r * qback / 4.0


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