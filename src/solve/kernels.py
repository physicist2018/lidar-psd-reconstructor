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

import numpy as np
import miepython as mie
from scipy.interpolate import CubicSpline

# (1 мкм)^2 = 1e-12 м^2 — перевод сечения из мкм^2 в м^2
UM2_TO_M2 = 1e-12


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
        """Физическое ядро на частицу: alpha — [м^2], beta — [м^2/ср]."""
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