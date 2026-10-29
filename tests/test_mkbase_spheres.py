"""Тесты ``mkbase_spheres`` — таблицы сечений сфер из miepython.

Проверяем единицы и конвенции: ``Cext = πa²·qext``, ``Cback = a²·qback/4``
(дифференциальное сечение dσ/dΩ(180°), как у mkbase/ADDA и solve), а также
согласованность ``Q = C/(πa²)``, в том числе после сглаживания.

Запуск:  uv run python tests/test_mkbase_spheres.py
"""

from __future__ import annotations

import math
import os
import tempfile

import miepython as mie
import numpy as np

from mkbase_spheres.mkbase_spheres import (
    compute_mie_cross_sections_units,
    generate_mie_tables,
    make_log_equidistant,
)

WL_UM = 2.0 * math.pi


def _read(path: str):
    return np.loadtxt(path, comments="#")


def test_cback_is_dsigma_and_q_consistent() -> None:
    """``Cback = a²·qback/4`` (dσ/dΩ), ``Qback = Cback/(πa²)``; не 4π."""
    x = np.geomspace(0.05, 20.0, 120)
    m_re, m_im = 1.45, -0.005
    cext, cabs, cback, qext, qabs, qback = compute_mie_cross_sections_units(
        x, m_re, m_im, WL_UM)
    qe, qs, qb, _g = mie.efficiencies_mx(complex(m_re, m_im), x)
    a = x  # λ_ref = 2π ⇒ радиус численно равен x
    cgeo = math.pi * a ** 2

    assert np.allclose(cext, cgeo * qe, rtol=1e-12)
    assert np.allclose(cabs, cgeo * (qe - qs), rtol=1e-12)
    assert np.allclose(cback, a ** 2 * qb / 4.0, rtol=1e-12)
    assert np.allclose(qext, cext / cgeo, rtol=1e-12)
    assert np.allclose(qabs, cabs / cgeo, rtol=1e-12)
    assert np.allclose(qback, cback / cgeo, rtol=1e-12)
    # ровно в 4π меньше «backscattering cross section» πa²·qback
    assert np.allclose(cback * 4.0 * math.pi, cgeo * qb, rtol=1e-12)


def test_generate_tables_unsmoothed_match_mie() -> None:
    """Записанный файл: Cext и Cback совпадают с miepython (dσ/dΩ)."""
    with tempfile.TemporaryDirectory() as td:
        generate_mie_tables([1.45], [0.0], 0.05, 20.0, 200, WL_UM, base_dir=td)
        d = _read(os.path.join(td, "mre_1.45_mim_0.00000", "data.txt"))
        # сечения записаны в узлах точной сетки (x в файле округлён до 6 знаков,
        # а рипл Mie чувствителен к x — сравниваем на исходной сетке)
        x = make_log_equidistant(0.05, 20.0, 200)
        qe, _qs, qb, _g = mie.efficiencies_mx(complex(1.45, 0.0), x)
        assert np.allclose(d[:, 1], math.pi * x ** 2 * qe, rtol=5e-6)
        assert np.allclose(d[:, 3], x ** 2 * qb / 4.0, rtol=5e-6)
        # столбцы Q остались согласованы с C/(πx²)
        assert np.allclose(d[:, 4], d[:, 1] / (math.pi * x ** 2), rtol=5e-6)
        assert np.allclose(d[:, 6], d[:, 3] / (math.pi * x ** 2), rtol=5e-6)


def test_generate_tables_smoothed_stay_consistent() -> None:
    """После сглаживания Q пересчитан из C: Q = C/(πx²)."""
    with tempfile.TemporaryDirectory() as td:
        generate_mie_tables([1.45], [0.0], 0.05, 20.0, 200, WL_UM,
                            base_dir=td, smooth=True, sigma_idx=2.0)
        d = _read(os.path.join(td, "mre_1.45_mim_0.00000", "data.txt"))
        x = d[:, 0]
        cgeo = math.pi * x ** 2
        assert np.allclose(d[:, 4], d[:, 1] / cgeo, rtol=5e-6)
        assert np.allclose(d[:, 5], d[:, 2] / cgeo, rtol=5e-6)
        assert np.allclose(d[:, 6], d[:, 3] / cgeo, rtol=5e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"все {len(fns)} проверок пройдены")
