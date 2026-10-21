"""Тесты табличного режима ядер (внешние таблицы сечений Cext/Cback).

Проверяем: парсинг файлов ``x C``, интерполяцию, перевод единиц (мкм²→м²),
пересчёт с эталонной длины волны таблицы ``λ_ref`` на длину волны канала
(``(λ/λ_ref)²``), маппинг r→x, покрытие по x (ограничение r_max), структуру
каталогов (каталог на m с файлами Cext.txt/Cback.txt), поведение при
выходе за пределы таблицы и самосогласованность с Mie-ядрами: таблицы,
снятые из miepython при λ_ref, должны воспроизводить ядра и результат
инверсии.
"""

from __future__ import annotations

import os
import tempfile

import miepython as mie
import numpy as np

from solve import (
    DEMO_ALPHA_WL_NM,
    DEMO_BETA_WL_NM,
    demo_true_distribution,
    forward_measurements,
)
from solve.inverse import scan_variants
from solve.kernels import Channel, KernelTable, TabularCrossSection, load_table_channels

M = 1.45 - 0.005j


def make_channels():
    return (
        [Channel("beta", wl * 1e-3) for wl in DEMO_BETA_WL_NM]
        + [Channel("alpha", wl * 1e-3) for wl in DEMO_ALPHA_WL_NM]
    )


def dump_tables(root: str, m: complex, channels, x_max: float, n: int = 400,
                linear: bool = False, ref_wl_um: float = 2.0 * np.pi) -> None:
    """Снять сечения C_ext/C_back (мкм²) из miepython в структуру каталогов.

    Каталог ``<root>/m{real:g}_i{imag:g}/``, файлы ``Cext.txt`` / ``Cback.txt``
    (по одной таблице на ``(m, вид)``) — тот же формат, что читает
    ``TabularCrossSection``. Сечения считаются при эталонной длине волны
    ``ref_wl_um`` (по умолчанию 2π мкм): для параметра размера ``x`` радиус
    ``r = x·ref_wl_um/(2π)``, ``C_ext = π r² q_ext``, ``C_back = r² q_back / 4``.
    При ``linear=True`` сетка x равномерная (шаг x_max/n), иначе —
    лог-эквидистантная (как в реальных таблицах).
    """
    mdir = os.path.join(root, f"m{m.real:g}_i{m.imag:g}")
    os.makedirs(mdir, exist_ok=True)
    x = (np.linspace(0.1, x_max, n) if linear
         else np.geomspace(0.1, x_max, n))
    qext, _qsca, qback, _g = mie.efficiencies_mx(m, x)
    r_um = x * ref_wl_um / (2.0 * np.pi)
    kinds = {ch.kind for ch in channels}
    for kind, prefix, C in (
        ("alpha", "Cext", np.pi * r_um ** 2 * qext),
        ("beta", "Cback", r_um ** 2 * qback / 4.0),
    ):
        if kind not in kinds:
            continue
        with open(os.path.join(mdir, f"{prefix}.txt"), "w",
                  encoding="utf-8") as fh:
            for xi, ci in zip(x, C):
                fh.write(f"{xi:.6e}  {ci:.6e}\n")


def test_tabular_parse_and_interp() -> None:
    """Парсинг `x C` с комментариями, интерполяция, границы."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "Cext_355.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(
                "# тестовая таблица\n"
                "0.1  1.0\n"
                "0.5  2.0\n"
                "1.0  4.0\n"
                "2.0  16.0\n"
                "\n"
                "5.0  100.0\n"
            )
        tbl = TabularCrossSection.from_file(p)
        assert tbl.x_min == 0.1 and tbl.x_max == 5.0
        # в узлах интерполяция точная
        assert np.allclose(tbl.cross_section(np.array([0.1, 1.0, 5.0])),
                           [1.0, 4.0, 100.0])
        # монотонность между узлами (данные монотонно растут)
        xs = np.geomspace(0.1, 5.0, 100)
        vals = tbl.cross_section(xs)
        assert np.all(np.diff(vals) > 0)


def test_table_out_of_range_error() -> None:
    """Выход за пределы диапазона x — ошибка; с extrapolate=True — работает."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "Cback_532.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("0.1  1.0\n0.5  2.0\n1.0  4.0\n")
        tbl = TabularCrossSection.from_file(p)
        try:
            tbl.cross_section(np.array([3.0]))
            raise AssertionError("ожидалось исключение при x > x_max")
        except ValueError:
            pass
        tbl_ex = TabularCrossSection.from_file(p, extrapolate=True)
        assert np.isfinite(tbl_ex.cross_section(np.array([3.0]))[0])


def test_table_kernels_match_mie() -> None:
    """Табличные ядра воспроизводят Mie-ядра для того же (m, λ).

    Точечная проверка по максимуму относительной ошибки неинформативна для
    обратного рассеяния: в глубоких минимумах рипла q_back значение C→0, и
    относительная ошибка стремится к бесконечности при малой абсолютной.
    Поэтому оцениваем медиану и 95-й процентиль относительной ошибки.
    """
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        # плотная равномерная по x сетка — воспроизводит рипл
        dump_tables(td, M, channels, x_max=40.0, n=2000, linear=True)
        r = np.geomspace(0.05, 2.0, 400)   # x <= 2π·2/0.355 ≈ 35.4 < 40
        for ch in channels:
            ch_tbl, cap = load_table_channels(td, M, [ch])
            assert cap > 2.0  # покрытие позволяет r до 2 мкм
            k_mie = ch.kernel(M, r)
            k_tab = ch_tbl[0].kernel(M, r)
            rel = np.abs(k_tab - k_mie) / np.maximum(k_mie, 1e-30)
            med = np.median(rel)
            p95 = np.percentile(rel, 95)
            assert med < 1e-3, f"{ch.name}: медиана rel err = {med:.4g}"
            assert p95 < 2e-2, f"{ch.name}: 95-процентиль rel err = {p95:.4g}"


def test_load_table_channels_structure() -> None:
    """Именование каталогов/файлов и расчёт ограничения r_max по покрытию."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=20.0)
        chs, cap = load_table_channels(td, M, channels)
        assert len(chs) == len(channels)
        assert all(c.table is not None for c in chs)
        # канал с самой короткой длиной волны ограничивает r_max
        lam_min = min(ch.wl_um for ch in channels)
        expected = 20.0 * lam_min / (2.0 * np.pi)
        assert np.isclose(cap, expected)
        # у каждого канала своя таблица с ожидаемым покрытием
        for c, c0 in zip(chs, channels):
            assert c0.kind == c.kind and c0.wl_um == c.wl_um
            assert np.isclose(c.table.x_max, 20.0)
            assert c.table_ref_wl_um == 2.0 * np.pi


def test_table_layout_per_m() -> None:
    """Структура: каталог на m, внутри Cext.txt / Cback.txt (без _<нм>)."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=20.0)
        mdir = os.path.join(td, f"m{M.real:g}_i{M.imag:g}")
        assert os.path.isfile(os.path.join(mdir, "Cext.txt"))
        assert os.path.isfile(os.path.join(mdir, "Cback.txt"))
        assert not any("_355" in fn or "_532" in fn or "_1064" in fn
                       for fn in os.listdir(mdir))
        chs, _ = load_table_channels(td, M, channels)
        assert all(c.table is not None for c in chs)
        assert all(c.table_ref_wl_um == 2.0 * np.pi for c in chs)
        # каналы одного вида (alpha/beta) делят одну загруженную таблицу
        alphas = [c for c in chs if c.kind == "alpha"]
        assert len({id(c.table) for c in alphas}) == 1


def test_table_ref_lambda_scaling() -> None:
    """Пересчёт (λ/λ_ref)²: таблица при 2π воспроизводит Mie; смена λ_ref
    меняет ядро в постоянное число раз."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=40.0, n=2000, linear=True)
        r = np.geomspace(0.2, 2.0, 50)
        for ch in channels:
            ch_ref, _ = load_table_channels(td, M, [ch], ref_wl_um=2.0 * np.pi)
            ch_one, _ = load_table_channels(td, M, [ch], ref_wl_um=1.0)
            k_ref = ch_ref[0].kernel(M, r)
            k_one = ch_one[0].kernel(M, r)
            # таблица при 2π → совпадение с Mie-ядром канала
            k_mie = ch.kernel(M, r)
            rel = np.median(np.abs(k_ref - k_mie) / np.maximum(k_mie, 1e-30))
            assert rel < 1e-3, f"{ch.name}: медиана rel err = {rel:.4g}"
            # λ_ref = 1 вместо 2π → постоянный множитель (2π)²
            assert np.allclose(k_one / k_ref, (2.0 * np.pi) ** 2, rtol=1e-6)


def test_tabular_inversion_matches_mie() -> None:
    """Сквозная инверсия в табличном режиме ≈ инверсии в Mie-режиме."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=60.0, n=800)
        r_max = 3.0  # покрытие позволяет (x_max=60, λ_min=0.355 → r_max≈3.4)

        # Mie-режим
        kern_mie = KernelTable(M, channels, 0.05, r_max, dx_max=0.1)
        g1 = forward_measurements(channels, kern_mie, demo_true_distribution,
                                  0.05, r_max)

        # табличный режим
        chs_tbl, _cap = load_table_channels(td, M, channels)
        kern_tbl = KernelTable(M, chs_tbl, 0.05, r_max, dx_max=0.1)

        r0g = np.geomspace(0.05, 0.3, 2)
        r1g = np.geomspace(2.0, r_max, 2)
        sols_mie, _ = scan_variants(channels, {M: kern_mie}, g1, [2], r0g, r1g,
                                    min_ratio=4.0, gamma_select="fixed",
                                    gamma_rel_fixed=0.1, n_gamma=5, order=12,
                                    max_log_span=0.2)
        sols_tbl, _ = scan_variants(channels, {M: kern_tbl}, g1, [2], r0g, r1g,
                                    min_ratio=4.0, gamma_select="fixed",
                                    gamma_rel_fixed=0.1, n_gamma=5, order=12,
                                    max_log_span=0.2)
        assert len(sols_mie) == len(sols_tbl)
        # инверсия плохо обусловлена: в низкоранговых решениях амплитуды ~1e7
        # с почти нулевыми компонентами, поэтому малые отличия ядер усиливаются;
        # сравниваем невязки по всем и коэффициенты/согласие лучшего решения
        for sm, st in zip(sols_mie, sols_tbl):
            assert abs(sm.err - st.err) < 1e-2
        best_m, best_t = sols_mie[0], sols_tbl[0]
        assert np.allclose(best_m.C, best_t.C, rtol=2e-3, atol=1e-9)
        assert np.allclose(best_m.g_fit, best_t.g_fit, rtol=2e-3)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"все {len(fns)} проверок пройдены")
