"""Тесты табличного режима ядер (внешние таблицы сечений mkbase).

Проверяем: чтение ``data.txt`` формата ``mkbase`` / ``mkbase_spheres``
(заголовок ``# x Cext Cabs Cback Qext Qabs Qback [n]``, выбор колонки по
имени), интерполяцию, перевод единиц (мкм²→м²), пересчёт с эталонной длины
волны таблицы ``λ_ref`` на длину волны канала (``(λ/λ_ref)²``), маппинг r→x,
покрытие по x (ограничение r_max), структуру каталогов
(``mre_<real>_mim_<imag>/data.txt``), диагностику при отсутствии нужного
``m``, поведение при выходе за пределы таблицы и самосогласованность с
Mie-ядрами: таблицы, снятые из miepython при λ_ref, должны воспроизводить
ядра и результат инверсии.
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
from solve.kernels import (
    Channel,
    KernelTable,
    TabularCrossSection,
    available_m_values,
    load_table_channels,
    m_table_dirname,
    table_m_grid,
)

M = 1.45 - 0.005j


def make_channels():
    return (
        [Channel("beta", wl * 1e-3) for wl in DEMO_BETA_WL_NM]
        + [Channel("alpha", wl * 1e-3) for wl in DEMO_ALPHA_WL_NM]
    )


def write_data_file(path: str, rows, header: str | None = None,
                    extra_comments=()) -> None:
    """Записать ``data.txt`` в формате mkbase (rows — готовые строки)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header or "# x Cext Cabs Cback Qext Qabs Qback\n")
        for line in extra_comments:
            fh.write(line + "\n")
        for line in rows:
            fh.write(line + "\n")


def dump_tables(root: str, m: complex, channels, x_max: float, n: int = 400,
                linear: bool = False, ref_wl_um: float = 2.0 * np.pi,
                subdir: str = "") -> None:
    """Снять сечения из miepython в ``data.txt`` формата mkbase.

    Каталог ``<root>/<subdir>/mre_<real:.2f>_mim_<imag:.5f>/data.txt``,
    столбцы ``x Cext Cabs Cback Qext Qabs Qback``. Сечения считаются при
    эталонной длине волны ``ref_wl_um`` (по умолчанию 2π мкм): для параметра
    размера ``x`` радиус ``r = x·ref_wl_um/(2π)``, ``C_ext = π r² q_ext``,
    ``C_abs = π r² (q_ext − q_sca)``, ``C_back = r² q_back / 4``.
    При ``linear=True`` сетка x равномерная (шаг x_max/n), иначе —
    лог-эквидистантная (как в реальных таблицах).
    """
    mdir = os.path.join(root, subdir, m_table_dirname(m))
    os.makedirs(mdir, exist_ok=True)
    x = (np.linspace(0.1, x_max, n) if linear
         else np.geomspace(0.1, x_max, n))
    qext, qsca, qback, _g = mie.efficiencies_mx(m, x)
    r_um = x * ref_wl_um / (2.0 * np.pi)
    cext = np.pi * r_um ** 2 * qext
    cabs = np.pi * r_um ** 2 * (qext - qsca)
    cback = r_um ** 2 * qback / 4.0
    px2 = np.pi * x * x
    with open(os.path.join(mdir, "data.txt"), "w", encoding="utf-8") as fh:
        fh.write("# x Cext Cabs Cback Qext Qabs Qback\n")
        for xi, ce, ca, cb, p2 in zip(x, cext, cabs, cback, px2):
            fh.write(f"{xi:.6e} {ce:.6e} {ca:.6e} {cb:.6e} "
                     f"{ce / p2:.6e} {ca / p2:.6e} {cb / p2:.6e}\n")


def test_data_file_parse_and_interp() -> None:
    """Чтение mkbase-файла: комментарии, выбор колонки, интерполяция."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.txt")
        write_data_file(
            p,
            ["0.1 1.0 0.5 0.10 31.8 15.9 3.18",
             "0.5 2.0 1.0 0.20 2.55 1.27 0.25",
             "1.0 4.0 2.0 0.40 1.27 0.64 0.13",
             "2.0 16.0 8.0 1.60 1.27 0.64 0.13",
             "3.0 36.0 18.0 3.60 1.27 0.64 0.13",
             "4.0 64.0 32.0 6.40 1.27 0.64 0.13",
             "5.0 100.0 50.0 10.0 1.27 0.64 0.13"],
            extra_comments=[
                "# Units: x (dimless), Cext/Cabs (um^2), "
                "Cback (um^2/sr), Q (dimensionless)",
                "# Smoothed with Gaussian kernel, sigma_idx=1.00",
            ],
        )
        ext = TabularCrossSection.from_data_file(p, "Cext")
        back = TabularCrossSection.from_data_file(p, "Cback")
        assert ext.x_min == 0.1 and ext.x_max == 5.0
        xs = np.array([0.1, 1.0, 5.0])
        # в узлах интерполяция точная
        assert np.allclose(ext.cross_section(xs), [1.0, 4.0, 100.0])
        assert np.allclose(back.cross_section(xs), [0.10, 0.40, 10.0])
        # монотонность между узлами (данные монотонно растут)
        grid = np.geomspace(0.1, 5.0, 100)
        assert np.all(np.diff(ext.cross_section(grid)) > 0)


def test_data_file_columns_by_name() -> None:
    """Колонки выбираются по имени: порядок и лишний столбец ``n`` не важны."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.txt")
        write_data_file(
            p,
            ["0.1 7.0 3.0 1.0 0.5 0.4 0.3 5",
             "0.5 8.0 4.0 2.0 0.6 0.5 0.4 5",
             "1.0 9.0 5.0 3.0 0.7 0.6 0.5 5"],
            header="# x Cback Cabs Cext Qext Qabs Qback n\n",
        )
        ext = TabularCrossSection.from_data_file(p, "Cext")
        back = TabularCrossSection.from_data_file(p, "Cback")
        assert np.allclose(ext.C, [1.0, 2.0, 3.0])
        assert np.allclose(back.C, [7.0, 8.0, 9.0])


def test_data_file_requires_header() -> None:
    """Без заголовка mkbase файл не читается; отсутствие колонки — ошибка."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.txt")
        write_data_file(p, ["0.1 1.0 0.5", "0.5 2.0 1.0", "1.0 4.0 2.0"],
                        header="# просто комментарий\n")
        try:
            TabularCrossSection.from_data_file(p, "Cext")
            raise AssertionError("ожидалась ошибка об отсутствии заголовка")
        except ValueError as exc:
            assert "заголовок" in str(exc)

        # заголовок есть, но нужной колонки в нём нет
        write_data_file(p, ["0.1 1.0", "0.5 2.0", "1.0 4.0"],
                        header="# x Cext\n")
        try:
            TabularCrossSection.from_data_file(p, "Cback")
            raise AssertionError("ожидалась ошибка об отсутствии колонки")
        except ValueError as exc:
            assert "Cback" in str(exc)


def test_table_out_of_range_error() -> None:
    """Выход за пределы диапазона x — ошибка; с extrapolate=True — работает."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.txt")
        write_data_file(
            p,
            ["0.1 1.0 0.5 0.1 31.8 15.9 3.2",
             "0.5 2.0 1.0 0.2 2.5 1.3 0.2",
             "1.0 4.0 2.0 0.4 1.3 0.6 0.1"],
        )
        tbl = TabularCrossSection.from_data_file(p, "Cext")
        try:
            tbl.cross_section(np.array([3.0]))
            raise AssertionError("ожидалось исключение при x > x_max")
        except ValueError:
            pass
        tbl_ex = TabularCrossSection.from_data_file(p, "Cext", extrapolate=True)
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


def test_cback_convention_4pi() -> None:
    """Cback = 4π·dσ/dΩ (старые mkbase_spheres/miedir) читается с cback_scale."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=40.0, n=2000, linear=True)
        # перезапишем Cback/Qback в конвенцию «backscattering cross section»
        path = os.path.join(td, m_table_dirname(M), "data.txt")
        d = np.loadtxt(path, comments="#")
        d[:, 3] *= 4.0 * np.pi
        d[:, 6] *= 4.0 * np.pi
        np.savetxt(path, d, fmt="%.10e",
                   header="x Cext Cabs Cback Qext Qabs Qback")

        r = np.geomspace(0.05, 2.0, 300)
        for ch in channels:
            k_mie = ch.kernel(M, r)
            bad, _ = load_table_channels(td, M, [ch])
            good, _ = load_table_channels(td, M, [ch],
                                         cback_scale=1.0 / (4.0 * np.pi))
            rel_bad = np.median(np.abs(bad[0].kernel(M, r) / k_mie - 1.0))
            rel_good = np.median(np.abs(good[0].kernel(M, r) / k_mie - 1.0))
            if ch.kind == "beta":
                assert rel_bad > 5.0, f"{ch.name}: ожидался сдвиг 4π"
            assert rel_good < 2e-3, f"{ch.name}: rel err = {rel_good:.4g}"


def test_load_table_channels_structure() -> None:
    """Именование каталогов и расчёт ограничения r_max по покрытию."""
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
    """Структура: mre_<real>_mim_<imag>/data.txt (формат mkbase)."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, M, channels, x_max=20.0)
        mdir = os.path.join(td, m_table_dirname(M))
        assert m_table_dirname(M) == "mre_1.45_mim_-0.00500"
        assert os.path.isfile(os.path.join(mdir, "data.txt"))
        assert not os.path.isfile(os.path.join(mdir, "Cext.txt"))
        assert not os.path.isfile(os.path.join(mdir, "Cback.txt"))
        chs, _ = load_table_channels(td, M, channels)
        assert all(c.table is not None for c in chs)
        assert all(c.table_ref_wl_um == 2.0 * np.pi for c in chs)
        # каналы одного вида (alpha/beta) делят одну загруженную таблицу
        for kind in ("alpha", "beta"):
            same = [c for c in chs if c.kind == kind]
            assert len({id(c.table) for c in same}) == 1


def test_negative_zero_imag_dirname() -> None:
    """Im(m) = −0.0 нормализуется: каталог ``mim_0.00000``, а не ``-0.00000``."""
    m = complex(1.45, -0.0)
    assert m_table_dirname(m) == "mre_1.45_mim_0.00000"
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, complex(1.45, 0.0), channels, x_max=20.0)
        chs, _ = load_table_channels(td, m, channels)
        assert all(c.table is not None for c in chs)


def test_missing_m_diagnostics() -> None:
    """Ошибка подсказывает доступные m и куда указывать --tables."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        # 1) передан корень базы (base), а не base/average или base/shapeNN
        os.makedirs(os.path.join(td, "average"))
        os.makedirs(os.path.join(td, "shape01"))
        try:
            load_table_channels(td, M, channels)
            raise AssertionError("ожидалась ошибка при отсутствии каталога m")
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "average" in msg and "shape" in msg

        # 2) таблица другого m есть, а нужного — нет: список доступных m
        dump_tables(td, complex(1.30, 0.0), channels, x_max=20.0)
        try:
            load_table_channels(td, M, channels)
            raise AssertionError("ожидалась ошибка при отсутствии нужного m")
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "mre_1.45_mim_-0.00500" in msg
            assert "1.3+0j" in msg


def test_available_m_values_and_grid() -> None:
    """Сетка перебора — только m с готовыми таблицами; флаги её сужают."""
    channels = make_channels()
    with tempfile.TemporaryDirectory() as td:
        dump_tables(td, complex(1.45, 0.0), channels, x_max=20.0)
        dump_tables(td, complex(1.45, 0.0168), channels, x_max=20.0)
        dump_tables(td, complex(1.50, 0.0), channels, x_max=20.0)
        # каталог без data.txt не считается таблицей
        os.makedirs(os.path.join(td, "mre_1.60_mim_0.00000"))

        expected = [complex(1.45, 0.0), complex(1.45, 0.0168), complex(1.50, 0.0)]
        assert available_m_values(td) == expected
        assert table_m_grid(td) == expected
        # --m-reals / --m-imags сужают набор (округление как в именах каталогов)
        assert table_m_grid(td, [1.45], None) == [
            complex(1.45, 0.0), complex(1.45, 0.0168)]
        assert table_m_grid(td, None, [0.0168]) == [complex(1.45, 0.0168)]
        assert table_m_grid(td, [1.50], [0.0]) == [complex(1.50, 0.0)]

        # фильтр без совпадений — ошибка с перечнем доступных m
        try:
            table_m_grid(td, [1.90], None)
            raise AssertionError("ожидалась ошибка для отсутствующего Re(m)")
        except FileNotFoundError as exc:
            assert "доступные m" in str(exc)

        # все m из сетки действительно загружаются
        for mv in table_m_grid(td):
            chs, _ = load_table_channels(td, mv, channels)
            assert all(c.table is not None for c in chs)


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
