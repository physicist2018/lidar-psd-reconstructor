"""Тесты ``mkbase`` — сборка базы сечений из результатов расчётов ADDA.

Проверяем: парсинг имён каталогов (в т.ч. маркера ``eps3``), парсинг файлов
``CrossSec``/``mueller``, сборку пошape-файлов и общий агрегат ``average/``
с интерполяцией на каноническую сетку ``x`` и предупреждением о
неканонических ``x``, режим ``nx`` (лог-эквидистантная сетка), а также
автоопределение «большой» структуры (``NN/mReal/mImag/xSize/run*``).

Запуск:  uv run python tests/test_mkbase.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile

from mkbase.mkbase import _shape_interps, build, parse_crosssec, parse_x, read_last_s11


def _cext(x: float) -> float:
    return x * x


def _cabs(x: float) -> float:
    return x


def _cback(x: float) -> float:
    return 2.0 * x


def _write_point(d: str, x: float, cext: "float | None" = None,
                 cabs: "float | None" = None,
                 cback: "float | None" = None) -> None:
    """Файлы ``CrossSec`` и ``mueller`` в каталоге прогона ``d``."""
    cext = _cext(x) if cext is None else cext
    cabs = _cabs(x) if cabs is None else cabs
    cback = _cback(x) if cback is None else cback
    with open(os.path.join(d, "CrossSec"), "w", encoding="utf-8") as fh:
        fh.write(f"Cext\t= {cext!r}\nQext\t= 0\n"
                 f"Cabs\t= {cabs!r}\nQabs\t= 0\n")
    with open(os.path.join(d, "mueller"), "w", encoding="utf-8") as fh:
        fh.write("theta s11 s12\n0.00 9.0 0.0\n"
                 f"180.00 {cback!r} 0.0\n")


def _write_run(root: str, shape: str, mr: str, mi: str, x: float,
               eps: bool = False) -> str:
    """Создать листовой каталог классической структуры."""
    name = f"{shape}_{mr}{mi}_x{x:g}" + ("eps3" if eps else "")
    d = os.path.join(root, shape, mr, mi, name)
    os.makedirs(d, exist_ok=True)
    _write_point(d, x)
    return d


def _write_large_run(root: str, shape: str, mre: str, mim: str, x: float,
                     run: str = "run000_read_g64", complete: bool = True,
                     mtime: "float | None" = None,
                     cext: "float | None" = None) -> str:
    """Создать прогон «большой» структуры ``<shape>/<mre>/<mim>/<x>/<run>``."""
    d = os.path.join(root, shape, mre, mim, f"{x:09.4f}", run)
    os.makedirs(d, exist_ok=True)
    if complete:
        _write_point(d, x, cext=cext)
    else:
        with open(os.path.join(d, "log"), "w", encoding="utf-8") as fh:
            fh.write("incomplete\n")
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def _make_tree(root: str) -> None:
    """2 формы, 1 ``mre``, 2 ``mim``; в ``mi0.00215`` у shape02 узел 3 → 3.5."""
    nodes = [1.0, 2.0, 3.0, 4.0, 5.0]
    for shape in ("shape01", "shape02"):
        for x in nodes:
            _write_run(root, shape, "mr1.00", "mi0", x)
    for x in nodes:
        _write_run(root, "shape01", "mr1.00", "mi0.00215", x)
    for x in [1.0, 2.0, 3.5, 4.0, 5.0]:
        _write_run(root, "shape02", "mr1.00", "mi0.00215", x, eps=(x == 3.5))


def _read(path: str):
    header = None
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("#"):
                header = ln.strip()
            elif ln.strip():
                rows.append(ln.split())
    return header, rows


def test_parse_helpers() -> None:
    """``parse_x`` (в т.ч. ``eps3``), ``parse_crosssec``, ``read_last_s11``."""
    assert parse_x("shape01_mr1.30mi0_x0.2951128") == 0.2951128
    assert parse_x("shape01_mr1.30mi0.1344_x21.2730108eps3") == 21.2730108
    assert parse_x("0.2951128") is None          # «голое» имя без ``_x``
    assert parse_x("log") is None
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "CrossSec")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("Cext\t= 2.667137275e-07\nQext\t= 8.8e-06\n"
                     "Cabs\t= 3.732907645e-28\nQabs\t= 1.2e-26\n")
        assert parse_crosssec(p) == (2.667137275e-07, 3.732907645e-28)
        q = os.path.join(td, "mueller")
        with open(q, "w", encoding="utf-8") as fh:
            fh.write("theta s11 s12\n0.00 3.2e-08 0\n180.00 3.1659075995e-08 0\n")
        assert read_last_s11(q) == 3.1659075995e-08


def test_build_structure_and_average() -> None:
    """Пошape-файлы и ``average/``: структура, ``n``, интерполяция, ``Q``."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "res"), os.path.join(td, "base")
        os.makedirs(src)
        _make_tree(src)
        stats = build(src, out)

        assert stats["layouts"] == "adda"
        assert stats["catalogs"] == 4     # 2 формы × 2 mim
        assert stats["avg"] == 2          # 1 mre × 2 mim
        assert stats["grid"] == 5         # канонические узлы 1..5
        assert stats["interp"] == 1       # только узел 3 у shape02/mi0.00215
        assert stats["warns"] == 1        # предупреждение о неканоническом 3.5

        hdr, rows = _read(os.path.join(out, "shape01", "mre_1.00_mim_0.00000",
                                       "data.txt"))
        assert hdr == "# x Cext Cabs Cback Qext Qabs Qback"
        assert [float(r[0]) for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert float(rows[2][1]) == 9.0 and float(rows[2][2]) == 3.0

        hdr, rows = _read(os.path.join(out, "average", "mre_1.00_mim_0.00000",
                                       "data.txt"))
        assert hdr == "# x Cext Cabs Cback Qext Qabs Qback n"
        assert [float(r[0]) for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert all(int(r[7]) == 2 for r in rows)
        x = float(rows[2][0])
        cext, cabs, cback, qext = (float(rows[2][i]) for i in (1, 2, 3, 4))
        assert (cext, cabs, cback) == (9.0, 3.0, 6.0)
        assert abs(qext - cext / (math.pi * x * x)) < 1e-15

        # узел 3 отсутствует у shape02 → интерполируется (x² → точно 9.0)
        hdr, rows = _read(os.path.join(out, "average", "mre_1.00_mim_0.00215",
                                       "data.txt"))
        assert [float(r[0]) for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(float(rows[2][1]) - 9.0) < 1e-6
        assert int(rows[2][7]) == 2


def test_log_grid_for_shapes_and_average() -> None:
    """``nx``: лог-эквидистантная сетка группы — и в ``shapeNN/``, и в ``average/``."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "res"), os.path.join(td, "base")
        os.makedirs(src)
        _make_tree(src)
        stats = build(src, out, nx=7)

        assert stats["grid"] == 7
        assert stats["interp"] == 20      # 4 таблицы × 5 внутренних узлов
        assert stats["warns"] == 0        # расхождение сеток больше не важно

        expected = [5.0 ** (i / 6.0) for i in range(7)]
        for rel in (("shape01", "mre_1.00_mim_0.00000"),
                    ("shape02", "mre_1.00_mim_0.00215"),
                    ("average", "mre_1.00_mim_0.00000"),
                    ("average", "mre_1.00_mim_0.00215")):
            hdr, rows = _read(os.path.join(out, *rel, "data.txt"))
            assert len(rows) == 7
            xs = [float(r[0]) for r in rows]
            assert all(abs(a - b) <= 1e-12 * b for a, b in zip(xs, expected))
            ratios = [b / a for a, b in zip(xs, xs[1:])]
            assert all(abs(r - ratios[0]) < 1e-12 for r in ratios)
            assert all(abs(float(r[1]) - float(r[0]) ** 2) < 1e-9 for r in rows)

        hdr, rows = _read(os.path.join(out, "average",
                                       "mre_1.00_mim_0.00215", "data.txt"))
        assert all(int(r[7]) == 2 for r in rows)

        # края (= исходные узлы) берутся из данных, внутренние — интерполяция
        hdr, rows = _read(os.path.join(out, "shape01", "mre_1.00_mim_0.00000",
                                       "data.txt"))
        assert float(rows[0][1]) == 1.0 and float(rows[-1][1]) == 25.0

        try:
            build(src, out, nx=1)
        except ValueError:
            pass
        else:
            raise AssertionError("nx=1 должен вызывать ValueError")


def test_average_warns_on_noncanonical_x() -> None:
    """Предупреждение о значениях x вне общей канонической сетки."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "res"), os.path.join(td, "base")
        os.makedirs(src)
        _make_tree(src)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            build(src, out)
        assert "неканоническ" in err.getvalue()
        assert "3.5" in err.getvalue()


def test_large_layout_autodetect_and_average() -> None:
    """«Большая» структура: автоопределение, имена выходов, ``average/``."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "large"), os.path.join(td, "base")
        os.makedirs(src)
        nodes = [1.0, 2.0, 82.7438]
        for x in nodes:
            _write_large_run(src, "01", "1.30", "0.000", x)
            _write_large_run(src, "shape02", "1.30", "0.000", x)

        stats = build(src, out)
        assert stats["layouts"] == "large"
        assert stats["catalogs"] == 2
        assert stats["avg"] == 1
        assert stats["grid"] == 3
        assert stats["warns"] == 0

        # «01» → shape01, «shape02» — как есть; mim — 5 знаков; x — из xSize
        hdr, rows = _read(os.path.join(out, "shape01", "mre_1.30_mim_0.00000",
                                       "data.txt"))
        assert hdr == "# x Cext Cabs Cback Qext Qabs Qback"
        assert [float(r[0]) for r in rows] == nodes
        assert float(rows[1][1]) == 4.0 and float(rows[1][2]) == 2.0

        hdr, rows = _read(os.path.join(out, "shape02", "mre_1.30_mim_0.00000",
                                       "data.txt"))
        assert [float(r[0]) for r in rows] == nodes

        hdr, rows = _read(os.path.join(out, "average", "mre_1.30_mim_0.00000",
                                       "data.txt"))
        assert [int(r[7]) for r in rows] == [2, 2, 2]


def test_large_layout_run_selection() -> None:
    """Неполные прогоны пропускаются; из полных — последний по mtime."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "large"), os.path.join(td, "base")
        os.makedirs(src)
        _write_large_run(src, "01", "1.30", "0.010", 1.0,
                         run="run000_incomplete", complete=False, mtime=1000)
        _write_large_run(src, "01", "1.30", "0.010", 1.0,
                         run="run001_old", mtime=2000, cext=4.0)
        _write_large_run(src, "01", "1.30", "0.010", 1.0,
                         run="run002_new", mtime=3000, cext=9.0)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            stats = build(src, out)
        assert stats["layouts"] == "large"
        assert stats["warns"] == 1
        assert "полных прогонов 2" in err.getvalue()

        hdr, rows = _read(os.path.join(out, "shape01", "mre_1.30_mim_0.01000",
                                       "data.txt"))
        assert len(rows) == 1 and float(rows[0][1]) == 9.0


def test_large_layout_nx_grid() -> None:
    """«Большая» структура совместима с ``nx`` (общая сетка и ``average/``)."""
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, "large"), os.path.join(td, "base")
        os.makedirs(src)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            _write_large_run(src, "01", "1.30", "0.000", x)
            _write_large_run(src, "02", "1.30", "0.000", x)

        stats = build(src, out, nx=7)
        assert stats["grid"] == 7
        for rel in (("shape01", "mre_1.30_mim_0.00000"),
                    ("shape02", "mre_1.30_mim_0.00000"),
                    ("average", "mre_1.30_mim_0.00000")):
            hdr, rows = _read(os.path.join(out, *rel, "data.txt"))
            assert len(rows) == 7
            assert all(abs(float(r[1]) - float(r[0]) ** 2) < 1e-9
                       for r in rows)


def test_shape_interps_nonpositive_uses_linear() -> None:
    """Положительные каналы — сплайн в лог-лог; ``C <= 0`` — линейно."""
    xs = [1.0, 2.0, 3.5, 4.0, 5.0]
    rows = [(x, _cext(x), x - 2.5, _cback(x)) for x in xs]
    ip_cext, ip_cabs, ip_cback = _shape_interps(xs, rows)
    assert abs(ip_cext(3.0) - 9.0) < 1e-9        # x² линейна в лог-лог
    assert abs(ip_cabs(3.0) - 0.5) < 1e-12       # (x-2.5), в т.ч. ≤ 0
    assert abs(ip_cback(3.0) - 6.0) < 1e-9
    assert ip_cext(0.5) is None                  # вне диапазона — без экстраполяции


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"все {len(fns)} проверок пройдены")
