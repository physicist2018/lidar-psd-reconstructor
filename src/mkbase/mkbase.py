#!/usr/bin/env python3
"""mkbase — сборка базы сечений для программы поиска решений.

Вход — каталог с результатами расчётов ADDA следующей структуры::

    results_aggl/
      shapeNN/
        mr<real>/
          mi<imag>/
            shapeNN_mr<real>mi<imag>_x<X>[eps3]/
              CrossSec  mueller  log  log_orient_avg

Для каждой тройки ``(shape, mre, mim)`` собирается файл ``data.txt`` со
столбцами ``x Cext Cabs Cback Qext Qabs Qback``, строки отсортированы по
возрастанию ``x``. Выход повторяет структуру::

    <out>/shapeNN/mre_<real>_mim_<imag>/data.txt

Дополнительно строится агрегат по всем формам в ``<out>/average/``::

    <out>/average/mre_<real>_mim_<imag>/data.txt

Для каждого ``(mre, mim)`` каждая форма интерполируется на общую каноническую
сетку ``x`` и усредняется (среднее арифметическое с равными весами). Сетка —
это узлы ``x``, присутствующие во всех формах хотя бы одной группы; так
сглаживаются редкие расхождения исходных сеток (например, узел 8 в части
прогонов задан как ``mre + 0.0076925`` вместо геометрического ``1.5576925``).
Интерполяция — кубический сплайн в лог-лог (как в ``solve``; для каналов
с ``C <= 0``, например ``Cabs`` при ``mim = 0``, — линейная). В агрегат
добавляется столбец ``n`` — число усреднённых форм.

Вместо канонической сетки можно задать **лог-эквидистантную** сетку из ``nx``
узлов (``--nx``): для каждой группы ``(mre, mim)`` берётся диапазон
``[min x, max x]`` по всем её точкам, и на нём строится ``np.geomspace``.
На эту сетку переинтерполируются и пошape-таблицы (``shapeNN/``), так что все
таблицы группы лежат в одних и тех же узлах ``x``. Узлы вне диапазона
конкретной формы пропускаются.

Источники величин:

* ``Shape``, ``mre``, ``mim`` — из имён каталогов;
* ``x`` — из имени листового каталога (суффикс ``_x<число>``; маркер ``eps3``
  режима ``adda -eps 3`` отбрасывается); это «size along x-axis»;
* ``Cext``, ``Cabs`` — из файла ``CrossSec``;
* ``Cback`` — последнее значение столбца ``s11`` файла ``mueller`` (θ = 180°);
* ``Qext``, ``Qabs``, ``Qback`` = ``C / (π·x²)``.

Пример запуска::

    python src/mkbase/mkbase.py results_aggl -o base
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import CubicSpline

HEADER = "# x Cext Cabs Cback Qext Qabs Qback"
AVG_HEADER = "# x Cext Cabs Cback Qext Qabs Qback n"

_SHAPE_RE = re.compile(r"^shape(\d+)$")
_MR_RE = re.compile(r"^mr(-?\d+(?:\.\d+)?)$")
_MI_RE = re.compile(r"^mi(-?\d+(?:\.\d+)?)$")
_X_RE = re.compile(r"(\d+(?:\.\d+)?)")

# сколько байт читать с конца mueller (хватает на последнюю строку)
_TAIL_BYTES = 8192

# допуск на совпадение значений x (для сравнения с канонической сеткой)
_X_TOL = 1e-9


def warn(msg: str) -> None:
    """Предупреждение в stderr."""
    print(f"предупреждение: {msg}", file=sys.stderr)


def _subdirs(path: str) -> "list[str]":
    """Отсортированные имена подкаталогов."""
    with os.scandir(path) as it:
        return sorted(e.name for e in it if e.is_dir())


def parse_x(name: str) -> "float | None":
    """Параметр x из имени каталога: ``..._x21.2730108eps3`` → 21.2730108."""
    if "_x" not in name:
        return None
    tail = name.rsplit("_x", 1)[1]
    m = _X_RE.match(tail)
    return float(m.group(1)) if m else None


def parse_crosssec(path: str) -> "tuple[float, float]":
    """Cext и Cabs из файла CrossSec (строки вида ``Cext = <число>``)."""
    values: "dict[str, float]" = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or "=" not in s:
                continue
            key, _, val = s.partition("=")
            values[key.strip()] = float(val.strip())
    return values["Cext"], values["Cabs"]


def read_last_s11(path: str) -> float:
    """Последнее значение столбца s11 в файле mueller (θ = 180°)."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - _TAIL_BYTES))
        chunk = fh.read()
    for raw in reversed(chunk.split(b"\n")):
        s = raw.strip()
        if s:
            toks = s.split()
            if len(toks) < 2:
                raise ValueError("строка без столбца s11: "
                                 f"{s.decode(errors='replace')!r}")
            return float(toks[1])
    raise ValueError("файл пуст")


def load_point(xdir: str, x: float) -> "tuple[float, float, float, float] | None":
    """(x, Cext, Cabs, Cback) для листового каталога, либо None при пропуске."""
    cs = os.path.join(xdir, "CrossSec")
    mu = os.path.join(xdir, "mueller")
    if not os.path.isfile(cs) or not os.path.isfile(mu):
        warn(f"пропуск {xdir}: нет CrossSec и/или mueller")
        return None
    try:
        cext, cabs = parse_crosssec(cs)
        cback = read_last_s11(mu)
    except (OSError, ValueError, KeyError) as exc:
        warn(f"пропуск {xdir}: {exc}")
        return None
    return x, cext, cabs, cback


def _fmt(v: float) -> str:
    """Полная точность float (round-trip repr)."""
    return repr(float(v))


def _group_ranges(data) -> "dict[tuple[float, float], tuple[float, float]]":
    """Диапазон x группы ``(mre, mim)``: (min, max) по всем формам и точкам."""
    ranges: "dict[tuple[float, float], tuple[float, float]]" = {}
    for (shape, mre, mim), rows in data.items():
        key = (mre, mim)
        xs = [r[0] for r in rows]
        lo, hi = min(xs), max(xs)
        prev = ranges.get(key)
        ranges[key] = (lo, hi) if prev is None else (min(prev[0], lo),
                                                     max(prev[1], hi))
    return ranges


def _log_grid(xmin: float, xmax: float, nx: int) -> "list[float]":
    """Лог-эквидистантная сетка из ``nx`` узлов на ``[xmin, xmax]``."""
    if xmax <= xmin:
        return [float(xmin)]
    return [float(v) for v in np.geomspace(xmin, xmax, nx)]


def _canonical_grid(data) -> "list[float]":
    """Общие узлы x: присутствуют во всех формах хотя бы одной группы (mre, mim)."""
    n_shapes: "dict[tuple[float, float], set[str]]" = defaultdict(set)
    cov: "dict[tuple[float, float], Counter]" = defaultdict(Counter)
    for (shape, mre, mim), rows in data.items():
        key = (mre, mim)
        n_shapes[key].add(shape)
        for row in rows:
            cov[key][row[0]] += 1
    grid: "set[float]" = set()
    for key, cnt in cov.items():
        full = len(n_shapes[key])
        for x, c in cnt.items():
            if c == full:
                grid.add(x)
    return sorted(grid)


def _report_grid_anomalies(data, grid) -> int:
    """Предупреждение о значениях x вне общей сетки. Возвращает число предупреждений."""
    grid_set = set(grid)
    counts: "Counter[float]" = Counter()
    for rows in data.values():
        for row in rows:
            counts[row[0]] += 1
    extra = sorted(x for x in counts if x not in grid_set)
    if not extra:
        return 0
    listing = ", ".join(f"{x:g} (\u00d7{counts[x]})" for x in extra)
    warn(f"сетка x неоднородна между формами; неканонические значения "
         f"(интерполируются на общую сетку): {listing}")
    return 1


def _exact_index(xs, node):
    """Индекс узла xs, совпадающего с node (в пределах допуска), либо None."""
    for i, x in enumerate(xs):
        if abs(node - x) <= _X_TOL * max(1.0, node):
            return i
    return None


def _make_loglog_spline(xs_np, ys):
    """Интерполятор по кубическому сплайну в лог-лог (все значения > 0)."""
    spl = CubicSpline(np.log(xs_np), np.log(ys))
    x0, x1 = xs_np[0], xs_np[-1]

    def interp(node):
        if node < x0 or node > x1:
            return None
        return float(np.exp(spl(math.log(node))))

    return interp


def _make_linear(xs_np, ys):
    """Интерполятор по линейной интерполяции (для каналов с ``C <= 0``)."""
    x0, x1 = xs_np[0], xs_np[-1]

    def interp(node):
        if node < x0 or node > x1:
            return None
        return float(np.interp(node, xs_np, ys))

    return interp


def _shape_interps(xs, rows):
    """Интерполяторы каналов ``(node -> значение | None)`` для одной формы.

    Положительные каналы — кубический сплайн в лог-лог (как в ``solve``);
    каналы с ``C <= 0`` (например ``Cabs`` при ``mim = 0``) — линейная
    интерполяция (сплайн по почти нулевым знакопеременным данным нестабилен).
    Вне диапазона ``xs`` возвращается None (экстраполяции нет).
    """
    xs_np = np.asarray(xs, dtype=float)
    interps = []
    for c in range(1, 4):
        ys = np.asarray([row[c] for row in rows], dtype=float)
        interps.append(_make_loglog_spline(xs_np, ys) if np.all(ys > 0)
                       else _make_linear(xs_np, ys))
    return interps


def _resample(rows, grid) -> "tuple[list, int]":
    """Значения каналов формы на узлах ``grid``; ``([(x, Cext, Cabs, Cback)], n_interp)``.

    Узел вне диапазона формы пропускается (экстраполяции нет).
    """
    xs = [r[0] for r in rows]
    interps = None
    out = []
    n_interp = 0
    for node in grid:
        i = _exact_index(xs, node)
        if i is not None:
            out.append((node, rows[i][1], rows[i][2], rows[i][3]))
            continue
        if interps is None:
            interps = _shape_interps(xs, rows)
        vals = [ip(node) for ip in interps]
        if any(val is None for val in vals):
            continue
        out.append((node, vals[0], vals[1], vals[2]))
        n_interp += 1
    return out, n_interp


def _write_averages(out_dir: str, data, grid, grids=None,
                    resampled=None) -> "tuple[int, int, int]":
    """Агрегат по формам. Возвращает (каталогов, интерполяций, значений).

    ``grids`` — сетка на каждую группу ``(mre, mim)`` (режим ``--nx``);
    ``resampled`` — уже переинтерполированные на неё пошape-таблицы.
    """
    groups: "dict[tuple[float, float], list]" = defaultdict(list)
    for key, rows in data.items():
        groups[(key[1], key[2])].append((key, sorted(rows)))

    n_files = 0
    n_interp = 0
    n_vals = 0
    for (mre, mim), curves in sorted(groups.items()):
        nodes = grids[(mre, mim)] if grids is not None else grid
        per_node: "dict[float, list]" = {x: [] for x in nodes}
        for key, rows in curves:
            if resampled is not None:
                for node, cext, cabs, cback in resampled[key]:
                    per_node[node].append((cext, cabs, cback))
                    n_vals += 1
                continue
            xs = [r[0] for r in rows]
            interps = None
            for node in nodes:
                i = _exact_index(xs, node)
                if i is not None:
                    v = (rows[i][1], rows[i][2], rows[i][3])
                else:
                    if interps is None:
                        interps = _shape_interps(xs, rows)
                    vals = [ip(node) for ip in interps]
                    if any(val is None for val in vals):
                        continue
                    v = (vals[0], vals[1], vals[2])
                    n_interp += 1
                per_node[node].append(v)
                n_vals += 1

        d = os.path.join(out_dir, "average", f"mre_{mre:.2f}_mim_{mim:.5f}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.txt"), "w", encoding="utf-8") as fh:
            fh.write(AVG_HEADER + "\n")
            for node in nodes:
                vals = per_node[node]
                if not vals:
                    continue
                n = len(vals)
                cext = math.fsum(v[0] for v in vals) / n
                cabs = math.fsum(v[1] for v in vals) / n
                cback = math.fsum(v[2] for v in vals) / n
                px2 = math.pi * node * node
                fh.write(" ".join(_fmt(z) for z in
                                  (node, cext, cabs, cback,
                                   cext / px2, cabs / px2, cback / px2))
                         + f" {n}\n")
        n_files += 1
    return n_files, n_interp, n_vals


def build(results_dir: str, out_dir: str, nx: "int | None" = None) -> "dict[str, int]":
    """Сборка базы. Возвращает статистику.

    ``nx`` — число узлов лог-эквидистантной сетки на группу ``(mre, mim)``;
    ``None`` — каноническая сетка (общие узлы исходных таблиц).
    """
    if nx is not None and nx < 2:
        raise ValueError(f"nx должен быть >= 2: {nx}")
    warn_count = 0
    data: "dict[tuple[str, float, float], list]" = {}
    seen: "set[tuple[str, float, float]]" = set()

    for shape in _subdirs(results_dir):
        if not _SHAPE_RE.match(shape):
            continue
        shape_path = os.path.join(results_dir, shape)
        for mr in _subdirs(shape_path):
            m = _MR_RE.match(mr)
            if not m:
                continue
            mre = float(m.group(1))
            mr_path = os.path.join(shape_path, mr)
            for mi in _subdirs(mr_path):
                mm = _MI_RE.match(mi)
                if not mm:
                    continue
                mim = float(mm.group(1))
                key = (shape, mre, mim)
                seen.add(key)
                mi_path = os.path.join(mr_path, mi)
                for xdir in _subdirs(mi_path):
                    x = parse_x(xdir)
                    if x is None:
                        warn(f"пропуск {os.path.join(mi_path, xdir)}: "
                             "не распознан x в имени каталога")
                        warn_count += 1
                        continue
                    point = load_point(os.path.join(mi_path, xdir), x)
                    if point is None:
                        warn_count += 1
                        continue
                    data.setdefault(key, []).append(point)

    for shape, mre, mim in sorted(seen - set(data)):
        warn(f"нет валидных точек для {shape}/mre_{mre:.2f}_mim_{mim:.5f}")
        warn_count += 1

    # --- сетка для таблиц ---
    grids = None
    resampled = None
    grid: "list[float]" = []
    if nx is None:
        grid = _canonical_grid(data)
        warn_count += _report_grid_anomalies(data, grid)
    else:
        grids = {}
        for key, (lo, hi) in _group_ranges(data).items():
            if hi <= lo:
                warn(f"группа mre_{key[0]:.2f}_mim_{key[1]:.5f}: один узел x "
                     f"({lo:g}), сетка nx={nx} вырождается")
                warn_count += 1
            grids[key] = _log_grid(lo, hi, nx)
        resampled = {}

    # --- пошape-файлы ---
    n_catalogs = 0
    n_points = 0
    n_resample = 0
    for (shape, mre, mim), rows in data.items():
        rows.sort(key=lambda r: r[0])
        table = rows
        if grids is not None:
            nodes = grids[(mre, mim)]
            table, n_int = _resample(rows, nodes)
            resampled[(shape, mre, mim)] = table
            n_resample += n_int
            if len(table) < len(nodes):
                warn(f"{shape}/mre_{mre:.2f}_mim_{mim:.5f}: {len(nodes) - len(table)} "
                     "узлов сетки вне диапазона x формы — пропущены")
                warn_count += 1
        d = os.path.join(out_dir, shape, f"mre_{mre:.2f}_mim_{mim:.5f}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.txt"), "w", encoding="utf-8") as fh:
            fh.write(HEADER + "\n")
            for x, cext, cabs, cback in table:
                px2 = math.pi * x * x
                fh.write(" ".join(_fmt(v) for v in
                                  (x, cext, cabs, cback,
                                   cext / px2, cabs / px2, cback / px2)) + "\n")
        n_catalogs += 1
        n_points += len(rows)

    # --- агрегат по формам ---
    n_avg, n_interp, n_avg_vals = _write_averages(
        out_dir, data, grid, grids=grids, resampled=resampled)

    return {
        "catalogs": n_catalogs,
        "points": n_points,
        "avg": n_avg,
        "grid": max(len(g) for g in grids.values()) if grids else len(grid),
        "interp": n_resample if grids is not None else n_interp,
        "avg_vals": n_avg_vals,
        "warns": warn_count,
    }


def _nx_type(s: str) -> int:
    """Положительное целое >= 2 для ``--nx``."""
    v = int(s)
    if v < 2:
        raise argparse.ArgumentTypeError(f"nx должен быть >= 2: {s}")
    return v


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mkbase",
        description="Сборка базы сечений (x, Cext, Cabs, Cback, Qext, Qabs, Qback) "
                    "из каталога результатов ADDA; плюс агрегат по формам в average/.")
    p.add_argument("results",
                   help="каталог с результатами расчётов (например results_aggl)")
    p.add_argument("-o", "--out", default="base",
                   help="каталог для базы (по умолчанию: base)")
    p.add_argument("-n", "--nx", type=_nx_type, default=None,
                   help="число узлов лог-эквидистантной сетки x на группу "
                        "(mre, mim); переинтерполируются и пошape-таблицы, "
                        "и average/ (по умолчанию — канонические узлы "
                        "исходных таблиц)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.results):
        print(f"ошибка: каталог не найден: {args.results}", file=sys.stderr)
        return 2
    st = build(args.results, args.out, nx=args.nx)
    grid_kind = "лог-эквидистантная" if args.nx is not None else "каноническая"
    print(f"готово: {st['catalogs']} пошape каталогов ({st['points']} точек); "
          f"average — {st['avg']} каталогов, сетка {grid_kind} {st['grid']} узлов, "
          f"интерполяций {st['interp']} из {st['avg_vals']}; "
          f"{st['warns']} предупреждений → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
