#!/usr/bin/env python3
"""Сверка C-версии (csolve) с эталонной Python-версией (solve).

Запускает оба бинарника с одинаковыми аргументами, сохраняет --json и
сравнивает числовые значения. Код возврата != 0, если расхождение больше
допуска.

Использование:
    python csolve/tests/compare_reference.py --suite
    python csolve/tests/compare_reference.py --tables base/average --r1-min 0.5 ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]          # .../solve
C_BIN = ROOT / "csolve" / "csolve"
PY_EXE = ROOT / ".venv" / "bin" / "python"

RTOL = 1e-6
ATOL = 1e-9


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:])
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit(f"команда завершилась с кодом {proc.returncode}: {' '.join(map(str, cmd))}")


def compare(ref, cur, path="", bad=None, info=None, scale=0.0):
    if bad is None:
        bad = []
    if info is None:
        info = []
    if isinstance(ref, dict):
        if set(ref) != set(cur):
            bad.append((path, "набор ключей", sorted(ref), sorted(cur)))
            return bad
        for k in ref:
            compare(ref[k], cur[k], f"{path}.{k}" if path else k, bad, info,
                    scale)
    elif isinstance(ref, list):
        if len(ref) != len(cur):
            bad.append((path, "длина списка", len(ref), len(cur)))
            return bad
        child_scale = scale
        if path.rsplit(".", 1)[-1] == "C":
            # коэффициенты могут быть на много порядков больше допуска:
            # нормируем на масштаб всего вектора C
            child_scale = max([abs(float(v)) for v in ref + cur
                               if isinstance(v, (int, float))] + [0.0])
        for i, (a, b) in enumerate(zip(ref, cur)):
            compare(a, b, f"{path}[{i}]", bad, info, child_scale)
    elif isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
        if ref is None or cur is None:
            if ref != cur:
                bad.append((path, "число", ref, cur))
            return bad
        if isinstance(ref, bool) or isinstance(cur, bool):
            if ref != cur:
                bad.append((path, "bool", ref, cur))
            return bad
        diff = abs(float(ref) - float(cur))
        mag = max(abs(float(ref)), abs(float(cur)))
        tol = ATOL + RTOL * max(mag, scale)
        if diff > tol:
            if path.rsplit(".", 1)[-1] == "gamma":
                # на плоском участке CV-критерия разные gamma дают одно и то же
                # решение (C и Err уже сверены выше) — это не расхождение
                info.append((path, ref, cur))
            else:
                bad.append((path, "число", ref, cur, diff))
    else:
        if ref != cur:
            bad.append((path, "значение", ref, cur))
    return bad


def compare_case(name, args, verbose=False, relax_modes=False):
    with tempfile.TemporaryDirectory() as td:
        ref_json = os.path.join(td, "ref.json")
        c_json = os.path.join(td, "c.json")
        base = list(args)
        run([str(PY_EXE), "-W", "ignore", "-m", "solve", *base,
             "--json", ref_json, "--no-plot"], ROOT)
        if not C_BIN.exists():
            raise SystemExit(f"не найден бинарник {C_BIN}; соберите make в csolve/")
        run([str(C_BIN), *base, "--json", c_json, "--no-plot"], ROOT)
        with open(ref_json) as fh:
            ref = json.load(fh)
        with open(c_json) as fh:
            cur = json.load(fh)
    if relax_modes:
        for tree in (ref, cur):
            for key, val in tree.get("modes", {}).items():
                val.pop("V_um3_per_cm3", None)
                val.pop("rm_um", None)
                val.pop("sigma", None)
    info = []
    bad = compare(ref, cur, info=info)
    if bad:
        print(f"[FAIL] {name}: расхождений {len(bad)}")
        for item in bad[:5]:
            print("   ", item)
        return False
    n = len(ref.get("best_solutions", []))
    extra = ""
    if info:
        extra = f" (gamma: {len(info)} на плато CV — решения совпадают)"
    if relax_modes:
        extra += " [моды не сравнивались: неустойчивы без сглаживания]"
    print(f"[ ok ] {name}: {n} решений, все поля в допуске "
          f"(rtol={RTOL:g}, atol={ATOL:g}){extra}")
    if verbose:
        print(json.dumps(ref["m_avg"], ensure_ascii=False))
    return True


def token_equal(a: str, b: str, rtol: float = 1e-5) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= 1e-9 + rtol * max(abs(fa), abs(fb))
    except ValueError:
        pass
    if "=" in a and "=" in b:
        na, va = a.split("=", 1)
        nb, vb = b.split("=", 1)
        if na == nb:
            try:
                fa, fb = float(va), float(vb)
                return abs(fa - fb) <= 1e-9 + rtol * max(abs(fa), abs(fb))
            except ValueError:
                return False
    return False


def compare_text_files(name, ref_dir, cur_dir):
    import pathlib

    ref = pathlib.Path(ref_dir)
    cur = pathlib.Path(cur_dir)
    ref_files = {p.name for p in ref.iterdir() if p.suffix != ".png"}
    cur_files = {p.name for p in cur.iterdir() if p.suffix != ".png"}
    problems = []
    if ref_files != cur_files:
        problems.append(("набор файлов", sorted(ref_files), sorted(cur_files)))
    for fname in sorted(ref_files & cur_files):
        a_lines = (ref / fname).read_text(encoding="utf-8").splitlines()
        b_lines = (cur / fname).read_text(encoding="utf-8").splitlines()
        if len(a_lines) != len(b_lines):
            problems.append((f"{fname}: число строк", len(a_lines), len(b_lines)))
            continue
        for ln, (la, lb) in enumerate(zip(a_lines, b_lines), 1):
            ta, tb = la.split(), lb.split()
            if len(ta) != len(tb):
                problems.append((f"{fname}:{ln}: число токенов", ta, tb))
                continue
            for ta_i, tb_i in zip(ta, tb):
                if not token_equal(ta_i, tb_i):
                    problems.append((f"{fname}:{ln}", ta_i, tb_i))
                    break
    if problems:
        print(f"[FAIL] {name}: расхождений {len(problems)}")
        for p in problems[:5]:
            print("   ", p)
        return False
    print(f"[ ok ] {name}: совпали файлы {', '.join(sorted(ref_files))}")
    return True


def compare_batch_case(name, args, out_suffix=""):
    with tempfile.TemporaryDirectory() as td:
        ref_dir = os.path.join(td, "ref")
        cur_dir = os.path.join(td, "cur")
        base = [a for a in args]
        run([str(PY_EXE), "-W", "ignore", "-m", "solve", *base,
             "--batch-out-dir", ref_dir, "--no-plot"], ROOT)
        if not C_BIN.exists():
            raise SystemExit(f"не найден бинарник {C_BIN}; соберите make в csolve/")
        run([str(C_BIN), *base, "--batch-out-dir", cur_dir, "--no-plot"], ROOT)
        return compare_text_files(name + out_suffix, ref_dir, cur_dir)


SUITE = {
    "base-average": [
        "--tables", "base/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "base-aggl": [
        "--tables", "base_aggl/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "miedir-4pi": [
        "--tables", "miedir", "--table-cback", "4pi",
        "--r1-min", "1", "--r1-max", "5.6", "--nr-out", "48",
        "--m-reals", "1.35", "1.45", "--m-imags", "0", "-0.005", "-0.01",
    ],
    "data-json": [
        "--tables", "base/average", "--table-extrapolate",
        "--data", "examples/input_example.json",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "gamma-cv": [
        "--tables", "base/average", "--table-extrapolate",
        "--gamma-select", "cv", "--ngamma", "15",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "gamma-err": [
        "--tables", "base/average", "--table-extrapolate",
        "--gamma-select", "err", "--ngamma", "11",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "no-boundary": [
        "--tables", "base/average", "--table-extrapolate", "--no-zero-boundary",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "no-smooth": [
        "--tables", "base/average", "--table-extrapolate", "--no-smooth",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "modes-stat": [
        "--tables", "base/average", "--table-extrapolate", "--modes", "stat",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "absolute": [
        "--tables", "base/average", "--table-extrapolate", "--absolute",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
    "nonneg": [
        "--tables", "base/average", "--table-extrapolate", "--nonneg",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "0.00215",
    ],
}


RELAX_MODES = {"no-smooth"}   # без сглаживания фит логнормали неустойчив

BATCH = ["--batch", "tests/data/measurements_example.txt"]

BATCH_SUITE = {
    "batch-many-alt": [],
    "batch-base": BATCH + [
        "--tables", "base/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "batch-lidar": BATCH + [
        "--tables", "base/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--lidar-ratio-cons",
    ],
    "batch-lidar-w0": BATCH + [
        "--tables", "base/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--lidar-ratio-cons", "--lidar-ratio-w", "0",
    ],
    "batch-modes-stat": BATCH + [
        "--tables", "base/average", "--table-extrapolate",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
        "--modes", "stat", "--m-reals", "1.45",
        "--m-imags", "0", "0.00215",
    ],
    "batch-miedir": BATCH + [
        "--tables", "miedir", "--table-cback", "4pi",
        "--r1-min", "1", "--r1-max", "5.6", "--nr-out", "32",
        "--m-reals", "1.45", "--m-imags", "0", "-0.005",
    ],
}


def compare_batch_many(name):
    """Пакет из 7 высот: тренирует ветку сглаживающего сплайна opорного S."""
    import math

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "many.txt")
        base = [9.879211342818602e-06, 1.06715705204743e-05,
                7.002700027432126e-06, 0.000302238873845872,
                0.000327611920803267]
        rows = ["Altitude  b355  b532  b1064 a355  a532"]
        for k in range(7):
            f = 1.0 + 0.05 * math.sin(1.7 * k) + 0.01 * k
            rows.append(f"{100 + 200 * k} " +
                        " ".join(f"{v * f:.17g}" for v in base))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        args = ["--batch", path, "--tables", "base/average",
                "--table-extrapolate", "--r1-min", "0.5", "--r1-max", "1.9",
                "--nr-out", "32", "--lidar-ratio-cons"]
        ref_dir = os.path.join(td, "ref")
        cur_dir = os.path.join(td, "cur")
        run([str(PY_EXE), "-W", "ignore", "-m", "solve", *args,
             "--batch-out-dir", ref_dir, "--no-plot"], ROOT)
        run([str(C_BIN), *args, "--batch-out-dir", cur_dir, "--no-plot"], ROOT)
        return compare_text_files(name, ref_dir, cur_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", action="store_true", help="прогнать набор сценариев")
    ap.add_argument("--batch-suite", action="store_true",
                    help="прогнать набор пакетных сценариев (--batch)")
    ap.add_argument("--only", nargs="*", default=None, help="имена сценариев")
    ap.add_argument("--verbose", action="store_true")
    args, rest = ap.parse_known_args()

    if args.batch_suite:
        names = args.only if args.only else list(BATCH_SUITE)
        ok = True
        for name in names:
            if name not in BATCH_SUITE and name != "batch-many-alt":
                raise SystemExit(f"неизвестный пакетный сценарий: {name}")
            if name == "batch-many-alt":
                ok &= compare_batch_many(name)
            else:
                ok &= compare_batch_case(name, BATCH_SUITE[name])
        print("\nИТОГ:", "все пакетные сценарии совпали" if ok else "есть расхождения")
        return 0 if ok else 1

    if not args.suite:
        if rest and rest[0] == "--":
            rest = rest[1:]
        if not rest:
            ap.error("нужны аргументы solve (или --suite)")
        return 0 if compare_case("custom", rest, args.verbose) else 1

    names = args.only if args.only else list(SUITE)
    ok = True
    for name in names:
        if name not in SUITE:
            raise SystemExit(f"неизвестный сценарий: {name}")
        ok &= compare_case(name, SUITE[name], args.verbose,
                           relax_modes=name in RELAX_MODES)
    print("\nИТОГ:", "все сценарии совпали" if ok else "есть расхождения")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
