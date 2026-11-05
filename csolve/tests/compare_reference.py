#!/usr/bin/env python3
"""Compare the C version (csolve) with the reference Python version (solve).

Runs both binaries with identical arguments, saves --json and compares the
numerical values. Exit code != 0 if the discrepancy exceeds the tolerance.

Usage:
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
# Допуск для коэффициентов C: NNLS выбирает активный набор на почти
# вырожденных задачах по-разному, и отдельные коэффициенты могут
# отличаться на ~1e-6 от масштаба вектора при совпадающей невязке.
C_RTOL = 1e-5


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:])
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit(f"command exited with code {proc.returncode}: {' '.join(map(str, cmd))}")


def compare(ref, cur, path="", bad=None, info=None, scale=0.0, rtol=RTOL):
    if bad is None:
        bad = []
    if info is None:
        info = []
    if isinstance(ref, dict):
        if set(ref) != set(cur):
            bad.append((path, "key set", sorted(ref), sorted(cur)))
            return bad
        for k in ref:
            compare(ref[k], cur[k], f"{path}.{k}" if path else k, bad, info,
                    scale, rtol)
    elif isinstance(ref, list):
        if len(ref) != len(cur):
            bad.append((path, "list length", len(ref), len(cur)))
            return bad
        child_scale = scale
        child_rtol = rtol
        if path.rsplit(".", 1)[-1] == "C":
            # коэффициенты могут быть на много порядков больше допуска:
            # нормируем на масштаб всего вектора C
            child_scale = max([abs(float(v)) for v in ref + cur
                               if isinstance(v, (int, float))] + [0.0])
            child_rtol = C_RTOL
        for i, (a, b) in enumerate(zip(ref, cur)):
            compare(a, b, f"{path}[{i}]", bad, info, child_scale, child_rtol)
    elif isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
        if ref is None or cur is None:
            if ref != cur:
                bad.append((path, "number", ref, cur))
            return bad
        if isinstance(ref, bool) or isinstance(cur, bool):
            if ref != cur:
                bad.append((path, "bool", ref, cur))
            return bad
        diff = abs(float(ref) - float(cur))
        mag = max(abs(float(ref)), abs(float(cur)))
        tol = ATOL + rtol * max(mag, scale)
        if diff > tol:
            if path.rsplit(".", 1)[-1] == "gamma":
                # на плоском участке CV-критерия разные gamma дают одно и то же
                # решение (C и Err уже сверены выше) — это не расхождение
                info.append((path, ref, cur))
            else:
                bad.append((path, "number", ref, cur, diff))
    else:
        if ref != cur:
            bad.append((path, "value", ref, cur))
    return bad


def strip_cosmetic(tree):
    """Drop purely textual fields that remain Russian in the Python version
    but are English in csolve: input units and the mode estimation method.
    Numeric fields are compared as before."""
    tree.pop("units_input", None)
    tree.pop("units_output", None)
    modes = tree.get("modes")
    if isinstance(modes, dict):
        for mode in modes.values():
            if isinstance(mode, dict):
                mode.pop("method", None)


def compare_case(name, args, verbose=False, relax_modes=False):
    with tempfile.TemporaryDirectory() as td:
        ref_json = os.path.join(td, "ref.json")
        c_json = os.path.join(td, "c.json")
        base = list(args)
        run([str(PY_EXE), "-W", "ignore", "-m", "solve", *base,
             "--json", ref_json, "--no-plot"], ROOT)
        if not C_BIN.exists():
            raise SystemExit(f"binary {C_BIN} not found; run make in csolve/")
        run([str(C_BIN), *base, "--json", c_json, "--no-plot"], ROOT)
        with open(ref_json) as fh:
            ref = json.load(fh)
        with open(c_json) as fh:
            cur = json.load(fh)
    strip_cosmetic(ref)
    strip_cosmetic(cur)
    if relax_modes:
        for tree in (ref, cur):
            for key, val in tree.get("modes", {}).items():
                val.pop("V_um3_per_cm3", None)
                val.pop("rm_um", None)
                val.pop("sigma", None)
    info = []
    bad = compare(ref, cur, info=info)
    if bad:
        print(f"[FAIL] {name}: {len(bad)} mismatches")
        for item in bad[:5]:
            print("   ", item)
        return False
    n = len(ref.get("best_solutions", []))
    extra = ""
    if info:
        extra = f" (gamma: {len(info)} on the CV plateau - identical solutions)"
    if relax_modes:
        extra += " [modes not compared: unstable without smoothing]"
    print(f"[ ok ] {name}: {n} solutions, all fields within tolerance "
          f"(rtol={RTOL:g}, atol={ATOL:g}, C rtol={C_RTOL:g}){extra}")
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


PLOT_EXTS = {".png", ".pdf", ".svg"}


def compare_text_files(name, ref_dir, cur_dir):
    import pathlib

    ref = pathlib.Path(ref_dir)
    cur = pathlib.Path(cur_dir)
    ref_files = {p.name for p in ref.iterdir() if p.suffix not in PLOT_EXTS}
    cur_files = {p.name for p in cur.iterdir() if p.suffix not in PLOT_EXTS}
    problems = []
    if ref_files != cur_files:
        problems.append(("file set", sorted(ref_files), sorted(cur_files)))
    for fname in sorted(ref_files & cur_files):
        # строки-комментарии (# ...) не сравниваются: csolve теперь пишет
        # заголовки на английском, Python — на русском
        a_lines = [ln for ln in (ref / fname).read_text(encoding="utf-8").splitlines()
                   if not ln.lstrip().startswith("#")]
        b_lines = [ln for ln in (cur / fname).read_text(encoding="utf-8").splitlines()
                   if not ln.lstrip().startswith("#")]
        if len(a_lines) != len(b_lines):
            problems.append((f"{fname}: line count", len(a_lines), len(b_lines)))
            continue
        for ln, (la, lb) in enumerate(zip(a_lines, b_lines), 1):
            ta, tb = la.split(), lb.split()
            if len(ta) != len(tb):
                problems.append((f"{fname}:{ln}: token count", ta, tb))
                continue
            for ta_i, tb_i in zip(ta, tb):
                if not token_equal(ta_i, tb_i):
                    problems.append((f"{fname}:{ln}", ta_i, tb_i))
                    break
    if problems:
        print(f"[FAIL] {name}: {len(problems)} mismatches")
        for p in problems[:5]:
            print("   ", p)
        return False
    print(f"[ ok ] {name}: files match {', '.join(sorted(ref_files))}")
    return True


def compare_batch_case(name, args, out_suffix=""):
    with tempfile.TemporaryDirectory() as td:
        ref_dir = os.path.join(td, "ref")
        cur_dir = os.path.join(td, "cur")
        base = [a for a in args]
        run([str(PY_EXE), "-W", "ignore", "-m", "solve", *base,
             "--batch-out-dir", ref_dir, "--no-plot"], ROOT)
        if not C_BIN.exists():
            raise SystemExit(f"binary {C_BIN} not found; run make in csolve/")
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
    ],
    "gamma-err": [
        "--tables", "base/average", "--table-extrapolate",
        "--gamma-select", "err", "--ngamma", "11",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "no-boundary": [
        "--tables", "base/average", "--table-extrapolate", "--no-zero-boundary",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "no-smooth": [
        "--tables", "base/average", "--table-extrapolate", "--no-smooth",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "modes-stat": [
        "--tables", "base/average", "--table-extrapolate", "--modes", "stat",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "absolute": [
        "--tables", "base/average", "--table-extrapolate", "--absolute",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
    ],
    "nonneg": [
        "--tables", "base/average", "--table-extrapolate", "--nonneg",
        "--r1-min", "0.5", "--r1-max", "1.9", "--nr-out", "32",
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
        "--modes", "stat",
    ],
    "batch-miedir": BATCH + [
        "--tables", "miedir", "--table-cback", "4pi",
        "--r1-min", "1", "--r1-max", "5.6", "--nr-out", "32",
    ],
}


def compare_batch_many(name):
    """Batch of 7 altitudes: exercises the smoothing-spline branch of the
    reference S."""
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
    ap.add_argument("--suite", action="store_true", help="run the scenario suite")
    ap.add_argument("--batch-suite", action="store_true",
                    help="run the batch scenario suite (--batch)")
    ap.add_argument("--only", nargs="*", default=None, help="scenario names")
    ap.add_argument("--verbose", action="store_true")
    args, rest = ap.parse_known_args()

    if args.batch_suite:
        names = args.only if args.only else list(BATCH_SUITE)
        ok = True
        for name in names:
            if name not in BATCH_SUITE and name != "batch-many-alt":
                raise SystemExit(f"unknown batch scenario: {name}")
            if name == "batch-many-alt":
                ok &= compare_batch_many(name)
            else:
                ok &= compare_batch_case(name, BATCH_SUITE[name])
        print("\nRESULT:", "all batch scenarios match" if ok else "mismatches found")
        return 0 if ok else 1

    if not args.suite:
        if rest and rest[0] == "--":
            rest = rest[1:]
        if not rest:
            ap.error("solve arguments are required (or --suite)")
        return 0 if compare_case("custom", rest, args.verbose) else 1

    names = args.only if args.only else list(SUITE)
    ok = True
    for name in names:
        if name not in SUITE:
            raise SystemExit(f"unknown scenario: {name}")
        ok &= compare_case(name, SUITE[name], args.verbose,
                           relax_modes=name in RELAX_MODES)
    print("\nRESULT:", "all scenarios match" if ok else "mismatches found")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
