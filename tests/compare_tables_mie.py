#!/usr/bin/env python3
"""Сверка внешних таблиц сечений с miepython.

Помогает понять, почему ``solve --tables`` даёт не то же, что Mie-ядра:

* сравнивает ``Cext`` и ``Cback`` из ``data.txt`` с miepython при том же ``m``;
* для ``Cback`` проверяет обе конвенции — ``dσ/dΩ`` и ``4π·dσ/dΩ``
  (последнюю писали ``mkbase_spheres``/``miedir`` до исправления);
* проверяет согласованность столбцов ``Q`` с ``C/(π·x²)``.

Метрики устойчивые: для осциллирующего ``Cback`` в глубоких минимумах рипла
относительная ошибка неинформативна, поэтому печатаются медиана отношения и
медиана/p95 относительного отклонения.

Для таблиц агрегатов (``base/average``, ``base/shapeNN``) расхождение с Mie —
норма: это другие частицы, а не ошибка конвенции.

Запуск::

    uv run python tests/compare_tables_mie.py miedir
    uv run python tests/compare_tables_mie.py miedir --m 1.45+0.005j
    uv run python tests/compare_tables_mie.py base/average --m 1.45+0j
"""

from __future__ import annotations

import argparse
import os
import sys

import miepython as mie
import numpy as np

from solve.kernels import TABLE_REF_WL_UM, available_m_values, m_table_dirname

FOUR_PI = 4.0 * np.pi

# порог, ниже которого считаем, что конвенция подходит
_OK = 0.05


def _stats(table: np.ndarray, model: np.ndarray):
    """(медиана отношения, медиана |ratio−1|, p95 |ratio−1|) на общих точках."""
    mask = (np.isfinite(table) & np.isfinite(model) & (table > 0) & (model > 0))
    if mask.sum() < 3:
        return None
    ratio = table[mask] / model[mask]
    rel = np.abs(ratio - 1.0)
    return (float(np.median(ratio)), float(np.median(rel)),
            float(np.percentile(rel, 95)))


def _fmt(v: float) -> str:
    return f"{100.0 * v:.2f}%" if v < 0.1 else f"{100.0 * v:.0f}%"


def _report_m(root: str, m: complex, ref_wl_um: float) -> str:
    """Сверить один каталог m. Возвращает конвенцию Cback: sigma/4pi/other."""
    path = os.path.join(root, m_table_dirname(m), "data.txt")
    d = np.loadtxt(path, comments="#")
    if d.ndim != 2 or d.shape[1] < 7:
        print(f"m = {m.real:g}{m.imag:+g}j: ожидались 7 столбцов, "
              f"получено {d.shape[1] if d.ndim == 2 else 1} — пропуск")
        return "other"
    x = d[:, 0]
    qext, _qsca, qback, _g = mie.efficiencies_mx(m, x)
    # при λ_ref = 2π абсцисса x численно равна радиусу a
    a = x * ref_wl_um / (2.0 * np.pi)
    cext_mie = np.pi * a ** 2 * qext
    cback_mie = a ** 2 * qback / 4.0

    ce = _stats(d[:, 1], cext_mie)
    cb_ds = _stats(d[:, 3], cback_mie)
    cb_4pi = _stats(d[:, 3], FOUR_PI * cback_mie)

    print(f"m = {m.real:g}{m.imag:+g}j")
    if ce is None:
        print("  Cext: нет общих точек для сравнения")
    else:
        print(f"  Cext : median ratio {ce[0]:.4f}, "
              f"med|rel| {_fmt(ce[1])}, p95 {_fmt(ce[2])}")
    if cb_ds is None or cb_4pi is None:
        print("  Cback: нет общих точек для сравнения")
        return "other"

    ok_ds = cb_ds[1] < _OK
    ok_4pi = cb_4pi[1] < _OK
    print(f"  Cback: как dσ/dΩ    med|rel| {_fmt(cb_ds[1])}, p95 {_fmt(cb_ds[2])}")
    print(f"         как 4π·dσ/dΩ med|rel| {_fmt(cb_4pi[1])}, p95 {_fmt(cb_4pi[2])}")
    if ok_ds and not ok_4pi:
        print("         → конвенция dσ/dΩ (ok)")
        verdict = "sigma"
    elif ok_4pi and not ok_ds:
        print("         → Cback записан как 4π·dσ/dΩ: "
              "используйте --table-cback 4pi")
        verdict = "4pi"
    elif not ok_ds and not ok_4pi:
        print("         → ни dσ/dΩ, ни 4π·dσ/dΩ: таблица не описывает сферы "
              "(для агрегатов это ожидаемо) — сверка с Mie неприменима")
        verdict = "other"
    else:
        verdict = "sigma"

    px2 = np.pi * x ** 2
    q_pairs = ((d[:, 4], d[:, 1] / px2), (d[:, 6], d[:, 3] / px2))
    q_dev = []
    for col, ref in q_pairs:
        mask = (np.isfinite(col) & np.isfinite(ref) & (ref != 0) & (col != 0))
        q_dev.append(float(np.median(np.abs(col[mask] / ref[mask] - 1.0)))
                     if mask.any() else float("nan"))
    print(f"  Q    : |Qext−Cext/(πx²)| {_fmt(q_dev[0])}, "
          f"|Qback−Cback/(πx²)| {_fmt(q_dev[1])}")
    return verdict


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="compare_tables_mie",
        description="Сверка таблиц сечений (формат mkbase) с miepython: "
                    "Cext/Cback, конвенция Cback, согласованность Q.")
    p.add_argument("root", help="каталог с подкаталогами mre_<real>_mim_<imag>/")
    p.add_argument("--m", action="append", default=None,
                   help="ограничить m (можно несколько), например 1.45+0.005j")
    p.add_argument("--ref-lambda-um", type=float, default=TABLE_REF_WL_UM,
                   help=f"длина волны таблиц, мкм [{TABLE_REF_WL_UM:g}]")
    args = p.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"ошибка: каталог не найден: {args.root}", file=sys.stderr)
        return 2
    try:
        ms = available_m_values(args.root)
    except (OSError, FileNotFoundError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    if not ms:
        print(f"ошибка: в {args.root} нет mre_*/data.txt", file=sys.stderr)
        return 2
    if args.m:
        want = [complex(s) for s in args.m]
        ms = [z for z in ms if any(abs(z - w) < 5e-6 for w in want)]
        if not ms:
            print("ошибка: ни одно m не найдено в таблицах", file=sys.stderr)
            return 2

    print(f"каталог: {args.root} ({len(ms)} m, λ_ref = {args.ref_lambda_um:g} мкм)")
    verdicts = {}
    for m in ms:
        v = _report_m(args.root, m, args.ref_lambda_um)
        verdicts[v] = verdicts.get(v, 0) + 1
    n_ds = verdicts.get("sigma", 0)
    n_4pi = verdicts.get("4pi", 0)
    n_other = verdicts.get("other", 0)
    print()
    if n_other == 0 and n_4pi == 0:
        print("итог: Cback всех проверенных таблиц = dσ/dΩ "
              "(--table-cback dsigma)")
    elif n_other == 0 and n_ds == 0:
        print("итог: Cback всех проверенных таблиц = 4π·dσ/dΩ "
              "(старые mkbase_spheres/miedir) → --table-cback 4pi")
    else:
        print(f"итог: dσ/dΩ — {n_ds}, 4π·dσ/dΩ — {n_4pi}, "
              f"не сферы/не определено — {n_other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
