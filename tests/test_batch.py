"""Тесты пакетной обработки (--batch).

Проверяем: парсинг txt, построение таблицы ядер ровно один раз,
и полный набор выходных файлов для нескольких высот.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from solve.batch import parse_batch_txt, run_batch
from solve.kernels import KernelTable


DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "measurements_example.txt")


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        batch=DATA_FILE,
        batch_out_dir=None,
        degree=[1, 2, 3, 4],
        n_basis=None,
        r0_min=0.02, r0_max=0.5, nr0=6,
        r1_min=2.0, r1_max=10.0, nr1=4,
        min_ratio=4.0,
        best=5,
        gamma_min=-7.0, gamma_max=5.0, n_gamma=41,
        gamma_select="fixed", gamma_rel=0.1,
        nonneg=False, no_zero_boundary=False, no_smooth=False,
        smooth_lam=1e-4, r_split=0.5, absolute=False,
        dx_max=0.1, no_jit=False, quad_order=24, log_span=0.1,
        m_real=1.45, m_imag=-0.005,
        m_reals=None, m_imags=None,
        noise=0.0, seed=1, out="result.png", json=None, no_plot=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parse_batch_txt():
    channels, altitudes, data = parse_batch_txt(DATA_FILE)
    kinds = [(c.kind, c.wl_um * 1e3) for c in channels]
    assert kinds == [("beta", 355.0), ("beta", 532.0), ("beta", 1064.0),
                     ("alpha", 355.0), ("alpha", 532.0)]
    assert altitudes.tolist() == [100.0, 500.0]
    assert data.shape == (2, 5)
    assert np.all(data > 0)


def test_parse_batch_txt_bad_header(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("H b355\n1 2e-6\n", encoding="utf-8")
    # 'H' допустим; проверим заведомо неверный заголовок
    bad.write_text("Foo b355\n1 2e-6\n", encoding="utf-8")
    try:
        parse_batch_txt(str(bad))
        raise AssertionError("ожидалось исключение")
    except ValueError:
        pass


def test_run_batch_outputs(tmp_path):
    out = tmp_path / "out"
    args = _args(batch_out_dir=str(out), no_plot=True)
    run_batch(args)

    files = sorted(p.name for p in out.iterdir())
    for name in ("moments.txt", "refractive.txt", "errors.txt",
                 "dist_100.txt", "dist_500.txt"):
        assert name in files, f"нет файла {name}: {files}"

    # moments.txt: 2 строки данных, колонки H Vf Vc V rmf rmc rm reff N σ...
    rows = [ln.split() for ln in (out / "moments.txt").read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    assert len(rows) == 2
    assert len(rows[0]) == 12
    h1 = float(rows[0][0])
    assert h1 in (100.0, 500.0)

    # refractive.txt: H Re(m) Im(m)
    refr = []
    for ln in (out / "refractive.txt").read_text(encoding="utf-8").splitlines():
        if ln and not ln.startswith("#"):
            t = ln.split()
            refr.append((float(t[0]), float(t[1]), float(t[2])))
    assert len(refr) == 2
    assert all(abs(im) <= 0 for _, _, im in refr)

    # errors.txt: H + ошибки по 5 каналам + Err
    err = [ln.split() for ln in (out / "errors.txt").read_text(encoding="utf-8").splitlines()
           if ln and not ln.startswith("#")]
    assert len(err) == 2
    assert len(err[0]) == 7  # H + 5 каналов + Err

    # dist_100.txt: комментарии в начале + столбцы r и dV/dln r
    dist = (out / "dist_100.txt").read_text(encoding="utf-8").splitlines()
    assert dist[0].startswith("# высота")
    assert any("входные данные" in ln for ln in dist[:3])
    body = [ln.split() for ln in dist if ln and not ln.startswith("#")]
    assert len(body) > 10
    assert len(body[0]) == 2
    assert float(body[0][1]) >= 0  # распределение неотрицательно


def test_kernel_table_built_once(monkeypatch, tmp_path):
    """Таблица ядер строится ровно один раз на весь пакет."""
    out = tmp_path / "out"
    calls = {"n": 0}
    orig = KernelTable.__init__

    def counting_init(self, *a, **kw):
        calls["n"] += 1
        orig(self, *a, **kw)

    monkeypatch.setattr(KernelTable, "__init__", counting_init)
    args = _args(batch_out_dir=str(out), no_plot=True,
                 m_reals=[1.45], m_imags=[-0.005])
    run_batch(args)
    assert calls["n"] == 1  # одна таблица ядер на пакет (не на высоту)
