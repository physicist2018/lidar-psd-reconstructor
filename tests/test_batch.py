"""Тесты пакетной обработки (--batch).

Проверяем: парсинг txt, построение таблицы ядер ровно один раз,
и полный набор выходных файлов для нескольких высот.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from solve.batch import (
    BatchChannel,
    _apply_lidar_ratio_constraint,
    _common_lidar_wavelengths,
    _windowed_median_ref,
    parse_batch_txt,
    run_batch,
)
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
        gamma_min=-7.0, gamma_max=5.0, ngamma=41,
        gamma_select="fixed", gamma_rel=0.1,
        nonneg=False, no_zero_boundary=False, no_smooth=False,
        smooth_lam=1e-4, r_split=0.5, absolute=False,
        modes="auto",
        dx_max=0.1, no_jit=False, quad_order=24, log_span=0.1,
        m_real=1.45, m_imag=-0.005,
        m_reals=None, m_imags=None,
        tables=None, table_extrapolate=False,
        noise=0.0, seed=1, out="result.png", json=None, no_plot=True,
        lidar_ratio_cons=False, lidar_ratio_w=0.5, lidar_ratio_iter=3,
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
    assert all(im <= 0 for _, _, im in refr)  # поглощающий аэрозоль: Im(m) <= 0

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


def test_lidar_ratio_file_written(tmp_path):
    """--lidar-ratio-cons: создаётся lidar_ratios.txt с S по общим длинам волн."""
    out = tmp_path / "out"
    args = _args(batch_out_dir=str(out), no_plot=True, lidar_ratio_cons=True)
    run_batch(args)
    path = out / "lidar_ratios.txt"
    assert path.exists()
    rows = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    assert len(rows) == 2                       # по строке на высоту
    assert len(rows[0]) == 3                    # H + S_355 + S_532
    assert all(float(v) > 0 for v in rows[0][1:])


def test_lidar_ratio_off_no_file(tmp_path):
    """Без --lidar-ratio-cons файл lidar_ratios.txt не создаётся."""
    out = tmp_path / "out"
    args = _args(batch_out_dir=str(out), no_plot=True)
    run_batch(args)
    assert not (out / "lidar_ratios.txt").exists()


def test_common_lidar_wavelengths():
    """Общие длины волн β∩α ищутся по kind/wl (не по порядку в заголовке)."""
    ch = [BatchChannel("beta", 0.532, "b532"), BatchChannel("alpha", 0.355, "a355"),
          BatchChannel("beta", 0.355, "b355"), BatchChannel("alpha", 1.064, "a1064")]
    common = _common_lidar_wavelengths(ch)
    assert common == [(0.355, 2, 1)]


def test_lidar_ratio_rerank():
    """Решение с чуть худшим Err, но типичным S, обходит выброс по S."""
    from types import SimpleNamespace

    def sol(name, err, s_ratio):
        return SimpleNamespace(name=name, err=err,
                               g_fit=np.array([1e-5, s_ratio * 1e-5], dtype=float))

    channels = [BatchChannel("beta", 0.355, "b355"),
                BatchChannel("alpha", 0.355, "a355")]
    common = _common_lidar_wavelengths(channels)
    assert len(common) == 1

    per_alt = [
        [sol("A0", 0.05, 30.0), sol("B0", 0.06, 31.0)],
        [sol("A1", 0.05, 30.0), sol("B1", 0.06, 31.0)],
        [sol("A2", 0.05, 100.0), sol("B2", 0.10, 30.0)],
    ]
    altitudes = [100.0, 300.0, 500.0]
    args = SimpleNamespace(lidar_ratio_w=0.5, lidar_ratio_iter=3, best=1)
    _apply_lidar_ratio_constraint(per_alt, altitudes, common, args)
    assert [s.name for s in per_alt[0]] == ["A0", "B0"]
    assert [s.name for s in per_alt[1]] == ["A1", "B1"]
    # высота-выброс: наверх выходит решение с типичным S, а не с лучшим Err
    assert per_alt[2][0].name == "B2"


def test_windowed_median_ref():
    """Скользящая медиана по высоте: окно 3, у краёв сужается."""
    S = np.array([[30.0], [30.0], [100.0], [30.0]])
    out = _windowed_median_ref(S).ravel()
    # h=0: [30,30] -> 30; h=1: [30,30,100] -> 30; h=2: [30,100,30] -> 30;
    # h=3 (край, окно из двух): [100,30] -> 65 (медиана двух = среднее)
    expected = [30.0, 30.0, 30.0, 65.0]
    assert np.allclose(out, expected)
