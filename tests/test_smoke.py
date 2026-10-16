"""Быстрые проверки ключевых блоков: базис, ядра, инверсия, ансамбль.

Запуск:  uv run python tests/test_smoke.py
(также совместимо с pytest:  uv run pytest tests/ -q)
"""

from __future__ import annotations

import numpy as np

from solve import (
    DEMO_ALPHA_WL_NM,
    DEMO_BETA_WL_NM,
    demo_true_distribution,
    forward_measurements,
)
from solve.basis import bspline_basis, design_matrix, quadrature_grid
from solve.inverse import (
    ensemble_average,
    residual_err,
    scan_variants,
    second_diff_matrix,
    solve_regularized,
)
from solve.kernels import Channel, KernelTable

M = 1.45 - 0.005j


def make_channels():
    return (
        [Channel("beta", wl * 1e-3) for wl in DEMO_BETA_WL_NM]
        + [Channel("alpha", wl * 1e-3) for wl in DEMO_ALPHA_WL_NM]
    )


def test_basis_partition_of_unity() -> None:
    """Сумма базисных функций равна 1 на [r0, r1] для всех степеней 1..4."""
    for k in (1, 2, 3, 4):
        basis, _knots = bspline_basis(0.1, 5.0, k, 5)
        r = np.geomspace(0.1, 5.0, 200)
        total = sum(b(r) for b in basis)
        assert np.allclose(total, 1.0, atol=1e-10), f"степень {k}"
        # вне окна scipy-BSpline (extrapolate=False) даёт NaN — это ожидаемо
        # (обнуление вне окна выполняется в Solution.distribution)
        r_out = np.array([0.05, 7.0])
        for b in basis:
            assert np.all(np.isnan(b(r_out))), f"степень {k}"


def test_boundary_zero() -> None:
    """Граничные условия: каждая функция внутреннего базиса (и любое
    разложение по нему) равна нулю в точках r0 и r1."""
    from solve import forward_measurements

    # 1) базис: все функции interior_only-базиса нулевые на концах
    for k in (1, 2, 3, 4):
        basis, _knots = bspline_basis(0.1, 5.0, k, 5, interior_only=True)
        assert len(basis) == 3  # M - 2
        for b in basis:
            v0 = float(b(np.array([0.1]))[0])
            v1 = float(b(np.array([5.0]))[0])
            assert abs(v0) < 1e-12, f"степень {k}: B(r0)={v0}"
            assert abs(v1) < 1e-12, f"степень {k}: B(r1)={v1}"

    # 2) сквозной прогон: n(r0) = n(r1) = 0 у каждого решения
    channels = make_channels()
    kern = KernelTable(M, channels, 0.05, 5.0, dx_max=0.1)
    g1 = forward_measurements(channels, kern, demo_true_distribution, 0.05, 5.0)
    sols, _scale = scan_variants(
        channels, {M: kern}, g1, [2], np.geomspace(0.05, 0.3, 2),
        np.geomspace(2.0, 5.0, 2),
        min_ratio=4.0, gamma_select="fixed", gamma_rel_fixed=0.1, n_gamma=5,
        zero_boundary=True, order=12, max_log_span=0.2,
    )
    assert sols
    for s in sols:
        assert len(s.C) == 3  # M - 2 коэффициентов
        n0 = float(s.distribution(np.array([s.r0]))[0])
        n1 = float(s.distribution(np.array([s.r1]))[0])
        assert abs(n0) < 1e-9, f"n(r0)={n0}"
        assert abs(n1) < 1e-9, f"n(r1)={n1}"


def test_second_diff_matrix() -> None:
    L = second_diff_matrix(5)
    assert L.shape == (3, 5)
    assert np.array_equal(L[0], [1, -2, 1, 0, 0])
    assert np.array_equal(L[2], [0, 0, 1, -2, 1])


def test_design_matrix_parts() -> None:
    """A согласована с квадратурой: A[j,i] = sum_q w_q K_j B_i."""
    channels = make_channels()
    kern = KernelTable(M, channels, 0.1, 5.0, dx_max=0.1)
    basis, knots = bspline_basis(0.1, 5.0, 2, 5)
    A = design_matrix(channels, kern, basis, knots, order=12, max_log_span=0.2)
    r_q, w_q = quadrature_grid(np.unique(knots), order=12, max_log_span=0.2)
    b_vals = np.array([b(r_q) for b in basis])
    k_vals = np.array([kern.kernel(j, r_q) for j in range(len(channels))])
    A_ref = (k_vals * w_q) @ b_vals.T
    assert np.allclose(A, A_ref, rtol=1e-10)


def test_kernels_positive() -> None:
    channels = make_channels()
    kern = KernelTable(M, channels, 0.05, 5.0, dx_max=0.1)
    r = np.geomspace(0.05, 5.0, 300)
    for j in range(len(channels)):
        k = kern.kernel(j, r)
        assert np.all(np.isfinite(k)) and np.all(k > 0)


def test_solve_regression_small() -> None:
    """Сквозной прогон на малой сетке: решение конечно, Err мал на обучении."""
    channels = make_channels()
    kern = KernelTable(M, channels, 0.05, 5.0, dx_max=0.1)
    g1 = forward_measurements(channels, kern, demo_true_distribution, 0.05, 5.0)
    r0g = np.geomspace(0.05, 0.3, 2)
    r1g = np.geomspace(2.0, 5.0, 2)
    sols, _scale = scan_variants(
        channels, {M: kern}, g1, [2], r0g, r1g,
        min_ratio=4.0, gamma_select="fixed", gamma_rel_fixed=0.1, n_gamma=5,
        order=12, max_log_span=0.2,
    )
    assert len(sols) >= 1
    for s in sols:
        assert np.all(np.isfinite(s.C))
        assert np.isfinite(s.err) and s.err >= 0
        assert s.m == M
    rg = np.geomspace(0.05, 5.0, 100)
    n_avg, ns, m_avg = ensemble_average(sols, 3, rg)
    ok = np.isfinite(n_avg)
    assert ok.any()
    # незначимые отрицательные значения допустимы (неограниченное решение),
    # но не на уровне амплитуды распределения
    peak = max(np.nanmax(ns), 1.0)
    assert n_avg[ok].min() > -0.1 * peak
    assert abs(m_avg - M) < 1e-12


def test_scan_over_refractive_index() -> None:
    """Перебор по сетке m: решения с разными m, средний m корректен."""
    channels = make_channels()
    g1 = forward_measurements(channels, KernelTable(M, channels, 0.05, 5.0, dx_max=0.1),
                              demo_true_distribution, 0.05, 5.0)
    m1 = 1.40 - 0.01j
    m2 = M
    m3 = 1.50 + 0j
    tables = {m1: KernelTable(m1, channels, 0.05, 5.0, dx_max=0.1),
              m2: KernelTable(m2, channels, 0.05, 5.0, dx_max=0.1),
              m3: KernelTable(m3, channels, 0.05, 5.0, dx_max=0.1)}
    sols, _scale = scan_variants(
        channels, tables, g1, [1], np.geomspace(0.05, 0.3, 2),
        np.geomspace(2.0, 5.0, 2),
        min_ratio=4.0, gamma_select="fixed", gamma_rel_fixed=0.1, n_gamma=5,
        order=12, max_log_span=0.2,
    )
    ms_seen = {s.m for s in sols}
    assert ms_seen == {m1, m2, m3}
    rg = np.geomspace(0.05, 5.0, 50)
    _n_avg, _ns, m_avg = ensemble_average(sols, 3, rg)
    mean_ref = complex(sum(s.m for s in sols[:3]) / 3)
    assert abs(m_avg - mean_ref) < 1e-12


def test_size_moments_physical() -> None:
    """Моменты положительны и разумны для неотрицательного распределения."""
    from solve import size_moments

    r = np.geomspace(0.05, 5.0, 200)
    n = demo_true_distribution(r)  # в м^-3·мкм^-1, амплитуды ~10^8
    mom = size_moments(n, r)
    assert mom["N_m-3"] > 0
    assert mom["vol_mean_um3"] > 0
    assert mom["vol_conc_um3_per_m3"] > 0
    assert 0.3 < mom["r_eff_um"] < 1.5  # моды 0.08/0.45 мкм
    assert mom["cov_frac"] == 1.0
    # отрицательные артефакты отбрасываются: моменты не отрицательны
    n2 = n.copy()
    n2[::5] = -n2[::5]
    mom2 = size_moments(n2, r)
    assert mom2["N_m-3"] >= 0 and mom2["r_eff_um"] >= 0


def test_configurable_n_basis() -> None:
    """Число базисных функций настраивается; валидация ограничений."""
    import warnings

    channels = make_channels()
    kern = KernelTable(M, channels, 0.05, 5.0, dx_max=0.1)
    g1 = forward_measurements(channels, kern, demo_true_distribution, 0.05, 5.0)
    r0g = np.geomspace(0.05, 0.3, 2)
    r1g = np.geomspace(2.0, 5.0, 2)
    common = dict(min_ratio=4.0, gamma_select="fixed", gamma_rel_fixed=0.1,
                  n_gamma=5, order=12, max_log_span=0.2)

    # по умолчанию: n_basis = число каналов (5); с ГУ длина C = 3
    sols, _ = scan_variants(channels, {M: kern}, g1, [2], r0g, r1g, **common)
    assert all(len(s.C) == 3 for s in sols)

    # явное n_basis=7: с ГУ длина C = 5; без ГУ: 7
    sols7, _ = scan_variants(channels, {M: kern}, g1, [2], r0g, r1g,
                             n_basis=7, **common)
    assert sols7 and all(len(s.C) == 5 for s in sols7)
    sols4, _ = scan_variants(channels, {M: kern}, g1, [2], r0g, r1g,
                             n_basis=4, zero_boundary=False, **common)
    assert sols4 and all(len(s.C) == 4 for s in sols4)

    # валидация: невозможные конфигурации -> ValueError
    def expect_error(**kw) -> None:
        try:
            scan_variants(channels, {M: kern}, g1, [2], r0g, r1g, **common, **kw)
        except ValueError:
            return
        raise AssertionError(f"ожидалась ошибка: {kw}")

    expect_error(n_basis=2)                    # < степень+1 = 3
    expect_error(n_basis=2, zero_boundary=True)  # ГУ требуют n_basis >= 3

    # предупреждение о недоопределённости (неизвестных > измерений)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scan_variants(channels, {M: kern}, g1, [2], r0g, r1g, n_basis=8, **common)
        assert any("недоопределен" in str(x.message) for x in w)


def test_weights_change_result() -> None:
    """Веса 1/g1 (относительные) меняют решение переопределённой системы."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 5)) + 3.0
    g1 = np.abs(rng.normal(size=6)) + 1.0
    c_abs = solve_regularized(A, g1, 1e-3, weights=np.ones(6))
    c_rel = solve_regularized(A, g1, 1e-3, weights=1.0 / g1)
    assert not np.allclose(c_abs, c_rel)
    e_abs = residual_err(A, c_abs, g1)
    e_rel = residual_err(A, c_rel, g1)
    # относительные веса должны давать меньшую остаточную невязку Err
    assert e_rel <= e_abs + 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"все {len(fns)} проверок пройдены")