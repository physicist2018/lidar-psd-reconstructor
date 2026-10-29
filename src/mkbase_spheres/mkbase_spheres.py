import os
import itertools
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import lognorm
import miepython as mp


def make_log_equidistant(x0: float, x1: float, nx: int) -> np.ndarray:
    if x0 <= 0 or x1 <= 0:
        raise ValueError("x0 и x1 должны быть > 0 для логарифмической сетки.")
    return np.exp(np.linspace(np.log(x0), np.log(x1), nx))


def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Возвращает дискретное гауссово ядро (нормированное на 1) для свёртки.
    sigma — в единицах индекса (шагов сетки).
    truncate — сколько сигм брать в каждую сторону.
    """
    radius = int(truncate * sigma + 0.5)
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_arrays(arrays: list[np.ndarray], sigma_idx: float) -> list[np.ndarray]:
    """
    Сглаживает каждый массив из списка arrays свёрткой с 1D-гауссовым ядром.
    arrays: список 1D массивов одинаковой длины.
    sigma_idx: ширина ядра в единицах шага сетки по индексу.
    Возвращает список сглаженных массивов.
    """
    if sigma_idx <= 0:
        return arrays
    kernel = gaussian_kernel_1d(sigma_idx)
    smoothed = [convolve1d(arr, kernel, mode='reflect') for arr in arrays]
    return smoothed


def compute_mie_cross_sections_units(
    x: np.ndarray, m_re: float, m_im: float, wavelength_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = complex(m_re, m_im)
    qext, qsca, qback, _g = mp.efficiencies_mx(m, x)
    qext = np.asarray(qext, dtype=float)
    qsca = np.asarray(qsca, dtype=float)
    qback = np.asarray(qback, dtype=float)
    qabs = qext - qsca

    a_um = (x * wavelength_um) / (2.0 * np.pi)
    Cgeo_um2 = np.pi * (a_um ** 2)  # πa^2 в мкм^2

    Cext = qext * Cgeo_um2
    Cabs = qabs * Cgeo_um2
    # Cback — дифференциальное сечение обратного рассеяния dσ/dΩ(180°) в мкм²/ср.
    # miepython: qback = 4|S(π)|²/x², а dσ/dΩ = a²·qback/4 (см. solve/kernels.py).
    # Так же пишет mkbase из ADDA (s11(θ) = dσ/dΩ, проверено по ∫s11 dΩ = Csca).
    Cback = qback * (a_um ** 2) / 4.0

    Qext = qext
    Qabs = qabs
    # Q = C/(πa²) — то же определение, что у mkbase
    Qback = Cback / Cgeo_um2

    return Cext, Cabs, Cback, Qext, Qabs, Qback


def format_m_value(val: float, is_real: bool = True) -> str:
    """Формат доли показателя преломления как у mkbase: `1.40`, `0.01000`."""
    return f"{val:.2f}" if is_real else f"{val:.5f}"


def generate_mie_tables(
    mre_values,
    mim_values,
    x0,
    x1,
    nx,
    wavelength_um,
    base_dir="base",
    smooth=False,
    sigma_idx=1.0,
):
    os.makedirs(base_dir, exist_ok=True)
    x_grid = make_log_equidistant(x0, x1, nx)
    cgeo_um2 = np.pi * ((x_grid * wavelength_um) / (2.0 * np.pi)) ** 2

    for m_re, m_im in itertools.product(mre_values, mim_values):
        mre_str = format_m_value(m_re, is_real=True)
        mim_str = format_m_value(m_im, is_real=False)
        subdir_name = f"mre_{mre_str}_mim_{mim_str}"
        subdir = Path(base_dir) / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)

        Cext, Cabs, Cback, Qext, Qabs, Qback = compute_mie_cross_sections_units(
            x_grid, m_re, m_im, wavelength_um
        )

        if smooth:
            # сглаживаем только сечения, Q пересчитываем из них:
            # иначе столбцы C и Q, сглаженные по отдельности, расходятся
            Cext, Cabs, Cback = smooth_arrays([Cext, Cabs, Cback], sigma_idx)
            Qext = Cext / cgeo_um2
            Qabs = Cabs / cgeo_um2
            Qback = Cback / cgeo_um2

        data_path = subdir / "data.txt"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("# x Cext Cabs Cback Qext Qabs Qback\n")
            f.write(
                "# Units: x (dimless), Cext/Cabs (um^2), Cback (um^2/sr), Q (dimensionless)\n"
            )
            if smooth:
                f.write(f"# Smoothed with Gaussian kernel, sigma_idx={sigma_idx:.2f}\n")
            for i in range(len(x_grid)):
                f.write(
                    f"{x_grid[i]:.6e} "
                    f"{Cext[i]:.6e} {Cabs[i]:.6e} {Cback[i]:.6e} "
                    f"{Qext[i]:.6e} {Qabs[i]:.6e} {Qback[i]:.6e}\n"
                )

    print(f"Таблицы рассеяния созданы в '{base_dir}'.")
    if smooth:
        print(f"Включено сглаживание: sigma_idx = {sigma_idx:.2f}")


def plot_cross_sections(
    x_grid: np.ndarray,
    Cext_raw: np.ndarray,
    Cext_sm: np.ndarray,
    Cback_raw: np.ndarray,
    Cback_sm: np.ndarray,
    out_path: str = "mie_Cext_Cback.png",
    m_re: float = 1.4,
    m_im: float = 0.0,
    sigma_idx: float = 1.0,
) -> bool:
    """
    Графики Cext(x) и Cback(x): сглаженные и несглаженные (log-log).
    Сохраняет в out_path (как solve: Agg + savefig).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[предупреждение] matplotlib недоступен: {exc}")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, raw, sm, ylabel, title in (
        (ax1, Cext_raw, Cext_sm, "Cext, мкм²", "Cext(x)"),
        (ax2, Cback_raw, Cback_sm, "Cback, мкм²/ср", "Cback(x)"),
    ):
        ax.semilogx(x_grid, raw/(np.pi*x_grid*x_grid), "C0-", lw=1.4, label="несглаженный")
        ax.semilogx(x_grid, sm/(np.pi*x_grid*x_grid), "C1--", lw=1.8,
                  label=f"сглаженный (σ_idx={sigma_idx:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title},  m = {m_re:.3f}{m_im:+.4f}j")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"график сохранён: {out_path}")
    return True


# -----------------------------------------------------------------------------
# Блок тестов для проверки расчётов коэффициентов ослабления
# -----------------------------------------------------------------------------
def test_extinction_backscatter(
    m_re=1.40,
    m_im=0.001,
    x0=0.1,
    x1=100.0,
    nx=500,
    wavelength_um=2.0 * np.pi,
    mu_lnr=1.0,      # медиана радиуса в мкм
    sigma_lnr=0.4,   # ширина логнормального распределения
    N0=1e6,          # концентрация в см^-3
    sigma_smooth=1.5,
    plot_path="mie_Cext_Cback.png",
):
    """
    Тест: сравниваем коэффициенты ослабления (extinction) и обратного рассеяния (backscatter)
    для заданного распределения частиц по размерам:
      - по «сырой» таблице (без сглаживания)
      - по сглаженной таблице (свёртка с гауссом)

    Распределение: логнормальное по радиусу a (в мкм):
      n(a) da = N0 * pdf_lognorm(a; mu_lnr, sigma_lnr) da

    Интегралы считаем по сетке x, переходя к радиусу: a = x * lambda / (2π).
    Коэффициенты:
      beta_ext = ∫ Cext(a) * n(a) da
      beta_back = ∫ Cback(a) * n(a) da
    """

    x_grid = make_log_equidistant(x0, x1, nx)

    # Сырые значения
    Cext_raw, Cabs_raw, Cback_raw, _, _, _ = compute_mie_cross_sections_units(
        x_grid, m_re, m_im, wavelength_um
    )

    # Сглаженные значения
    arrays = [Cext_raw, Cabs_raw, Cback_raw]
    Cext_sm, Cabs_sm, Cback_sm = smooth_arrays(arrays, sigma_smooth)

    # Переходим от x к радиусу a в мкм
    a_um = (x_grid * wavelength_um) / (2.0 * np.pi)

    # Логнормальное распределение по радиусу (в мкм)
    # scipy lognorm: scale = exp(mu) — медиана, s = sigma
    n_a = N0 * lognorm.pdf(a_um, s=sigma_lnr, scale=float(np.exp(mu_lnr)))

    # Шаг по радиусу: интегрируем по r: ∫ C(r)·n(r) dr
    beta_ext_raw = np.trapezoid(Cext_raw * n_a, a_um)
    beta_back_raw = np.trapezoid(Cback_raw * n_a, a_um)

    beta_ext_sm = np.trapezoid(Cext_sm * n_a, a_um)
    beta_back_sm = np.trapezoid(Cback_sm * n_a, a_um)

    rel_diff_ext = abs(beta_ext_raw - beta_ext_sm) / max(beta_ext_raw, 1e-30)
    rel_diff_back = abs(beta_back_raw - beta_back_sm) / max(beta_back_raw, 1e-30)

    print("\n=== Тест: сравнение коэффициентов ослабления и обратного рассеяния ===")
    print(f"Показатель преломления: m = {m_re} + i{m_im}")
    print(f"Распределение частиц: логнормальное, mu_ln(r)={mu_lnr}, sigma_ln(r)={sigma_lnr}")
    print(f"Концентрация N0 = {N0:.2e} см^-3")
    print(f"Сглаживание: sigma_idx = {sigma_smooth:.2f}")
    print()
    print(f"beta_ext (raw)   = {beta_ext_raw:.6e} см^-1")
    print(f"beta_ext (smooth) = {beta_ext_sm:.6e} см^-1")
    print(f"Относительная разница beta_ext: {rel_diff_ext:.4%}")
    print()
    print(f"beta_back (raw)  = {beta_back_raw:.6e} см^-1·ср^-1")
    print(f"beta_back (smooth)= {beta_back_sm:.6e} см^-1·ср^-1")
    print(f"Относительная разница beta_back: {rel_diff_back:.4%}")
    print("=====================================================================\n")

    plot_cross_sections(
        x_grid, Cext_raw, Cext_sm, Cback_raw, Cback_sm,
        out_path=plot_path, m_re=m_re, m_im=m_im, sigma_idx=sigma_smooth,
    )

    return {
        "beta_ext_raw": beta_ext_raw,
        "beta_ext_smooth": beta_ext_sm,
        "beta_back_raw": beta_back_raw,
        "beta_back_smooth": beta_back_sm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Генерация таблиц рассеяния Ми с опциональным сглаживанием."
    )
    parser.add_argument("--mre", nargs="+", type=float, default=[1.30, 1.35, 1.40],
                        help="Действительные части показателя преломления.")
    parser.add_argument("--mim", nargs="+", type=float, default=[0.0, 0.001, 0.01],
                        help="Мнимые части показателя преломления.")
    parser.add_argument("--x0", type=float, default=0.1, help="Нижняя граница x.")
    parser.add_argument("--x1", type=float, default=100.0, help="Верхняя граница x.")
    parser.add_argument("--nx", type=int, default=500, help="Число точек сетки.")
    parser.add_argument("--lambda", dest="wavelength_um", type=float,
                        default=2.0 * np.pi, help="Длина волны в мкм.")
    parser.add_argument("--base-dir", type=str, default="base", help="Корневая папка.")
    parser.add_argument("--smooth", action="store_true",
                        help="Включить сглаживание сечений и факторов эффективности.")
    parser.add_argument("--sigma-idx", type=float, default=1.0,
                        help="Ширина гауссова ядра в единицах индекса сетки.")
    parser.add_argument("--test", action="store_true",
                        help="Запустить тестовый расчёт коэффициентов ослабления.")

    args = parser.parse_args()

    if args.test:
        test_extinction_backscatter(
            m_re=args.mre[0],
            m_im=args.mim[0],
            x0=args.x0,
            x1=args.x1,
            nx=args.nx,
            wavelength_um=args.wavelength_um,
            sigma_smooth=args.sigma_idx,
        )
    else:
        generate_mie_tables(
            args.mre,
            args.mim,
            args.x0,
            args.x1,
            args.nx,
            args.wavelength_um,
            base_dir=args.base_dir,
            smooth=args.smooth,
            sigma_idx=args.sigma_idx,
        )


if __name__ == "__main__":
    main()
