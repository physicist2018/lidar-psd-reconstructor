/* Текстовый вывод результатов (сообщения совпадают с Python-версией). */
#include "csolve.h"

#include <stdio.h>
#include <string.h>

static void channel_name(const channel_t *ch, char *buf, size_t n)
{
    snprintf(buf, n, "%s(%g nm)", ch->kind == CH_BETA ? "beta" : "alpha",
             ch->wl_um * 1e3);
}

void report_summary(const args_t *a, const solutions_t *sols,
                    const double *g1, const channel_t *ch, int nch,
                    const double *n_avg, const double *g_fit,
                    const double *r_grid, int nr, double m_avg_re,
                    double m_avg_im)
{
    int G = a->best < sols->n ? a->best : sols->n;
    printf("\n=== Solution ensemble (scan over m, r0, r1) ===\n");
    printf("variants scanned: %d; averaging: G = %d\n", sols->n, G);
    printf("%4s %13s %7s %9s %9s %10s %9s\n",
           "rank", "m", "degree", "r0, µm", "r1, µm", "gamma", "Err");
    for (int t = 0; t < sols->n; t++) {
        const solution_t *s = &sols->sol[t];
        const char *mark = t == 0 ? " <--" : "";
        printf("%4d %6.3f%+7.3fj %7d %9.4g %9.4g %10.3g %9.5f%s\n",
               t + 1, s->m_re, s->m_im, s->degree, s->r0, s->r1,
               s->gamma, s->err, mark);
        if (t + 1 >= G)
            break;
    }
    printf("\nmean refractive index (over %d best): "
           "m = %.3f%+.3fj\n", G, m_avg_re, m_avg_im);

    printf("\n=== Mean distribution function (dV/dln r) ===\n");
    printf("key points (full grid — in --json, if requested):\n");
    int npts = nr < 7 ? nr : 7;
    for (int k = 0; k < npts; k++) {
        int i = npts > 1
                    ? (int)((double)k * (double)(nr - 1) / (double)(npts - 1) + 1e-9)
                    : 0;
        if (i >= nr)
            i = nr - 1;
        double r = r_grid[i];
        if (!isfinite(n_avg[i])) {
            printf("  r = %9.4g µm,  dV/dln r = no coverage "
                   "(outside the windows of the best solutions)\n", r);
            continue;
        }
        printf("  r = %9.4g µm,  dV/dln r = %.4g µm³/cm³\n", r,
               n_avg[i] / CM3_PER_M3);
    }

    printf("\n=== Channel agreement (mean solution) ===\n");
    char name[64];
    double err_avg = 0.0;
    for (int j = 0; j < nch; j++) {
        channel_name(&ch[j], name, sizeof(name));
        double rel = fabs(g_fit[j] - g1[j]) / g1[j] * 100.0;
        printf("  %-14s g1 = %.6g   g = %.6g   |g-g1|/g1 = %.2f%%\n",
               name, g1[j], g_fit[j], rel);
        err_avg += fabs(g_fit[j] - g1[j]) / g1[j];
    }
    err_avg /= (double)nch;
    printf("  Err (mean solution) = %.5f\n", err_avg);
}

void report_moments(const moments_t *m)
{
    printf("\n=== Integral characteristics "
           "(estimates over the covered region) ===\n");
    printf("  covered radius range: %.4g–%.4g µm (grid fraction: %.0f%%)\n",
           m->r_cov_min_um, m->r_cov_max_um, m->cov_frac * 100.0);
    printf("  number concentration N      = %.4g cm⁻³\n", m->N_cm3);
    printf("  volume concentration C_v    = %.4g µm³/cm³\n",
           m->vol_conc_um3_per_cm3);
    printf("  mean particle volume ⟨V⟩    = %.4g µm³ (%.4g m³)\n",
           m->vol_mean_um3, m->vol_mean_um3 * UM3_TO_M3);
    printf("  effective radius r_eff      = %.4g µm\n", m->r_eff_um);
}

void report_modes(const modes_t *m, double r_split, const char *method)
{
    printf("\n=== Log-normal mode parameters (dV/dln r) ===\n");
    printf("  mode split boundary: %g µm (--r-split)\n", r_split);
    printf("  parameter estimation method: %s (--modes)\n", method);
    printf("  %-8s %12s %9s %7s   note\n",
           "mode", "V, µm³/cm³", "rm, µm", "sigma");
    const char *names[3] = {"fine", "coarse", "overall"};
    const cs_mode_t *vals[3] = {&m->fine, &m->coarse, &m->overall};
    for (int i = 0; i < 3; i++) {
        const cs_mode_t *v = vals[i];
        if (v->ok) {
            char note[128];
            if (strcmp(v->method, "fit") == 0)
                note[0] = '\0';
            else
                snprintf(note, sizeof(note), "(%s)", v->method);
            printf("  %-8s %12.4g %9.4g %7.4g   %s\n", names[i],
                   v->V / CM3_PER_M3, v->rm, v->sigma, note);
        } else {
            printf("  %-8s %12s %9s %7s   %s\n", names[i], "—", "—", "—",
                   v->method);
        }
    }
}
