/* Итоговый профиль одной высоты: ансамбль, расчётные g, моменты и моды.
 * Общий код одиночного (main) и пакетного (batch) режимов.
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

static int m_index_of(const double *mre, const double *mim, int nm,
                      double re, double im)
{
    for (int i = 0; i < nm; i++)
        if (mre[i] == re && mim[i] == im)
            return i;
    return -1;
}

void profile_finalize(profile_t *p, const args_t *a,
                      const channel_t *ch, int nch,
                      const ktable_t *kts, const double *mre,
                      const double *mim, int nm,
                      const solutions_t *sols, const geom_t *geom,
                      double r_min, double r_max)
{
    (void)ch;
    memset(p, 0, sizeof(*p));
    int nr = a->nr_out;
    p->nr = nr;
    p->r_grid = geomspace(r_min, r_max, nr);
    p->n_avg = dbl_array(nr);
    p->n_avg_raw = dbl_array(nr);

    double *n_best = NULL;
    double mre_avg = 0.0, mim_avg = 0.0;
    ensemble_average(sols, a->best, geom, p->r_grid, nr, !a->no_smooth,
                     a->smooth_lam, p->n_avg, &n_best, &mre_avg, &mim_avg);
    ensemble_average(sols, a->best, geom, p->r_grid, nr, 0, a->smooth_lam,
                     p->n_avg_raw, NULL, NULL, NULL);
    p->n_best = n_best;
    int G = a->best;
    if (G < 1)
        G = 1;
    if (G > sols->n)
        G = sols->n;
    p->n_best_count = G;
    p->m_re = mre_avg;
    p->m_im = mim_avg;

    /* расчётные g по усреднённым ядрам G лучших решений */
    int *best_idx = int_array(G);
    for (int t = 0; t < G; t++) {
        best_idx[t] = m_index_of(mre, mim, nm, sols->sol[t].m_re,
                                 sols->sol[t].m_im);
        if (best_idx[t] < 0)
            cs_fatal("внутренняя ошибка: m = %g%+gj отсутствует в сетке ядер",
                     sols->sol[t].m_re, sols->sol[t].m_im);
    }

    double breaks[2] = {r_min, r_max};
    double *rq = NULL, *wq = NULL;
    int nq = 0;
    quadrature_grid(breaks, 2, a->quad_order, a->log_span, &rq, &wq, &nq);
    double *navg0 = dbl_array(nr);
    for (int i = 0; i < nr; i++)
        navg0[i] = isfinite(p->n_avg[i]) ? p->n_avg[i] : 0.0;
    double *nq_vals = dbl_array(nq);
    for (int k = 0; k < nq; k++)
        nq_vals[k] = interp_clamped(rq[k], p->r_grid, navg0, nr);

    p->g_fit = dbl_array(nch);
    for (int j = 0; j < nch; j++) {
        double s = 0.0;
        for (int k = 0; k < nq; k++) {
            double ksum = 0.0;
            for (int t = 0; t < G; t++)
                ksum += ktable_kernel(&kts[best_idx[t]], j, rq[k]);
            double kavg = ksum / (double)G;
            double r = rq[k];
            double q = 3.0 / (4.0 * CS_PI * r * r * r * r);
            s += wq[k] * kavg * q * nq_vals[k];
        }
        p->g_fit[j] = s;
    }

    p->moments = size_moments(p->n_avg, p->r_grid, nr);

    /* кратность покрытия точек сетки лучшими решениями — веса анализа мод */
    p->count = dbl_array(nr);
    for (int t = 0; t < G; t++) {
        const solution_t *s = &sols->sol[t];
        for (int i = 0; i < nr; i++)
            if (p->r_grid[i] >= s->r0 && p->r_grid[i] <= s->r1)
                p->count[i] += 1.0;
    }
    p->modes = mode_analysis(p->r_grid, p->n_avg, nr, a->r_split, p->count,
                             a->modes);

    free(best_idx);
    free(rq);
    free(wq);
    free(navg0);
    free(nq_vals);
}

void profile_free(profile_t *p)
{
    if (!p)
        return;
    free(p->r_grid);
    free(p->n_avg);
    free(p->n_avg_raw);
    free(p->n_best);
    free(p->g_fit);
    free(p->count);
    memset(p, 0, sizeof(*p));
}
