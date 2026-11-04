/* Усреднение по лучшим решениям и сглаживание среднего.
 *
 * Среднее считается только по решениям, чьё окно [r0, r1] покрывает точку r;
 * точки без покрытия остаются NaN. Сглаживание повторяет scipy
 * make_smoothing_spline через SPD-постановку в B-сплайн-базисе:
 *   min sum_i w_i (y_i - f_i)^2 + lam * integral (f'')^2 dx,
 * с нормированными координатами и усиленными весами краёв (1e3).
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

static double dist_at(const solution_t *s, const geom_t *g, double r,
                      double *full)
{
    if (r < s->r0 || r > s->r1)
        return 0.0;
    bspline_values(g->knots, g->n_knots, g->degree, r, full);
    int off = g->interior_only ? 1 : 0;
    double sum = 0.0;
    for (int i = 0; i < g->n_unknown; i++)
        sum += s->C[i] * full[off + i];
    return sum;
}

double solution_distribution(const solution_t *s, const geom_t *g, double r)
{
    double *full = dbl_array(g->n_basis);
    double v = dist_at(s, g, r, full);
    free(full);
    return v;
}

void smooth_distribution(const double *r_grid, int n, double *n_avg,
                         const double *w, double lam_rel)
{
    int cnt = 0;
    int *ok = int_array(n);
    for (int i = 0; i < n; i++) {
        if (isfinite(n_avg[i])) {
            ok[i] = 1;
            cnt++;
        }
    }
    if (cnt < 4) {
        free(ok);
        return;
    }

    double *t = dbl_array(cnt);
    double *y = dbl_array(cnt);
    double *wts = dbl_array(cnt);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (!ok[i])
            continue;
        t[k] = log(r_grid[i]);
        y[k] = n_avg[i];
        wts[k] = w ? w[i] : 1.0;
        k++;
    }
    double t0 = t[0], t1 = t[cnt - 1];
    if (!(t1 > t0)) {
        free(ok); free(t); free(y); free(wts);
        return;
    }
    for (int i = 0; i < cnt; i++)
        t[i] = (t[i] - t0) / (t1 - t0);

    double scale = 0.0;
    for (int i = 0; i < cnt; i++)
        if (fabs(y[i]) > scale)
            scale = fabs(y[i]);
    if (!(scale > 0.0) || !isfinite(scale)) {
        free(ok); free(t); free(y); free(wts);
        return;
    }
    double *ys = dbl_array(cnt);
    for (int i = 0; i < cnt; i++)
        ys[i] = y[i] / scale;

    double wmax = 0.0;
    for (int i = 0; i < cnt; i++)
        if (wts[i] > wmax)
            wmax = wts[i];
    double wden = wmax > 1e-300 ? wmax : 1e-300;
    for (int i = 0; i < cnt; i++)
        wts[i] /= wden;
    wts[0] = 1e3;
    wts[cnt - 1] = 1e3;

    if (cnt < 5) {
        free(ok); free(t); free(y); free(ys); free(wts);
        return;
    }

    double *fit = dbl_array(cnt);
    smoothing_spline_fit(t, ys, wts, cnt, lam_rel, t, cnt, fit);

    /* fit соответствует сжатому набору точек — разложить обратно */
    k = 0;
    for (int i = 0; i < n; i++) {
        if (!ok[i])
            continue;
        n_avg[i] = fit[k] * scale;
        k++;
    }

    free(ok); free(t); free(y); free(ys); free(wts); free(fit);
}

void ensemble_average(const solutions_t *sols, int n_best,
                      const geom_t *geom, const double *r_grid, int nr,
                      int smooth, double lam_rel,
                      double *n_avg, double **n_best_mat, double *m_avg_re,
                      double *m_avg_im)
{
    int G = n_best;
    if (G < 1)
        G = 1;
    if (G > sols->n)
        G = sols->n;
    double *total = dbl_array(nr);
    double *count = dbl_array(nr);
    double *full = dbl_array(geom[0].n_basis);
    double *mat = NULL;
    if (n_best_mat)
        mat = xcalloc((size_t)(G > 0 ? G : 1) * (size_t)nr, sizeof(double));

    double mre = 0.0, mim = 0.0;
    for (int t = 0; t < G; t++) {
        const solution_t *s = &sols->sol[t];
        const geom_t *g = &geom[s->geom_index];
        for (int i = 0; i < nr; i++) {
            double v = dist_at(s, g, r_grid[i], full);
            if (mat)
                mat[(size_t)t * nr + i] = v;
            if (r_grid[i] >= s->r0 && r_grid[i] <= s->r1) {
                total[i] += v;
                count[i] += 1.0;
            }
        }
        mre += s->m_re;
        mim += s->m_im;
    }
    if (G > 0) {
        mre /= (double)G;
        mim /= (double)G;
    }
    for (int i = 0; i < nr; i++)
        n_avg[i] = count[i] > 0.0 ? total[i] / count[i] : NAN;
    if (smooth)
        smooth_distribution(r_grid, nr, n_avg, count, lam_rel);

    if (m_avg_re)
        *m_avg_re = mre;
    if (m_avg_im)
        *m_avg_im = mim;
    if (n_best_mat)
        *n_best_mat = mat;
    else
        free(mat);
    free(total);
    free(count);
    free(full);
}
