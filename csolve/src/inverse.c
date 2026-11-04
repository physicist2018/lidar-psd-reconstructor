/* Обратная задача: перебор (m, r0, r1, степень) с регуляризацией по кривизне.
 *
 * min ||W (A C - g1)||^2 + gamma^2 ||L C||^2, L — матрица вторых разностей.
 * gamma выбирается фиксированным (доля ||A||_2), по LOO-кросс-валидации
 * (закрытая формула через leverage) или по минимуму невязки Err.
 */
#include "csolve.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

static int n_lrows(int nu)
{
    return nu > 2 ? nu - 2 : 0;
}

/* Расширенная система [W A; gamma L] c = [W g1n; 0]. */
static void build_aug(const double *A, int nch, int nu, double gamma,
                      const double *weights, const double *g1n,
                      double *aug, double *rhs)
{
    int nL = n_lrows(nu);
    for (int j = 0; j < nch; j++) {
        for (int i = 0; i < nu; i++)
            aug[(size_t)j * nu + i] = weights[j] * A[(size_t)j * nu + i];
        rhs[j] = weights[j] * g1n[j];
    }
    for (int k = 0; k < nL; k++) {
        double *row = aug + (size_t)(nch + k) * nu;
        for (int i = 0; i < nu; i++)
            row[i] = 0.0;
        row[k] = gamma;
        row[k + 1] = -2.0 * gamma;
        row[k + 2] = gamma;
        rhs[nch + k] = 0.0;
    }
}

static void solve_gamma(const double *A, int nch, int nu, double gamma,
                        const double *weights, const double *g1n, int nonneg,
                        double *c)
{
    int nL = n_lrows(nu);
    int rows = nch + nL;
    double *aug = xcalloc((size_t)rows * (size_t)nu, sizeof(double));
    double *rhs = dbl_array(rows);
    build_aug(A, nch, nu, gamma, weights, g1n, aug, rhs);
    if (nonneg)
        nnls(aug, rows, nu, rhs, c);
    else
        lstsq(aug, rows, nu, rhs, c);
    free(aug);
    free(rhs);
}

static double residual_err(const double *A, int nch, int nu, const double *c,
                           const double *g1n)
{
    double sum = 0.0;
    for (int j = 0; j < nch; j++) {
        double p = 0.0;
        for (int i = 0; i < nu; i++)
            p += A[(size_t)j * nu + i] * c[i];
        sum += fabs(p - g1n[j]) / g1n[j];
    }
    return sum / (double)nch;
}

/* Брутфорс LOO-кросс-валидация (нужна для NNLS и вырожденных leverage). */
static double cv_bruteforce(const double *A, int nch, int nu, double gamma,
                            const double *weights, const double *g1n,
                            int nonneg)
{
    int nL = n_lrows(nu);
    int rows = (nch - 1) + nL;
    double *aug = xcalloc((size_t)(rows > 0 ? rows : 1) * (size_t)nu,
                          sizeof(double));
    double *rhs = dbl_array(rows > 0 ? rows : 1);
    double *c = dbl_array(nu);
    double *wsub = dbl_array(nch);
    double *gsub = dbl_array(nch);
    double total = 0.0;
    for (int skip = 0; skip < nch; skip++) {
        int r = 0;
        for (int j = 0; j < nch; j++) {
            if (j == skip)
                continue;
            for (int i = 0; i < nu; i++)
                aug[(size_t)r * nu + i] =
                    weights[j] * A[(size_t)j * nu + i];
            rhs[r] = weights[j] * g1n[j];
            r++;
        }
        for (int k = 0; k < nL; k++) {
            double *row = aug + (size_t)(nch - 1 + k) * nu;
            for (int i = 0; i < nu; i++)
                row[i] = 0.0;
            row[k] = gamma;
            row[k + 1] = -2.0 * gamma;
            row[k + 2] = gamma;
            rhs[nch - 1 + k] = 0.0;
        }
        if (nonneg)
            nnls(aug, rows, nu, rhs, c);
        else
            lstsq(aug, rows, nu, rhs, c);
        double pred = 0.0;
        for (int i = 0; i < nu; i++)
            pred += A[(size_t)skip * nu + i] * c[i];
        total += fabs(pred - g1n[skip]) / g1n[skip];
    }
    (void)wsub;
    (void)gsub;
    free(aug);
    free(rhs);
    free(c);
    free(wsub);
    free(gsub);
    return total;
}

/* LOO-оценка закрытой формулой; ok=0 — нужен брутфорс. */
static double cv_closed(const double *A, int nch, int nu, double gamma,
                        const double *weights, const double *g1n, int *ok)
{
    int nL = n_lrows(nu);
    int rows = nch + nL;
    double *aug = xcalloc((size_t)rows * (size_t)nu, sizeof(double));
    double *rhs = dbl_array(rows);
    build_aug(A, nch, nu, gamma, weights, g1n, aug, rhs);

    svd_t sv;
    svd_compute(&sv, aug, rows, nu);
    double smax = 0.0;
    for (int k = 0; k < nu; k++)
        if (sv.s[k] > smax)
            smax = sv.s[k];
    double cutoff = CS_EPS * (double)(rows > nu ? rows : nu) * smax;
    int *keep = int_array(nu);
    double *coef = dbl_array(nu);
    for (int k = 0; k < nu; k++) {
        keep[k] = sv.s[k] > cutoff && sv.s[k] != 0.0;
        if (!keep[k]) {
            coef[k] = 0.0;
            continue;
        }
        double dot = 0.0;
        for (int i = 0; i < rows; i++)
            dot += sv.w[(size_t)i * nu + k] * rhs[i];
        coef[k] = dot / (sv.s[k] * sv.s[k]);
    }
    double *c = dbl_array(nu);
    for (int i = 0; i < nu; i++) {
        double s = 0.0;
        for (int k = 0; k < nu; k++)
            s += sv.v[(size_t)i * nu + k] * coef[k];
        c[i] = s;
    }
    /* leverage строк A: s_ii = sum_p ( (V^T a_i)_p / s_p )^2 */
    double score = 0.0;
    *ok = 1;
    for (int j = 0; j < nch; j++) {
        double lev = 0.0;
        for (int p = 0; p < nu; p++) {
            if (!keep[p])
                continue;
            double q = 0.0;
            for (int i = 0; i < nu; i++)
                q += sv.v[(size_t)i * nu + p] * A[(size_t)j * nu + i];
            lev += (q / sv.s[p]) * (q / sv.s[p]);
        }
        double denom = 1.0 - weights[j] * weights[j] * lev;
        double pred = 0.0;
        for (int i = 0; i < nu; i++)
            pred += A[(size_t)j * nu + i] * c[i];
        if (!(denom > 0.0)) {
            *ok = 0;
            break;
        }
        score += fabs((g1n[j] - pred) / denom) / g1n[j];
    }
    free(aug);
    free(rhs);
    free(c);
    free(coef);
    free(keep);
    svd_free(&sv);
    return score;
}

/* ---------------- устойчивая сортировка ---------------- */
static void merge_sort(solution_t *a, solution_t *tmp, int lo, int hi)
{
    if (hi - lo < 2)
        return;
    int mid = (lo + hi) / 2;
    merge_sort(a, tmp, lo, mid);
    merge_sort(a, tmp, mid, hi);
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (a[j].err < a[i].err)
            tmp[k++] = a[j++];
        else
            tmp[k++] = a[i++];
    }
    while (i < mid)
        tmp[k++] = a[i++];
    while (j < hi)
        tmp[k++] = a[j++];
    for (int t = lo; t < hi; t++)
        a[t] = tmp[t];
}

void sort_solutions(solution_t *sol, int n)
{
    if (n < 2)
        return;
    solution_t *tmp = xcalloc((size_t)n, sizeof(solution_t));
    merge_sort(sol, tmp, 0, n);
    free(tmp);
}

void solutions_free(solutions_t *s)
{
    if (!s)
        return;
    for (int i = 0; i < s->n; i++) {
        free(s->sol[i].C);
        free(s->sol[i].g_fit);
    }
    free(s->sol);
    memset(s, 0, sizeof(*s));
}

void scan_variants(const design_t *d, const double *g1, int nch,
                   int gamma_select, double gamma_rel_fixed,
                   double gamma_exp_min, double gamma_exp_max, int n_gamma,
                   int nonneg, int relative, solutions_t *out)
{
    memset(out, 0, sizeof(*out));
    out->n = 0;
    if (d->n_w == 0 || d->n_m == 0) {
        out->scale = 1.0;
        return;
    }
    for (int j = 0; j < nch; j++)
        if (!(g1[j] > 0.0))
            cs_fatal("все измеренные коэффициенты g1 должны быть положительными");
    double scale = g1[0];
    for (int j = 1; j < nch; j++)
        if (g1[j] > scale)
            scale = g1[j];
    double *g1n = dbl_array(nch);
    double *weights = dbl_array(nch);
    for (int j = 0; j < nch; j++) {
        g1n[j] = g1[j] / scale;
        weights[j] = relative ? 1.0 / g1n[j] : 1.0;
    }
    if (d->nu > nch)
        fprintf(stderr, "[предупреждение] неизвестных на окно (%d) больше "
                        "измерений (%d): решение опирается главным образом на "
                        "регуляризацию (система недоопределена)\n",
                d->nu, nch);

    int nu = d->nu;
    int nm = d->n_m, nw = d->n_w;
    int total = nm * nw;
    double *grel = dbl_array(n_gamma > 0 ? n_gamma : 1);
    int ng = 1;
    if (gamma_select == GAMMA_FIXED) {
        grel[0] = gamma_rel_fixed;
    } else {
        ng = n_gamma > 0 ? n_gamma : 1;
        double *lin = linspace(gamma_exp_min, gamma_exp_max, ng);
        for (int k = 0; k < ng; k++)
            grel[k] = pow(10.0, lin[k]);
        free(lin);
    }

    double *c_norm = dbl_array(nu);
    double *c = dbl_array(nu);
    double *c_abs = dbl_array(nu);
    double *c_best = dbl_array(nu);
    double *gfit = dbl_array(nch);

    out->sol = xcalloc((size_t)total, sizeof(solution_t));
    out->n = total;
    out->scale = scale;

    int t = 0;
    for (int im = 0; im < nm; im++) {
        for (int iw = 0; iw < nw; iw++) {
            const double *A = d->A + (((size_t)im * nw + iw) * nch * nu);
            double s0 = d->s0[(size_t)im * nw + iw];
            double gamma = 0.0;

            if (gamma_select == GAMMA_FIXED) {
                gamma = grel[0] * s0;
                if (nonneg)
                    solve_gamma(A, nch, nu, gamma, weights, g1n, 1, c_norm);
                else
                    solve_gamma(A, nch, nu, gamma, weights, g1n, 0, c_norm);
            } else {
                double best = INFINITY;
                for (int k = 0; k < ng; k++) {
                    double gk = grel[k] * s0;
                    double score;
                    if (gamma_select == GAMMA_CV) {
                        if (nonneg) {
                            score = cv_bruteforce(A, nch, nu, gk, weights,
                                                  g1n, 1);
                        } else {
                            int ok = 0;
                            score = cv_closed(A, nch, nu, gk, weights, g1n,
                                              &ok);
                            if (!ok)
                                score = cv_bruteforce(A, nch, nu, gk, weights,
                                                      g1n, 0);
                        }
                    } else {
                        solve_gamma(A, nch, nu, gk, weights, g1n, nonneg, c);
                        score = residual_err(A, nch, nu, c, g1n);
                    }
                    if (score < best) {
                        best = score;
                        gamma = gk;
                    }
                }
                solve_gamma(A, nch, nu, gamma, weights, g1n, nonneg, c_norm);
            }

            for (int i = 0; i < nu; i++)
                c_abs[i] = c_norm[i] * scale;
            for (int j = 0; j < nch; j++) {
                double p = 0.0;
                for (int i = 0; i < nu; i++)
                    p += A[(size_t)j * nu + i] * c_abs[i];
                gfit[j] = p;
            }
            double err = 0.0;
            for (int j = 0; j < nch; j++)
                err += fabs(gfit[j] - g1[j]) / g1[j];
            err /= (double)nch;

            solution_t *sol = &out->sol[t];
            sol->m_re = d->m_re[im];
            sol->m_im = d->m_im[im];
            sol->degree = d->degree[iw];
            sol->r0 = d->r0[iw];
            sol->r1 = d->r1[iw];
            sol->gamma = gamma;
            sol->C = dbl_array(nu);
            memcpy(sol->C, c_abs, sizeof(double) * (size_t)nu);
            sol->nu = nu;
            sol->geom_index = iw;
            sol->err = err;
            sol->g_fit = dbl_array(nch);
            memcpy(sol->g_fit, gfit, sizeof(double) * (size_t)nch);
            sol->id = t;
            (void)c_best;
            t++;
        }
    }

    sort_solutions(out->sol, out->n);
    free(g1n);
    free(weights);
    free(grel);
    free(c_norm);
    free(c);
    free(c_abs);
    free(c_best);
    free(gfit);
}
