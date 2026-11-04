/* Матрицы A для всех (m, окно) за один проход.
 *
 * A[j,i] = int_{r0}^{r1} K_j(r) q(r) B_i(r) dr, q(r) = 3/(4 pi r^4),
 * где K_j — табличное ядро, B_i — базисные B-сплайны окна.
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

void design_build(design_t *d, const channel_t *ch, int nch,
                  const ktable_t *kts, int n_m,
                  const double *m_re, const double *m_im,
                  const int *degrees, int n_degrees,
                  const double *r0_grid, int nr0,
                  const double *r1_grid, int nr1,
                  double min_ratio, int n_basis_in, int zero_boundary,
                  int order, double max_log_span)
{
    (void)ch;
    memset(d, 0, sizeof(*d));
    d->nch = nch;
    d->n_basis = n_basis_in > 0 ? n_basis_in : nch;
    d->zero_boundary = zero_boundary;
    d->order = order;
    d->max_log_span = max_log_span;

    int cap = n_degrees * nr0 * nr1;
    int *deg = int_array(cap > 0 ? cap : 1);
    double *r0s = dbl_array(cap > 0 ? cap : 1);
    double *r1s = dbl_array(cap > 0 ? cap : 1);
    int nw = 0;
    for (int a = 0; a < n_degrees; a++) {
        for (int b = 0; b < nr0; b++) {
            for (int c = 0; c < nr1; c++) {
                if (r1_grid[c] < min_ratio * r0_grid[b])
                    continue;
                deg[nw] = degrees[a];
                r0s[nw] = r0_grid[b];
                r1s[nw] = r1_grid[c];
                nw++;
            }
        }
    }
    d->n_w = nw;
    d->n_m = n_m;
    d->degree = deg;
    d->r0 = r0s;
    d->r1 = r1s;
    d->m_re = dbl_array(n_m);
    d->m_im = dbl_array(n_m);
    memcpy(d->m_re, m_re, sizeof(double) * (size_t)n_m);
    memcpy(d->m_im, m_im, sizeof(double) * (size_t)n_m);

    d->geom = xcalloc((size_t)(nw > 0 ? nw : 1), sizeof(geom_t));
    for (int w = 0; w < nw; w++)
        geom_build(&d->geom[w], r0s[w], r1s[w], deg[w], d->n_basis,
                   zero_boundary, order, max_log_span);

    int nu = nw > 0 ? d->geom[0].n_unknown : 0;
    d->nu = nu;
    size_t total = (size_t)n_m * (size_t)nw * (size_t)nch *
                   (size_t)(nu > 0 ? nu : 1);
    d->A = xcalloc(total, sizeof(double));
    d->s0 = xcalloc((size_t)(n_m > 0 ? n_m : 1) * (size_t)(nw > 0 ? nw : 1),
                    sizeof(double));

    for (int im = 0; im < n_m; im++) {
        for (int w = 0; w < nw; w++) {
            const geom_t *g = &d->geom[w];
            int nn = g->n_nodes;
            int nunk = g->n_unknown;
            double *Ab = d->A + (((size_t)im * nw + w) * nch *
                                 (size_t)(nunk > 0 ? nunk : 1));
            /* ln r для узлов окна считается один раз на все каналы */
            double *logr = dbl_array(nn > 0 ? nn : 1);
            for (int q = 0; q < nn; q++) {
                double rc = g->nodes[q];
                if (rc < kts[im].r_min)
                    rc = kts[im].r_min;
                if (rc > kts[im].r_max)
                    rc = kts[im].r_max;
                logr[q] = log(rc);
            }
            for (int j = 0; j < nch; j++) {
                int hint = -1;
                double *row = Ab + (size_t)j * nunk;
                for (int q = 0; q < nn; q++) {
                    double k = exp(spline_eval_hint(&kts[im].sp[j], logr[q],
                                                    &hint));
                    double wgt = k * g->dvol[q] * g->weights[q];
                    const double *bv = g->values + (size_t)q * nunk;
                    for (int i = 0; i < nunk; i++)
                        row[i] += wgt * bv[i];
                }
            }
            d->s0[(size_t)im * nw + w] = svd_norm2(Ab, nch, nunk);
            free(logr);
        }
    }
}

void design_free(design_t *d)
{
    if (!d)
        return;
    for (int w = 0; w < d->n_w; w++)
        geom_free(&d->geom[w]);
    free(d->geom);
    free(d->degree);
    free(d->r0);
    free(d->r1);
    free(d->m_re);
    free(d->m_im);
    free(d->A);
    free(d->s0);
    memset(d, 0, sizeof(*d));
}
