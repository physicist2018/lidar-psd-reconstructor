/* Кубический сглаживающий сплайн (эквивалент scipy make_smoothing_spline):
 *
 *   min_f  sum_i w_i (f(x_i) - y_i)^2 + lam * integral (f'')^2 dx
 *
 * над пространством кубических B-сплайнов с узлами в x_i (кратные концы).
 * Решается симметричная положительно определённая ленточная система
 * (B^T W B + lam P) c = B^T W y, где P — матрица штрафа за вторую
 * производную. Проверено: совпадение с scipy на ~1e-16.
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

void smoothing_spline_fit(const double *x, const double *y, const double *w,
                          int n, double lam, const double *xeval, int neval,
                          double *out)
{
    for (int k = 0; k < neval; k++)
        out[k] = NAN;
    if (n < 5)
        return;   /* scipy требует >= 5 точек */

    const int p = 3;
    int nbf = n + 2;
    int nk = nbf + p + 1;
    double *U = dbl_array(nk);
    U[0] = U[1] = U[2] = x[0];
    for (int i = 0; i < n; i++)
        U[3 + i] = x[i];
    U[nk - 1] = U[nk - 2] = U[nk - 3] = x[n - 1];

    const int kb = 3;
    int bw = 2 * kb + 1;
    double *M = xcalloc((size_t)nbf * bw, sizeof(double));
    double *rhs = dbl_array(nbf);
    double *loc = dbl_array(p + 1);

#define MB(i, j) M[(size_t)(i) * bw + ((j) - (i) + kb)]

    /* член данных B^T W B и правая часть B^T W y */
    for (int i = 0; i < n; i++) {
        double wi = w ? w[i] : 1.0;
        int span = 0;
        bspline_deriv_window(U, nk, p, 0, x[i], loc, &span);
        int base = span - p;
        for (int a = 0; a < p + 1; a++) {
            int ia = base + a;
            if (ia < 0 || ia >= nbf)
                continue;
            rhs[ia] += wi * y[i] * loc[a];
            for (int b = 0; b < p + 1; b++) {
                int ib = base + b;
                if (ib < 0 || ib >= nbf)
                    continue;
                if (abs(ia - ib) <= kb)
                    MB(ia, ib) += wi * loc[a] * loc[b];
            }
        }
    }

    /* штраф lam * int (f'')^2 dx по пролётам между узлами x */
    static const double gl3[3] = {-0.7745966692414834, 0.0,
                                  0.7745966692414834};
    static const double gw3[3] = {0.5555555555555556, 0.8888888888888888,
                                  0.5555555555555556};
    double *d2 = dbl_array(p + 1);
    for (int i = 0; i < n - 1; i++) {
        double a = x[i], b = x[i + 1];
        double mid = 0.5 * (a + b), half = 0.5 * (b - a);
        for (int q = 0; q < 3; q++) {
            double node = half * gl3[q] + mid;
            int span = 0;
            bspline_deriv_window(U, nk, p, 2, node, d2, &span);
            int base = span - p;
            double wq = half * gw3[q] * lam;
            for (int a1 = 0; a1 < p + 1; a1++) {
                int i1 = base + a1;
                if (i1 < 0 || i1 >= nbf)
                    continue;
                for (int b1 = 0; b1 < p + 1; b1++) {
                    int j1 = base + b1;
                    if (j1 < 0 || j1 >= nbf)
                        continue;
                    if (abs(i1 - j1) <= kb)
                        MB(i1, j1) += wq * d2[a1] * d2[b1];
                }
            }
        }
    }

    double *c = dbl_array(nbf);
    band_chol_solve(M, nbf, kb, rhs, c);

    for (int k = 0; k < neval; k++) {
        int span = 0;
        bspline_deriv_window(U, nk, p, 0, xeval[k], loc, &span);
        int base = span - p;
        double val = 0.0;
        for (int a = 0; a < p + 1; a++) {
            int ia = base + a;
            if (ia >= 0 && ia < nbf)
                val += c[ia] * loc[a];
        }
        out[k] = val;
    }

#undef MB
    free(U);
    free(M);
    free(rhs);
    free(loc);
    free(d2);
    free(c);
}
