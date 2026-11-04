/* Кубический сплайн с краевыми условиями not-a-knot.
 *
 * Строится по вторым производным M_i в узлах; интерполяция и экстраполяция
 * совпадают с scipy.interpolate.CubicSpline (bc_type='not-a-knot'), который
 * используется в Python-версии. Для n = 3 функция вырождается в параболическую
 * интерполяцию через три точки (как в scipy), для n = 2 — в линейную.
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

static void fill_coeffs(spline_t *s, const double *M)
{
    int n = s->n;
    for (int i = 0; i < n - 1; i++) {
        double h = s->x[i + 1] - s->x[i];
        double delta = (s->y[i + 1] - s->y[i]) / h;
        s->c[i] = 0.5 * M[i];
        s->b[i] = delta - (2.0 * M[i] + M[i + 1]) * h / 6.0;
        s->d[i] = (M[i + 1] - M[i]) / (6.0 * h);
    }
}

void spline_init(spline_t *s, const double *x, const double *y, int n)
{
    memset(s, 0, sizeof(*s));
    s->n = n;
    s->x = dbl_array(n);
    s->y = dbl_array(n);
    memcpy(s->x, x, sizeof(double) * (size_t)n);
    memcpy(s->y, y, sizeof(double) * (size_t)n);
    if (n <= 1)
        return;
    s->b = dbl_array(n - 1);
    s->c = dbl_array(n - 1);
    s->d = dbl_array(n - 1);

    if (n == 2) {
        double h = x[1] - x[0];
        s->b[0] = (y[1] - y[0]) / h;
        return;
    }

    double *M = dbl_array(n);
    if (n == 3) {
        double h0 = x[1] - x[0], h1 = x[2] - x[1];
        double d0 = (y[1] - y[0]) / h0, d1 = (y[2] - y[1]) / h1;
        double m = 2.0 * (d1 - d0) / (h0 + h1);
        M[0] = M[1] = M[2] = m;
    } else {
        /* Неизвестные M_1 .. M_{n-2}; после исключения M_0 и M_{n-1}
         * получается трёхдиагональная система (n-2) x (n-2). */
        int m = n - 2;
        double *a = dbl_array(m);
        double *b = dbl_array(m);
        double *c = dbl_array(m);
        double *r = dbl_array(m);
        double *h = dbl_array(n - 1);
        double *delta = dbl_array(n - 1);
        for (int i = 0; i < n - 1; i++) {
            h[i] = x[i + 1] - x[i];
            delta[i] = (y[i + 1] - y[i]) / h[i];
        }
        /* k = 0 — узел 1 (первая краевая строка исключена) */
        b[0] = (h[0] + h[1]) * (h[0] + 2.0 * h[1]);
        c[0] = h[1] * h[1] - h[0] * h[0];
        r[0] = 6.0 * h[1] * (delta[1] - delta[0]);
        /* внутренние строки, узел i = k+1, i = 2 .. n-3 */
        for (int k = 1; k <= m - 2; k++) {
            int i = k + 1;
            a[k] = h[i - 1];
            b[k] = 2.0 * (h[i - 1] + h[i]);
            c[k] = h[i];
            r[k] = 6.0 * (delta[i] - delta[i - 1]);
        }
        /* k = m-1 — узел n-2 (последняя краевая строка исключена) */
        if (m - 1 > 0) {
            int k = m - 1;
            a[k] = h[n - 3] * h[n - 3] - h[n - 2] * h[n - 2];
            b[k] = (h[n - 3] + h[n - 2]) * (2.0 * h[n - 3] + h[n - 2]);
            r[k] = 6.0 * h[n - 3] * (delta[n - 2] - delta[n - 3]);
        }
        /* прямая прогонка (матрица строго диагонально доминирует) */
        for (int k = 1; k < m; k++) {
            double w = a[k] / b[k - 1];
            b[k] -= w * c[k - 1];
            r[k] -= w * r[k - 1];
        }
        double *sol = dbl_array(m);
        sol[m - 1] = r[m - 1] / b[m - 1];
        for (int k = m - 2; k >= 0; k--)
            sol[k] = (r[k] - c[k] * sol[k + 1]) / b[k];
        for (int k = 0; k < m; k++)
            M[k + 1] = sol[k];
        /* восстановление M_0 и M_{n-1} из краевых условий */
        M[0] = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
        M[n - 1] = (-h[n - 2] * M[n - 3] +
                    (h[n - 2] + h[n - 3]) * M[n - 2]) / h[n - 3];
        free(a); free(b); free(c); free(r); free(h); free(delta); free(sol);
    }
    fill_coeffs(s, M);
    free(M);
}

double spline_eval(const spline_t *s, double t)
{
    int n = s->n;
    if (n <= 0)
        return NAN;
    if (n == 1)
        return s->y[0];
    int lo;
    if (t <= s->x[0]) {
        lo = 0;
    } else if (t >= s->x[n - 1]) {
        lo = n - 2;
    } else {
        int hi = n - 1;
        lo = 0;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (t < s->x[mid])
                hi = mid;
            else
                lo = mid;
        }
    }
    double dx = t - s->x[lo];
    return s->y[lo] + dx * (s->b[lo] + dx * (s->c[lo] + dx * s->d[lo]));
}

double spline_eval_hint(const spline_t *s, double t, int *hint)
{
    int n = s->n;
    if (n <= 0)
        return NAN;
    if (n == 1)
        return s->y[0];
    int lo = *hint;
    if (lo < 0)
        lo = 0;
    if (lo > n - 2)
        lo = n - 2;
    if (t <= s->x[0]) {
        lo = 0;
    } else if (t >= s->x[n - 1]) {
        lo = n - 2;
    } else {
        while (lo > 0 && t < s->x[lo])
            lo--;
        while (lo < n - 2 && t >= s->x[lo + 1])
            lo++;
    }
    *hint = lo;
    double dx = t - s->x[lo];
    return s->y[lo] + dx * (s->b[lo] + dx * (s->c[lo] + dx * s->d[lo]));
}

void spline_free(spline_t *s)
{
    if (!s)
        return;
    free(s->x);
    free(s->y);
    free(s->b);
    free(s->c);
    free(s->d);
    memset(s, 0, sizeof(*s));
}
