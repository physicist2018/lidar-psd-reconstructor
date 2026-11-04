/* Линейная алгебра: SVD (односторонний Якоби), МНК с отсечкой как в numpy,
 * ленточная Холецкого и NNLS (Lawson-Hanson).
 *
 * Матрицы в этом проекте маленькие (число каналов + неизвестных на окно,
 * либо n x 3 для фита логнормали), поэтому односторонний Якоби быстрее и
 * устойчивее, чем попытка воспроизвести LAPACK dgesdd.
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

void svd_compute(svd_t *out, const double *a, int m, int n)
{
    memset(out, 0, sizeof(*out));
    out->m = m;
    out->n = n;
    if (m <= 0 || n <= 0) {
        out->s = dbl_array(1);
        return;
    }
    int rows = m > n ? m : n;
    double *w = xcalloc((size_t)rows * (size_t)n, sizeof(double));
    double *v = xcalloc((size_t)n * (size_t)n, sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            w[(size_t)i * n + j] = a[(size_t)i * n + j];
    for (int i = 0; i < n; i++)
        v[(size_t)i * n + i] = 1.0;

    const int max_sweeps = 60;
    const double tol = 1e-16;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        int off = 0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < rows; i++) {
                    double wp = w[(size_t)i * n + p];
                    double wq = w[(size_t)i * n + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if (alpha == 0.0 || beta == 0.0)
                    continue;
                if (fabs(gamma) <= tol * sqrt(alpha * beta))
                    continue;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                           (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = c * t;
                for (int i = 0; i < rows; i++) {
                    double wp = w[(size_t)i * n + p];
                    double wq = w[(size_t)i * n + q];
                    w[(size_t)i * n + p] = c * wp - s * wq;
                    w[(size_t)i * n + q] = s * wp + c * wq;
                }
                for (int i = 0; i < n; i++) {
                    double vp = v[(size_t)i * n + p];
                    double vq = v[(size_t)i * n + q];
                    v[(size_t)i * n + p] = c * vp - s * vq;
                    v[(size_t)i * n + q] = s * vp + c * vq;
                }
                off++;
            }
        }
        if (off == 0)
            break;
    }

    out->w = w;
    out->v = v;
    out->s = dbl_array(n);
    for (int k = 0; k < n; k++) {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            double x = w[(size_t)i * n + k];
            sum += x * x;
        }
        out->s[k] = sqrt(sum);
    }
}

void svd_free(svd_t *s)
{
    if (!s)
        return;
    free(s->w);
    free(s->s);
    free(s->v);
    memset(s, 0, sizeof(*s));
}

double svd_norm2(const double *a, int m, int n)
{
    svd_t s;
    svd_compute(&s, a, m, n);
    double mx = 0.0;
    for (int k = 0; k < n; k++)
        if (s.s[k] > mx)
            mx = s.s[k];
    svd_free(&s);
    return mx;
}

void lstsq(const double *a, int m, int n, const double *b, double *x)
{
    svd_t sv;
    svd_compute(&sv, a, m, n);
    double smax = 0.0;
    for (int k = 0; k < n; k++)
        if (sv.s[k] > smax)
            smax = sv.s[k];
    double cutoff = CS_EPS * (double)(m > n ? m : n) * smax;
    double *coef = dbl_array(n);
    for (int k = 0; k < n; k++) {
        if (!(sv.s[k] > cutoff) || sv.s[k] == 0.0) {
            coef[k] = 0.0;
            continue;
        }
        double dot = 0.0;
        for (int i = 0; i < m; i++)
            dot += sv.w[(size_t)i * n + k] * b[i];
        coef[k] = dot / (sv.s[k] * sv.s[k]);
    }
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
            sum += sv.v[(size_t)i * n + k] * coef[k];
        x[i] = sum;
    }
    free(coef);
    svd_free(&sv);
}

void band_chol_solve(double *band, int n, int kb, const double *b, double *x)
{
    int w = 2 * kb + 1;
    for (int i = 0; i < n; i++) {
        int j0 = i - kb > 0 ? i - kb : 0;
        for (int j = j0; j <= i; j++) {
            double s = band[(size_t)i * w + (j - i + kb)];
            int k0 = i - kb > 0 ? i - kb : 0;
            if (j - kb > k0)
                k0 = j - kb;
            for (int k = k0; k < j; k++)
                s -= band[(size_t)i * w + (k - i + kb)] *
                     band[(size_t)j * w + (k - j + kb)];
            if (j == i) {
                if (!(s > 0.0))
                    s = 1e-300;   /* страховка от потери положительности */
                band[(size_t)i * w + kb] = sqrt(s);
            } else {
                band[(size_t)i * w + (j - i + kb)] =
                    s / band[(size_t)j * w + kb];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        double s = b[i];
        int k0 = i - kb > 0 ? i - kb : 0;
        for (int k = k0; k < i; k++)
            s -= band[(size_t)i * w + (k - i + kb)] * x[k];
        x[i] = s / band[(size_t)i * w + kb];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = x[i];
        for (int k = i + 1; k < n && k <= i + kb; k++)
            s -= band[(size_t)k * w + (i - k + kb)] * x[k];
        x[i] = s / band[(size_t)i * w + kb];
    }
}

static void lstsq_subset(const double *a, int m, int n, const int *cols,
                         int np, const double *b, double *coef)
{
    double *sub = xcalloc((size_t)m * (size_t)(np > 0 ? np : 1), sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < np; j++)
            sub[(size_t)i * np + j] = a[(size_t)i * n + cols[j]];
    lstsq(sub, m, np, b, coef);
    free(sub);
}

void nnls(const double *a, int m, int n, const double *b, double *x)
{
    for (int i = 0; i < n; i++)
        x[i] = 0.0;
    if (n <= 0)
        return;
    int *passive = int_array(n);
    int *cols = int_array(n);
    double *w = dbl_array(n);
    double *z = dbl_array(n);
    double *res = dbl_array(m);

    /* начальная норма градиента — для относительного допуска */
    double g0 = 0.0;
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int i = 0; i < m; i++)
            s += a[(size_t)i * n + j] * b[i];
        if (fabs(s) > g0)
            g0 = fabs(s);
    }
    if (g0 == 0.0)
        g0 = 1.0;
    const double tol = 1e-12 * g0;
    const int max_iter = 3 * n + 10;

    for (int iter = 0; iter < max_iter; iter++) {
        /* градиент A^T (b - A x) */
        for (int i = 0; i < m; i++) {
            double s = b[i];
            for (int j = 0; j < n; j++)
                s -= a[(size_t)i * n + j] * x[j];
            res[i] = s;
        }
        int best = -1;
        double bestv = tol;
        for (int j = 0; j < n; j++) {
            if (passive[j])
                continue;
            double s = 0.0;
            for (int i = 0; i < m; i++)
                s += a[(size_t)i * n + j] * res[i];
            w[j] = s;
            if (s > bestv) {
                bestv = s;
                best = j;
            }
        }
        if (best < 0)
            break;
        passive[best] = 1;

        for (int inner = 0; inner < max_iter; inner++) {
            int np = 0;
            for (int j = 0; j < n; j++)
                if (passive[j])
                    cols[np++] = j;
            lstsq_subset(a, m, n, cols, np, b, z);
            int allpos = 1;
            double alpha = 1.0;
            for (int k = 0; k < np; k++) {
                int j = cols[k];
                if (z[k] <= 0.0) {
                    allpos = 0;
                    double denom = x[j] - z[k];
                    if (denom > 0.0) {
                        double r = x[j] / denom;
                        if (r < alpha)
                            alpha = r;
                    } else {
                        alpha = 0.0;
                    }
                }
            }
            if (allpos) {
                for (int k = 0; k < np; k++)
                    x[cols[k]] = z[k];
                break;
            }
            for (int k = 0; k < np; k++) {
                int j = cols[k];
                x[j] += alpha * (z[k] - x[j]);
            }
            for (int j = 0; j < n; j++)
                if (passive[j] && x[j] <= 1e-14)
                    passive[j] = 0;
        }
    }
    free(passive);
    free(cols);
    free(w);
    free(z);
    free(res);
}
