/* B-сплайны на логарифмически эквидистантной сетке узлов, квадратуры
 * Гаусса-Лежандра и геометрия окна [r0, r1].
 */
#include "csolve.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- Гаусс-Лежандр ---------------- */
void gl_nodes_weights(int order, double *x, double *w)
{
    int n = order;
    if (n <= 0)
        return;
    for (int i = 0; i < n; i++) {
        double z = cos(CS_PI * ((double)i + 0.75) / ((double)n + 0.5));
        double pp = 1.0;
        for (int it = 0; it < 100; it++) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= n; j++) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / (double)j;
            }
            pp = (double)n * (z * p1 - p2) / (z * z - 1.0);
            double z1 = z;
            z = z1 - p1 / pp;
            if (fabs(z - z1) <= 1e-16)
                break;
        }
        /* пересчёт производной в сошедшейся точке */
        double p1 = 1.0, p2 = 0.0;
        for (int j = 1; j <= n; j++) {
            double p3 = p2;
            p2 = p1;
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / (double)j;
        }
        pp = (double)n * (z * p1 - p2) / (z * z - 1.0);
        /* узлы по возрастанию (как np.polynomial.legendre.leggauss) */
        x[n - 1 - i] = z;
        w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}

/* ---------------- B-сплайны ---------------- */
static int find_span(const double *U, int nbf, int p, double u)
{
    int n = nbf - 1;
    if (u <= U[p])
        return p;
    if (u >= U[n + 1])
        return n;
    int low = p, high = n + 1;
    int mid = (low + high) / 2;
    while (u < U[mid] || u >= U[mid + 1]) {
        if (u < U[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

void bspline_values(const double *U, int nk, int p, double t, double *out)
{
    int nbf = nk - p - 1;
    if (nbf <= 0)
        return;
    memset(out, 0, sizeof(double) * (size_t)nbf);
    /* на правом конце переходим к пределу слева (полуоткрытое соглашение) */
    double u = t;
    if (u >= U[nbf])
        u = nextafter(U[nbf], -INFINITY);
    if (u < U[p])
        u = U[p];

    int span = find_span(U, nbf, p, u);
    double N[16], left[16], right[16];
    if (p + 1 > 16)
        cs_fatal("B-spline degree too large");
    N[0] = 1.0;
    for (int j = 1; j <= p; j++) {
        left[j] = u - U[span + 1 - j];
        right[j] = U[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; r++) {
            double denom = right[r + 1] + left[j - r];
            double temp = denom != 0.0 ? N[r] / denom : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
    for (int j = 0; j <= p; j++) {
        int idx = span - p + j;
        if (idx >= 0 && idx < nbf)
            out[idx] = N[j];
    }
}

void bspline_deriv_values(const double *U, int nk, int p, int deriv, double t,
                          double *out)
{
    int nbf = nk - p - 1;
    if (nbf <= 0)
        return;
    if (deriv == 0) {
        bspline_values(U, nk, p, t, out);
        return;
    }
    if (deriv > p) {
        memset(out, 0, sizeof(double) * (size_t)nbf);
        return;
    }
    /* база: производные степени p-1 */
    double *prev = dbl_array(nbf + 1);
    bspline_deriv_values(U, nk, p - 1, deriv - 1, t, prev);
    for (int i = 0; i < nbf; i++) {
        double d1 = 0.0, d2 = 0.0;
        double den1 = U[i + p] - U[i];
        double den2 = U[i + p + 1] - U[i + 1];
        if (den1 > 0.0)
            d1 = prev[i] / den1;
        if (den2 > 0.0)
            d2 = prev[i + 1] / den2;
        out[i] = (double)p * (d1 - d2);
    }
    free(prev);
}

/* --- локальные значения (только p+1 ненулевых) --- */
static void local_basis(const double *U, int nk, int q, double u, int i0,
                        int m, double *out)
{
    for (int k = 0; k < m; k++)
        out[k] = 0.0;
    int nbfq = nk - q - 1;
    double uu = u;
    if (uu >= U[nbfq])
        uu = nextafter(U[nbfq], -INFINITY);
    if (uu < U[q])
        uu = U[q];
    int sq = find_span(U, nbfq, q, uu);
    double N[16], left[16], right[16];
    N[0] = 1.0;
    for (int j = 1; j <= q; j++) {
        left[j] = uu - U[sq + 1 - j];
        right[j] = U[sq + j] - uu;
        double saved = 0.0;
        for (int r = 0; r < j; r++) {
            double denom = right[r + 1] + left[j - r];
            double temp = denom != 0.0 ? N[r] / denom : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
    for (int j = 0; j <= q; j++) {
        int idx = sq - q + j;
        if (idx >= i0 && idx < i0 + m)
            out[idx - i0] = N[j];
    }
}

static void local_deriv(const double *U, int nk, int p, int deriv, double u,
                        int i0, int m, double *out)
{
    if (deriv <= 0) {
        local_basis(U, nk, p, u, i0, m, out);
        return;
    }
    if (deriv > p) {
        for (int k = 0; k < m; k++)
            out[k] = 0.0;
        return;
    }
    double *v = dbl_array(m + 1);
    local_deriv(U, nk, p - 1, deriv - 1, u, i0, m + 1, v);
    for (int k = 0; k < m; k++) {
        int i = i0 + k;
        double den1 = U[i + p] - U[i];
        double den2 = U[i + p + 1] - U[i + 1];
        double d1 = den1 > 0.0 ? v[k] / den1 : 0.0;
        double d2 = den2 > 0.0 ? v[k + 1] / den2 : 0.0;
        out[k] = (double)p * (d1 - d2);
    }
    free(v);
}

void bspline_deriv_window(const double *U, int nk, int p, int deriv, double t,
                          double *out, int *span_out)
{
    int nbf = nk - p - 1;
    double u = t;
    if (u >= U[nbf])
        u = nextafter(U[nbf], -INFINITY);
    if (u < U[p])
        u = U[p];
    int s = find_span(U, nbf, p, u);
    local_deriv(U, nk, p, deriv, u, s - p, p + 1, out);
    if (span_out)
        *span_out = s;
}

/* ---------------- составная квадратура ---------------- */
typedef struct {
    double *v;
    int n, cap;
} vec_t;

static void vec_push(vec_t *d, double x)
{
    if (d->n == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 512;
        d->v = xrealloc(d->v, sizeof(double) * (size_t)d->cap);
    }
    d->v[d->n++] = x;
}

void quadrature_grid(const double *breaks, int nb, int order,
                     double max_log_span, double **nodes, double **weights,
                     int *out_n)
{
    vec_t nx = {0}, nw = {0};
    double *gx = dbl_array(order);
    double *gw = dbl_array(order);
    gl_nodes_weights(order, gx, gw);
    for (int i = 0; i < nb - 1; i++) {
        double a = breaks[i], b = breaks[i + 1];
        if (!(b > a))
            continue;
        int n_seg = (int)ceil(log(b / a) / max_log_span);
        if (n_seg < 1)
            n_seg = 1;
        double la = log(a), lb = log(b);
        double *bounds = linspace(la, lb, n_seg + 1);
        for (int k = 0; k < n_seg; k++) {
            double aa = exp(bounds[k]), bb = exp(bounds[k + 1]);
            double mid = 0.5 * (bb + aa);
            double half = 0.5 * (bb - aa);
            for (int q = 0; q < order; q++) {
                vec_push(&nx, half * gx[q] + mid);
                vec_push(&nw, half * gw[q]);
            }
        }
        free(bounds);
    }
    free(gx);
    free(gw);
    *nodes = nx.v;
    *weights = nw.v;
    *out_n = nx.n;
}

/* ---------------- геометрия окна ---------------- */
void geom_build(geom_t *g, double r0, double r1, int degree, int n_basis,
                int interior_only, int order, double max_log_span)
{
    memset(g, 0, sizeof(*g));
    int p_internal = n_basis - degree - 1;
    if (p_internal < 0)
        cs_fatal("expected n_basis >= degree + 1");
    if (interior_only && n_basis < 3)
        cs_fatal("boundary conditions n(r0)=n(r1)=0 require n_basis >= 3");

    int nk = n_basis + degree + 1;
    double *knots = dbl_array(nk);
    for (int i = 0; i <= degree; i++) {
        knots[i] = r0;
        knots[nk - 1 - i] = r1;
    }
    if (p_internal > 0) {
        double *inter = geomspace(r0, r1, p_internal + 2);
        for (int i = 0; i < p_internal; i++)
            knots[degree + 1 + i] = inter[i + 1];
        free(inter);
    }

    /* уникальные узлы (breaks) */
    double *breaks = dbl_array(nk);
    int nb = 0;
    for (int i = 0; i < nk; i++) {
        if (nb == 0 || knots[i] != breaks[nb - 1])
            breaks[nb++] = knots[i];
    }

    double *nodes = NULL, *weights = NULL;
    int nn = 0;
    quadrature_grid(breaks, nb, order, max_log_span, &nodes, &weights, &nn);

    int nu = n_basis - (interior_only ? 2 : 0);
    double *values = xcalloc((size_t)nn * (size_t)(nu > 0 ? nu : 1),
                             sizeof(double));
    double *full = dbl_array(n_basis);
    for (int q = 0; q < nn; q++) {
        bspline_values(knots, nk, degree, nodes[q], full);
        for (int i = 0; i < nu; i++)
            values[(size_t)q * nu + i] =
                full[interior_only ? i + 1 : i];
    }
    free(full);
    double *dvol = dbl_array(nn);
    for (int q = 0; q < nn; q++) {
        double r = nodes[q];
        dvol[q] = 3.0 / (4.0 * CS_PI * r * r * r * r);
    }

    g->degree = degree;
    g->n_basis = n_basis;
    g->n_unknown = nu;
    g->interior_only = interior_only;
    g->r0 = r0;
    g->r1 = r1;
    g->n_knots = nk;
    g->knots = knots;
    g->n_nodes = nn;
    g->nodes = nodes;
    g->weights = weights;
    g->values = values;
    g->dvol = dvol;
    free(breaks);
}

void geom_free(geom_t *g)
{
    if (!g)
        return;
    free(g->knots);
    free(g->nodes);
    free(g->weights);
    free(g->values);
    free(g->dvol);
    memset(g, 0, sizeof(*g));
}
