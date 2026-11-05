/* Интегральные характеристики распределения и параметры логнормальных мод. */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

static double per_cm3(double v)
{
    return v / CM3_PER_M3;
}

static double q_dvol(double r)
{
    return 3.0 / (4.0 * CS_PI * r * r * r * r);
}

double lognormal_volume(double r, double V, double rg, double sigma)
{
    double s = log(sigma);
    double z = log(r / rg);
    return V / (sqrt(2.0 * CS_PI) * s) * exp(-(z * z) / (2.0 * s * s));
}

double demo_true_distribution(double r)
{
    return lognormal_volume(r, 4.0e7, 0.15, 1.5) +
           lognormal_volume(r, 6.0e7, 1.2, 1.7);
}

moments_t size_moments(const double *n_avg, const double *r_grid, int n)
{
    moments_t m;
    memset(&m, 0, sizeof(m));
    double *f = dbl_array(n);
    double *lnr = dbl_array(n);
    double *fr = dbl_array(n);
    double *nq = dbl_array(n);
    int cov = 0;
    double rlo = NAN, rhi = NAN;
    for (int i = 0; i < n; i++) {
        lnr[i] = log(r_grid[i]);
        int ok = isfinite(n_avg[i]);
        if (ok) {
            double v = n_avg[i] > 0.0 ? n_avg[i] : 0.0;
            f[i] = v;
            fr[i] = v / r_grid[i];
            nq[i] = v * q_dvol(r_grid[i]);
            cov++;
            if (cov == 1 || r_grid[i] < rlo)
                rlo = r_grid[i];
            if (cov == 1 || r_grid[i] > rhi)
                rhi = r_grid[i];
        }
    }
    double v_conc = trapz(f, lnr, n);
    double m_ln1 = trapz(fr, lnr, n);
    double n_num = trapz(nq, r_grid, n);

    m.vol_conc_um3_per_cm3 = per_cm3(v_conc);
    m.N_cm3 = per_cm3(n_num);
    m.vol_mean_um3 = n_num > 0.0 ? v_conc / n_num : NAN;
    m.r_eff_um = m_ln1 > 0.0 ? v_conc / m_ln1 : NAN;
    m.r_cov_min_um = rlo;
    m.r_cov_max_um = rhi;
    m.cov_frac = n > 0 ? (double)cov / (double)n : 0.0;
    free(f);
    free(lnr);
    free(fr);
    free(nq);
    return m;
}

/* ---------------- логнормальные моды ---------------- */
static cs_mode_t fit_lognormal_ln(const double *x, const double *y,
                               const double *w, int n)
{
    cs_mode_t out;
    out.V = NAN;
    out.rm = NAN;
    out.sigma = NAN;
    out.ok = 0;
    out.method = "";
    if (n < 3)
        return out;
    double *a = xcalloc((size_t)n * 3, sizeof(double));
    double *b = dbl_array(n);
    for (int i = 0; i < n; i++) {
        double sw = 1.0;
        if (w) {
            if (w[i] <= 0.0)
                continue;
            sw = sqrt(w[i]);
        }
        a[(size_t)i * 3 + 0] = sw;
        a[(size_t)i * 3 + 1] = sw * x[i];
        a[(size_t)i * 3 + 2] = sw * x[i] * x[i];
        b[i] = sw * y[i];
    }
    double coef[3];
    lstsq(a, n, 3, b, coef);
    free(a);
    free(b);
    double ca = coef[0], cb = coef[1], cc = coef[2];
    if (cc >= 0.0 || !isfinite(cc))
        return out;
    double mu = -cb / (2.0 * cc);
    double s2 = -1.0 / (2.0 * cc);
    double s = sqrt(s2);
    double v = exp(ca - cb * cb / (4.0 * cc)) * sqrt(2.0 * CS_PI) * s;
    if (!(v > 0.0) || !isfinite(v) || !isfinite(mu))
        return out;
    out.V = v;
    out.rm = exp(mu);
    out.sigma = exp(s);
    out.ok = 1;
    out.method = "fit";
    return out;
}

static cs_mode_t analyze(const double *r, const double *f, const double *w,
                      int n, int method)
{
    cs_mode_t out;
    out.V = NAN;
    out.rm = NAN;
    out.sigma = NAN;
    out.ok = 0;
    out.method = "insufficient points";
    if (n < 3)
        return out;
    double *fr = dbl_array(n);
    double *lnr = dbl_array(n);
    for (int i = 0; i < n; i++) {
        fr[i] = f[i] > 0.0 ? f[i] : 0.0;
        lnr[i] = log(r[i]);
    }
    double v = trapz(fr, lnr, n);
    free(fr);
    free(lnr);
    if (!(v > 0.0) || !isfinite(v)) {
        out.method = "no positive volume";
        return out;
    }
    out.V = v;
    int *pos = int_array(n);
    int np = 0;
    for (int i = 0; i < n; i++)
        if (f[i] > 0.0)
            pos[np++] = i;
    if (np < 3) {
        out.method = "few points with F > 0";
        return out;
    }
    if (method != 2) {
        double *x = dbl_array(np);
        double *y = dbl_array(np);
        double *wsub = w ? dbl_array(np) : NULL;
        for (int k = 0; k < np; k++) {
            x[k] = log(r[pos[k]]);
            y[k] = log(f[pos[k]]);
            if (wsub)
                wsub[k] = w[pos[k]];
        }
        cs_mode_t fit = fit_lognormal_ln(x, y, wsub, np);
        free(x);
        free(y);
        free(wsub);
        if (fit.ok) {
            free(pos);
            out.rm = fit.rm;
            out.sigma = fit.sigma;
            out.ok = 1;
            out.method = "fit";
            return out;
        }
    }
    /* моментная оценка по объёмным весам F */
    double *xf = dbl_array(np);
    double *yf = dbl_array(np);
    for (int k = 0; k < np; k++) {
        xf[k] = log(r[pos[k]]);
        yf[k] = f[pos[k]];
    }
    double wsum = 0.0, mu = 0.0;
    for (int k = 0; k < np; k++)
        wsum += yf[k];
    for (int k = 0; k < np; k++)
        mu += yf[k] * xf[k];
    mu /= wsum;
    double s2 = 0.0;
    for (int k = 0; k < np; k++) {
        double d = xf[k] - mu;
        s2 += yf[k] * d * d;
    }
    s2 /= wsum;
    if (!(s2 > 0.0) || !isfinite(s2))
        s2 = 0.0;
    out.rm = exp(mu);
    out.sigma = exp(sqrt(s2));
    out.ok = 1;
    out.method = "moments";
    free(xf);
    free(yf);
    free(pos);
    return out;
}

static cs_mode_t analyze_filtered(const double *r_grid, const double *f,
                               const double *w, int n, double lo, double hi,
                               int method)
{
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (r_grid[i] >= lo && r_grid[i] <= hi)
            cnt++;
    double *r = dbl_array(cnt);
    double *fv = dbl_array(cnt);
    double *wv = w ? dbl_array(cnt) : NULL;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (r_grid[i] < lo || r_grid[i] > hi)
            continue;
        r[k] = r_grid[i];
        fv[k] = f[i];
        if (wv)
            wv[k] = w[i];
        k++;
    }
    cs_mode_t m = analyze(r, fv, wv, cnt, method);
    free(r);
    free(fv);
    free(wv);
    return m;
}

modes_t mode_analysis(const double *r_grid, const double *f_avg, int n,
                      double r_split, const double *w, int method)
{
    modes_t out;
    memset(&out, 0, sizeof(out));
    double *r = dbl_array(n);
    double *f = dbl_array(n);
    double *wv = w ? dbl_array(n) : NULL;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(f_avg[i]))
            continue;
        r[cnt] = r_grid[i];
        f[cnt] = f_avg[i];
        if (wv)
            wv[cnt] = w[i];
        cnt++;
    }
    if (cnt < 4) {
        cs_mode_t bad;
        memset(&bad, 0, sizeof(bad));
        bad.V = bad.rm = bad.sigma = NAN;
        bad.ok = 0;
        bad.method = "insufficient points";
        out.fine = out.coarse = out.overall = bad;
        free(r);
        free(f);
        free(wv);
        return out;
    }
    double r0 = r[0], r1 = r[0];
    for (int i = 1; i < cnt; i++) {
        if (r[i] < r0)
            r0 = r[i];
        if (r[i] > r1)
            r1 = r[i];
    }
    out.fine = analyze_filtered(r, f, wv, cnt, r0, r_split, method);
    out.coarse = analyze_filtered(r, f, wv, cnt, r_split, r1, method);
    out.overall = analyze_filtered(r, f, wv, cnt, r0, r1, method);
    free(r);
    free(f);
    free(wv);
    return out;
}
