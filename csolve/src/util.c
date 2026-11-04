#include "csolve.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void cs_fatal(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("ошибка: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        cs_fatal("не хватает памяти (%zu байт)", n);
    return p;
}

void *xcalloc(size_t n, size_t size)
{
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p)
        cs_fatal("не хватает памяти (%zu x %zu байт)", n, size);
    return p;
}

void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q)
        cs_fatal("не хватает памяти (%zu байт)", n);
    return q;
}

char *xstrdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *p = xmalloc(n);
    memcpy(p, s, n);
    return p;
}

char *xasprintf(const char *fmt, ...)
{
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
        cs_fatal("ошибка форматирования строки");
    char *p = xmalloc((size_t)n + 1);
    vsnprintf(p, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return p;
}

double *dbl_array(int n)
{
    return xcalloc((size_t)(n > 0 ? n : 1), sizeof(double));
}

int *int_array(int n)
{
    return xcalloc((size_t)(n > 0 ? n : 1), sizeof(int));
}

double *linspace(double a, double b, int n)
{
    double *out = dbl_array(n);
    if (n <= 0)
        return out;
    if (n == 1) {
        out[0] = a;
        return out;
    }
    double step = (b - a) / (double)(n - 1);
    for (int i = 0; i < n; i++)
        out[i] = a + step * (double)i;
    out[n - 1] = b;
    return out;
}

double *geomspace(double a, double b, int n)
{
    double *out = dbl_array(n);
    if (n <= 0)
        return out;
    if (n == 1) {
        out[0] = a;
        return out;
    }
    double la = log(a), lb = log(b);
    double step = (lb - la) / (double)(n - 1);
    for (int i = 0; i < n; i++)
        out[i] = exp(la + step * (double)i);
    out[n - 1] = exp(lb);
    return out;
}

int dbl_eq(double a, double b)
{
    return a == b;
}

double trapz(const double *y, const double *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n - 1; i++)
        sum += 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1]);
    return sum;
}

double interp_clamped(double x, const double *xs, const double *ys, int n)
{
    if (n <= 0)
        return NAN;
    if (x <= xs[0])
        return ys[0];
    if (x >= xs[n - 1])
        return ys[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x < xs[mid])
            hi = mid;
        else
            lo = mid;
    }
    double dx = xs[hi] - xs[lo];
    double w = dx > 0.0 ? (x - xs[lo]) / dx : 0.0;
    return ys[lo] + w * (ys[hi] - ys[lo]);
}
