/* Юнит-тесты csolve: сплайн, квадратуры, B-сплайны, Холецкого, NNLS,
 * разбор таблиц и сглаживание (эталоны получены из Python/scipy). */
#include "csolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check_close(const char *what, double got, double want, double tol)
{
    double d = fabs(got - want);
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    if (d > tol * scale) {
        printf("FAIL %-38s got %.17g want %.17g (d=%.3g)\n", what, got, want, d);
        failures++;
    } else {
        printf("ok   %s\n", what);
    }
}

static void check(const char *what, int cond)
{
    if (!cond) {
        printf("FAIL %s\n", what);
        failures++;
    } else {
        printf("ok   %s\n", what);
    }
}

static void test_spline(void)
{
    double x[5] = {0.1, 0.5, 1.0, 2.0, 3.0};
    double y[5] = {1.0, 2.0, 4.0, 16.0, 9.0};
    spline_t s;
    spline_init(&s, x, y, 5);
    double pts[5] = {0.2, 0.7, 1.5, 2.5, 3.5};
    double want[5] = {1.3600000000000003, 2.46, 9.8125, 17.4375, -14.4375};
    for (int i = 0; i < 5; i++) {
        char name[64];
        snprintf(name, sizeof(name), "spline not-a-knot t=%.1f", pts[i]);
        check_close(name, spline_eval(&s, pts[i]), want[i], 1e-12);
    }
    spline_free(&s);

    double x3[3] = {0.1, 0.5, 1.0};
    double y3[3] = {1.0, 2.0, 4.0};
    spline_init(&s, x3, y3, 3);
    double p3[3] = {0.2, 0.7, 1.5};
    double w3[3] = {1.2, 2.6999999999999997, 6.833333333333333};
    for (int i = 0; i < 3; i++) {
        char name[64];
        snprintf(name, sizeof(name), "spline n=3 t=%.1f", p3[i]);
        check_close(name, spline_eval(&s, p3[i]), w3[i], 1e-12);
    }
    spline_free(&s);
}

static void test_gl(void)
{
    int order = 24;
    double *x = dbl_array(order), *w = dbl_array(order);
    gl_nodes_weights(order, x, w);
    double s8 = 0.0, s2 = 0.0;
    for (int i = 0; i < order; i++) {
        s8 += w[i] * pow(x[i], 8);
        s2 += w[i] * x[i] * x[i];
    }
    check_close("Гаусс-Лежандр: int x^8 = 2/9", s8, 2.0 / 9.0, 1e-14);
    check_close("Гаусс-Лежандр: int x^2 = 2/3", s2, 2.0 / 3.0, 1e-14);
    /* узлы по возрастанию */
    int asc = 1;
    for (int i = 1; i < order; i++)
        if (!(x[i] > x[i - 1]))
            asc = 0;
    check("Гаусс-Лежандр: узлы возрастают", asc);
    free(x);
    free(w);
}

static void test_quadrature(void)
{
    double breaks[2] = {1.0, 2.0};
    double *nodes = NULL, *weights = NULL;
    int n = 0;
    quadrature_grid(breaks, 2, 24, 0.1, &nodes, &weights, &n);
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += weights[i] / nodes[i];
    check_close("квадратура: int_1^2 dx/x = ln2", s, log(2.0), 1e-12);
    free(nodes);
    free(weights);
}

static void test_bspline(void)
{
    geom_t g;
    geom_build(&g, 0.1, 10.0, 3, 6, 0, 24, 0.1);
    double *full = dbl_array(6);
    int ok = 1;
    for (int k = 0; k < 25; k++) {
        double t = 0.1 * pow(100.0, (double)k / 24.0);
        bspline_values(g.knots, g.n_knots, g.degree, t, full);
        double s = 0.0;
        for (int i = 0; i < 6; i++) {
            s += full[i];
            if (full[i] < -1e-14)
                ok = 0;
        }
        if (fabs(s - 1.0) > 1e-12)
            ok = 0;
    }
    check("B-сплайны: разбиение единицы и неотрицательность", ok);
    /* в узлах квадратуры матрица values тоже даёт разбиение единицы */
    int ok2 = 1;
    for (int q = 0; q < g.n_nodes; q++) {
        double s = 0.0;
        for (int i = 0; i < g.n_unknown; i++)
            s += g.values[(size_t)q * g.n_unknown + i];
        if (fabs(s - 1.0) > 1e-12)
            ok2 = 0;
    }
    check("B-сплайны: разбиение единицы в узлах квадратуры", ok2);
    free(full);
    geom_free(&g);
}

static void test_band_chol(void)
{
    const int n = 4, kb = 1, bw = 2 * kb + 1;
    double M[4][4] = {{4, 1, 0, 0}, {1, 4, 1, 0},
                      {0, 1, 4, 1}, {0, 0, 1, 4}};
    double xt[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4], band[4 * 3];
    memset(band, 0, sizeof(band));
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++) {
            b[i] += M[i][j] * xt[j];
            if (abs(i - j) <= kb)
                band[i * bw + (j - i + kb)] = M[i][j];
        }
    }
    double x[4];
    band_chol_solve(band, n, kb, b, x);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (fabs(x[i] - xt[i]) > 1e-12)
            ok = 0;
    check("ленточная Холецкого: решение SPD-системы", ok);
}

static void test_nnls(void)
{
    double a[6] = {1, 0, 0, 1, 1, 1};   /* 3 x 2, row-major */
    double b[3] = {1.0, -1.0, 0.0};
    double x[2];
    nnls(a, 3, 2, b, x);
    check_close("NNLS: x[1] = 0 (граница)", x[1], 0.0, 1e-10);
    check_close("NNLS: x[0] = 0.5", x[0], 0.5, 1e-10);
}

static void test_dirname(void)
{
    char *d1 = m_table_dirname(1.45, -0.005);
    check("имя каталога m (обычное)",
          strcmp(d1, "mre_1.45_mim_-0.00500") == 0);
    free(d1);
    char *d2 = m_table_dirname(1.45, -0.0);
    check("имя каталога m (отрицательный ноль)",
          strcmp(d2, "mre_1.45_mim_0.00000") == 0);
    free(d2);
    double re = 0, im = 0;
    check("разбор имени каталога", parse_m_dirname("mre_1.45_mim_-0.00500", &re, &im));
    check_close("разбор имени: Re", re, 1.45, 1e-15);
    check_close("разбор имени: Im", im, -0.005, 1e-15);
    check("разбор имени: постороннее имя",
          !parse_m_dirname("average", &re, &im));
}

static void test_table_parse(void)
{
    const char *path = "/tmp/csolve_unit_table.txt";
    FILE *f = fopen(path, "w");
    if (!f) {
        check("таблица: создать временный файл", 0);
        return;
    }
    fputs("# Units: x (dimless), C (um^2)\n"
          "# x Cback Cext Qext n\n"
          "0.1 7.0 1.0 0.5 5\n"
          "0.5 8.0 2.0 0.6 5\n"
          "1.0 9.0 4.0 0.7 5\n"
          "2.0 10.0 16.0 0.8 5\n", f);
    fclose(f);
    table_t *t = table_load(path, 0);
    check("таблица: обе колонки найдены", t->has_cext && t->has_cback);
    check_close("таблица: x_min", t->x_min, 0.1, 1e-15);
    check_close("таблица: x_max", t->x_max, 2.0, 1e-15);
    check_close("таблица: Cext в узле", table_cross_section(t, 0, 0.5), 2.0, 1e-12);
    check_close("таблица: Cback в узле", table_cross_section(t, 1, 1.0), 9.0, 1e-12);
    check_close("таблица: интерполяция внутри",
                table_cross_section(t, 0, 0.25), 1.5, 0.2);
    table_free(t);
    remove(path);
}

static void test_smoothing(void)
{
    double r[8] = {0.05, 0.08469069900482264, 0.14345028995850928,
                   0.24297810658061286, 0.41155971378360795,
                   0.6971055968511699, 1.1807672055499936, 2.0};
    double y[8] = {1.0, 2.0, 1.5, 3.0, 2.2, 2.8, 1.7, 2.5};
    double w[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    double want[8] = {1.0000605683247177, 1.7972784813389382,
                      1.8136974164873019, 2.684401088600474,
                      2.465905496820976, 2.5324755824187664,
                      1.9235917468040047, 2.4999220818628123};
    double n_avg[8];
    memcpy(n_avg, y, sizeof(y));
    smooth_distribution(r, 8, n_avg, w, 1e-4);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabs(n_avg[i] - want[i]) > 1e-9 * fabs(want[i]) + 1e-12)
            ok = 0;
    check("сглаживание: совпадение с scipy make_smoothing_spline", ok);
    if (!ok)
        for (int i = 0; i < 8; i++)
            printf("   [%d] got %.17g want %.17g\n", i, n_avg[i], want[i]);
}

static void test_smoothing_lam1(void)
{
    /* опорный профиль лидарного отношения: lam=1.0, единичные веса */
    double x[7] = {0.0, 1.0 / 6.0, 2.0 / 6.0, 0.5, 4.0 / 6.0, 5.0 / 6.0, 1.0};
    double y[7] = {30.0, 30.4, 29.7, 30.2, 30.9, 30.1, 30.5};
    double want[7] = {30.032290318114548, 30.106986571862308,
                      30.181759424004692, 30.257218107418254,
                      30.332666884010926, 30.407299445676983,
                      30.481779248902395};
    double out[7];
    smoothing_spline_fit(x, y, NULL, 7, 1.0, x, 7, out);
    int ok = 1;
    for (int i = 0; i < 7; i++)
        if (fabs(out[i] - want[i]) > 1e-9 * fabs(want[i]) + 1e-12)
            ok = 0;
    check("сглаживание lam=1.0 без весов (опорный профиль S)", ok);
    if (!ok)
        for (int i = 0; i < 7; i++)
            printf("   [%d] got %.17g want %.17g\n", i, out[i], want[i]);
}

int main(void)
{
    printf("== csolve unit tests ==\n");
    test_spline();
    test_gl();
    test_quadrature();
    test_bspline();
    test_band_chol();
    test_nnls();
    test_dirname();
    test_table_parse();
    test_smoothing();
    test_smoothing_lam1();
    printf("\n%s (%d ошибок)\n", failures ? "ПРОВАЛ" : "все тесты пройдены",
           failures);
    return failures ? 1 : 0;
}
