/* csolve — точка входа: табличный режим solve на C. */
#include "csolve.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int time_enabled;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void phase(const char *name, double t0)
{
    if (time_enabled)
        fprintf(stderr, "[time] %-24s %.3f s\n", name, now_sec() - t0);
}

/* ---------------- вспомогательное ---------------- */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        cs_fatal("не удалось открыть файл: %s", path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0)
        cs_fatal("не удалось прочитать файл: %s", path);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

static void channel_name(const channel_t *ch, char *buf, size_t n)
{
    snprintf(buf, n, "%s(%g nm)", ch->kind == CH_BETA ? "beta" : "alpha",
             ch->wl_um * 1e3);
}

static int cmp_int(const void *pa, const void *pb)
{
    int a = *(const int *)pa, b = *(const int *)pb;
    return (a > b) - (a < b);
}

/* ---------------- JSON-вход ---------------- */
static void load_json_input(const char *path, double *mre, double *mim,
                            channel_t *ch, int *nch, double *g1)
{
    char *text = read_file(path);
    jval_t *root = json_parse(text);
    free(text);
    if (root->t != JOBJ)
        cs_fatal("входной JSON должен быть объектом");
    jval_t *m = json_get(root, "m");
    if (m && m->t == JOBJ) {
        jval_t *re = json_get(m, "real");
        jval_t *im = json_get(m, "imag");
        if (re && re->t == JNUM)
            *mre = re->num;
        if (im && im->t == JNUM)
            *mim = im->num;
    }
    const char *keys[2] = {"backscatter", "extinction"};
    ch_kind_t kinds[2] = {CH_BETA, CH_ALPHA};
    int n = 0;
    for (int k = 0; k < 2; k++) {
        jval_t *grp = json_get(root, keys[k]);
        if (!grp || grp->t != JOBJ)
            continue;
        jval_t *wl = json_get(grp, "wavelengths_nm");
        jval_t *co = json_get(grp, "coefficients");
        if (!wl || !co || wl->t != JARR || co->t != JARR)
            continue;
        int cnt = wl->n < co->n ? wl->n : co->n;
        for (int i = 0; i < cnt; i++) {
            if (wl->items[i]->t != JNUM || co->items[i]->t != JNUM)
                continue;
            if (n >= 32)
                cs_fatal("слишком много каналов во входных данных");
            ch[n].kind = kinds[k];
            ch[n].wl_um = wl->items[i]->num * 1e-3;
            ch[n].tab = NULL;
            ch[n].ref_wl_um = TABLE_REF_WL_UM;
            ch[n].cback_scale = 1.0;
            g1[n] = co->items[i]->num;
            n++;
        }
    }
    json_free(root);
    if (n == 0)
        cs_fatal("в данных нет ни одного канала");
    *nch = n;
}

/* ---------------- демо ---------------- */
static void demo_m_target(double *tre, double *tim, const args_t *a)
{
    *tre = a->has_m_real ? a->m_real : 1.45;
    *tim = a->has_m_imag ? a->m_imag : -0.005;
}

static int m_index(const double *mre, const double *mim, int nm,
                   double re, double im)
{
    for (int i = 0; i < nm; i++)
        if (mre[i] == re && mim[i] == im)
            return i;
    return -1;
}

static void resolve_demo_m(double *tre, double *tim, const args_t *a,
                           const double *mre, const double *mim, int nm)
{
    int *cand = int_array(nm);
    int nc = 0;
    for (int i = 0; i < nm; i++) {
        int ok = 1;
        if (a->has_m_real) {
            char want[64], have[64];
            snprintf(want, sizeof(want), "%.2f", a->m_real);
            snprintf(have, sizeof(have), "%.2f", mre[i]);
            if (strcmp(want, have) != 0)
                ok = 0;
        }
        if (ok && a->has_m_imag) {
            char want[64], have[64];
            snprintf(want, sizeof(want), "%.5f", a->m_imag);
            snprintf(have, sizeof(have), "%.5f", mim[i]);
            if (strcmp(want, have) != 0)
                ok = 0;
        }
        if (ok)
            cand[nc++] = i;
    }
    if (nc == 0) {
        char *list = xstrdup("");
        for (int i = 0; i < nm; i++) {
            char *part = xasprintf("%g%+gj", mre[i], mim[i]);
            char *tmp = xasprintf("%s%s%s", list, i ? ", " : "", part);
            free(list);
            free(part);
            list = tmp;
        }
        cs_fatal("истинный m демо = %g%+gj отсутствует в табличной сетке m: %s "
                 "— задайте --m-real/--m-imag из этого списка",
                 *tre, *tim, list);
    }
    int best = cand[0];
    for (int k = 1; k < nc; k++) {
        int i = cand[k];
        double d1 = fabs(mre[i] - *tre) + fabs(mim[i] - *tim);
        double d2 = fabs(mre[best] - *tre) + fabs(mim[best] - *tim);
        if (d1 < d2)
            best = i;
    }
    if (mre[best] != *tre || mim[best] != *tim) {
        printf("[табличный режим] истинный m демо: используется "
               "%g%+gj (ближайший к %g%+gj из таблиц)\n",
               mre[best], mim[best], *tre, *tim);
    }
    *tre = mre[best];
    *tim = mim[best];
    free(cand);
}

/* ---------------- прямой расчёт (демо) ---------------- */
static void forward_measurements(const channel_t *ch, int nch,
                                 const ktable_t *kt, double (*nf)(double),
                                 double r_min, double r_max, int order,
                                 double span, double *out)
{
    (void)ch;
    double breaks[2] = {r_min, r_max};
    double *nodes = NULL, *weights = NULL;
    int nn = 0;
    quadrature_grid(breaks, 2, order, span, &nodes, &weights, &nn);
    for (int j = 0; j < nch; j++) {
        double s = 0.0;
        for (int k = 0; k < nn; k++) {
            double r = nodes[k];
            double q = 3.0 / (4.0 * CS_PI * r * r * r * r);
            s += weights[k] * ktable_kernel(kt, j, r) * q * nf(r);
        }
        out[j] = s;
    }
    free(nodes);
    free(weights);
}

static uint64_t rng_state;
static uint64_t rng_next(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

static double rng_uniform(void)
{
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(void)
{
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 <= 0.0)
        u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * CS_PI * u2);
}



int main(int argc, char **argv)
{
    time_enabled = getenv("CSOLVE_TIME") != NULL;
    double t_all = now_sec();
    args_t a;
    parse_args(argc, argv, &a);

    if (!a.tables)
        cs_fatal("C-версия поддерживает только табличный режим — "
                 "задайте --tables DIR (Mie через miepython не переносился)");
    if (a.nr_out < 2)
        cs_fatal("--nr-out должен быть >= 2");
    if (a.nr0 < 1 || a.nr1 < 1)
        cs_fatal("--nr0 и --nr1 должны быть >= 1");
    if (!(a.r0_min > 0.0))
        cs_fatal("--r0-min должен быть > 0");
    if (!(a.dx_max > 0.0))
        cs_fatal("--dx-max должен быть > 0");
    if (!(a.log_span > 0.0))
        cs_fatal("--log-span должен быть > 0");
    if (a.quad_order < 1)
        cs_fatal("--quad-order должен быть >= 1");
    if (a.best < 1)
        a.best = 1;   /* в Python G = max(1, min(best, n)) */
    if (a.gamma_rel < 0.0)
        cs_fatal("--gamma-rel должен быть >= 0");
    if (a.ngamma < 1)
        cs_fatal("--ngamma должен быть >= 1");

    qsort(a.degrees, (size_t)a.n_degrees, sizeof(int), cmp_int);
    int nd = 0;
    for (int i = 0; i < a.n_degrees; i++) {
        if (a.degrees[i] < 1 || a.degrees[i] > 4)
            cs_fatal("степень сплайна должна быть в диапазоне 1..4");
        if (i == 0 || a.degrees[i] != a.degrees[i - 1])
            a.degrees[nd++] = a.degrees[i];
    }
    a.n_degrees = nd;

    if (a.lidar_ratio_w < 0.0)
        cs_fatal("--lidar-ratio-w должен быть >= 0");
    if (a.lidar_ratio_iter < 1)
        cs_fatal("--lidar-ratio-iter должен быть >= 1");
    if (a.lidar_ratio_cons && !a.batch)
        printf("[предупреждение] --lidar-ratio-cons действует только в "
               "пакетном режиме (--batch); игнорируется\n");

    if (a.batch) {
        run_batch(&a);
        args_free(&a);
        phase("всего", t_all);
        return 0;
    }

    channel_t base_ch[32];
    double g1_store[32];
    int nch = 0;
    int demo = a.data == NULL;
    double true_re = 1.45, true_im = -0.005;

    if (!demo) {
        load_json_input(a.data, &true_re, &true_im, base_ch, &nch, g1_store);
        printf("данные: %s; каналов: %d\n", a.data, nch);
    } else {
        demo_m_target(&true_re, &true_im, &a);
        double bwl[3] = {355.0, 532.0, 1064.0};
        double awl[2] = {355.0, 532.0};
        for (int i = 0; i < 3; i++) {
            base_ch[nch].kind = CH_BETA;
            base_ch[nch].wl_um = bwl[i] * 1e-3;
            base_ch[nch].tab = NULL;
            base_ch[nch].ref_wl_um = TABLE_REF_WL_UM;
            base_ch[nch].cback_scale = 1.0;
            nch++;
        }
        for (int i = 0; i < 2; i++) {
            base_ch[nch].kind = CH_ALPHA;
            base_ch[nch].wl_um = awl[i] * 1e-3;
            base_ch[nch].tab = NULL;
            base_ch[nch].ref_wl_um = TABLE_REF_WL_UM;
            base_ch[nch].cback_scale = 1.0;
            nch++;
        }
        printf("демонстрация на синтетических данных\n");
    }

    char name[64];
    for (int j = 0; j < nch; j++) {
        channel_name(&base_ch[j], name, sizeof(name));
        printf("  канал: %s\n", name);
    }
    printf("входные единицы: β [1/(м·ср)], α [1/м]\n");
    printf("граничные условия n(r0)=n(r1)=0 (каждое элементарное решение): %s\n",
           a.no_zero_boundary ? "выключено (--no-zero-boundary)" : "включено");
    if (a.no_smooth)
        printf("сглаживание среднего по лучшим решениям: выключено (--no-smooth)\n");
    else
        printf("сглаживание среднего по лучшим решениям: включено (rel-lam=%g)\n",
               a.smooth_lam);
    int nb = a.n_basis > 0 ? a.n_basis : nch;
    if (!a.no_zero_boundary)
        printf("базисных функций на окно: %d (с ГУ неизвестных: %d)\n",
               nb, nb - 2);
    else
        printf("базисных функций на окно: %d\n", nb);

    /* ---- сетка m из таблиц ---- */
    double *mre = NULL, *mim = NULL;
    int nm = 0;
    table_m_grid(a.tables, a.m_reals, a.n_m_reals, a.m_imags, a.n_m_imags,
                 &mre, &mim, &nm);
    if (nm == 0)
        cs_fatal("сетка показателя преломления пуста");

    if (demo)
        resolve_demo_m(&true_re, &true_im, &a, mre, mim, nm);

    printf("показатель преломления m (исходный/истинный) = %g %+gj\n",
           true_re, true_im);
    printf("сетка показателя преломления (перебор):\n");
    for (int i = 0; i < nm; i++)
        printf("  m = %.3f%+.3fj\n", mre[i], mim[i]);

    /* ---- покрытие таблиц и построение ядер ---- */
    double cback_scale = a.table_cback_4pi ? 1.0 / (4.0 * CS_PI) : 1.0;
    printf("строим таблицы ядер (%d значений m)...\n", nm);
    double capmin = INFINITY;
    for (int i = 0; i < nm; i++) {
        channels_t c;
        load_table_channels(a.tables, mre[i], mim[i], base_ch, nch,
                            a.table_extrapolate, a.table_ref_wl,
                            cback_scale, &c);
        if (c.cap < capmin)
            capmin = c.cap;
        channels_free(&c);
    }
    double r1_max_eff = a.r1_max < capmin ? a.r1_max : capmin;
    if (r1_max_eff < a.r1_min)
        cs_fatal("покрытие таблиц по x ограничивает r1_max = %g мкм < "
                 "r1_min = %g — расширьте таблицы или уменьшите --r1-min",
                 r1_max_eff, a.r1_min);

    printf("табличный режим ядер (внешние сечения), корень: %s\n", a.tables);
    printf("  λ_ref таблиц = %g мкм (сечения пересчитываются как (λ/λ_ref)²)\n",
           a.table_ref_wl);
    printf("  конвенция Cback = %s\n", a.table_cback_4pi ? "4pi" : "dsigma");
    {
        channels_t c0;
        load_table_channels(a.tables, mre[0], mim[0], base_ch, nch,
                            a.table_extrapolate, a.table_ref_wl,
                            cback_scale, &c0);
        for (int j = 0; j < nch; j++) {
            channel_name(&c0.ch[j], name, sizeof(name));
            printf("  %-14s x_max = %g, r_max = %.3g мкм\n", name,
                   c0.ch[j].tab->x_max,
                   c0.ch[j].tab->x_max * c0.ch[j].wl_um / (2.0 * CS_PI));
        }
        channels_free(&c0);
    }
    printf("  эффективный r1_max = %g мкм (задан: %g) — окна и сетка r "
           "ограничены\n", r1_max_eff, a.r1_max);

    ktable_t *kts = xcalloc((size_t)nm, sizeof(ktable_t));
    double t_phase = now_sec();
    for (int i = 0; i < nm; i++) {
        channels_t c;
        load_table_channels(a.tables, mre[i], mim[i], base_ch, nch,
                            a.table_extrapolate, a.table_ref_wl,
                            cback_scale, &c);
        ktable_build(&kts[i], &c, a.r0_min, r1_max_eff, a.dx_max);
        channels_free(&c);
    }
    printf("  таблица: %d точек по r на каждое m\n", kts[0].n);
    phase("таблицы ядер", t_phase);

    /* ---- измерения ---- */
    double *g1 = dbl_array(nch);
    double (*n_true_fn)(double) = NULL;
    if (demo) {
        int idx = m_index(mre, mim, nm, true_re, true_im);
        ktable_t tmp;
        const ktable_t *tk;
        if (idx >= 0) {
            tk = &kts[idx];
        } else {
            printf("  строим таблицу для истинного m демо: m = %g%+gj\n",
                   true_re, true_im);
            channels_t c;
            load_table_channels(a.tables, true_re, true_im, base_ch, nch,
                                a.table_extrapolate, a.table_ref_wl,
                                cback_scale, &c);
            ktable_build(&tmp, &c, a.r0_min, r1_max_eff, a.dx_max);
            channels_free(&c);
            tk = &tmp;
        }
        forward_measurements(base_ch, nch, tk, demo_true_distribution,
                             a.r0_min, r1_max_eff, 24, 0.05, g1);
        if (a.noise > 0.0) {
            rng_state = a.seed ? (uint64_t)a.seed * 6364136223846793005ULL +
                                     1442695040888963407ULL
                               : 88172645463325252ULL;
            for (int j = 0; j < nch; j++)
                g1[j] *= 1.0 + a.noise * rng_normal();
        }
        printf("измеренные коэффициенты g1: [");
        for (int j = 0; j < nch; j++)
            printf("%s%.6g", j ? " " : "", g1[j]);
        printf("]\n");
        if (idx < 0)
            ktable_free(&tmp);
        n_true_fn = demo_true_distribution;
    } else {
        memcpy(g1, g1_store, sizeof(double) * (size_t)nch);
    }

    /* ---- перебор ---- */
    double *r0_grid = geomspace(a.r0_min, a.r0_max, a.nr0);
    double *r1_grid = geomspace(a.r1_min, r1_max_eff, a.nr1);

    printf("перебор (m, r0, r1, степень)...\n");
    design_t design;
    t_phase = now_sec();
    design_build(&design, base_ch, nch, kts, nm, mre, mim,
                 a.degrees, a.n_degrees, r0_grid, a.nr0, r1_grid, a.nr1,
                 a.min_ratio, a.n_basis, !a.no_zero_boundary,
                 a.quad_order, a.log_span);
    phase("матрицы A", t_phase);

    solutions_t sols;
    t_phase = now_sec();
    scan_variants(&design, g1, nch, a.gamma_select, a.gamma_rel,
                  a.gamma_min, a.gamma_max, a.ngamma, a.nonneg,
                  !a.absolute, &sols);
    phase("перебор (m, окно)", t_phase);
    if (sols.n == 0)
        cs_fatal("нет ни одного допустимого окна [r0, r1] — проверьте "
                 "границы и min-ratio");

    profile_t prof;
    t_phase = now_sec();
    profile_finalize(&prof, &a, base_ch, nch, kts, mre, mim, nm, &sols,
                     design.geom, a.r0_min, r1_max_eff);
    phase("ансамбль и сглаживание", t_phase);

    report_summary(&a, &sols, g1, base_ch, nch, prof.n_avg, prof.g_fit,
                   prof.r_grid, prof.nr, prof.m_re, prof.m_im);
    report_moments(&prof.moments);

    const char *mode_name = a.modes == 1 ? "fit" : a.modes == 2 ? "stat" : "auto";
    report_modes(&prof.modes, a.r_split, mode_name);

    if (a.json) {
        write_json(a.json, &sols, prof.r_grid, prof.nr, prof.n_avg,
                   prof.n_avg_raw, base_ch, nch, g1, prof.g_fit,
                   prof.m_re, prof.m_im, &prof.moments, &prof.modes);
        printf("результаты сохранены: %s\n", a.json);
    }

    if (!a.no_plot) {
        double *n_true_vals = NULL;
        if (n_true_fn) {
            n_true_vals = dbl_array(prof.nr);
            for (int i = 0; i < prof.nr; i++)
                n_true_vals[i] = n_true_fn(prof.r_grid[i]);
        }
        plot_results(a.out, prof.r_grid, prof.nr, n_true_vals, prof.n_avg,
                     prof.n_avg_raw, prof.n_best, prof.n_best_count,
                     base_ch, nch, g1, prof.g_fit, prof.m_re, prof.m_im);
        free(n_true_vals);
    }

    /* ---- уборка ---- */
    profile_free(&prof);
    for (int i = 0; i < nm; i++)
        ktable_free(&kts[i]);
    free(kts);
    free(mre);
    free(mim);
    free(r0_grid);
    free(r1_grid);
    free(g1);
    design_free(&design);
    solutions_free(&sols);
    args_free(&a);
    phase("всего", t_all);
    return 0;
}
