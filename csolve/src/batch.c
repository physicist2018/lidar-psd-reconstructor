/* Пакетная обработка (--batch): несколько высот из одного txt.
 *
 * Формат входа:
 *   Altitude  b355  b532  b1064 a355 a532
 *   100       9.87e-06 ...
 * Имена каналов: b<нм> — обратное рассеяние β, a<нм> — ослабление α.
 *
 * Таблица ядер и матрицы A строятся один раз на весь пакет; далее каждая
 * высота решается независимо. При --lidar-ratio-cons решения всех высот
 * удерживаются и переранжируются по близости S = α/β к сглаженному по
 * высотам опорному профилю.
 */
#define _POSIX_C_SOURCE 200809L

#include "csolve.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* разбор txt                                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    double *v;
    int n, cap;
} dvec_t;

/* Сводные метрики по высотам для итоговых графиков. */
typedef struct {
    double *err;
    double *V, *Vc, *Vf;
    double *rmf, *rmc, *reff;
    double *Nc;
    double *sgf, *sgc, *sgo;
    double *m_re, *m_im;
    double *S;      /* nl x n_alt (по длинам волн) */
    int n_alt, nl;
} metrics_t;

static void dvec_push(dvec_t *d, double x)
{
    if (d->n == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 64;
        d->v = xrealloc(d->v, sizeof(double) * (size_t)d->cap);
    }
    d->v[d->n++] = x;
}

static int split_tokens(char *s, char **toks, int max)
{
    int n = 0;
    char *p = s;
    while (*p) {
        while (*p && isspace((unsigned char)*p))
            p++;
        if (!*p)
            break;
        if (n < max)
            toks[n++] = p;
        while (*p && !isspace((unsigned char)*p))
            p++;
        if (*p)
            *p++ = '\0';
    }
    return n;
}

void parse_batch_txt(const char *path, batch_input_t *b)
{
    memset(b, 0, sizeof(*b));
    FILE *f = fopen(path, "r");
    if (!f)
        cs_fatal("не удалось открыть файл: %s", path);

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int have_header = 0;
    int have_row = 0;
    dvec_t alt = {0}, vals = {0};

    while ((len = getline(&line, &cap, f)) != -1) {
        (void)len;
        char *s = line;
        while (*s && isspace((unsigned char)*s))
            s++;
        if (!*s || *s == '#')
            continue;
        char *end = s + strlen(s);
        while (end > s && isspace((unsigned char)end[-1]))
            *--end = '\0';
        char *toks[64];
        int nt = split_tokens(s, toks, 64);
        if (nt == 0)
            continue;

        if (!have_header) {
            char low[32];
            size_t nl = strlen(toks[0]);
            if (nl >= sizeof(low))
                nl = sizeof(low) - 1;
            for (size_t i = 0; i < nl; i++)
                low[i] = (char)tolower((unsigned char)toks[0][i]);
            low[nl] = '\0';
            if (strcmp(low, "altitude") != 0 && strcmp(low, "alt") != 0 &&
                strcmp(low, "h") != 0)
                cs_fatal("первое поле заголовка должно быть 'Altitude', "
                         "получено '%s'", toks[0]);
            b->nch = nt - 1;
            if (b->nch <= 0)
                cs_fatal("в заголовке нет ни одного канала "
                         "(ожидается b/a<Нм>)");
            b->ch = xcalloc((size_t)b->nch, sizeof(batch_channel_t));
            for (int i = 1; i < nt; i++) {
                const char *nm = toks[i];
                char c0 = (char)tolower((unsigned char)nm[0]);
                int digits = nm[1] != '\0';
                for (const char *p = nm + 1; *p; p++)
                    if (!isdigit((unsigned char)*p))
                        digits = 0;
                if (!digits || (c0 != 'b' && c0 != 'a'))
                    cs_fatal("неизвестное имя канала '%s' "
                             "(ожидается b<нм> или a<нм>)", nm);
                batch_channel_t *ch = &b->ch[i - 1];
                ch->kind = c0 == 'b' ? CH_BETA : CH_ALPHA;
                ch->wl_um = (double)atoi(nm + 1) * 1e-3;
                snprintf(ch->name, sizeof(ch->name), "%s", nm);
            }
            have_header = 1;
            continue;
        }

        if (nt != b->nch + 1)
            cs_fatal("ожидалось %d коэффициентов на строку, получено %d: '%s'",
                     b->nch, nt - 1, s);
        char *e = NULL;
        double hval = strtod(toks[0], &e);
        if (e == toks[0] || *e)
            cs_fatal("не удалось разобрать строку '%s'", s);
        dvec_push(&alt, hval);
        for (int i = 0; i < b->nch; i++) {
            double v = strtod(toks[i + 1], &e);
            if (e == toks[i + 1] || *e)
                cs_fatal("не удалось разобрать строку '%s'", s);
            dvec_push(&vals, v);
        }
        have_row = 1;
    }
    fclose(f);
    free(line);

    if (!have_header)
        cs_fatal("файл %s пуст", path);
    if (!have_row)
        cs_fatal("в файле %s нет строк с высотами", path);
    b->n_alt = alt.n;
    b->alt = alt.v;
    b->data = vals.v;
}

void batch_input_free(batch_input_t *b)
{
    if (!b)
        return;
    free(b->ch);
    free(b->alt);
    free(b->data);
    memset(b, 0, sizeof(*b));
}

/* ------------------------------------------------------------------ */
/* общие длины волн и лидарное отношение                               */
/* ------------------------------------------------------------------ */
typedef struct {
    double wl;
    int ib;   /* индекс канала β */
    int ia;   /* индекс канала α */
} common_t;

static common_t *common_wavelengths(const batch_input_t *b, int *n_out)
{
    double wls[64];
    int ib[64], ia[64];
    int nw = 0;
    for (int j = 0; j < b->nch; j++) {
        double wl = b->ch[j].wl_um;
        int k = -1;
        for (int i = 0; i < nw; i++)
            if (wls[i] == wl) {
                k = i;
                break;
            }
        if (k < 0) {
            k = nw++;
            wls[k] = wl;
            ib[k] = -1;
            ia[k] = -1;
        }
        /* как словарь в Python: для одинаковых wl побеждает последний */
        if (b->ch[j].kind == CH_BETA)
            ib[k] = j;
        else
            ia[k] = j;
    }
    common_t *out = xcalloc((size_t)(nw > 0 ? nw : 1), sizeof(common_t));
    int n = 0;
    for (int i = 0; i < nw; i++)
        if (ib[i] >= 0 && ia[i] >= 0) {
            out[n].wl = wls[i];
            out[n].ib = ib[i];
            out[n].ia = ia[i];
            n++;
        }
    /* сортировка по длине волны */
    for (int i = 1; i < n; i++) {
        common_t key = out[i];
        int j = i - 1;
        while (j >= 0 && out[j].wl > key.wl) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
    *n_out = n;
    return out;
}

static double lidar_ratio_value(const double *g_fit, const common_t *c, int k)
{
    double b = g_fit[c[k].ib];
    double a = g_fit[c[k].ia];
    return (b > 0.0 && a > 0.0) ? a / b : NAN;
}

/* ------------------------------------------------------------------ */
/* медианы и опорный профиль S                                         */
/* ------------------------------------------------------------------ */
static int cmp_double(const void *pa, const void *pb)
{
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

static double nanmedian(const double *v, int n)
{
    double *tmp = dbl_array(n > 0 ? n : 1);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i]))
            tmp[m++] = v[i];
    double out = NAN;
    if (m > 0) {
        qsort(tmp, (size_t)m, sizeof(double), cmp_double);
        out = (m % 2) ? tmp[m / 2] : 0.5 * (tmp[m / 2 - 1] + tmp[m / 2]);
    }
    free(tmp);
    return out;
}

/* скользящая медиана (окно 3, у краёв сужается); S — n_alt x nl */
static void windowed_median_ref(const double *S, int n_alt, int nl,
                                double *out)
{
    double *col = dbl_array(n_alt);
    for (int h = 0; h < n_alt; h++) {
        int lo = h > 0 ? h - 1 : 0;
        int hi = h + 2 < n_alt ? h + 2 : n_alt;
        for (int li = 0; li < nl; li++) {
            int m = 0;
            for (int k = lo; k < hi; k++)
                col[m++] = S[(size_t)k * nl + li];
            out[(size_t)h * nl + li] = nanmedian(col, m);
        }
    }
    free(col);
}

static void windowed_median_column(const double *S, int n_alt, int nl, int li,
                                   double *out)
{
    double *col = dbl_array(n_alt);
    for (int h = 0; h < n_alt; h++) {
        int lo = h > 0 ? h - 1 : 0;
        int hi = h + 2 < n_alt ? h + 2 : n_alt;
        int m = 0;
        for (int k = lo; k < hi; k++)
            col[m++] = S[(size_t)k * nl + li];
        out[(size_t)h * nl + li] = nanmedian(col, m);
    }
    free(col);
}

/* np.maximum(v, floor) с сохранением NaN (как numpy) */
static double np_max_floor(double v, double floor_v)
{
    if (isnan(v))
        return NAN;
    return v > floor_v ? v : floor_v;
}

static void smooth_ref_profile(const double *alt, const double *S, int n_alt,
                               int nl, double *out)
{
    for (int i = 0; i < n_alt * nl; i++)
        out[i] = NAN;
    if (n_alt < 5) {
        windowed_median_ref(S, n_alt, nl, out);
        return;
    }
    double *t = dbl_array(n_alt);
    for (int i = 0; i < n_alt; i++)
        t[i] = alt[i] - alt[0];
    double span = t[n_alt - 1] - t[0];
    if (!(span > 0.0)) {
        free(t);
        windowed_median_ref(S, n_alt, nl, out);
        return;
    }
    for (int i = 0; i < n_alt; i++)
        t[i] /= span;

    int *ok = int_array(n_alt);
    for (int li = 0; li < nl; li++) {
        int cnt = 0;
        for (int h = 0; h < n_alt; h++)
            if (isfinite(S[(size_t)h * nl + li]))
                ok[cnt++] = h;
        if (cnt < 5) {
            windowed_median_column(S, n_alt, nl, li, out);
            continue;
        }
        double scale = 0.0;
        for (int k = 0; k < cnt; k++) {
            double v = fabs(S[(size_t)ok[k] * nl + li]);
            if (v > scale)
                scale = v;
        }
        if (!(scale > 0.0) || !isfinite(scale)) {
            windowed_median_column(S, n_alt, nl, li, out);
            continue;
        }
        double *xv = dbl_array(cnt);
        double *yv = dbl_array(cnt);
        double *fit = dbl_array(cnt);
        for (int k = 0; k < cnt; k++) {
            xv[k] = t[ok[k]];
            yv[k] = S[(size_t)ok[k] * nl + li] / scale;
        }
        smoothing_spline_fit(xv, yv, NULL, cnt, 1.0, xv, cnt, fit);
        for (int k = 0; k < cnt; k++)
            out[(size_t)ok[k] * nl + li] = fit[k] * scale;
        free(xv);
        free(yv);
        free(fit);
    }
    free(t);
    free(ok);
}

/* ------------------------------------------------------------------ */
/* устойчивая сортировка решений по score                              */
/* ------------------------------------------------------------------ */
static void merge_order(const double *score, int *order, int *tmp,
                        int lo, int hi)
{
    if (hi - lo < 2)
        return;
    int mid = (lo + hi) / 2;
    merge_order(score, order, tmp, lo, mid);
    merge_order(score, order, tmp, mid, hi);
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (score[order[j]] < score[order[i]])
            tmp[k++] = order[j++];
        else
            tmp[k++] = order[i++];
    }
    while (i < mid)
        tmp[k++] = order[i++];
    while (j < hi)
        tmp[k++] = order[j++];
    for (int t = lo; t < hi; t++)
        order[t] = tmp[t];
}

static void reorder_by_score(solutions_t *sols, const double *score)
{
    int n = sols->n;
    int *order = int_array(n);
    int *tmp = int_array(n);
    for (int i = 0; i < n; i++)
        order[i] = i;
    merge_order(score, order, tmp, 0, n);
    solution_t *buf = xmalloc(sizeof(solution_t) * (size_t)(n > 0 ? n : 1));
    for (int k = 0; k < n; k++)
        buf[k] = sols->sol[order[k]];
    memcpy(sols->sol, buf, sizeof(solution_t) * (size_t)n);
    free(buf);
    free(order);
    free(tmp);
}

static int top_ids_equal(const solutions_t *sols, int g, const int *prev,
                         int off)
{
    for (int k = 0; k < g; k++)
        if (sols->sol[k].id != prev[off + k])
            return 0;
    return 1;
}

static void store_top_ids(const solutions_t *sols, int g, int *prev, int off)
{
    for (int k = 0; k < g; k++)
        prev[off + k] = sols->sol[k].id;
}

static void apply_lidar_constraint(solutions_t *per, int n_alt,
                                   const double *alt, const common_t *common,
                                   int nl, double w, int max_iter, int best,
                                   const double *const *S)
{
    int dbg = getenv("CSOLVE_LIDAR_DEBUG") != NULL;
    int *g_eff = int_array(n_alt);
    int *k_eff = int_array(n_alt);
    for (int h = 0; h < n_alt; h++) {
        int g = best;
        if (g > per[h].n)
            g = per[h].n;
        if (g < 1)
            g = 1;
        g_eff[h] = g;
        int k = 3 * g_eff[h] > 9 ? 3 * g_eff[h] : 9;
        if (k > per[h].n)
            k = per[h].n;
        k_eff[h] = k;
    }
    int gmax = 1;
    for (int h = 0; h < n_alt; h++)
        if (g_eff[h] > gmax)
            gmax = g_eff[h];

    int *prev = int_array(n_alt * gmax);
    int have_prev = 0;

    int maxpool = n_alt;
    for (int h = 0; h < n_alt; h++)
        if (k_eff[h] > maxpool)
            maxpool = k_eff[h];

    double *est = dbl_array(n_alt * nl);
    double *ref = dbl_array(n_alt * nl);
    double *pool = dbl_array(maxpool > 0 ? maxpool : 1);
    double *log_ref = dbl_array(nl > 0 ? nl : 1);
    int iter = max_iter > 1 ? max_iter : 1;
    (void)common;

    for (int it = 0; it < iter; it++) {
        /* опорный профиль: медиана top-K решений каждой высоты, сглаженная */
        for (int h = 0; h < n_alt; h++) {
            int k = k_eff[h];
            for (int li = 0; li < nl; li++) {
                for (int t = 0; t < k; t++)
                    pool[t] = S[h][(size_t)per[h].sol[t].id * nl + li];
                est[(size_t)h * nl + li] = nanmedian(pool, k);
            }
        }
        smooth_ref_profile(alt, est, n_alt, nl, ref);
        /* остаточные NaN — медиана по столбцу профиля */
        for (int li = 0; li < nl; li++) {
            for (int h = 0; h < n_alt; h++)
                pool[h] = ref[(size_t)h * nl + li];
            double fill = nanmedian(pool, n_alt);
            if (isfinite(fill))
                for (int h = 0; h < n_alt; h++)
                    if (!isfinite(ref[(size_t)h * nl + li]))
                        ref[(size_t)h * nl + li] = fill;
        }
        if (dbg) {
            printf("iter %d: ref = [", it);
            for (int h = 0; h < n_alt; h++) {
                printf("%s[", h ? ", " : "");
                for (int li = 0; li < nl; li++)
                    printf("%s%.15g", li ? ", " : "",
                           ref[(size_t)h * nl + li]);
                printf("]");
            }
            printf("]\n");
        }

        for (int h = 0; h < n_alt; h++) {
            int n = per[h].n;
            double *score = dbl_array(n);
            for (int li = 0; li < nl; li++) {
                double rv = np_max_floor(ref[(size_t)h * nl + li], 1e-300);
                log_ref[li] = log(rv);
            }
            for (int t = 0; t < n; t++) {
                const double *sv = S[h] + (size_t)per[h].sol[t].id * nl;
                double sum = 0.0;
                int cnt = 0;
                for (int li = 0; li < nl; li++) {
                    double lv = log(np_max_floor(sv[li], 1e-300));
                    double d = fabs(lv - log_ref[li]);
                    if (isfinite(d)) {
                        sum += d;
                        cnt++;
                    }
                }
                double dev = cnt > 0 ? sum / (double)cnt : 0.0;
                if (!isfinite(dev))
                    dev = 0.0;
                score[t] = per[h].sol[t].err + w * dev;
            }
            reorder_by_score(&per[h], score);
            if (dbg) {
                for (int t = 0; t < n && t < 3; t++) {
                    const solution_t *s = &per[h].sol[t];
                    const double *sv = S[h] + (size_t)s->id * nl;
                    double sum = 0.0;
                    int cnt = 0;
                    for (int li = 0; li < nl; li++) {
                        double lv = log(np_max_floor(sv[li], 1e-300));
                        double rv = log(np_max_floor(ref[(size_t)h * nl + li],
                                                     1e-300));
                        double d = fabs(lv - rv);
                        if (isfinite(d)) {
                            sum += d;
                            cnt++;
                        }
                    }
                    double dev = cnt > 0 ? sum / (double)cnt : 0.0;
                    printf("  h=%g top%d: (score=%g, err=%g, dev=%g, id=%d, "
                           "m=%g%+gj, deg=%d, r0=%.5g, r1=%.5g) S=[%g, %g]\n",
                           alt[h], t, s->err + w * dev, s->err, dev, s->id,
                           s->m_re, s->m_im, s->degree, s->r0, s->r1,
                           sv[0], nl > 1 ? sv[1] : 0.0);
                }
            }
            free(score);
        }

        int changed = !have_prev;
        int off = 0;
        for (int h = 0; h < n_alt; h++) {
            if (have_prev && !top_ids_equal(&per[h], g_eff[h], prev, off))
                changed = 1;
            off += gmax;
        }
        if (!changed)
            break;
        off = 0;
        for (int h = 0; h < n_alt; h++) {
            store_top_ids(&per[h], g_eff[h], prev, off);
            off += gmax;
        }
        have_prev = 1;
    }

    free(prev);
    free(est);
    free(ref);
    free(pool);
    free(log_ref);
    free(g_eff);
    free(k_eff);
}

/* ------------------------------------------------------------------ */
/* вывод файлов                                                        */
/* ------------------------------------------------------------------ */
static void mkdir_p(const char *path)
{
    char *tmp = xstrdup(path);
    size_t n = strlen(tmp);
    for (size_t i = 1; i < n; i++) {
        if (tmp[i] != '/')
            continue;
        tmp[i] = '\0';
        if (mkdir(tmp, 0777) != 0 && errno != EEXIST)
            cs_fatal("не удалось создать каталог %s: %s", tmp,
                     strerror(errno));
        tmp[i] = '/';
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST)
        cs_fatal("не удалось создать каталог %s: %s", tmp, strerror(errno));
    free(tmp);
}

static char *default_out_dir(const char *batch_path)
{
    /* как os.path.abspath: без разыменования симлинков */
    char *abs;
    if (batch_path[0] == '/')
        abs = xstrdup(batch_path);
    else {
        char cwd[4096];
        if (!getcwd(cwd, sizeof(cwd)))
            cs_fatal("не удалось получить текущий каталог");
        abs = xasprintf("%s/%s", cwd, batch_path);
    }
    char *slash = strrchr(abs, '/');
    if (!slash) {
        free(abs);
        return xstrdup(".");
    }
    if (slash == abs) {
        slash[1] = '\0';
        return abs;
    }
    *slash = '\0';
    return abs;
}

static FILE *open_append(const char *dir, const char *name)
{
    char *p = xasprintf("%s/%s", dir, name);
    FILE *f = fopen(p, "a");
    if (!f)
        cs_fatal("не удалось открыть файл для записи: %s", p);
    free(p);
    return f;
}

static void write_profile(const char *out_dir, double h,
                          const batch_input_t *b, const double *g1,
                          const profile_t *p, double *out_err)
{
    char *path = xasprintf("%s/dist_%g.txt", out_dir, h);
    FILE *fd = fopen(path, "w");
    if (!fd)
        cs_fatal("не удалось открыть файл для записи: %s", path);
    fprintf(fd, "# высота H = %g м\n", h);
    fputs("# входные данные (β [1/(м·ср)], α [1/м]): ", fd);
    for (int j = 0; j < b->nch; j++)
        fprintf(fd, "%s%s=%g", j ? " " : "", b->ch[j].name, g1[j]);
    fputs("\n# r, мкм  dV/dln r, мкм³/см³\n", fd);
    for (int i = 0; i < p->nr; i++) {
        double nv = p->n_avg[i] / CM3_PER_M3;
        if (isfinite(nv))
            fprintf(fd, "%g  %g\n", p->r_grid[i], nv);
    }
    fclose(fd);
    free(path);

    const cs_mode_t *mf = &p->modes.fine;
    const cs_mode_t *mc = &p->modes.coarse;
    const cs_mode_t *mo = &p->modes.overall;
    FILE *fm = open_append(out_dir, "moments.txt");
    fprintf(fm, "%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n", h,
            mf->V / CM3_PER_M3, mc->V / CM3_PER_M3,
            p->moments.vol_conc_um3_per_cm3, mf->rm, mc->rm, mo->rm,
            p->moments.r_eff_um, p->moments.N_cm3, mf->sigma, mc->sigma,
            mo->sigma);
    fclose(fm);

    FILE *fr = open_append(out_dir, "refractive.txt");
    fprintf(fr, "%g  %g  %g\n", h, p->m_re, p->m_im);
    fclose(fr);

    FILE *fe = open_append(out_dir, "errors.txt");
    fprintf(fe, "%g", h);
    double err = 0.0;
    for (int j = 0; j < b->nch; j++) {
        double rel = fabs(p->g_fit[j] - g1[j]) / g1[j];
        fprintf(fe, "  %g", rel);
        err += rel;
    }
    err /= (double)b->nch;
    fprintf(fe, "  %g\n", err);
    fclose(fe);
    if (out_err)
        *out_err = err;
}

static void emit_profile(const args_t *a, const char *out_dir, double h,
                         const batch_input_t *b, const channel_t *solve_ch,
                         const double *g1, const solutions_t *sols,
                         const ktable_t *kts, const double *mre,
                         const double *mim, int nm, const geom_t *geom,
                         double r1_max, const common_t *common, int nl,
                         int write_lidar, metrics_t *met, int hi)
{
    profile_t prof;
    profile_finalize(&prof, a, solve_ch, b->nch, kts, mre, mim, nm, sols,
                     geom, a->r0_min, r1_max);
    double err = NAN;
    write_profile(out_dir, h, b, g1, &prof, &err);

    if (met) {
        met->err[hi] = err;
        met->V[hi] = prof.moments.vol_conc_um3_per_cm3;
        met->Vc[hi] = prof.modes.coarse.V / CM3_PER_M3;
        met->Vf[hi] = prof.modes.fine.V / CM3_PER_M3;
        met->rmf[hi] = prof.modes.fine.rm;
        met->rmc[hi] = prof.modes.coarse.rm;
        met->reff[hi] = prof.moments.r_eff_um;
        met->Nc[hi] = prof.moments.N_cm3;
        met->sgf[hi] = prof.modes.fine.sigma;
        met->sgc[hi] = prof.modes.coarse.sigma;
        met->sgo[hi] = prof.modes.overall.sigma;
        met->m_re[hi] = prof.m_re;
        met->m_im[hi] = prof.m_im;
        for (int k = 0; k < nl && met->nl > 0; k++)
            met->S[(size_t)k * met->n_alt + hi] =
                lidar_ratio_value(prof.g_fit, common, k);
    }

    if (!a->no_plot) {
        char *pp = xasprintf("%s/plot_%g.png", out_dir, h);
        plot_profile(pp, prof.r_grid, prof.nr, prof.n_avg, solve_ch, b->nch,
                     g1, prof.g_fit, h);
        free(pp);
    }

    char *s_str = xstrdup("");
    if (write_lidar) {
        FILE *fl = open_append(out_dir, "lidar_ratios.txt");
        fprintf(fl, "%g", h);
        for (int k = 0; k < nl; k++) {
            double v = lidar_ratio_value(prof.g_fit, common, k);
            if (isfinite(v))
                fprintf(fl, "  %g", v);
            else
                fputs("  nan", fl);
        }
        fputc('\n', fl);
        fclose(fl);
        char *acc = xstrdup("  S = ");
        for (int k = 0; k < nl; k++) {
            double v = lidar_ratio_value(prof.g_fit, common, k);
            char *part = isfinite(v) ? xasprintf("%g", v) : xstrdup("nan");
            char *tmp = xasprintf("%s%s%s", acc, k ? ", " : "", part);
            free(acc);
            free(part);
            acc = tmp;
        }
        free(s_str);
        s_str = acc;
    }

    printf("  высота %g м: ⟨m⟩ = %.3f%+.3fj, N = %.3g см⁻³%s\n", h,
           prof.m_re, prof.m_im, prof.moments.N_cm3, s_str);
    free(s_str);
    profile_free(&prof);
}

/* ------------------------------------------------------------------ */
/* основной сценарий                                                   */
/* ------------------------------------------------------------------ */
void run_batch(const args_t *a)
{
    batch_input_t b;
    parse_batch_txt(a->batch, &b);

    double *mre = NULL, *mim = NULL;
    int nm = 0;
    table_m_grid(a->tables, a->m_reals, a->n_m_reals, a->m_imags,
                 a->n_m_imags, &mre, &mim, &nm);

    printf("пакетная обработка: %s\n", a->batch);
    printf("  высот: %d; каналов: %d\n", b.n_alt, b.nch);
    for (int j = 0; j < b.nch; j++)
        printf("  канал: %s (%s, %g нм)\n", b.ch[j].name,
               b.ch[j].kind == CH_BETA ? "beta" : "alpha",
               b.ch[j].wl_um * 1e3);
    printf("  единицы β [1/(м·ср)], α [1/м]\n");
    if (a->lidar_ratio_cons)
        printf("ограничение плавности лидарного отношения S=α/β с высотой: "
               "включено (вес %g, итераций <= %d)\n", a->lidar_ratio_w,
               a->lidar_ratio_iter);
    else
        printf("ограничение плавности лидарного отношения S=α/β с высотой: "
               "выключено\n");

    /* каналы решения */
    channel_t *solve_ch = xcalloc((size_t)b.nch, sizeof(channel_t));
    for (int j = 0; j < b.nch; j++) {
        solve_ch[j].kind = b.ch[j].kind;
        solve_ch[j].wl_um = b.ch[j].wl_um;
        solve_ch[j].tab = NULL;
        solve_ch[j].ref_wl_um = TABLE_REF_WL_UM;
        solve_ch[j].cback_scale = 1.0;
    }

    double cback_scale = a->table_cback_4pi ? 1.0 / (4.0 * CS_PI) : 1.0;

    /* покрытие таблиц -> r1_max_eff */
    double capmin = INFINITY;
    for (int i = 0; i < nm; i++) {
        channels_t c;
        load_table_channels(a->tables, mre[i], mim[i], solve_ch, b.nch,
                            a->table_extrapolate, a->table_ref_wl,
                            cback_scale, &c);
        if (c.cap < capmin)
            capmin = c.cap;
        channels_free(&c);
    }
    double r1_max_eff = a->r1_max < capmin ? a->r1_max : capmin;
    if (r1_max_eff < a->r1_min)
        cs_fatal("покрытие таблиц по x ограничивает r1_max = %g мкм < "
                 "r1_min = %g — расширьте таблицы или уменьшите --r1-min",
                 r1_max_eff, a->r1_min);

    printf("табличный режим ядер (внешние сечения), корень: %s\n", a->tables);
    printf("  λ_ref таблиц = %g мкм (сечения пересчитываются как (λ/λ_ref)²)\n",
           a->table_ref_wl);
    printf("  конвенция Cback = %s\n", a->table_cback_4pi ? "4pi" : "dsigma");
    {
        channels_t c0;
        load_table_channels(a->tables, mre[0], mim[0], solve_ch, b.nch,
                            a->table_extrapolate, a->table_ref_wl,
                            cback_scale, &c0);
        for (int j = 0; j < b.nch; j++) {
            char cname[64];
            snprintf(cname, sizeof(cname), "%s(%g nm)",
                     solve_ch[j].kind == CH_BETA ? "beta" : "alpha",
                     solve_ch[j].wl_um * 1e3);
            printf("  %-14s x_max = %g, r_max = %.3g мкм\n", cname,
                   c0.ch[j].tab->x_max,
                   c0.ch[j].tab->x_max * c0.ch[j].wl_um / (2.0 * CS_PI));
        }
        channels_free(&c0);
    }
    printf("  эффективный r1_max = %g мкм (задан: %g) — окна и сетка r "
           "ограничены\n", r1_max_eff, a->r1_max);

    /* таблицы ядер — один раз на пакет */
    printf("строим таблицы ядер (%d значений m, один раз на пакет)...\n", nm);
    ktable_t *kts = xcalloc((size_t)nm, sizeof(ktable_t));
    for (int i = 0; i < nm; i++) {
        channels_t c;
        load_table_channels(a->tables, mre[i], mim[i], solve_ch, b.nch,
                            a->table_extrapolate, a->table_ref_wl,
                            cback_scale, &c);
        ktable_build(&kts[i], &c, a->r0_min, r1_max_eff, a->dx_max);
        channels_free(&c);
    }
    printf("  таблица: %d точек по r на каждое m\n", kts[0].n);

    /* матрицы A — тоже один раз */
    double *r0_grid = geomspace(a->r0_min, a->r0_max, a->nr0);
    double *r1_grid = geomspace(a->r1_min, r1_max_eff, a->nr1);
    design_t design;
    design_build(&design, solve_ch, b.nch, kts, nm, mre, mim, a->degrees,
                 a->n_degrees, r0_grid, a->nr0, r1_grid, a->nr1,
                 a->min_ratio, a->n_basis, !a->no_zero_boundary,
                 a->quad_order, a->log_span);
    printf("  матрицы A предвычислены: %d окон × %d m\n", design.n_w, nm);

    char *out_dir = a->batch_out_dir ? xstrdup(a->batch_out_dir)
                                     : default_out_dir(a->batch);
    mkdir_p(out_dir);

    /* шапки общих файлов */
    {
        char *p = xasprintf("%s/moments.txt", out_dir);
        FILE *f = fopen(p, "w");
        if (!f)
            cs_fatal("не удалось открыть файл для записи: %s", p);
        fputs("# H  Vf  Vc  V  rmf  rmc  rm  reff  N  sigma_f  sigma_c  sigma\n"
              "# V — мкм³/см³; rm/reff — мкм; N — см⁻³; sigma — безразм.\n", f);
        fclose(f);
        free(p);
    }
    {
        char *p = xasprintf("%s/refractive.txt", out_dir);
        FILE *f = fopen(p, "w");
        if (!f)
            cs_fatal("не удалось открыть файл для записи: %s", p);
        fputs("# H  Re(m)  Im(m)\n", f);
        fclose(f);
        free(p);
    }
    {
        char *p = xasprintf("%s/errors.txt", out_dir);
        FILE *f = fopen(p, "w");
        if (!f)
            cs_fatal("не удалось открыть файл для записи: %s", p);
        fputs("# H  ", f);
        for (int j = 0; j < b.nch; j++)
            fprintf(f, "%s%s", j ? "  " : "", b.ch[j].name);
        fputs("  Err\n# относительные ошибки |g_fit-g1|/g1 "
              "(Err = среднее по каналам)\n", f);
        fclose(f);
        free(p);
    }

    int nl = 0;
    common_t *common = common_wavelengths(&b, &nl);
    int write_lidar = a->lidar_ratio_cons && nl > 0;
    int retain = a->lidar_ratio_cons && b.n_alt >= 2 && nl > 0;

    /* сводные метрики по высотам — для итоговых графиков */
    metrics_t met;
    met.n_alt = b.n_alt;
    met.nl = nl;
    met.err = dbl_array(b.n_alt);
    met.V = dbl_array(b.n_alt);
    met.Vc = dbl_array(b.n_alt);
    met.Vf = dbl_array(b.n_alt);
    met.rmf = dbl_array(b.n_alt);
    met.rmc = dbl_array(b.n_alt);
    met.reff = dbl_array(b.n_alt);
    met.Nc = dbl_array(b.n_alt);
    met.sgf = dbl_array(b.n_alt);
    met.sgc = dbl_array(b.n_alt);
    met.sgo = dbl_array(b.n_alt);
    met.m_re = dbl_array(b.n_alt);
    met.m_im = dbl_array(b.n_alt);
    met.S = dbl_array(b.n_alt * (nl > 0 ? nl : 1));

    if (write_lidar) {
        char *p = xasprintf("%s/lidar_ratios.txt", out_dir);
        FILE *f = fopen(p, "w");
        if (!f)
            cs_fatal("не удалось открыть файл для записи: %s", p);
        fputs("# H  ", f);
        for (int k = 0; k < nl; k++)
            fprintf(f, "%sS_%g", k ? "  " : "", common[k].wl * 1e3);
        fputs("\n# восстановленное лидарное отношение S = α/β "
              "по общим длинам волн\n", f);
        fclose(f);
        free(p);
    }

    if (retain) {
        solutions_t *per = xcalloc((size_t)b.n_alt, sizeof(solutions_t));
        for (int h = 0; h < b.n_alt; h++) {
            const double *g1 = b.data + (size_t)h * b.nch;
            printf("перебор (m, r0, r1, степень)...\n");
            scan_variants(&design, g1, b.nch, a->gamma_select, a->gamma_rel,
                          a->gamma_min, a->gamma_max, a->ngamma, a->nonneg,
                          !a->absolute, &per[h]);
            if (per[h].n == 0)
                cs_fatal("нет ни одного допустимого окна [r0, r1]");
            printf("  высота %g м: вариантов %d\n", b.alt[h], per[h].n);
        }

        printf("  применяем ограничение: общие длины волн ");
        for (int k = 0; k < nl; k++)
            printf("%s%g нм", k ? ", " : "", common[k].wl * 1e3);
        putchar('\n');

        /* S по каждому решению (ключ — исходный id) */
        double **S = xcalloc((size_t)b.n_alt, sizeof(double *));
        for (int h = 0; h < b.n_alt; h++) {
            int n = per[h].n;
            S[h] = dbl_array(n * (nl > 0 ? nl : 1));
            for (int t = 0; t < n; t++)
                for (int k = 0; k < nl; k++)
                    S[h][(size_t)per[h].sol[t].id * nl + k] =
                        lidar_ratio_value(per[h].sol[t].g_fit, common, k);
        }
        apply_lidar_constraint(per, b.n_alt, b.alt, common, nl,
                               a->lidar_ratio_w, a->lidar_ratio_iter, a->best,
                               (const double *const *)S);
        for (int h = 0; h < b.n_alt; h++)
            free(S[h]);
        free(S);

        for (int h = 0; h < b.n_alt; h++) {
            const double *g1 = b.data + (size_t)h * b.nch;
            emit_profile(a, out_dir, b.alt[h], &b, solve_ch, g1, &per[h], kts,
                         mre, mim, nm, design.geom, r1_max_eff, common, nl,
                         write_lidar, &met, h);
            solutions_free(&per[h]);
        }
        free(per);
    } else {
        if (a->lidar_ratio_cons) {
            if (b.n_alt < 2)
                printf("  [предупреждение] --lidar-ratio-cons: нужно >= 2 "
                       "высот; ограничение не применяется\n");
            else
                printf("  [предупреждение] --lidar-ratio-cons: нет общих "
                       "длин волн β∩α для лидарного отношения; ограничение "
                       "не применяется\n");
        }
        for (int h = 0; h < b.n_alt; h++) {
            const double *g1 = b.data + (size_t)h * b.nch;
            solutions_t sols;
            printf("перебор (m, r0, r1, степень)...\n");
            scan_variants(&design, g1, b.nch, a->gamma_select, a->gamma_rel,
                          a->gamma_min, a->gamma_max, a->ngamma, a->nonneg,
                          !a->absolute, &sols);
            if (sols.n == 0)
                cs_fatal("нет ни одного допустимого окна [r0, r1]");
            printf("  высота %g м: вариантов %d\n", b.alt[h], sols.n);
            emit_profile(a, out_dir, b.alt[h], &b, solve_ch, g1, &sols, kts,
                         mre, mim, nm, design.geom, r1_max_eff, common, nl,
                         write_lidar, &met, h);
            solutions_free(&sols);
        }
    }

    if (!a->no_plot) {
        char *pp;
        pp = xasprintf("%s/refractive.png", out_dir);
        plot_refractive_profile(pp, b.alt, met.m_re, met.m_im, b.n_alt);
        free(pp);

        pp = xasprintf("%s/error.png", out_dir);
        {
            double *err_pct = dbl_array(b.n_alt);
            for (int i = 0; i < b.n_alt; i++)
                err_pct[i] = met.err[i] * 100.0;
            const double *vals[1] = {err_pct};
            const char *names[1] = {"Err"};
            plot_profiles(pp, "Средняя относительная ошибка восстановления, %",
                          "Err, %", 1, b.alt, vals, names, b.n_alt);
            free(err_pct);
        }
        free(pp);

        pp = xasprintf("%s/volumes.png", out_dir);
        {
            const double *vals[3] = {met.V, met.Vc, met.Vf};
            const char *names[3] = {"V", "Vc", "Vf"};
            plot_profiles(pp, "Объёмная концентрация мод",
                          "V, мкм³/см³", 3, b.alt, vals, names, b.n_alt);
        }
        free(pp);

        pp = xasprintf("%s/radii.png", out_dir);
        {
            const double *vals[3] = {met.rmf, met.rmc, met.reff};
            const char *names[3] = {"rmf", "rmc", "reff"};
            plot_profiles(pp, "Радиусы мод", "r, мкм", 3, b.alt, vals, names,
                          b.n_alt);
        }
        free(pp);

        pp = xasprintf("%s/concentration.png", out_dir);
        {
            const double *vals[1] = {met.Nc};
            const char *names[1] = {"N"};
            plot_profiles(pp, "Числовая концентрация", "N, см⁻³", 1, b.alt,
                          vals, names, b.n_alt);
        }
        free(pp);

        pp = xasprintf("%s/sigma.png", out_dir);
        {
            const double *vals[3] = {met.sgf, met.sgc, met.sgo};
            const char *names[3] = {"sigma_f", "sigma_c", "sigma"};
            plot_profiles(pp, "Ширины логнормальных мод", "sigma", 3, b.alt,
                          vals, names, b.n_alt);
        }
        free(pp);

        if (nl > 0) {
            pp = xasprintf("%s/lidar_ratios.png", out_dir);
            const double *svals[32];
            const char *snames[32];
            char *sbuf[32];
            for (int k = 0; k < nl && k < 32; k++) {
                svals[k] = met.S + (size_t)k * b.n_alt;
                /* фигурные скобки — чтобы вся длина волны была подстрочной
                 * (без них gnuplot делает индексом только один символ) */
                sbuf[k] = xasprintf("S_{%g}", common[k].wl * 1e3);
                snames[k] = sbuf[k];
            }
            plot_profiles(pp, "Лидарное отношение S = α/β", "S = α/β",
                          nl < 32 ? nl : 32, b.alt, svals, snames, b.n_alt);
            for (int k = 0; k < nl && k < 32; k++)
                free(sbuf[k]);
            free(pp);
        }
    }

    free(met.err);
    free(met.V);
    free(met.Vc);
    free(met.Vf);
    free(met.rmf);
    free(met.rmc);
    free(met.reff);
    free(met.Nc);
    free(met.sgf);
    free(met.sgc);
    free(met.sgo);
    free(met.m_re);
    free(met.m_im);
    free(met.S);

    printf("пакетная обработка завершена; результаты: %s\n", out_dir);

    free(common);
    free(out_dir);
    design_free(&design);
    free(r0_grid);
    free(r1_grid);
    for (int i = 0; i < nm; i++)
        ktable_free(&kts[i]);
    free(kts);
    free(mre);
    free(mim);
    free(solve_ch);
    batch_input_free(&b);
}
