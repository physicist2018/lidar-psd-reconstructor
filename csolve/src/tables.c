/* Внешние таблицы сечений в формате mkbase / mkbase_spheres.
 *
 * Каталог --tables содержит подкаталоги mre_<real:.2f>_mim_<imag:.5f>/data.txt.
 * data.txt: заголовок-комментарий вида
 *   # x Cext Cabs Cback Qext Qabs Qback [n]
 * и числовые строки. Нужные столбцы выбираются ПО ИМЕНИ из заголовка,
 * комментарии и пустые строки пропускаются.
 */
#define _POSIX_C_SOURCE 200809L

#include "csolve.h"

#include <ctype.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ---------------- динамический вектор ---------------- */
typedef struct {
    double *v;
    int n, cap;
} dvec;

static void dvec_push(dvec *d, double x)
{
    if (d->n == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 256;
        d->v = xrealloc(d->v, sizeof(double) * (size_t)d->cap);
    }
    d->v[d->n++] = x;
}

/* ---------------- разбор строки на токены ---------------- */
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

static int token_index(char **toks, int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(toks[i], name) == 0)
            return i;
    return -1;
}

static int file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int dir_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* ---------------- имена каталогов m ---------------- */
char *m_table_dirname(double re, double im)
{
    /* нормализуем отрицательный ноль: -0.0 + 0.0 -> +0.0 */
    if (re == 0.0)
        re = 0.0;
    if (im == 0.0)
        im = 0.0;
    return xasprintf("mre_%.2f_mim_%.5f", re, im);
}

int parse_m_dirname(const char *name, double *re, double *im)
{
    if (strncmp(name, "mre_", 4) != 0)
        return 0;
    const char *sep = strstr(name + 4, "_mim_");
    if (!sep)
        return 0;
    char *end = NULL;
    double r = strtod(name + 4, &end);
    if (end != sep)
        return 0;
    double i = strtod(sep + 5, &end);
    if (!end || *end != '\0')
        return 0;
    *re = r;
    *im = i;
    return 1;
}

/* ---------------- чтение data.txt ---------------- */
static void load_column_splines(table_t *t)
{
    if (t->has_cext) {
        double *lx = dbl_array(t->n), *ly = dbl_array(t->n);
        for (int i = 0; i < t->n; i++) {
            lx[i] = log(t->x[i]);
            ly[i] = log(t->cext[i]);
        }
        spline_init(&t->sp_cext, lx, ly, t->n);
        free(lx);
        free(ly);
    }
    if (t->has_cback) {
        double *lx = dbl_array(t->n), *ly = dbl_array(t->n);
        for (int i = 0; i < t->n; i++) {
            lx[i] = log(t->x[i]);
            ly[i] = log(t->cback[i]);
        }
        spline_init(&t->sp_cback, lx, ly, t->n);
        free(lx);
        free(ly);
    }
}

table_t *table_load(const char *path, int extrapolate)
{
    FILE *fh = fopen(path, "r");
    if (!fh)
        cs_fatal("cross-section table file not found: %s", path);

    int ix_ext = -1, ic_ext = -1, ix_back = -1, ic_back = -1;
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    /* проход 1: индексы столбцов по заголовку */
    while ((len = getline(&line, &cap, fh)) != -1) {
        (void)len;
        char *s = line;
        while (*s && isspace((unsigned char)*s))
            s++;
        if (*s != '#')
            continue;
        s++;
        char *toks[64];
        int nt = split_tokens(s, toks, 64);
        if (nt == 0 || strcmp(toks[0], "x") != 0)
            continue;
        int ix = token_index(toks, nt, "x");
        if (ic_ext < 0) {
            int ic = token_index(toks, nt, "Cext");
            if (ic >= 0) {
                ix_ext = ix;
                ic_ext = ic;
            }
        }
        if (ic_back < 0) {
            int ic = token_index(toks, nt, "Cback");
            if (ic >= 0) {
                ix_back = ix;
                ic_back = ic;
            }
        }
    }

    if (ic_ext < 0 && ic_back < 0) {
        fclose(fh);
        free(line);
        cs_fatal("%s: mkbase-format header with columns "
                 "'x' and 'Cext' not found (expected, for example, "
                 "'# x Cext Cabs Cback Qext Qabs Qback')", path);
    }
    if (ic_ext >= 0 && ic_back >= 0 && ix_ext != ix_back) {
        fclose(fh);
        free(line);
        cs_fatal("%s: columns Cext and Cback come from headers with "
                 "different x positions — ambiguous table format", path);
    }

    int ix = ic_ext >= 0 ? ix_ext : ix_back;
    int need = ix + 1;
    if (ic_ext >= 0 && ic_ext + 1 > need)
        need = ic_ext + 1;
    if (ic_back >= 0 && ic_back + 1 > need)
        need = ic_back + 1;

    dvec xs = {0}, ve = {0}, vb = {0};
    rewind(fh);
    while ((len = getline(&line, &cap, fh)) != -1) {
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
        if (nt < need) {
            fclose(fh);
            free(line);
            free(xs.v);
            free(ve.v);
            free(vb.v);
            cs_fatal("%s: no required column in row: '%s'", path, s);
        }
        dvec_push(&xs, strtod(toks[ix], NULL));
        if (ic_ext >= 0)
            dvec_push(&ve, strtod(toks[ic_ext], NULL));
        if (ic_back >= 0)
            dvec_push(&vb, strtod(toks[ic_back], NULL));
    }
    fclose(fh);
    free(line);

    int n = xs.n;
    if (n < 3) {
        free(xs.v);
        free(ve.v);
        free(vb.v);
        cs_fatal("%s: insufficient points (>= 3), found %d", path, n);
    }
    for (int i = 1; i < n; i++) {
        if (!(xs.v[i] > xs.v[i - 1])) {
            free(xs.v);
            free(ve.v);
            free(vb.v);
            cs_fatal("%s: x must be strictly increasing", path);
        }
    }
    for (int i = 0; i < n; i++) {
        if (ic_ext >= 0 && (!(ve.v[i] > 0.0) || !isfinite(ve.v[i]))) {
            free(xs.v);
            free(ve.v);
            free(vb.v);
            cs_fatal("%s: Cext cross sections must be > 0 and finite", path);
        }
        if (ic_back >= 0 && (!(vb.v[i] > 0.0) || !isfinite(vb.v[i]))) {
            free(xs.v);
            free(ve.v);
            free(vb.v);
            cs_fatal("%s: Cback cross sections must be > 0 and finite", path);
        }
    }
    double dx_max = 0.0;
    for (int i = 1; i < n; i++) {
        double d = xs.v[i] - xs.v[i - 1];
        if (d > dx_max)
            dx_max = d;
    }
    if (dx_max > 1.0)
        fprintf(stderr, "[warning] table %s: large step in x "
                        "(max \xce\x94x = %g); Mie ripple may not be resolved\n",
                path, dx_max);

    table_t *t = xcalloc(1, sizeof(*t));
    t->n = n;
    t->x = xs.v;
    t->x_min = xs.v[0];
    t->x_max = xs.v[n - 1];
    t->extrapolate = extrapolate;
    t->path = xstrdup(path);
    if (ic_ext >= 0) {
        t->has_cext = 1;
        t->cext = ve.v;
    } else {
        free(ve.v);
    }
    if (ic_back >= 0) {
        t->has_cback = 1;
        t->cback = vb.v;
    } else {
        free(vb.v);
    }
    load_column_splines(t);
    return t;
}

void table_free(table_t *t)
{
    if (!t)
        return;
    free(t->x);
    free(t->cext);
    free(t->cback);
    spline_free(&t->sp_cext);
    spline_free(&t->sp_cback);
    free(t->path);
    free(t);
}

double table_cross_section(const table_t *t, int use_cback, double x)
{
    if (!t->extrapolate && (x < t->x_min || x > t->x_max))
        cs_fatal("size parameter x outside table range [%g, %g] — "
                 "extend the table range or enable --table-extrapolate (%s)",
                 t->x_min, t->x_max, t->path);
    const spline_t *sp = use_cback ? &t->sp_cback : &t->sp_cext;
    return exp(spline_eval(sp, log(x)));
}

/* ---------------- каталог m ---------------- */
typedef struct {
    double re, im;
} mv_t;

static int mv_cmp(const void *pa, const void *pb)
{
    const mv_t *a = pa, *b = pb;
    if (a->re < b->re)
        return -1;
    if (a->re > b->re)
        return 1;
    if (a->im < b->im)
        return -1;
    if (a->im > b->im)
        return 1;
    return 0;
}

static mv_t *collect_available(const char *root, int *out_n)
{
    *out_n = 0;
    DIR *d = opendir(root);
    if (!d)
        return NULL;
    mv_t *vals = NULL;
    int n = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        double re, im;
        if (!parse_m_dirname(e->d_name, &re, &im))
            continue;
        char *path = xasprintf("%s/%s/data.txt", root, e->d_name);
        int ok = file_exists(path);
        free(path);
        if (!ok)
            continue;
        int dup = 0;
        for (int i = 0; i < n; i++)
            if (vals[i].re == re && vals[i].im == im) {
                dup = 1;
                break;
            }
        if (dup)
            continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            vals = xrealloc(vals, sizeof(mv_t) * (size_t)cap);
        }
        vals[n].re = re;
        vals[n].im = im;
        n++;
    }
    closedir(d);
    if (n > 1)
        qsort(vals, (size_t)n, sizeof(mv_t), mv_cmp);
    *out_n = n;
    return vals;
}

static int str_cmp(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static char *layout_hint(const char *root)
{
    if (!dir_exists(root))
        return xasprintf("; directory %s does not exist", root);
    DIR *d = opendir(root);
    if (!d)
        return xstrdup("");
    char *names[64];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < 64) {
        const char *name = e->d_name;
        int digits_ok = 1;
        if (strncmp(name, "shape", 5) == 0) {
            if (!name[5])
                digits_ok = 0;
            for (const char *p = name + 5; *p; p++)
                if (!isdigit((unsigned char)*p))
                    digits_ok = 0;
        }
        int match = (strcmp(name, "average") == 0) ||
                    ((strncmp(name, "shape", 5) == 0) && digits_ok);
        if (!match)
            continue;
        char *path = xasprintf("%s/%s", root, name);
        if (dir_exists(path))
            names[n++] = xstrdup(name);
        free(path);
    }
    closedir(d);
    if (n == 0)
        return xstrdup("");
    qsort(names, (size_t)n, sizeof(char *), str_cmp);
    char *joined = xstrdup("");
    for (int i = 0; i < n; i++) {
        char *tmp = xasprintf("%s%s%s", joined, i ? ", " : "", names[i]);
        free(joined);
        joined = tmp;
    }
    char *hint = xasprintf("; %s contains %s — specify --tables %s/average "
                           "or --tables <root>/<shapeNN>", root, joined, root);
    free(joined);
    for (int i = 0; i < n; i++)
        free(names[i]);
    return hint;
}

static char *m_values_str(const mv_t *v, int n)
{
    char *out = xstrdup("");
    for (int i = 0; i < n && i < 10; i++) {
        char *part = xasprintf("%g%+gj", v[i].re, v[i].im);
        char *tmp = xasprintf("%s%s%s", out, i ? ", " : "", part);
        free(out);
        free(part);
        out = tmp;
    }
    if (n > 10) {
        char *tmp = xasprintf("%s, ...", out);
        free(out);
        out = tmp;
    }
    return out;
}

void table_m_grid(const char *root, double **out_re, double **out_im,
                  int *out_n)
{
    int nav = 0;
    mv_t *avail = collect_available(root, &nav);
    if (!dir_exists(root)) {
        free(avail);
        char *hint = layout_hint(root);
        char *msg = xasprintf("table directory not found: %s%s", root, hint);
        free(hint);
        cs_fatal("%s", msg);
    }
    if (nav == 0) {
        free(avail);
        char *hint = layout_hint(root);
        char *msg = xasprintf("table directory %s contains no subdirectories "
                              "mre_<real>_mim_<imag>/data.txt%s", root, hint);
        free(hint);
        cs_fatal("%s", msg);
    }
    double *re = dbl_array(nav), *im = dbl_array(nav);
    for (int i = 0; i < nav; i++) {
        re[i] = avail[i].re;
        im[i] = avail[i].im;
    }
    free(avail);
    *out_re = re;
    *out_im = im;
    *out_n = nav;
}

/* ---------------- кэш загруженных таблиц ---------------- */
typedef struct {
    char *root;
    double re, im;
    int extrapolate;
    table_t *tab;
} tcache_ent;

static tcache_ent *cache;
static int cache_n;

static void missing_table_error(const char *root, const char *mdir,
                                double mre, double mim)
{
    char *dir = m_table_dirname(mre, mim);
    char *msg = xasprintf("table directory not found: %s (expected "
                          "%s/data.txt — mkbase/mkbase_spheres format)",
                          mdir, dir);
    free(dir);
    if (!dir_exists(root)) {
        char *hint = layout_hint(root);
        cs_fatal("%s%s", msg, hint);
    }
    int nav = 0;
    mv_t *avail = collect_available(root, &nav);
    if (nav > 0) {
        char *list = m_values_str(avail, nav);
        cs_fatal("%s; available m in %s: %s", msg, root, list);
    }
    free(avail);
    char *hint = layout_hint(root);
    cs_fatal("%s%s", msg, hint);
}

static table_t *table_cache_get(const char *root, double re, double im,
                                int extrapolate)
{
    for (int i = 0; i < cache_n; i++) {
        tcache_ent *e = &cache[i];
        if (e->extrapolate == extrapolate && e->re == re && e->im == im &&
            strcmp(e->root, root) == 0)
            return e->tab;
    }
    char *dir = m_table_dirname(re, im);
    char *mdir = xasprintf("%s/%s", root, dir);
    char *path = xasprintf("%s/data.txt", mdir);
    if (!file_exists(path))
        missing_table_error(root, mdir, re, im);
    table_t *t = table_load(path, extrapolate);
    cache = xrealloc(cache, sizeof(tcache_ent) * (size_t)(cache_n + 1));
    cache[cache_n].root = xstrdup(root);
    cache[cache_n].re = re;
    cache[cache_n].im = im;
    cache[cache_n].extrapolate = extrapolate;
    cache[cache_n].tab = t;
    cache_n++;
    free(dir);
    free(mdir);
    free(path);
    return t;
}

void channels_free(channels_t *c)
{
    if (!c)
        return;
    free(c->ch);
    c->ch = NULL;
    c->n = 0;
}

void load_table_channels(const char *root, double mre, double mim,
                         const channel_t *base, int nch,
                         int extrapolate, double ref_wl, double cback_scale,
                         channels_t *out)
{
    table_t *tab = table_cache_get(root, mre, mim, extrapolate);
    out->n = nch;
    out->ch = xcalloc((size_t)(nch > 0 ? nch : 1), sizeof(channel_t));
    out->cap = INFINITY;
    for (int i = 0; i < nch; i++) {
        channel_t ch = base[i];
        ch.tab = tab;
        ch.ref_wl_um = ref_wl;
        ch.cback_scale = ch.kind == CH_BETA ? cback_scale : 1.0;
        if (ch.kind == CH_ALPHA && !tab->has_cext) {
            cs_fatal("%s: mkbase-format header with columns "
                     "'x' and 'Cext' not found (expected, for example, "
                     "'# x Cext Cabs Cback Qext Qabs Qback')", tab->path);
        }
        if (ch.kind == CH_BETA && !tab->has_cback) {
            cs_fatal("%s: mkbase-format header with columns "
                     "'x' and 'Cback' not found (expected, for example, "
                     "'# x Cext Cabs Cback Qext Qabs Qback')", tab->path);
        }
        out->ch[i] = ch;
        double cap = tab->x_max * ch.wl_um / (2.0 * CS_PI);
        if (cap < out->cap)
            out->cap = cap;
    }
}
