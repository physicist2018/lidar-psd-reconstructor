/* Command-line parsing via argtable3 (vendored, third_party/argtable3).
 *
 * All options are declared in the argtable table below. argtable3 performs
 * the parsing, the value checks (arg_rex patterns for enumerations and a
 * custom checkfn installed on restricted numeric options), the error
 * reporting (arg_print_errors) and the --help output
 * (arg_print_syntax / arg_print_glossary). */
#include "csolve.h"
#include "argtable3.h"

#include <stdlib.h>
#include <string.h>

/* Patterns for arg_rex: enumerations. */

/* ------------------------------------------------------------------ */
/* range checks performed by argtable3 (custom checkfn + errorfn)      */
/* ------------------------------------------------------------------ */

enum { CS_ERR_RANGE = 1000 };

typedef struct {
    void *parent;
    const char *name;
    double lo, hi;
    int integer;
    int allow_zero;     /* the value 0 is accepted as well */
    int strict_lo;      /* the lower bound is exclusive */
    arg_errorfn *orig_errorfn;
} cs_range_t;

static cs_range_t cs_ranges[32];
static int cs_nranges;

static cs_range_t *cs_range_find(const void *parent)
{
    for (int i = 0; i < cs_nranges; i++)
        if (cs_ranges[i].parent == parent)
            return &cs_ranges[i];
    return NULL;
}

static int cs_range_check(void *parent)
{
    cs_range_t *r = cs_range_find(parent);
    if (!r)
        return 0;

    double v;
    int ok = 1;
    if (r->integer) {
        const arg_int_t *p = parent;
        for (int i = 0; i < p->count && ok; i++) {
            v = (double)p->ival[i];
            ok = !(v < r->lo || v > r->hi);
        }
        return ok ? 0 : CS_ERR_RANGE;
    }

    const arg_dbl_t *p = parent;
    for (int i = 0; i < p->count && ok; i++) {
        v = p->dval[i];
        if (!isfinite(v)) {
            ok = 0;
            break;
        }
        ok = r->strict_lo ? (v > r->lo) : (v >= r->lo);
        if (v > r->hi)
            ok = 0;
        if (!ok && r->allow_zero && v == 0.0)
            ok = 1;
    }
    return ok ? 0 : CS_ERR_RANGE;
}

static void cs_range_error(void *parent, arg_dstr_t ds, int error,
                           const char *argval, const char *progname)
{
    cs_range_t *r = cs_range_find(parent);
    if (!r || error != CS_ERR_RANGE) {
        /* parse-time errors (bad number etc.) keep the library wording */
        if (r && r->orig_errorfn) {
            r->orig_errorfn(parent, ds, error, argval, progname);
            return;
        }
        arg_dstr_catf(ds, "%s: invalid option value\n", progname);
        return;
    }
    if (r->allow_zero) {
        arg_dstr_catf(ds, "%s: option --%s must be 0 or between %g and %g\n",
                      progname, r->name, r->lo, r->hi);
    } else if (r->hi >= HUGE_VAL) {
        if (r->strict_lo)
            arg_dstr_catf(ds, "%s: option --%s must be > %g\n",
                          progname, r->name, r->lo);
        else
            arg_dstr_catf(ds, "%s: option --%s must be >= %g\n",
                          progname, r->name, r->lo);
    } else {
        arg_dstr_catf(ds, "%s: option --%s must be between %g and %g\n",
                      progname, r->name, r->lo, r->hi);
    }
}

static void cs_restrict(void *parent, const char *name, double lo, double hi,
                        int integer, int allow_zero, int strict_lo)
{
    cs_range_t *r = &cs_ranges[cs_nranges++];
    arg_hdr_t *h = parent;

    r->parent = parent;
    r->name = name;
    r->lo = lo;
    r->hi = hi;
    r->integer = integer;
    r->allow_zero = allow_zero;
    r->strict_lo = strict_lo;
    r->orig_errorfn = h->errorfn;
    h->checkfn = cs_range_check;
    h->errorfn = cs_range_error;
}

/* ------------------------------------------------------------------ */
/* help                                                                */
/* ------------------------------------------------------------------ */

static void print_help(void **argtable)
{
    fputs("csolve - retrieval of the particle size distribution from lidar\n"
          "coefficients using external cross-section tables "
          "(C port of the tabular mode of solve).\n"
          "\n"
          "Usage:\n"
          "  csolve", stdout);
    arg_print_syntax(stdout, argtable, "\n");
    fputs("\nOptions:\n", stdout);
    arg_print_glossary(stdout, argtable, "  %-32s %s\n");
    fputs("\nMie mode (miepython) is not implemented in the C version: "
          "kernels come from tables only.\n", stdout);
}

/* ------------------------------------------------------------------ */
/* greedy multi-value options                                          */
/* ------------------------------------------------------------------ */

/* argtable3 accepts several values only by repeating the option, so a tail
 * such as --degree 1 2 3 4 is expanded into
 * --degree 1 --degree 2 --degree 3 --degree 4. */
static const char *multivalue_name(const char *tok)
{
    static const char *const names[] = {"--degree"};
    for (int k = 0; k < (int)(sizeof(names) / sizeof(names[0])); k++) {
        size_t n = strlen(names[k]);
        if (strncmp(tok, names[k], n) == 0 && (tok[n] == '\0' || tok[n] == '='))
            return names[k];
    }
    return NULL;
}

static char **expand_multivalue(int argc, char **argv, int *out_argc)
{
    /* worst case: every value needs a repeated option name */
    char **out = xmalloc(sizeof(char *) * (size_t)(2 * argc + 1));
    const char *mv = NULL;      /* current multi-value option */
    int have_value = 0;         /* a value was already seen for it */
    int n = 0;

    out[n++] = argv[0];
    for (int i = 1; i < argc; i++) {
        const char *tok = argv[i];
        if (strncmp(tok, "--", 2) == 0) {
            mv = multivalue_name(tok);
            have_value = mv && strchr(tok, '=') != NULL;
            out[n++] = argv[i];
            continue;
        }
        if (!mv) {
            out[n++] = argv[i];
            continue;
        }
        if (have_value)
            out[n++] = (char *)mv;   /* repeat the option name */
        out[n++] = argv[i];
        have_value = 1;
    }
    *out_argc = n;
    return out;
}

void parse_args(int argc, char **argv, args_t *a)
{
    memset(a, 0, sizeof(*a));
    a->degrees[0] = 1; a->degrees[1] = 2; a->degrees[2] = 3; a->degrees[3] = 4;
    a->n_degrees = 4;
    a->table_ref_wl = TABLE_REF_WL_UM;
    a->r0_min = 0.02; a->r0_max = 0.5; a->nr0 = 6;
    a->r1_min = 2.0; a->r1_max = 10.0; a->nr1 = 4;
    a->min_ratio = 4.0;
    a->best = 5;
    a->gamma_min = -7.0; a->gamma_max = 5.0; a->ngamma = 41;
    a->gamma_select = GAMMA_FIXED;
    a->gamma_rel = 0.1;
    a->smooth_lam = 1e-4;
    a->r_split = 0.5;
    a->modes = 0;
    a->dx_max = 0.1;
    a->nr_out = 256;
    a->quad_order = 24;
    a->log_span = 0.1;
    a->seed = 1;
    a->out = NULL;
    a->plot_format = PLOT_FMT_PNG;
    a->plot_dpi = 150.0;
    a->plot_font_size = 0.0;
    a->lidar_ratio_w = 0.5;
    a->lidar_ratio_iter = 3;

    cs_nranges = 0;

    arg_lit_t *help = arg_lit0("h", "help", "show this help and exit");
    arg_lit_t *no_jit = arg_lit0(NULL, "no-jit",
                                 "accepted and ignored (no Mie backend)");

    arg_str_t *data = arg_str0(NULL, "data", "<JSON>",
                               "measured coefficients; demo data if omitted");
    arg_str_t *tables = arg_strn(NULL, "tables", "<DIR>", 1, 1,
                                 "directory with mre_<real>_mim_<imag>/data.txt "
                                 "(required, tabular mode only)");
    arg_lit_t *table_extrapolate = arg_lit0(NULL, "table-extrapolate",
                                            "allow cross-section extrapolation in x");
    arg_dbl_t *table_ref_wl = arg_dbl0(NULL, "table-ref-lambda-um", "<X>",
                                       "reference wavelength of tables, um [2pi]");
    arg_rex_t *table_cback = arg_rex0(NULL, "table-cback", "^(dsigma|4pi)$",
                                      "{dsigma|4pi}", 0,
                                      "Cback column convention [dsigma]");

    arg_int_t *degree = arg_intn(NULL, "degree", "<1|2|3|4>", 0, 8,
                                 "B-spline degrees: 1, 2, 3 or 4 [1 2 3 4]");
    arg_int_t *n_basis = arg_int0(NULL, "n-basis", "<N>",
                                  "basis functions per window [number of channels]");
    arg_dbl_t *r0_min = arg_dbl0(NULL, "r0-min", "<X>",
                                 "lower bound of r0 grid, > 0 [0.02]");
    arg_dbl_t *r0_max = arg_dbl0(NULL, "r0-max", "<X>",
                                 "upper bound of r0 grid [0.5]");
    arg_int_t *nr0 = arg_int0(NULL, "nr0", "<N>",
                              "number of r0 values, >= 1 [6]");
    arg_dbl_t *r1_min = arg_dbl0(NULL, "r1-min", "<X>",
                                 "lower bound of r1 grid [2]");
    arg_dbl_t *r1_max = arg_dbl0(NULL, "r1-max", "<X>",
                                 "upper bound of r1 grid [10]");
    arg_int_t *nr1 = arg_int0(NULL, "nr1", "<N>",
                              "number of r1 values, >= 1 [4]");
    arg_dbl_t *min_ratio = arg_dbl0(NULL, "min-ratio", "<X>",
                                    "drop windows with r1 < X*r0 [4]");
    arg_int_t *best = arg_int0(NULL, "best", "<G>",
                               "number of best solutions G [5]");

    arg_rex_t *gamma_select = arg_rex0(NULL, "gamma-select",
                                       "^(fixed|cv|err)$", "{fixed|cv|err}", 0,
                                       "gamma selection rule [fixed]");
    arg_dbl_t *gamma_rel = arg_dbl0(NULL, "gamma-rel", "<X>",
                                    "gamma/||A||_2, >= 0 [0.1]");
    arg_dbl_t *gamma_min = arg_dbl0(NULL, "gamma-min", "<X>", "min lg gamma [-7]");
    arg_dbl_t *gamma_max = arg_dbl0(NULL, "gamma-max", "<X>", "max lg gamma [5]");
    arg_int_t *ngamma = arg_int0(NULL, "ngamma", "<N>",
                                 "number of gamma values, >= 1 [41]");

    arg_lit_t *nonneg = arg_lit0(NULL, "nonneg", "constrain C >= 0");
    arg_lit_t *no_zero_boundary = arg_lit0(NULL, "no-zero-boundary",
                                           "no boundary conditions n(r0)=n(r1)=0");
    arg_lit_t *no_smooth = arg_lit0(NULL, "no-smooth",
                                    "disable smoothing of the mean");
    arg_dbl_t *smooth_lam = arg_dbl0(NULL, "smooth-lam", "<X>",
                                     "smoothing parameter [1e-4]");
    arg_dbl_t *r_split = arg_dbl0(NULL, "r-split", "<X>",
                                  "fine/coarse boundary [0.5]");
    arg_rex_t *modes = arg_rex0(NULL, "modes", "^(auto|fit|stat)$",
                                "{auto|fit|stat}", 0,
                                "mode parameters estimation [auto]");
    arg_lit_t *absolute = arg_lit0(NULL, "absolute",
                                   "least squares on absolute residuals");
    arg_dbl_t *dx_max = arg_dbl0(NULL, "dx-max", "<X>",
                                 "max step in x for kernel tables, > 0 [0.1]");
    arg_int_t *nr_out = arg_int0(NULL, "nr-out", "<N>",
                                 "points in output grid, >= 2 [256]");
    arg_int_t *quad_order = arg_int0(NULL, "quad-order", "<N>",
                                     "Gauss-Legendre order, >= 1 [24]");
    arg_dbl_t *log_span = arg_dbl0(NULL, "log-span", "<X>",
                                   "max subsegment width in ln r, > 0 [0.1]");

    arg_dbl_t *m_real = arg_dbl0(NULL, "m-real", "<X>", "true Re(m) for demo");
    arg_dbl_t *m_imag = arg_dbl0(NULL, "m-imag", "<X>", "true Im(m) for demo");

    arg_dbl_t *noise = arg_dbl0(NULL, "noise", "<X>", "demo noise level [0]");
    arg_int_t *seed = arg_int0(NULL, "seed", "<N>", "RNG seed [1]");
    arg_str_t *out = arg_str0(NULL, "out", "<FILE>",
                              "plot output file [result.<fmt>]");
    arg_rex_t *plot_format = arg_rex0(NULL, "plot-format", "^(png|pdf|svg)$",
                                      "{png|pdf|svg}", 0,
                                      "plot format [png]");
    arg_dbl_t *plot_dpi = arg_dbl0(NULL, "plot-dpi", "<N>",
                                   "resolution for png/svg, dpi, > 0 [150]");
    arg_dbl_t *plot_font_size = arg_dbl0(NULL, "plot-font-size", "<pt>",
                                         "font size of axis labels and series "
                                         "labels, pt (0 or 4..72) "
                                         "[0 - terminal default]");
    arg_str_t *json = arg_str0(NULL, "json", "<JSON>", "save results to JSON");
    arg_lit_t *no_plot = arg_lit0(NULL, "no-plot", "do not build plots");

    arg_str_t *batch = arg_str0(NULL, "batch", "<TXT>",
                                "batch processing from txt "
                                "(Altitude b<nm>... a<nm>...)");
    arg_str_t *batch_out_dir = arg_str0(NULL, "batch-out-dir", "<DIR>",
                                        "batch output directory [next to TXT]");
    arg_lit_t *lidar_ratio_cons = arg_lit0(NULL, "lidar-ratio-cons",
                                           "constrain smoothness of the lidar "
                                           "ratio S = alpha/beta with altitude");
    arg_dbl_t *lidar_ratio_w = arg_dbl0(NULL, "lidar-ratio-w", "<X>",
                                        "weight of S deviation penalty, >= 0 [0.5]");
    arg_int_t *lidar_ratio_iter = arg_int0(NULL, "lidar-ratio-iter", "<N>",
                                           "iterations to refine reference S, "
                                           ">= 1 [3]");

    /* Restricted numeric options: the checks are executed by argtable3. */
    cs_restrict(degree, "degree", 1, 4, 1, 0, 0);
    cs_restrict(nr0, "nr0", 1, HUGE_VAL, 1, 0, 0);
    cs_restrict(nr1, "nr1", 1, HUGE_VAL, 1, 0, 0);
    cs_restrict(nr_out, "nr-out", 2, HUGE_VAL, 1, 0, 0);
    cs_restrict(quad_order, "quad-order", 1, HUGE_VAL, 1, 0, 0);
    cs_restrict(ngamma, "ngamma", 1, HUGE_VAL, 1, 0, 0);
    cs_restrict(lidar_ratio_iter, "lidar-ratio-iter", 1, HUGE_VAL, 1, 0, 0);
    cs_restrict(r0_min, "r0-min", 0, HUGE_VAL, 0, 0, 1);
    cs_restrict(dx_max, "dx-max", 0, HUGE_VAL, 0, 0, 1);
    cs_restrict(log_span, "log-span", 0, HUGE_VAL, 0, 0, 1);
    cs_restrict(plot_dpi, "plot-dpi", 0, HUGE_VAL, 0, 0, 1);
    cs_restrict(gamma_rel, "gamma-rel", 0, HUGE_VAL, 0, 0, 0);
    cs_restrict(lidar_ratio_w, "lidar-ratio-w", 0, HUGE_VAL, 0, 0, 0);
    cs_restrict(plot_font_size, "plot-font-size", 4, 72, 0, 1, 0);

    arg_end_t *end = arg_end(20);
    void *argtable[] = {
        help, no_jit,
        data, tables, table_extrapolate, table_ref_wl, table_cback,
        degree, n_basis, r0_min, r0_max, nr0, r1_min, r1_max, nr1,
        min_ratio, best,
        gamma_select, gamma_rel, gamma_min, gamma_max, ngamma,
        nonneg, no_zero_boundary, no_smooth, smooth_lam, r_split,
        modes, absolute, dx_max, nr_out, quad_order, log_span,
        m_real, m_imag,
        noise, seed, out, plot_format, plot_dpi, plot_font_size, json,
        no_plot,
        batch, batch_out_dir, lidar_ratio_cons, lidar_ratio_w,
        lidar_ratio_iter,
        end,
    };
    const size_t ntab = sizeof(argtable) / sizeof(argtable[0]);

    int nargv = 0;
    char **eargv = expand_multivalue(argc, argv, &nargv);
    int nerrors = arg_parse(nargv, eargv, argtable);
    free(eargv);

    if (help->count > 0) {
        print_help(argtable);
        arg_freetable(argtable, ntab);
        exit(0);
    }
    if (nerrors > 0) {
        arg_print_errors(stderr, end, argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        arg_freetable(argtable, ntab);
        exit(1);
    }

    if (data->count > 0)
        a->data = data->sval[0];
    a->tables = tables->sval[0];
    a->table_extrapolate = table_extrapolate->count > 0;
    if (table_ref_wl->count > 0)
        a->table_ref_wl = table_ref_wl->dval[0];
    if (table_cback->count > 0)
        a->table_cback_4pi = strcmp(table_cback->sval[0], "4pi") == 0;

    if (degree->count > 0) {
        a->n_degrees = degree->count;
        for (int i = 0; i < a->n_degrees; i++)
            a->degrees[i] = degree->ival[i];
    }
    if (n_basis->count > 0)
        a->n_basis = n_basis->ival[0];
    if (r0_min->count > 0)
        a->r0_min = r0_min->dval[0];
    if (r0_max->count > 0)
        a->r0_max = r0_max->dval[0];
    if (nr0->count > 0)
        a->nr0 = nr0->ival[0];
    if (r1_min->count > 0)
        a->r1_min = r1_min->dval[0];
    if (r1_max->count > 0)
        a->r1_max = r1_max->dval[0];
    if (nr1->count > 0)
        a->nr1 = nr1->ival[0];
    if (min_ratio->count > 0)
        a->min_ratio = min_ratio->dval[0];
    if (best->count > 0)
        a->best = best->ival[0];

    if (gamma_select->count > 0) {
        const char *v = gamma_select->sval[0];
        a->gamma_select = strcmp(v, "cv") == 0 ? GAMMA_CV
                          : strcmp(v, "err") == 0 ? GAMMA_ERR
                                                   : GAMMA_FIXED;
    }
    if (gamma_rel->count > 0)
        a->gamma_rel = gamma_rel->dval[0];
    if (gamma_min->count > 0)
        a->gamma_min = gamma_min->dval[0];
    if (gamma_max->count > 0)
        a->gamma_max = gamma_max->dval[0];
    if (ngamma->count > 0)
        a->ngamma = ngamma->ival[0];

    a->nonneg = nonneg->count > 0;
    a->no_zero_boundary = no_zero_boundary->count > 0;
    a->no_smooth = no_smooth->count > 0;
    if (smooth_lam->count > 0)
        a->smooth_lam = smooth_lam->dval[0];
    if (r_split->count > 0)
        a->r_split = r_split->dval[0];
    if (modes->count > 0) {
        const char *v = modes->sval[0];
        a->modes = strcmp(v, "stat") == 0 ? 2 : strcmp(v, "fit") == 0 ? 1 : 0;
    }
    a->absolute = absolute->count > 0;
    if (dx_max->count > 0)
        a->dx_max = dx_max->dval[0];
    if (nr_out->count > 0)
        a->nr_out = nr_out->ival[0];
    if (quad_order->count > 0)
        a->quad_order = quad_order->ival[0];
    if (log_span->count > 0)
        a->log_span = log_span->dval[0];

    if (m_real->count > 0) {
        a->has_m_real = 1;
        a->m_real = m_real->dval[0];
    }
    if (m_imag->count > 0) {
        a->has_m_imag = 1;
        a->m_imag = m_imag->dval[0];
    }

    if (noise->count > 0)
        a->noise = noise->dval[0];
    if (seed->count > 0)
        a->seed = seed->ival[0];
    if (out->count > 0)
        a->out = out->sval[0];
    if (plot_format->count > 0) {
        const char *v = plot_format->sval[0];
        a->plot_format = strcmp(v, "pdf") == 0 ? PLOT_FMT_PDF
                         : strcmp(v, "svg") == 0 ? PLOT_FMT_SVG
                                                  : PLOT_FMT_PNG;
    }
    if (plot_dpi->count > 0)
        a->plot_dpi = plot_dpi->dval[0];
    if (plot_font_size->count > 0)
        a->plot_font_size = plot_font_size->dval[0];
    if (json->count > 0)
        a->json = json->sval[0];
    a->no_plot = no_plot->count > 0;

    if (batch->count > 0)
        a->batch = batch->sval[0];
    if (batch_out_dir->count > 0)
        a->batch_out_dir = batch_out_dir->sval[0];
    a->lidar_ratio_cons = lidar_ratio_cons->count > 0;
    if (lidar_ratio_w->count > 0)
        a->lidar_ratio_w = lidar_ratio_w->dval[0];
    if (lidar_ratio_iter->count > 0)
        a->lidar_ratio_iter = lidar_ratio_iter->ival[0];

    /* no_jit is deliberately ignored: there is no Mie backend in C */
    (void)no_jit;

    arg_freetable(argtable, ntab);
}
