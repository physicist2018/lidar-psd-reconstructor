/* csolve — табличный солвер восстановления распределения частиц по размерам.
 *
 * C-переписывание табличного режима (--tables) Python-пакета solve:
 * ядра берутся из внешних таблиц сечений Cext(x)/Cback(x), далее тот же
 * конвейер (B-сплайны, квадратуры Гаусса-Лежандра, регуляризованный МНК,
 * перебор окон, ансамбль, моменты и логнормальные моды).
 */
#ifndef CSOLVE_H
#define CSOLVE_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CS_PI M_PI
#define CS_EPS 2.2204460492503131e-16
#define UM2_TO_M2 1e-12
#define TABLE_REF_WL_UM (2.0 * CS_PI)
#define CM3_PER_M3 1e6
#define UM3_TO_M3 1e-18

/* ------------------------------------------------------------------ */
/* ошибки и память                                                     */
/* ------------------------------------------------------------------ */
void cs_fatal(const char *fmt, ...);
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
char *xasprintf(const char *fmt, ...);
double *dbl_array(int n);
int *int_array(int n);

/* ------------------------------------------------------------------ */
/* утилиты                                                             */
/* ------------------------------------------------------------------ */
double *linspace(double a, double b, int n);
double *geomspace(double a, double b, int n);
int dbl_eq(double a, double b);
double trapz(const double *y, const double *x, int n);
/* линейная интерполяция с зажимом на концах (как np.interp) */
double interp_clamped(double x, const double *xs, const double *ys, int n);

/* ------------------------------------------------------------------ */
/* кубический сплайн (not-a-knot)                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    int n;
    double *x, *y;
    double *b, *c, *d;   /* коэффициенты на интервалах 0..n-2 */
} spline_t;

void spline_init(spline_t *s, const double *x, const double *y, int n);
double spline_eval(const spline_t *s, double t);
/* то же, но с подсказкой интервала для монотонно возрастающих t */
double spline_eval_hint(const spline_t *s, double t, int *hint);
void spline_free(spline_t *s);

/* ------------------------------------------------------------------ */
/* линейная алгебра                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    int m, n;
    double *w;    /* матрица после вращений: W = U*S (m x n) */
    double *s;    /* сингулярные числа (n) */
    double *v;    /* V (n x n) */
} svd_t;

void svd_compute(svd_t *out, const double *a, int m, int n);
void svd_free(svd_t *s);
double svd_norm2(const double *a, int m, int n);
/* МНК: min ||a x - b||, отсечка как в numpy lstsq (rcond=None) */
void lstsq(const double *a, int m, int n, const double *b, double *x);
/* ленточная Холецкого: симметричная положительно определённая матрица,
 * полуширина ленты kb; band[i*(2kb+1) + (j-i+kb)] = M[i][j] */
void band_chol_solve(double *band, int n, int kb, const double *b, double *x);
/* NNLS (Lawson-Hanson): min ||a x - b||, x >= 0 */
void nnls(const double *a, int m, int n, const double *b, double *x);

/* ------------------------------------------------------------------ */
/* таблицы сечений                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    int n;
    double *x, *cext, *cback;
    int has_cext, has_cback;
    spline_t sp_cext, sp_cback;    /* ln C от ln x */
    double x_min, x_max;
    int extrapolate;
    char *path;
} table_t;

table_t *table_load(const char *path, int extrapolate);
void table_free(table_t *t);
double table_cross_section(const table_t *t, int use_cback, double x);

char *m_table_dirname(double re, double im);
int parse_m_dirname(const char *name, double *re, double *im);

void table_m_grid(const char *root,
                  const double *m_reals, int n_reals,
                  const double *m_imags, int n_imags,
                  double **out_re, double **out_im, int *out_n);

/* ------------------------------------------------------------------ */
/* каналы и таблица ядер                                               */
/* ------------------------------------------------------------------ */
typedef enum { CH_ALPHA = 0, CH_BETA = 1 } ch_kind_t;

typedef struct {
    ch_kind_t kind;
    double wl_um;
    table_t *tab;          /* NULL => Mie (не поддерживается в C-версии) */
    double ref_wl_um;
    double cback_scale;
} channel_t;

typedef struct {
    channel_t *ch;
    int n;
    double cap;            /* min x_max*wl/(2*pi) по каналам */
} channels_t;

void channels_free(channels_t *c);
void load_table_channels(const char *root, double mre, double mim,
                         const channel_t *base, int nch,
                         int extrapolate, double ref_wl, double cback_scale,
                         channels_t *out);

typedef struct {
    int n;
    double *r;
    double r_min, r_max;
    int nch;
    spline_t *sp;          /* ln K от ln r, по каналу */
} ktable_t;

void ktable_build(ktable_t *kt, const channels_t *chs, double r_min,
                  double r_max, double dx_max);
void ktable_free(ktable_t *kt);
double ktable_kernel(const ktable_t *kt, int j, double r);
double ktable_kernels_at(const ktable_t *kt, int j, double r);

/* ------------------------------------------------------------------ */
/* базис, квадратуры, геометрия окна                                   */
/* ------------------------------------------------------------------ */
void gl_nodes_weights(int order, double *x, double *w);

void quadrature_grid(const double *breaks, int nb, int order,
                     double max_log_span, double **nodes, double **weights,
                     int *out_n);

/* значения всех B-сплайнов степени p на векторе узлов U (nk узлов),
 * nbf = nk - p - 1 функций; out должен иметь размер nbf */
void bspline_values(const double *U, int nk, int p, double t, double *out);
/* значения производной порядка `deriv` */
void bspline_deriv_values(const double *U, int nk, int p, int deriv, double t,
                          double *out);

/* локальные значения производной: p+1 чисел для индексов span-p .. span;
 * возвращает span через span_out. O(p) на точку, без полных массивов. */
void bspline_deriv_window(const double *U, int nk, int p, int deriv, double t,
                          double *out, int *span_out);

typedef struct {
    int degree, n_basis, n_unknown, interior_only;
    double r0, r1;
    int n_knots, n_nodes;
    double *knots, *nodes, *weights;
    double *values;   /* n_nodes x n_unknown */
    double *dvol;     /* n_nodes */
} geom_t;

void geom_build(geom_t *g, double r0, double r1, int degree, int n_basis,
                int interior_only, int order, double max_log_span);
void geom_free(geom_t *g);

/* ------------------------------------------------------------------ */
/* матрицы A и решения                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    double m_re, m_im;
    int degree;
    double r0, r1;
    double gamma;
    double *C;        /* n_unknown */
    int nu;           /* число коэффициентов C */
    int geom_index;
    double err;
    double *g_fit;    /* nch */
    int id;           /* исходный номер решения (для переранжирования) */
} solution_t;

typedef struct {
    int n_m, n_w, nch, nu;
    double *A;        /* [n_m][n_w][nch][nu] */
    double *s0;       /* [n_m][n_w] */
    int *degree;
    double *r0, *r1;
    double *m_re, *m_im;
    geom_t *geom;
    int zero_boundary;
    int order;
    double max_log_span;
    int n_basis;
} design_t;

void design_build(design_t *d, const channel_t *ch, int nch,
                  const ktable_t *kts, int n_m,
                  const double *m_re, const double *m_im,
                  const int *degrees, int n_degrees,
                  const double *r0_grid, int nr0,
                  const double *r1_grid, int nr1,
                  double min_ratio, int n_basis, int zero_boundary,
                  int order, double max_log_span);
void design_free(design_t *d);

typedef struct {
    solution_t *sol;
    int n;
    double scale;
} solutions_t;

enum { GAMMA_FIXED = 0, GAMMA_CV = 1, GAMMA_ERR = 2 };

void scan_variants(const design_t *d, const double *g1, int nch,
                   int gamma_select, double gamma_rel_fixed,
                   double gamma_exp_min, double gamma_exp_max, int n_gamma,
                   int nonneg, int relative, solutions_t *out);
void solutions_free(solutions_t *s);
void sort_solutions(solution_t *sol, int n);

/* ------------------------------------------------------------------ */
/* ансамбль и сглаживание                                              */
/* ------------------------------------------------------------------ */
double solution_distribution(const solution_t *s, const geom_t *g, double r);

void ensemble_average(const solutions_t *sols, int n_best,
                      const geom_t *geom, const double *r_grid, int nr,
                      int smooth, double lam_rel,
                      double *n_avg, double **n_best_mat, double *m_avg_re,
                      double *m_avg_im);
void smooth_distribution(const double *r_grid, int n, double *n_avg,
                         const double *w, double lam_rel);

/* Сглаживающий сплайн (scipy make_smoothing_spline):
 *   min sum_i w_i (y_i - f_i)^2 + lam * int (f'')^2 dx
 * на базисе кубических B-сплайнов с узлами в x. x должны возрастать,
 * xeval — точки вычисления. w = NULL — единичные веса. */
void smoothing_spline_fit(const double *x, const double *y, const double *w,
                          int n, double lam, const double *xeval, int neval,
                          double *out);

/* ------------------------------------------------------------------ */
/* итоговый профиль одной высоты (общий для одиночного и пакетного)     */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* интегральные характеристики и моды                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    double N_cm3;
    double vol_mean_um3;
    double vol_conc_um3_per_cm3;
    double r_eff_um;
    double r_cov_min_um, r_cov_max_um;
    double cov_frac;
} moments_t;

moments_t size_moments(const double *n_avg, const double *r_grid, int n);

typedef struct {
    double V, rm, sigma;
    int ok;
    const char *method;
} cs_mode_t;

typedef struct {
    cs_mode_t fine, coarse, overall;
} modes_t;

modes_t mode_analysis(const double *r_grid, const double *f_avg, int n,
                      double r_split, const double *w, int method);

/* демо */
double lognormal_volume(double r, double V, double rg, double sigma);
double demo_true_distribution(double r);

/* ------------------------------------------------------------------ */
/* JSON                                                                */
/* ------------------------------------------------------------------ */
typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } jtype_t;

typedef struct jval {
    jtype_t t;
    double num;
    int bval;
    char *str;
    struct jval **items;
    int n;
    char **keys;
    struct jval **vals;
    int nk;
} jval_t;

jval_t *json_parse(const char *text);
void json_free(jval_t *v);
jval_t *json_get(const jval_t *obj, const char *key);

/* ------------------------------------------------------------------ */
/* аргументы CLI                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *data;
    const char *tables;
    int table_extrapolate;
    double table_ref_wl;
    int table_cback_4pi;

    int degrees[8];
    int n_degrees;
    int n_basis;               /* 0 => авто */
    double r0_min, r0_max; int nr0;
    double r1_min, r1_max; int nr1;
    double min_ratio;
    int best;
    double gamma_min, gamma_max; int ngamma;
    int gamma_select;
    double gamma_rel;
    int nonneg;
    int no_zero_boundary;
    int no_smooth;
    double smooth_lam;
    double r_split;
    int modes;                 /* 0 auto, 1 fit, 2 stat */
    int absolute;
    double dx_max;
    int nr_out;
    int quad_order;
    double log_span;

    int has_m_real, has_m_imag;
    double m_real, m_imag;
    double *m_reals, *m_imags;
    int n_m_reals, n_m_imags;

    double noise;
    int seed;
    const char *out;
    const char *json;
    int no_plot;

    const char *batch;
    const char *batch_out_dir;
    int lidar_ratio_cons;
    double lidar_ratio_w;
    int lidar_ratio_iter;
} args_t;

void parse_args(int argc, char **argv, args_t *a);
void args_free(args_t *a);

/* ------------------------------------------------------------------ */
/* итоговый профиль одной высоты (общий для одиночного и пакетного)     */
/* ------------------------------------------------------------------ */
typedef struct {
    int nr;
    double *r_grid;
    double *n_avg;
    double *n_avg_raw;
    double *n_best;      /* n_best_count x nr */
    int n_best_count;
    double m_re, m_im;
    double *g_fit;
    moments_t moments;
    modes_t modes;
    double *count;
} profile_t;

void profile_finalize(profile_t *p, const args_t *a,
                      const channel_t *ch, int nch,
                      const ktable_t *kts, const double *mre,
                      const double *mim, int nm,
                      const solutions_t *sols, const geom_t *geom,
                      double r_min, double r_max);
void profile_free(profile_t *p);

/* ------------------------------------------------------------------ */
/* вывод                                                               */
/* ------------------------------------------------------------------ */
void report_summary(const args_t *a, const solutions_t *sols,
                    const double *g1, const channel_t *ch, int nch,
                    const double *n_avg, const double *g_fit,
                    const double *r_grid, int nr, double m_avg_re,
                    double m_avg_im);
void report_moments(const moments_t *m);
void report_modes(const modes_t *m, double r_split, const char *method);

void write_json(const char *path, const solutions_t *sols,
                const double *r_grid, int nr,
                const double *n_avg, const double *n_avg_raw,
                const channel_t *ch, int nch, const double *g1,
                const double *g_fit, double m_avg_re, double m_avg_im,
                const moments_t *mom, const modes_t *modes);

int plot_results(const char *out_path, const double *r_grid, int nr,
                 const double *n_true, const double *n_avg,
                 const double *n_avg_raw, const double *n_best,
                 int n_best_count, const channel_t *ch, int nch,
                 const double *g1, const double *g_fit,
                 double m_avg_re, double m_avg_im);

/* Компактный график одной высоты пакетного режима (2 панели). */
int plot_profile(const char *out_path, const double *r_grid, int nr,
                 const double *n_avg, const channel_t *ch, int nch,
                 const double *g1, const double *g_fit, double h);

/* Сводный вертикальный профиль показателя преломления по высотам:
 * одна панель, две оси X (низ — Re(m), верх — Im(m)), высота — по Y. */
int plot_refractive_profile(const char *out_path, const double *alt,
                            const double *m_re, const double *m_im, int n);

/* Сводный вертикальный профиль произвольных величин: ось Y — высота,
 * ось X — значения (несколько серий). Логарифмическая шкала X
 * включается автоматически, если все значения положительны и их
 * размах больше 30x. NaN-значения пропускаются. */
int plot_profiles(const char *out_path, const char *title,
                  const char *xlabel, int n_series, const double *alt,
                  const double *const *vals, const char *const *names, int n);

/* ------------------------------------------------------------------ */
/* пакетная обработка (--batch)                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    ch_kind_t kind;
    double wl_um;
    char name[32];
} batch_channel_t;

typedef struct {
    batch_channel_t *ch;
    int nch;
    double *alt;
    double *data;    /* n_alt x nch, row-major */
    int n_alt;
} batch_input_t;

void parse_batch_txt(const char *path, batch_input_t *b);
void batch_input_free(batch_input_t *b);
void run_batch(const args_t *a);

#endif /* CSOLVE_H */
