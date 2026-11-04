/* График: два панно (распределение и согласие по каналам) через gnuplot. */
#define _POSIX_C_SOURCE 200809L

#include "csolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *plot_format_ext(int fmt)
{
    switch (fmt) {
    case PLOT_FMT_PDF:
        return "pdf";
    case PLOT_FMT_SVG:
        return "svg";
    default:
        return "png";
    }
}

/* Терминал gnuplot под выбранный формат. Размеры фигуры задаются
 * в дюймах; для png/svg они переводятся в пиксели через dpi,
 * для pdf передаются как есть (вектор). */
static void set_terminal(FILE *f, const plot_opts_t *po, double w_in,
                         double h_in)
{
    double dpi = po && po->dpi > 0.0 ? po->dpi : 150.0;
    int fmt = po ? po->fmt : PLOT_FMT_PNG;
    int wp = (int)(w_in * dpi + 0.5);
    int hp = (int)(h_in * dpi + 0.5);
    if (wp < 1)
        wp = 1;
    if (hp < 1)
        hp = 1;
    switch (fmt) {
    case PLOT_FMT_PDF:
        fprintf(f, "set terminal pdfcairo size %.3f,%.3f enhanced "
                   "font 'DejaVu Sans,10'\n", w_in, h_in);
        break;
    case PLOT_FMT_SVG:
        fprintf(f, "set terminal svg size %d,%d enhanced "
                   "font 'DejaVu Sans,10'\n", wp, hp);
        break;
    default:
        fprintf(f, "set terminal pngcairo size %d,%d enhanced "
                   "font 'DejaVu Sans,10'\n", wp, hp);
        break;
    }
}

static void write_avg(const char *path, const double *r, const double *avg,
                      const double *raw, int n)
{
    FILE *f = fopen(path, "w");
    if (!f)
        cs_fatal("не удалось записать %s", path);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%.17g ", r[i]);
        if (isfinite(avg[i]))
            fprintf(f, "%.17g", avg[i] / CM3_PER_M3);
        else
            fputs("NaN", f);
        fputc(' ', f);
        if (isfinite(raw[i]))
            fprintf(f, "%.17g", raw[i] / CM3_PER_M3);
        else
            fputs("NaN", f);
        fputc('\n', f);
    }
    fclose(f);
}

static void write_true(const char *path, const double *r, const double *v, int n)
{
    FILE *f = fopen(path, "w");
    if (!f)
        cs_fatal("не удалось записать %s", path);
    for (int i = 0; i < n; i++)
        fprintf(f, "%.17g %.17g\n", r[i], v[i] / CM3_PER_M3);
    fclose(f);
}

static void write_sol(const char *path, const double *r, const double *v, int n)
{
    FILE *f = fopen(path, "w");
    if (!f)
        cs_fatal("не удалось записать %s", path);
    for (int i = 0; i < n; i++)
        fprintf(f, "%.17g %.17g\n", r[i], v[i] / CM3_PER_M3);
    fclose(f);
}

static void write_bars(const char *path, const channel_t *ch, int nch,
                       const double *g1, const double *g_fit)
{
    FILE *f = fopen(path, "w");
    if (!f)
        cs_fatal("не удалось записать %s", path);
    for (int j = 0; j < nch; j++)
        fprintf(f, "%c%g %.17g %.17g\n",
                ch[j].kind == CH_BETA ? 'b' : 'a', ch[j].wl_um * 1e3,
                g1[j], g_fit[j]);
    fclose(f);
}

int plot_profile(const plot_opts_t *po, const char *out_path,
                 const double *r_grid, int nr, const double *n_avg,
                 const channel_t *ch, int nch, const double *g1,
                 const double *g_fit, double h)
{
    char tmpl[256];
    int mk = 0;
    for (int attempt = 0; attempt < 100; attempt++) {
        snprintf(tmpl, sizeof(tmpl), "/tmp/csolve_%ld_%d", (long)getpid(),
                 attempt);
        if (mkdir(tmpl, 0700) == 0) {
            mk = 1;
            break;
        }
    }
    if (!mk) {
        fprintf(stderr, "[предупреждение] не удалось создать временный каталог "
                        "для графика\n");
        return 0;
    }
    char *dir = tmpl;
    char *p_prof = xasprintf("%s/prof.dat", dir);
    char *p_bars = xasprintf("%s/bars.dat", dir);
    char *p_script = xasprintf("%s/plot.gp", dir);

    FILE *fd = fopen(p_prof, "w");
    if (!fd)
        cs_fatal("не удалось записать %s", p_prof);
    for (int i = 0; i < nr; i++) {
        fprintf(fd, "%.17g ", r_grid[i]);
        double v = n_avg[i] / CM3_PER_M3;
        if (isnan(v))
            fputs("NaN", fd);
        else
            fprintf(fd, "%.17g", v > 1e-30 ? v : 1e-30);
        fputc('\n', fd);
    }
    fclose(fd);
    write_bars(p_bars, ch, nch, g1, g_fit);

    FILE *f = fopen(p_script, "w");
    if (!f) {
        fprintf(stderr, "[предупреждение] не удалось записать скрипт gnuplot\n");
        free(p_prof); free(p_bars); free(p_script);
        return 0;
    }
    fputs("set encoding utf8\n", f);
    set_terminal(f, po, 8.0, 3.0667);
    fprintf(f, "set output \"%s\"\n", out_path);
    fputs("set datafile missing 'NaN'\n", f);
    fprintf(f, "set title 'H = %g м'\n", h);
    fputs("set multiplot layout 1,2\n", f);
    fputs("set logscale x\n", f);
    fputs("set xlabel 'r, мкм'\n", f);
    fputs("set ylabel 'dV/dln r, мкм³/см³'\n", f);
    fputs("set grid\n", f);
    fprintf(f, "plot \"%s\" using 1:2 with lines lw 2 lc rgb '#1f77b4' "
               "notitle\n", p_prof);
    fputs("unset logscale x\n", f);
    fputs("set xlabel ''\n", f);
    fputs("set ylabel 'коэффициент'\n", f);
    fputs("set style data histograms\n", f);
    fputs("set style fill solid 0.7 border -1\n", f);
    fputs("set boxwidth 0.4\n", f);
    fputs("set xtics rotate by 20 right\n", f);
    fprintf(f, "plot \"%s\" using 2:xtic(1) title 'измеренные g1', "
               "'' using 3 title 'расчётные g'\n", p_bars);
    fputs("unset multiplot\n", f);
    fclose(f);

    char *cmd = xasprintf("gnuplot \"%s\"", p_script);
    int rc = system(cmd);
    free(cmd);
    int ok = (rc == 0) && access(out_path, R_OK) == 0;
    if (ok) {
        unlink(p_prof);
        unlink(p_bars);
        unlink(p_script);
        rmdir(dir);
    } else {
        fprintf(stderr, "[предупреждение] не удалось построить график %s "
                        "(скрипт: %s)\n", out_path, p_script);
    }
    free(p_prof);
    free(p_bars);
    free(p_script);
    return ok;
}

int plot_refractive_profile(const plot_opts_t *po, const char *out_path,
                            const double *alt, const double *m_re,
                            const double *m_im, int n)
{
    if (n <= 0)
        return 0;
    char tmpl[256];
    int mk = 0;
    for (int attempt = 0; attempt < 100; attempt++) {
        snprintf(tmpl, sizeof(tmpl), "/tmp/csolve_%ld_%d", (long)getpid(),
                 attempt);
        if (mkdir(tmpl, 0700) == 0) {
            mk = 1;
            break;
        }
    }
    if (!mk) {
        fprintf(stderr, "[предупреждение] не удалось создать временный каталог "
                        "для графика\n");
        return 0;
    }
    char *dir = tmpl;
    char *p_data = xasprintf("%s/refr.dat", dir);
    char *p_script = xasprintf("%s/plot.gp", dir);

    /* данные в порядке возрастания высоты */
    int *idx = int_array(n);
    for (int i = 0; i < n; i++)
        idx[i] = i;
    for (int i = 1; i < n; i++) {
        int key = idx[i], j = i - 1;
        while (j >= 0 && alt[idx[j]] > alt[key]) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    double re_lo = m_re[idx[0]], re_hi = re_lo;
    double im_lo = m_im[idx[0]], im_hi = im_lo;
    double h_lo = alt[idx[0]], h_hi = h_lo;
    for (int k = 1; k < n; k++) {
        int i = idx[k];
        if (m_re[i] < re_lo) re_lo = m_re[i];
        if (m_re[i] > re_hi) re_hi = m_re[i];
        if (m_im[i] < im_lo) im_lo = m_im[i];
        if (m_im[i] > im_hi) im_hi = m_im[i];
        if (alt[i] < h_lo) h_lo = alt[i];
        if (alt[i] > h_hi) h_hi = alt[i];
    }
    double re_mag = fmax(fabs(re_lo), fabs(re_hi));
    double im_mag = fmax(fabs(im_lo), fabs(im_hi));
    double re_pad = 0.06 * (re_hi - re_lo);
    if (0.02 * re_mag > re_pad)
        re_pad = 0.02 * re_mag;
    if (re_pad <= 0.0)
        re_pad = 0.01;
    double im_pad = 0.06 * (im_hi - im_lo);
    if (0.02 * im_mag > im_pad)
        im_pad = 0.02 * im_mag;
    if (im_pad <= 0.0)
        im_pad = 0.001;
    double h_mag = fmax(fabs(h_lo), fabs(h_hi));
    double h_pad = 0.04 * (h_hi - h_lo);
    if (0.02 * h_mag > h_pad)
        h_pad = 0.02 * h_mag;
    if (h_pad <= 0.0)
        h_pad = 10.0;

    FILE *fd = fopen(p_data, "w");
    if (!fd)
        cs_fatal("не удалось записать %s", p_data);
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        fprintf(fd, "%.17g %.17g %.17g\n", alt[i], m_re[i], m_im[i]);
    }
    fclose(fd);
    free(idx);

    FILE *f = fopen(p_script, "w");
    if (!f) {
        fprintf(stderr, "[предупреждение] не удалось записать скрипт gnuplot\n");
        free(p_data);
        free(p_script);
        return 0;
    }
    fputs("set encoding utf8\n", f);
    set_terminal(f, po, 6.0, 4.8);
    fprintf(f, "set output \"%s\"\n", out_path);
    fputs("set datafile missing 'NaN'\n", f);
    fputs("set xlabel 'Re(m)'\n", f);
    fputs("set x2label 'Im(m)'\n", f);
    fputs("set ylabel 'Высота H, м'\n", f);
    fputs("set grid\n", f);
    fputs("set xtics nomirror\n", f);
    fputs("set x2tics\n", f);
    fputs("set ytics nomirror\n", f);
    fputs("set key top right\n", f);
    fprintf(f, "set xrange [%.10g:%.10g]\n", re_lo - re_pad, re_hi + re_pad);
    fprintf(f, "set x2range [%.10g:%.10g]\n", im_lo - im_pad, im_hi + im_pad);
    fprintf(f, "set yrange [%.10g:%.10g]\n", h_lo - h_pad, h_hi + h_pad);
    fprintf(f, "set title 'Показатель преломления по высотам'\n");
    fprintf(f, "plot \"%s\" using 2:1 with linespoints axes x1y1 "
               "lw 1.8 pt 7 ps 1.1 lc rgb '#1f77b4' title 'Re(m)', \\\n"
               "     \"%s\" using 3:1 with linespoints axes x2y1 "
               "lw 1.8 pt 5 ps 1.1 lc rgb '#d62728' title 'Im(m)'\n",
            p_data, p_data);
    fclose(f);

    char *cmd = xasprintf("gnuplot \"%s\"", p_script);
    int rc = system(cmd);
    free(cmd);
    int ok = (rc == 0) && access(out_path, R_OK) == 0;
    if (ok) {
        unlink(p_data);
        unlink(p_script);
        rmdir(dir);
    } else {
        fprintf(stderr, "[предупреждение] не удалось построить график %s "
                        "(скрипт: %s)\n", out_path, p_script);
    }
    free(p_data);
    free(p_script);
    return ok;
}

int plot_profiles(const plot_opts_t *po, const char *out_path,
                  const char *title, const char *xlabel, int n_series,
                  const double *alt, const double *const *vals,
                  const char *const *names, int n)
{
    if (n <= 0 || n_series <= 0)
        return 0;
    char tmpl[256];
    int mk = 0;
    for (int attempt = 0; attempt < 100; attempt++) {
        snprintf(tmpl, sizeof(tmpl), "/tmp/csolve_%ld_%d", (long)getpid(),
                 attempt);
        if (mkdir(tmpl, 0700) == 0) {
            mk = 1;
            break;
        }
    }
    if (!mk) {
        fprintf(stderr, "[предупреждение] не удалось создать временный каталог "
                        "для графика\n");
        return 0;
    }
    char *dir = tmpl;
    char *p_data = xasprintf("%s/prof.dat", dir);
    char *p_script = xasprintf("%s/plot.gp", dir);

    FILE *fd = fopen(p_data, "w");
    if (!fd)
        cs_fatal("не удалось записать %s", p_data);
    for (int i = 0; i < n; i++) {
        fprintf(fd, "%.17g", alt[i]);
        for (int s = 0; s < n_series; s++) {
            double v = vals[s][i];
            fputc(' ', fd);
            if (isfinite(v))
                fprintf(fd, "%.17g", v);
            else
                fputs("NaN", fd);
        }
        fputc('\n', fd);
    }
    fclose(fd);

    /* диапазоны */
    double lo = INFINITY, hi = -INFINITY, h_lo = INFINITY, h_hi = -INFINITY;
    int any = 0;
    for (int s = 0; s < n_series; s++)
        for (int i = 0; i < n; i++) {
            double v = vals[s][i];
            if (!isfinite(v))
                continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            any = 1;
        }
    for (int i = 0; i < n; i++) {
        if (!isfinite(alt[i]))
            continue;
        if (alt[i] < h_lo) h_lo = alt[i];
        if (alt[i] > h_hi) h_hi = alt[i];
    }
    if (!any) {
        lo = 0.0;
        hi = 1.0;
    }
    int use_log = (lo > 0.0 && hi > 0.0 && hi / lo > 30.0);
    double mag = fmax(fabs(lo), fabs(hi));
    if (use_log) {
        lo /= 1.3;
        hi *= 1.3;
    } else if (hi > lo) {
        double pad = 0.08 * (hi - lo);
        if (0.05 * mag > pad)
            pad = 0.05 * mag;
        lo -= pad;
        hi += pad;
    } else {
        double pad = mag > 0.0 ? 0.05 * mag : 0.5;
        lo -= pad;
        hi += pad;
    }
    if (!(hi > lo)) {
        lo -= 0.5;
        hi += 0.5;
    }
    double h_mag = fmax(fabs(h_lo), fabs(h_hi));
    double h_pad = 0.04 * (h_hi - h_lo);
    if (0.02 * h_mag > h_pad)
        h_pad = 0.02 * h_mag;
    if (h_pad <= 0.0)
        h_pad = 10.0;
    h_lo -= h_pad;
    h_hi += h_pad;

    static const char *colors[6] = {"'#1f77b4'", "'#d62728'", "'#2ca02c'",
                                    "'#9467bd'", "'#ff7f0e'", "'#8c564b'"};
    static const int points[6] = {7, 5, 9, 11, 13, 15};

    FILE *f = fopen(p_script, "w");
    if (!f) {
        fprintf(stderr, "[предупреждение] не удалось записать скрипт gnuplot\n");
        free(p_data);
        free(p_script);
        return 0;
    }
    fputs("set encoding utf8\n", f);
    set_terminal(f, po, 6.0, 4.8);
    fprintf(f, "set output \"%s\"\n", out_path);
    fputs("set datafile missing 'NaN'\n", f);
    fprintf(f, "set title '%s'\n", title);
    fprintf(f, "set xlabel '%s'\n", xlabel);
    fputs("set ylabel 'Высота H, м'\n", f);
    fputs("set grid\n", f);
    fputs("set key top right\n", f);
    if (use_log)
        fputs("set logscale x\n", f);
    else
        fputs("unset logscale x\n", f);
    fprintf(f, "set xrange [%.10g:%.10g]\n", lo, hi);
    fprintf(f, "set yrange [%.10g:%.10g]\n", h_lo, h_hi);
    fputs("plot ", f);
    for (int s = 0; s < n_series; s++) {
        fprintf(f, "%s\"%s\" using %d:1 with linespoints lw 1.8 pt %d "
                   "ps 1.1 lc rgb %s title '%s'",
                s ? ", \\\n     " : "", p_data, s + 2, points[s % 6],
                colors[s % 6], names[s]);
    }
    fputc('\n', f);
    fclose(f);

    char *cmd = xasprintf("gnuplot \"%s\"", p_script);
    int rc = system(cmd);
    free(cmd);
    int ok = (rc == 0) && access(out_path, R_OK) == 0;
    if (ok) {
        unlink(p_data);
        unlink(p_script);
        rmdir(dir);
    } else {
        fprintf(stderr, "[предупреждение] не удалось построить график %s "
                        "(скрипт: %s)\n", out_path, p_script);
    }
    free(p_data);
    free(p_script);
    return ok;
}

int plot_results(const plot_opts_t *po, const char *out_path,
                 const double *r_grid, int nr,
                 const double *n_true, const double *n_avg,
                 const double *n_avg_raw, const double *n_best,
                 int n_best_count, const channel_t *ch, int nch,
                 const double *g1, const double *g_fit,
                 double m_avg_re, double m_avg_im)
{
    char tmpl[256];
    int mk = 0;
    for (int attempt = 0; attempt < 100; attempt++) {
        snprintf(tmpl, sizeof(tmpl), "/tmp/csolve_%ld_%d", (long)getpid(),
                 attempt);
        if (mkdir(tmpl, 0700) == 0) {
            mk = 1;
            break;
        }
    }
    if (!mk) {
        fprintf(stderr, "[предупреждение] не удалось создать временный каталог "
                        "для графика\n");
        return 0;
    }
    char *dir = tmpl;
    char *p_avg = xasprintf("%s/avg.dat", dir);
    char *p_true = xasprintf("%s/true.dat", dir);
    char *p_bars = xasprintf("%s/bars.dat", dir);
    char *p_script = xasprintf("%s/plot.gp", dir);
    write_avg(p_avg, r_grid, n_avg, n_avg_raw, nr);
    if (n_true)
        write_true(p_true, r_grid, n_true, nr);
    write_bars(p_bars, ch, nch, g1, g_fit);

    char *sols_files[64];
    int ns = n_best_count < 64 ? n_best_count : 64;
    for (int i = 0; i < ns; i++) {
        sols_files[i] = xasprintf("%s/sol_%d.dat", dir, i);
        write_sol(sols_files[i], r_grid, n_best + (size_t)i * nr, nr);
    }

    FILE *f = fopen(p_script, "w");
    if (!f) {
        fprintf(stderr, "[предупреждение] не удалось записать скрипт gnuplot\n");
        return 0;
    }
    fputs("set encoding utf8\n", f);
    set_terminal(f, po, 8.6667, 3.4667);
    fprintf(f, "set output \"%s\"\n", out_path);
    fputs("set datafile missing 'NaN'\n", f);
    fputs("set multiplot layout 1,2\n", f);
    fputs("set logscale x\n", f);
    fputs("set xlabel 'r, мкм'\n", f);
    fputs("set ylabel 'dV/dln r, мкм³/см³'\n", f);
    fprintf(f, "set title 'Объёмное распределение частиц по размерам, "
               "⟨m⟩ = %.3f%+.3fj'\n", m_avg_re, m_avg_im);
    fputs("set grid\n", f);
    fputs("set key top right\n", f);
    fputs("plot ", f);
    int first = 1;
    if (n_true) {
        fprintf(f, "\"%s\" using 1:2 with lines dt 2 lw 1.6 "
                   "lc rgb 'black' title 'истинное n(r)'", p_true);
        first = 0;
    }
    for (int i = 0; i < ns; i++) {
        fprintf(f, "%s\"%s\" using 1:2 with lines lw 1 lc rgb '#bbbbbb' notitle",
                first ? "" : ", \\\n     ", sols_files[i]);
        first = 0;
    }
    fprintf(f, "%s\"%s\" using 1:3 with lines lw 1 dt 3 "
               "lc rgb 'dark-orange' title 'среднее (несглаженное)'",
            first ? "" : ", \\\n     ", p_avg);
    fprintf(f, ", \\\n     \"%s\" using 1:2 with lines lw 2.6 "
               "lc rgb '#1f77b4' title 'среднее по лучшим'\n", p_avg);
    fputs("unset logscale x\n", f);
    fputs("set xlabel ''\n", f);
    fputs("set ylabel 'коэффициент, отн. ед.'\n", f);
    fputs("set title 'Согласие измеренных и расчётных коэффициентов'\n", f);
    fputs("set style data histograms\n", f);
    fputs("set style fill solid 0.7 border -1\n", f);
    fputs("set boxwidth 0.4\n", f);
    fputs("set xtics rotate by 20 right\n", f);
    fprintf(f, "plot \"%s\" using 2:xtic(1) title 'измеренные g1', "
               "'' using 3 title 'расчётные g (усредн.)'\n", p_bars);
    fputs("unset multiplot\n", f);
    fclose(f);

    char *cmd = xasprintf("gnuplot \"%s\"", p_script);
    int rc = system(cmd);
    free(cmd);
    int ok = (rc == 0) && access(out_path, R_OK) == 0;

    if (!ok) {
        for (int i = 0; i < ns; i++)
            free(sols_files[i]);
        fprintf(stderr, "[предупреждение] не удалось построить график "
                        "(скрипт оставлен для диагностики: %s)\n", p_script);
        free(p_avg); free(p_true); free(p_bars); free(p_script);
        return 0;
    }
    for (int i = 0; i < ns; i++) {
        unlink(sols_files[i]);
        free(sols_files[i]);
    }
    unlink(p_avg);
    unlink(p_true);
    unlink(p_bars);
    unlink(p_script);
    rmdir(dir);
    free(p_avg);
    free(p_true);
    free(p_bars);
    free(p_script);
    printf("график сохранён: %s\n", out_path);
    return 1;
}
