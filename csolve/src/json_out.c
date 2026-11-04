/* Запись результатов в JSON (структура совпадает с Python-версией). */
#include "csolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void wnum(FILE *f, double v)
{
    if (isfinite(v))
        fprintf(f, "%.17g", v);
    else
        fputs("null", f);
}

static void wnum_cm(FILE *f, double v)
{
    wnum(f, v / CM3_PER_M3);
}

void write_json(const char *path, const solutions_t *sols,
                const double *r_grid, int nr,
                const double *n_avg, const double *n_avg_raw,
                const channel_t *ch, int nch, const double *g1,
                const double *g_fit, double m_avg_re, double m_avg_im,
                const moments_t *mom, const modes_t *modes)
{
    FILE *f = fopen(path, "w");
    if (!f)
        cs_fatal("не удалось открыть файл для записи: %s", path);

    fputs("{\n  \"m_avg\": {\"real\": ", f);
    wnum(f, m_avg_re);
    fputs(", \"imag\": ", f);
    wnum(f, m_avg_im);
    fputs("},\n  \"moments\": {\n", f);
    fputs("    \"N_cm-3\": ", f); wnum(f, mom->N_cm3);
    fputs(",\n    \"vol_mean_um3\": ", f); wnum(f, mom->vol_mean_um3);
    fputs(",\n    \"vol_conc_um3_per_cm3\": ", f);
    wnum(f, mom->vol_conc_um3_per_cm3);
    fputs(",\n    \"r_eff_um\": ", f); wnum(f, mom->r_eff_um);
    fputs(",\n    \"r_cov_min_um\": ", f); wnum(f, mom->r_cov_min_um);
    fputs(",\n    \"r_cov_max_um\": ", f); wnum(f, mom->r_cov_max_um);
    fputs(",\n    \"cov_frac\": ", f); wnum(f, mom->cov_frac);
    fputs("\n  },\n  \"units_input\": {\"beta\": \"1/(м*ср)\", "
          "\"alpha\": \"1/м\"},\n  \"channels\": [\n", f);
    for (int j = 0; j < nch; j++) {
        fprintf(f, "    {\"kind\": \"%s\", \"wl_nm\": ",
                ch[j].kind == CH_BETA ? "beta" : "alpha");
        wnum(f, ch[j].wl_um * 1e3);
        fputs(", \"g1\": ", f);
        wnum(f, g1[j]);
        fputs(", \"g_fit\": ", f);
        wnum(f, g_fit[j]);
        fputs(j + 1 < nch ? "},\n" : "}\n", f);
    }
    fputs("  ],\n  \"r_um\": [", f);
    for (int i = 0; i < nr; i++) {
        if (i)
            fputc(',', f);
        wnum(f, r_grid[i]);
    }
    fputs("],\n  \"n_avg\": [", f);
    for (int i = 0; i < nr; i++) {
        if (i)
            fputc(',', f);
        wnum_cm(f, n_avg[i]);
    }
    fputs("],\n  \"best_solutions\": [\n", f);
    for (int t = 0; t < sols->n; t++) {
        const solution_t *s = &sols->sol[t];
        fprintf(f, "    {\"rank\": %d, \"m_real\": ", t + 1);
        wnum(f, s->m_re);
        fputs(", \"m_imag\": ", f);
        wnum(f, s->m_im);
        fprintf(f, ", \"degree\": %d, \"r0\": ", s->degree);
        wnum(f, s->r0);
        fputs(", \"r1\": ", f);
        wnum(f, s->r1);
        fputs(", \"gamma\": ", f);
        wnum(f, s->gamma);
        fputs(", \"err\": ", f);
        wnum(f, s->err);
        fputs(", \"C\": [", f);
        int nu = s->nu;
        for (int i = 0; i < nu; i++) {
            if (i)
                fputc(',', f);
            wnum(f, s->C[i]);
        }
        fputs("]}", f);
        if (t + 1 < sols->n)
            fputc(',', f);
        fputc('\n', f);
    }
    fputs("  ],\n  \"n_avg_raw\": [", f);
    for (int i = 0; i < nr; i++) {
        if (i)
            fputc(',', f);
        wnum_cm(f, n_avg_raw[i]);
    }
    fputs("],\n  \"modes\": {\n", f);
    const char *names[3] = {"fine", "coarse", "overall"};
    const cs_mode_t *vals[3] = {&modes->fine, &modes->coarse, &modes->overall};
    for (int k = 0; k < 3; k++) {
        const cs_mode_t *v = vals[k];
        fprintf(f, "    \"%s\": {\"V_um3_per_cm3\": ", names[k]);
        if (isfinite(v->V))
            wnum_cm(f, v->V);
        else
            fputs("null", f);
        fputs(", \"rm_um\": ", f);
        wnum(f, v->rm);
        fputs(", \"sigma\": ", f);
        wnum(f, v->sigma);
        fprintf(f, ", \"ok\": %s, \"method\": \"%s\"}%s\n",
                v->ok ? "true" : "false", v->method,
                k < 2 ? "," : "");
    }
    fputs("  }\n}\n", f);
    fclose(f);
}
