/* Таблица ядер на сетке r и её интерполяция (ln K от ln r).
 *
 * Шаг сетки по r выбирается так, чтобы шаг параметра размера x для самой
 * короткой длины волны не превосходил dx_max: dr = dx_max*lam_min/(2*pi).
 */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

void ktable_build(ktable_t *kt, const channels_t *chs, double r_min,
                  double r_max, double dx_max)
{
    memset(kt, 0, sizeof(*kt));
    int nch = chs->n;
    if (nch <= 0)
        cs_fatal("нет ни одного канала");
    double lam_min = chs->ch[0].wl_um;
    for (int j = 1; j < nch; j++)
        if (chs->ch[j].wl_um < lam_min)
            lam_min = chs->ch[j].wl_um;
    double dr = dx_max * lam_min / (2.0 * CS_PI);
    if (!(dr > 0.0))
        cs_fatal("некорректный шаг сетки r (dx-max = %g)", dx_max);

    int n = (int)ceil((r_max - r_min) / dr);
    if (n < 0)
        n = 0;
    double *r = dbl_array(n + 1);
    for (int i = 0; i < n; i++)
        r[i] = r_min + (double)i * dr;
    int nr = n;
    if (nr == 0 || r[nr - 1] < r_max - 0.5 * dr)
        r[nr++] = r_max;

    kt->n = nr;
    kt->r = r;
    kt->r_min = r_min;
    kt->r_max = r_max;
    kt->nch = nch;
    kt->sp = xcalloc((size_t)nch, sizeof(spline_t));

    double *lr = dbl_array(nr);
    double *val = dbl_array(nr);
    for (int i = 0; i < nr; i++)
        lr[i] = log(r[i]);
    for (int j = 0; j < nch; j++) {
        const channel_t *ch = &chs->ch[j];
        if (!ch->tab)
            cs_fatal("Mie-режим не поддерживается C-версией — задайте --tables");
        double scale = UM2_TO_M2 * pow(ch->wl_um / ch->ref_wl_um, 2.0);
        for (int i = 0; i < nr; i++) {
            double x = 2.0 * CS_PI * r[i] / ch->wl_um;
            double C = table_cross_section(ch->tab, ch->kind == CH_BETA, x);
            double K = scale * C;
            if (ch->kind == CH_BETA)
                K *= ch->cback_scale;
            if (!(K > 0.0))
                cs_fatal("ядро K(r) должно быть строго положительным на сетке "
                         "(канал %d, r = %g)", j, r[i]);
            val[i] = log(K);
        }
        spline_init(&kt->sp[j], lr, val, nr);
    }
    free(lr);
    free(val);
}

void ktable_free(ktable_t *kt)
{
    if (!kt)
        return;
    for (int j = 0; j < kt->nch; j++)
        spline_free(&kt->sp[j]);
    free(kt->sp);
    free(kt->r);
    memset(kt, 0, sizeof(*kt));
}

double ktable_kernel(const ktable_t *kt, int j, double r)
{
    double rc = r;
    if (rc < kt->r_min)
        rc = kt->r_min;
    if (rc > kt->r_max)
        rc = kt->r_max;
    return exp(spline_eval(&kt->sp[j], log(rc)));
}

double ktable_kernels_at(const ktable_t *kt, int j, double r)
{
    return ktable_kernel(kt, j, r);
}
