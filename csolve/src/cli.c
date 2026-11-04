/* Разбор командной строки. */
#include "csolve.h"

#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    puts("csolve — восстановление распределения частиц по размерам из лидарных\n"
         "коэффициентов по внешним таблицам сечений (C-версия табличного режима solve).\n"
         "\n"
         "Использование:\n"
         "  csolve --tables DIR [--data JSON] [опции]\n"
         "\n"
         "Обязательные:\n"
         "  --tables DIR              каталог с mre_<real>_mim_<imag>/data.txt\n"
         "\n"
         "Данные и таблицы:\n"
         "  --data JSON               измеренные коэффициенты (иначе демо)\n"
         "  --table-extrapolate       разрешить экстраполяцию сечения по x\n"
         "  --table-ref-lambda-um X   эталонная длина волны таблиц, мкм [2pi]\n"
         "  --table-cback {dsigma|4pi} конвенция столбца Cback [dsigma]\n"
         "\n"
         "Сетки и перебор:\n"
         "  --degree K...             степени B-сплайнов [1 2 3 4]\n"
         "  --n-basis N               базисных функций на окно [число каналов]\n"
         "  --r0-min/--r0-max/--nr0   сетка r0 [0.02 0.5 6]\n"
         "  --r1-min/--r1-max/--nr1   сетка r1 [2 10 4]\n"
         "  --min-ratio X             r1 >= X*r0 [4]\n"
         "  --best G                  число лучших решений [5]\n"
         "  --m-reals/--m-imags ...   сузить сетку m (значения из таблиц)\n"
         "  --m-real/--m-imag X       истинный m демо\n"
         "\n"
         "Регуляризация и вывод:\n"
         "  --gamma-select {fixed|cv|err}  [fixed]\n"
         "  --gamma-rel X             gamma/||A||_2 [0.1]\n"
         "  --gamma-min/--gamma-max/--ngamma  [-7 5 41]\n"
         "  --nonneg                  ограничение C >= 0\n"
         "  --no-zero-boundary        без ГУ n(r0)=n(r1)=0\n"
         "  --no-smooth               без сглаживания среднего\n"
         "  --smooth-lam X            параметр сглаживания [1e-4]\n"
         "  --r-split X               граница fine/coarse [0.5]\n"
         "  --modes {auto|fit|stat}   расчёт параметров мод [auto]\n"
         "  --absolute                МНК по абсолютным невязкам\n"
         "  --dx-max X                шаг по x в таблице ядер [0.1]\n"
         "  --nr-out N                точек выходной сетки [256]\n"
         "  --quad-order N            порядок Гаусса-Лежандра [24]\n"
         "  --log-span X              макс. ширина подсегмента в ln r [0.1]\n"
         "  --noise X --seed N        шум демо и зерно [0 1]\n"
         "  --out PNG                 файл графика [result.png]\n"
         "  --json JSON               сохранить результаты\n"
         "  --no-plot                 не строить график\n"
         "  -h, --help                эта справка\n"
         "\n"
         "Пакетная обработка (--batch):\n"
         "  --batch TXT               txt со столбцами Altitude b<нм>... a<нм>...\n"
         "  --batch-out-dir DIR       каталог результатов [рядом с TXT]\n"
         "  --lidar-ratio-cons        ограничить плавность S = α/β с высотой\n"
         "  --lidar-ratio-w X         вес штрафа за отклонение S [0.5]\n"
         "  --lidar-ratio-iter N      итераций уточнения опорного S [3]\n");
}

static double parse_double(const char *s, const char *flag)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || *end != '\0')
        cs_fatal("опция %s: ожидается число, получено '%s'", flag, s);
    return v;
}

static int parse_int(const char *s, const char *flag)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0')
        cs_fatal("опция %s: ожидается целое число, получено '%s'", flag, s);
    return (int)v;
}

static const char *need_value(int argc, char **argv, int *i,
                              const char *inline_val, const char *flag)
{
    if (inline_val)
        return inline_val;
    if (*i + 1 >= argc)
        cs_fatal("опция %s требует значения", flag);
    return argv[++(*i)];
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
    a->out = "result.png";
    a->lidar_ratio_w = 0.5;
    a->lidar_ratio_iter = 3;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage();
            exit(0);
        }
        if (strncmp(arg, "--", 2) != 0)
            cs_fatal("неожиданный аргумент: %s", arg);

        char namebuf[128];
        const char *name = arg;
        const char *inline_val = NULL;
        const char *eq = strchr(arg, '=');
        if (eq) {
            size_t n = (size_t)(eq - arg);
            if (n >= sizeof(namebuf))
                cs_fatal("слишком длинное имя опции");
            memcpy(namebuf, arg, n);
            namebuf[n] = '\0';
            name = namebuf;
            inline_val = eq + 1;
        }

#define IS(x) (strcmp(name, x) == 0)
        if (IS("--data")) {
            a->data = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--tables")) {
            a->tables = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--table-extrapolate")) {
            a->table_extrapolate = 1;
        } else if (IS("--table-ref-lambda-um")) {
            a->table_ref_wl = parse_double(
                need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--table-cback")) {
            const char *v = need_value(argc, argv, &i, inline_val, name);
            if (strcmp(v, "dsigma") == 0)
                a->table_cback_4pi = 0;
            else if (strcmp(v, "4pi") == 0)
                a->table_cback_4pi = 1;
            else
                cs_fatal("--table-cback: ожидается dsigma или 4pi, получено '%s'", v);
        } else if (IS("--degree")) {
            int nd = 0;
            if (inline_val)
                a->degrees[nd++] = parse_int(inline_val, name);
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                if (nd >= 8)
                    cs_fatal("--degree: слишком много значений");
                a->degrees[nd++] = parse_int(argv[++i], name);
            }
            if (nd == 0)
                cs_fatal("--degree требует хотя бы одно значение");
            a->n_degrees = nd;
        } else if (IS("--n-basis")) {
            a->n_basis = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--r0-min")) {
            a->r0_min = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--r0-max")) {
            a->r0_max = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--nr0")) {
            a->nr0 = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--r1-min")) {
            a->r1_min = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--r1-max")) {
            a->r1_max = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--nr1")) {
            a->nr1 = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--min-ratio")) {
            a->min_ratio = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--best")) {
            a->best = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--gamma-select")) {
            const char *v = need_value(argc, argv, &i, inline_val, name);
            if (strcmp(v, "fixed") == 0)
                a->gamma_select = GAMMA_FIXED;
            else if (strcmp(v, "cv") == 0)
                a->gamma_select = GAMMA_CV;
            else if (strcmp(v, "err") == 0)
                a->gamma_select = GAMMA_ERR;
            else
                cs_fatal("--gamma-select: ожидается fixed, cv или err");
        } else if (IS("--gamma-rel")) {
            a->gamma_rel = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--gamma-min")) {
            a->gamma_min = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--gamma-max")) {
            a->gamma_max = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--ngamma")) {
            a->ngamma = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--nonneg")) {
            a->nonneg = 1;
        } else if (IS("--no-zero-boundary")) {
            a->no_zero_boundary = 1;
        } else if (IS("--no-smooth")) {
            a->no_smooth = 1;
        } else if (IS("--smooth-lam")) {
            a->smooth_lam = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--r-split")) {
            a->r_split = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--modes")) {
            const char *v = need_value(argc, argv, &i, inline_val, name);
            if (strcmp(v, "auto") == 0)
                a->modes = 0;
            else if (strcmp(v, "fit") == 0)
                a->modes = 1;
            else if (strcmp(v, "stat") == 0)
                a->modes = 2;
            else
                cs_fatal("--modes: ожидается auto, fit или stat");
        } else if (IS("--absolute")) {
            a->absolute = 1;
        } else if (IS("--dx-max")) {
            a->dx_max = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--nr-out")) {
            a->nr_out = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--quad-order")) {
            a->quad_order = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--log-span")) {
            a->log_span = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--m-real")) {
            a->has_m_real = 1;
            a->m_real = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--m-imag")) {
            a->has_m_imag = 1;
            a->m_imag = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--m-reals") || IS("--m-imags")) {
            int is_reals = IS("--m-reals");
            double **dst = is_reals ? &a->m_reals : &a->m_imags;
            int *cnt = is_reals ? &a->n_m_reals : &a->n_m_imags;
            if (inline_val) {
                *dst = xrealloc(*dst, sizeof(double) * (size_t)(*cnt + 1));
                (*dst)[(*cnt)++] = parse_double(inline_val, name);
            }
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                *dst = xrealloc(*dst, sizeof(double) * (size_t)(*cnt + 1));
                (*dst)[(*cnt)++] = parse_double(argv[++i], name);
            }
            if (*cnt == 0)
                cs_fatal("%s требует хотя бы одно значение", name);
        } else if (IS("--noise")) {
            a->noise = parse_double(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--seed")) {
            a->seed = parse_int(need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--out")) {
            a->out = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--json")) {
            a->json = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--no-plot")) {
            a->no_plot = 1;
        } else if (IS("--batch")) {
            a->batch = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--batch-out-dir")) {
            a->batch_out_dir = need_value(argc, argv, &i, inline_val, name);
        } else if (IS("--lidar-ratio-cons")) {
            a->lidar_ratio_cons = 1;
        } else if (IS("--lidar-ratio-w")) {
            a->lidar_ratio_w = parse_double(
                need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--lidar-ratio-iter")) {
            a->lidar_ratio_iter = parse_int(
                need_value(argc, argv, &i, inline_val, name), name);
        } else if (IS("--no-jit")) {
            /* Mie-бэкенда в C-версии нет — флаг принимается и игнорируется */
        } else {
            cs_fatal("неизвестный аргумент: %s (см. --help)", arg);
        }
#undef IS
    }
}

void args_free(args_t *a)
{
    free(a->m_reals);
    free(a->m_imags);
    a->m_reals = a->m_imags = NULL;
}
