/* Минимальный JSON-парсер (DOM) для входного файла измерений. */
#include "csolve.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *s;
} jp_t;

static void skip_ws(jp_t *p)
{
    while (*p->s && isspace((unsigned char)*p->s))
        p->s++;
}

static jval_t *new_val(jtype_t t)
{
    jval_t *v = xcalloc(1, sizeof(*v));
    v->t = t;
    return v;
}

static void utf8_put(char **out, unsigned cp)
{
    char *o = *out;
    if (cp < 0x80) {
        *o++ = (char)cp;
    } else if (cp < 0x800) {
        *o++ = (char)(0xC0 | (cp >> 6));
        *o++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *o++ = (char)(0xE0 | (cp >> 12));
        *o++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *o++ = (char)(0x80 | (cp & 0x3F));
    }
    *out = o;
}

static char *parse_string(jp_t *p)
{
    if (*p->s != '"')
        cs_fatal("JSON: ожидалась строка");
    p->s++;
    size_t cap = strlen(p->s) + 1;
    char *buf = xmalloc(cap);
    char *o = buf;
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (c == '\\') {
            char e = *p->s++;
            switch (e) {
            case 'n': *o++ = '\n'; break;
            case 't': *o++ = '\t'; break;
            case 'r': *o++ = '\r'; break;
            case 'b': *o++ = '\b'; break;
            case 'f': *o++ = '\f'; break;
            case '"': *o++ = '"'; break;
            case '\\': *o++ = '\\'; break;
            case '/': *o++ = '/'; break;
            case 'u': {
                char hex[5] = {0};
                for (int i = 0; i < 4; i++) {
                    if (!isxdigit((unsigned char)*p->s))
                        cs_fatal("JSON: некорректный \\u escape");
                    hex[i] = *p->s++;
                }
                unsigned cp = (unsigned)strtoul(hex, NULL, 16);
                utf8_put(&o, cp);
                break;
            }
            default:
                cs_fatal("JSON: неизвестный escape \\%c", e);
            }
        } else {
            *o++ = c;
        }
    }
    if (*p->s != '"')
        cs_fatal("JSON: незавершённая строка");
    p->s++;
    *o = '\0';
    return buf;
}

static jval_t *parse_value(jp_t *p);

static jval_t *parse_array(jp_t *p)
{
    jval_t *v = new_val(JARR);
    p->s++;   /* '[' */
    skip_ws(p);
    if (*p->s == ']') {
        p->s++;
        return v;
    }
    for (;;) {
        jval_t *item = parse_value(p);
        v->items = xrealloc(v->items, sizeof(jval_t *) * (size_t)(v->n + 1));
        v->items[v->n++] = item;
        skip_ws(p);
        if (*p->s == ',') {
            p->s++;
            continue;
        }
        if (*p->s == ']') {
            p->s++;
            return v;
        }
        cs_fatal("JSON: ожидалась ',' или ']'");
    }
}

static jval_t *parse_object(jp_t *p)
{
    jval_t *v = new_val(JOBJ);
    p->s++;   /* '{' */
    skip_ws(p);
    if (*p->s == '}') {
        p->s++;
        return v;
    }
    for (;;) {
        skip_ws(p);
        char *key = parse_string(p);
        skip_ws(p);
        if (*p->s != ':')
            cs_fatal("JSON: ожидалось ':'");
        p->s++;
        jval_t *val = parse_value(p);
        v->keys = xrealloc(v->keys, sizeof(char *) * (size_t)(v->nk + 1));
        v->vals = xrealloc(v->vals, sizeof(jval_t *) * (size_t)(v->nk + 1));
        v->keys[v->nk] = key;
        v->vals[v->nk] = val;
        v->nk++;
        skip_ws(p);
        if (*p->s == ',') {
            p->s++;
            continue;
        }
        if (*p->s == '}') {
            p->s++;
            return v;
        }
        cs_fatal("JSON: ожидалась ',' или '}'");
    }
}

static jval_t *parse_value(jp_t *p)
{
    skip_ws(p);
    if (*p->s == '{')
        return parse_object(p);
    if (*p->s == '[')
        return parse_array(p);
    if (*p->s == '"') {
        jval_t *v = new_val(JSTR);
        v->str = parse_string(p);
        return v;
    }
    if (strncmp(p->s, "true", 4) == 0) {
        p->s += 4;
        jval_t *v = new_val(JBOOL);
        v->bval = 1;
        return v;
    }
    if (strncmp(p->s, "false", 5) == 0) {
        p->s += 5;
        jval_t *v = new_val(JBOOL);
        v->bval = 0;
        return v;
    }
    if (strncmp(p->s, "null", 4) == 0) {
        p->s += 4;
        return new_val(JNULL);
    }
    char *end = NULL;
    double d = strtod(p->s, &end);
    if (end == p->s)
        cs_fatal("JSON: неожиданный символ '%c'", *p->s);
    p->s = end;
    jval_t *v = new_val(JNUM);
    v->num = d;
    return v;
}

jval_t *json_parse(const char *text)
{
    jp_t p = {text};
    jval_t *v = parse_value(&p);
    skip_ws(&p);
    if (*p.s)
        cs_fatal("JSON: лишние данные после значения");
    return v;
}

void json_free(jval_t *v)
{
    if (!v)
        return;
    free(v->str);
    for (int i = 0; i < v->n; i++)
        json_free(v->items[i]);
    free(v->items);
    for (int i = 0; i < v->nk; i++) {
        free(v->keys[i]);
        json_free(v->vals[i]);
    }
    free(v->keys);
    free(v->vals);
    free(v);
}

jval_t *json_get(const jval_t *obj, const char *key)
{
    if (!obj || obj->t != JOBJ)
        return NULL;
    for (int i = 0; i < obj->nk; i++)
        if (strcmp(obj->keys[i], key) == 0)
            return obj->vals[i];
    return NULL;
}
