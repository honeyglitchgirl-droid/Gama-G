/* Gama-G native runtime -- implementation.
 *
 * Every function here has a counterpart in the reference interpreter, named in
 * its comment.  Where the two could differ, the difference is written down
 * rather than left to be found by a user.
 */

#include "gamag_rt.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

GFault  g_fault = { "", "", "", 0 };
jmp_buf g_fault_jmp;

/* ================================================================== */
/* Arena                                                              */
/* ================================================================== */

#define G_CHUNK (1u << 20)

/* A chunk owns a run of bytes.  Chunks are kept in a list, newest first, so a
 * mark can name the chunk it was taken in and everything allocated after it
 * lives in that chunk or in the ones in front of it. */
typedef struct GChunk {
    struct GChunk *next;
    size_t size;                    /* bytes owned by this chunk */
    size_t used;                    /* bytes handed out */
    size_t reserved;                /* keeps the payload 16-byte aligned */
} GChunk;

static GChunk *g_arena_head = NULL;     /* newest chunk still holding values */
static GChunk *g_arena_pool = NULL;     /* released chunks, reused before malloc */
static size_t  g_live_bytes = 0;        /* bytes handed out right now */
static size_t  g_peak_bytes = 0;        /* high-water mark of g_live_bytes */
static size_t  g_arena_total = 0;       /* cumulative bytes taken from malloc */
static int     g_arena_stats_hooked = 0;

static void arena_stats(void)
{
    const char *want = getenv("GG_ARENA_STATS");
    if (want == NULL || want[0] == '\0')
        return;
    fprintf(stderr, "gama-g arena: live=%lu peak=%lu total=%lu\n",
            (unsigned long)g_live_bytes, (unsigned long)g_peak_bytes,
            (unsigned long)g_arena_total);
}

static void arena_push(size_t need)
{
    /* A chunk that was released is reused rather than returned to malloc: the
     * program's footprint then settles at its high-water mark instead of
     * churning, which is what a long-running program wants. */
    if (need <= G_CHUNK && g_arena_pool != NULL) {
        GChunk *c = g_arena_pool;
        g_arena_pool = c->next;
        c->next = g_arena_head;
        c->used = 0;
        g_arena_head = c;
        return;
    }
    size_t n = need > G_CHUNK ? need : G_CHUNK;
    GChunk *c = (GChunk *)malloc(sizeof(GChunk) + n);
    if (!c) {
        fprintf(stderr, "gama-g native: out of memory\n");
        exit(G_EXIT_RUNTIME);
    }
    c->next = g_arena_head;
    c->size = n;
    c->used = 0;
    c->reserved = 0;
    g_arena_head = c;
    g_arena_total += n;
    if (!g_arena_stats_hooked) {
        g_arena_stats_hooked = 1;
        atexit(arena_stats);
    }
}

void *g_alloc(size_t n)
{
    n = (n + 15u) & ~(size_t)15u;
    GChunk *c = g_arena_head;
    if (c == NULL || c->size - c->used < n) {
        arena_push(n);
        c = g_arena_head;
    }
    char *base = (char *)(c + 1);
    void *p = base + c->used;
    c->used += n;
    g_live_bytes += n;
    if (g_live_bytes > g_peak_bytes)
        g_peak_bytes = g_live_bytes;
    return p;
}

/* Bytes handed out and not yet released.  This is the number that says whether
 * a program leaks; `g_arena_bytes` below is the cumulative figure and only ever
 * grows. */
size_t g_arena_live(void) { return g_live_bytes; }
size_t g_arena_peak(void) { return g_peak_bytes; }
size_t g_arena_bytes(void) { return g_arena_total; }

GArenaMark g_arena_mark(void)
{
    GArenaMark m;
    m.chunk = g_arena_head;
    m.used = g_arena_head ? g_arena_head->used : 0;
    return m;
}

/* Release everything allocated since `m`.
 *
 * Marks nest, because a function marks on entry and releases before it
 * returns, and calls nest: the newest mark is always released first.  That is
 * what makes freeing whole chunks in a loop correct -- there is never a live
 * value behind the mark being released.
 *
 * Only chunks are freed, and only to the pool.  A value allocated since the
 * mark and still in use would be a bug in the caller's decision to release,
 * and the release decision is conservative by construction: see
 * `cgen.release_eligible`. */
void g_arena_release(GArenaMark m)
{
    while (g_arena_head != m.chunk) {
        GChunk *c = g_arena_head;
        g_arena_head = c->next;
        g_live_bytes -= c->used;
        c->used = 0;
        c->next = g_arena_pool;
        g_arena_pool = c;
    }
    if (g_arena_head != NULL && g_arena_head->used > m.used) {
        g_live_bytes -= (g_arena_head->used - m.used);
        g_arena_head->used = m.used;
    }
}

void g_arena_reset(void)
{
    while (g_arena_head != NULL) {
        GChunk *c = g_arena_head;
        g_arena_head = c->next;
        c->used = 0;
        c->next = g_arena_pool;
        g_arena_pool = c;
    }
    g_live_bytes = 0;
}

char *g_strdup_n(const char *s, size_t n)
{
    char *p = (char *)g_alloc(n + 1);
    if (n) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

char *g_strdup(const char *s) { return g_strdup_n(s, strlen(s)); }

/* ================================================================== */
/* Faults                                                             */
/* ================================================================== */

void g_raise(const char *kind, const char *pos, const char *fmt, ...)
{
    va_list ap;
    g_fault.kind = kind;
    if (pos && *pos) snprintf(g_fault.pos, sizeof g_fault.pos, "%s", pos);
    else             g_fault.pos[0] = '\0';
    va_start(ap, fmt);
    vsnprintf(g_fault.message, sizeof g_fault.message, fmt, ap);
    va_end(ap);
    g_fault.active = 1;
    longjmp(g_fault_jmp, 1);
}

/* ================================================================== */
/* Constructors                                                       */
/* ================================================================== */

GValue g_unit(void)           { GValue v; v.tag = GV_UNIT;  v.u.p = NULL;  return v; }
GValue g_bool(int b)          { GValue v; v.tag = GV_BOOL;  v.u.b = !!b;   return v; }
GValue g_int(int64_t i)       { GValue v; v.tag = GV_INT;   v.u.i = i;     return v; }
GValue g_float(double f)      { GValue v; v.tag = GV_FLOAT; v.u.f = f;     return v; }

GValue g_text_n(const char *s, size_t n)
{
    GText *t = (GText *)g_alloc(sizeof(GText));
    t->len = n;
    t->data = g_strdup_n(s, n);
    GValue v; v.tag = GV_TEXT; v.u.s = t; return v;
}

GValue g_text(const char *s) { return g_text_n(s, strlen(s)); }

static GValue seq_new(GTag tag, size_t n, const GValue *items)
{
    GList *l = (GList *)g_alloc(sizeof(GList));
    l->len = n;
    l->cap = n ? n : 1;
    l->items = (GValue *)g_alloc(sizeof(GValue) * l->cap);
    for (size_t i = 0; i < n; i++) l->items[i] = items[i];
    GValue v; v.tag = tag; v.u.l = l; return v;
}

GValue g_list_new(size_t n, const GValue *items)  { return seq_new(GV_LIST,  n, items); }
GValue g_tuple_new(size_t n, const GValue *items) { return seq_new(GV_TUPLE, n, items); }

GValue g_map_new(void)
{
    GMap *m = (GMap *)g_alloc(sizeof(GMap));
    m->len = 0; m->cap = 4;
    m->keys = (GValue *)g_alloc(sizeof(GValue) * m->cap);
    m->vals = (GValue *)g_alloc(sizeof(GValue) * m->cap);
    GValue v; v.tag = GV_MAP; v.u.m = m; return v;
}

void g_list_push(GValue list, GValue item)
{
    GList *l = list.u.l;
    if (l->len == l->cap) {
        size_t nc = l->cap * 2;
        GValue *ni = (GValue *)g_alloc(sizeof(GValue) * nc);
        memcpy(ni, l->items, sizeof(GValue) * l->len);
        l->items = ni; l->cap = nc;
    }
    l->items[l->len++] = item;
}

void g_map_set(GValue map, GValue key, GValue val)
{
    GMap *m = map.u.m;
    for (size_t i = 0; i < m->len; i++) {
        if (g_equal(m->keys[i], key)) { m->vals[i] = val; return; }
    }
    if (m->len == m->cap) {
        size_t nc = m->cap * 2;
        GValue *nk = (GValue *)g_alloc(sizeof(GValue) * nc);
        GValue *nv = (GValue *)g_alloc(sizeof(GValue) * nc);
        memcpy(nk, m->keys, sizeof(GValue) * m->len);
        memcpy(nv, m->vals, sizeof(GValue) * m->len);
        m->keys = nk; m->vals = nv; m->cap = nc;
    }
    m->keys[m->len] = key;
    m->vals[m->len] = val;
    m->len++;
}

GValue g_map_get(GValue map, GValue key, const char *pos)
{
    GMap *m = map.u.m;
    for (size_t i = 0; i < m->len; i++)
        if (g_equal(m->keys[i], key)) return m->vals[i];
    g_raise("KeyError", pos, "no key %s in the map", g_display(key, 0));
    return g_unit(); /* unreachable */
}

int g_map_has(GValue map, GValue key)
{
    GMap *m = map.u.m;
    for (size_t i = 0; i < m->len; i++)
        if (g_equal(m->keys[i], key)) return 1;
    return 0;
}

GValue g_variant(const char *kind, const char *tag, size_t n, const GValue *args)
{
    GNamed *r = (GNamed *)g_alloc(sizeof(GNamed));
    r->kind = kind ? kind : "";
    r->len = n; r->cap = n ? n : 1;
    r->names = (char **)g_alloc(sizeof(char *) * r->cap);
    r->vals = (GValue *)g_alloc(sizeof(GValue) * r->cap);
    r->names[0] = g_strdup(tag);
    for (size_t i = 0; i < n; i++) r->vals[i] = args[i];
    GValue v; v.tag = GV_VARIANT; v.u.n = r; return v;
}

GValue g_record(const char *name, size_t n, const char **fields, const GValue *vals)
{
    GNamed *r = (GNamed *)g_alloc(sizeof(GNamed));
    r->kind = name ? name : "";
    r->len = n; r->cap = n ? n : 1;
    r->names = (char **)g_alloc(sizeof(char *) * r->cap);
    r->vals = (GValue *)g_alloc(sizeof(GValue) * r->cap);
    for (size_t i = 0; i < n; i++) { r->names[i] = g_strdup(fields[i]); r->vals[i] = vals[i]; }
    GValue v; v.tag = GV_RECORD; v.u.n = r; return v;
}

GValue g_option(int some, GValue value)
{
    GNamed *r = (GNamed *)g_alloc(sizeof(GNamed));
    r->kind = some ? "some" : "none";
    r->len = some ? 1 : 0; r->cap = 1;
    r->names = (char **)g_alloc(sizeof(char *));
    r->vals = (GValue *)g_alloc(sizeof(GValue));
    r->names[0] = g_strdup("value");
    if (some) r->vals[0] = value;
    GValue v; v.tag = GV_OPTION; v.u.n = r; return v;
}

GValue g_result(int ok, GValue value)
{
    GNamed *r = (GNamed *)g_alloc(sizeof(GNamed));
    r->kind = ok ? "ok" : "fail";
    r->len = 1; r->cap = 1;
    r->names = (char **)g_alloc(sizeof(char *));
    r->vals = (GValue *)g_alloc(sizeof(GValue));
    r->names[0] = g_strdup("value");
    r->vals[0] = value;
    GValue v; v.tag = GV_RESULT; v.u.n = r; return v;
}

GValue g_secret(const char *label, GValue inner)
{
    GSecret *s = (GSecret *)g_alloc(sizeof(GSecret));
    s->label = label ? label : "value";
    s->inner = (GValue *)g_alloc(sizeof(GValue));
    *s->inner = inner;
    GValue v; v.tag = GV_SECRET; v.u.sec = s; return v;
}

/* ================================================================== */
/* Rendering                                                          */
/* ================================================================== */

/* A small growable string, on the arena. */
typedef struct { char *p; size_t len, cap; } SBuf;

static void sb_init(SBuf *b) { b->cap = 64; b->p = (char *)g_alloc(b->cap); b->len = 0; b->p[0] = '\0'; }
static void sb_putn(SBuf *b, const char *s, size_t n)
{
    if (b->len + n + 1 > b->cap) {
        size_t nc = (b->len + n + 1) * 2;
        char *np = (char *)g_alloc(nc);
        memcpy(np, b->p, b->len + 1);
        b->p = np; b->cap = nc;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void sb_puts(SBuf *b, const char *s) { sb_putn(b, s, strlen(s)); }
static void sb_putc(SBuf *b, char c) { sb_putn(b, &c, 1); }
static void sb_putf(SBuf *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof tmp) { sb_putn(b, tmp, (size_t)n); return; }
    char *big = (char *)g_alloc((size_t)n + 1);
    va_start(ap, fmt); vsnprintf(big, (size_t)n + 1, fmt, ap); va_end(ap);
    sb_putn(b, big, (size_t)n);
}

/* `values.py::display` for a float.
 *
 * Python's repr() is the shortest decimal string that round-trips, rendered in
 * fixed notation when `-4 < decpt <= 16` and in exponential notation otherwise
 * (CPython: `format_float_short` with mode 'r' and `Py_DTSF_ADD_DOT_0`).
 * `%.17g` does not reproduce that, so the digits are found by scanning for the
 * shortest round-trip and then laid out by Python's own rule. */
void g_repr_float(double v, char *out, size_t n)
{
    if (isnan(v)) { snprintf(out, n, "nan"); return; }
    if (isinf(v)) { snprintf(out, n, v > 0 ? "inf" : "-inf"); return; }

    char buf[64];
    int prec = 17;
    for (int p = 0; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p, v);
        if (strtod(buf, NULL) == v) { prec = p; break; }
    }
    snprintf(buf, sizeof buf, "%.*e", prec, v);

    /* buf is [-]d[.ddd]e[+-]XX */
    char *p = buf;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    char digits[64]; size_t nd = 0;
    while (*p && *p != 'e' && *p != 'E') {
        if (*p != '.' && nd < sizeof digits - 1) digits[nd++] = *p;
        p++;
    }
    digits[nd] = '\0';
    int expo = (*p == 'e' || *p == 'E') ? atoi(p + 1) : 0;
    int decpt = expo + 1;                    /* value = 0.digits * 10^decpt */
    while (nd > 1 && digits[nd - 1] == '0') digits[--nd] = '\0';

    SBuf b; sb_init(&b);
    if (neg) sb_putc(&b, '-');

    if (decpt <= -4 || decpt > 16) {
        sb_putc(&b, digits[0]);
        if (nd > 1) { sb_putc(&b, '.'); sb_puts(&b, digits + 1); }
        sb_putf(&b, "e%c%02d", decpt - 1 < 0 ? '-' : '+',
                decpt - 1 < 0 ? -(decpt - 1) : decpt - 1);
    } else if (decpt <= 0) {
        sb_puts(&b, "0.");
        for (int i = 0; i < -decpt; i++) sb_putc(&b, '0');
        sb_puts(&b, digits);
    } else if ((size_t)decpt >= nd) {
        sb_puts(&b, digits);
        for (int i = 0; i < decpt - (int)nd; i++) sb_putc(&b, '0');
        sb_puts(&b, ".0");
    } else {
        char head[64]; memcpy(head, digits, (size_t)decpt); head[decpt] = '\0';
        sb_puts(&b, head);
        sb_putc(&b, '.');
        sb_puts(&b, digits + decpt);
    }
    snprintf(out, n, "%s", b.p);
}

/* `values.py::type_name` */
const char *g_type_name(GValue v)
{
    switch (v.tag) {
    case GV_BOOL:   return "Bool";
    case GV_INT:    return "I64";
    case GV_FLOAT:  return "F64";
    case GV_TEXT:   return v.u.s->len == 1 ? "Char" : "Text";
    case GV_BYTES:  return "Bytes";
    case GV_UNIT:   return "Unit";
    case GV_LIST:   return "List";
    case GV_MAP:    return "Map";
    case GV_TUPLE:  return "Tuple";
    case GV_RECORD: return v.u.n->kind;
    case GV_VARIANT:return "Variant";
    case GV_OPTION: return "Option";
    case GV_RESULT: return "Result";
    case GV_SECRET: return "Secret";
    case GV_FUNC:   return "Function";
    }
    return "Any";
}

static void display_into(SBuf *b, GValue v, int allow_secret);

char *g_display(GValue v, int allow_secret)
{
    SBuf b; sb_init(&b);
    display_into(&b, v, allow_secret);
    return b.p;
}

char *g_show(GValue v) { return g_display(v, 0); }

static void display_into(SBuf *b, GValue v, int allow_secret)
{
    switch (v.tag) {
    case GV_UNIT: sb_puts(b, "()"); return;
    case GV_BOOL: sb_puts(b, v.u.b ? "true" : "false"); return;
    case GV_INT:  sb_putf(b, "%lld", (long long)v.u.i); return;
    case GV_FLOAT: {
        double f = v.u.f;
        char out[64];
        if (isnan(f))            snprintf(out, sizeof out, "nan");
        else if (isinf(f))       snprintf(out, sizeof out, f > 0 ? "inf" : "-inf");
        else if (f == floor(f) && fabs(f) < 1e16)
                                 snprintf(out, sizeof out, "%.1f", f);
        else                     g_repr_float(f, out, sizeof out);
        sb_puts(b, out);
        return;
    }
    case GV_TEXT: sb_putn(b, v.u.s->data, v.u.s->len); return;
    case GV_SECRET:
        if (!allow_secret)
            g_raise("SecretLeak", "",
                    "cannot convert <secret %s> to Text", v.u.sec->label);
        display_into(b, *v.u.sec->inner, 1);
        return;
    case GV_LIST:
    case GV_TUPLE: {
        GList *l = v.u.l;
        sb_putc(b, '[');
        for (size_t i = 0; i < l->len; i++) {
            if (i) sb_puts(b, ", ");
            display_into(b, l->items[i], allow_secret);
        }
        sb_putc(b, ']');
        return;
    }
    case GV_MAP: {
        GMap *m = v.u.m;
        sb_putc(b, '{');
        for (size_t i = 0; i < m->len; i++) {
            if (i) sb_puts(b, ", ");
            display_into(b, m->keys[i], allow_secret);
            sb_puts(b, ": ");
            display_into(b, m->vals[i], allow_secret);
        }
        sb_putc(b, '}');
        return;
    }
    case GV_OPTION:
        if (strcmp(v.u.n->kind, "none") == 0) { sb_puts(b, "none"); return; }
        sb_puts(b, "some(");
        display_into(b, v.u.n->vals[0], allow_secret);
        sb_putc(b, ')');
        return;
    case GV_RESULT:
        sb_putf(b, "%s(", v.u.n->kind);
        display_into(b, v.u.n->vals[0], allow_secret);
        sb_putc(b, ')');
        return;
    case GV_VARIANT: {
        GNamed *r = v.u.n;
        sb_puts(b, r->names[0]);
        if (r->len > 1) {
            sb_putc(b, '(');
            for (size_t i = 1; i < r->len; i++) {
                if (i > 1) sb_puts(b, ", ");
                display_into(b, r->vals[i], allow_secret);
            }
            sb_putc(b, ')');
        }
        return;
    }
    case GV_RECORD: {
        GNamed *r = v.u.n;
        sb_putf(b, "%s { ", r->kind);
        for (size_t i = 0; i < r->len; i++) {
            if (i) sb_puts(b, ", ");
            sb_putf(b, "%s: ", r->names[i]);
            display_into(b, r->vals[i], allow_secret);
        }
        sb_puts(b, " }");
        return;
    }
    case GV_BYTES: sb_putf(b, "bytes(%p)", v.u.p); return;
    case GV_FUNC:  sb_puts(b, "<function>"); return;
    }
}

char *g_to_text(GValue v) { return g_display(v, 0); }

/* ================================================================== */
/* Predicates                                                         */
/* ================================================================== */

/* `values.py::truthy` */
int g_truthy(GValue v, const char *pos)
{
    switch (v.tag) {
    case GV_BOOL:  return v.u.b;
    case GV_OPTION:return strcmp(v.u.n->kind, "some") == 0;
    case GV_RESULT:return strcmp(v.u.n->kind, "ok") == 0;
    case GV_UNIT:  return 0;
    case GV_INT:   return v.u.i != 0;
    case GV_FLOAT: return v.u.f != 0.0;
    case GV_TEXT:  return v.u.s->len != 0;
    case GV_LIST:
    case GV_TUPLE: return v.u.l->len != 0;
    case GV_MAP:   return v.u.m->len != 0;
    default:
        g_raise("TypeFault", pos, "cannot use %s as a Bool", g_type_name(v));
        return 0;
    }
}

/* `ops.py::values_equal` */
int g_equal(GValue a, GValue b)
{
    if (a.tag == GV_SECRET || b.tag == GV_SECRET)
        g_raise("SecretLeak", "", "cannot compare secret values directly");
    if (a.tag == GV_BOOL || b.tag == GV_BOOL) {
        if (a.tag != b.tag) return 0;
        return a.u.b == b.u.b;
    }
    if (a.tag == GV_UNIT || b.tag == GV_UNIT)
        return a.tag == GV_UNIT && b.tag == GV_UNIT;
    if (a.tag != b.tag) {
        /* I64 against F64 compares by value, as Python's int == float does. */
        if ((a.tag == GV_INT && b.tag == GV_FLOAT) ||
            (a.tag == GV_FLOAT && b.tag == GV_INT)) {
            double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
            double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
            return x == y;
        }
        return 0;
    }
    switch (a.tag) {
    case GV_INT:   return a.u.i == b.u.i;
    case GV_FLOAT: return a.u.f == b.u.f;
    case GV_TEXT:  return a.u.s->len == b.u.s->len &&
                          memcmp(a.u.s->data, b.u.s->data, a.u.s->len) == 0;
    case GV_LIST:
    case GV_TUPLE: {
        GList *x = a.u.l, *y = b.u.l;
        if (x->len != y->len) return 0;
        for (size_t i = 0; i < x->len; i++) if (!g_equal(x->items[i], y->items[i])) return 0;
        return 1;
    }
    case GV_MAP: {
        GMap *x = a.u.m, *y = b.u.m;
        if (x->len != y->len) return 0;
        for (size_t i = 0; i < x->len; i++) {
            if (!g_map_has(b, x->keys[i])) return 0;
            if (!g_equal(g_map_get(b, x->keys[i], ""), g_map_get(a, x->keys[i], ""))) return 0;
        }
        return 1;
    }
    case GV_OPTION:
    case GV_RESULT:
    case GV_VARIANT:
        if (strcmp(a.u.n->kind, b.u.n->kind) != 0) return 0;
        if (a.u.n->len != b.u.n->len) return 0;
        for (size_t i = 0; i < a.u.n->len; i++)
            if (!g_equal(a.u.n->vals[i], b.u.n->vals[i])) return 0;
        return 1;
    case GV_RECORD:
        if (strcmp(a.u.n->kind, b.u.n->kind) != 0) return 0;
        if (a.u.n->len != b.u.n->len) return 0;
        for (size_t i = 0; i < a.u.n->len; i++) {
            if (strcmp(a.u.n->names[i], b.u.n->names[i]) != 0) return 0;
            if (!g_equal(a.u.n->vals[i], b.u.n->vals[i])) return 0;
        }
        return 1;
    case GV_UNIT: return 1;
    default: return 0;
    }
}

/* ================================================================== */
/* Operators -- `ops.py::binop`, `unop`, `compare`                     */
/* ================================================================== */

static int is_numeric(GValue v) { return v.tag == GV_INT || v.tag == GV_FLOAT; }

GValue g_compare(const char *op, GValue a, GValue b, const char *pos)
{
    if (a.tag == GV_SECRET || b.tag == GV_SECRET)
        g_raise("SecretLeak", pos, "cannot order secret values");
    if ((a.tag == GV_BOOL) != (b.tag == GV_BOOL))
        g_raise("TypeFault", pos, "cannot order a Bool against a non-Bool");
    int r;
    if (is_numeric(a) && is_numeric(b)) {
        double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
        double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
        /* Integer-versus-integer must not lose precision through double. */
        if (a.tag == GV_INT && b.tag == GV_INT) {
            r = a.u.i < b.u.i ? -1 : (a.u.i > b.u.i ? 1 : 0);
        } else {
            r = x < y ? -1 : (x > y ? 1 : 0);
        }
    } else if (a.tag == GV_TEXT && b.tag == GV_TEXT) {
        size_t n = a.u.s->len < b.u.s->len ? a.u.s->len : b.u.s->len;
        int c = n ? memcmp(a.u.s->data, b.u.s->data, n) : 0;
        r = c ? (c < 0 ? -1 : 1)
              : (a.u.s->len < b.u.s->len ? -1 : (a.u.s->len > b.u.s->len ? 1 : 0));
    } else {
        g_raise("TypeFault", pos, "cannot order %s against %s",
                g_type_name(a), g_type_name(b));
        return g_unit();
    }
    if (!strcmp(op, "<"))  return g_bool(r < 0);
    if (!strcmp(op, ">"))  return g_bool(r > 0);
    if (!strcmp(op, "<=")) return g_bool(r <= 0);
    return g_bool(r >= 0);
}

/* A result too large for the exact arithmetic the interpreter performs. */
static void too_big(const char *pos)
{
    g_raise("IntegerOverflow", pos,
            "the native backend computes integers in 64 bits, and this result "
            "does not fit; the reference interpreter computes it exactly");
}

/* The scalar-specialized path calls these rather than repeating the messages,
 * so the two paths raise the same fault with the same text by construction
 * instead of by inspection. */
void g_int_overflow(const char *pos) { too_big(pos); }

void g_zero_division(const char *pos, const char *what)
{
    g_raise("DivideByZero", pos, "%s", what);
}

void g_div_zero_values(const char *pos, GValue a, GValue b)
{
    g_raise("DivideByZero", pos, "division by zero: %s / %s",
            g_display(a, 0), g_display(b, 0));
}

static GValue num_binop(const char *op, GValue a, GValue b, const char *pos)
{
    int both_int = (a.tag == GV_INT && b.tag == GV_INT);

    if (!strcmp(op, "+") || !strcmp(op, "-") || !strcmp(op, "*")) {
        if (both_int) {
            int64_t r; int ov;
            if (!strcmp(op, "+"))      ov = __builtin_add_overflow(a.u.i, b.u.i, &r);
            else if (!strcmp(op, "-")) ov = __builtin_sub_overflow(a.u.i, b.u.i, &r);
            else                       ov = __builtin_mul_overflow(a.u.i, b.u.i, &r);
            if (ov) too_big(pos);
            return g_int(r);
        }
        double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
        double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
        if (!strcmp(op, "+")) return g_float(x + y);
        if (!strcmp(op, "-")) return g_float(x - y);
        return g_float(x * y);
    }

    if (!strcmp(op, "/")) {
        if ((both_int && b.u.i == 0) || (!both_int && (b.tag == GV_INT ? b.u.i : b.u.f) == 0))
            g_div_zero_values(pos, a, b);
        if (both_int) {
            if (b.u.i == 0) g_raise("DivideByZero", pos, "division by zero");
            /* INT64_MIN / -1 overflows; Python would give 9223372036854775808. */
            if (a.u.i == INT64_MIN && b.u.i == -1) too_big(pos);
            return g_int(a.u.i / b.u.i);   /* truncates toward zero, as trunc_div */
        }
        double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
        double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
        return g_float(x / y);
    }

    if (!strcmp(op, "%")) {
        if (both_int) {
            if (b.u.i == 0) g_raise("DivideByZero", pos, "modulo by zero");
            if (a.u.i == INT64_MIN && b.u.i == -1) return g_int(0);
            return g_int(a.u.i % b.u.i);   /* sign of the dividend, as trunc_mod */
        }
        double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
        double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
        if (y == 0) g_raise("DivideByZero", pos, "modulo by zero");
        return g_float(fmod(x, y));
    }

    if (!strcmp(op, "**")) {
        if (both_int) {
            if (b.u.i < 0) {
                /* Python returns a float for a negative integer exponent. */
                return g_float(pow((double)a.u.i, (double)b.u.i));
            }
            int64_t r = 1;
            for (int64_t k = 0; k < b.u.i; k++) {
                int64_t t;
                if (__builtin_mul_overflow(r, a.u.i, &t)) too_big(pos);
                r = t;
            }
            return g_int(r);
        }
        double x = a.tag == GV_INT ? (double)a.u.i : a.u.f;
        double y = b.tag == GV_INT ? (double)b.u.i : b.u.f;
        double r = pow(x, y);
        /* ops.py: an integral result of two ints comes back as an int. */
        if (both_int && r == floor(r) && fabs(r) < 9.2e18) return g_int((int64_t)r);
        return g_float(r);
    }

    g_raise("BadGIR", pos, "unknown binary operator `%s`", op);
    return g_unit();
}

GValue g_binop(const char *op, GValue a, GValue b, const char *pos)
{
    if (!strcmp(op, "and")) return g_bool(g_truthy(a, pos) && g_truthy(b, pos));
    if (!strcmp(op, "or"))  return g_bool(g_truthy(a, pos) || g_truthy(b, pos));
    if (!strcmp(op, "=="))  return g_bool(g_equal(a, b));
    if (!strcmp(op, "!="))  return g_bool(!g_equal(a, b));
    if (!strcmp(op, "<") || !strcmp(op, ">") ||
        !strcmp(op, "<=") || !strcmp(op, ">="))
        return g_compare(op, a, b, pos);

    if (a.tag == GV_SECRET || b.tag == GV_SECRET)
        g_raise("SecretLeak", pos, "cannot perform arithmetic on a secret value");
    if (a.tag == GV_BOOL || b.tag == GV_BOOL)
        g_raise("TypeFault", pos, "operator `%s` is not defined for Bool operands", op);

    if (!strcmp(op, "+")) {
        if (a.tag == GV_TEXT && b.tag == GV_TEXT) {
            GText *t = (GText *)g_alloc(sizeof(GText));
            t->len = a.u.s->len + b.u.s->len;
            char *d = (char *)g_alloc(t->len + 1);
            memcpy(d, a.u.s->data, a.u.s->len);
            memcpy(d + a.u.s->len, b.u.s->data, b.u.s->len);
            d[t->len] = '\0';
            t->data = d;
            GValue v; v.tag = GV_TEXT; v.u.s = t; return v;
        }
        if (a.tag == GV_LIST && b.tag == GV_LIST) {
            GList *x = a.u.l, *y = b.u.l;
            GValue *items = (GValue *)g_alloc(sizeof(GValue) * (x->len + y->len + 1));
            memcpy(items, x->items, sizeof(GValue) * x->len);
            memcpy(items + x->len, y->items, sizeof(GValue) * y->len);
            return g_list_new(x->len + y->len, items);
        }
        if (a.tag == GV_TEXT || b.tag == GV_TEXT)
            g_raise("TypeFault", pos, "`+` cannot combine Text with a non-Text value");
    } else if (a.tag == GV_TEXT || b.tag == GV_TEXT) {
        g_raise("TypeFault", pos, "operator `%s` is not defined for Text operands", op);
    }

    if (!is_numeric(a) || !is_numeric(b))
        g_raise("TypeFault", pos, "operator `%s` is not defined for %s and %s",
                op, g_type_name(a), g_type_name(b));

    return num_binop(op, a, b, pos);
}

GValue g_unop(const char *op, GValue v, const char *pos)
{
    if (!strcmp(op, "!") || !strcmp(op, "not")) return g_bool(!g_truthy(v, pos));
    if (!strcmp(op, "-")) {
        if (v.tag == GV_BOOL)
            g_raise("TypeFault", pos, "unary `-` is not defined for Bool");
        if (v.tag == GV_INT) {
            if (v.u.i == INT64_MIN) too_big(pos);
            return g_int(-v.u.i);
        }
        if (v.tag == GV_FLOAT) return g_float(-v.u.f);
        g_raise("TypeFault", pos, "unary `-` is not defined for %s", g_type_name(v));
    }
    if (!strcmp(op, "+")) return v;
    g_raise("BadGIR", pos, "unknown unary operator `%s`", op);
    return v;
}

/* ================================================================== */
/* Range checks -- `vm.py::store` and `vm.py::coerce_return`           */
/* ================================================================== */

GValue g_store_int(GValue v, int64_t lo, int64_t hi, const char *type_name,
                   const char *slot, const char *pos)
{
    (void)slot;  /* named in the interpreter's context; kept for parity */
    if (v.tag == GV_INT && (v.u.i < lo || v.u.i > hi))
        g_raise("IntegerOverflow", pos,
                "value %lld does not fit in %s (range %lld to %lld)",
                (long long)v.u.i, type_name, (long long)lo, (long long)hi);
    return v;
}

GValue g_return_int(GValue v, int64_t lo, int64_t hi, const char *type_name,
                    const char *fn, const char *pos)
{
    if (v.tag == GV_INT && (v.u.i < lo || v.u.i > hi))
        g_raise("IntegerOverflow", pos,
                "`%s` returned %lld, which does not fit in %s",
                fn, (long long)v.u.i, type_name);
    return v;
}

/* ================================================================== */
/* Indexing and fields                                                 */
/* ================================================================== */

GValue g_index(GValue obj, GValue idx, const char *pos)
{
    long long i;
    if (idx.tag == GV_INT) i = (long long)idx.u.i;
    else g_raise("TypeFault", pos, "index must be an integer, not %s",
                 g_type_name(idx));

    switch (obj.tag) {
    case GV_LIST:
    case GV_TUPLE: {
        GList *l = obj.u.l;
        long long n = (long long)l->len;
        long long k = i < 0 ? i + n : i;
        if (k < 0 || k >= n)
            g_raise("IndexOutOfRange", pos,
                    "index %lld is out of range for a List of length %lld", i, n);
        return l->items[k];
    }
    case GV_TEXT: {
        GText *t = obj.u.s;
        long long n = (long long)t->len;
        long long k = i < 0 ? i + n : i;
        if (k < 0 || k >= n)
            g_raise("IndexOutOfRange", pos,
                    "index %lld is out of range for Text of length %lld", i, n);
        return g_text_n(t->data + k, 1);
    }
    case GV_MAP:
        return g_map_get(obj, idx, pos);
    default:
        g_raise("TypeFault", pos, "cannot index %s", g_type_name(obj));
        return g_unit();
    }
}

void g_set_index(GValue obj, GValue idx, GValue val, const char *pos)
{
    if (obj.tag == GV_LIST) {
        GList *l = obj.u.l;
        long long n = (long long)l->len;
        long long k = idx.tag == GV_INT ? (long long)idx.u.i : 0;
        if (k < 0) k += n;
        if (k < 0 || k >= n)
            g_raise("IndexOutOfRange", pos,
                    "index %lld is out of range for a List of length %lld",
                    (long long)(idx.tag == GV_INT ? idx.u.i : 0), n);
        l->items[k] = val;
        return;
    }
    if (obj.tag == GV_MAP) { g_map_set(obj, idx, val); return; }
    g_raise("TypeFault", pos, "cannot assign into %s", g_type_name(obj));
}

GValue g_field(GValue obj, const char *name, const char *pos)
{
    if (obj.tag == GV_SECRET)
        g_raise("SecretLeak", pos, "cannot read field `.%s` of a secret value",
                name);

    /* `vm.py::op_field` gives Result and Option these three names, and they are
     * not the same thing: `.value` unwraps (and faults on the error branch),
     * while `.error` reads the payload without unwrapping. */
    if (obj.tag == GV_RESULT) {
        if (!strcmp(name, "ok"))    return g_bool(strcmp(obj.u.n->kind, "ok") == 0);
        if (!strcmp(name, "value")) {
            if (strcmp(obj.u.n->kind, "ok") != 0)
                g_raise("TypeFault", pos, "called unwrap on fail(%s)",
                        g_display(obj.u.n->vals[0], 0));
            return obj.u.n->vals[0];
        }
        if (!strcmp(name, "error")) return obj.u.n->vals[0];
    }
    if (obj.tag == GV_OPTION) {
        int some = strcmp(obj.u.n->kind, "some") == 0;
        if (!strcmp(name, "some") || !strcmp(name, "is_some")) return g_bool(some);
        if (!strcmp(name, "value")) {
            if (!some) g_raise("TypeFault", pos, "called unwrap on none");
            return obj.u.n->vals[0];
        }
    }
    if (obj.tag == GV_VARIANT) {
        GNamed *r = obj.u.n;
        if (!strcmp(name, "tag")) return g_text(r->names[0]);
        if (!strcmp(name, "args")) {
            size_t n = r->len > 1 ? r->len - 1 : 0;
            GValue *items = (GValue *)g_alloc(sizeof(GValue) * (n ? n : 1));
            for (size_t i = 0; i < n; i++) items[i] = r->vals[i + 1];
            return g_list_new(n, items);
        }
    }
    if (obj.tag == GV_RECORD || obj.tag == GV_VARIANT ||
        obj.tag == GV_OPTION || obj.tag == GV_RESULT) {
        GNamed *r = obj.u.n;
        size_t start = (obj.tag == GV_VARIANT) ? 1 : 0;
        for (size_t i = start; i < r->len; i++)
            if (!strcmp(r->names[i], name)) return r->vals[i];
        /* A variant's payload is also reachable positionally, as `v.0`. */
        if (obj.tag == GV_VARIANT && r->len > 1)
            for (size_t i = 1; i < r->len; i++) {
                char buf[32];
                snprintf(buf, sizeof buf, "%zu", i - 1);
                if (!strcmp(buf, name)) return r->vals[i];
            }
    }
    if (obj.tag == GV_MAP) {
        GValue k = g_text(name);
        if (g_map_has(obj, k)) return g_map_get(obj, k, pos);
    }
    g_raise("NoSuchField", pos, "%s has no field `%s`", g_type_name(obj), name);
    return g_unit();
}

/* ================================================================== */
/* Output                                                              */
/* ================================================================== */

/* `std/library.py::_print`: space-separated, then a newline.  `print_raw`
 * writes the same text with no terminator, and `eprint` goes to stderr. */
static void emit_joined(FILE *out, size_t n, const GValue *args, int newline)
{
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(' ', out);
        fputs(g_display(args[i], 0), out);
    }
    if (newline) fputc('\n', out);
}

void g_println_n(size_t n, const GValue *args)   { emit_joined(stdout, n, args, 1); }
void g_print_raw_n(size_t n, const GValue *args) { emit_joined(stdout, n, args, 0); }
void g_eprint_n(size_t n, const GValue *args)    { emit_joined(stderr, n, args, 1); fflush(stderr); }

void g_print(GValue v)    { fputs(g_display(v, 0), stdout); }
void g_println(GValue v)  { fputs(g_display(v, 0), stdout); fputc('\n', stdout); }
void g_print_raw(GValue v){ fputs(g_display(v, 0), stdout); }
void g_eprint(GValue v)   { fputs(g_display(v, 0), stderr); fputc('\n', stderr); }

/* ================================================================== */
/* Builtins -- each mirrors a `@reg` entry in `std/library.py`          */
/* ================================================================== */

/* Number of UTF-8 code points, because `len` of a Text counts characters in
 * the interpreter (Python `len(str)`), not bytes. */
static size_t utf8_len(const char *s, size_t n)
{
    size_t c = 0;
    for (size_t i = 0; i < n; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80) c++;
    return c;
}

GValue g_len(GValue v)
{
    switch (v.tag) {
    case GV_TEXT:  return g_int((int64_t)utf8_len(v.u.s->data, v.u.s->len));
    case GV_LIST:
    case GV_TUPLE: return g_int((int64_t)v.u.l->len);
    case GV_MAP:   return g_int((int64_t)v.u.m->len);
    case GV_BYTES: return g_int(0);
    default:
        g_raise("TypeFault", "", "`len` does not apply to %s", g_type_name(v));
        return g_int(0);
    }
}

/* `vm.py::op_cast` */
GValue g_cast(GValue v, const char *target, const char *pos)
{
    if (v.tag == GV_SECRET)
        g_raise("SecretLeak", pos, "cannot cast a secret value");
    if (!strcmp(target, "I64") || !strncmp(target, "I", 1) ||
        !strncmp(target, "U", 1)) {
        if (v.tag == GV_FLOAT) {
            if (v.u.f != floor(v.u.f))
                g_raise("TypeFault", pos,
                        "cannot cast %s to %s without truncation",
                        g_display(v, 0), target);
            return g_int((int64_t)v.u.f);
        }
        if (v.tag == GV_INT) return v;
        if (v.tag == GV_BOOL) return g_int(v.u.b ? 1 : 0);
    }
    if (!strcmp(target, "F64")) {
        if (v.tag == GV_INT) return g_float((double)v.u.i);
        if (v.tag == GV_FLOAT) return v;
        if (v.tag == GV_BOOL) return g_float(v.u.b ? 1.0 : 0.0);
    }
    if (!strcmp(target, "Text")) return g_text(g_to_text(v));
    if (!strcmp(target, "Bool")) return g_bool(g_truthy(v, pos));
    return v;
}

/* `std/library.py::_float` */
GValue g_to_float(GValue v, const char *pos)
{
    if (v.tag == GV_INT) return g_float((double)v.u.i);
    if (v.tag == GV_FLOAT) return v;
    if (v.tag == GV_BOOL) return g_float(v.u.b ? 1.0 : 0.0);
    if (v.tag == GV_TEXT) {
        char *end = NULL;
        errno = 0;
        double d = strtod(v.u.s->data, &end);
        while (*end && isspace((unsigned char)*end)) end++;
        if (errno || end == v.u.s->data || *end)
            g_raise("TypeFault", pos, "cannot parse \"%s\" as a float",
                    v.u.s->data);
        return g_float(d);
    }
    g_raise("TypeFault", pos, "cannot convert %s to F64", g_type_name(v));
    return g_float(0);
}

/* `std/library.py::_int`.  Python's `int(s, 0)` rejects a leading zero that is
 * not a recognised prefix, where C's base-0 strtoll would read it as octal, so
 * the base is chosen explicitly. */
GValue g_to_int(GValue v, const char *pos)
{
    if (v.tag == GV_BOOL) return g_int(v.u.b ? 1 : 0);
    if (v.tag == GV_INT) return v;
    if (v.tag == GV_FLOAT) {
        if (v.u.f != floor(v.u.f))
            g_raise("TypeFault", pos,
                    "cannot convert %s to I64 without truncation",
                    g_display(v, 0));
        return g_int((int64_t)v.u.f);
    }
    if (v.tag == GV_TEXT) {
        const char *p = v.u.s->data;
        while (*p && isspace((unsigned char)*p)) p++;
        int neg = 0;
        if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
        int base = 10;
        if ((p[0] == '0') && (p[1] == 'x' || p[1] == 'X')) { base = 16; p += 2; }
        else if ((p[0] == '0') && (p[1] == 'o' || p[1] == 'O')) { base = 8; p += 2; }
        else if ((p[0] == '0') && (p[1] == 'b' || p[1] == 'B')) { base = 2; p += 2; }
        else if (p[0] == '0' && p[1] && isdigit((unsigned char)p[1]))
            g_raise("TypeFault", pos, "cannot parse \"%s\" as an integer",
                    v.u.s->data);
        char *end = NULL;
        errno = 0;
        long long r = strtoll(p, &end, base);
        while (*end && isspace((unsigned char)*end)) end++;
        if (errno == ERANGE || end == p || *end)
            g_raise("TypeFault", pos, "cannot parse \"%s\" as an integer",
                    v.u.s->data);
        return g_int(neg ? -r : r);
    }
    g_raise("TypeFault", pos, "cannot convert %s to I64", g_type_name(v));
    return g_int(0);
}

GValue g_abs(GValue v, const char *pos)
{
    if (v.tag == GV_INT) {
        if (v.u.i == INT64_MIN)
            g_raise("IntegerOverflow", pos,
                    "the native backend computes integers in 64 bits, and this "
                    "result does not fit; the reference interpreter computes it "
                    "exactly");
        return g_int(v.u.i < 0 ? -v.u.i : v.u.i);
    }
    if (v.tag == GV_FLOAT) return g_float(fabs(v.u.f));
    g_raise("TypeFault", pos, "`abs` does not apply to %s", g_type_name(v));
    return g_int(0);
}

GValue g_contains(GValue hay, GValue needle, const char *pos)
{
    if (hay.tag == GV_TEXT) {
        if (needle.tag != GV_TEXT)
            g_raise("TypeFault", pos, "cannot search Text for %s",
                    g_type_name(needle));
        if (needle.u.s->len > hay.u.s->len) return g_bool(0);
        for (size_t i = 0; i + needle.u.s->len <= hay.u.s->len; i++)
            if (!memcmp(hay.u.s->data + i, needle.u.s->data, needle.u.s->len))
                return g_bool(1);
        return g_bool(0);
    }
    if (hay.tag == GV_LIST || hay.tag == GV_TUPLE) {
        for (size_t i = 0; i < hay.u.l->len; i++)
            if (g_equal(hay.u.l->items[i], needle)) return g_bool(1);
        return g_bool(0);
    }
    if (hay.tag == GV_MAP) return g_bool(g_map_has(hay, needle));
    g_raise("TypeFault", pos, "`contains` does not apply to %s", g_type_name(hay));
    return g_bool(0);
}

static GValue minmax(GValue a, GValue b, const char *pos, int want_min)
{
    GValue c = g_compare(want_min ? "<" : ">", a, b, pos);
    return c.u.b ? a : b;
}

GValue g_min(GValue a, GValue b, const char *pos) { return minmax(a, b, pos, 1); }
GValue g_max(GValue a, GValue b, const char *pos) { return minmax(a, b, pos, 0); }

GValue g_min_of(GValue v, const char *pos)
{
    if (v.tag != GV_LIST && v.tag != GV_TUPLE)
        g_raise("TypeFault", pos, "`min` expects a List or two values");
    if (!v.u.l->len) g_raise("ValueError", pos, "`min` of an empty List");
    GValue best = v.u.l->items[0];
    for (size_t i = 1; i < v.u.l->len; i++) best = minmax(best, v.u.l->items[i], pos, 1);
    return best;
}

GValue g_max_of(GValue v, const char *pos)
{
    if (v.tag != GV_LIST && v.tag != GV_TUPLE)
        g_raise("TypeFault", pos, "`max` expects a List or two values");
    if (!v.u.l->len) g_raise("ValueError", pos, "`max` of an empty List");
    GValue best = v.u.l->items[0];
    for (size_t i = 1; i < v.u.l->len; i++) best = minmax(best, v.u.l->items[i], pos, 0);
    return best;
}

GValue g_sum(GValue v, const char *pos)
{
    if (v.tag != GV_LIST && v.tag != GV_TUPLE)
        g_raise("TypeFault", pos, "`sum` expects a List");
    GValue acc = g_int(0);
    for (size_t i = 0; i < v.u.l->len; i++)
        acc = g_binop("+", acc, v.u.l->items[i], pos);
    return acc;
}

/* `sorted` uses Python's stable sort over `compare`.  Insertion sort is stable
 * and adequate for the sizes the differential corpus uses; the ordering
 * relation, not the algorithm, is what must match. */
GValue g_sorted(GValue v, const char *pos)
{
    if (v.tag != GV_LIST && v.tag != GV_TUPLE)
        g_raise("TypeFault", pos, "`sorted` expects a List");
    size_t n = v.u.l->len;
    GValue *items = (GValue *)g_alloc(sizeof(GValue) * (n ? n : 1));
    memcpy(items, v.u.l->items, sizeof(GValue) * n);
    for (size_t i = 1; i < n; i++) {
        GValue key = items[i];
        size_t j = i;
        while (j > 0 && g_truthy(g_compare(">", items[j - 1], key, pos), pos)) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
    return g_list_new(n, items);
}

/* `std/library.py::_range`: range(n), range(a,b), range(a,b,step) */
GValue g_range(int nargs, const GValue *args, const char *pos)
{
    int64_t start = 0, stop = 0, step = 1;
    if (nargs == 1) { stop = g_to_int(args[0], pos).u.i; }
    else if (nargs == 2) { start = g_to_int(args[0], pos).u.i;
                           stop = g_to_int(args[1], pos).u.i; }
    else if (nargs == 3) { start = g_to_int(args[0], pos).u.i;
                           stop = g_to_int(args[1], pos).u.i;
                           step = g_to_int(args[2], pos).u.i;
                           if (step == 0)
                               g_raise("ValueError", pos, "`range` step cannot be zero"); }
    else g_raise("ValueError", pos, "`range` takes at most 3 arguments");

    size_t cap = 16, n = 0;
    GValue *items = (GValue *)g_alloc(sizeof(GValue) * cap);
    if (step > 0) {
        for (int64_t i = start; i < stop; i += step) {
            if (n == cap) { cap *= 2; GValue *ni = (GValue *)g_alloc(sizeof(GValue)*cap);
                            memcpy(ni, items, sizeof(GValue)*n); items = ni; }
            items[n++] = g_int(i);
        }
    } else {
        for (int64_t i = start; i > stop; i += step) {
            if (n == cap) { cap *= 2; GValue *ni = (GValue *)g_alloc(sizeof(GValue)*cap);
                            memcpy(ni, items, sizeof(GValue)*n); items = ni; }
            items[n++] = g_int(i);
        }
    }
    return g_list_new(n, items);
}

/* The hidden `__range` builtin behind `a..b` and `a..=b`. */
GValue g_range_span(GValue start, GValue end, int inclusive, const char *pos)
{
    double a = g_to_float(start, pos).u.f;
    double b = g_to_float(end, pos).u.f;
    double step = a <= b ? 1.0 : -1.0;
    double limit = inclusive ? b + step : b;
    size_t cap = 16, n = 0;
    GValue *items = (GValue *)g_alloc(sizeof(GValue) * cap);
    for (double x = a; (step > 0 ? x < limit : x > limit); x += step) {
        if (n == cap) { cap *= 2; GValue *ni = (GValue *)g_alloc(sizeof(GValue)*cap);
                        memcpy(ni, items, sizeof(GValue)*n); items = ni; }
        items[n++] = (start.tag == GV_INT && end.tag == GV_INT)
                     ? g_int((int64_t)x) : g_float(x);
    }
    return g_list_new(n, items);
}

GValue g_map_literal(size_t n, const GValue *items)
{
    GValue m = g_map_new();
    for (size_t i = 0; i + 1 < n; i += 2) g_map_set(m, items[i], items[i + 1]);
    return m;
}

/* `math.*`.  `math.round` is half away from zero and returns I64, matching
 * `std/library.py::_math_round` -- not C's nearbyint, which follows the
 * current rounding mode. */
GValue g_math_round(GValue x, const char *pos) { return g_int((int64_t)round(g_to_float(x, pos).u.f)); }
GValue g_math_floor(GValue x, const char *pos) { return g_int((int64_t)floor(g_to_float(x, pos).u.f)); }
GValue g_math_ceil (GValue x, const char *pos) { return g_int((int64_t)ceil (g_to_float(x, pos).u.f)); }
GValue g_math_sqrt (GValue x, const char *pos) { return g_float(sqrt(g_to_float(x, pos).u.f)); }
GValue g_math_log  (GValue x, const char *pos) { return g_float(log (g_to_float(x, pos).u.f)); }
GValue g_math_exp  (GValue x, const char *pos) { return g_float(exp (g_to_float(x, pos).u.f)); }
GValue g_math_pow  (GValue a, GValue b, const char *pos)
{ return g_float(pow(g_to_float(a, pos).u.f, g_to_float(b, pos).u.f)); }
GValue g_math_clamp(GValue x, GValue lo, GValue hi, const char *pos)
{ return g_min(g_max(x, lo, pos), hi, pos); }

/* ================================================================== */
/* Capabilities (spec section 12)                                      */
/* ================================================================== */

/* Authority is a decision the deployment makes, never the program.  The
 * generated binary therefore takes its grants from the command line, and the
 * program's own `grant`/`authority` declarations are honoured only when the
 * caller has not asked otherwise -- the same choice `ggc run` makes, in the
 * same place, for the same reason: a program that can confer a capability on
 * itself by writing it down has ambient authority under another name.
 *
 * The coverage relation below is a transcription of
 * `compiler/gamag/capabilities.py`.  It is deliberately not a second
 * definition.  A backend that reasoned about capabilities differently from
 * the checker would deny programs the checker had accepted, and the denial
 * would look like a bug in the program rather than a disagreement between two
 * halves of one toolchain. */

GCaps g_caps;

/*: The permissions of the vocabulary, in the order `capabilities.PERMISSIONS`
 *: lists them.  A written name is split on the matching suffix. */
static const char *const G_PERMISSIONS[] = {
    "Read", "Write", "Connect", "Sign", "Spawn", "Expose", "Load", NULL
};

static void g_cap_trim(char *s)
{
    char *p = s;
    while (*p && isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
}

/* Split a written capability into resource and permission.  Two spellings are
 * accepted, both from the specification: `PatientRead` (bare) and
 * `PatientStore[Read]` (section 12's qualified form). */
static void g_cap_parse(const char *name, char *resource, size_t rn,
                        char *permission, size_t pn)
{
    resource[0] = '\0';
    permission[0] = '\0';
    size_t n = strlen(name);

    if (n > 1 && name[n - 1] == ']') {
        const char *open = strchr(name, '[');
        if (open) {
            size_t rlen = (size_t)(open - name);
            if (rlen >= rn) rlen = rn - 1;
            memcpy(resource, name, rlen);
            resource[rlen] = '\0';
            size_t plen = n - (size_t)(open - name) - 2;   /* inside [ ] */
            if (plen >= pn) plen = pn - 1;
            memcpy(permission, open + 1, plen);
            permission[plen] = '\0';
            g_cap_trim(permission);
            /* `PatientStore[Read]` and `PatientRead` name the same access. */
            size_t rl = strlen(resource);
            if (rl > 5 && strcmp(resource + rl - 5, "Store") == 0)
                resource[rl - 5] = '\0';
            return;
        }
    }
    for (size_t i = 0; G_PERMISSIONS[i]; i++) {
        size_t pl = strlen(G_PERMISSIONS[i]);
        if (n > pl && strcmp(name + n - pl, G_PERMISSIONS[i]) == 0) {
            size_t rlen = n - pl;
            if (rlen >= rn) rlen = rn - 1;
            memcpy(resource, name, rlen);
            resource[rlen] = '\0';
            snprintf(permission, pn, "%s", G_PERMISSIONS[i]);
            return;
        }
    }
    snprintf(resource, rn, "%s", name);   /* no permission: the whole resource */
}

/*: Which permissions entail which others, on the same resource.  A decision,
 *: written down with its reason in `capabilities.py`: writing a store entails
 *: reading it back, and exposing or loading a secret reads it first.  The
 *: actions (`Connect`, `Sign`, `Spawn`) entail nothing. */
static const char *g_cap_implies(const char *permission)
{
    if (strcmp(permission, "Write") == 0)  return "Read";
    if (strcmp(permission, "Expose") == 0) return "Read";
    if (strcmp(permission, "Load") == 0)   return "Read";
    return NULL;
}

static int g_cap_one_grants(const char *held, const char *wanted)
{
    char hr[G_RES_MAX], hp[G_PERM_MAX], wr[G_RES_MAX], wp[G_PERM_MAX];
    g_cap_parse(held, hr, sizeof hr, hp, sizeof hp);
    g_cap_parse(wanted, wr, sizeof wr, wp, sizeof wp);

    if (strcmp(hr, wr) != 0) return 0;
    if (hp[0] == '\0') return 1;     /* unrestricted on this resource */
    if (wp[0] == '\0') return 0;     /* the demand is the whole resource */
    if (strcmp(hp, wp) == 0) return 1;
    const char *implied = g_cap_implies(hp);
    return implied && strcmp(implied, wp) == 0;
}

/* Whether the granted set satisfies a demand, using coverage rather than
 * membership: as in `covers()`, `"*"` covers everything, and an intent holding
 * `PatientWrite` satisfies a demand for `PatientRead` at run time exactly as it
 * did at compile time.  A membership test here would deny programs the checker
 * accepted. */
int g_cap_covers(const char *wanted)
{
    for (size_t i = 0; i < g_caps.len; i++)
        if (strcmp(g_caps.names[i], "*") == 0) return 1;
    for (size_t i = 0; i < g_caps.len; i++)
        if (g_cap_one_grants(g_caps.names[i], wanted)) return 1;
    return 0;
}

void g_cap_reset(void)
{
    g_caps.len = 0;
    g_caps.denials = 0;
    g_caps.strict_authority = 0;
}

void g_cap_grant(const char *name)
{
    if (!name || !*name) return;
    if (g_caps.len >= G_CAP_MAX) {
        fprintf(stderr, "warning: more than %d grants; `%s` ignored\n",
                G_CAP_MAX, name);
        return;
    }
    /* A repeated grant is one grant: the interpreter's grants are a set, and
     * a list that grew with every repetition would make `--grant FileRead`
     * twice mean something else than once. */
    for (size_t i = 0; i < g_caps.len; i++)
        if (strcmp(g_caps.names[i], name) == 0) return;
    g_caps.names[g_caps.len++] = name;
}

int g_cap_granted_count(void) { return (int)g_caps.len; }

long g_cap_denials(void) { return g_caps.denials; }

/* Spec section 12, enforced at the point of use.  The message is the
 * interpreter's, word for word, so that a program which is denied under one
 * and permitted under the other is a real difference rather than two
 * spellings of the same refusal. */
void g_cap_require(const char *capability, const char *what, const char *pos)
{
    if (g_cap_covers(capability)) return;
    g_caps.denials++;
    if (what && *what)
        g_raise("CapabilityViolation", pos,
                "operation requires the `%s` capability, which was not "
                "granted for %s", capability, what);
    g_raise("CapabilityViolation", pos,
            "operation requires the `%s` capability, which was not granted",
            capability);
}

/* ================================================================== */
/* Capability-gated standard library                                   */
/* ================================================================== */

/* `io.read_file` and friends.  These are the operations the capability system
 * exists for, so they are the ones that make enforcement observable rather
 * than notional: without them the native runtime could only refuse whole
 * programs, which is not a security model. */

static char *g_read_whole_file(const char *path, const char *pos, int for_write)
{
    FILE *fh = fopen(path, for_write ? "wb" : "rb");
    if (!fh) {
        if (for_write)
            g_raise("BuiltinFault", pos, "cannot write '%s': %s", path,
                    strerror(errno));
        g_raise("BuiltinFault", pos, "cannot read '%s': %s", path,
                strerror(errno));
    }
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) g_raise("BuiltinFault", pos, "out of memory reading '%s'", path);
    for (;;) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *grown = (char *)realloc(buf, cap);
            if (!grown) { free(buf);
                g_raise("BuiltinFault", pos, "out of memory reading '%s'", path); }
            buf = grown;
        }
        size_t got = fread(buf + len, 1, cap - len - 1, fh);
        len += got;
        if (got == 0) break;
    }
    int failed = ferror(fh);
    fclose(fh);
    if (failed) {
        free(buf);
        g_raise("BuiltinFault", pos, "cannot read '%s': %s", path,
                strerror(errno));
    }
    buf[len] = '\0';
    return buf;
}

GValue g_io_read_file(GValue path, const char *pos)
{
    const char *p = g_to_text(path);
    g_cap_require("FileRead", "io.read_file", pos);
    char *data = g_read_whole_file(p, pos, 0);
    GValue v = g_text_n(data, strlen(data));
    free(data);
    return v;
}

GValue g_io_write_file(GValue path, GValue content, const char *pos)
{
    const char *p = g_to_text(path);
    g_cap_require("FileWrite", "io.write_file", pos);
    const char *text = g_to_text(content);
    FILE *fh = fopen(p, "wb");
    if (!fh)
        g_raise("BuiltinFault", pos, "cannot write '%s': %s", p,
                strerror(errno));
    size_t n = strlen(text);
    int ok = (n == 0) || fwrite(text, 1, n, fh) == n;
    int err = errno;
    fclose(fh);
    if (!ok)
        g_raise("BuiltinFault", pos, "cannot write '%s': %s", p, strerror(err));
    return g_unit();
}

GValue g_io_append_file(GValue path, GValue content, const char *pos)
{
    const char *p = g_to_text(path);
    g_cap_require("FileWrite", "io.append_file", pos);
    const char *text = g_to_text(content);
    FILE *fh = fopen(p, "ab");
    if (!fh)
        g_raise("BuiltinFault", pos, "cannot write '%s': %s", p,
                strerror(errno));
    size_t n = strlen(text);
    int ok = (n == 0) || fwrite(text, 1, n, fh) == n;
    int err = errno;
    fclose(fh);
    if (!ok)
        g_raise("BuiltinFault", pos, "cannot write '%s': %s", p, strerror(err));
    return g_unit();
}

GValue g_io_exists(GValue path, const char *pos)
{
    const char *p = g_to_text(path);
    g_cap_require("FileRead", "io.exists", pos);
    FILE *fh = fopen(p, "rb");
    if (fh) { fclose(fh); return g_bool(1); }
    return g_bool(0);
}

/* ================================================================== */
/* Entry point                                                         */
/* ================================================================== */

int g_run(int argc, char **argv,
          const char *const *declared_grants, size_t declared_grants_n)
{
    g_caps.declared = declared_grants;
    g_caps.declared_n = declared_grants_n;
    static const char *caller_grants[G_CAP_MAX];
    size_t caller_n = 0;
    int strict = 0;
    int show_authority = 0;

    g_cap_reset();
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--grant") == 0 && i + 1 < argc) {
            if (caller_n < G_CAP_MAX) caller_grants[caller_n++] = argv[++i];
            else i++;
        } else if (strncmp(a, "--grant=", 8) == 0) {
            if (caller_n < G_CAP_MAX) caller_grants[caller_n++] = a + 8;
        } else if (strcmp(a, "--strict-authority") == 0) {
            strict = 1;
        } else if (strcmp(a, "--authority") == 0) {
            show_authority = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            printf("usage: %s [--grant CAP]... [--strict-authority] "
                   "[--authority]\n\n"
                   "  --grant CAP         grant a capability the program did "
                   "not ask for (repeatable)\n"
                   "  --strict-authority  grant only what --grant names, "
                   "ignoring the capabilities the program declares; pass this\n"
                   "                      when running code you have not read\n"
                   "  --authority         print the authority this run has and "
                   "exit\n", argv[0]);
            return G_EXIT_OK;
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "unknown option: %s (try --help)\n", a);
            return G_EXIT_USAGE;
        }
    }

    /* Authority comes from the caller.  The program's own declarations are
     * honoured here because the user chose to run this file and its grant
     * lines are visible in it -- the same decision `ggc run` makes, and
     * `--strict-authority` refuses it.  This is the only place the choice is
     * taken, and it is one line long so that it stays reviewable. */
    g_caps.strict_authority = strict;
    if (!strict)
        for (size_t i = 0; i < declared_grants_n; i++)
            g_cap_grant(declared_grants[i]);
    for (size_t i = 0; i < caller_n; i++)
        g_cap_grant(caller_grants[i]);

    if (show_authority) {
        printf("authority: %d capability name(s), %s\n", g_cap_granted_count(),
               strict ? "program declarations refused (--strict-authority)"
                      : "including the program's own declarations");
        for (size_t i = 0; i < g_caps.len; i++)
            printf("  %s\n", g_caps.names[i]);
        return G_EXIT_OK;
    }

    if (setjmp(g_fault_jmp)) {
        fflush(stdout);
        if (g_fault.pos[0])
            fprintf(stderr, "runtime fault [%s] at %s: %s\n",
                    g_fault.kind, g_fault.pos, g_fault.message);
        else
            fprintf(stderr, "runtime fault [%s]: %s\n",
                    g_fault.kind, g_fault.message);
        return G_EXIT_RUNTIME;
    }
    g_main(argc, argv);
    fflush(stdout);
    return G_EXIT_OK;
}
