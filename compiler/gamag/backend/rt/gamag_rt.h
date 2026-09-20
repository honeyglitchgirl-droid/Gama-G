/* Gama-G native runtime (spec section 22, stages 11-12).
 *
 * This is the support library for programs the native CPU backend emits.  Its
 * contract is narrow and absolute: for every value and every operator, it must
 * produce what `compiler/gamag/runtime/ops.py` and `values.py` produce, because
 * the only honest way to know a backend is correct is to run both and compare
 * (`ggc difftest`, audit priority 7).
 *
 * Three properties of the reference interpreter are easy to get wrong here, so
 * they are written down rather than left to be discovered:
 *
 *   1. Integers are exact in Python and range-checked at *store* and *return*
 *      against the declared type (`vm.py::store`).  C integers wrap.  So every
 *      arithmetic result is computed with the overflow-checking builtins and
 *      the range check happens where the interpreter puts it, not where C would.
 *   2. `display()` formats an integral float as `%.1f` and otherwise uses
 *      Python's shortest-round-trip repr, which switches to exponential outside
 *      `-4 < decpt <= 16`.  `%.17g` does not match that; `g_repr_float` does.
 *   3. Integer division truncates toward zero and `%` takes the sign of the
 *      dividend (`ops.py::trunc_div`, `trunc_mod`).  That is C's own behaviour,
 *      so it is used directly rather than reimplemented.
 *
 * Memory: an arena.  Nothing is freed until the process exits.  This is a
 * deliberate, documented limitation -- the language's deterministic-destruction
 * guarantee (spec section 8) is a property the *memory model* computes over the
 * derived graph, and `ggc memory` reports it; the native runtime does not yet
 * act on those extents.  Scalars live in C locals and cost nothing.  See
 * docs/DESIGN_v1_0.md, "What the native backend does not do".
 */

#ifndef GAMAG_RT_H
#define GAMAG_RT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <setjmp.h>

/* ------------------------------------------------------------------ */
/* Values                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    GV_UNIT, GV_BOOL, GV_INT, GV_FLOAT, GV_TEXT, GV_BYTES,
    GV_LIST, GV_MAP, GV_TUPLE, GV_RECORD, GV_VARIANT,
    GV_OPTION, GV_RESULT, GV_SECRET, GV_FUNC
} GTag;

typedef struct GValue GValue;

typedef struct GText  { size_t len; char *data; } GText;              /* NUL-terminated */
typedef struct GList  { size_t len, cap; GValue *items; } GList;
/* Map keys are values, not strings: the interpreter's `dict` accepts any
 * hashable key, and keying by the *rendered* form would make `1` and `"1"`
 * collide.  Lookup is a linear scan with `g_equal`, which is O(n) and correct;
 * the examples and the differential corpus are small, and a hash table is a
 * performance change that must not change semantics. */
typedef struct GMap   { size_t len, cap; GValue *keys; GValue *vals; } GMap;
typedef struct GNamed { size_t len, cap; char **names; GValue *vals;
                        const char *kind; } GNamed;   /* record / variant */
typedef struct GSecret { const char *label; GValue *inner; } GSecret;

struct GValue {
    GTag tag;
    union {
        int       b;
        int64_t   i;
        double    f;
        GText    *s;
        GList    *l;
        GMap     *m;
        GNamed   *n;      /* tuple, record, variant */
        GSecret  *sec;
        void     *p;
    } u;
};

/* ------------------------------------------------------------------ */
/* Faults                                                              */
/* ------------------------------------------------------------------ */

#define G_MSG_MAX 1024

typedef struct GFault {
    const char *kind;
    char        message[G_MSG_MAX];
    char        pos[160];
    int         active;
} GFault;

extern GFault   g_fault;
extern jmp_buf  g_fault_jmp;

/* Exit statuses, matching `compiler/gamag/cli/main.py`. */
#define G_EXIT_OK      0
#define G_EXIT_RUNTIME 3

void g_raise(const char *kind, const char *pos, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4), noreturn))
#endif
    ;

/* ------------------------------------------------------------------ */
/* Arena                                                               */
/* ------------------------------------------------------------------ */

void  *g_alloc(size_t n);
char  *g_strdup_n(const char *s, size_t n);
char  *g_strdup(const char *s);
void   g_arena_reset(void);
size_t g_arena_bytes(void);

/* ------------------------------------------------------------------ */
/* Constructors                                                        */
/* ------------------------------------------------------------------ */

GValue g_unit(void);
GValue g_bool(int b);
GValue g_int(int64_t i);
GValue g_float(double f);
GValue g_text(const char *s);
GValue g_text_n(const char *s, size_t n);
GValue g_list_new(size_t n, const GValue *items);
GValue g_tuple_new(size_t n, const GValue *items);
GValue g_map_new(void);
GValue g_variant(const char *kind, const char *tag, size_t n, const GValue *args);
GValue g_option(int some, GValue value);
GValue g_result(int ok, GValue value);
GValue g_secret(const char *label, GValue inner);
GValue g_record(const char *name, size_t n, const char **fields, const GValue *vals);

void   g_list_push(GValue list, GValue item);
void   g_map_set(GValue map, GValue key, GValue val);
GValue g_map_get(GValue map, GValue key, const char *pos);
int    g_map_has(GValue map, GValue key);

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

/* `values.py::display`.  Returns arena memory. */
char  *g_display(GValue v, int allow_secret);
/* `values.py::to_text` -- display with the secret barrier enforced. */
char  *g_to_text(GValue v);
/* Python's repr() for a double, exactly.  Writes into `out`. */
void   g_repr_float(double v, char *out, size_t n);
/* `values.py::type_name`. */
const char *g_type_name(GValue v);
/* `ops.py::display` of a value inside a fault message. */
char  *g_show(GValue v);

/* ------------------------------------------------------------------ */
/* Predicates and operators                                            */
/* ------------------------------------------------------------------ */

int    g_truthy(GValue v, const char *pos);
int    g_equal(GValue a, GValue b);
GValue g_compare(const char *op, GValue a, GValue b, const char *pos);
GValue g_binop(const char *op, GValue a, GValue b, const char *pos);
GValue g_unop(const char *op, GValue v, const char *pos);
GValue g_cast(GValue v, const char *target, const char *pos);

/* `vm.py::store`'s range check for an integer slot. */
GValue g_store_int(GValue v, int64_t lo, int64_t hi, const char *type_name,
                   const char *slot, const char *pos);
/* `vm.py::coerce_return`'s range check. */
GValue g_return_int(GValue v, int64_t lo, int64_t hi, const char *type_name,
                    const char *fn, const char *pos);

/* ------------------------------------------------------------------ */
/* Indexing                                                            */
/* ------------------------------------------------------------------ */

GValue g_index(GValue obj, GValue idx, const char *pos);
void   g_set_index(GValue obj, GValue idx, GValue val, const char *pos);
GValue g_field(GValue obj, const char *name, const char *pos);

/* ------------------------------------------------------------------ */
/* Output                                                              */
/* ------------------------------------------------------------------ */

void g_println_n(size_t n, const GValue *args);
void g_print_raw_n(size_t n, const GValue *args);
void g_eprint_n(size_t n, const GValue *args);
void g_print(GValue v);
void g_println(GValue v);
void g_print_raw(GValue v);
void g_eprint(GValue v);

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Builtins                                                            */
/* ------------------------------------------------------------------ */

GValue g_len(GValue v);
GValue g_to_float(GValue v, const char *pos);
GValue g_to_int(GValue v, const char *pos);
GValue g_abs(GValue v, const char *pos);
GValue g_contains(GValue hay, GValue needle, const char *pos);
GValue g_min(GValue a, GValue b, const char *pos);
GValue g_max(GValue a, GValue b, const char *pos);
GValue g_min_of(GValue v, const char *pos);
GValue g_max_of(GValue v, const char *pos);
GValue g_sum(GValue v, const char *pos);
GValue g_sorted(GValue v, const char *pos);
GValue g_range(int nargs, const GValue *args, const char *pos);
GValue g_range_span(GValue start, GValue end, int inclusive, const char *pos);
GValue g_map_literal(size_t n, const GValue *items);
GValue g_math_round(GValue x, const char *pos);
GValue g_math_floor(GValue x, const char *pos);
GValue g_math_ceil(GValue x, const char *pos);
GValue g_math_sqrt(GValue x, const char *pos);
GValue g_math_log(GValue x, const char *pos);
GValue g_math_exp(GValue x, const char *pos);
GValue g_math_pow(GValue a, GValue b, const char *pos);
GValue g_math_clamp(GValue x, GValue lo, GValue hi, const char *pos);

/* Emitted by the backend. */
GValue g_main(int argc, char **argv);

int g_run(int argc, char **argv);

#endif /* GAMAG_RT_H */
