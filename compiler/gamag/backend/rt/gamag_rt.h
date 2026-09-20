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

#include <math.h>
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

/* Exit statuses, matching `compiler/gamag/cli/main.py` exactly.  They did not:
 * a runtime fault exited 3 here and 2 under `ggc run`, and the differential
 * harness hid the difference by normalising the interpreter's status to the
 * native one -- so the comparison could never have caught it.  `3` is a usage
 * error, and a fault is not a usage error. */
#define G_EXIT_OK        0
#define G_EXIT_COMPILE   1
#define G_EXIT_RUNTIME   2
#define G_EXIT_USAGE     3

/* ------------------------------------------------------------------ */
/* Capabilities (spec section 12)                                      */
/* ------------------------------------------------------------------ */

#define G_CAP_MAX  64
#define G_RES_MAX  96
#define G_PERM_MAX 24

typedef struct GCaps {
    const char *names[G_CAP_MAX];
    size_t      len;
    int         strict_authority;
    long        denials;
    /*: What the program declared, supplied by the generated `main`. */
    const char *const *declared;
    size_t      declared_n;
} GCaps;

extern GCaps g_caps;


int  g_cap_covers(const char *wanted);
void g_cap_grant(const char *name);
void g_cap_reset(void);
void g_cap_require(const char *capability, const char *what, const char *pos);
int  g_cap_granted_count(void);
long g_cap_denials(void);

void g_raise(const char *kind, const char *pos, const char *fmt, ...)
/* A specialized function's boxed wrapper exists for callers that were not
 * specialized.  When a program has none -- the common case of a leaf helper
 * called only from specialized code -- the wrapper is a definition nothing
 * references, and `-Wall -Wextra` says so.  Saying "this may be unused" is more
 * honest than deleting the wrapper, which would make the emitter's output
 * depend on which functions happen to call it. */
#if defined(__GNUC__) || defined(__clang__)
#  define G_MAYBE_UNUSED __attribute__((unused))
#else
#  define G_MAYBE_UNUSED
#endif

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

/* Arena release path.  A mark records a position; releasing it returns every
 * byte allocated since to the arena.  Marks nest and are released newest
 * first, because a function marks on entry and releases before returning.
 *
 * This exists so that a program which repeatedly calls a function does not
 * grow without bound: without it the arena only ever grew, and a long-running
 * program leaked by construction. */
typedef struct {
    struct GChunk *chunk;           /* struct GChunk is private to the .c file */
    size_t         used;
} GArenaMark;

GArenaMark g_arena_mark(void);
void       g_arena_release(GArenaMark mark);

/* Bytes handed out and not released (the leak measure) and its high-water
 * mark.  `g_arena_bytes` is cumulative and only ever grows. */
size_t g_arena_live(void);
size_t g_arena_peak(void);

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
GValue g_io_read_file(GValue path, const char *pos);
GValue g_io_write_file(GValue path, GValue content, const char *pos);
GValue g_io_append_file(GValue path, GValue content, const char *pos);
GValue g_io_exists(GValue path, const char *pos);

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

/* ================================================================== */
/* Scalar specialization                                              */
/* ================================================================== */
/*
 * `ggc native` emits unboxed C for a function whose values are all scalars.
 * The generic path represents every value as a `GValue` and dispatches every
 * operator through `g_binop`, which selects the operation by comparing the
 * operator's *name* with `strcmp` at run time and returns a tagged union by
 * value.  That is correct and it is what makes a `while` loop cost roughly a
 * hundred nanoseconds per iteration; a loop of integer arithmetic compiled
 * this way measured about 900x the same loop in C.
 *
 * These helpers exist so the specialized path can keep the *semantics* of
 * `g_binop`/`g_unop` exactly -- including which fault is raised, and with what
 * message -- while the arguments and the result stay in machine registers.
 * They are `static inline` so the C compiler sees them in the generated
 * translation unit and can optimize across them rather than calling out.
 *
 * Every one of them mirrors a branch of `num_binop` or `g_unop` in
 * `gamag_rt.c`.  If one of those changes, this must change with it; the
 * differential test is what catches the pair going out of step.
 */

/* Raise the same fault `too_big` raises.  Declared here, defined in the
 * runtime, so the inline helpers below can call it. */
void g_int_overflow(const char *pos);

/* Raise the same fault `num_binop` raises for a zero divisor. */
void g_zero_division(const char *pos, const char *what);
/* The one place "division by zero: %s / %s" is written. */
void g_div_zero_values(const char *pos, GValue a, GValue b);

/*
 * Unboxing at a call boundary.  A specialized function keeps its values in
 * machine registers, but it can still be *called* from a function that was not
 * specialized -- `main` calls a helper, a helper calls the next one down.  So
 * every specialized function is emitted twice: the body under one name, and a
 * wrapper under the generic name that unboxes its arguments, calls the body,
 * and boxes the result.  Unboxing happens here, once per call, instead of on
 * every operation inside the callee.
 *
 * The tag check cannot fail in a program that type-checked: the slot's declared
 * type is what the checker enforced, and it is the same declaration that chose
 * the C type.  It is checked anyway, because a silent reinterpretation of the
 * wrong union member is the one failure mode that would make the two emitters
 * disagree about a program without saying so.
 */
static inline int64_t g_unbox_i64(GValue v, const char *fn, const char *slot)
{
    if (v.tag != GV_INT)
        g_raise("TypeFault", "", "`%s`: parameter `%s` is declared I64 but "
                "holds %s", fn, slot, g_display(v, 0));
    return v.u.i;
}

static inline double g_unbox_f64(GValue v, const char *fn, const char *slot)
{
    if (v.tag != GV_FLOAT)
        g_raise("TypeFault", "", "`%s`: parameter `%s` is declared F64 but "
                "holds %s", fn, slot, g_display(v, 0));
    return v.u.f;
}

static inline int g_unbox_bool(GValue v, const char *fn, const char *slot)
{
    if (v.tag != GV_BOOL)
        g_raise("TypeFault", "", "`%s`: parameter `%s` is declared Bool but "
                "holds %s", fn, slot, g_display(v, 0));
    return v.u.b ? 1 : 0;
}

static inline int64_t g_iadd(int64_t a, int64_t b, const char *pos)
{
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) g_int_overflow(pos);
    return r;
}

static inline int64_t g_isub(int64_t a, int64_t b, const char *pos)
{
    int64_t r;
    if (__builtin_sub_overflow(a, b, &r)) g_int_overflow(pos);
    return r;
}

static inline int64_t g_imul(int64_t a, int64_t b, const char *pos)
{
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) g_int_overflow(pos);
    return r;
}

static inline int64_t g_idiv(int64_t a, int64_t b, const char *pos)
{
    if (b == 0) g_div_zero_values(pos, g_int(a), g_int(b));
    if (a == INT64_MIN && b == -1) g_int_overflow(pos);
    return a / b;                    /* truncates toward zero, as trunc_div */
}

static inline int64_t g_imod(int64_t a, int64_t b, const char *pos)
{
    if (b == 0) g_zero_division(pos, "modulo by zero");
    if (a == INT64_MIN && b == -1) return 0;
    return a % b;                    /* sign of the dividend, as trunc_mod */
}

/* Unary minus, which can overflow only at INT64_MIN (spec: `-(-2^63)` does not
 * fit in I64, and the interpreter computes it exactly). */
static inline int64_t g_ineg(int64_t a, const char *pos)
{
    if (a == INT64_MIN) g_int_overflow(pos);
    return -a;
}

/*
 * Float division.  The result is a `double` either way, but a failure has to
 * name the operands the way `g_binop` would, and `g_display` renders an `I64`
 * 1 as "1" while it renders an `F64` 1.0 as "1.0".  Which operand is which is
 * a static property of the instruction, so the code generator picks the
 * wrapper and the unboxing cost is on the fault path only.
 */
static inline double g_fdiv_f_f(double a, double b, const char *pos)
{
    if (b == 0.0) g_div_zero_values(pos, g_float(a), g_float(b));
    return a / b;
}

static inline double g_fdiv_i_f(int64_t a, double b, const char *pos)
{
    if (b == 0.0) g_div_zero_values(pos, g_int(a), g_float(b));
    return (double)a / b;
}

static inline double g_fdiv_f_i(double a, int64_t b, const char *pos)
{
    if (b == 0) g_div_zero_values(pos, g_float(a), g_int(b));
    return a / (double)b;
}

static inline double g_fmod(double a, double b, const char *pos)
{
    /* "modulo by zero" carries no operands, so unlike division the two
     * operand kinds cannot show up in the message. */
    if (b == 0.0) g_zero_division(pos, "modulo by zero");
    return fmod(a, b);
}

/* Emitted by the backend. */
GValue g_main(int argc, char **argv);

/* `declared_grants` is what the *program* asks for, which the code
 * generator passes in rather than the runtime reaching for: a runtime that
 * referenced a symbol only generated programs define could not be linked on
 * its own, and it is linked on its own by the tests that check its value
 * formatting against the interpreter. */
int g_run(int argc, char **argv,
          const char *const *declared_grants, size_t declared_grants_n);

#endif /* GAMAG_RT_H */
