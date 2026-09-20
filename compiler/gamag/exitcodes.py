"""The exit statuses `ggc` promises, defined once.

These were written down twice -- once in :mod:`gamag.cli.main` and once as
``G_EXIT_*`` in the C runtime -- and the two disagreed: a runtime fault exited
``2`` from ``ggc run`` and ``3`` from a natively compiled binary, because ``3``
is the usage-error status.  Nothing caught it, and the reason is worth keeping:
the differential harness normalised *both* sides to its own convention before
comparing them, so the one tool whose job was to compare the two back ends had
been told to ignore their exit statuses.

A number that two components must agree on belongs in one place.  The C runtime
cannot import this module, so it carries the same numbers with a comment saying
where they come from, and ``tests/test_backends.py`` asserts that the two
definitions still match -- a comment cannot fail.
"""

from __future__ import annotations

#: The program ran and finished.
EXIT_OK = 0
#: The program did not compile.
EXIT_COMPILE = 1
#: The program compiled and then faulted at run time.
EXIT_RUNTIME = 2
#: `ggc` was invoked wrongly: an unknown flag, a missing one.
EXIT_USAGE = 3
#: The program's own `test` declarations ran and at least one failed.
EXIT_TEST = 4

#: The statuses a run can end in, for a table-driven test or a report.
ALL = {
    "ok": EXIT_OK,
    "compile-error": EXIT_COMPILE,
    "runtime-fault": EXIT_RUNTIME,
    "usage": EXIT_USAGE,
    "test-failed": EXIT_TEST,
}
