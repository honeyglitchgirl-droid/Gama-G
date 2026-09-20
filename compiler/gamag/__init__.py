"""Gama-G reference toolchain (spec section 2).

This package is the reference implementation of the Gama-G language:
`ggc` compiler front end, Gama IR (GIR), the Gama Abstract Execution
Model (GAEM) reference interpreter, and the standard library.
"""

import os


def _read_version() -> str:
    """Read the version from the repository's VERSION file.

    Three places need to agree -- VERSION, pyproject.toml and this module -- and
    a constant duplicated into all three drifts.  VERSION is the source of
    truth; `tests/test_enforcement.py::Packaging` checks the other two against
    it.  The literal below is only a fallback for an installed wheel that did
    not ship the file, and it must be kept equal to it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(here, "VERSION")
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                break
            if text:
                return text
        here = os.path.dirname(here)
    return "1.0.1"


__version__ = _read_version()
GIR_VERSION = "1.0"
SPEC_VERSION = "1.0"
