"""Make the test suite runner-independent.

The tests import a shared helper as a top-level module (`import support`),
which works under `python3 -m unittest discover -s tests` because unittest puts
the start directory on sys.path.  pytest does not do that by default, so
collecting the same directory failed with::

    ModuleNotFoundError: No module named 'support'

pytest loads conftest.py before collecting, so putting the two directories the
suite needs on sys.path here makes both runners work without any test module
having to know which one is driving it.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMPILER = os.path.join(_REPO, "compiler")

for path in (_HERE, _COMPILER):
    if path not in sys.path:
        sys.path.insert(0, path)
