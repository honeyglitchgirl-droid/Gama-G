"""The standard library.

Each module registers its builtins into one table when it is imported, so this
package's job is to import them in an order that leaves everything registered --
and then to refresh the module-name table, which `library` snapshots at its own
import time and which therefore does not know about anything registered after
it.
"""

from . import library  # noqa: F401  -- owns the builtin registry
from . import ffi      # noqa: F401  -- spec section 29, audit priority 12
from . import interop  # noqa: F401  -- spec section 24, audit priority 15
from . import enterprise  # noqa: F401  -- spec section 25, audit priority 16

# Everything is registered; make the checker's view of module names agree with
# the registry the builtins actually went into.
library.refresh_module_names()
