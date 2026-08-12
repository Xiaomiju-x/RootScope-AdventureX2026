"""Place the v2 compatibility package ahead of the installer's working tree."""

from pathlib import Path
import sys


_OVERLAY = str(Path(__file__).resolve().parent)
sys.path[:] = [_OVERLAY, *(entry for entry in sys.path if entry != _OVERLAY)]

# Cache the shim package before Python inserts the ``-m`` working directory at
# sys.path[0]; otherwise the stale bundled app/__init__.py wins resolution.
import app  # noqa: E402,F401
