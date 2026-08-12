"""Import-only compatibility shim for the immutable field bundle v2.

The frozen v2 core intentionally packages the ``app.edge`` and ``app.llm``
subpackages, but its top-level ``app/__init__.py`` still imports modules that
are not part of that release.  Putting this directory first on ``PYTHONPATH``
during installation avoids executing those stale exports while
``pkgutil.extend_path`` makes the original, hash-verified subpackages visible.

This shim changes no file in the immutable bundle, grants no hardware or
execution authority, and is needed only for the v2 installer/smoke commands.
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_WORKTREE_APP = (Path.cwd() / "app").resolve()
if _WORKTREE_APP.is_dir() and str(_WORKTREE_APP) not in __path__:
    __path__.append(str(_WORKTREE_APP))
