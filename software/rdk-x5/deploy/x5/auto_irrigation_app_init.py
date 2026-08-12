"""Minimal package marker for the RootScope auto-irrigation sidecar.

The sidecar intentionally ships only ``app.hardware`` and ``app.serial``.
Keeping this package initializer empty prevents importing the full RootScope
application graph (configuration, runtime and release modules) before the
one-cycle runner reaches its explicit safety gates.
"""

__all__: list[str] = []
