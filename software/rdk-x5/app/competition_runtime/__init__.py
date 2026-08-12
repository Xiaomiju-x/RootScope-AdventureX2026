"""Competition-only, zero-authority runtime adapters.

This package is additive.  It does not alter the immutable field bundle, the
frozen CPU classifier, or any actuator/state-machine interface.
"""

from .bpu_shadow_client import BpuShadowClient

__all__ = ["BpuShadowClient"]
