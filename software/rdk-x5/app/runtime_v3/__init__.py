"""RootScope v3 board-side runtime contracts.

The package contains only advisory scheduling and inference adapters.  It has
no camera, serial, GPIO, pump, service-manager, or network client.
"""

from .resource_broker import (
    ResourceBroker,
    ResourceDecision,
    ResourceSnapshot,
    RuntimePhase,
    Workload,
)

__all__ = [
    "ResourceBroker",
    "ResourceDecision",
    "ResourceSnapshot",
    "RuntimePhase",
    "Workload",
]
