"""Dependency-light offline dashboard server."""

from .state_store import SnapshotStore, default_snapshot

__all__ = ["SnapshotStore", "default_snapshot"]
