"""Public, device-free reference interfaces for RootScope."""

from .contracts import ActionProposal, Decision, EvidenceBundle
from .gate import evaluate_evidence

__all__ = ["ActionProposal", "Decision", "EvidenceBundle", "evaluate_evidence"]
__version__ = "0.1.0"

