"""Read-only local language-model helpers for RootScope."""

from .read_only_explainer import (
    ExplanationConfig,
    build_explanation_messages,
    deterministic_explanation,
    explain_snapshot,
    parse_explanation_response,
)

__all__ = [
    "ExplanationConfig",
    "build_explanation_messages",
    "deterministic_explanation",
    "explain_snapshot",
    "parse_explanation_response",
]
