"""RootScope-Ω read-only knowledge and language-model boundary.

This package deliberately has no device, actuator, serial, GPIO, state-machine,
tool-call, or external-network integration.  It turns immutable source records
into cited read-only explanations and records the resulting claims.
"""

from .llm import (
    AUTHORITY,
    RESPONSE_SCHEMA,
    KnowledgeRequest,
    ReadOnlyKnowledgeService,
    Role,
)
from .store import (
    ClaimLedgerRecord,
    KnowledgeChunk,
    KnowledgeContractError,
    KnowledgeStore,
    SearchHit,
    SourceRecord,
)

__all__ = [
    "AUTHORITY",
    "RESPONSE_SCHEMA",
    "ClaimLedgerRecord",
    "KnowledgeChunk",
    "KnowledgeContractError",
    "KnowledgeRequest",
    "KnowledgeStore",
    "ReadOnlyKnowledgeService",
    "Role",
    "SearchHit",
    "SourceRecord",
]
