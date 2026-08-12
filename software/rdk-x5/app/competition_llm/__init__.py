"""4 GB X5 one-call/three-role competition LLM/RAG sidecar."""

from .contracts import (
    AUTHORITY,
    MAX_MODEL_TOKENS,
    CompetitionLlmError,
    CorpusChunk,
    LoopbackConfig,
)
from .corpus import load_corpus, retrieve
from .competition_rag import (
    CompetitionRagIndex,
    load_competition_rag,
    run_competition_rag_microcluster,
)
from .runtime import LoopbackOpenAIClient, run_competition_microcluster

__all__ = [
    "AUTHORITY",
    "MAX_MODEL_TOKENS",
    "CompetitionLlmError",
    "CorpusChunk",
    "CompetitionRagIndex",
    "LoopbackConfig",
    "LoopbackOpenAIClient",
    "load_corpus",
    "load_competition_rag",
    "retrieve",
    "run_competition_microcluster",
    "run_competition_rag_microcluster",
]
