"""RootScope-Ω FTS5/BM25 and three-role read-only knowledge adapter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple

from app.omega_knowledge import (
    KnowledgeChunk,
    KnowledgeRequest,
    KnowledgeStore,
    ReadOnlyKnowledgeService,
    Role,
    SourceRecord,
)

from .contracts import canonical_sha256


_SECTION_RE = re.compile(
    r"^##\s+(K\d{2})\s+(.+?)\s*$\n+(.+?)(?=^##\s+K\d{2}\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)


class TextOnlyModel(Protocol):
    model_id: str

    def generate(
        self,
        messages: tuple[Mapping[str, str], ...],
        *,
        response_schema: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class KnowledgeContext:
    responses: Tuple[Mapping[str, Any], ...]
    claims: Tuple[Mapping[str, Any], ...]
    integrity_report: Mapping[str, Any]
    claim_ledger_root: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "responses": [dict(item) for item in self.responses],
            "claims": [dict(item) for item in self.claims],
            "integrity_report": dict(self.integrity_report),
            "claim_ledger_root": self.claim_ledger_root,
        }


def _load_corpus(store: KnowledgeStore, path: Path) -> None:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    source = SourceRecord(
        source_id="rootscope-field-knowledge-v1",
        title="RootScope Omega Field Knowledge v1",
        locator="configs/omega/field_knowledge.v1.md",
        source_type="LOCAL_FILE",
        version="v1",
        license="PROJECT_INTERNAL",
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    chunks = []
    for section_id, title, body in _SECTION_RE.findall(text):
        content = f"{title.strip()}\n\n{body.strip()}"
        chunks.append(
            KnowledgeChunk.from_text(
                chunk_id=section_id.lower(),
                source_id=source.source_id,
                paragraph_id=section_id.lower(),
                text=content,
            )
        )
    if len(chunks) != 7:
        raise ValueError("field knowledge corpus must contain exactly K01-K07")
    store.add_documents(source, chunks)


def _query_for(case_id: str, role: Role) -> str:
    case_topic = {
        "CASE01_NORMAL_VERIFIED": "ACK mass loss target wetting completion evidence",
        "CASE02_OOD_RECAPTURE": "OOD glare occlusion RECAPTURE operator review",
        "CASE03_ACK_WITHOUT_MASS_LOSS": "ACK without mass loss blockage reweigh",
        "CASE04_NEIGHBOR_SPILL": "neighbor spill leakage hydraulic crosstalk locked",
        "CASE05_STALE_TAMPER_ESTOP": "stale evidence payload hash estop firmware disconnected",
    }.get(case_id, "RootScope evidence safety boundary")
    role_topic = {
        Role.EVIDENCE_EXPLAINER: "Evidence DAG Hybrid Belief State Decision Receipt",
        Role.SAFETY_AUDITOR: "zero authority fail closed safety",
        Role.DEFENSE_QA: "RDK X5 CPU BPU qualification field profile",
    }[role]
    return f"{case_topic} {role_topic}"


def run_knowledge_roles(
    *,
    case_id: str,
    evidence_refs: Tuple[str, ...],
    corpus_path: Path,
    model: Optional[TextOnlyModel] = None,
    compact_edge_prompt: bool = False,
) -> KnowledgeContext:
    with KnowledgeStore(":memory:") as store:
        _load_corpus(store, corpus_path)
        service = ReadOnlyKnowledgeService(store)
        responses = []
        run_id = f"knowledge-{case_id.lower()}"
        for role in Role:
            request = KnowledgeRequest(
                role=role,
                query=_query_for(case_id, role),
                run_id=run_id,
                evidence_refs=evidence_refs,
                max_hits=1 if compact_edge_prompt else 6,
            )
            responses.append(
                service.answer(
                    request,
                    model=model,
                    passage_char_limit=320 if compact_edge_prompt else 2_400,
                    include_schema_in_user_payload=not compact_edge_prompt,
                    max_model_tokens=192 if compact_edge_prompt else 384,
                )
            )
        claims = tuple(
            record.to_dict() for record in store.claims_for_run(run_id)
        )
        integrity = store.integrity_report()
        capsule = {
            "schema_version": "rootscope.omega.claim-ledger-capsule.v1",
            "case_id": case_id,
            "response_hashes": [
                response["provenance"]["response_sha256"]
                for response in responses
            ],
            "claims": list(claims),
            "integrity": integrity,
            "authority": {
                "execution_authority": False,
                "physical_authority": False,
            },
        }
        return KnowledgeContext(
            responses=tuple(responses),
            claims=claims,
            integrity_report=integrity,
            claim_ledger_root=canonical_sha256(capsule),
        )
