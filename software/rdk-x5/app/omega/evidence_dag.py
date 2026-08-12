"""Immutable-node Evidence DAG with deterministic snapshot roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schemas import (
    AuthorityBoundary,
    EvidenceKind,
    EvidenceNode,
    OmegaContractError,
    canonical_sha256,
    require_exact_keys,
    require_sha256,
)


DAG_SNAPSHOT_SCHEMA = "rootscope.omega.evidence-dag-snapshot.v1"


class EvidenceDagError(OmegaContractError):
    pass


@dataclass(frozen=True)
class EvidenceDagSnapshot:
    node_count: int
    node_ids: tuple[str, ...]
    root_node_ids: tuple[str, ...]
    leaf_node_ids: tuple[str, ...]
    node_hashes: tuple[tuple[str, str], ...]
    authority: AuthorityBoundary
    snapshot_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.node_count, bool)
            or not isinstance(self.node_count, int)
            or self.node_count < 0
        ):
            raise EvidenceDagError("node_count must be a non-negative integer")
        if tuple(sorted(set(self.node_ids))) != self.node_ids:
            raise EvidenceDagError("node_ids must be unique and sorted")
        if self.node_count != len(self.node_ids):
            raise EvidenceDagError("node_count does not match node_ids")
        if tuple(sorted(set(self.root_node_ids))) != self.root_node_ids:
            raise EvidenceDagError("root_node_ids must be unique and sorted")
        if tuple(sorted(set(self.leaf_node_ids))) != self.leaf_node_ids:
            raise EvidenceDagError("leaf_node_ids must be unique and sorted")
        if not set(self.root_node_ids).issubset(self.node_ids):
            raise EvidenceDagError("unknown root node")
        if not set(self.leaf_node_ids).issubset(self.node_ids):
            raise EvidenceDagError("unknown leaf node")
        if tuple(sorted(self.node_hashes)) != self.node_hashes:
            raise EvidenceDagError("node_hashes must be sorted by node_id")
        if tuple(node_id for node_id, _ in self.node_hashes) != self.node_ids:
            raise EvidenceDagError("node_hashes do not exactly cover node_ids")
        for _, digest in self.node_hashes:
            require_sha256(digest, "node hash")
        if not isinstance(self.authority, AuthorityBoundary):
            raise EvidenceDagError("authority must be zero-authority capsule")
        expected = canonical_sha256(self.unsigned_dict())
        if not self.snapshot_sha256:
            object.__setattr__(self, "snapshot_sha256", expected)
        elif require_sha256(self.snapshot_sha256, "snapshot_sha256") != expected:
            raise EvidenceDagError("DAG snapshot hash mismatch")

    @property
    def root_sha256(self) -> str:
        """Stable public name used by runtime integration."""

        return self.snapshot_sha256

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DAG_SNAPSHOT_SCHEMA,
            "node_count": self.node_count,
            "node_ids": list(self.node_ids),
            "root_node_ids": list(self.root_node_ids),
            "leaf_node_ids": list(self.leaf_node_ids),
            "node_hashes": [
                {"node_id": node_id, "content_sha256": digest}
                for node_id, digest in self.node_hashes
            ],
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "snapshot_sha256": self.snapshot_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceDagSnapshot":
        expected = {
            "schema_version",
            "node_count",
            "node_ids",
            "root_node_ids",
            "leaf_node_ids",
            "node_hashes",
            "authority",
            "snapshot_sha256",
        }
        require_exact_keys(value, expected, "DAG snapshot")
        if value["schema_version"] != DAG_SNAPSHOT_SCHEMA:
            raise EvidenceDagError("DAG snapshot schema mismatch")
        for name in ("node_ids", "root_node_ids", "leaf_node_ids", "node_hashes"):
            if not isinstance(value[name], list):
                raise EvidenceDagError(f"{name} must be an array")
        hashes: list[tuple[str, str]] = []
        for index, record in enumerate(value["node_hashes"]):
            require_exact_keys(
                record, {"node_id", "content_sha256"}, f"node_hashes[{index}]"
            )
            hashes.append((record["node_id"], record["content_sha256"]))
        return cls(
            node_count=value["node_count"],
            node_ids=tuple(value["node_ids"]),
            root_node_ids=tuple(value["root_node_ids"]),
            leaf_node_ids=tuple(value["leaf_node_ids"]),
            node_hashes=tuple(hashes),
            authority=AuthorityBoundary.from_dict(value["authority"]),
            snapshot_sha256=value["snapshot_sha256"],
        )


class EvidenceDAG:
    """Append-only in-memory graph of hash-bound evidence nodes.

    Parents must already exist.  This one rule makes cycle creation impossible,
    prevents dangling evidence, and makes every rejected mutation atomic.
    """

    def __init__(self, nodes: Iterable[EvidenceNode] = ()) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        self._children: dict[str, set[str]] = {}
        self._insertion_order: list[str] = []
        for node in nodes:
            self.add(node)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    @property
    def root_sha256(self) -> str:
        return self.snapshot().root_sha256

    def add(self, node: EvidenceNode) -> bool:
        if not isinstance(node, EvidenceNode):
            raise EvidenceDagError("DAG accepts only strict EvidenceNode objects")
        # Re-parse the serial form so a forged/subclassed object cannot bypass
        # content-hash and exact-authority validation.
        checked = EvidenceNode.from_dict(node.to_dict())
        existing = self._nodes.get(checked.node_id)
        if existing is not None:
            if existing.content_sha256 == checked.content_sha256:
                return False
            raise EvidenceDagError("node_id conflict with different content")
        missing = tuple(parent for parent in checked.parents if parent not in self._nodes)
        if missing:
            raise EvidenceDagError(f"evidence node has unknown parents: {missing}")

        # No operation above mutates the graph.  Commit only after every gate.
        self._nodes[checked.node_id] = checked
        self._children[checked.node_id] = set()
        for parent in checked.parents:
            self._children[parent].add(checked.node_id)
        self._insertion_order.append(checked.node_id)
        return True

    def get(self, node_id: str) -> EvidenceNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise EvidenceDagError(f"unknown evidence node: {node_id}") from exc

    def require_all(self, node_ids: Iterable[str]) -> tuple[EvidenceNode, ...]:
        return tuple(self.get(node_id) for node_id in node_ids)

    def nodes_by_kind(self, kind: EvidenceKind) -> tuple[EvidenceNode, ...]:
        if not isinstance(kind, EvidenceKind):
            raise EvidenceDagError("kind must be EvidenceKind")
        return tuple(
            self._nodes[node_id]
            for node_id in self._insertion_order
            if self._nodes[node_id].kind is kind
        )

    def latest(self, kind: EvidenceKind) -> EvidenceNode | None:
        nodes = self.nodes_by_kind(kind)
        if not nodes:
            return None
        return max(nodes, key=lambda node: (node.observed_at_ms, node.node_id))

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        self.get(node_id)
        visited: set[str] = set()
        stack = list(self._nodes[node_id].parents)
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._nodes[current].parents)
        return tuple(sorted(visited))

    def validate(self) -> EvidenceDagSnapshot:
        for node_id, node in self._nodes.items():
            checked = EvidenceNode.from_dict(node.to_dict())
            if checked.node_id != node_id:
                raise EvidenceDagError("DAG key/node identity mismatch")
            for parent in node.parents:
                if parent not in self._nodes:
                    raise EvidenceDagError("DAG contains a dangling parent")
                if node_id not in self._children.get(parent, set()):
                    raise EvidenceDagError("DAG reverse edge mismatch")
                if parent not in self.ancestors(node_id):
                    raise EvidenceDagError("DAG ancestor index mismatch")
        return self.snapshot()

    def snapshot(self) -> EvidenceDagSnapshot:
        node_ids = tuple(sorted(self._nodes))
        roots = tuple(
            node_id for node_id in node_ids if not self._nodes[node_id].parents
        )
        leaves = tuple(
            node_id for node_id in node_ids if not self._children.get(node_id)
        )
        hashes = tuple(
            (node_id, self._nodes[node_id].content_sha256) for node_id in node_ids
        )
        return EvidenceDagSnapshot(
            node_count=len(node_ids),
            node_ids=node_ids,
            root_node_ids=roots,
            leaf_node_ids=leaves,
            node_hashes=hashes,
            authority=AuthorityBoundary(),
        )


# Runtime-facing name: an evidence record is the immutable node contract.
EvidenceRecord = EvidenceNode
