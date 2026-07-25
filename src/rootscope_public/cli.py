"""Command-line entry point for synthetic, device-free fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audit import receipt_sha256
from .contracts import EvidenceBundle
from .gate import evaluate_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a synthetic RootScope evidence fixture."
    )
    parser.add_argument("fixture", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    evidence = EvidenceBundle.from_mapping(raw)
    proposal = evaluate_evidence(evidence).to_dict()
    receipt = {
        "schema": "rootscope.public.proposal.v1",
        "input_kind": "synthetic_fixture",
        "proposal": proposal,
    }
    receipt["sha256"] = receipt_sha256(receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

