"""Canonical hashing for public, synthetic proposal receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def receipt_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

