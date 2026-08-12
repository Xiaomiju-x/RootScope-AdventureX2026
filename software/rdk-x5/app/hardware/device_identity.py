"""Pure-data USB identity contracts for RootScope hardware.

This module deliberately performs no discovery.  Production code must receive
an identity captured during commissioning; it must never select the first
``ttyUSB``/``video`` device it happens to find.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Optional


_HEX4_RE = re.compile(r"^[0-9a-f]{4}$")
_SAFE_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+,-]{0,255}$")


def _safe_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SAFE_TEXT_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty safe string")


@dataclass(frozen=True)
class UsbDeviceIdentity:
    """Expected immutable/stable identity for one explicitly enrolled device."""

    alias: str
    vid: str
    pid: str
    serial_number: Optional[str] = None
    id_path: Optional[str] = None
    interface_number: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias.startswith("/dev/"):
            raise ValueError("alias must be an explicit /dev path")
        if any(character.isspace() for character in self.alias):
            raise ValueError("alias cannot contain whitespace")
        if not _HEX4_RE.fullmatch(self.vid):
            raise ValueError("vid must be four lowercase hexadecimal digits")
        if not _HEX4_RE.fullmatch(self.pid):
            raise ValueError("pid must be four lowercase hexadecimal digits")
        if self.serial_number is None and self.id_path is None:
            raise ValueError("identity requires serial_number or frozen id_path")
        for field_name in ("serial_number", "id_path", "interface_number"):
            value = getattr(self, field_name)
            if value is not None:
                _safe_text(value, field_name)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "alias": self.alias,
            "vid": self.vid,
            "pid": self.pid,
            "serial_number": self.serial_number,
            "id_path": self.id_path,
            "interface_number": self.interface_number,
        }

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()
