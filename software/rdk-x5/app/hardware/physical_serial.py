"""No-side-effect interface for the future physical USB-TTL adapter.

Importing this module never enumerates or opens a device.  The default opener
remains disabled.  The POSIX implementation must be constructed explicitly and
opens only a commissioned alias after matching its frozen udev identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .device_identity import UsbDeviceIdentity


ROOTSCOPE_F407_ALIAS = "/dev/rootscope_f407"
ROOTSCOPE_F103_ALIAS = "/dev/rootscope_stm32"
ROOTSCOPE_F407_BAUDRATE = 115_200
ROOTSCOPE_ALLOWED_SERIAL_ALIASES = frozenset(
    {ROOTSCOPE_F407_ALIAS, ROOTSCOPE_F103_ALIAS}
)


class PhysicalSerialDisabled(RuntimeError):
    """Raised when a physical adapter was not explicitly installed/authorized."""


@dataclass(frozen=True)
class PhysicalSerialOpenRequest:
    """Complete request for one explicit, exclusive serial open.

    No default port fallback is permitted.  ``physical_authority_granted`` is
    intentionally required to be ``True`` before an implementation may act;
    callers cannot accidentally rely on a truthy string.
    """

    identity: UsbDeviceIdentity
    baudrate: int = ROOTSCOPE_F407_BAUDRATE
    read_timeout_ms: int = 50
    write_timeout_ms: int = 100
    exclusive: bool = True
    allow_port_enumeration: bool = False
    physical_authority_granted: bool = False

    def __post_init__(self) -> None:
        if self.identity.alias not in ROOTSCOPE_ALLOWED_SERIAL_ALIASES:
            raise ValueError(
                "physical serial alias must be one of "
                f"{sorted(ROOTSCOPE_ALLOWED_SERIAL_ALIASES)}"
            )
        for field_name in ("baudrate", "read_timeout_ms", "write_timeout_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.baudrate != ROOTSCOPE_F407_BAUDRATE:
            raise ValueError("baudrate must match the frozen serial ICD")
        if self.exclusive is not True:
            raise ValueError("physical serial must be opened in exclusive mode")
        if self.allow_port_enumeration is not False:
            raise ValueError("port enumeration is forbidden")
        if not isinstance(self.physical_authority_granted, bool):
            raise ValueError("physical_authority_granted must be boolean")


@runtime_checkable
class SerialByteTransport(Protocol):
    """The only byte-level object the unique writer may own."""

    @property
    def backend_id(self) -> str: ...

    @property
    def device_identity_sha256(self) -> str: ...

    @property
    def is_open(self) -> bool: ...

    def write(self, data: bytes) -> int: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


@runtime_checkable
class PhysicalSerialOpener(Protocol):
    """Explicit injection point for a later audited pyserial implementation."""

    def open_explicit(self, request: PhysicalSerialOpenRequest) -> SerialByteTransport:
        """Open exactly ``request.identity.alias`` or raise without fallback."""


class DisabledPhysicalSerialOpener:
    """E0 production default: deterministic refusal with zero device I/O."""

    def open_explicit(self, request: PhysicalSerialOpenRequest) -> SerialByteTransport:
        del request
        raise PhysicalSerialDisabled(
            "physical serial implementation is absent in E0; no port was opened"
        )


class _PosixTermiosTransport:
    """One already-verified Linux tty owned by the unique serial writer."""

    def __init__(
        self,
        *,
        file_descriptor: int,
        alias: str,
        identity_sha256: str,
        read_timeout_ms: int,
    ) -> None:
        self._fd = file_descriptor
        self._alias = alias
        self._identity_sha256 = identity_sha256
        self._read_timeout_ms = read_timeout_ms

    @property
    def backend_id(self) -> str:
        return f"POSIX_TERMIOS_EXPLICIT:{self._alias}"

    @property
    def device_identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def is_open(self) -> bool:
        return self._fd >= 0

    def write(self, data: bytes) -> int:
        import os

        if not self.is_open:
            raise OSError("serial transport is closed")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("serial write requires bytes-like data")
        return os.write(self._fd, bytes(data))

    def read(self, size: int) -> bytes:
        import os
        import select

        if not self.is_open:
            raise OSError("serial transport is closed")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("serial read size must be a positive integer")
        readable, _, _ = select.select(
            [self._fd], [], [], self._read_timeout_ms / 1000.0
        )
        return os.read(self._fd, size) if readable else b""

    def close(self) -> None:
        import os

        if self._fd >= 0:
            descriptor = self._fd
            self._fd = -1
            os.close(descriptor)


class PosixExplicitSerialOpener:
    """Audited Linux opener for a pre-enrolled `/dev/rootscope_*` alias.

    It calls ``udevadm`` only for the exact alias in the request; it never lists
    or scans serial ports.  The returned descriptor is placed in exclusive,
    raw 115200-8-N-1 mode.  Construction has no side effects.
    """

    @staticmethod
    def _udev_properties(alias: str) -> dict[str, str]:
        import subprocess

        completed = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={alias}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        properties: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key:
                properties[key] = value
        return properties

    @staticmethod
    def _verify_identity(
        expected: UsbDeviceIdentity, properties: dict[str, str]
    ) -> None:
        observed_vid = properties.get("ID_VENDOR_ID", "").lower()
        observed_pid = properties.get("ID_MODEL_ID", "").lower()
        if observed_vid != expected.vid or observed_pid != expected.pid:
            raise PhysicalSerialDisabled(
                "USB VID:PID mismatch for commissioned serial alias"
            )
        if expected.serial_number is not None and (
            properties.get("ID_SERIAL_SHORT") != expected.serial_number
        ):
            raise PhysicalSerialDisabled(
                "USB serial number mismatch for commissioned serial alias"
            )
        if expected.id_path is not None and (
            properties.get("ID_PATH") != expected.id_path
        ):
            raise PhysicalSerialDisabled(
                "USB physical path mismatch for commissioned serial alias"
            )
        if expected.interface_number is not None and (
            properties.get("ID_USB_INTERFACE_NUM") != expected.interface_number
        ):
            raise PhysicalSerialDisabled(
                "USB interface mismatch for commissioned serial alias"
            )

    def open_explicit(self, request: PhysicalSerialOpenRequest) -> SerialByteTransport:
        import fcntl
        import os
        import sys
        import termios

        if sys.platform != "linux":
            raise PhysicalSerialDisabled(
                "POSIX physical serial opener is available on Linux only"
            )
        if request.physical_authority_granted is not True:
            raise PhysicalSerialDisabled(
                "physical_authority_granted=True is required for a real open"
            )
        if not os.path.exists(request.identity.alias):
            raise PhysicalSerialDisabled(
                f"commissioned serial alias is absent: {request.identity.alias}"
            )

        properties = self._udev_properties(request.identity.alias)
        self._verify_identity(request.identity, properties)

        flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        descriptor = os.open(request.identity.alias, flags)
        try:
            # Linux TIOCEXCL: reject any second opener until this descriptor
            # closes, enforcing the one-writer contract at the kernel boundary.
            fcntl.ioctl(descriptor, termios.TIOCEXCL)
            attributes = termios.tcgetattr(descriptor)
            attributes[0] = 0
            attributes[1] = 0
            attributes[2] = (
                termios.CLOCAL | termios.CREAD | termios.CS8
            )
            attributes[3] = 0
            attributes[4] = termios.B115200
            attributes[5] = termios.B115200
            attributes[6][termios.VMIN] = 0
            attributes[6][termios.VTIME] = 0
            termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
            termios.tcflush(descriptor, termios.TCIOFLUSH)
            current_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            fcntl.fcntl(
                descriptor,
                fcntl.F_SETFL,
                current_flags & ~os.O_NONBLOCK,
            )
        except Exception:
            os.close(descriptor)
            raise

        return _PosixTermiosTransport(
            file_descriptor=descriptor,
            alias=request.identity.alias,
            identity_sha256=request.identity.identity_sha256,
            read_timeout_ms=request.read_timeout_ms,
        )
