"""Send only fail-closed ASCII commands, then print readable status lines."""

from __future__ import annotations

import os
import select
import termios
import time


DEVICE = "/dev/rootscope_stm32"


def main() -> int:
    fd = os.open(DEVICE, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
        os.write(fd, b"STOP\r\nSTATUS\r\nIOSTATUS\r\n")

        data = bytearray()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                try:
                    data.extend(os.read(fd, 4096))
                except BlockingIOError:
                    pass
        text = bytes(data).decode("ascii", errors="ignore")
        for line in text.replace("\r", "\n").splitlines():
            if line.startswith(("ACK,", "STATUS,", "IOSTATUS,")):
                print(line)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
