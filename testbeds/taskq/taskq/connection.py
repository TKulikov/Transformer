"""Framed socket connection shared by client and server."""

from __future__ import annotations

from .protocol import decode_message, encode_message
from .utils import retry


class Connection:
    """A length-prefixed message channel over a socket."""

    def __init__(self, sock):
        self._sock = sock
        self._buffer = b""

    def send(self, msg):
        """Encode and send a Message."""
        self._sock.sendall(encode_message(msg))

    def _read_exact(self, n):
        while len(self._buffer) < n:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed")
            self._buffer += chunk
        head, self._buffer = self._buffer[:n], self._buffer[n:]
        return head

    def recv(self):
        """Read one framed Message (with retry on transient reads)."""
        import struct

        header = retry(lambda: self._read_exact(struct.calcsize("!I")))
        (length,) = struct.unpack("!I", header)
        body = self._read_exact(length)
        return decode_message(header + body)

    def close(self):
        """Close the underlying socket."""
        self._sock.close()
