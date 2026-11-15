"""Framed JSON protocol."""

from __future__ import annotations

import json
import struct

_HEADER = struct.Struct("!I")


def encode_message(msg_type, payload, msg_id=0):
    """Serialize a message to a length-prefixed JSON frame."""
    body = json.dumps({"type": msg_type, "payload": payload, "id": msg_id}).encode()
    return _HEADER.pack(len(body)) + body


def decode_message(frame):
    """Parse a length-prefixed JSON frame."""
    (length,) = _HEADER.unpack(frame[: _HEADER.size])
    data = json.loads(frame[_HEADER.size : _HEADER.size + length].decode())
    return data["type"], data.get("payload", {}), data.get("id", 0)
