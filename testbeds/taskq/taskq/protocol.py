"""Wire protocol: length-prefixed JSON messages."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

from .errors import ProtocolError

_HEADER = struct.Struct("!I")


@dataclass
class Message:
    """A single protocol message with a type and a payload dict."""

    type: str
    payload: dict = field(default_factory=dict)
    msg_id: int = 0


def encode_message(msg):
    """Serialize a Message to a length-prefixed JSON byte frame."""
    body = json.dumps({"type": msg.type, "payload": msg.payload, "id": msg.msg_id}).encode("utf-8")
    return _HEADER.pack(len(body)) + body


def decode_message(frame):
    """Parse a length-prefixed JSON frame into a Message."""
    if len(frame) < _HEADER.size:
        raise ProtocolError("frame shorter than header")
    (length,) = _HEADER.unpack(frame[: _HEADER.size])
    body = frame[_HEADER.size : _HEADER.size + length]
    if len(body) != length:
        raise ProtocolError("truncated frame body")
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    return Message(type=data["type"], payload=data.get("payload", {}), msg_id=data.get("id", 0))
