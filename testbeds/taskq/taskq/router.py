"""Message routing to registered handlers."""

from __future__ import annotations

from .errors import ProtocolError
from .protocol import Message


class Router:
    """Dispatches incoming messages to a handler by message type."""

    def __init__(self):
        self._routes = {}

    def register(self, msg_type, handler):
        """Bind a message type to a callable handler."""
        self._routes[msg_type] = handler
        return self

    def dispatch(self, msg):
        """Route a Message to its handler and return the reply Message."""
        handler = self._routes.get(msg.type)
        if handler is None:
            raise ProtocolError(f"no route for message type {msg.type!r}")
        payload = handler(msg.payload)
        return Message(type=f"{msg.type}_reply", payload=payload, msg_id=msg.msg_id)
