"""The task queue client."""

from __future__ import annotations

from .connection import Connection
from .protocol import Message
from .utils import parse_duration, retry


class Client:
    """Submits tasks and polls their results over a Connection."""

    def __init__(self, conn, timeout="5s"):
        self._conn = conn
        self._timeout = parse_duration(timeout)
        self._next_id = 1

    def _request(self, msg_type, payload):
        msg = Message(type=msg_type, payload=payload, msg_id=self._next_id)
        self._next_id += 1
        self._conn.send(msg)
        return self._conn.recv()

    def submit(self, tasks):
        """Submit a list of task specs; return accepted ids."""
        reply = retry(lambda: self._request("submit", {"tasks": tasks}))
        return reply.payload["accepted"]

    def status(self, task_id):
        """Poll a task's status and result."""
        reply = self._request("status", {"id": task_id})
        return reply.payload

    def cancel(self, task_id):
        """Cancel a pending task."""
        reply = self._request("cancel", {"id": task_id})
        return reply.payload

    def close(self):
        """Close the connection."""
        self._conn.close()
