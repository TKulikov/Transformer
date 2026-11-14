"""Token-based authentication middleware."""

from __future__ import annotations

from .errors import ProtocolError
from .middleware import Middleware
from .utils import dedupe


class AuthMiddleware(Middleware):
    """Rejects requests whose token is not in the allowed set."""

    def __init__(self, inner, tokens):
        super().__init__(inner)
        self._allowed = set(dedupe(tokens))

    def before(self, payload):
        token = payload.get("token")
        if token not in self._allowed:
            raise ProtocolError("invalid or missing token")
        return payload
