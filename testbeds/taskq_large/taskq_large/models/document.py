"""Domain model for document resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Document:
    """A single document record."""

    pages: float = 0.0
    title: str = ''
    format: str = 'pdf'
    indexed: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_document(**kwargs):
    """Construct a document from keyword fields."""
    return Document(**kwargs)
