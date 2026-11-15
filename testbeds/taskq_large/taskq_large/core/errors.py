"""Shared exception types."""


class TaskqError(Exception):
    """Base error."""


class NotFound(TaskqError):
    """Raised when a record is missing."""
