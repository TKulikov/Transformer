"""Exception hierarchy for the task queue."""


class TaskQueueError(Exception):
    """Base class for all taskq errors."""


class ProtocolError(TaskQueueError):
    """Raised when a wire message cannot be parsed or is malformed."""


class TaskNotFound(TaskQueueError):
    """Raised when a task id is unknown to the store."""

    def __init__(self, task_id):
        super().__init__(f"task {task_id!r} not found")
        self.task_id = task_id


class RetryExhausted(TaskQueueError):
    """Raised when an operation fails after all retry attempts."""

    def __init__(self, attempts, last_error):
        super().__init__(f"failed after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error
