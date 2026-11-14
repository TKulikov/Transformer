"""taskq — a small client-server task queue used as a memory-test codebase.

Realistic multi-file project: one class per file, meaningful names,
cross-module helper calls. The server accepts tasks over a framed socket
protocol, schedules and executes them via workers, and persists results
in an in-memory store; the client submits and polls tasks.
"""

__version__ = "0.3.1"
