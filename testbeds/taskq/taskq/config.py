"""Server configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .utils import parse_duration


@dataclass
class Config:
    """Runtime configuration for the server and client."""

    host: str = "127.0.0.1"
    port: int = 9100
    workers: int = 4
    max_retries: int = 3
    task_timeout: int = 5000

    @classmethod
    def from_dict(cls, data):
        """Build a Config, parsing human-readable duration fields."""
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            if key == "task_timeout":
                value = parse_duration(value, cfg.task_timeout)
            setattr(cfg, key, value)
        return cfg
