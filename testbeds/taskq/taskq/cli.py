"""Command-line interface for the client."""

from __future__ import annotations

import argparse

from .utils import parse_duration


class CLI:
    """Parses command-line arguments into client operations."""

    def __init__(self, client):
        self._client = client

    def build_parser(self):
        parser = argparse.ArgumentParser(prog="taskq")
        sub = parser.add_subparsers(dest="command")
        submit = sub.add_parser("submit")
        submit.add_argument("name")
        submit.add_argument("--id", required=True)
        status = sub.add_parser("status")
        status.add_argument("id")
        sub.add_parser("cancel").add_argument("id")
        return parser

    def run(self, argv):
        """Execute the parsed command against the client."""
        args = self.build_parser().parse_args(argv)
        if args.command == "submit":
            return self._client.submit([{"id": args.id, "name": args.name}])
        if args.command == "status":
            return self._client.status(args.id)
        if args.command == "cancel":
            return self._client.cancel(args.id)
        return None

    def timeout_ms(self, text):
        """Parse a timeout argument into milliseconds."""
        return parse_duration(text)
