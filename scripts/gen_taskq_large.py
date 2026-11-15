"""Генератор большого мультифайлового проекта taskq_large (~50-90k токенов).

Реалистичная layered-архитектура: на каждый ресурс — model / repository /
service / handler (один класс на файл), говорящие имена, межфайловые вызовы
хелперов из core.utils (retry, backoff_delay, parse_duration, clamp, chunked,
dedupe, slugify, compute_total). Домены различаются полями и предметной
логикой (не клоны). Цель — тестbed для shallow vs flat на масштабе + множестве
файлов (диверсификация по файлам).

Запуск: .venv/bin/python scripts/gen_taskq_large.py [--resources 18]
Выход:  testbeds/taskq_large/taskq_large/... + meta.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "testbeds" / "taskq_large" / "taskq_large"

# name: (num_field, str_field, extra_fields, domain_expr, domain_doc)
RESOURCES = {
    "invoice": ("amount", "number", [("due_days", "int", "30"), ("paid", "bool", "False")],
                "total * (1.0 + 0.02 * overdue)", "amount with a late penalty"),
    "order": ("total", "reference", [("items", "int", "1"), ("status", "str", "'new'")],
             "total / max(items, 1)", "average value per item"),
    "product": ("price", "sku", [("stock", "int", "0"), ("category", "str", "'general'")],
               "price * stock", "total stock value"),
    "payment": ("amount", "method", [("currency", "str", "'USD'"), ("settled", "bool", "False")],
               "amount if settled else 0.0", "settled amount"),
    "shipment": ("weight", "destination", [("carrier", "str", "'local'"), ("distance", "int", "10")],
                "weight * distance * 0.01", "estimated freight cost"),
    "ticket": ("priority", "title", [("assignee", "str", "'unassigned'"), ("severity", "int", "1")],
              "priority * severity", "weighted urgency"),
    "session": ("duration", "token", [("device", "str", "'web'"), ("active", "bool", "True")],
               "duration if active else 0", "active session time"),
    "metric": ("value", "name", [("unit", "str", "'count'"), ("tags", "str", "''")],
              "value * value", "squared magnitude"),
    "contract": ("value", "party", [("term_months", "int", "12"), ("signed", "bool", "False")],
                "value / max(term_months, 1)", "monthly contract value"),
    "visit": ("duration", "page", [("referrer", "str", "'direct'"), ("bounce", "bool", "False")],
             "0 if bounce else duration", "engaged visit time"),
    "signal": ("strength", "source", [("frequency", "int", "1"), ("noise", "int", "0")],
              "strength - noise", "net signal strength"),
    "batch": ("size", "kind", [("checksum", "str", "''"), ("valid", "bool", "True")],
             "size if valid else 0", "valid batch size"),
    "route": ("latency", "path", [("method", "str", "'GET'"), ("hits", "int", "0")],
             "latency * hits", "cumulative latency"),
    "account": ("balance", "owner", [("tier", "str", "'free'"), ("frozen", "bool", "False")],
               "0.0 if frozen else balance", "available balance"),
    "event": ("payload_size", "kind", [("source", "str", "'app'"), ("processed", "bool", "False")],
             "payload_size if processed else 0", "processed payload"),
    "document": ("pages", "title", [("format", "str", "'pdf'"), ("indexed", "bool", "False")],
                "pages if indexed else 0", "indexed page count"),
    "subscription": ("price", "plan", [("renews", "bool", "True"), ("period_months", "int", "1")],
                     "price / max(period_months, 1)", "monthly price"),
    "notification": ("priority", "channel", [("message", "str", "''"), ("read", "bool", "False")],
                    "0 if read else priority", "unread priority"),
}

UTILS = '''"""Cross-cutting helpers shared by every layer (recall targets)."""

from __future__ import annotations

import time


def now_ms():
    """Current wall-clock time in integer milliseconds."""
    return int(time.time() * 1000)


def clamp(value, lo, hi):
    """Clamp value into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def parse_duration(text, default=0):
    """Parse '500ms' / '2s' / '1m' into milliseconds."""
    if isinstance(text, (int, float)):
        return int(text)
    text = str(text).strip().lower()
    for suffix, factor in {"ms": 1, "s": 1000, "m": 60_000}.items():
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * factor)
            except ValueError:
                return default
    try:
        return int(text)
    except ValueError:
        return default


def parse_int(text, default=0):
    """Parse an int from text with a fallback."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def backoff_delay(attempt, base=100, cap=30_000):
    """Exponential backoff delay in ms for a zero-based attempt."""
    return clamp(base * (2 ** attempt), base, cap)


def retry(fn, attempts=3, base=100, sleep=time.sleep):
    """Call fn() with exponential backoff; raise on exhaustion."""
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                sleep(backoff_delay(attempt, base) / 1000.0)
    raise RuntimeError(f"failed after {attempts} attempts: {last}")


def chunked(items, size):
    """Yield successive fixed-size chunks from a sequence."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def dedupe(items):
    """Stable deduplication preserving first-occurrence order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def slugify(text):
    """Lowercase dash-separated slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def compute_total(rows, field):
    """Sum a numeric field over dict rows."""
    return sum(float(r.get(field, 0) or 0) for r in rows)
'''


def cap(name):
    return name.capitalize()


def gen_model(name, spec):
    num, strf, extra, _, _ = spec
    fields = [(num, "float", "0.0"), (strf, "str", "''")] + extra
    lines = [f'"""Domain model for {name} resources."""', "",
             "from __future__ import annotations", "",
             "from dataclasses import dataclass, field", "",
             "from ..core.utils import now_ms", "", "",
             "@dataclass", f"class {cap(name)}:", f'    """A single {name} record."""', ""]
    for fname, ftype, fdefault in fields:
        lines.append(f"    {fname}: {ftype} = {fdefault}")
    lines += ["    created_ms: int = field(default_factory=now_ms)", "", "",
              f"def new_{name}(**kwargs):", f'    """Construct a {name} from keyword fields."""',
              f"    return {cap(name)}(**kwargs)", ""]
    return "\n".join(lines)


def gen_repo(name, spec):
    num, strf, extra, _, _ = spec
    return f'''"""Repository for {name} records (persistence layer)."""

from __future__ import annotations

from ..core.utils import chunked, dedupe
from ..models.{name} import {cap(name)}


class {cap(name)}Repository:
    """In-memory store of {name} records with batch helpers."""

    def __init__(self):
        self._rows = {{}}

    def put(self, record):
        """Insert or replace a record keyed by its {strf}."""
        self._rows[record.{strf}] = record
        return record

    def get(self, key):
        """Return a record by {strf} or None."""
        return self._rows.get(key)

    def delete(self, key):
        """Remove a record by {strf}."""
        return self._rows.pop(key, None)

    def list(self):
        """All records in insertion order."""
        return list(self._rows.values())

    def keys(self):
        """Deduplicated record keys."""
        return dedupe(list(self._rows.keys()))

    def bulk_put(self, records):
        """Insert records in fixed-size batches."""
        count = 0
        for batch in chunked(records, 64):
            for record in batch:
                self.put(record)
                count += 1
        return count

    def filter_by(self, field, value):
        """Records whose field equals value."""
        return [r for r in self._rows.values() if getattr(r, field, None) == value]
'''


def gen_service(name, spec):
    num, strf, extra, expr, doc = spec
    return f'''"""Business logic for {name} resources (service layer)."""

from __future__ import annotations

from ..core.utils import (backoff_delay, clamp, compute_total, dedupe,
                          parse_int, retry, slugify)
from ..models.{name} import new_{name}
from ..repositories.{name}_repo import {cap(name)}Repository


class {cap(name)}Service:
    """Validates, transforms and aggregates {name} records."""

    def __init__(self, repository=None):
        self._repo = repository or {cap(name)}Repository()

    def create(self, payload):
        """Validate a payload and store a new {name}."""
        record = new_{name}(
            {num}=clamp(float(payload.get("{num}", 0)), 0.0, 1e9),
            {strf}=slugify(payload.get("{strf}", "")),
        )
        return self._repo.put(record)

    def validate(self, payload):
        """Return True when the numeric field is within a safe range."""
        value = parse_int(payload.get("{num}"), -1)
        return value == clamp(value, 0, 1_000_000)

    def domain_metric(self, record):
        """Compute the {name}-specific metric: {doc}."""
        total = float(record.{num})
        items = float(getattr(record, "items", 1) or 1)
        overdue = float(getattr(record, "due_days", 0) or 0)
        stock = float(getattr(record, "stock", 0) or 0)
        distance = float(getattr(record, "distance", 0) or 0)
        severity = float(getattr(record, "severity", 1) or 1)
        term_months = float(getattr(record, "term_months", 1) or 1)
        period_months = float(getattr(record, "period_months", 1) or 1)
        hits = float(getattr(record, "hits", 0) or 0)
        noise = float(getattr(record, "noise", 0) or 0)
        settled = bool(getattr(record, "settled", True))
        active = bool(getattr(record, "active", True))
        bounce = bool(getattr(record, "bounce", False))
        valid = bool(getattr(record, "valid", True))
        frozen = bool(getattr(record, "frozen", False))
        processed = bool(getattr(record, "processed", True))
        indexed = bool(getattr(record, "indexed", True))
        read = bool(getattr(record, "read", False))
        return {expr}

    def summarize(self, records):
        """Aggregate the numeric field and the domain metric over records."""
        rows = [r.__dict__ for r in records]
        total = compute_total(rows, "{num}")
        metric = sum(self.domain_metric(r) for r in records)
        return {{"n": len(records), "total": round(total, 3), "metric": round(metric, 3)}}

    def process(self, records):
        """Deduplicate and store records with retry on transient failure."""
        unique = dedupe(records)
        return retry(lambda: self._repo.bulk_put(unique))

    def reprocess_failed(self, records):
        """Retry processing with explicit exponential backoff."""
        for attempt in range(3):
            try:
                return self._repo.bulk_put(records)
            except Exception:  # noqa: BLE001
                _ = backoff_delay(attempt)
        return 0
'''


def gen_handler(name, spec):
    return f'''"""Request handler for {name} resources (API layer)."""

from __future__ import annotations

from ..core.utils import parse_duration
from ..services.{name}_service import {cap(name)}Service


class {cap(name)}Handler:
    """Routes {name} CRUD requests to the service."""

    def __init__(self, service=None):
        self._service = service or {cap(name)}Service()

    def create(self, payload):
        """Create a {name} and return its key."""
        record = self._service.create(payload)
        return {{"created": record.__dict__}}

    def list(self, payload):
        """List {name} records, optionally filtered."""
        field = payload.get("filter_field")
        if field:
            rows = self._service._repo.filter_by(field, payload.get("filter_value"))
        else:
            rows = self._service._repo.list()
        return {{"items": [r.__dict__ for r in rows]}}

    def summary(self, payload):
        """Return an aggregate summary of {name} records."""
        return self._service.summarize(self._service._repo.list())

    def timeout_ms(self, payload):
        """Parse a request timeout into milliseconds."""
        return parse_duration(payload.get("timeout", "5s"))
'''


CORE_ERRORS = '''"""Shared exception types."""


class TaskqError(Exception):
    """Base error."""


class NotFound(TaskqError):
    """Raised when a record is missing."""
'''

CORE_PROTOCOL = '''"""Framed JSON protocol."""

from __future__ import annotations

import json
import struct

_HEADER = struct.Struct("!I")


def encode_message(msg_type, payload, msg_id=0):
    """Serialize a message to a length-prefixed JSON frame."""
    body = json.dumps({"type": msg_type, "payload": payload, "id": msg_id}).encode()
    return _HEADER.pack(len(body)) + body


def decode_message(frame):
    """Parse a length-prefixed JSON frame."""
    (length,) = _HEADER.unpack(frame[: _HEADER.size])
    data = json.loads(frame[_HEADER.size : _HEADER.size + length].decode())
    return data["type"], data.get("payload", {}), data.get("id", 0)
'''


def gen_server(names):
    imports = "\n".join(f"from ..handlers.{n}_handler import {cap(n)}Handler" for n in names)
    regs = "\n".join(f'        self.routes["{n}"] = {cap(n)}Handler()' for n in names)
    return f'''"""API server wiring all resource handlers."""

from __future__ import annotations

from ..core.protocol import decode_message, encode_message
{imports}


class Server:
    """Dispatches framed requests to the matching resource handler."""

    def __init__(self):
        self.routes = {{}}
{regs}

    def handle(self, frame):
        """Decode a request, route it, and encode the reply."""
        msg_type, payload, msg_id = decode_message(frame)
        resource, _, action = msg_type.partition(".")
        handler = self.routes.get(resource)
        if handler is None:
            return encode_message("error", {{"error": "unknown resource"}}, msg_id)
        method = getattr(handler, action or "list")
        result = method(payload)
        return encode_message(f"{{msg_type}}_reply", result, msg_id)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", type=int, default=18)
    args = ap.parse_args()
    names = list(RESOURCES)[: args.resources]

    for sub in ("core", "models", "repositories", "services", "handlers"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        (OUT / sub / "__init__.py").write_text(f'"""taskq_large {sub} package."""\n')
    (OUT / "__init__.py").write_text('"""taskq_large — large multi-file memory testbed."""\n\n__version__ = "0.1.0"\n')
    (OUT / "core" / "utils.py").write_text(UTILS)
    (OUT / "core" / "errors.py").write_text(CORE_ERRORS)
    (OUT / "core" / "protocol.py").write_text(CORE_PROTOCOL)

    for n in names:
        spec = RESOURCES[n]
        (OUT / "models" / f"{n}.py").write_text(gen_model(n, spec))
        (OUT / "repositories" / f"{n}_repo.py").write_text(gen_repo(n, spec))
        (OUT / "services" / f"{n}_service.py").write_text(gen_service(n, spec))
        (OUT / "handlers" / f"{n}_handler.py").write_text(gen_handler(n, spec))
    (OUT / "server.py").write_text(gen_server(names))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    total = 0
    nfunc = nclass = nfiles = 0
    for f in sorted(OUT.rglob("*.py")):
        src = f.read_text()
        ast.parse(src)  # validity gate
        total += len(tok(src)["input_ids"])
        tree = ast.parse(src)
        nfunc += sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) for x in ast.walk(tree))
        nclass += sum(isinstance(x, ast.ClassDef) for x in ast.walk(tree))
        nfiles += 1
    meta = {"tokens": total, "files": nfiles, "functions": nfunc, "classes": nclass,
            "resources": names}
    (OUT.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    print("generated:", OUT)
    print(json.dumps({k: v for k, v in meta.items() if k != "resources"}))


if __name__ == "__main__":
    main()
