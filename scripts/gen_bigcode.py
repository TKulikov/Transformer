"""Генератор большого связного Python-файла (~100k токенов) для стресс-теста.

Файл — единая библиотека analytics_toolkit: сначала базовые хелперы, затем
сотни тематических функций и классы, которые ВЫЗЫВАЮТ ранние хелперы (это
даёт цели для проб на переиспользование из памяти). Детерминирован (seed),
валиден (ast.parse), без внешних зависимостей. В конце — закомментированный
план следующих функций: естественная точка «дописать».

Запуск: .venv/bin/python scripts/gen_bigcode.py [--target-tokens 100000]
Выход:  results/chunk_code/stress100k/bigcode.py + meta.json
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "stress100k"

HELPERS = '''"""analytics_toolkit — single-file data processing library (generated)."""

from __future__ import annotations

import math
import re
from collections import defaultdict


def normalize_records(records):
    """Drop malformed rows and strip string fields."""
    out = []
    for row in records:
        if not isinstance(row, dict):
            continue
        out.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return out


def filter_active(records):
    """Keep only rows whose status field marks them active."""
    return [r for r in records if r.get("status", "active") == "active"]


def compute_total(records, field):
    """Sum a numeric field over records, ignoring non-numeric values."""
    total = 0.0
    for r in records:
        v = r.get(field)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += float(v)
    return total


def group_by(records, key):
    """Group records into a dict of lists by the given field."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)
    return dict(groups)


def moving_average(values, window):
    """Simple moving average; window is clamped to the series length."""
    if not values:
        return []
    window = max(1, min(window, len(values)))
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def clamp(value, lo, hi):
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


def slugify(text):
    """Lowercase slug with dashes instead of non-alphanumeric runs."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_int(text, default=0):
    """Parse an int from text, falling back to default."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def format_table(rows, headers):
    """Render rows as a fixed-width plain-text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\\n".join(lines)


def dedupe(items):
    """Stable deduplication preserving first occurrence order."""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
'''

TOPICS = ["invoice", "sensor", "ticket", "order", "session", "payment", "shipment",
          "survey", "metric", "contract", "visit", "signal", "batch", "route"]
FIELDS = ["amount", "score", "qty", "duration", "weight", "price", "latency", "count"]


def fn_summary(i, topic, field, rng):
    return f'''
def summarize_{topic}_{i}(records):
    """Summary statistics for {topic} records over '{field}'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "{field}")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("{field}", 0.0) for r in rows if isinstance(r.get("{field}"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {{"topic": "{topic}", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}}
'''


def fn_transform(i, topic, field, rng):
    cap = rng.choice([10, 50, 100, 500])
    return f'''
def transform_{topic}_{i}(records):
    """Group {topic} records and cap per-group '{field}' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "{field}")
        out.append({{"region": name, "n": len(group),
                    "total": clamp(total, 0.0, {cap}.0),
                    "slug": slugify(name)}})
    return out
'''


def fn_validate(i, topic, field, rng):
    lo, hi = rng.choice([(0, 100), (1, 1000), (-50, 50)])
    return f'''
def validate_{topic}_{i}(payload):
    """Validate a {topic} payload: '{field}' must be in [{lo}, {hi}]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("{field}"), default={lo} - 1)
    if value != clamp(value, {lo}, {hi}):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))
'''


def fn_render(i, topic, field, rng):
    return f'''
def render_{topic}_{i}(records):
    """Plain-text report for {topic} records."""
    rows = transform_{topic}_{i}(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "{topic} report: {field}"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\\n" + body
'''


def fn_aggregate(i, topic, field, rng):
    win = rng.choice([3, 5, 7, 12])
    return f'''
def aggregate_{topic}_{i}(values):
    """Smoothed mean of a {topic} series (window={win})."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, {win})
    return round(sum(smoothed) / len(smoothed), 4)
'''


def cls_store(i, topic, field, rng):
    return f'''
class {topic.capitalize()}Store{i}:
    """In-memory store for {topic} records with helper-based analytics."""

    def __init__(self):
        self._rows = []

    def add(self, record):
        """Append one record after normalization."""
        self._rows.extend(normalize_records([record]))
        return len(self._rows)

    def active(self):
        """Active rows only."""
        return filter_active(self._rows)

    def total(self):
        """Total over '{field}' for active rows."""
        return compute_total(self.active(), "{field}")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "{field}")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])
'''

TEMPLATES = [fn_summary, fn_transform, fn_validate, fn_render, fn_aggregate]


def generate(target_tokens: int) -> tuple[str, dict]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    rng = random.Random(42)
    parts = [HELPERS]
    i = 0
    n_tokens = len(tok(HELPERS)["input_ids"])
    while n_tokens < target_tokens:
        topic = TOPICS[i % len(TOPICS)]
        field = FIELDS[(i // len(TOPICS)) % len(FIELDS)]
        if i % 9 == 8:
            piece = cls_store(i, topic, field, rng)
        else:
            piece = TEMPLATES[i % len(TEMPLATES)](i, topic, field, rng)
        parts.append(piece)
        n_tokens += len(tok(piece)["input_ids"])
        i += 1
    parts.append('''

# --- planned next (not implemented yet) -----------------------------------
# build_summary_report(records): normalize, keep active, total 'amount',
#   group by region and render one table with a smoothed trend per region.
# trend_report(values_by_group): smooth each group's series, then average.
# export_active_total(records): active rows only, total 'amount'.
''')
    text = "".join(parts)
    tree = ast.parse(text)  # валидность — обязательна
    n_funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    n_classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    meta = {"tokens": len(tok(text)["input_ids"]), "lines": text.count("\n") + 1,
            "functions": n_funcs, "classes": n_classes, "generated_units": i,
            "target_tokens": target_tokens, "seed": 42}
    return text, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=100000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    text, meta = generate(args.target_tokens)
    (OUT / "bigcode.py").write_text(text, encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("generated:", OUT / "bigcode.py")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
