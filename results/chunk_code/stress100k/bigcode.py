"""analytics_toolkit — single-file data processing library (generated)."""

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
    return "\n".join(lines)


def dedupe(items):
    """Stable deduplication preserving first occurrence order."""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def summarize_invoice_0(records):
    """Summary statistics for invoice records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_1(records):
    """Group sensor records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_2(payload):
    """Validate a ticket payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_3(records):
    """Plain-text report for order records."""
    rows = transform_order_3(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_4(values):
    """Smoothed mean of a session series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_5(records):
    """Summary statistics for payment records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_6(records):
    """Group shipment records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_survey_7(payload):
    """Validate a survey payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class MetricStore8:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_contract_9(values):
    """Smoothed mean of a contract series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_10(records):
    """Summary statistics for visit records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_11(records):
    """Group signal records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_batch_12(payload):
    """Validate a batch payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_13(records):
    """Plain-text report for route records."""
    rows = transform_route_13(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_14(values):
    """Smoothed mean of a invoice series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_15(records):
    """Summary statistics for sensor records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_16(records):
    """Group ticket records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class OrderStore17:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_session_18(records):
    """Plain-text report for session records."""
    rows = transform_session_18(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_19(values):
    """Smoothed mean of a payment series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_20(records):
    """Summary statistics for shipment records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_21(records):
    """Group survey records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_metric_22(payload):
    """Validate a metric payload: 'score' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_23(records):
    """Plain-text report for contract records."""
    rows = transform_contract_23(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_24(values):
    """Smoothed mean of a visit series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_25(records):
    """Summary statistics for signal records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class BatchStore26:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_route_27(payload):
    """Validate a route payload: 'score' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_28(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_28(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_29(values):
    """Smoothed mean of a sensor series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_30(records):
    """Summary statistics for ticket records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_31(records):
    """Group order records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_session_32(payload):
    """Validate a session payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_33(records):
    """Plain-text report for payment records."""
    rows = transform_payment_33(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_34(values):
    """Smoothed mean of a shipment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

class SurveyStore35:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_metric_36(records):
    """Group metric records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_contract_37(payload):
    """Validate a contract payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_38(records):
    """Plain-text report for visit records."""
    rows = transform_visit_38(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_39(values):
    """Smoothed mean of a signal series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_40(records):
    """Summary statistics for batch records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_41(records):
    """Group route records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_42(payload):
    """Validate a invoice payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_43(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_43(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class TicketStore44:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_order_45(records):
    """Summary statistics for order records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_46(records):
    """Group session records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_payment_47(payload):
    """Validate a payment payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_48(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_48(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_49(values):
    """Smoothed mean of a survey series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_50(records):
    """Summary statistics for metric records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_51(records):
    """Group contract records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_visit_52(payload):
    """Validate a visit payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class SignalStore53:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_batch_54(values):
    """Smoothed mean of a batch series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_55(records):
    """Summary statistics for route records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_56(records):
    """Group invoice records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_57(payload):
    """Validate a sensor payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_58(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_58(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_59(values):
    """Smoothed mean of a order series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_60(records):
    """Summary statistics for session records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_61(records):
    """Group payment records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

class ShipmentStore62:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_survey_63(records):
    """Plain-text report for survey records."""
    rows = transform_survey_63(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_64(values):
    """Smoothed mean of a metric series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_65(records):
    """Summary statistics for contract records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_66(records):
    """Group visit records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_signal_67(payload):
    """Validate a signal payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_68(records):
    """Plain-text report for batch records."""
    rows = transform_batch_68(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_69(values):
    """Smoothed mean of a route series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_70(records):
    """Summary statistics for invoice records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SensorStore71:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_ticket_72(payload):
    """Validate a ticket payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_73(records):
    """Plain-text report for order records."""
    rows = transform_order_73(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_74(values):
    """Smoothed mean of a session series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_75(records):
    """Summary statistics for payment records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_76(records):
    """Group shipment records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_survey_77(payload):
    """Validate a survey payload: 'price' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_78(records):
    """Plain-text report for metric records."""
    rows = transform_metric_78(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_79(values):
    """Smoothed mean of a contract series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

class VisitStore80:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_signal_81(records):
    """Group signal records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_batch_82(payload):
    """Validate a batch payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_83(records):
    """Plain-text report for route records."""
    rows = transform_route_83(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_84(values):
    """Smoothed mean of a invoice series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_85(records):
    """Summary statistics for sensor records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_86(records):
    """Group ticket records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_order_87(payload):
    """Validate a order payload: 'latency' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_88(records):
    """Plain-text report for session records."""
    rows = transform_session_88(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class PaymentStore89:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_shipment_90(records):
    """Summary statistics for shipment records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_91(records):
    """Group survey records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_metric_92(payload):
    """Validate a metric payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_93(records):
    """Plain-text report for contract records."""
    rows = transform_contract_93(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_94(values):
    """Smoothed mean of a visit series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_95(records):
    """Summary statistics for signal records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_96(records):
    """Group batch records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_route_97(payload):
    """Validate a route payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class InvoiceStore98:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_sensor_99(values):
    """Smoothed mean of a sensor series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_100(records):
    """Summary statistics for ticket records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_101(records):
    """Group order records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_session_102(payload):
    """Validate a session payload: 'count' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_103(records):
    """Plain-text report for payment records."""
    rows = transform_payment_103(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_104(values):
    """Smoothed mean of a shipment series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_105(records):
    """Summary statistics for survey records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_106(records):
    """Group metric records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class ContractStore107:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_visit_108(records):
    """Plain-text report for visit records."""
    rows = transform_visit_108(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_109(values):
    """Smoothed mean of a signal series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_110(records):
    """Summary statistics for batch records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_111(records):
    """Group route records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_112(payload):
    """Validate a invoice payload: 'amount' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_113(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_113(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_114(values):
    """Smoothed mean of a ticket series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_115(records):
    """Summary statistics for order records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SessionStore116:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_payment_117(payload):
    """Validate a payment payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_118(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_118(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_119(values):
    """Smoothed mean of a survey series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_120(records):
    """Summary statistics for metric records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_121(records):
    """Group contract records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_122(payload):
    """Validate a visit payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_123(records):
    """Plain-text report for signal records."""
    rows = transform_signal_123(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_124(values):
    """Smoothed mean of a batch series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class RouteStore125:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_invoice_126(records):
    """Group invoice records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_127(payload):
    """Validate a sensor payload: 'score' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_128(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_128(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_129(values):
    """Smoothed mean of a order series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_130(records):
    """Summary statistics for session records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_131(records):
    """Group payment records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_132(payload):
    """Validate a shipment payload: 'score' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_133(records):
    """Plain-text report for survey records."""
    rows = transform_survey_133(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class MetricStore134:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_contract_135(records):
    """Summary statistics for contract records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_136(records):
    """Group visit records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_signal_137(payload):
    """Validate a signal payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_138(records):
    """Plain-text report for batch records."""
    rows = transform_batch_138(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_139(values):
    """Smoothed mean of a route series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_140(records):
    """Summary statistics for invoice records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_141(records):
    """Group sensor records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_142(payload):
    """Validate a ticket payload: 'qty' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class OrderStore143:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_session_144(values):
    """Smoothed mean of a session series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_145(records):
    """Summary statistics for payment records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_146(records):
    """Group shipment records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_survey_147(payload):
    """Validate a survey payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_148(records):
    """Plain-text report for metric records."""
    rows = transform_metric_148(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_149(values):
    """Smoothed mean of a contract series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_150(records):
    """Summary statistics for visit records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_151(records):
    """Group signal records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

class BatchStore152:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_route_153(records):
    """Plain-text report for route records."""
    rows = transform_route_153(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_154(values):
    """Smoothed mean of a invoice series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_155(records):
    """Summary statistics for sensor records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_156(records):
    """Group ticket records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_order_157(payload):
    """Validate a order payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_158(records):
    """Plain-text report for session records."""
    rows = transform_session_158(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_159(values):
    """Smoothed mean of a payment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_160(records):
    """Summary statistics for shipment records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SurveyStore161:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_metric_162(payload):
    """Validate a metric payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_163(records):
    """Plain-text report for contract records."""
    rows = transform_contract_163(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_164(values):
    """Smoothed mean of a visit series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_165(records):
    """Summary statistics for signal records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_166(records):
    """Group batch records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_route_167(payload):
    """Validate a route payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_168(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_168(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_169(values):
    """Smoothed mean of a sensor series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class TicketStore170:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_order_171(records):
    """Group order records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_session_172(payload):
    """Validate a session payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_173(records):
    """Plain-text report for payment records."""
    rows = transform_payment_173(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_174(values):
    """Smoothed mean of a shipment series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_175(records):
    """Summary statistics for survey records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_176(records):
    """Group metric records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_contract_177(payload):
    """Validate a contract payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_178(records):
    """Plain-text report for visit records."""
    rows = transform_visit_178(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class SignalStore179:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_batch_180(records):
    """Summary statistics for batch records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_181(records):
    """Group route records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_182(payload):
    """Validate a invoice payload: 'price' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_183(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_183(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_184(values):
    """Smoothed mean of a ticket series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_185(records):
    """Summary statistics for order records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_186(records):
    """Group session records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_payment_187(payload):
    """Validate a payment payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class ShipmentStore188:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_survey_189(values):
    """Smoothed mean of a survey series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_190(records):
    """Summary statistics for metric records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_191(records):
    """Group contract records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_192(payload):
    """Validate a visit payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_193(records):
    """Plain-text report for signal records."""
    rows = transform_signal_193(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_194(values):
    """Smoothed mean of a batch series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_195(records):
    """Summary statistics for route records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_196(records):
    """Group invoice records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

class SensorStore197:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_ticket_198(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_198(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_199(values):
    """Smoothed mean of a order series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_200(records):
    """Summary statistics for session records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_201(records):
    """Group payment records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_202(payload):
    """Validate a shipment payload: 'latency' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_203(records):
    """Plain-text report for survey records."""
    rows = transform_survey_203(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_204(values):
    """Smoothed mean of a metric series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_205(records):
    """Summary statistics for contract records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class VisitStore206:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_signal_207(payload):
    """Validate a signal payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_208(records):
    """Plain-text report for batch records."""
    rows = transform_batch_208(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_209(values):
    """Smoothed mean of a route series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_210(records):
    """Summary statistics for invoice records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_211(records):
    """Group sensor records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_212(payload):
    """Validate a ticket payload: 'count' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_213(records):
    """Plain-text report for order records."""
    rows = transform_order_213(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_214(values):
    """Smoothed mean of a session series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

class PaymentStore215:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_shipment_216(records):
    """Group shipment records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_survey_217(payload):
    """Validate a survey payload: 'count' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_218(records):
    """Plain-text report for metric records."""
    rows = transform_metric_218(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_219(values):
    """Smoothed mean of a contract series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_220(records):
    """Summary statistics for visit records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_221(records):
    """Group signal records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_batch_222(payload):
    """Validate a batch payload: 'count' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_223(records):
    """Plain-text report for route records."""
    rows = transform_route_223(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class InvoiceStore224:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_sensor_225(records):
    """Summary statistics for sensor records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_226(records):
    """Group ticket records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_order_227(payload):
    """Validate a order payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_228(records):
    """Plain-text report for session records."""
    rows = transform_session_228(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_229(values):
    """Smoothed mean of a payment series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_230(records):
    """Summary statistics for shipment records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_231(records):
    """Group survey records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_metric_232(payload):
    """Validate a metric payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class ContractStore233:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_visit_234(values):
    """Smoothed mean of a visit series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_235(records):
    """Summary statistics for signal records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_236(records):
    """Group batch records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_route_237(payload):
    """Validate a route payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_238(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_238(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_239(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_240(records):
    """Summary statistics for ticket records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_241(records):
    """Group order records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

class SessionStore242:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_payment_243(records):
    """Plain-text report for payment records."""
    rows = transform_payment_243(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_244(values):
    """Smoothed mean of a shipment series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_245(records):
    """Summary statistics for survey records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_246(records):
    """Group metric records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_contract_247(payload):
    """Validate a contract payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_248(records):
    """Plain-text report for visit records."""
    rows = transform_visit_248(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_249(values):
    """Smoothed mean of a signal series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_250(records):
    """Summary statistics for batch records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class RouteStore251:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_invoice_252(payload):
    """Validate a invoice payload: 'qty' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_253(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_253(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_254(values):
    """Smoothed mean of a ticket series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_255(records):
    """Summary statistics for order records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_256(records):
    """Group session records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_payment_257(payload):
    """Validate a payment payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_258(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_258(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_259(values):
    """Smoothed mean of a survey series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

class MetricStore260:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_contract_261(records):
    """Group contract records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_visit_262(payload):
    """Validate a visit payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_263(records):
    """Plain-text report for signal records."""
    rows = transform_signal_263(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_264(values):
    """Smoothed mean of a batch series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_265(records):
    """Summary statistics for route records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_266(records):
    """Group invoice records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_267(payload):
    """Validate a sensor payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_268(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_268(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class OrderStore269:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_session_270(records):
    """Summary statistics for session records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_271(records):
    """Group payment records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_272(payload):
    """Validate a shipment payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_273(records):
    """Plain-text report for survey records."""
    rows = transform_survey_273(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_274(values):
    """Smoothed mean of a metric series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_275(records):
    """Summary statistics for contract records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_276(records):
    """Group visit records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_signal_277(payload):
    """Validate a signal payload: 'duration' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class BatchStore278:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_route_279(values):
    """Smoothed mean of a route series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_280(records):
    """Summary statistics for invoice records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_281(records):
    """Group sensor records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_282(payload):
    """Validate a ticket payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_283(records):
    """Plain-text report for order records."""
    rows = transform_order_283(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_284(values):
    """Smoothed mean of a session series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_285(records):
    """Summary statistics for payment records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_286(records):
    """Group shipment records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

class SurveyStore287:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_metric_288(records):
    """Plain-text report for metric records."""
    rows = transform_metric_288(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_289(values):
    """Smoothed mean of a contract series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_290(records):
    """Summary statistics for visit records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_291(records):
    """Group signal records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_batch_292(payload):
    """Validate a batch payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_293(records):
    """Plain-text report for route records."""
    rows = transform_route_293(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_294(values):
    """Smoothed mean of a invoice series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_295(records):
    """Summary statistics for sensor records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class TicketStore296:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_order_297(payload):
    """Validate a order payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_298(records):
    """Plain-text report for session records."""
    rows = transform_session_298(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_299(values):
    """Smoothed mean of a payment series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_300(records):
    """Summary statistics for shipment records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_301(records):
    """Group survey records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_metric_302(payload):
    """Validate a metric payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_303(records):
    """Plain-text report for contract records."""
    rows = transform_contract_303(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_304(values):
    """Smoothed mean of a visit series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

class SignalStore305:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_batch_306(records):
    """Group batch records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_route_307(payload):
    """Validate a route payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_308(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_308(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_309(values):
    """Smoothed mean of a sensor series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_310(records):
    """Summary statistics for ticket records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_311(records):
    """Group order records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_session_312(payload):
    """Validate a session payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_313(records):
    """Plain-text report for payment records."""
    rows = transform_payment_313(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class ShipmentStore314:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_survey_315(records):
    """Summary statistics for survey records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_316(records):
    """Group metric records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_contract_317(payload):
    """Validate a contract payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_318(records):
    """Plain-text report for visit records."""
    rows = transform_visit_318(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_319(values):
    """Smoothed mean of a signal series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_320(records):
    """Summary statistics for batch records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_321(records):
    """Group route records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_322(payload):
    """Validate a invoice payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class SensorStore323:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_ticket_324(values):
    """Smoothed mean of a ticket series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_325(records):
    """Summary statistics for order records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_326(records):
    """Group session records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_payment_327(payload):
    """Validate a payment payload: 'count' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_328(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_328(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_329(values):
    """Smoothed mean of a survey series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_330(records):
    """Summary statistics for metric records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_331(records):
    """Group contract records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

class VisitStore332:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_signal_333(records):
    """Plain-text report for signal records."""
    rows = transform_signal_333(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_334(values):
    """Smoothed mean of a batch series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_335(records):
    """Summary statistics for route records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_336(records):
    """Group invoice records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_337(payload):
    """Validate a sensor payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_338(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_338(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_339(values):
    """Smoothed mean of a order series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_340(records):
    """Summary statistics for session records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class PaymentStore341:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_shipment_342(payload):
    """Validate a shipment payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_343(records):
    """Plain-text report for survey records."""
    rows = transform_survey_343(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_344(values):
    """Smoothed mean of a metric series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_345(records):
    """Summary statistics for contract records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_346(records):
    """Group visit records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_signal_347(payload):
    """Validate a signal payload: 'amount' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_348(records):
    """Plain-text report for batch records."""
    rows = transform_batch_348(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_349(values):
    """Smoothed mean of a route series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

class InvoiceStore350:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_sensor_351(records):
    """Group sensor records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_352(payload):
    """Validate a ticket payload: 'score' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_353(records):
    """Plain-text report for order records."""
    rows = transform_order_353(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_354(values):
    """Smoothed mean of a session series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_355(records):
    """Summary statistics for payment records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_356(records):
    """Group shipment records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_survey_357(payload):
    """Validate a survey payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_358(records):
    """Plain-text report for metric records."""
    rows = transform_metric_358(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class ContractStore359:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_visit_360(records):
    """Summary statistics for visit records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_361(records):
    """Group signal records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_batch_362(payload):
    """Validate a batch payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_363(records):
    """Plain-text report for route records."""
    rows = transform_route_363(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_364(values):
    """Smoothed mean of a invoice series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_365(records):
    """Summary statistics for sensor records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_366(records):
    """Group ticket records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_order_367(payload):
    """Validate a order payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class SessionStore368:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_payment_369(values):
    """Smoothed mean of a payment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_370(records):
    """Summary statistics for shipment records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_371(records):
    """Group survey records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_metric_372(payload):
    """Validate a metric payload: 'qty' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_373(records):
    """Plain-text report for contract records."""
    rows = transform_contract_373(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_374(values):
    """Smoothed mean of a visit series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_375(records):
    """Summary statistics for signal records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_376(records):
    """Group batch records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class RouteStore377:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_invoice_378(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_378(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_379(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_380(records):
    """Summary statistics for ticket records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_381(records):
    """Group order records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_session_382(payload):
    """Validate a session payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_383(records):
    """Plain-text report for payment records."""
    rows = transform_payment_383(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_384(values):
    """Smoothed mean of a shipment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_385(records):
    """Summary statistics for survey records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class MetricStore386:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_contract_387(payload):
    """Validate a contract payload: 'duration' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_388(records):
    """Plain-text report for visit records."""
    rows = transform_visit_388(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_389(values):
    """Smoothed mean of a signal series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_390(records):
    """Summary statistics for batch records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_391(records):
    """Group route records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_392(payload):
    """Validate a invoice payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_393(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_393(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_394(values):
    """Smoothed mean of a ticket series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

class OrderStore395:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_session_396(records):
    """Group session records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_payment_397(payload):
    """Validate a payment payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_398(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_398(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_399(values):
    """Smoothed mean of a survey series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_400(records):
    """Summary statistics for metric records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_401(records):
    """Group contract records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_visit_402(payload):
    """Validate a visit payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_403(records):
    """Plain-text report for signal records."""
    rows = transform_signal_403(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class BatchStore404:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_route_405(records):
    """Summary statistics for route records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_406(records):
    """Group invoice records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_407(payload):
    """Validate a sensor payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_408(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_408(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_409(values):
    """Smoothed mean of a order series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_410(records):
    """Summary statistics for session records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_411(records):
    """Group payment records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_412(payload):
    """Validate a shipment payload: 'price' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class SurveyStore413:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_metric_414(values):
    """Smoothed mean of a metric series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_415(records):
    """Summary statistics for contract records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_416(records):
    """Group visit records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_signal_417(payload):
    """Validate a signal payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_418(records):
    """Plain-text report for batch records."""
    rows = transform_batch_418(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_419(values):
    """Smoothed mean of a route series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_420(records):
    """Summary statistics for invoice records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_421(records):
    """Group sensor records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

class TicketStore422:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_order_423(records):
    """Plain-text report for order records."""
    rows = transform_order_423(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_424(values):
    """Smoothed mean of a session series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_425(records):
    """Summary statistics for payment records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_426(records):
    """Group shipment records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_survey_427(payload):
    """Validate a survey payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_428(records):
    """Plain-text report for metric records."""
    rows = transform_metric_428(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_429(values):
    """Smoothed mean of a contract series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_430(records):
    """Summary statistics for visit records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SignalStore431:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_batch_432(payload):
    """Validate a batch payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_433(records):
    """Plain-text report for route records."""
    rows = transform_route_433(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_434(values):
    """Smoothed mean of a invoice series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_435(records):
    """Summary statistics for sensor records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_436(records):
    """Group ticket records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_order_437(payload):
    """Validate a order payload: 'count' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_438(records):
    """Plain-text report for session records."""
    rows = transform_session_438(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_439(values):
    """Smoothed mean of a payment series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class ShipmentStore440:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_survey_441(records):
    """Group survey records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_metric_442(payload):
    """Validate a metric payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_443(records):
    """Plain-text report for contract records."""
    rows = transform_contract_443(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_444(values):
    """Smoothed mean of a visit series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_445(records):
    """Summary statistics for signal records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_446(records):
    """Group batch records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_route_447(payload):
    """Validate a route payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_448(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_448(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class SensorStore449:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_ticket_450(records):
    """Summary statistics for ticket records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_451(records):
    """Group order records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_session_452(payload):
    """Validate a session payload: 'amount' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_453(records):
    """Plain-text report for payment records."""
    rows = transform_payment_453(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_454(values):
    """Smoothed mean of a shipment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_455(records):
    """Summary statistics for survey records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_456(records):
    """Group metric records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_contract_457(payload):
    """Validate a contract payload: 'amount' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class VisitStore458:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_signal_459(values):
    """Smoothed mean of a signal series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_460(records):
    """Summary statistics for batch records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_461(records):
    """Group route records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_462(payload):
    """Validate a invoice payload: 'score' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_463(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_463(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_464(values):
    """Smoothed mean of a ticket series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_465(records):
    """Summary statistics for order records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_466(records):
    """Group session records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

class PaymentStore467:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_shipment_468(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_468(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_469(values):
    """Smoothed mean of a survey series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_470(records):
    """Summary statistics for metric records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_471(records):
    """Group contract records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_472(payload):
    """Validate a visit payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_473(records):
    """Plain-text report for signal records."""
    rows = transform_signal_473(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_474(values):
    """Smoothed mean of a batch series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_475(records):
    """Summary statistics for route records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class InvoiceStore476:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_sensor_477(payload):
    """Validate a sensor payload: 'qty' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_478(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_478(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_479(values):
    """Smoothed mean of a order series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_480(records):
    """Summary statistics for session records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_481(records):
    """Group payment records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_482(payload):
    """Validate a shipment payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_483(records):
    """Plain-text report for survey records."""
    rows = transform_survey_483(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_484(values):
    """Smoothed mean of a metric series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class ContractStore485:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_visit_486(records):
    """Group visit records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_signal_487(payload):
    """Validate a signal payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_488(records):
    """Plain-text report for batch records."""
    rows = transform_batch_488(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_489(values):
    """Smoothed mean of a route series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_490(records):
    """Summary statistics for invoice records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_491(records):
    """Group sensor records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_492(payload):
    """Validate a ticket payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_493(records):
    """Plain-text report for order records."""
    rows = transform_order_493(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class SessionStore494:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_payment_495(records):
    """Summary statistics for payment records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_496(records):
    """Group shipment records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_survey_497(payload):
    """Validate a survey payload: 'duration' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_498(records):
    """Plain-text report for metric records."""
    rows = transform_metric_498(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_499(values):
    """Smoothed mean of a contract series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_500(records):
    """Summary statistics for visit records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_501(records):
    """Group signal records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_batch_502(payload):
    """Validate a batch payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class RouteStore503:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_invoice_504(values):
    """Smoothed mean of a invoice series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_505(records):
    """Summary statistics for sensor records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_506(records):
    """Group ticket records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_order_507(payload):
    """Validate a order payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_508(records):
    """Plain-text report for session records."""
    rows = transform_session_508(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_509(values):
    """Smoothed mean of a payment series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_510(records):
    """Summary statistics for shipment records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_511(records):
    """Group survey records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class MetricStore512:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_contract_513(records):
    """Plain-text report for contract records."""
    rows = transform_contract_513(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_514(values):
    """Smoothed mean of a visit series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_515(records):
    """Summary statistics for signal records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_516(records):
    """Group batch records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_route_517(payload):
    """Validate a route payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_518(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_518(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_519(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_520(records):
    """Summary statistics for ticket records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class OrderStore521:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_session_522(payload):
    """Validate a session payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_523(records):
    """Plain-text report for payment records."""
    rows = transform_payment_523(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_524(values):
    """Smoothed mean of a shipment series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_525(records):
    """Summary statistics for survey records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_526(records):
    """Group metric records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_contract_527(payload):
    """Validate a contract payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_528(records):
    """Plain-text report for visit records."""
    rows = transform_visit_528(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_529(values):
    """Smoothed mean of a signal series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

class BatchStore530:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_route_531(records):
    """Group route records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_532(payload):
    """Validate a invoice payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_533(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_533(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_534(values):
    """Smoothed mean of a ticket series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_535(records):
    """Summary statistics for order records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_536(records):
    """Group session records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_payment_537(payload):
    """Validate a payment payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_538(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_538(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class SurveyStore539:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_metric_540(records):
    """Summary statistics for metric records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_541(records):
    """Group contract records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_542(payload):
    """Validate a visit payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_543(records):
    """Plain-text report for signal records."""
    rows = transform_signal_543(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_544(values):
    """Smoothed mean of a batch series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_545(records):
    """Summary statistics for route records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_546(records):
    """Group invoice records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_547(payload):
    """Validate a sensor payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class TicketStore548:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_order_549(values):
    """Smoothed mean of a order series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_550(records):
    """Summary statistics for session records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_551(records):
    """Group payment records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_552(payload):
    """Validate a shipment payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_553(records):
    """Plain-text report for survey records."""
    rows = transform_survey_553(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_554(values):
    """Smoothed mean of a metric series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_555(records):
    """Summary statistics for contract records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_556(records):
    """Group visit records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

class SignalStore557:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_batch_558(records):
    """Plain-text report for batch records."""
    rows = transform_batch_558(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_559(values):
    """Smoothed mean of a route series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_560(records):
    """Summary statistics for invoice records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_561(records):
    """Group sensor records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_562(payload):
    """Validate a ticket payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_563(records):
    """Plain-text report for order records."""
    rows = transform_order_563(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_564(values):
    """Smoothed mean of a session series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_565(records):
    """Summary statistics for payment records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class ShipmentStore566:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_survey_567(payload):
    """Validate a survey payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_568(records):
    """Plain-text report for metric records."""
    rows = transform_metric_568(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_569(values):
    """Smoothed mean of a contract series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_570(records):
    """Summary statistics for visit records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_571(records):
    """Group signal records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_batch_572(payload):
    """Validate a batch payload: 'amount' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_573(records):
    """Plain-text report for route records."""
    rows = transform_route_573(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_574(values):
    """Smoothed mean of a invoice series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

class SensorStore575:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_ticket_576(records):
    """Group ticket records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_order_577(payload):
    """Validate a order payload: 'score' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_578(records):
    """Plain-text report for session records."""
    rows = transform_session_578(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_579(values):
    """Smoothed mean of a payment series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_580(records):
    """Summary statistics for shipment records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_581(records):
    """Group survey records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_metric_582(payload):
    """Validate a metric payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_583(records):
    """Plain-text report for contract records."""
    rows = transform_contract_583(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class VisitStore584:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_signal_585(records):
    """Summary statistics for signal records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_586(records):
    """Group batch records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_route_587(payload):
    """Validate a route payload: 'score' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_588(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_588(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_589(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_590(records):
    """Summary statistics for ticket records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_591(records):
    """Group order records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_session_592(payload):
    """Validate a session payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class PaymentStore593:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_shipment_594(values):
    """Smoothed mean of a shipment series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_595(records):
    """Summary statistics for survey records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_596(records):
    """Group metric records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_contract_597(payload):
    """Validate a contract payload: 'qty' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_598(records):
    """Plain-text report for visit records."""
    rows = transform_visit_598(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_599(values):
    """Smoothed mean of a signal series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_600(records):
    """Summary statistics for batch records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_601(records):
    """Group route records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class InvoiceStore602:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_sensor_603(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_603(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_604(values):
    """Smoothed mean of a ticket series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_605(records):
    """Summary statistics for order records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_606(records):
    """Group session records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_payment_607(payload):
    """Validate a payment payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_608(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_608(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_609(values):
    """Smoothed mean of a survey series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_610(records):
    """Summary statistics for metric records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class ContractStore611:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_visit_612(payload):
    """Validate a visit payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_613(records):
    """Plain-text report for signal records."""
    rows = transform_signal_613(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_614(values):
    """Smoothed mean of a batch series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_615(records):
    """Summary statistics for route records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_616(records):
    """Group invoice records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_617(payload):
    """Validate a sensor payload: 'weight' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_618(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_618(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_619(values):
    """Smoothed mean of a order series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class SessionStore620:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_payment_621(records):
    """Group payment records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_622(payload):
    """Validate a shipment payload: 'weight' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_623(records):
    """Plain-text report for survey records."""
    rows = transform_survey_623(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_624(values):
    """Smoothed mean of a metric series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_625(records):
    """Summary statistics for contract records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_626(records):
    """Group visit records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_signal_627(payload):
    """Validate a signal payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_628(records):
    """Plain-text report for batch records."""
    rows = transform_batch_628(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class RouteStore629:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_invoice_630(records):
    """Summary statistics for invoice records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_631(records):
    """Group sensor records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_632(payload):
    """Validate a ticket payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_633(records):
    """Plain-text report for order records."""
    rows = transform_order_633(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_634(values):
    """Smoothed mean of a session series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_635(records):
    """Summary statistics for payment records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_636(records):
    """Group shipment records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_survey_637(payload):
    """Validate a survey payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class MetricStore638:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_contract_639(values):
    """Smoothed mean of a contract series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_visit_640(records):
    """Summary statistics for visit records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "visit", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_signal_641(records):
    """Group signal records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_batch_642(payload):
    """Validate a batch payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_643(records):
    """Plain-text report for route records."""
    rows = transform_route_643(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_644(values):
    """Smoothed mean of a invoice series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_645(records):
    """Summary statistics for sensor records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_646(records):
    """Group ticket records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

class OrderStore647:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_session_648(records):
    """Plain-text report for session records."""
    rows = transform_session_648(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_payment_649(values):
    """Smoothed mean of a payment series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_shipment_650(records):
    """Summary statistics for shipment records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_651(records):
    """Group survey records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_metric_652(payload):
    """Validate a metric payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_653(records):
    """Plain-text report for contract records."""
    rows = transform_contract_653(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_654(values):
    """Smoothed mean of a visit series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_655(records):
    """Summary statistics for signal records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class BatchStore656:
    """In-memory store for batch records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_route_657(payload):
    """Validate a route payload: 'latency' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_invoice_658(records):
    """Plain-text report for invoice records."""
    rows = transform_invoice_658(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "invoice report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_sensor_659(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_660(records):
    """Summary statistics for ticket records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_661(records):
    """Group order records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_session_662(payload):
    """Validate a session payload: 'count' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_663(records):
    """Plain-text report for payment records."""
    rows = transform_payment_663(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_664(values):
    """Smoothed mean of a shipment series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

class SurveyStore665:
    """In-memory store for survey records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_metric_666(records):
    """Group metric records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_contract_667(payload):
    """Validate a contract payload: 'count' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_visit_668(records):
    """Plain-text report for visit records."""
    rows = transform_visit_668(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: count"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_669(values):
    """Smoothed mean of a signal series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_670(records):
    """Summary statistics for batch records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_671(records):
    """Group route records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_672(payload):
    """Validate a invoice payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_673(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_673(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class TicketStore674:
    """In-memory store for ticket records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_order_675(records):
    """Summary statistics for order records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_session_676(records):
    """Group session records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_payment_677(payload):
    """Validate a payment payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_678(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_678(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: amount"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_679(values):
    """Smoothed mean of a survey series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_680(records):
    """Summary statistics for metric records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_681(records):
    """Group contract records and cap per-group 'amount' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "amount")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_682(payload):
    """Validate a visit payload: 'amount' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("amount"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class SignalStore683:
    """In-memory store for signal records with helper-based analytics."""

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
        """Total over 'amount' for active rows."""
        return compute_total(self.active(), "amount")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "amount")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_batch_684(values):
    """Smoothed mean of a batch series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_route_685(records):
    """Summary statistics for route records over 'amount'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "amount")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("amount", 0.0) for r in rows if isinstance(r.get("amount"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "route", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_invoice_686(records):
    """Group invoice records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_687(payload):
    """Validate a sensor payload: 'score' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_688(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_688(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_689(values):
    """Smoothed mean of a order series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_690(records):
    """Summary statistics for session records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_691(records):
    """Group payment records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

class ShipmentStore692:
    """In-memory store for shipment records with helper-based analytics."""

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
        """Total over 'score' for active rows."""
        return compute_total(self.active(), "score")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "score")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_survey_693(records):
    """Plain-text report for survey records."""
    rows = transform_survey_693(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_metric_694(values):
    """Smoothed mean of a metric series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_contract_695(records):
    """Summary statistics for contract records over 'score'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "score")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("score", 0.0) for r in rows if isinstance(r.get("score"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_696(records):
    """Group visit records and cap per-group 'score' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "score")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_signal_697(payload):
    """Validate a signal payload: 'score' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("score"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_698(records):
    """Plain-text report for batch records."""
    rows = transform_batch_698(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: score"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_699(values):
    """Smoothed mean of a route series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_700(records):
    """Summary statistics for invoice records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SensorStore701:
    """In-memory store for sensor records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_ticket_702(payload):
    """Validate a ticket payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_order_703(records):
    """Plain-text report for order records."""
    rows = transform_order_703(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "order report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_session_704(values):
    """Smoothed mean of a session series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_705(records):
    """Summary statistics for payment records over 'qty'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "qty")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("qty", 0.0) for r in rows if isinstance(r.get("qty"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_706(records):
    """Group shipment records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_survey_707(payload):
    """Validate a survey payload: 'qty' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_metric_708(records):
    """Plain-text report for metric records."""
    rows = transform_metric_708(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "metric report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_contract_709(values):
    """Smoothed mean of a contract series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

class VisitStore710:
    """In-memory store for visit records with helper-based analytics."""

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
        """Total over 'qty' for active rows."""
        return compute_total(self.active(), "qty")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "qty")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_signal_711(records):
    """Group signal records and cap per-group 'qty' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "qty")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_batch_712(payload):
    """Validate a batch payload: 'qty' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("qty"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_route_713(records):
    """Plain-text report for route records."""
    rows = transform_route_713(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "route report: qty"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_invoice_714(values):
    """Smoothed mean of a invoice series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_sensor_715(records):
    """Summary statistics for sensor records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "sensor", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_ticket_716(records):
    """Group ticket records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_order_717(payload):
    """Validate a order payload: 'duration' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_session_718(records):
    """Plain-text report for session records."""
    rows = transform_session_718(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "session report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class PaymentStore719:
    """In-memory store for payment records with helper-based analytics."""

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
        """Total over 'duration' for active rows."""
        return compute_total(self.active(), "duration")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "duration")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_shipment_720(records):
    """Summary statistics for shipment records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "shipment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_survey_721(records):
    """Group survey records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 50.0),
                    "slug": slugify(name)})
    return out

def validate_metric_722(payload):
    """Validate a metric payload: 'duration' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_contract_723(records):
    """Plain-text report for contract records."""
    rows = transform_contract_723(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "contract report: duration"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_visit_724(values):
    """Smoothed mean of a visit series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_signal_725(records):
    """Summary statistics for signal records over 'duration'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "duration")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("duration", 0.0) for r in rows if isinstance(r.get("duration"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "signal", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_batch_726(records):
    """Group batch records and cap per-group 'duration' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "duration")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_route_727(payload):
    """Validate a route payload: 'duration' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("duration"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class InvoiceStore728:
    """In-memory store for invoice records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_sensor_729(values):
    """Smoothed mean of a sensor series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_ticket_730(records):
    """Summary statistics for ticket records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "ticket", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_order_731(records):
    """Group order records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_session_732(payload):
    """Validate a session payload: 'weight' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("weight"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_payment_733(records):
    """Plain-text report for payment records."""
    rows = transform_payment_733(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "payment report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_shipment_734(values):
    """Smoothed mean of a shipment series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_survey_735(records):
    """Summary statistics for survey records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "survey", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_metric_736(records):
    """Group metric records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

class ContractStore737:
    """In-memory store for contract records with helper-based analytics."""

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
        """Total over 'weight' for active rows."""
        return compute_total(self.active(), "weight")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "weight")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def render_visit_738(records):
    """Plain-text report for visit records."""
    rows = transform_visit_738(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "visit report: weight"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_signal_739(values):
    """Smoothed mean of a signal series (window=3)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 3)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_batch_740(records):
    """Summary statistics for batch records over 'weight'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "weight")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("weight", 0.0) for r in rows if isinstance(r.get("weight"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "batch", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_route_741(records):
    """Group route records and cap per-group 'weight' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "weight")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_invoice_742(payload):
    """Validate a invoice payload: 'price' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_sensor_743(records):
    """Plain-text report for sensor records."""
    rows = transform_sensor_743(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "sensor report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_ticket_744(values):
    """Smoothed mean of a ticket series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_order_745(records):
    """Summary statistics for order records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "order", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

class SessionStore746:
    """In-memory store for session records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def validate_payment_747(payload):
    """Validate a payment payload: 'price' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_shipment_748(records):
    """Plain-text report for shipment records."""
    rows = transform_shipment_748(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "shipment report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_survey_749(values):
    """Smoothed mean of a survey series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_metric_750(records):
    """Summary statistics for metric records over 'price'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "price")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("price", 0.0) for r in rows if isinstance(r.get("price"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "metric", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_contract_751(records):
    """Group contract records and cap per-group 'price' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "price")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_visit_752(payload):
    """Validate a visit payload: 'price' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("price"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_signal_753(records):
    """Plain-text report for signal records."""
    rows = transform_signal_753(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "signal report: price"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_batch_754(values):
    """Smoothed mean of a batch series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

class RouteStore755:
    """In-memory store for route records with helper-based analytics."""

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
        """Total over 'price' for active rows."""
        return compute_total(self.active(), "price")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "price")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def transform_invoice_756(records):
    """Group invoice records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_sensor_757(payload):
    """Validate a sensor payload: 'latency' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_ticket_758(records):
    """Plain-text report for ticket records."""
    rows = transform_ticket_758(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "ticket report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_order_759(values):
    """Smoothed mean of a order series (window=7)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 7)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_session_760(records):
    """Summary statistics for session records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "session", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_payment_761(records):
    """Group payment records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 500.0),
                    "slug": slugify(name)})
    return out

def validate_shipment_762(payload):
    """Validate a shipment payload: 'latency' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_survey_763(records):
    """Plain-text report for survey records."""
    rows = transform_survey_763(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "survey report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

class MetricStore764:
    """In-memory store for metric records with helper-based analytics."""

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
        """Total over 'latency' for active rows."""
        return compute_total(self.active(), "latency")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "latency")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def summarize_contract_765(records):
    """Summary statistics for contract records over 'latency'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "latency")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("latency", 0.0) for r in rows if isinstance(r.get("latency"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "contract", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_visit_766(records):
    """Group visit records and cap per-group 'latency' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "latency")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_signal_767(payload):
    """Validate a signal payload: 'latency' must be in [1, 1000]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("latency"), default=1 - 1)
    if value != clamp(value, 1, 1000):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

def render_batch_768(records):
    """Plain-text report for batch records."""
    rows = transform_batch_768(records)
    table_rows = [(r["slug"], r["n"], r["total"]) for r in rows]
    header = "batch report: latency"
    body = format_table(table_rows, ["region", "n", "total"])
    return header + "\n" + body

def aggregate_route_769(values):
    """Smoothed mean of a route series (window=5)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 5)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_invoice_770(records):
    """Summary statistics for invoice records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "invoice", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_sensor_771(records):
    """Group sensor records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 100.0),
                    "slug": slugify(name)})
    return out

def validate_ticket_772(payload):
    """Validate a ticket payload: 'count' must be in [0, 100]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=0 - 1)
    if value != clamp(value, 0, 100):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))

class OrderStore773:
    """In-memory store for order records with helper-based analytics."""

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
        """Total over 'count' for active rows."""
        return compute_total(self.active(), "count")

    def by_region(self):
        """Active rows grouped by region."""
        return group_by(self.active(), "region")

    def export(self):
        """Render the store as a text table."""
        groups = self.by_region()
        rows = [(slugify(k), len(v), compute_total(v, "count")) for k, v in sorted(groups.items())]
        return format_table(rows, ["region", "n", "total"])

def aggregate_session_774(values):
    """Smoothed mean of a session series (window=12)."""
    clean = dedupe([float(v) for v in values if isinstance(v, (int, float))])
    if not clean:
        return 0.0
    smoothed = moving_average(clean, 12)
    return round(sum(smoothed) / len(smoothed), 4)

def summarize_payment_775(records):
    """Summary statistics for payment records over 'count'."""
    rows = normalize_records(records)
    rows = filter_active(rows)
    total = compute_total(rows, "count")
    n = len(rows)
    mean = total / n if n else 0.0
    values = [r.get("count", 0.0) for r in rows if isinstance(r.get("count"), (int, float))]
    spread = (max(values) - min(values)) if values else 0.0
    return {"topic": "payment", "n": n, "total": round(total, 2),
            "mean": round(mean, 3), "spread": round(spread, 3)}

def transform_shipment_776(records):
    """Group shipment records and cap per-group 'count' totals."""
    rows = normalize_records(records)
    groups = group_by(rows, "region")
    out = []
    for name, group in sorted(groups.items()):
        total = compute_total(group, "count")
        out.append({"region": name, "n": len(group),
                    "total": clamp(total, 0.0, 10.0),
                    "slug": slugify(name)})
    return out

def validate_survey_777(payload):
    """Validate a survey payload: 'count' must be in [-50, 50]."""
    if not isinstance(payload, dict):
        return False
    value = parse_int(payload.get("count"), default=-50 - 1)
    if value != clamp(value, -50, 50):
        return False
    name = payload.get("name", "")
    return bool(slugify(str(name)))


# --- planned next (not implemented yet) -----------------------------------
# build_summary_report(records): normalize, keep active, total 'amount',
#   group by region and render one table with a smoothed trend per region.
# trend_report(values_by_group): smooth each group's series, then average.
# export_active_total(records): active rows only, total 'amount'.
