"""Offline tests for codemem.prefetch (step TTL, scan window, eviction, dedup)."""

from __future__ import annotations

from codemem.prefetch import PrefetchBank


def test_ttl_one_only_current_step():
    p = PrefetchBank(ttl_steps=1)
    p.put(0, [1, 2])
    assert p.scan(0) == [1, 2]
    assert p.scan(1) == []              # step 0 expired at step 1


def test_ttl_two_window_and_advance():
    p = PrefetchBank(ttl_steps=2)
    p.put(0, [1])
    p.put(1, [2])
    assert p.scan(1) == [1, 2]          # steps 0 and 1 live
    assert p.scan(2) == [2]             # step 0 expired
    p.advance(2)
    assert len(p) == 1                  # batch 0 evicted


def test_scan_dedup_order_stable():
    p = PrefetchBank(ttl_steps=1)
    p.put(0, [3, 3, 1])
    assert p.scan(0) == [3, 1]


def test_invalid_ttl_raises():
    try:
        PrefetchBank(ttl_steps=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for ttl_steps=0")