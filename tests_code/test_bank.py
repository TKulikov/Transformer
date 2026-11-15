"""Offline tests for codemem.bank (static keys, maturity tiers, use scoring)."""

from __future__ import annotations

import torch

from codemem.bank import CodeBank, Tier


def test_add_static_embedding_and_ids():
    bank = CodeBank()
    emb = torch.tensor([1.0, 0.0])
    a = bank.add("def f(): pass", emb)
    b = bank.add("def g(): pass", torch.tensor([0.0, 1.0]))
    assert (a, b) == (0, 1)
    emb[0] = 99.0                       # mutating caller tensor must not change key
    assert torch.equal(bank.get(a).emb, torch.tensor([1.0, 0.0]))
    assert len(bank) == 2


def test_record_use_promotes_and_scores():
    bank = CodeBank(use_boost=0.05)
    cid = bank.add("x", torch.tensor([1.0, 0.0]))
    assert bank.get(cid).tier == Tier.WRITTEN
    assert bank.bonus(cid) == 0.0
    bank.record_use(cid, success=True)
    assert bank.get(cid).tier == Tier.USED
    assert bank.get(cid).uses == 1 and bank.get(cid).successes == 1
    assert abs(bank.bonus(cid) - 0.05) < 1e-9
    bank.record_use(cid, success=False)
    assert bank.get(cid).uses == 2 and bank.get(cid).successes == 1


def test_promote_verified_bonus_and_no_downgrade():
    bank = CodeBank(use_boost=0.05, verified_boost=0.1)
    cid = bank.add("x", torch.tensor([1.0, 0.0]))
    bank.promote(cid, Tier.VERIFIED)
    assert abs(bank.bonus(cid) - 0.1) < 1e-9
    bank.promote(cid, Tier.WRITTEN)     # ignored, never downgrades
    assert bank.get(cid).tier == Tier.VERIFIED