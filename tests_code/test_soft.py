"""Offline tests for soft routing and goal+need blended retrieval."""

from __future__ import annotations

import torch

from codemem.bank import CodeBank
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from codemem.tree import GoalTree


def _setup():
    bank = CodeBank()
    tree = GoalTree(bank, centered=False, mmr_lambda=1.0)
    root = tree.add_goal("app", torch.tensor([0.5, 0.5]))
    a = tree.add_goal("A", torch.tensor([1.0, 0.0]), parent=root)
    b = tree.add_goal("B", torch.tensor([0.0, 1.0]), parent=root)
    ca = bank.add("a fn", torch.tensor([0.9, 0.1]))
    cb = bank.add("b fn", torch.tensor([0.1, 0.9]))
    tree.nodes[a].chunk_ids.append(ca); tree.nodes[a].centroid.update(bank.get(ca).emb)
    tree.nodes[b].chunk_ids.append(cb); tree.nodes[b].centroid.update(bank.get(cb).emb)
    return bank, tree, a, b, ca, cb


def test_route_soft_weights_sum_to_one():
    bank, tree, a, b, ca, cb = _setup()
    out = tree.route_soft(torch.tensor([0.95, 0.05]), m=2)
    assert abs(sum(w for _, w in out) - 1.0) < 1e-6
    assert out[0][0] == a and out[0][1] > out[1][1]


def test_route_soft_m1_is_hard():
    bank, tree, a, b, ca, cb = _setup()
    out = tree.route_soft(torch.tensor([0.95, 0.05]), m=1)
    assert out == [(a, 1.0)]


def test_retrieve_soft_goal_plus_need():
    bank, tree, a, b, ca, cb = _setup()
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=1.0, centered=False)
    out = r.retrieve_soft(torch.tensor([1.0, 0.0]), torch.tensor([0.95, 0.05]), 0, m=2, need_weight=0.5)
    assert out[0] == ca


def test_blend_disambiguates_via_need():
    # ambiguous goal (leans A) but need points at B -> blend routes to B
    bank, tree, a, b, ca, cb = _setup()
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=1.0, centered=False)
    out = r.retrieve_soft(torch.tensor([0.6, 0.4]), torch.tensor([0.1, 0.9]), 0, m=2, need_weight=0.5)
    assert out[0] == cb


def test_need_weight_zero_goal_dominates():
    bank, tree, a, b, ca, cb = _setup()
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=1.0, centered=False)
    out = r.retrieve_soft(torch.tensor([1.0, 0.0]), torch.tensor([0.1, 0.9]), 0, m=2, need_weight=0.0)
    assert out[0] == ca                 # goal-only routing ignores the B-pointing need


def test_route_soft_by_centroid_uses_function_mean():
    # node A label points at [1,0] but its function centroid is [0.2,0.8] (near B);
    # centroid routing of a B-ish query must prefer A (whose functions are B-like).
    bank = CodeBank()
    tree = GoalTree(bank, centered=False)
    root = tree.add_goal("app", torch.tensor([0.5, 0.5]))
    a = tree.add_goal("A", torch.tensor([1.0, 0.0]), parent=root)   # label says A
    b = tree.add_goal("B", torch.tensor([0.0, 1.0]), parent=root)
    ca = bank.add("a fn", torch.tensor([0.2, 0.8]))                 # but functions are B-like
    cb = bank.add("b fn", torch.tensor([0.9, 0.1]))
    tree.nodes[a].chunk_ids.append(ca); tree.nodes[a].centroid.update(bank.get(ca).emb)
    tree.nodes[b].chunk_ids.append(cb); tree.nodes[b].centroid.update(bank.get(cb).emb)
    q = torch.tensor([0.1, 0.9])
    label_top = tree.route_soft(q, m=1, by_centroid=False)[0][0]
    centroid_top = tree.route_soft(q, m=1, by_centroid=True)[0][0]
    assert label_top == b               # label routing picks B
    assert centroid_top == a            # centroid routing picks A (functions match query)


def test_adaptive_high_threshold_falls_back_to_flat():
    bank, tree, a, b, ca, cb = _setup()
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=0.5, centered=False)
    # threshold 2.0 can never be met -> always flat over all chunks
    out = r.retrieve_adaptive(torch.tensor([1.0, 0.0]), torch.tensor([0.95, 0.05]), 0, conf_threshold=2.0)
    assert out[0] == ca                 # flat picks the most need-relevant chunk


def test_adaptive_low_threshold_uses_goal_routing():
    bank, tree, a, b, ca, cb = _setup()
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=0.5, centered=False)
    out = r.retrieve_adaptive(torch.tensor([1.0, 0.0]), torch.tensor([0.95, 0.05]), 0, conf_threshold=-1.0)
    assert out[0] == ca                 # confident goal routing also picks ca