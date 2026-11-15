"""Offline tests for codemem.tree (routing, centered search, write-back)."""

from __future__ import annotations

import torch

from codemem.bank import CodeBank
from codemem.tree import GoalTree


def _tree():
    bank = CodeBank()
    return bank, GoalTree(bank, tau_assimilate=0.6)


def test_route_descends_to_matching_child():
    bank, tree = _tree()
    root = tree.add_goal("app", torch.tensor([0.5, 0.5, 0.0]))
    a = tree.add_goal("auth", torch.tensor([1.0, 0.0, 0.0]), parent=root)
    tree.add_goal("db", torch.tensor([0.0, 1.0, 0.0]), parent=root)
    assert tree.route(torch.tensor([0.95, 0.05, 0.0])) == a


def test_write_back_assimilates_similar_chunk():
    bank, tree = _tree()
    root = tree.add_goal("auth", torch.tensor([1.0, 0.0]))
    cid = bank.add("def hash(): ...", torch.tensor([0.95, 0.05]))
    node = tree.write_back(cid, torch.tensor([1.0, 0.0]))
    assert node == root
    assert tree.nodes[root].chunk_ids == [cid]
    assert bank.get(cid).goal_id == root
    assert len(tree.nodes[root].centroid) == 1


def test_write_back_creates_solved_node_when_dissimilar():
    bank, tree = _tree()
    root = tree.add_goal("auth", torch.tensor([1.0, 0.0]))
    c1 = bank.add("hash", torch.tensor([0.95, 0.05]))
    tree.write_back(c1, torch.tensor([1.0, 0.0]))      # seeds root
    c2 = bank.add("db connect", torch.tensor([0.0, 1.0]))
    node = tree.write_back(c2, torch.tensor([1.0, 0.0]))
    assert node != root
    new = tree.nodes[node]
    assert new.parent == root and new.solved is True
    assert new.chunk_ids == [c2]
    assert tree.nodes[root].chunk_ids == [c1]          # root unchanged


def test_search_centered_returns_members():
    bank, tree = _tree()
    root = tree.add_goal("auth", torch.tensor([1.0, 0.0]))
    c1 = bank.add("a", torch.tensor([1.0, 0.1]))
    c2 = bank.add("b", torch.tensor([1.0, -0.1]))
    tree.write_back(c1, torch.tensor([1.0, 0.0]))
    tree.write_back(c2, torch.tensor([1.0, 0.0]))
    out = tree.search(root, torch.tensor([1.0, 0.05]), k=2)
    assert out == [c1, c2]              # c1 aligns with need deviation first