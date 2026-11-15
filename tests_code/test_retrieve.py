"""Offline tests for codemem.retrieve (unified cascade, prefetch not privileged)."""

from __future__ import annotations

import torch

from codemem.bank import CodeBank
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from codemem.tree import GoalTree


def test_unified_ranking_demotes_prefetch_clone():
    bank = CodeBank()
    tree = GoalTree(bank, tau_assimilate=0.6)
    root = tree.add_goal("auth", torch.tensor([1.0, 0.0]))
    c_tree = bank.add("tree fn", torch.tensor([0.9, 0.1]))
    c_tree2 = bank.add("tree fn2", torch.tensor([0.9, -0.1]))
    tree.write_back(c_tree, torch.tensor([1.0, 0.0]))
    tree.write_back(c_tree2, torch.tensor([1.0, 0.0]))
    c_pre = bank.add("prefetch clone", torch.tensor([0.9, 0.12]))   # near-clone, not in tree
    prefetch = PrefetchBank(ttl_steps=1)
    prefetch.put(0, [c_pre])
    r = Retriever(tree, bank, prefetch, top_k=2, mmr_lambda=0.5)
    out = r.retrieve(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), step=0, k=2)
    assert out[0] == c_tree             # most relevant tree candidate first
    assert c_pre not in out             # prefetch clone demoted by diversity, no privilege


def test_retrieve_relevance_order_uncentered():
    bank = CodeBank()
    tree = GoalTree(bank, tau_assimilate=0.6, centered=False)
    root = tree.add_goal("auth", torch.tensor([1.0, 0.0]))
    c1 = bank.add("best", torch.tensor([0.95, 0.05]))
    c2 = bank.add("worse", torch.tensor([0.7, 0.7]))
    tree.write_back(c1, torch.tensor([1.0, 0.0]))
    tree.write_back(c2, torch.tensor([1.0, 0.0]))
    r = Retriever(tree, bank, PrefetchBank(), top_k=2, mmr_lambda=1.0, centered=False)
    out = r.retrieve(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), step=0)
    assert out[0] == c1                 # pure relevance picks the closer chunk


def test_retrieve_empty_tree():
    bank = CodeBank()
    tree = GoalTree(bank)
    tree.add_goal("app", torch.tensor([1.0, 0.0]))
    r = Retriever(tree, bank, PrefetchBank())
    assert r.retrieve(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), step=0) == []