"""Tests for the v2 mechanics: EMA merge, RAG bank upsert/gist-retrieval, tone."""

from __future__ import annotations

import pytest
import torch

from tmem.memory import Memory
from tmem.semantic import RagBank, repetition_rate


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


class TestEmaMerge:
    def make(self, k):
        m = Memory(4, size=4, merge_k=k)
        return m

    def test_ema_bounds_norm(self):
        m = self.make(0.25)
        v = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        m.update(v.unsqueeze(0), crit=0.95, max_to_new=0.4, window_crit=-1e9)
        idx = int(torch.nonzero(m.occupied)[0])
        for _ in range(20):  # repeated merges of the same direction
            m.update(v.unsqueeze(0), crit=0.95, max_to_new=0.4, window_crit=-1e9)
        assert float(m.context[0, idx].norm()) < 20.0  # stable, no (1+k)^m blowup
        assert m.stats["merges"] >= 15

    def test_ema_k1_is_replacement(self):
        m = self.make(1.0)
        a = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 1.0, 1e-3, 0.0]])  # close enough to merge
        m.update(a.unsqueeze(0), window_crit=-1e9)
        m.update(b.unsqueeze(0), crit=0.95, window_crit=-1e9)
        idx = int(torch.nonzero(m.occupied)[0])
        assert torch.allclose(m.context[0, idx], b[0], atol=1e-6)

    def test_v1_mean_is_default(self):
        m = Memory(4, size=4)
        assert m.merge_k is None
        assert torch.allclose(m.consume(torch.ones(4), torch.zeros(4)), torch.full((4,), 0.5))

    def test_bad_k(self):
        with pytest.raises(ValueError):
            Memory(4, merge_k=0.0)
        with pytest.raises(ValueError):
            Memory(4, merge_k=1.5)


class TestRagBank:
    def fresh(self):
        return RagBank(4, crit=0.95, capacity=8, centered=False)

    def test_add_and_retrieve_exact(self):
        bank = self.fresh()
        bank.add("cat is Barsik", unit(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        bank.add("city Helsinki", unit(torch.tensor([0.0, 1.0, 0.0, 0.0])))
        got = bank.retrieve(torch.tensor([0.1, 0.95, 0.0, 0.0]), k=1)
        assert got[0][0] == "city Helsinki"
        assert bank.stats["adds"] == 2

    def test_upsert_overwrites_stale_fact(self):
        bank = self.fresh()
        bank.add("My cat is Barsik", unit(torch.tensor([1.0, 0.1, 0.0, 0.0])))
        bank.add("My cat is Simba", unit(torch.tensor([1.0, 0.05, 0.0, 0.0])))  # sim > .95
        assert bank.stats["upserts"] == 1 and bank.stats["adds"] == 1
        assert bank.texts == ["My cat is Simba"]

    def test_dissimilar_both_kept(self):
        bank = self.fresh()
        bank.add("a", unit(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        bank.add("b", unit(torch.tensor([0.0, 1.0, 0.0, 0.0])))
        assert bank.texts == ["a", "b"]

    def test_gist_shifts_ranking(self):
        bank = self.fresh()
        bank.add("parser fact", unit(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        bank.add("topic cat", unit(torch.tensor([0.0, 1.0, 0.05, 0.0])))
        q = unit(torch.tensor([0.9, 0.25, 0.0, 0.0]))
        gist = unit(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        top_plain = bank.retrieve(q, k=1, lam=0.0)[0][0]
        top_gist = bank.retrieve(q, k=1, gist=gist, lam=1.0)[0][0]
        assert top_plain == "parser fact"
        assert top_gist == "topic cat"  # gist-conditioning surfaces topic-relevant fact

    def test_capacity_evicts_oldest(self):
        bank = RagBank(2, crit=0.95, capacity=2)
        bank.add("a", torch.tensor([1.0, 0.0]))
        bank.add("b", torch.tensor([0.0, 1.0]))
        bank.add("c", torch.tensor([0.7, 0.7]))
        assert "a" not in bank.texts and bank.stats["evictions"] == 1

    def test_empty_bank(self):
        assert self.fresh().retrieve(torch.ones(4), k=3) == []

    def test_questions_are_not_stored(self):
        """A probe question must never upsert-overwrite the fact it asks."""
        bank = self.fresh()
        fact = unit(torch.tensor([1.0, 0.1, 0.0, 0.0]))
        bank.add("My cat is Barsik", fact)
        bank.add("How is my cat named?", unit(torch.tensor([1.0, 0.15, 0.0, 0.0])))
        assert bank.texts == ["My cat is Barsik"]
        assert bank.stats["adds"] == 1 and bank.stats["upserts"] == 0


class TestCenteredBank:
    """v3: anisotropy cure — shared-mean component must not collapse the bank."""

    def biased(self, i: int, d: int = 8) -> torch.Tensor:
        """Embeddings with a heavy shared bias (mutual cos > 0.95) + small
        orthogonal signals — exactly the real-model pathology."""
        base = torch.ones(d) * 3.0
        sig = torch.zeros(d)
        sig[i] = 0.9
        return unit(base + sig)

    def test_uncentered_collapses(self):
        bank = RagBank(8, crit=0.95, capacity=16, centered=False)
        for i in range(4):
            bank.add(f"fact {i}", self.biased(i))
        assert bank.stats["upserts"] >= 2  # facts ate each other

    def test_centered_keeps_and_finds(self):
        bank = RagBank(8, crit=0.95, capacity=16, centered=True)
        for i in range(4):
            bank.add(f"fact {i}", self.biased(i))
        assert bank.stats["upserts"] == 0 and bank.stats["adds"] == 4
        hit = bank.retrieve(self.biased(2), k=1)[0][0]
        assert hit == "fact 2"

    def test_centered_still_upserts_true_duplicate(self):
        bank = RagBank(8, crit=0.95, capacity=16, centered=True)
        bank.add("cat is Barsik", self.biased(1))
        bank.add("cat is Barsik.", self.biased(1) * 1.001)  # same utterance
        assert bank.stats["upserts"] == 1

    def test_mean_tracks_topic(self):
        bank = RagBank(8, centered=True)
        bank.add("a", self.biased(0))
        assert float(bank.mean.norm()) > 0  # centroid starts following the dialogue


class TestToneMetrics:
    def test_repetition_rate(self):
        assert repetition_rate("I I I I I I") == 0.75  # 4 trigrams, all identical
        assert repetition_rate("completely varied word sequence here") == 0.0
        assert repetition_rate("short") == 0.0

    def test_tone_alignment_on_tiny(self, causal_bundle):
        from tmem.semantic import embed_utterance, tone_alignment

        bank = RagBank(causal_bundle.hidden_size, crit=0.99, device=causal_bundle.device)
        t1 = embed_utterance(causal_bundle, "hello there friend")
        t2 = embed_utterance(causal_bundle, "another utterance entirely")
        bank.add("hello there friend", t1)
        bank.add("another utterance entirely", t2)
        score = tone_alignment(causal_bundle, t1, bank)
        assert score == pytest.approx(1.0, abs=0.02)  # answer identical to a stored span
        assert bank.retrieve(t1, k=1)[0][0] == "hello there friend"


class TestStrategyV2Plumbing:
    def test_rag_run_produces_tone_and_notes(self, causal_bundle):
        from tmem.dialogue import Strategy, run_session
        from tmem.qa import SESSIONS

        run = run_session(
            causal_bundle, SESSIONS[0],
            Strategy("v2", context="window", window=2, rag=True, rag_lambda=0.5,
                     memory=True, memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25),
            max_new_tokens=3, seed=0,
        )
        s = run.summary()
        assert s["tone_mean"] is not None
        assert run.bank.stats["adds"] + run.bank.stats["upserts"] > 0
        assert s["embed_tokens"] > 0
        assert any(t.injected is not None and t.injected for t in run.turns)  # bank actually filled prompts
        assert isinstance(s["attention_flops_proxy"], int)


class TestBankLayerEmbedding:
    def test_bank_layer_indices_fallback(self, causal_bundle):
        # GPT-2 has no layer_types → model middle, matching kvinject fallback
        idxs = causal_bundle.bank_layer_indices()
        assert idxs == [1]  # 2-layer tiny GPT-2

    def test_embed_bank_layer(self, causal_bundle):
        from tmem.semantic import embed_utterance

        v_bank = embed_utterance(causal_bundle, "sprint ends thursday", layer="bank")
        v_last = embed_utterance(causal_bundle, "sprint ends thursday", layer="last")
        assert v_bank.shape == v_last.shape == (causal_bundle.hidden_size,)
        assert not torch.allclose(v_bank, v_last)
        cos = torch.nn.functional.cosine_similarity(v_bank, v_last, dim=0)
        assert -1.0 <= float(cos) <= 1.0

    def test_embed_bank_layer_unknown_raises(self, causal_bundle):
        from tmem.semantic import embed_utterance

        with pytest.raises(ValueError, match="unknown embed layer"):
            embed_utterance(causal_bundle, "x", layer="nope")


class TestBankSlotKeys:
    def test_rag_bank_key_run(self, causal_bundle):
        """rag_key_source='bank': keys are latent-bank slot vectors."""
        from tmem.dialogue import Strategy, run_session
        from tmem.qa import SESSIONS

        run = run_session(
            causal_bundle, SESSIONS[0],
            Strategy("bk", context="window", window=2, rag=True, rag_key_source="bank"),
            max_new_tokens=3, seed=0,
        )
        assert run.corr is not None                    # bank created even without injection
        assert run.corr.stats["update_calls"] > 0      # utterances were poured into it
        s = run.summary()
        assert run.bank.stats["adds"] + run.bank.stats["upserts"] > 0
        assert any(t.injected for t in run.turns)      # slot-keyed notes reached prompts

    def test_bank_slot_key_differs_from_raw(self, causal_bundle):
        """The slot key is a bank vector, not the raw mean activation."""
        from tmem.dialogue import _bank_slot_key
        from tmem.memory import CorrectionLayer
        from tmem.semantic import embed_utterance

        corr = CorrectionLayer(causal_bundle.hidden_size, size=32, window_crit=0.0)
        key, top1, n = _bank_slot_key(causal_bundle, corr, "remember Barsik the cat")
        raw = embed_utterance(causal_bundle, "remember Barsik the cat", layer="bank")
        assert key.shape == raw.shape == (causal_bundle.hidden_size,)
        assert top1 is not None and top1.shape == raw.shape
        assert n > 0
        assert corr.stats["update_calls"] == 1         # pour happened exactly once


class TestAnchoredBank:
    def test_cluster_gate_ranks_within_cluster(self):
        """Facts anchored to the query's cluster win; cross-cluster loses."""
        d = 8
        bank = RagBank(d, crit=0.99, anchored=True, anchor_gate=0.9)
        a1 = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])       # cluster A anchor
        a2 = torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0])     # cluster B anchor
        # same anchor, distinct deviations (facts inside one cluster)
        bank.add("fact A1", torch.tensor([1.0, 0, 0.5, 0, 0, 0, 0, 0]), anchor=a1)
        bank.add("fact A2", torch.tensor([1.0, 0, 0, 0.5, 0, 0, 0, 0]), anchor=a1)
        bank.add("fact B1", torch.tensor([0.0, 1.0, 0, 0, 0.9, 0, 0, 0]), anchor=a2)

        # query in cluster A with deviation toward A1's direction
        q = torch.tensor([1.0, 0, 0.5, 0.01, 0, 0, 0, 0])
        out = bank.retrieve(q, k=2, anchor=a1)
        texts = [t for t, _, _ in out]
        assert "fact B1" not in texts, "cluster B must be gated out"
        assert texts[0] == "fact A1"

    def test_no_eligible_cluster_falls_back(self):
        d = 8
        bank = RagBank(d, crit=0.99, anchored=True, anchor_gate=0.9)
        a1 = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        bank.add("fact A1", torch.tensor([1.0, 0, 0.5, 0, 0, 0, 0, 0]), anchor=a1)
        # query from an unrelated cluster: gate excludes, fallback returns something
        q = torch.tensor([0.0, 1.0, 0, 0, 1.0, 0, 0, 0])
        out = bank.retrieve(q, k=1, anchor=torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0.001]))
        assert len(out) == 1  # fallback: globally ranked, not an empty result

    def test_anchored_ignores_running_mean(self):
        """Anchored mode centres by the anchor, not by the running centroid."""
        d = 4
        bank = RagBank(d, crit=0.99, centered=True, anchored=True)
        a = torch.tensor([1.0, 0, 0, 0])
        v = torch.tensor([1.0, 0.3, 0, 0])
        bank.add("t", v, anchor=a)
        stored = bank.embeds[0]
        assert torch.allclose(stored, (v - a) / (v - a).norm(), atol=1e-5)
