"""Tests for the corpus generator: determinism, coverage, validators (offline)."""

from __future__ import annotations

import pytest

from tmem.corpus import (
    PRESETS,
    TOPICS,
    CorpusSpec,
    FactSpec,
    Persona,
    Topic,
    _combos,
    build_corpus_session,
    effective_turns,
    validate_session,
    validate_topics,
)


class TestPools:
    def test_topics_are_valid(self):
        assert validate_topics() == []

    def test_topic_coverage(self):
        assert len(TOPICS) == 6
        for topic in TOPICS:
            assert len(topic.facts) >= 8
            assert sum(1 for f in topic.facts if f.update) >= 2
            assert topic.twins and len(topic.holdout) >= 2 and len(topic.distractors) >= 3

    def test_validator_catches_keyword_collision(self):
        broken = Topic(
            name="broken", label="broken topic",
            facts=(
                FactSpec("a", "My cat is Anna.", ("anna",), "Who is the cat?", "Which cat?"),
                FactSpec("b", "My dog is Hanna.", ("hanna",), "Who is the dog?", "Which dog?"),
            ) + tuple(
                FactSpec(f"f{i}", f"Filler fact number {i} about zeta{i}.", (f"zeta{i}",),
                         f"Question {i}?", f"Paraphrase {i}?")
                for i in range(6)
            ),
            twins=(
                (FactSpec("t1", "Twin one is Alpha.", ("alpha",), "Q1?", "P1?"),
                 FactSpec("t2", "Twin two is Beta.", ("beta",), "Q2?", "P2?")),
            ),
            holdout=(
                FactSpec("h1", "Holdout one is Gamma.", ("gamma",), "Qh1?", "Ph1?"),
                FactSpec("h2", "Holdout two is Delta.", ("delta",), "Qh2?", "Ph2?"),
            ),
            distractors=("d1", "d2", "d3"),
            fillers=("f1", "f2", "f3"),
        )
        problems = validate_topics((broken,))
        assert any("colliding keywords" in p and "anna" in p for p in problems)

    def test_validator_catches_leak_into_filler(self):
        topic = next(t for t in TOPICS if t.name == "work")
        leaked = Topic(**{**topic.__dict__, "fillers": (*topic.fillers, "Margarita is a great name.")})
        assert any("leaks" in p for p in validate_topics((leaked,)))


class TestPresets:
    @pytest.mark.parametrize("preset,size", [("pilot", 12), ("main", 48), ("ablation", 16), ("sweep", 8)])
    def test_preset_sizes(self, preset, size):
        specs = PRESETS[preset]()
        assert len(specs) == size
        assert len({s.spec_id for s in specs}) == size

    def test_balanced_prefix_covers_all_axes(self):
        combos = _combos()[:12]
        assert len({t for t, _, _ in combos}) == 6       # all topics
        assert len({s for _, _, s in combos}) == 5       # all scenarios
        assert len({p for _, p, _ in combos}) == 4       # all personas

    def test_main_covers_lengths(self):
        lengths = {s.target_turns for s in PRESETS["main"]()}
        assert lengths == {100, 200}

    def test_spec_json_roundtrip(self):
        spec = PRESETS["pilot"]()[0]
        assert CorpusSpec.from_dict(spec.to_dict()) == spec


class TestGenerator:
    def test_deterministic(self):
        spec = PRESETS["main"]()[0]
        a, b = build_corpus_session(spec), build_corpus_session(spec)
        assert a.user_turns() == b.user_turns()
        assert [(p.after_turn, p.question, p.kind) for p in a.probes] == \
               [(p.after_turn, p.question, p.kind) for p in b.probes]

    def test_different_seed_differs(self):
        spec = PRESETS["main"]()[0]
        other = CorpusSpec(**{**spec.to_dict(), "seed": spec.seed + 1, "spec_id": spec.spec_id + "-x"})
        assert build_corpus_session(spec).user_turns() != build_corpus_session(other).user_turns()

    def test_all_pilot_sessions_valid(self):
        for spec in PRESETS["pilot"]():
            session = build_corpus_session(spec)
            assert len(session.user_turns()) == spec.target_turns
            assert validate_session(session) == [], spec.spec_id

    def test_lag_zones_all_covered_at_100_turns(self):
        spec = PRESETS["main"]()[0]
        session = build_corpus_session(spec)
        zones = set()
        for p in session.probes:
            if p.plant_turn is None:
                continue
            lag = p.after_turn - p.plant_turn
            zones.add("short" if lag <= 5 else "mid" if lag <= 20 else "long" if lag <= 50 else "vlong")
        assert zones == {"short", "mid", "long", "vlong"}

    def test_probe_kinds_present(self):
        spec = next(s for s in PRESETS["main"]() if s.scenario == "updates")
        session = build_corpus_session(spec)
        kinds = {p.kind for p in session.probes}
        assert {"direct", "paraphrase", "multihop", "temporal", "unanswerable"} <= kinds

    def test_temporal_probe_expects_old_value(self):
        spec = next(s for s in PRESETS["main"]() if s.scenario == "updates")
        session = build_corpus_session(spec)
        temporal = [p for p in session.probes if p.kind == "temporal"]
        assert temporal
        turns = " ".join(effective_turns(session)).lower()
        for p in temporal:
            assert any(kw.lower() in turns for kw in p.expected)   # old value planted
            assert any(kw.lower() in turns for kw in p.stale)      # new value planted too
            assert set(p.expected).isdisjoint(p.stale)

    def test_multihop_has_two_expected_parts(self):
        spec = PRESETS["main"]()[0]
        session = build_corpus_session(spec)
        multihop = [p for p in session.probes if p.kind == "multihop"]
        assert multihop and all(len(p.expected) >= 2 for p in multihop)

    def test_unanswerable_value_absent_from_dialogue(self):
        for spec in PRESETS["pilot"]():
            session = build_corpus_session(spec)
            turns = " ".join(effective_turns(session)).lower()
            for p in session.probes:
                if p.kind == "unanswerable":
                    assert p.forbidden
                    for kw in p.forbidden:
                        assert kw.lower() not in turns, (spec.spec_id, kw)

    def test_adversarial_twins_carry_stale(self):
        spec = next(s for s in PRESETS["main"]() if s.scenario == "adversarial")
        session = build_corpus_session(spec)
        with_stale = [p for p in session.probes if p.stale]
        assert with_stale, "adversarial session must mark twin values as stale"

    @pytest.mark.parametrize("persona", ["terse", "chatty", "sloppy", "formal"])
    def test_persona_keeps_fact_values_verbatim(self, persona):
        spec = CorpusSpec(spec_id=f"persona-{persona}", topic="travel", persona=persona,
                          scenario="baseline", target_turns=60, seed=42, n_facts=8,
                          probes_per_zone=2, n_unanswerable=1)
        session = build_corpus_session(spec)
        assert validate_session(session) == []

    def test_multifact_packs_two_facts_per_turn(self):
        spec = next(s for s in PRESETS["main"]() if s.scenario == "multifact")
        session = build_corpus_session(spec)
        combined = [t for t in session.user_turns() if " also, " in t.lower()]
        assert combined, "multifact scenario must combine statements"
        assert validate_session(session) == []
