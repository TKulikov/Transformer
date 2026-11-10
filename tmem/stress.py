"""Stress evaluation of the associative ("semantic") memory.

Builds long dialogues that plant many facts across the context (early /
middle / late zones), *update* some facts mid-dialogue (old value must be
superseded), ask paraphrased probes far from their plants, and include an
adversarial session with near-duplicate facts (tests memory merging and
confusion). For every strategy we measure recall by zone and lag, stale
(superseded-value) rate, consistency against the classic-cache reference
answer, memory internals (stores/merges/hits/evictions) and all efficiency
costs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dialogue import DialogueRun, Strategy
from .kvcache import KVConfig
from .qa import DialogSession, Probe

FILLER = "Tell me more about that, expanding on the same subject a bit."


@dataclass(frozen=True)
class Fact:
    key: str
    plant: str
    expected: tuple[str, ...]
    question: str
    paraphrase: str
    update: str | None = None
    new_expected: tuple[str, ...] = ()
    stale_topic: str | None = None


FACTS = [
    Fact("name", "Remember this: my name is Alex.", ("alex",), "What is my name?", "How should I address me again?"),
    Fact("cat", "My pet cat is named Barsik.", ("barsik",), "How is my cat named?", "Tell me the name of my feline pet.",
         update="Actually, my cat has been renamed to Simba.", new_expected=("simba",)),
    Fact("parser", "I am writing a Rust parser called Oxidum.", ("oxidum",),
         "What is the name of my Rust parser?", "Which project am I building in Rust?"),
    Fact("version", "Its current version is 0.4.2.", ("0.4.2",), "What version is my parser?", "Which release number does it have now?"),
    Fact("city", "Our conference takes place in Helsinki.", ("helsinki",),
         "Which city hosts the conference?", "Where is the conference held?",
         update="Correction: the conference moved to Tallinn.", new_expected=("tallinn",)),
    Fact("date", "The conference starts on March 14.", ("march",), "What month does the conference start?", "When does the conference begin?"),
    Fact("booking", "My hotel booking reference is HG-77120.", ("77120",),
         "What is my hotel booking reference?", "Which reservation number do I have?"),
    Fact("color", "My favorite color is teal.", ("teal",), "What is my favorite color?", "Which color do I like most?"),
    Fact("book", "I am currently reading the novel Dune.", ("dune",), "Which book am I reading?", "What novel is on my nightstand?"),
    Fact("serial", "My laptop serial number is ZX9-4410.", ("4410",), "What is my laptop serial?", "Which serial does my computer have?"),
    Fact("car", "My car is a blue Volvo.", ("volvo",), "What car do I have?", "Which brand is my car?"),
    Fact("sister", "My sister works as a marine biologist.", ("marine",), "What does my sister do?", "Which field does my sister work in?"),
    Fact("guitar", "I bought a Fender Stratocaster guitar.", ("stratocaster",), "Which guitar did I buy?", "What instrument did I get?"),
    Fact("diet", "Since February I am gluten free.", ("gluten",), "What is my diet?", "What do I avoid eating?"),
    Fact("trek", "Next summer I plan an Everest base camp trek.", ("everest",), "What trek am I planning?", "Which mountain expedition is in my plans?"),
    Fact("language", "I am learning Finnish at B1 level.", ("finnish",), "Which language am I learning?", "What is my study language?"),
]

ADVERSARIAL_FACTS = [
    Fact("parser1", "My first project is a Rust parser called Oxidum.", ("oxidum",),
         "What is the name of my first project?", "Which one is the parser?", stale_topic="oxidus"),
    Fact("parser2", "My second project is a Rust compiler called Oxidus.", ("oxidus",),
         "What is the name of my second project?", "Which one is the compiler?"),
    Fact("cat", "My cat Barsik is gray.", ("barsik",), "How is my cat named?", "What color is my pet Barsik?"),
    Fact("dog", "My dog Barks is brown.", ("barks",), "How is my dog named?", "What color is my pet Barks?"),
    Fact("conf", "The main conference is on March 14 in Helsinki.", ("helsinki",),
         "Which city hosts the main conference?", "Where is the main conference held?"),
    Fact("webinar", "The webinar is on March 21 in Tampere.", ("tampere",),
         "Which city hosts the webinar?", "Where is the webinar held?"),
]


def _zones(n: int, total: int) -> str:
    return "early" if n < total / 3 else ("middle" if n < 2 * total / 3 else "late")


def build_long_session(n_facts: int = 9, seed: int = 0, paraphrase_every: int = 2) -> DialogSession:
    """Facts spread across the dialogue, updates mid-way, probes at the end."""
    rng = random.Random(seed)
    facts = FACTS[:n_facts] if n_facts <= len(FACTS) else rng.sample(FACTS, n_facts)
    turns: list[str] = ["Let's talk about my life, projects and plans."]
    plants: dict[str, int] = {}
    expected: dict[str, tuple[str, ...]] = {}
    stale: dict[str, list[str]] = {}

    for f in facts:
        turns.append(f.plant)
        plants[f.key] = len(turns) - 1
        expected[f.key] = f.expected
        turns.append(FILLER)

    for f in facts:  # updates replace plants later in the dialogue
        if f.update:
            turns.append(f.update)
            plants[f.key] = len(turns) - 1
            stale[f.key] = list(f.expected)
            expected[f.key] = f.new_expected
            turns.append(FILLER)

    probes: list[Probe] = []
    order = facts[:]
    rng.shuffle(order)
    for i, f in enumerate(order):
        is_paraphrase = i % paraphrase_every == paraphrase_every - 1
        question = f.paraphrase if is_paraphrase else f.question
        probes.append(
            Probe(
                after_turn=len(turns),
                question=question,
                expected=list(expected[f.key]),
                plant_turn=plants[f.key],
                stale=stale.get(f.key, []),
                paraphrase=is_paraphrase,
            )
        )
        turns.append(question)
        if i % 3 == 2:
            turns.append(FILLER)
    return DialogSession(name=f"stress-long-{n_facts}f", opening=turns, fillers=[], probes=probes)


def build_adversarial_session() -> DialogSession:
    """Similar-sounding facts; probes must not confuse them (stale = the twin value)."""
    turns: list[str] = ["I will describe several similar things; keep them apart."]
    plants, probes, expected, stale = {}, [], {}, {}
    for f in ADVERSARIAL_FACTS:
        # stale = the "twin" value the probe must not mix up with the answer
        turns.append(f.plant)
        plants[f.key] = len(turns) - 1
        expected[f.key] = f.expected
        stale[f.key] = [f.stale_topic] if f.stale_topic else []
        turns.append(FILLER)
    for i, f in enumerate(ADVERSARIAL_FACTS):
        question = f.question if i % 2 == 0 else f.paraphrase
        probes.append(
            Probe(
                after_turn=len(turns),
                question=question,
                expected=list(expected[f.key]),
                plant_turn=plants[f.key],
                stale=stale[f.key],
                paraphrase=bool(i % 2),
            )
        )
        turns.append(question)
    return DialogSession(name="stress-adversarial", opening=turns, fillers=[], probes=probes)


def stress_sessions() -> list[DialogSession]:
    return [build_long_session(), build_adversarial_session()]


# ----------------------------------------------------------------- battery100
def build_battery_100(target_turns: int = 105, seed: int = 0) -> DialogSession:
    """>100-turn dialogue: 16 facts across all zones, 2 updates, 24 probes."""
    rng = random.Random(seed)
    facts = FACTS[:16]
    turns: list[str] = ["Let's have a long talk about my life, work and plans."]
    plants, expected, stale = {}, {}, {}

    slots = [max(1, round(1 + i * (target_turns * 0.55) / len(facts))) for i in range(len(facts))]
    for f, at in zip(facts, slots):
        while len(turns) <= at:
            turns.append(FILLER)
        turns[at] = f.plant
        plants[f.key] = at
        expected[f.key] = f.expected

    updaters = [f for f in facts if f.update]
    updates = list(zip(updaters, [round(target_turns * 0.5), round(target_turns * 0.62)]))
    for f, at in updates:
        at = max(at, len(turns))
        while len(turns) <= at:
            turns.append(FILLER)
        turns[at] = f.update
        plants[f.key] = at
        stale[f.key] = list(expected[f.key])
        expected[f.key] = f.new_expected

    probes: list[Probe] = []
    order = facts[:] + rng.sample(facts, min(8, len(facts)))  # extra paraphrased repeats
    rng.shuffle(order)
    for i, f in enumerate(order):
        at = max(target_turns - 22 + i, len(turns))  # probes live in the final third
        while len(turns) <= at:
            turns.append(FILLER)
        question = f.paraphrase if i % 2 else f.question
        turns[at] = question
        probes.append(Probe(after_turn=at, question=question, expected=list(expected[f.key]),
                            plant_turn=plants[f.key], stale=stale.get(f.key, []), paraphrase=bool(i % 2)))
    while len(turns) < target_turns:
        turns.append(FILLER)
    return DialogSession(name=f"battery-{len(turns)}t-{len(probes)}q", opening=turns, fillers=[], probes=probes)


def battery_strategies(budget: int = 128) -> list[Strategy]:
    """Return archival conditions and the comparable training-free controls."""
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy(f"sparse_sliding_{budget}", context="full", kv=KVConfig.sparse(budget, "sliding", 4, 64)),
        Strategy(f"sparse_topk_{budget}", context="full", kv=KVConfig.sparse(budget, "topk", 4, 64)),
        Strategy(f"snapkv_{budget}", context="full",
                 kv=KVConfig.sparse(budget, "snapkv", observation=64, kernel_size=7)),
        Strategy(f"pyramidkv_{budget}", context="full",
                 kv=KVConfig.sparse(budget, "pyramidkv", observation=8, pyramid_beta=20)),
        Strategy("rag_text_only", context="window", window=5, rag=True, rag_lambda=0.0, rag_centered=False),
        Strategy("window2_only", context="window", window=2),
        Strategy("window5_only", context="window", window=5),
        Strategy("v1_memory", context="window", window=2, memory=True, memory_window_crit=0.5),
        Strategy("v2_memory_ema", context="window", window=2, memory=True, memory_window_crit=0.5,
                 memory_crit=0.95, memory_merge_k=0.25),
        Strategy("v2_rag_plain", context="window", window=2, rag=True, rag_lambda=0.0),
        Strategy("v2_rag_gist", context="window", window=2, rag=True, rag_lambda=0.5, memory=True,
                 memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25),
        Strategy("v2_full", context="window", window=5, rag=True, rag_lambda=0.5, memory=True,
                 memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25),
        Strategy("v3_rag_centroid", context="window", window=2, rag=True, rag_lambda=0.0, rag_centered=True),
        Strategy("v3_centroid_plus_gist", context="window", window=2, rag=True, rag_lambda=0.5, rag_centered=True,
                 memory=True, memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25),
        Strategy("v4_kv_mid", context="window", window=2, memory_kv=True, memory_window_crit=0.5,
                 memory_crit=0.95, memory_merge_k=0.25, kv_slots=32),
        Strategy("v4_rag_kv", context="window", window=5, memory_kv=True, rag=True, rag_lambda=0.0,
                 rag_centered=True, memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25, kv_slots=32),
    ]


def stress_strategies(budget: int = 128) -> list[Strategy]:
    """Condition grid: references, KV pressure, v1 memory, v2 (EMA + RAG bank)."""
    mem = dict(memory=True, context="window", window=2, memory_window_crit=0.5)
    mem_v2 = dict(memory=True, context="window", memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    return [
        Strategy("classic_kv_full", context="full"),                                        # reference (gold)
        Strategy(f"sparse_sliding_{budget}", context="full", kv=KVConfig.sparse(budget, "sliding", 4, 64)),
        Strategy("sparse_sliding_32", context="full", kv=KVConfig.sparse(32, "sliding", 4, 16)),
        Strategy(f"sparse_topk_{budget}", context="full", kv=KVConfig.sparse(budget, "topk", 4, 64)),
        Strategy("window2_only", context="window", window=2),                                # short buffer, no layers
        Strategy("window5_only", context="window", window=5),
        Strategy("v1_memory", **mem),                                                        # notebook defaults
        Strategy("v2_memory_ema", **mem_v2, window=2),                                       # EMA k=.25, crit=.95
        Strategy("v2_rag_plain", context="window", window=2, rag=True, rag_lambda=0.0),      # bank without gist
        Strategy("v2_rag_gist", **mem_v2, window=2, rag=True, rag_lambda=0.5),               # gist-conditioned bank
        Strategy("v2_full", **mem_v2, window=5, rag=True, rag_lambda=0.5),                  # buffer5 + both layers
    ]


# ---------------------------------------------------------------- analysis
def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def analyze_session(runs: dict[str, DialogueRun], reference: str = "classic_kv_full") -> dict:
    """Compute conditional recall, update errors, latency distribution and costs."""
    ref_run = runs.get(reference)
    total_turns = len(ref_run.turns) if ref_run else max((len(r.turns) for r in runs.values()), default=1)
    ref_answers = {d["question"]: d["answer"] for d in ref_run.summary()["probe_details"]} if ref_run else {}
    out = {}
    for name, run in runs.items():
        details = run.summary()["probe_details"]
        cons = []
        lag_scores = {"0-5": [], "6-12": [], "13-20": [], "21+": []}
        zone_acc: dict[str, list[float]] = {}
        kind_acc: dict[str, list[float]] = {}
        for d in details:
            score = d["score"] or 0.0
            zone = _zones(d["plant_turn"] or 0, total_turns)
            zone_acc.setdefault(zone, []).append(score)
            kind_acc.setdefault(d.get("kind") or "direct", []).append(score)
            lag = d["lag"]
            if lag is not None:
                bucket = "0-5" if lag <= 5 else ("6-12" if lag <= 12 else ("13-20" if lag <= 20 else "21+"))
                lag_scores[bucket].append(score)
            if d["question"] in ref_answers:
                cons.append(_jaccard(d["answer"], ref_answers[d["question"]]))
        n = max(len(details), 1)
        s = run.summary()
        direct = [d["score"] or 0.0 for d in details if not d.get("paraphrase")]
        paraphrase = [d["score"] or 0.0 for d in details if d.get("paraphrase")]
        updated = [d for d in details if d["stale_score"] is not None]
        latencies = sorted(t.decode.latency_s for t in run.turns)
        mid = len(latencies) // 2
        median = (latencies[mid] if len(latencies) % 2 else (latencies[mid - 1] + latencies[mid]) / 2) if latencies else 0.0
        p90 = latencies[max(0, -(-9 * len(latencies) // 10) - 1)] if latencies else 0.0
        out[name] = {
            "recall_mean": round(sum(x["score"] or 0.0 for x in details) / n, 3),
            "recall_direct": round(sum(direct) / len(direct), 3) if direct else None,
            "recall_paraphrase": round(sum(paraphrase) / len(paraphrase), 3) if paraphrase else None,
            "recall_by_zone": {z: round(sum(v) / len(v), 3) for z, v in zone_acc.items()},
            "recall_by_kind": {k: round(sum(v) / len(v), 3) for k, v in kind_acc.items()},
            "recall_by_lag": {k: round(sum(v) / len(v), 3) if v else None for k, v in lag_scores.items()},
            "stale_rate": round(sum(bool(d["stale_score"]) for d in details) / n, 3),
            "stale_updated_rate": round(sum(bool(d["stale_score"]) for d in updated) / len(updated), 3) if updated else None,
            "consistency_vs_reference": round(sum(cons) / len(cons), 3) if cons else None,
            "tone_mean": s.get("tone_mean"),
            "rep_mean": s["rep_mean"],
            "prompt_tokens": s["prompt_tokens"],
            "generated_tokens": s["generated_tokens"],
            "embed_tokens": s.get("embed_tokens", 0),
            "cache_slots": s["cache_slots"],
            "attention_flops_proxy": s["attention_flops_proxy"],
            "max_cache_len": s["max_cache_len"],
            "mean_turn_latency_s": s["mean_turn_latency_s"],
            "median_turn_latency_s": round(median, 4),
            "p90_turn_latency_s": round(p90, 4),
            "peak_mem_mb": round(max((t.decode.peak_mem_mb or 0.0 for t in run.turns), default=0.0), 2) or None,
            "memory_stats": s.get("memory_stats"),
            "rag_stats": s.get("rag_stats"),
            "probe_details": details,
        }
    return out
