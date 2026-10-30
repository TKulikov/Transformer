"""Pre-scripted questions, dialog sessions and expected answers.

Two evaluation modes:

* ``GENERAL_QA`` — standalone questions with reference answers; every
  strategy answers them independently and the answers are scored against
  the reference keywords.
* ``SESSIONS`` — multi-turn dialogues where facts are planted early and
  probed several turns later (needle-in-a-growing-context). This is what
  separates the cache policies: with a full history + classic KV cache the
  needle stays visible; a sparse cache with a small budget may evict it;
  the associative-memory strategy is designed to bring it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import keyword_score


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str                      # reference answer (shown for the record)
    expected: list[str]              # keywords the model answer must contain


@dataclass(frozen=True)
class Probe:
    """Question asked at turn ``after_turn`` (replacing the scripted turn there).

    Stress mode fills ``plant_turn`` (where the fact was introduced),
    ``expected`` (current truth) and ``stale`` (a superseded value — an
    answer containing it shows memory/context confusion).
    """

    after_turn: int
    question: str
    expected: list[str]
    plant_turn: int | None = None
    stale: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DialogSession:
    name: str
    opening: list[str] = field(default_factory=list)   # user turns that plant facts
    fillers: list[str] = field(default_factory=list)   # user turns that grow context
    probes: list[Probe] = field(default_factory=list)

    def user_turns(self) -> list[str]:
        return self.opening + self.fillers


GENERAL_QA = [
    QAItem("What does a KV cache store in a transformer decoder?",
           "Key and value tensors of past tokens per attention layer",
           ["key", "value"]),
    QAItem("What is the main drawback of the classic KV cache?",
           "Its memory grows linearly with sequence length",
           ["memory", "grow"]),
    QAItem("What does a sparse KV cache do?",
           "It evicts or selects only important/visible tokens from the cache",
           ["evict"]),
    QAItem("In attention, what do Q, K and V stand for?",
           "Query, Key, Value",
           ["query", "key", "value"]),
    QAItem("What is the softmax used for in attention?",
           "Turning scores into a probability distribution over tokens",
           ["probab"]),
    QAItem("What does positional encoding provide?",
           "Information about token order/position",
           ["position"]),
]

_FILLER = "Please explain a little more, continuing the same discussion in detail."

SESSIONS = [
    DialogSession(
        name="coding-buddy",
        opening=[
            "Remember this: my name is Alex and my pet cat is named Barsik.",
            "I am writing a Rust parser called Oxidum, version 0.4.2.",
        ],
        fillers=[_FILLER] * 8,
        probes=[
            Probe(after_turn=6, question="What is my name?", expected=["alex"]),
            Probe(after_turn=8, question="How is my cat named?", expected=["barsik"]),
            Probe(after_turn=9, question="What is the name of my Rust parser?", expected=["oxidum"]),
        ],
    ),
    DialogSession(
        name="trip-planner",
        opening=[
            "Our conference starts on March 14 in Helsinki.",
            "The hotel booking reference is HG-77120.",
        ],
        fillers=[_FILLER] * 8,
        probes=[
            Probe(after_turn=6, question="Which city hosts the conference?", expected=["helsinki"]),
            Probe(after_turn=7, question="What date does the conference start?", expected=["march"]),
            Probe(after_turn=9, question="What is the hotel booking reference?", expected=["77120"]),
        ],
    ),
]


def score_probe_answer(text: str, probe: Probe) -> float:
    """Keyword coverage of a generated answer against the probe's expectation."""
    return keyword_score(text, probe.expected)


def score_qa_answer(text: str, item: QAItem) -> float:
    return keyword_score(text, item.expected)
