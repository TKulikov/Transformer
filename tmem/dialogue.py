"""Dialogue-level experiments: context strategy x memory x KV-cache policy.

Each :class:`Strategy` combines

* how the prompt context is assembled (full history vs sliding window),
* whether the associative :class:`~tmem.memory.CorrectionLayer` is injected,
* which KV-cache policy decoding uses (none / classic / sparse).

``run_session`` replays a scripted :class:`~tmem.qa.DialogSession` under a
strategy and records per-turn token usage, cache sizes, latency and probe
answers scored against the expected keywords.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk
from .generate import DecodeResult, decode_causal, decode_encdec
from .kvcache import KVConfig
from .metrics import keyword_score
from .qa import DialogSession, Probe, score_probe_answer
from .semantic import RagBank, embed_utterance, gist_of, repetition_rate, tone_alignment


@dataclass(frozen=True)
class Strategy:
    name: str
    context: str = "full"                 # "full" | "window"
    window: int = 3                       # recent *turns* kept when context == "window"
    memory: bool = False                  # inject CorrectionLayer
    memory_beta: float = 0.25             # additive weight (decoder flavour)
    memory_window_crit: float = 10.0      # activity gate of the memory
    memory_size: int = 512                # memory slots
    memory_retrieve: int = 256            # slots retrieved per call
    memory_crit: float = 0.8              # merge threshold (v2: 0.95)
    memory_merge_k: float | None = None   # EMA merge strength (None = v1 mean)
    memory_decay: bool = True             # TTL-динамика (False = статический банк, абляция живой памяти)
    rag: bool = False                     # text-preserving fact bank in prompt
    rag_crit: float = 0.95                # upsert dedup threshold
    rag_k: int = 4                        # facts injected per turn
    rag_lambda: float = 0.5               # gist-conditioned retrieval strength
    rag_centered: bool = False            # v3: deviation-from-topic scoring
    rag_layer: str = "last"               # слой RAG-эмбеддинга: last | mid (латентный) | mean
    memory_kv: bool = False               # v4: activated facts enter mid-layer attention as K/V
    kv_slots: int = 32                    # max memory keys/values per attention layer
    gen_chunks: bool = False              # chunk-memory: store generation chunks instead of user facts
    chunk_embed: str = "span"             # mean | span | attn | late (contextualised chunk embedding)
    chunk_key: str = "content"            # content (block layer activations) | request (legacy)
    chunk_key_layers: str = "mean"        # last | mid | mean — which layers form the content key
    chunk_score: str = "key"              # key (request→block) | value (request→chunk) | maxsim
    chunk_tokens: int = 128               # max tokens per stored chunk
    chunk_k: int = 4                      # chunks injected per turn
    chunk_upsert: bool = False            # cosine upsert (dangerous for code — off by default)
    chunk_centered: bool = True           # centre request keys by the running centroid
    cache: str = "use"                    # "use" | "none"
    kv: KVConfig = KVConfig.classic()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "context": self.context,
            "window": self.window,
            "memory": self.memory,
            "memory_beta": self.memory_beta,
            "memory_window_crit": self.memory_window_crit,
            "memory_size": self.memory_size,
            "memory_retrieve": self.memory_retrieve,
            "memory_crit": self.memory_crit,
            "memory_merge_k": self.memory_merge_k,
            "memory_decay": self.memory_decay,
            "rag": self.rag,
            "rag_crit": self.rag_crit,
            "rag_k": self.rag_k,
                "rag_lambda": self.rag_lambda,
            "rag_centered": self.rag_centered,
            "rag_layer": self.rag_layer,
            "memory_kv": self.memory_kv,
            "kv_slots": self.kv_slots,
            "gen_chunks": self.gen_chunks,
            "chunk_embed": self.chunk_embed,
            "chunk_key": self.chunk_key,
            "chunk_key_layers": self.chunk_key_layers,
            "chunk_score": self.chunk_score,
            "chunk_tokens": self.chunk_tokens,
            "chunk_k": self.chunk_k,
            "chunk_upsert": self.chunk_upsert,
            "chunk_centered": self.chunk_centered,
            "cache": self.cache,
            "kv": {
                "mode": self.kv.mode,
                "strategy": self.kv.strategy,
                "budget": self.kv.budget,
                "sink": self.kv.sink,
                "recent": self.kv.recent,
                "observation": self.kv.observation,
                "kernel_size": self.kv.kernel_size,
                "pyramid_beta": self.kv.pyramid_beta,
                "chunk_size": self.kv.chunk_size,
            },
        }


def default_strategies(budget: int = 128, sink: int = 4, recent: int = 64) -> list[Strategy]:
    """The full comparison grid used by the benchmark CLI."""
    return [
        Strategy("no_cache_full", context="full", cache="none"),
        Strategy("classic_kv_full", context="full", cache="use", kv=KVConfig.classic()),
        Strategy("sparse_sliding", context="full", cache="use", kv=KVConfig.sparse(budget, "sliding", sink, recent)),
        Strategy("sparse_topk", context="full", cache="use", kv=KVConfig.sparse(budget, "topk", sink, recent)),
        Strategy("window_only", context="window", window=3, cache="use", kv=KVConfig.classic()),
        Strategy("window_hybrid_memory", context="window", window=3, memory=True, cache="use", kv=KVConfig.classic()),
    ]


@dataclass
class TurnResult:
    user_text: str
    answer_text: str
    prompt_tokens: int
    decode: DecodeResult
    probe: Probe | None = None
    probe_score: float | None = None
    stale_score: float | None = None  # keyword hit on a superseded value (confusion)
    tone_score: float | None = None   # max cosine(answer, dialogue bank) — topic hold
    rep_rate: float = 0.0             # repeated word n-grams (degeneration gauge)
    injected: list[str] | None = None  # RAG notes placed into the prompt


@dataclass
class DialogueRun:
    strategy: Strategy
    turns: list[TurnResult] = field(default_factory=list)
    corr: "object | None" = None  # CorrectionLayer used by memory strategies (None otherwise)
    bank: "object | None" = None   # RagBank for rag strategies (None otherwise)
    chunk_bank: "object | None" = None  # ChunkBank for gen_chunks strategies (None otherwise)
    embed_tokens: int = 0          # extra tokens spent on bank/query embedding
    kv_injected: int = 0           # v4: memory slots injected into attention (total)
    kv_steps: int = 0              # v4: attention forwards that saw memory

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_generated_tokens(self) -> int:
        return sum(t.decode.steps for t in self.turns)

    @property
    def total_cache_slots(self) -> int:
        return sum(t.decode.total_cache_slots for t in self.turns)

    @property
    def mean_turn_latency(self) -> float:
        return sum(t.decode.latency_s for t in self.turns) / max(len(self.turns), 1)

    @property
    def probe_scores(self) -> dict[str, float]:
        return {f"turn{t.probe.after_turn}": t.probe_score for t in self.turns if t.probe_score is not None}

    @property
    def probe_details(self) -> list[dict]:
        out = []
        for i, t in enumerate(self.turns):
            if t.probe is None:
                continue
            lag = (i - t.probe.plant_turn) if t.probe.plant_turn is not None else None
            out.append(
                {
                    "turn": i,
                    "question": t.probe.question,
                    "plant_turn": t.probe.plant_turn,
                    "lag": lag,
                    "paraphrase": t.probe.paraphrase,
                    "kind": t.probe.kind,
                    "score": t.probe_score,
                    "stale_score": t.stale_score,
                    "answer": t.answer_text[:120],
                }
            )
        return out

    def summary(self) -> dict:
        stale = [t.stale_score for t in self.turns if t.stale_score is not None]
        s = {
            "strategy": self.strategy.describe(),
            "turns": len(self.turns),
            "prompt_tokens": self.total_prompt_tokens,
            "generated_tokens": self.total_generated_tokens,
            "cache_slots": self.total_cache_slots,
            "mean_turn_latency_s": round(self.mean_turn_latency, 4),
            "max_cache_len": max((max(t.decode.cache_max_lengths, default=0) for t in self.turns), default=0),
            "probe_scores": self.probe_scores,
            "probe_mean": (
                round(sum(self.probe_scores.values()) / len(self.probe_scores), 4) if self.probe_scores else None
            ),
            "stale_mean": round(sum(stale) / len(stale), 4) if stale else None,
            "probe_details": self.probe_details,
        }
        if self.corr is not None:
            s["memory_stats"] = self.corr.stats
        if self.kv_steps:
            s["kv_injected_slots"] = self.kv_injected
            s["kv_steps"] = self.kv_steps
        if self.bank is not None:
            s["rag_stats"] = dict(self.bank.stats, size=len(self.bank.texts))
        if self.chunk_bank is not None:
            s["chunk_stats"] = dict(self.chunk_bank.stats, size=len(self.chunk_bank.texts))
        tone = [t.tone_score for t in self.turns if t.tone_score is not None]
        s["tone_mean"] = round(sum(tone) / len(tone), 4) if tone else None
        s["rep_mean"] = round(sum(t.rep_rate for t in self.turns) / max(len(self.turns), 1), 4)
        s["embed_tokens"] = self.embed_tokens
        s["attention_flops_proxy"] = self.total_cache_slots  # proportional to 4*d*L*slots
        return s


def _context(history: list[str], strategy: Strategy) -> str:
    if strategy.context == "full":
        return " ".join(history)
    return " ".join(history[-2 * strategy.window :])


def _prompt(bundle, context: str, user_text: str, notes: list[str] | None = None) -> str:
    note_block = ""
    if notes:
        note_block = "Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    if bundle.kind == "causal":
        return f"Dialogue.\n{note_block}{context}\nUser: {user_text}\nAssistant:"
    return f"{note_block}{context} User: {user_text}".strip()


@torch.no_grad()
def run_session(
    bundle,
    session: DialogSession,
    strategy: Strategy,
    *,
    max_new_tokens: int = 24,
    seed: int = 0,
) -> DialogueRun:
    """Replay ``session`` under ``strategy``; returns a :class:`DialogueRun`."""
    run = DialogueRun(strategy=strategy)
    history: list[str] = []
    # Build the full turn plan: every scripted user turn plus probes at
    # their designated turn index.
    probes_by_turn = {}
    for probe in session.probes:
        probes_by_turn[probe.after_turn] = probe
    planned = session.user_turns()
    total_turns = max([len(planned), *(p.after_turn + 1 for p in session.probes)], default=1)

    handle = None
    corr = None
    bank = None
    if strategy.memory:
        hook, corr = bundle.make_memory_hook(
            size=strategy.memory_size,
            beta=strategy.memory_beta,
            window_crit=strategy.memory_window_crit,
            retrieve_count=strategy.memory_retrieve,
            crit=strategy.memory_crit,
            merge_k=strategy.memory_merge_k,
            decay=strategy.memory_decay,
        )
        handle = bundle.memory_target().register_forward_pre_hook(hook, with_kwargs=True)
    run.corr = corr
    kv_wrappers = None
    if strategy.memory_kv:
        from .kvinject import install_kv_memory, uninstall_kv_memory

        if corr is None:
            _, corr = bundle.make_memory_hook(
                size=strategy.memory_size, window_crit=strategy.memory_window_crit,
                crit=strategy.memory_crit, merge_k=strategy.memory_merge_k,
                decay=strategy.memory_decay,
            )
            run.corr = corr
        kv_wrappers = install_kv_memory(bundle.model, corr, max_slots=strategy.kv_slots)
    # The bank exists in every strategy for tone *measurement*; it feeds the
    # prompt (RAG mode) only when strategy.rag is set.
    bank = RagBank(bundle.hidden_size, crit=strategy.rag_crit, device=bundle.device, centered=strategy.rag_centered)
    run.bank = bank
    # Generation-chunk memory: stores the assistant's own output chunks,
    # keyed by the request that produced them (task → solution retrieval).
    chunk_bank = None
    if strategy.gen_chunks:
        chunk_bank = ChunkBank(bundle.hidden_size, centered=strategy.chunk_centered,
                               upsert=strategy.chunk_upsert, device=bundle.device)
    run.chunk_bank = chunk_bank

    try:
        turn_idx = 0
        while turn_idx < total_turns:
            probe = probes_by_turn.get(turn_idx)
            if probe is not None:
                user_text, question = probe.question, probe
            elif turn_idx < len(planned):
                user_text, question = planned[turn_idx], None
            else:
                turn_idx += 1
                continue

            notes: list[str] | None = None
            user_emb = None
            if bank is not None:
                user_emb, n_tok = embed_utterance(bundle, user_text, layer=strategy.rag_layer), _n_tokens(bundle, user_text)
                run.embed_tokens += n_tok
                if strategy.rag:
                    notes = [t for t, _, _ in bank.retrieve(user_emb, k=strategy.rag_k, gist=gist_of(corr), lam=strategy.rag_lambda)]
                if chunk_bank is not None and chunk_bank.texts:
                    q_tokens = None
                    if strategy.chunk_score == "maxsim":
                        _, q_tokens, n_q = embed_chunk(bundle, user_text, mode="late")
                        run.embed_tokens += n_q
                    chunk_notes = [t for t, _, _ in chunk_bank.retrieve(
                        user_emb, k=strategy.chunk_k, score=strategy.chunk_score, query_tokens=q_tokens)]
                    notes = (notes or []) + chunk_notes

            context = _context(history, strategy)
            prompt_text = bundle.to_chat(_prompt(bundle, context, user_text, notes))
            prompt_ids = bundle.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
            decode = _generate(bundle, prompt_ids, strategy, max_new_tokens=max_new_tokens, seed=seed + turn_idx)
            answer = bundle.tokenizer.decode(decode.token_ids, skip_special_tokens=True).strip()

            tone = None
            if bank is not None and answer:
                ans_emb = embed_utterance(bundle, answer, layer=strategy.rag_layer)
                run.embed_tokens += _n_tokens(bundle, answer)
                tone = tone_alignment(bundle, ans_emb, bank)
                bank.add(user_text, user_emb, turn_idx)
            elif bank is not None:
                bank.add(user_text, user_emb, turn_idx)

            # Ingest the generation itself: chunk it, embed each chunk in the
            # context that produced it (value), key it by the block's own
            # layer activations so pieces of one huge generation stay
            # distinguishable (legacy: key by the request).
            if chunk_bank is not None and answer:
                for ch in chunk_text(answer, max_tokens=strategy.chunk_tokens, tokenizer=bundle.tokenizer):
                    val_emb, val_tokens, n_ch = embed_chunk(
                        bundle, ch, context=prompt_text, mode=strategy.chunk_embed,
                        max_tokens=strategy.chunk_tokens)
                    run.embed_tokens += n_ch
                    if strategy.chunk_key == "content":
                        key_emb, n_key = embed_block_key(
                            bundle, ch, layers=strategy.chunk_key_layers,
                            max_tokens=strategy.chunk_tokens)
                        run.embed_tokens += n_key
                    else:
                        key_emb = user_emb
                    chunk_bank.add(ch, key_emb=key_emb, val_emb=val_emb, turn=turn_idx,
                                   val_tokens=val_tokens if strategy.chunk_embed == "late" else None)

            history.append(f"User: {user_text} Assistant: {answer}")
            score = score_probe_answer(answer, question) if question else None
            stale_score = None
            if question is not None and question.stale:
                stale_score = keyword_score(answer, question.stale)
            run.turns.append(
                TurnResult(
                    user_text=user_text,
                    answer_text=answer,
                    prompt_tokens=int(prompt_ids.shape[1]),
                    decode=decode,
                    probe=question,
                    probe_score=score,
                    stale_score=stale_score,
                    tone_score=tone,
                    rep_rate=repetition_rate(answer),
                    injected=notes,
                )
            )
            turn_idx += 1
    finally:
        if handle is not None:
            handle.remove()
        if kv_wrappers is not None:
            from .kvinject import uninstall_kv_memory

            uninstall_kv_memory(bundle.model)
            s_inj = sum(w.stats["injected_slots"] for w in kv_wrappers)
            s_steps = sum(w.stats["steps"] for w in kv_wrappers)
            run.kv_injected, run.kv_steps = s_inj, s_steps
    return run


def _n_tokens(bundle, text: str) -> int:
    return len(bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0])


def _generate(bundle, prompt_ids: torch.Tensor, strategy: Strategy, *, max_new_tokens: int, seed: int) -> DecodeResult:
    # Keep the *most recent* tail of the prompt: a growing dialogue must
    # never exceed the model's positional embedding range.
    prompt_ids = prompt_ids[:, -bundle.max_prompt_len(reserve=max_new_tokens) :]
    if bundle.kind == "causal":
        return decode_causal(
            bundle.model,
            prompt_ids.to(bundle.device),
            kv=strategy.kv,
            cache=strategy.cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_ids=bundle.eos_ids(),
            seed=seed,
        )
    return decode_encdec(
        bundle.model,
        prompt_ids.to(bundle.device),
        decoder_start_id=bundle.decoder_start_id(),
        kv=strategy.kv,
        cache=strategy.cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_ids=(1,),
        seed=seed,
    )
