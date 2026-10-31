"""(б) Train the model to READ the memory channel — LoRA for KV-injection.

v4 showed the injection channel is free on the GPU but unreadable: an
untrained model treats position-free memory keys/values as tone noise
(0/24 fact recall). This trains exactly that reader:

* facts are embedded (frozen base-model utterance states) and injected as
  extra K/V in the attention layers — same geometry as tmem.kvinject;
* the prompt deliberately does NOT contain the facts: only attending to
  memory can produce the right answer;
* LoRA (peft) adapts q/k/v/o; loss on the answer tokens;
* held-out eval with unseen entity/value pairings reports recall before/after.

VRAM etiquette: set_per_process_memory_fraction caps our process so a
parallel user training job can never be squeezed out.
"""

from __future__ import annotations

import random

import torch

from .models import ModelBundle

IGNORE = -100
FACT_TPL = "The name of my {thing} is {code}."
Q_TPL = "What is the name of my {thing}?"
ALT_Q_TPL = "Tell me the name of my {thing}."
ANSWER_TPL = "Your {thing} is called {code}."
FILLER = "Tell me a bit more about your day in a few sentences."

THINGS = ["cat", "dog", "robot", "parrot", "guitar", "bike", "laptop", "plant", "mug", "fridge",
          "lamp", "watch", "knife", "boot", "kettle"]


def _rand_code(rng: random.Random) -> str:
    """Unique pseudo-lexical code - impossible to memorize in weights."""
    letters = "BCDFGHJKLMNPRSTVWXYZ"
    return f"{rng.choice(letters)}{rng.randint(100, 999)}{rng.choice('aeiou')}{rng.randint(10, 99)}"


def _make_item(rng: random.Random) -> dict:
    thing = THINGS[rng.randrange(len(THINGS))]
    code = _rand_code(rng)
    facts = [FACT_TPL.format(thing=thing, code=code)]
    if rng.random() < 0.6:
        other = THINGS[rng.randrange(len(THINGS))]
        facts.append(FACT_TPL.format(thing=other, code=_rand_code(rng)))
    return {
        "facts": facts,
        "gold": code,
        "question": rng.choice([Q_TPL, ALT_Q_TPL]).format(thing=thing),
        "answer": ANSWER_TPL.format(thing=thing, code=code),
    }


def _items(n: int, rng: random.Random, offset: int = 0) -> list[dict]:
    """(fact, question, answer) with globally unique random values per pool:
    a model that has not learned to READ the injected memory cannot get the
    answer right even once - memorization shortcuts are closed."""
    return [_make_item(rng) for _ in range(n)]


class GradientKVWrap:
    """Attention forward that appends holder['mem'] as extra K/V (grads flow)."""

    def __init__(self, attn, holder: dict) -> None:
        import copy

        self.attn = attn
        self.holder = holder
        # dedicated read head: warm-started copies + zero-initialized gate,
        # so the channel starts closed and cannot corrupt base behaviour
        self.mem_k = copy.deepcopy(attn.k_proj)
        self.mem_v = copy.deepcopy(attn.v_proj)
        self.gate = torch.nn.Parameter(torch.zeros(1, device=attn.k_proj.weight.device))
        self.extra_params = [self.mem_k.weight, self.mem_v.weight, self.gate]

    def __call__(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        a = self.attn
        if self.holder.get("mem") is None:
            return a._pre_wrap_forward(hidden_states, position_embeddings, attention_mask,
                                       past_key_values=past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, a.head_dim)
        query_states, gate = torch.chunk(
            a.q_proj(hidden_states).view(*input_shape, -1, a.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        q = a.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = a.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, a.layer_idx)   # THE cache update (was missing!)

        mem = self.holder.get("mem")
        iface = ALL_ATTENTION_FUNCTIONS.get_interface(a.config._attn_implementation, M.eager_attention_forward)
        if attention_mask is not None:
            attention_mask = attention_mask.to(q.dtype)
        main_out, _ = iface(a, q, k, v, attention_mask, dropout=0.0, scaling=a.scaling, **kwargs)

        g = float(torch.tanh(self.gate))
        if mem is not None and (self.mem_k.training or g != 0.0):
            mk = a.k_norm(self.mem_k(mem.to(self.mem_k.weight.dtype)).view(1, mem.shape[1], -1, a.head_dim)).transpose(1, 2)
            mv = self.mem_v(mem.to(self.mem_v.weight.dtype)).view(1, mem.shape[1], -1, a.head_dim).transpose(1, 2)
            mem_out, _ = iface(a, q, mk.to(q.dtype), mv.to(q.dtype), None, dropout=0.0, scaling=a.scaling)
            attn_output = main_out + torch.tanh(self.gate).to(main_out.dtype) * mem_out
        else:
            attn_output = main_out  # gate closed in eval: bit-identical to base
        attn_output = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        return a.o_proj(attn_output), None


def _install(model, holder) -> list:
    wrapped = []
    for blk in model.model.layers:
        attn = getattr(blk, "self_attn", None)
        if attn is not None:
            attn._pre_wrap_forward = attn.forward
            attn.forward = GradientKVWrap(attn, holder)
            wrapped.append(attn)
    return wrapped


def _extra_params(wrapped: list):
    return [p for attn in wrapped for p in attn.forward.extra_params]


def _uninstall(wrapped: list) -> None:
    for attn in wrapped:
        attn.forward = attn._pre_wrap_forward
        del attn._pre_wrap_forward


def _encode(bundle: ModelBundle, text: str) -> torch.Tensor:
    ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=640)["input_ids"]
    return ids.to(bundle.device)


@torch.no_grad()
def _fact_states(bundle: ModelBundle, text: str, cap: int = 24) -> torch.Tensor:
    """Token-level last-hidden states of a fact sentence (lexical content
    survives here, which a single mean-pooled vector loses)."""
    ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=cap)["input_ids"].to(bundle.device)
    out = bundle.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0]  # (L, d)


def precompute_mem(bundle: ModelBundle, items: list[dict], max_rows: int = 64) -> None:
    """Freeze memory rows for each item (computed once with the base model)."""
    for item in items:
        parts = [_fact_states(bundle, f) for f in item["facts"]]
        item["mem_rows"] = torch.cat(parts, dim=0)[:max_rows].to(torch.bfloat16).cpu()


def _sample(bundle: ModelBundle, item: dict, rng: random.Random, facts_in_prompt: bool = False):
    """(input_ids, labels, mem) for one memory-QA training example."""
    ctx = f"{FILLER}\nAssistant: Of course! {FILLER}\n"
    if facts_in_prompt:
        ctx = " ".join(item["facts"]) + " " + ctx
        mem = None
    else:
        mem = item["mem_rows"].to(bundle.device)
    prompt = bundle.to_chat(f"Dialogue.\n{ctx}User: {item['question']}")
    full = prompt + item["answer"] + "\n"
    prompt_ids = _encode(bundle, prompt)[0]
    full_ids = _encode(bundle, full)[0]
    n_prompt = prompt_ids.shape[0]
    labels = full_ids.clone()
    labels[:n_prompt] = IGNORE
    return full_ids.unsqueeze(0), labels.unsqueeze(0), mem.unsqueeze(0) if mem is not None else None


@torch.no_grad()
def evaluate(bundle: ModelBundle, wrapped: list, holder: dict, items: list[dict], max_new=24,
             use_memory: bool = True, facts_in_prompt: bool = False) -> float:
    """Fraction of held-out answers whose gold value is produced."""
    from .compat import cache_length
    from .generate import pick_token

    hits = 0
    n_ans = len(_encode(bundle, items[0]["answer"] + "\n")[0])
    for item in items:
        ids, _, mem = _sample(bundle, item, random.Random(0), facts_in_prompt=facts_in_prompt)
        holder["mem"] = None if facts_in_prompt else (mem if use_memory else None)
        prompt_only = ids[:, : ids.shape[1] - n_ans]
        out = bundle.model(input_ids=prompt_only, attention_mask=torch.ones_like(prompt_only), use_cache=True)
        cur = out.logits[:, -1:]
        gen = []
        pos = prompt_only.shape[1] - 1
        for _ in range(max_new):
            nxt = pick_token(cur, do_sample=False, temperature=1.0, top_p=1.0, generator=None)
            gen.append(nxt)
            if nxt in bundle.eos_ids():
                break
            pos += 1
            step = torch.tensor([[nxt]], device=prompt_only.device)
            clen = cache_length(out.past_key_values)
            out = bundle.model(input_ids=step, past_key_values=out.past_key_values,
                               position_ids=torch.full((1, 1), pos, dtype=torch.long, device=step.device),
                               attention_mask=torch.ones(1, clen + 1, dtype=torch.long, device=step.device),
                               use_cache=True)
            cur = out.logits[:, -1:]
        text = bundle.tokenizer.decode(gen, skip_special_tokens=True)
        hits += int(item["gold"].lower() in text.lower())
    holder["mem"] = None
    return hits / max(len(items), 1)


def train(bundle: ModelBundle, steps: int = 300, lr: float = 2e-4, batch: int = 2,
          eval_every: int = 100, eval_n: int = 15, seed: int = 0, memory_fraction: float = 0.55,
          facts_in_prompt: bool = False, log=print) -> dict:
    """Fine-tune attention adapters; streaming fresh-code data (no memorization)."""
    from peft import LoraConfig, get_peft_model

    if bundle.device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    rng = random.Random(seed)
    peft_model = get_peft_model(
        bundle.model,
        LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                   lora_dropout=0.05),
    )
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    log(f"lora trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params | facts_in_prompt={facts_in_prompt}")

    holder: dict = {}
    wrapped = _install(bundle.model, holder)
    trainable = trainable + _extra_params(wrapped)  # dedicated read heads too
    log(f"with memory read heads: {sum(p.numel() for p in trainable)/1e6:.1f}M params")
    eval_items = _items(eval_n, rng, offset=7)
    precompute_mem(bundle, eval_items)
    halluc = evaluate(bundle, wrapped, holder, eval_items, use_memory=False)
    base_recall = evaluate(bundle, wrapped, holder, eval_items, facts_in_prompt=facts_in_prompt)
    log(f"baselines: without-facts {halluc:.2f} | as-trained-mode {base_recall:.2f}")

    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    for step in range(steps):
        bundle.model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=bundle.device)
        for _ in range(batch):
            item = _make_item(rng)          # fresh random code every micro-step
            if not facts_in_prompt:
                precompute_mem(bundle, [item])
            ids, labels, mem = _sample(bundle, item, rng, facts_in_prompt=facts_in_prompt)
            holder["mem"] = mem
            out = bundle.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
            total = total + out.loss
            del ids, labels, mem, out
        (total / batch).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(total.detach()) / batch)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            bundle.model.eval()
            r = evaluate(bundle, wrapped, holder, eval_items, facts_in_prompt=facts_in_prompt)
            log(f"step {step+1}: loss {sum(losses[-eval_every:])/eval_every:.3f} eval recall {r:.2f}")

    bundle.model.eval()
    final = evaluate(bundle, wrapped, holder, eval_items, facts_in_prompt=facts_in_prompt)
    log(f"recall AFTER training: {final:.2f}")
    _uninstall(wrapped)
    holder.clear()
    return {"hallucination_recall": halluc, "base_recall": base_recall,
            "final_recall": final, "loss_curve": losses[-10:]}


if __name__ == "__main__":
    import argparse

    from .models import load_model

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen35-4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--fraction", type=float, default=0.55)
    p.add_argument("--facts-in-prompt", action="store_true", help="control condition: facts visible in the prompt instead of injected memory")
    a = p.parse_args()
    b = load_model(a.model, device=a.device, dtype=torch.bfloat16)
    print(train(b, steps=a.steps, lr=a.lr, memory_fraction=a.fraction, facts_in_prompt=a.facts_in_prompt))