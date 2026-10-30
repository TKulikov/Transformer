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
from .semantic import embed_utterance

IGNORE = -100
FACT_TPL = "The name of my {thing} is {v}."
Q_TPL = "What is the name of my {thing}?"
ALT_Q_TPL = "Tell me the name of my {thing}."
ANSWER_TPL = "Your {thing} is called {v}."
FILLER = "Tell me a bit more about your day in a few sentences."

THINGS = ["cat", "dog", "robot", "parrot", "guitar", "bike", "laptop", "plant", "mug", "fridge",
          "lamp", "watch", "knife", "boot", "kettle"]
VALS = ["Barsik", "Zeus", "R2D2", "Kiwi", "Strat", "Brompton", "Thinkpad", "Fern", "Moka", "Nordic",
        "Lumen", "Timex", "Damask", "Vibram", "Breville"]


def _items(n: int, rng: random.Random, offset: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        thing = THINGS[(i + offset) % len(THINGS)]
        val = VALS[(i * 3 + offset + 1) % len(VALS)]
        facts = [FACT_TPL.format(thing=thing, v=val)]
        if rng.random() < 0.6:
            other = THINGS[(i + 7) % len(THINGS)]
            facts.append(FACT_TPL.format(thing=other, v=VALS[(i * 3 + offset + 4) % len(VALS)]))
        out.append({
            "facts": facts,
            "gold": val,
            "question": (Q_TPL if i % 2 else ALT_Q_TPL).format(thing=thing),
            "answer": ANSWER_TPL.format(thing=thing, v=val),
        })
    return out


class GradientKVWrap:
    """Attention forward that appends holder['mem'] as extra K/V (grads flow)."""

    def __init__(self, attn, holder: dict) -> None:
        self.attn = attn
        self.holder = holder

    def __call__(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        a = self.attn
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

        mem = self.holder.get("mem")
        mask = attention_mask
        if mem is not None:
            mk = a.k_norm(a.k_proj(mem).view(1, mem.shape[1], -1, a.head_dim)).transpose(1, 2)
            mv = a.v_proj(mem).view(1, mem.shape[1], -1, a.head_dim).transpose(1, 2)
            k = torch.cat([k, mk.to(k.dtype)], dim=-2)
            v = torch.cat([v, mv.to(v.dtype)], dim=-2)
            if mask is not None:
                b, hh, s_q, _ = mask.shape
                mask = torch.cat([mask, torch.zeros(b, hh, s_q, mem.shape[1], dtype=mask.dtype, device=mask.device)], dim=-1)

        iface = ALL_ATTENTION_FUNCTIONS.get_interface(a.config._attn_implementation, M.eager_attention_forward)
        attn_output, _ = iface(a, q, k, v, mask, dropout=0.0, scaling=a.scaling, **kwargs)
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


def _uninstall(wrapped: list) -> None:
    for attn in wrapped:
        attn.forward = attn._pre_wrap_forward
        del attn._pre_wrap_forward


def _encode(bundle: ModelBundle, text: str) -> torch.Tensor:
    ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=640)["input_ids"]
    return ids.to(bundle.device)


def _sample(bundle: ModelBundle, item: dict, rng: random.Random):
    """(input_ids, labels, mem) for one memory-QA training example."""
    ctx = f"{FILLER}\nAssistant: Of course! {FILLER}\n"
    prompt = bundle.to_chat(f"Dialogue.\n{ctx}User: {item['question']}")
    full = prompt + item["answer"] + "\n"
    prompt_ids = _encode(bundle, prompt)[0]
    full_ids = _encode(bundle, full)[0]
    n_prompt = prompt_ids.shape[0]
    labels = full_ids.clone()
    labels[:n_prompt] = IGNORE
    mem = torch.stack([embed_utterance(bundle, f) for f in item["facts"]]).to(torch.bfloat16)
    return full_ids.unsqueeze(0), labels.unsqueeze(0), mem.unsqueeze(0)


@torch.no_grad()
def evaluate(bundle: ModelBundle, wrapped: list, holder: dict, items: list[dict], max_new=24,
             use_memory: bool = True) -> float:
    """Fraction of held-out answers whose gold value is produced from memory."""
    from .compat import cache_length
    from .generate import pick_token

    hits = 0
    n_ans = len(_encode(bundle, items[0]["answer"] + "\n")[0])
    for item in items:
        ids, _, mem = _sample(bundle, item, random.Random(0))
        holder["mem"] = mem if use_memory else None
        prompt_only = ids[:, : ids.shape[1] - n_ans]
        out = bundle.model(input_ids=prompt_only, attention_mask=torch.ones_like(prompt_only), use_cache=True)
        cur = out.logits[:, -1:]
        gen = []
        for _ in range(max_new):
            nxt = pick_token(cur, do_sample=False, temperature=1.0, top_p=1.0, generator=None)
            gen.append(nxt)
            if nxt in bundle.eos_ids():
                break
            step = torch.tensor([[nxt]], device=prompt_only.device)
            out = bundle.model(input_ids=step, past_key_values=out.past_key_values,
                               attention_mask=torch.ones(1, cache_length(out.past_key_values) + 1,
                                                         dtype=torch.long, device=prompt_only.device),
                               use_cache=True)
            cur = out.logits[:, -1:]
        text = bundle.tokenizer.decode(gen, skip_special_tokens=True)
        hits += int(item["gold"].lower() in text.lower())
    holder["mem"] = None
    return hits / max(len(items), 1)


def train(bundle: ModelBundle, steps: int = 300, lr: float = 2e-4, batch: int = 2,
          eval_every: int = 100, eval_n: int = 15, seed: int = 0, memory_fraction: float = 0.55,
          log=print) -> dict:
    """Fine-tune attention adapters so injected memory becomes readable."""
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
    log(f"lora trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params")

    holder: dict = {}
    wrapped = _install(bundle.model, holder)
    train_items = _items(200, rng, offset=0)
    eval_items = _items(eval_n, rng, offset=7)  # unseen pairings
    halluc = evaluate(bundle, wrapped, holder, eval_items, use_memory=False)
    base_recall = evaluate(bundle, wrapped, holder, eval_items)
    log(f"recall WITHOUT memory (hallucination): {halluc:.2f} | WITH memory untrained: {base_recall:.2f}")

    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    for step in range(steps):
        bundle.model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=bundle.device)
        for _ in range(batch):
            item = rng.choice(train_items)
            ids, labels, mem = _sample(bundle, item, rng)
            holder["mem"] = mem
            mask = torch.ones_like(ids)
            out = bundle.model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            total = total + out.loss
            del ids, labels, mem, out
        (total / batch).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(total.detach()) / batch)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            bundle.model.eval()
            r = evaluate(bundle, wrapped, holder, eval_items)
            log(f"step {step+1}: loss {sum(losses[-eval_every:])/eval_every:.3f} eval recall {r:.2f}")

    bundle.model.eval()
    final = evaluate(bundle, wrapped, holder, eval_items)
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
    a = p.parse_args()
    b = load_model(a.model, device=a.device, dtype=torch.bfloat16)
    print(train(b, steps=a.steps, lr=a.lr, memory_fraction=a.fraction))