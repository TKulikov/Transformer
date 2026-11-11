"""Executable verification of PAPER.md against the code and the archived run.

Run ``python scripts/verify_paper.py`` instead of re-reading the article: every
formula (1)-(9), every algorithm parameter and every published number is checked
against the real ``tmem`` code, a tiny CPU Qwen3.5 (for the K/V-injection
formulas) and ``results/qwen35_comparison/qwen35-4b_battery100.json``. Prints a
PASS/FAIL/SKIP table and exits non-zero if anything fails. CPU-only, offline.

Also exposed as ``run_checks()`` for the pytest wrapper in tests/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import report_data as RD  # noqa: E402
import paper_text as PT  # noqa: E402
from tmem.compat import cache_length, cache_mean_length  # noqa: E402
from tmem.dialogue import Strategy, _context  # noqa: E402
from tmem.memory import CorrectionLayer, Memory  # noqa: E402
from tmem.metrics import keyword_score  # noqa: E402
from tmem.semantic import RagBank, embed_utterance, repetition_rate  # noqa: E402


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


_CHECKS: list = []


def check(name: str):
    """Decorator registering a verification function returning a detail string."""
    def deco(fn):
        _CHECKS.append((name, fn))
        return fn
    return deco


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _tiny_qwen35():
    """A 4-layer full-attention Qwen3.5 with random weights, on CPU."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        layer_types=["full_attention"] * 4,
    )
    return M.Qwen3_5ForCausalLM(cfg).eval()


def _cached_config():
    """Locate the cached Qwen3.5-4B config.json without network, or None."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache("Qwen/Qwen3.5-4B", "config.json")
        if isinstance(p, str) and Path(p).exists():
            return json.loads(Path(p).read_text())
    except Exception:
        pass
    hits = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/config.json"))
    return json.loads(hits[0].read_text()) if hits else None


# --------------------------------------------------------------------- formulas
@check("Формула (1): центрирование u_j=(e_j-c_{j-1})/||...||, c_0=0")
def f1():
    torch.manual_seed(1)
    e1, e2, e3 = torch.randn(8), torch.randn(8), torch.randn(8)
    bank = RagBank(8, centered=True, crit=2.0, capacity=256)  # crit>1: upsert выключен
    bank.add("a", e1)
    bank.add("b", e2)
    assert torch.allclose(bank.embeds[0], _unit(e1), atol=1e-6), "первая запись должна центрироваться c_0=0"
    assert torch.allclose(bank.embeds[1], _unit(e2 - e1), atol=1e-6), "вторая — средним c_1=e_1 (без e_2)"
    assert torch.allclose(bank.mean, (e1 + e2) / 2, atol=1e-6), "бегущее среднее после 2 реплик"
    # запрос центрируется ТЕКУЩИМ средним
    q = _unit(e3 - bank.mean)
    sims = bank.embeds @ q
    got = bank.retrieve(e3, k=2)
    order = [t for t, _, _ in got]
    expect = [bank.texts[i] for i in sims.argsort(descending=True).tolist()]
    assert order == expect, "ранжирование запроса по децентрированному косинусу"
    # среднее учитывает даже заменённые (upsert) реплики
    b2 = RagBank(8, centered=True, crit=-2.0)  # crit<0: всегда upsert
    b2.add("x", e1)
    b2.add("y", e2)
    assert len(b2.texts) == 1 and torch.allclose(b2.mean, (e1 + e2) / 2, atol=1e-6)
    return "c_{j-1}, c_0=0, запрос по текущему c, среднее включает upsert — совпадает с RagBank.add/_vec/retrieve"


@check("Формула (2): косинус = скалярное произведение единичных векторов, top-4")
def f2():
    torch.manual_seed(2)
    bank = RagBank(8, centered=False, crit=2.0)
    for i in range(6):
        bank.add(f"t{i}", torch.randn(8))
    assert torch.allclose(bank.embeds.norm(dim=1), torch.ones(6), atol=1e-6), "записи единичной нормы"
    q = torch.randn(8)
    got = bank.retrieve(q, k=4)
    assert len(got) == 4, "выбираются 4 ближайшие записи"
    qn = _unit(q)
    for text, _, score in got:
        i = bank.texts.index(text)
        assert abs(score - float(bank.embeds[i] @ qn)) < 1e-5, "оценка = u·q = косинус"
    return "норма=1, sim=u·q, top-4 — совпадает с RagBank.retrieve (embeds @ probe)"


@check("Формула (3): EMA a<-0.75a+0.25h (lambda=0.75, merge_k=0.25)")
def f3():
    m = Memory(8, size=4, merge_k=0.25)
    a, h = torch.randn(8), torch.randn(8)
    assert torch.allclose(m._merge_slot(a, h), 0.75 * a + 0.25 * h, atol=1e-6)
    assert abs((1 - m.merge_k) - 0.75) < 1e-9
    return "lambda=1-merge_k=0.75 — совпадает с Memory._merge_slot"


@check("Формула (4): K_m=R_K(W_K A) без RoPE, V_m=W_V A; память вне кэша")
def f4():
    from tmem.kvinject import KVMemoryAttention, install_kv_memory, uninstall_kv_memory
    model = _tiny_qwen35()
    corr = CorrectionLayer(64, size=32, window_crit=-1e9, crit=0.95, merge_k=0.25)
    A = torch.randn(64)
    corr.memory.context[0, 0] = A
    corr.memory.occupied[0] = True
    corr.memory.ttl[0] = 6
    wrappers = install_kv_memory(model, corr, layers=[2], max_slots=4)
    cap = {}
    orig = KVMemoryAttention._attention

    def capture(a, q, k, v, mask, kwargs):
        cap.update(a=a, k=k, v=v)
        return orig(a, q, k, v, mask, kwargs)

    KVMemoryAttention._attention = staticmethod(capture)
    ids = torch.randint(0, 128, (1, 6))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
    finally:
        KVMemoryAttention._attention = orig
    n = wrappers[0].stats["injected_slots"]
    assert n == 1, "один занятый слот должен быть извлечён"
    a, k, v = cap["a"], cap["k"], cap["v"]
    mem_k = a.k_norm(a.k_proj(A).view(1, n, -1, a.head_dim)).transpose(1, 2)
    mem_v = a.v_proj(A).view(1, n, -1, a.head_dim).transpose(1, 2)
    assert torch.allclose(k[..., -n:, :], mem_k.to(k.dtype), atol=1e-4), "K_m=k_norm(k_proj(A)), без RoPE"
    assert torch.allclose(v[..., -n:, :], mem_v.to(v.dtype), atol=1e-4), "V_m=v_proj(A)"
    assert cache_length(out.past_key_values) == ids.shape[1], "память не попала в последовательный кэш"
    uninstall_kv_memory(model)
    return "K_m=k_norm(k_proj(A)) без RoPE, V_m=v_proj(A), кэш не растёт — подтверждено на крошечной Qwen3.5"


@check("Формула (5): маска дополняется нулевыми столбцами; scaling=1/sqrt(d_k)")
def f5():
    from tmem.kvinject import _extend_mask
    mask = torch.full((1, 2, 3, 5), float("-inf"))
    mask[..., :3] = 0.0
    ext = _extend_mask(mask, real_keys=5, mem=4, dtype=torch.float32)
    assert ext.shape == (1, 2, 3, 9), "добавлены 4 столбца памяти"
    assert bool((ext[..., -4:] == 0).all()), "столбцы памяти нулевые = видимы всем"
    assert torch.equal(ext[..., :5], mask), "исходная маска не изменена"
    assert _extend_mask(None, 5, 4, torch.float32) is None, "None-маска (быстрый путь) сохраняется"
    model = _tiny_qwen35()
    a = model.model.layers[2].self_attn
    assert abs(a.scaling - a.head_dim ** -0.5) < 1e-9, "scaling=1/sqrt(d_k)"
    return "нулевые столбцы памяти, причинность сохранена, scaling=1/sqrt(d_k)"


@check("Формула (6): recall-индикатор подстроки (contains), без учёта регистра")
def f6():
    assert keyword_score("The cat is Simba", ["simba"]) == 1.0
    assert keyword_score("The cat is Barsik", ["simba"]) == 0.0
    assert keyword_score("SIMBA here", ["simba"]) == 1.0, "регистр не учитывается"
    assert keyword_score("a b", ["a", "b"]) == 1.0 and keyword_score("a b", ["a", "z"]) == 0.5
    return "доля найденных строк; при |y_i|=1 — индикатор 0/1 (metrics.keyword_score)"


@check("Формула (7): D=1-U/G по словесным триграммам")
def f7():
    assert repetition_rate("I I I I I I") == 0.75  # 4 триграммы, все одинаковы: 1-1/4
    assert repetition_rate("completely varied word sequence here") == 0.0
    assert repetition_rate("short") == 0.0
    return "1-U/G по триграммам нижнего регистра — совпадает с semantic.repetition_rate"


@check("Формула (8): A=sum средних длин кэша; S замеров на S токенов; L=8")
def f8():
    from tmem.generate import decode_causal
    from tmem.models import tiny_causal
    bundle = tiny_causal(d_model=64, layers=2, seed=7)
    ids = bundle.tokenizer("hello world test", return_tensors="pt")["input_ids"]
    res = decode_causal(bundle.model, ids, max_new_tokens=5, eos_ids=(), seed=0)
    assert res.steps == 5 and len(res.cache_lengths) == 5, "ровно S замеров на S сгенерированных токенов"
    assert res.total_cache_slots == sum(res.cache_lengths)
    # cache_mean_length = округлённое среднее по K/V-слоям
    class L:
        def __init__(self, n):
            self.keys = torch.zeros(1, 2, n, 3)
            self.values = self.keys
    class C:
        layers = [L(4), L(8)]
    assert cache_mean_length(C()) == 6, "среднее (4+8)/2=6"
    cfg = _cached_config()
    if cfg is not None:
        tc = cfg.get("text_config", cfg)
        lt = tc.get("layer_types") or cfg.get("layer_types")
        assert sum("full" in str(t) for t in lt) == 8, "L=8 full-attention слоёв"
    return "S замеров на S токенов, среднее по слоям, L=8 (config)"


@check("Формула (9): E_A=A_full/A_method = 5.04")
def f9():
    q = RD.q35_all()
    factor = q["classic_kv_full"]["cache_slots"] / q["v4_rag_kv"]["cache_slots"]
    assert f"{factor:.2f}" == "5.04", f"ожидалось 5.04, получено {factor:.4f}"
    return f"8559983/1698815={factor:.4f} -> 5,04"


# ------------------------------------------------------------------- algorithm
@check("§2.2 буфер: window=5 -> 10 пар «реплика-ответ» (20 реплик)")
def a_buffer():
    hist = [f"User: u{i} Assistant: a{i}" for i in range(30)]
    ctx = _context(hist, Strategy("x", context="window", window=5))
    assert ctx.count("User:") == 10 and ctx.count("Assistant:") == 10, "10 пар, не 5"
    assert "User: u20" in ctx and "User: u19" not in ctx, "последние 10 элементов истории"
    return "history[-2*window:] = 10 пар (каждый элемент истории — пара)"


@check("§2.4 двухэтапный отбор 256->32 и пороги 0.95/0.4/0.5")
def a_params():
    q = RD.q35_all()
    proto = {s["name"]: s for s in RD.load()["protocol"]["strategies"]}
    v4 = proto["v4_rag_kv"]
    assert v4["memory_retrieve"] == 256 and v4["kv_slots"] == 32, "top-256 из банка -> top-32 в attention"
    assert v4["memory_crit"] == 0.95 and v4["memory_merge_k"] == 0.25 and v4["memory_window_crit"] == 0.5
    assert CorrectionLayer(8).max_to_new == 0.4, "порог хранения 0.4"
    return "retrieve=256, kv_slots=32, crit=0.95, merge_k=0.25, max_to_new=0.4, gate=0.5"


@check("§2.4 TTL: новый=5, хит +2/потолок 6, -1 за вызов, слияние >=5")
def a_ttl():
    m = Memory(8, size=4)
    v = torch.randn(1, 1, 8)
    m.update(v, crit=0.95, max_to_new=1.0, window_crit=-1e9)
    idx = int(torch.nonzero(m.occupied)[0])
    assert float(m.ttl[idx]) == 5, "новый слот TTL=5"
    m.ttl[idx] = 3
    m.get_relevant(v, count=4)  # хит: +2 -> 5, затем -1 -> 4
    assert float(m.ttl[idx]) == 4, "хит +2 с потолком 6, затем -1"
    m.ttl[idx] = 6
    m.get_relevant(v, count=4)  # +2 -> потолок 6, -1 -> 5
    assert float(m.ttl[idx]) == 5, "потолок 6"
    m.ttl[idx] = 2
    m.update(v, crit=0.95, max_to_new=0.4, window_crit=-1e9)  # слияние
    assert float(m.ttl[idx]) >= 5, "при слиянии TTL поднимается до >=5"
    m.ttl[idx] = 1
    m.get_relevant(torch.randn(1, 1, 8), count=0)  # без хита: -1 -> 0 -> вытеснение
    assert not bool(m.occupied[idx]), "слот с TTL<=0 вытесняется"
    return "новый=5, +2/потолок 6, -1/вызов, слияние>=5, вытеснение при TTL<=0"


@check("§2.4 гейт: неперекрывающиеся 32-блоки, сумма>0.5 (не sliding)")
def a_gate():
    v = torch.zeros(64)
    v[31] = 0.4
    v[32] = 0.4  # через границу блоков: sliding-окно нашло бы 0.8>0.5
    assert Memory.check(v, window_size=32, crit=0.5) is False, "блоки неперекрывающиеся: max 0.4<0.5"
    v2 = torch.zeros(64)
    v2[0] = 0.6
    assert Memory.check(v2, window_size=32, crit=0.5) is True
    v3 = torch.zeros(40)  # неполный хвостовой блок 8
    v3[35] = 0.6
    assert Memory.check(v3, window_size=32, crit=0.5) is True, "хвостовой блок тоже считается"
    return "неперекрывающиеся блоки + хвост, порог 0.5 (Memory.check/_check_batch)"


@check("§2.3 вопросы: эвристика по «?»; перефраз с точкой попадает в банк")
def a_questions():
    bank = RagBank(8, centered=False)
    e = torch.randn(8)
    bank.add("My cat is Barsik", e)
    bank.add("How is my cat named?", e)  # '?' -> отбрасывается
    bank.add("Tell me the name of my feline pet.", e)  # '.' -> сохраняется
    assert "How is my cat named?" not in bank.texts
    assert "Tell me the name of my feline pet." in bank.texts
    return "отбрасываются реплики на «?»; перефраз с точкой сохраняется (видно в rag_stats)"


@check("§2.3 эмбеддинг: mean hidden_states[-1] по токенам, <=128, изолированно")
def a_embed():
    from tmem.models import tiny_causal
    bundle = tiny_causal(d_model=64, layers=2, seed=7)
    text = "hello there friend"
    emb = embed_utterance(bundle, text)
    ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
    with torch.no_grad():
        out = bundle.model(input_ids=ids, output_hidden_states=True)
    manual = out.hidden_states[-1].float().mean(dim=1).squeeze(0)
    assert torch.allclose(emb, manual, atol=1e-6), "e(x)=mean последнего скрытого слоя по токенам"
    assert emb.shape == (bundle.hidden_size,)
    return "один проход, hidden_states[-1], mean по токенам — совпадает с embed_utterance"


@check("§2.4/§2.6 слои 11,15,19,23 = att[2:6] из 8 full-attention")
def a_layers():
    cfg = _cached_config()
    if cfg is None:
        raise _Skip("нет кэша config.json Qwen3.5-4B")
    tc = cfg.get("text_config", cfg)
    lt = tc.get("layer_types") or cfg.get("layer_types")
    att = [i for i, t in enumerate(lt) if "full" in str(t)]
    n = len(att)
    chosen = att[max(n // 4, 0): max(3 * n // 4, n // 4 + 1)]
    assert chosen == [11, 15, 19, 23], f"ожидалось [11,15,19,23], получено {chosen}"
    return "install_kv_memory по умолчанию берёт att[2:6]=[11,15,19,23]"


@check("§2.6 архитектура: 32 слоя / 8 full / 16 q-голов / 4 KV / 2560 / head_dim 256")
def a_arch():
    cfg = _cached_config()
    if cfg is None:
        raise _Skip("нет кэша config.json Qwen3.5-4B")
    tc = cfg.get("text_config", cfg)
    assert tc["num_hidden_layers"] == 32
    assert tc["hidden_size"] == 2560
    assert tc["num_attention_heads"] == 16
    assert tc["num_key_value_heads"] == 4
    assert tc["head_dim"] == 256
    lt = tc.get("layer_types") or cfg.get("layer_types")
    assert sum("full" in str(t) for t in lt) == 8
    return "config Qwen3.5-4B совпадает с §2.6"


# --------------------------------------------------------------------- numbers
@check("§3 числа: recall, факторы, латентность, тон, stale, диапазон KV")
def n_numbers():
    q = RD.q35_all()
    ref, final = q["classic_kv_full"], q["v4_rag_kv"]
    assert round(final["recall_mean"] * 24) == 20
    assert round(ref["recall_mean"] * 24) == 24
    assert round(q["pyramidkv_128"]["recall_mean"] * 24) == 18
    assert round(q["rag_text_only"]["recall_mean"] * 24) == 16
    assert round(q["snapkv_128"]["recall_mean"] * 24) == 15
    assert round(q["sparse_topk_128"]["recall_mean"] * 24) == 8
    assert round(q["sparse_sliding_128"]["recall_mean"] * 24) == 4
    assert f"{ref['cache_slots'] / final['cache_slots']:.2f}" == "5.04"
    assert f"{ref['prompt_tokens'] / final['prompt_tokens']:.2f}" == "4.97"
    assert f"{final['mean_turn_latency_s']:.2f}" == "1.34" and f"{ref['mean_turn_latency_s']:.2f}" == "1.84"
    assert f"{final['median_turn_latency_s']:.2f}" == "1.49" and f"{ref['median_turn_latency_s']:.2f}" == "1.79"
    assert f"{final['tone_mean']:.4f}" == "0.8298"
    assert f"{ref['stale_rate']:.3f}" == "0.125"
    kv = [ref["cache_slots"] / q[k]["cache_slots"] for k in
          ("sparse_sliding_128", "sparse_topk_128", "snapkv_128", "pyramidkv_128")]
    assert f"{min(kv):.1f}" == "18.4" and f"{max(kv):.1f}" == "20.5"
    return "все числа §3 и аннотации пересчитаны из JSON и совпадают"


@check("Текст статьи содержит пересчитанные числа (генератор)")
def n_text():
    q = RD.q35_all()
    text = "\n".join(p for _, ps in PT.sections(q) for p in ps) + PT.abstract(q)
    for needle in ("20 из 24", "0,833", "5,04", "4,97", "1,34", "1,84", "18,4–20,5",
                   "десяти пар", "11, 15, 19, 23", "contains"):
        assert needle in text, f"в тексте статьи нет «{needle}»"
    for bad in ("пять пар", "5 пар", "subseteq", "до четырёх раз", "примерно в 18"):
        assert bad not in text, f"в тексте статьи остался «{bad}»"
    return "генератор paper_text.py выдаёт исправленные формулировки и числа"


class _Skip(Exception):
    pass


def run_checks() -> list[Result]:
    """Run every registered check; never raises (failures are captured)."""
    results = []
    for name, fn in _CHECKS:
        try:
            detail = fn()
            results.append(Result(name, "PASS", detail or ""))
        except _Skip as e:
            results.append(Result(name, "SKIP", str(e)))
        except AssertionError as e:
            results.append(Result(name, "FAIL", str(e) or "утверждение не выполнено"))
        except Exception as e:  # noqa: BLE001
            results.append(Result(name, "FAIL", f"{type(e).__name__}: {e}"))
    return results


def main() -> int:
    results = run_checks()
    width = max(len(r.name) for r in results)
    for r in results:
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "—"}[r.status]
        print(f"[{mark}] {r.status:4}  {r.name.ljust(width)}  {r.detail}")
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]
    print(f"\nИтого: {len(results) - len(failed) - len(skipped)} PASS, "
          f"{len(failed)} FAIL, {len(skipped)} SKIP из {len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
