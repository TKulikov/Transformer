"""Диагностика: меняет ли K/V-инжект живой памяти сами эмбеддинги?

Нулевой результат memctx_eval может означать (а) механизм не помогает, либо
(б) механизм не «зацепился» (эмбеддинги почти не изменились). Различаем:
сравниваем per-function ключи standard vs memory-contextualized (cosine).
cos~1.0 → механизм инертен; cos<<1 → engages, но не помогает.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/memctx_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from taskq_large_eval import PROJECT  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    std = SemTreeBank(nodes, space="mean", device=bundle.device, memory_context=False)
    std.ingest(bundle)
    ctx = SemTreeBank(nodes, space="mean", device=bundle.device, memory_context=True)
    ctx.ingest(bundle)
    print(f"живых слотов в memctx: {ctx.stats.get('memctx_slots')}", flush=True)
    for space in ("mean", "mid"):
        a, b = std.raw[space], ctx.raw[space]
        an = torch.nn.functional.normalize(a, dim=1)
        bn = torch.nn.functional.normalize(b, dim=1)
        cos = (an * bn).sum(dim=1)
        print(f"{space}: cos(std, memctx) mean={cos.mean():.4f} "
              f"min={cos.min():.4f} max={cos.max():.4f} "
              f"доля cos<0.99: {(cos < 0.99).float().mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
