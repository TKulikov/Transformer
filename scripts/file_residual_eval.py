"""Файл-резидуальный поиск на taskq_large (заказ 15.11, вечер).

Проверка предложения: ключ функции = латентный эмбеддинг в контексте класса
минус бегущее среднее ФАЙЛА (разреживание внутри файла); запрос сначала
выбирает файл по центроиду, затем вычитает общую часть файла и ищет внутри
плоско. Против клонов mid-ключей (провал flat_mid/hier_flat в search_retest).

Условия (те же 5 задач и генерация, что в search_retest):
* fileres_f1/f2/f3 — beta=0 (чистый контекстный эмбеддинг), k_files=1/2/3;
* fileres_fused_f2 — beta=1 (fused q_rec+тело), k_files=2 (rekey, без реингестии);
* flat_legacy / hier_flat — референсы (лидер ретеста и вчерашний лидер).

Диагностика: какие файлы выбрал stage 1 (находит ли core.utils для
dedupe/chunked). Выход: results/chunk_code/fileres/{summary.json,REPORT.md,<cond>.json}

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/file_residual_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from code_search_retest import build_flat_bank, generate, score_entry  # noqa: E402
from taskq_large_eval import PROJECT, TASKS, project_context  # noqa: E402
from tmem.fileres import FileResidualBank  # noqa: E402
from tmem.hiermem import HierLatentBank, build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "chunk_code" / "fileres"
K = 6
CONDITIONS = [
    ("fileres_f1", 0.0, 1),
    ("fileres_f2", 0.0, 2),
    ("fileres_f3", 0.0, 3),
    ("fileres_fused_f2", 1.0, 2),
]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    files = [p.read_text(encoding="utf-8") for p in sorted(PROJECT.rglob("*.py"))]
    nodes = build_project_tree(PROJECT)
    ctx = project_context()

    bank = FileResidualBank(nodes, beta=0.0, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_fileres = time.perf_counter() - t0
    print(f"fileres ингестия: {bank.stats} за {ingest_fileres:.0f}s", flush=True)

    flat_legacy, ingest_flat = build_flat_bank(bundle, files, "mean", "mid")
    print(f"flat_legacy: {len(flat_legacy.texts)} чанков за {ingest_flat:.0f}s", flush=True)
    hier = HierLatentBank(nodes, beta=1.0, device=bundle.device)
    t0 = time.perf_counter()
    hier.ingest(bundle)
    ingest_hier = time.perf_counter() - t0
    print(f"hier: {hier.stats} за {ingest_hier:.0f}s", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "project": "taskq_large",
               "k": K, "ingest_s": {"fileres": round(ingest_fileres, 1),
                                     "flat_legacy": round(ingest_flat, 1),
                                     "hier": round(ingest_hier, 1)},
               "conditions": {}}

    def run_cond(cond: str, retrieve_fn) -> None:
        print(f"\n=== {cond} ===", flush=True)
        t0 = time.perf_counter()
        entries = []
        for task in TASKS:
            notes, picked = retrieve_fn(task["user"])
            answer = generate(bundle, ctx, notes, task["user"])
            entry = score_entry(task, answer, notes)
            entry["files_picked"] = picked
            entries.append(entry)
            print(f"  {task['id']}: reuse={entry['reuse']} hits={entry['retrieval_hit']} "
                  f"files={picked}", flush=True)
        summary["conditions"][cond] = {"wall_s": round(time.perf_counter() - t0, 1),
                                       "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for cond, beta, k_files in CONDITIONS:
        bank.rekey(beta)

        def retrieve(q, _b=bank, _kf=k_files):
            res, names = _b.retrieve(bundle, q, k_files=_kf, k=K)
            return [n.text for n, _, _ in res], names

        run_cond(cond, retrieve)

    from tmem.semantic import embed_utterance

    def retrieve_flat(q):
        return [t for t, _, _ in flat_legacy.retrieve(embed_utterance(bundle, q, layer="last"),
                                                      k=K, score="key")], []

    def retrieve_hier(q):
        res = hier.retrieve(bundle, q, k_branches=5, k_children=2, alpha=0.5, mode="flat")
        return [n.text for n, _, _ in res], []

    run_cond("flat_legacy", retrieve_flat)
    run_cond("hier_flat", retrieve_hier)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Файл-резидуальный поиск (taskq_large): разреживание внутри файла",
        "",
        "Ключ = латентный эмбеддинг функции в контексте класса − центроид файла;",
        "запрос: stage 1 — файл по центроиду, stage 2 — (q − центроид) против residual-ключей.",
        f"k={s['k']} заметок. Ингестия (с): {s['ingest_s']}.",
        "",
        "| Условие | Reuse /10 | Retrieval hits /10 | Valid /5 |",
        "|---|---|---|---|",
    ]
    for cond, c in s["conditions"].items():
        ents = c["entries"]
        reuse = sum(sum(e["reuse"].values()) for e in ents)
        hits = sum(sum(e["retrieval_hit"].values()) for e in ents)
        valid = sum(e["valid_code"] for e in ents)
        lines.append(f"| {cond} | {reuse} | {hits} | {valid} |")
    lines += ["", "## Выбранные файлы (stage 1)", "",
              "| Условие | " + " | ".join(t["id"] for t in TASKS) + " |",
              "|---|" + "---|" * len(TASKS)]
    for cond, c in s["conditions"].items():
        cells = [", ".join(e.get("files_picked") or []) or "—" for e in c["entries"]]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "fileres_f1/f2/f3 — beta=0, k_files=1/2/3; fileres_fused_f2 — beta=1 (fused-ключ);",
              "flat_legacy / hier_flat — референсы из search_retest. Данные: `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
