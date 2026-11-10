"""Final aggregation of a corpus run: AGGREGATE.md + summary.json.

Usage:
    .venv/bin/python scripts/aggregate_corpus.py results/corpus/main-v1

Uses the same statistics kernel as INTERIM.md (tmem.corpus_stats), so
partial and final numbers are methodologically identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmem.corpus_stats import load_run, render_interim_md, summarize  # noqa: E402


def main(run_dir: str) -> int:
    path = Path(run_dir)
    if not (path / "run_config.json").exists():
        raise SystemExit(f"нет run_config.json в {path}")
    run = load_run(path)
    summary = summarize(run)
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = render_interim_md(path).replace("# Промежуточный отчёт:", "# Итоговый отчёт:", 1)
    (path / "AGGREGATE.md").write_text(md)
    print(f"диалогов: {summary['n_dialogues_done']}/{summary['n_dialogues_total']}")
    print(f"записано: {path / 'summary.json'}, {path / 'AGGREGATE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/corpus/main-v1"))
