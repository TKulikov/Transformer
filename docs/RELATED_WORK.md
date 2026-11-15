# Related work / оценка новизны (заказ 15.11)

Проверка prior art по ключевым механизмам tmem перед оформлением статьи.
**Вывод: почти каждый механизм уже опубликован, часть — очень близко и недавно
(ACL 2026). Текущий framing статьи (v4 latent K/V injection как сигнатурный
вклад) несостоятелен: он и не нов, и эмпирически не подтверждён нашими тестами.**

## Карта: наш механизм → prior art

| Наш компонент | Ближайший prior art | Что у них | Отличие (если есть) |
|---|---|---|---|
| **K/V-инъекция памяти в mid-attention, без RoPE** (v4, `kvinject.py`) | **Memorizing Transformers** (Wu et al., ICLR 2022, arXiv:2203.08913); **Memory³** (arXiv:2407.01178, JML 2024) | kNN K/V памяти в ОДНОМ среднем слое, position-free, learned gate; Memory³ — sparse attention K/V + Faiss IVFPQ, parallel position encoding | У них **обучение** (fine-tune / pretraining with memory). Мы training-free в frozen Qwen — и показываем NULL. Это контр-результат, не метод |
| **«mid-слой — зона смыслов, туда инжектить»** | **Memorizing Transformers, абляция слоя** (2022, Table 14) | Прямо показано: память в середине стека даёт лучший perplexity, ближе к входу/выходу — хуже | Наша гипотеза о mid-слое уже опубликована и проверена в 2022 |
| **Онлайн-кластеризация по порогу τ: похож → в группу, иначе новая** (`semtree.py`) | **EverMemOS** (ACL 2026, arXiv:2601.02163); **CLAG** (ACL Findings 2026) | EverMemOS: «new MemCell → nearest MemScene centroid; if sim ≥ τ assimilate + update, else new scene» — **дословно наш алгоритм**. CLAG: agent-driven clustering | Практически идентично. Отличие: мы кластеризуем hidden-state кода, не текстовые заметки |
| **Двухстадийный поиск: top групп по центроиду → внутри плоско** | **CLAG** (2026); **RAPTOR** (ICLR 2024); **IVF** (2010) | CLAG: «first filters clusters via centroid, then retrieves within selected clusters» — **наш two-stage**. RAPTOR: tree traversal / collapsed | Не ново как механизм |
| **Рекурсивное объединение похожих центроидов в глубокое дерево** | **RAPTOR** (Sarthi et al., ICLR 2024, arXiv:2401.18059) | Рекурсивно embed→cluster(GMM+UMAP)→summarize, дерево снизу вверх | RAPTOR использует внешний SBERT + LLM-саммари узлов; мы — центроиды hidden-state без саммари. Механизм дерева тот же |
| **Residual-ключи: ключ − среднее группы, поиск внутри** | **IVF + residual encoding** (Jégou, Douze, Schmid, TPAMI 2011; Faiss arXiv:2401.08281) | `by_residual=true` по умолчанию: кодировать отклонение от центроида кластера, искать по residual запроса | **Классика ANN 2010 года.** Математически идентично. Не ново |
| **Центрирование (вычитание бегущего среднего), deviation-пространство** | Ethayarajh (EMNLP 2019, D19-1006); Mu & Viswanath «all-but-the-top» (ICLR 2018); Bis et al. «Too Much in Common» (NAACL 2021); Su et al. whitening (2021) | Анизотропия contextual embeddings, mean-centering/whitening improves similarity | Хорошо известно. Фон, не вклад |
| **Иерархическая кластеризация функций кода по смыслу** | **FeatLens** (arXiv:2609.26480, 2026); **Hydra** (arXiv:2602.11671); **RepoMind** (ICPC 2026) | FeatLens: «hierarchical clustering on function-level representations, grouping functions supporting the same feature» — очень близко | FeatLens: внешние эмбеддинги + LLM feature-описания. Мы: только hidden-state frozen-модели |
| **Training-free память для кода** | **MEMCoder** (THU-Agent, 2026) | «training-free self-evolving memory framework for private-library code generation» | Приложение занято |

## Честный вердикт по новизне

1. **Сигнатурный claim статьи (v4 K/V-инъекция) — двойная проблема:**
   - не нов (Memorizing Transformers 2022, Memory³ 2024);
   - наши же абляции показывают, что training-free он **не работает** (recall/tone/thread/code — везде null).
   Его нельзя оставлять как основной вклад.

2. **Семантическое дерево / кластеризация / residual / two-stage — не новы**
   (RAPTOR, CLAG, EverMemOS, IVF). Механизмы опубликованы, местами дословно.

3. **Что потенциально остаётся (узко):**
   - **Эмпирический/абляционный вклад**: строгое разложение «что реально работает»
     в hidden-state памяти для кода — центрирование даёт прирост, а не инъекция;
     выбор слоя задачно-зависим; файловая группировка слепа к grab-bag utils,
     семантическая — лечит. Ценность = rigor + честные nulls. **Но сейчас n=1,
     5–10 проб — для публикации нужны повторы/сиды, ДИ, несколько бенчмарков.**
   - **Контр-результат к Memory³/Memorizing Transformers**: training-free K/V-инъекция
     в frozen модель не даёт выигрыша (они получали плюс ТОЛЬКО с обучением).
     Узкий, но полезный negative result.
   - **Hidden-state-only**: память = собственные активации frozen-модели, без
     внешнего энкодера и без LLM-саммари (в отличие от RAPTOR/FeatLens). Слабое
     отличие, само по себе не тянет на вклад.

## Варианты переформулировки статьи

- **A. Эмпирическая статья «What matters in hidden-state memory for code»** —
  контролируемые абляции + честные nulls. Требует: статистика (повторы, ДИ,
  значимость), ≥2–3 бенчмарка (наш taskq + RepoBench/CrossCodeEval/DevEval),
  сравнение с опубликованными baselines (RAPTOR, CLAG, FeatLens, обычный RAG).
- **B. Negative-result / reproduction study** — «training-free latent K/V memory
  does not help: a controlled study». Честно, но узко; нужен сильный experimental
  design и сравнение с trained-вариантом.
- **C. Инженерный/системный вклад** — если собрать working system с измеримым
  преимуществом на реальном repo-level бенчмарке против SOTA. Сейчас нет.

**Рекомендация:** не подавать текущий framing. Сначала (1) статистика и внешние
бенчмарки, (2) позиционирование относительно RAPTOR/CLAG/EverMemOS/Memorizing
Transformers/Memory³/FeatLens, (3) выбрать A или B. Механизмы цитировать как
известные; вклад — в эмпирическом разложении и негативных результатах.

## Ключевые ссылки
- RAPTOR: https://arxiv.org/abs/2401.18059 (ICLR 2024)
- Memorizing Transformers: https://arxiv.org/abs/2203.08913 (ICLR 2022)
- Memory³: https://arxiv.org/abs/2407.01178 (JML 2024)
- Product Quantization / IVF residual: Jégou et al., TPAMI 2011; Faiss https://arxiv.org/abs/2401.08281
- Anisotropy: Ethayarajh https://aclanthology.org/D19-1006 ; Mu & Viswanath ICLR 2018 ; Bis et al. https://aclanthology.org/2021.naacl-main.403
- CLAG: https://aclanthology.org/2026.findings-acl.824 (ACL Findings 2026)
- EverMemOS: https://arxiv.org/abs/2601.02163 (ACL 2026)
- FeatLens: https://arxiv.org/abs/2609.26480
- Hydra: https://arxiv.org/abs/2602.11671
- MEMCoder: https://github.com/THU-Agent/MEMCoder
- Repo-level RACG survey: https://arxiv.org/abs/2510.04905
