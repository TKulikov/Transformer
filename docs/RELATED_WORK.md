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

## Уточнение: связка ЦЕЛИКОМ (дерево + латентный слой + свои активации + код)

Вопрос: делал ли кто-то именно комбинацию «семантическое дерево, где ключи =
собственные латентные (mid) активации генератора, вспоминание в том же
пространстве, для кода, training-free»? Прицельный поиск нашёл очень близкое:

| Работа | Что делает | Насколько закрывает нашу связку |
|---|---|---|
| **ActiveMem** (arXiv:2609.33244, сен 2026) | «Dynamic Latent Memory Trees»: иерархическое дерево, инкрементально хранящее латентные абстракции СОБСТВЕННЫХ hidden states модели («current hidden representation = execution state»); бенчмарки включая KodCode, BigCodeBench | **Почти наша связка**: дерево + латентные свои активации + код. Отличие: dependency-aware execution trees для траекторий агента, не residual-кластеры статических функций для reuse |
| **SRSD** (EMNLP Findings 2026, aclanthology 2026.findings-emnlp.40) | Training-free: СОБСТВЕННЫЕ intermediate-layer hidden states как семантический индекс; **prompt-conditioned centering + L2-norm** (та же анизотропия, та же ссылка Ethayarajh 2019!); mid-to-late layer plateau + layer selection; **до 5.42× на project-level CODE generation** | Закрывает «свои mid-активации + центрирование + training-free + код». Отличие: плоский индекс для speculative decoding, не дерево памяти |
| **MemTree** (ICLR 2025, Accenture) | Динамическое дерево памяти: узел = текст + semantic embedding; онлайн-адаптация по сходству эмбеддингов нового и существующего | Онлайн семантическое дерево (текстовые эмбеддинги, не свои hidden-state) |
| **Hierarchical Embedding Fusion for RAG Code** (arXiv:2603.06593, фев 2026) | Иерархическое дерево эмбеддингов кода (chunk→file→module→repo), fusion маленьким трансформером, retrieved векторы → pseudo-tokens в генератор | Дерево кодовых эмбеддингов + инъекция в генератор. Но ВНЕШНИЙ энкодер (Qwen3-Embedding-8B) + ОБУЧАЕМЫЕ fuser/projector |
| **FlashMem** (ACL Findings 2026) | «Intrinsic Latent Memory via Computation Reuse»: backbone как pre-computed encoder, свои last hidden states → латентная память, без auxiliary encoder | Свои активации как память, без внешнего энкодера. Но last-layer, не дерево, не код |
| **Layer by Layer** (ICML, arXiv:2502.02013) | Intermediate layers consistently outperform final на 32 MTEB задачах | Наш вывод «mid лучше last» опубликован широко |
| **KV-Embedding** (arXiv:2601.01046, янв 2026) | Training-free эмбеддинги из internal states decoder-LLM, layer selection; «middle layers — самые дискриминативные» | Свои активации + mid + training-free (общий текст) |

**Честный итог по связке:** комбинация «свои латентные активации + центрирование +
дерево + код + training-free» в целом ЗАКРЫТА недавними работами (ActiveMem —
дерево из своих активаций для кода; SRSD — свои центрированные mid-активации
training-free для кода). Точный пересечение «residual-кластеры статических
функций кода в дереве из собственных mid-эмбеддингов, training-free, для reuse
хелперов» как единая статья, кажется, не занят — но это УЗКИЙ комбинаторный
зазор между очень свежими активными работами (2025–2026), и:
- каждый компонент опубликован, часть дословно (центрирование в SRSD — та же
  формула и та же ссылка);
- наши доказательства слабы (n=1, кастомный testbed, 9/10);
- ActiveMem/SRSD уже показывают жизнеспособность ядра («свои активации как
  латентная память/индекс для кода»).

Чтобы заявить вклад, нужно: (1) показать явное преимущество над
ActiveMem/SRSD/RAPTOR/CLAG на РЕАЛЬНЫХ code-бенчмарках со статистикой;
(2) артикулировать, что именно residual-дерево функций даёт сверх них.
Без этого — комбинаторная новизна, слабая для публикации.

## Разбор по компонентам: TTL-банк + gist «живой темы» + mid + дерево

Вопрос: есть ли у них именно наш механизм — латентный слот-банк, который
определяет общую тему (TTL-weighted gist), обновляется от запросов с TTL
(refresh при извлечении + decay + eviction), консьюмится из СРЕДНИХ слоёв,
и всё это в дереве? Разбор по частям (проверено по текстам статей):

| Компонент нашего механизма | Кто и что делает | Совпадает? |
|---|---|---|
| **TTL: refresh при извлечении + decay + eviction** | **MemoryBank** (Zhong et al., AAAI 2024): кривая Эббингауза, Importance = e^(−Δt/S), S = число извлечений, Δt сбрасывается при recall. **H-MEM** (EACL 2026), LUFY, FOREVER, MemoryOS — forgetting curve | Механизм TTL/decay ИЗВЕСТЕН, но у всех — для **ТЕКСТОВОЙ** памяти (скоринг реплик/саммари), не латентный слот-банк из активаций |
| **Латентный банк из скрытых состояний, mid-слой** | **MemDefrag** (EMNLP 2026): per-layer hidden states как память (MemoryLLM/M+), **средние слои несут tracing-сигнал** (attention density), training-free defragmentation + forgetting. **MemoryLLM/M+** — латентный пул | Mid-layer латентная память есть; но MemDefrag — defrag/forgetting по informativeness, **не TTL-gist**, не дерево; MemoryLLM/M+ — ОБУЧАЕМЫЕ модули |
| **Gist «живой темы» = TTL-взвешенное среднее слотов, condicionирует запрос (q + α·gist)** | Не найдено точного аналога. Ближайшее: FlashMem (last hidden state как summary), topic-aware retrieval | **ЭТОГО ТОЧНОГО МЕХАНИЗМА НЕ НАШЁЛ** — кандидат на узкую новизну |
| **Латентное ДЕРЕВО +遗忘 + слияние похожих** | **ActiveMem** (прочитан целиком): HLMT, узлы = (e_i, parent, children); НО e_i — выход ОБУЧАЕМОГО контроллера (MLP), не raw mid-активации; запрос h_t = **LAST-token** hidden state; retrieval = **жадный спуск** (argmax cos, stop по τ); pruning по **reward-EMA** (S_EMA < τ_del), НЕ TTL; merge похожих siblings (sim > τ_merge); инъекция = soft-prompt prefix; всё ОБУЧАЕТСЯ через GRPO (RL); для траекторий агента | Дерево + merge есть, но: trained (не training-free), reward-EMA (не TTL), last-token (не mid), жадный спуск (не flat), траектории (не статические функции кода). **Нашего TTL-gist-банка НЕТ** |

**Точный вывод:** связка «TTL-взвешенный gist живой темы, кондиционирующий
запрос, из латентного слот-банка средних слоёв, в training-free семантическом
дереве функций кода» — как единый механизм в найденных статьях НЕ встречается.
ActiveMem ближе всех по форме (латентное дерево), но принципиально иной:
обучаемый контроллер, reward-EMA вместо TTL, last-token запрос, жадный спуск.
TTL-decay сам по себе — MemoryBank (текст). Mid-layer латентная память —
MemDefrag (trained, без gist-запроса).

**ВАЖНЫЕ оговорки (честно):**
1. Поиск не исчерпывающий; «gist кондиционирует запрос» может существовать
   под другим именем (topic-aware / context-conditioned query expansion).
2. **Мы САМИ не изолировали пользу TTL-gist**: в факториале rag_lambda=0,
   в content-свипе gamma=0 лучший, в semtree gist вообще не используется.
   То есть novel-кандидат может оказаться ИНЕРТНЫМ — это надо проверить
   абляцией alpha=0 vs alpha>0 в дереве, прежде чем заявлять вклад.
3. Комбинаторная новизна без доказанного преимущества — слабая.

**Конкретный следующий эксперимент:** абляция TTL-gist в семантическом
дереве (alpha=0 / 0.25 / 0.5 / 1.0) на taskq_large + thread-тесте — помогает
 ли gist «живой темы» вообще, и если да, то это наш узкий novel-механизм.

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
- ActiveMem (latent memory trees): https://arxiv.org/abs/2609.33244
- SRSD (own mid-layer states, code): https://aclanthology.org/2026.findings-emnlp.40
- MemTree: ICLR 2025, Accenture
- Hierarchical Embedding Fusion (code): https://arxiv.org/abs/2603.06593
- FlashMem: https://aclanthology.org/2026.findings-acl.230
- Layer by Layer (mid-layer embeddings): https://arxiv.org/abs/2502.02013
- KV-Embedding: https://arxiv.org/abs/2601.01046
- MemoryBank (TTL/Ebbinghaus, текст): https://ojs.aaai.org/index.php/AAAI/article/view/29946 (AAAI 2024)
- MemDefrag (mid-layer latent memory tracing): https://aclanthology.org/2026.emnlp-main.1043
- H-MEM (hierarchical + forgetting): https://aclanthology.org/2026.eacl-long.15
- M+ (latent hidden-state memory): https://arxiv.org/abs/2502.00592
