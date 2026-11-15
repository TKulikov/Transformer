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
2. **Абляция ПРОВЕДЕНА (living_memory_eval) — УТОЧНЕНИЕ (важно, не confundать
   два механизма):**
   - **Живая память темы ДЛЯ ЭМБЕДДИНГОВ = бегущий центроид банка
     (центрирование, deviation scoring).** По докстрингу RagBank это
     «gist-conditioning for free»: и ключи, и запросы = отклонение от
     эволюционирующего среднего. ЭТО load-bearing — v3 (центрированный RAG)
     победитель всех тестов, факториал: центрирование +0.041 recall.
   - **Отдельный TTL слот-банк + явный gist (q+λ·gist) — ИЗБЫТОЧЕН поверх
     центрирования.** Абляция: код nobank=static=living=9/10 (банк/gist/TTL
     ничего не добавляют); диалог v3_nomem (центрирование БЕЗ слот-банка)
     confusion 0.000 лучший, добавление слот-банка вредит, явный gist
     монотонно ухудшает (confusion 0→0.429). Потому что центрирование УЖЕ
     кондиционирует темой бесплатно.
   - Исправленный вывод: ядро — живая тема-центроид в эмбеддингах (работает);
     отдельная TTL-слот-gist машинерия не нужна (дублирует центрирование).
     Прежняя формулировка «живая память не load-bearing» была ОШИБКОЙ
     смешения двух механизмов.
3. Комбинаторная новизна: центрирование/mean-subtraction известны (анизотропия,
   Mu & Viswanath 2018 и др.); специфичен инсайт «бегущий центроид банка = живая
   тема, deviation = неявный gist, явный TTL-gist избыточен» + эмпирическое
   разложение. Слабая, но честная новизна; требует статистики и бенчмарков.

**Итог для статьи:** ядро — живая тема-центроид в эмбеддингах (центрированный
RAG v3); явный TTL-gist избыточен. Честная история: «живая тема бесплатно
через центроид банка; латентные K/V/additive и явный TTL-gist не дают прироста
(иногда вредят); активный компонент — центрирование». Требует статистики
(повторы/сиды/ДИ) и нескольких бенчмарков перед подачей.

## ОБНОВЛЕНИЕ: баг маски + первый положительный сигнал (encoding-contextualization)

Найден баг в `kvinject._extend_mask`: при multi-token prefill (mask=None) SDPA
применял is_causal к расширенной длине ключей и маскировал memory-колонки как
«будущие» → K/V-инжект был ИНЕРТЕН при кодировании (cos эмбеддингов =1.0000
несмотря на 115 инъекций). Работал только в decode (q=1). Исправлено (`68add5b`):
явная causal-маска + видимые memory-колонки для prefill.

Следствия:
- Прежние null-результаты K/V→кодирование (memctx) — ЛОЖНООТРИЦАТЕЛЬНЫЕ.
- Прежние K/V→генерация (диалог) результаты тоже могли быть искажены (память
  была только в decode, не в prompt-prefill) — требуют перепроверки с фиксом.

После фикса memctx_eval (taskq_large, semtree mean/tau=0.6):
| Условие | Reuse | Hits | Def hits |
|---|---|---|---|
| standard | 9 | 7 | 4 |
| memctx_keys | 9 | 8 | 5 |
| memctx_both | 8 | 8 | 5 |

**Первый положительный сигнал живой памяти**: контекстуализация КЛЮЧЕЙ живой
памятью при кодировании улучшает retrieval (hits +1, def-hits +1). Это именно
предложение автора (эмбеддинг рождается в контексте живого TTL-банка) — и оно
НЕ повторено в prior art. НО: n=1, 5 задач, различия в 1 исход, reuse на
потолке (9/10) — сигнал слабый, нужны повторы/сиды/ДИ + более чувствительный
тест (диалог, сложный код). memctx_keys лучше memctx_both (контекстуализация
запроса слегка снижает reuse). Это кандидат на novel-механизм с доказанным
преимуществом — после статистического подтверждения.

## VERIFICATION (memctx_verify, 8 сидов + диалог с фиксом): сигнал НЕ подтвердился

Часть A (код, 8 сидов, парное сравнение, shuffle порядка ингестии):
- standard mean: reuse 8.0, hits 7.75, def-hits 4.75;
- memctx_keys mean: reuse 7.88, hits 8.0, def-hits 4.75;
- **hits Δ: +0.25, t=1.528, df=7 (крит 2.365) → НЕ значимо** (2/8 сидов +1, 6/8 без изменений);
- **def-hits Δ: 0.0 на ВСЕХ 8 сидах → эффект исчез** (прежний +1 — fluk одного сида);
- reuse Δ: −0.125, t=−1.0 → слегка отрицательно, не значимо.
- Важно: сам standard varies по порядку (hits 7–8, reuse 8); прежние «9/10» были
  order-specific. Одиночные сиды вводили в заблуждение.

Часть B (диалог thread, С ФИКСОМ маски):
| Условие | recall | confusion |
|---|---|---|
| v3_clean_last | 0.786 | **0.000** |
| additive_v1 | 0.643 | 0.143 |
| kv_v4 (фикс) | 0.738 | **0.286** |
K/V с фиксом стал активнее в prefill: recall вырос (0.548→0.738), но confusion
ТОЖЕ вырос (0.143→0.286). K/V по-прежнему ВРЕДИТ нити; чистый v3 (confusion 0.000)
лучший. Вердикт по K/V-генерации не изменился по существу.

**ИТОГ после строгой проверки:** novel-кандидат (живая TTL-память во всех формах:
gist-запрос, K/V-генерация, K/V-кодирование) НЕ даёт робастного преимущества.
Положительный сигнал memctx был шумом одного сида. Устойчивый результат:
чистый центрированный RAG (v3) — рабочая лошадка; центрирование — активный
компонент. Честный вклад статьи — эмпирический/негативный + инсайт о центроиде,
а не новый механизм. Методологический урок: n=1/order-specific результаты
обманчивы, нужна статистика.

## МЕХАНИКА НАЙДЕНА: memctx ≡ диверсификация (memctx_diverse, 8 порядков)

Разбор выигрыша memctx на исходном порядке (memctx_mechanism): единственный
gain — retry_backoff. standard top-6 = backoff_delay + 5 КЛОНОВ timeout_ms;
retry погребён. memctx сдвинул ключ retry на cos 0.9987 (тот же кластер) —
случайный tie-break, поднявший retry над клонами. Корень — FLOODING КЛОНАМИ.

Робастный механизм — явная диверсификация MMR (lam=0.5), 8 порядков:
| Комбинация | hits | def-hits |
|---|---|---|
| std_plain | 7.62 | 4.62 |
| std_diverse | 8.0 | 4.75 |
| mem_plain | 8.0 | 4.75 |
| mem_diverse | 8.0 | 4.75 |

- **diverse−plain (hits): +0.375, t=2.049 (df=7), 3+/0−/5=** — робастно (НИКОГДА
  не хуже, иногда лучше); маргинально значимо (p≈0.08 двусторонний, ≈0.04
  односторонний), эффект мал (потолок 8/10).
- **memctx−std diverse (hits): 0.0 на ВСЕХ 8 порядках** — memctx НЕ добавляет
  ничего поверх диверсификации.
- std_diverse = mem_plain = mem_diverse ПОКАЗАТЕЛЬНО идентичны на каждом порядке.

**ВЫВОД:** выигрыш «живой контекстуализации» был ХРУПКИМ ПРОКСИ диверсификации:
она случайно разбивала clone-dominated ранжирование. MMR делает то же самое
НАДЁЖНО и ДЕШЁВО (без K/V-инжекта при кодировании). Искомый робастный механизм —
**диверсификация против flooding клонами** в семантическом дереве. Интуиция
автора («механика должна дать результат») подтвердилась: механика найдена и
даёт робастный (never-hurts) результат. Новизна: MMR классический (Carbonell &
Goldstein 1998); специфичен эмпирический инсайт «clone flooding — бутылочное
горлышко; контекстуализация ≡ диверсификация».

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
