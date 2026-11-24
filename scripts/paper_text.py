"""Text of the unified scientific article; empirical values are computed from archived runs."""

from __future__ import annotations

import report_data as RD

TITLE = "Живая латентная память и текстовый retrieval в диалоговых LLM: компонентный анализ без дообучения"
EN_TITLE = "Living Latent Memory and Text Retrieval in Dialogue LLMs: A Training-Free Component Analysis"
KEYWORDS = (
    "большие языковые модели; диалоговая память; retrieval-augmented generation; "
    "латентная память; K/V-инъекция; центрирование эмбеддингов; Qwen3.5"
)
EN_KEYWORDS = (
    "large language models; dialogue memory; retrieval-augmented generation; "
    "latent memory; K/V injection; embedding centering; Qwen3.5"
)
ACCESS_DATE = "20.11.2026"
REPOSITORY_URL = "https://github.com/TKulikov/Transformer.git"


def _ref(description: str, url: str) -> str:
    """Format one bibliography entry with its source and access date."""
    return f"{description} — URL: {url} (дата обращения: {ACCESS_DATE})."


REFERENCES = [
    _ref("Xiao G. Efficient Streaming Language Models with Attention Sinks / G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis // ICLR. — 2024.", "https://arxiv.org/abs/2309.17453"),
    _ref("Zhang Z. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models / Z. Zhang et al. // NeurIPS. — 2023.", "https://arxiv.org/abs/2306.14048"),
    _ref("Li Y. SnapKV: LLM Knows What You are Looking for Before Generation / Y. Li et al. // NeurIPS. — 2024.", "https://arxiv.org/abs/2404.14469"),
    _ref("Cai Z. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling / Z. Cai et al. — 2024.", "https://arxiv.org/abs/2406.02069"),
    _ref("Liu Z. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache / Z. Liu et al. // ICML. — 2024.", "https://arxiv.org/abs/2402.02750"),
    _ref("Tang J. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference / J. Tang et al. // ICML. — 2024.", "https://arxiv.org/abs/2406.10774"),
    _ref("Lewis P. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks / P. Lewis et al. // NeurIPS. — 2020.", "https://arxiv.org/abs/2005.11401"),
    _ref("Jiang H. LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression / H. Jiang et al. // ACL. — 2024.", "https://arxiv.org/abs/2310.06839"),
    _ref("Packer C. MemGPT: Towards LLMs as Operating Systems / C. Packer et al. — 2023.", "https://arxiv.org/abs/2310.08560"),
    _ref("Zhong W. MemoryBank: Enhancing Large Language Models with Long-Term Memory / W. Zhong et al. — 2023.", "https://arxiv.org/abs/2305.10250"),
    _ref("Xu W. A-MEM: Agentic Memory for LLM Agents / W. Xu et al. // NeurIPS. — 2025.", "https://arxiv.org/abs/2502.12110"),
    _ref("Chhikara P. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory / P. Chhikara et al. — 2025.", "https://arxiv.org/abs/2504.19413"),
    _ref("Wu Y. Memorizing Transformers / Y. Wu et al. // ICLR. — 2022.", "https://arxiv.org/abs/2203.08913"),
    _ref("Wang W. Augmenting Language Models with Long-Term Memory / W. Wang et al. — 2023.", "https://arxiv.org/abs/2306.07174"),
    _ref("Bulatov A. Recurrent Memory Transformer / A. Bulatov, Y. Kuratov, M. Burtsev // NeurIPS. — 2022.", "https://arxiv.org/abs/2207.06881"),
    _ref("Mohtashami A. Landmark Attention: Random-Access Infinite Context Length for Transformers / A. Mohtashami, M. Jaggi // NeurIPS. — 2023.", "https://arxiv.org/abs/2305.16300"),
    _ref("Munkhdalai T. Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention / T. Munkhdalai, M. Faruqui, S. Gopal. — 2024.", "https://arxiv.org/abs/2404.07143"),
    _ref("Berges V.-P. Memory Layers at Scale / V.-P. Berges et al. — 2024.", "https://arxiv.org/abs/2412.09764"),
    _ref("Lample G. Large Memory Layers with Product Keys / G. Lample et al. // NeurIPS. — 2019.", "https://arxiv.org/abs/1907.05242"),
    _ref("Liu N. F. Lost in the Middle: How Language Models Use Long Contexts / N. F. Liu et al. // TACL. — 2024.", "https://arxiv.org/abs/2307.03172"),
    _ref("Gu A. Mamba: Linear-Time Sequence Modeling with Selective State Spaces / A. Gu, T. Dao. — 2023.", "https://arxiv.org/abs/2312.00752"),
    _ref("Yang S. Gated Delta Networks: Improving Mamba2 with Delta Rule / S. Yang et al. // ICLR. — 2025.", "https://arxiv.org/abs/2412.06464"),
    _ref("StreamingLLM: official implementation [Электронный ресурс] // GitHub.", "https://github.com/mit-han-lab/streaming-llm"),
    _ref("H2O: official implementation [Электронный ресурс] // GitHub.", "https://github.com/FMInference/H2O"),
    _ref("SnapKV: official implementation [Электронный ресурс] // GitHub.", "https://github.com/FasterDecoding/SnapKV"),
    _ref("KVCache-Factory: official implementation of PyramidKV [Электронный ресурс] // GitHub.", "https://github.com/Zefan-Cai/KVCache-Factory"),
    _ref("KIVI: official implementation [Электронный ресурс] // GitHub.", "https://github.com/jy-yuan/KIVI"),
    _ref("Quest: official implementation [Электронный ресурс] // GitHub.", "https://github.com/mit-han-lab/Quest"),
    _ref("Qwen3.5-4B: model card [Электронный ресурс] // Hugging Face.", "https://huggingface.co/Qwen/Qwen3.5-4B"),
    _ref("Yang W. Memory3: Language Modeling with Explicit Memory / W. Yang et al. — 2024.", "https://arxiv.org/abs/2407.01178"),
    _ref("Mu J. All-but-the-Top: Simple and Effective Postprocessing for Word Representations / J. Mu, P. Viswanath // ICLR. — 2018.", "https://arxiv.org/abs/1702.01417"),
    _ref("Ethayarajh K. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings / K. Ethayarajh // EMNLP. — 2019.", "https://aclanthology.org/D19-1006"),
    _ref("Wang Y. M+: Extending MemoryLLM with Scalable Long-Term Memory / Y. Wang et al. // ICML. — 2025.", "https://proceedings.mlr.press/v267/wang25au.html"),
    _ref("Yan R. MemDefrag: Latent Memory Defragmentation for Large Language Models / R. Yan, Z. Mao, Y. Guo. — 2026.", "https://arxiv.org/abs/2607.05969"),
]


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _integer(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _math(latex: str) -> str:
    return f"$$\n{latex}\n$$"


def abstract(q: dict) -> str:
    """Return the Russian abstract from the battery, core, and RAG archives."""
    core = RD.core_data()["metrics"]
    rag = RD.rag_data()
    dynamic = core["v4_kv_mid_dyn"]
    fixed = core["v4_kv_mid"]
    pyramid = core["pyramidkv_128"]
    best_pure = max(rag["metrics"][name]["recall_mean"] for name in RD.RAG_PURE_ORDER[1:7])
    best_hybrid = max(row["dual_recall"] for row in rag["pair_rows"])
    speed = (1 - dynamic["e2e_latency_median_s"] / pyramid["e2e_latency_median_s"]) * 100
    vram = (1 - dynamic["peak"] / pyramid["peak"]) * 100
    return (
        "Предложена training-free система живой диалоговой памяти: изменяемый банк скрытых состояний "
        "поддерживает EMA-слияние, TTL-забывание и динамическую подачу векторов в качестве дополнительных "
        "беспозиционных K/V в средние attention-слои замороженной модели. Система исследуется отдельно от "
        "текстового RAG и в сочетании с ним. На пяти main-диалогах Qwen3.5-4B динамическая инъекция получила "
        f"recall {_number(dynamic['recall_mean'])} против {_number(fixed['recall_mean'])} у фиксированной k32 "
        f"(p=0,027 после Холма) и {_number(pyramid['recall_mean'])} у PyramidKV. Относительно PyramidKV она "
        f"быстрее на {_number(speed, 1)}% и использует на {_number(vram, 1)}% меньше peak allocated VRAM, "
        "однако статистический паритет по качеству не установлен. На четырёх main-диалогах лучший чистый "
        f"RAG достиг {_number(best_pure)}, тогда как лучший гибрид с фиксированной K/V-инъекцией — "
        f"{_number(best_hybrid)}. Центрирование last-ключей улучшило RAG с 0,489 до 0,710 на всех четырёх "
        "диалогах. Таким образом, самостоятельный латентный канал демонстрирует работоспособность и ресурсные "
        "преимущества, но добавление K/V к текстовому retrieval может ухудшать качество. Работа одновременно "
        "описывает механизм, его сильные стороны и границы применимости; main-корпус остаётся незавершённым."
    )


def en_abstract(q: dict) -> str:
    """Return the English mirror of the current Russian abstract."""
    core = RD.core_data()["metrics"]
    rag = RD.rag_data()
    dynamic = core["v4_kv_mid_dyn"]
    fixed = core["v4_kv_mid"]
    pyramid = core["pyramidkv_128"]
    best_pure = max(rag["metrics"][name]["recall_mean"] for name in RD.RAG_PURE_ORDER[1:7])
    best_hybrid = max(row["dual_recall"] for row in rag["pair_rows"])
    return (
        "We propose a training-free living-memory system for dialogue: a mutable bank of hidden states uses "
        "EMA merging, TTL-based forgetting, and dynamically gated injection as position-free K/V pairs into "
        "middle attention layers of a frozen model. The mechanism is evaluated alone and together with text "
        f"RAG. On five main dialogues, dynamic injection obtains recall {dynamic['recall_mean']:.3f} versus "
        f"{fixed['recall_mean']:.3f} for fixed k32 (Holm-adjusted p=0.027) and {pyramid['recall_mean']:.3f} "
        "for PyramidKV. It reduces latency and peak allocated VRAM relative to PyramidKV, but statistical "
        "parity in quality is not established. On four main dialogues, the best RAG-only configuration reaches "
        f"recall {best_pure:.3f}, whereas the best fixed-K/V hybrid reaches {best_hybrid:.3f}. Last-layer "
        "centering improves RAG from 0.489 to 0.710 on all four dialogues. The standalone latent channel is "
        "functional and resource-efficient, while adding K/V to text retrieval can harm quality. The report "
        "covers both strengths and weaknesses; the main corpus remains incomplete."
    )


def sections(q: dict) -> list[tuple[str, list[str]]]:
    """Build the unified article sections with separately delimited display formulas."""
    core_data = RD.core_data()
    rag_data = RD.rag_data()
    core = core_data["metrics"]
    rag = rag_data["metrics"]
    ref = core["classic_kv_full"]
    stream = core["sparse_sliding_128"]
    topk = core["sparse_topk_128"]
    snap = core["snapkv_128"]
    pyramid = core["pyramidkv_128"]
    fixed = core["v4_kv_mid"]
    dynamic = core["v4_kv_mid_dyn"]
    core_p = core_data["pvalues"]
    center_p = rag_data["centering_pvalues"]
    speed = (1 - dynamic["e2e_latency_median_s"] / pyramid["e2e_latency_median_s"]) * 100
    vram = (1 - dynamic["peak"] / pyramid["peak"]) * 100
    prompt_factor = pyramid["prompt_tokens"] / dynamic["prompt_tokens"]
    attention_factor = dynamic["cache_slots"] / pyramid["cache_slots"]
    best_hybrid = max(rag_data["pair_rows"], key=lambda row: row["dual_recall"])
    return [
        ("ВВЕДЕНИЕ", [
            "Продолжительный диалог требует не только локальной связности, но и доступа к точным сведениям из "
            "далёкого прошлого. Полная история сохраняет исходные данные, однако увеличивает длину входа и "
            "KV-кэш. Сжатие кэша уменьшает число доступных состояний, но не восстанавливает удалённый факт. "
            "Текстовый retrieval возвращает дословные записи в промпт, тогда как латентная память способна "
            "влиять на вычисления без декодирования сохранённого состояния обратно в текст.",
            "В работе исследуется живая память, понимаемая как изменяемый банк активаций замороженной модели. "
            "Запись сопровождается слиянием похожих состояний и забыванием; на каждом шаге текущее скрытое "
            "состояние определяет число подключаемых векторов. Выбранные активации проецируются собственными "
            "матрицами слоя в дополнительные ключи и значения attention. Текстовый канал при этом может быть "
            "отключён, что позволяет измерять самостоятельный механизм без RAG.",
            "Цель — описать предложенную training-free конструкцию и установить, какие компоненты действительно "
            "отвечают за качество и стоимость. RQ1: как слой и центрирование влияют на текстовый retrieval? "
            "RQ2: насколько живая K/V-память работает без текстового канала и что меняет динамический гейт? "
            "RQ3: улучшает ли фиксированная K/V-инъекция соответствующий текстовый RAG? RQ4: какие сильные и "
            "слабые стороны воспроизводятся за пределами основного фактического теста?",
            "Вклад состоит из трёх частей. Во-первых, предложена конкретная комбинация общего изменяемого банка "
            "скрытых состояний, TTL/EMA-динамики, динамической активации и multi-layer K/V-инъекции без обучения. "
            "Во-вторых, выполнены раздельные абляции текстового, латентного и гибридного каналов. В-третьих, "
            "приведены как положительные, так и отрицательные результаты, включая случаи, где ранний вывод "
            "не сохранился после добавления новых сценариев. Новизна отдельных общих принципов не утверждается: "
            "их связь с литературой явно рассматривается ниже.",
        ]),
        ("1 ОБЗОР СУЩЕСТВУЮЩИХ РЕШЕНИЙ", [
            "1.1 Сжатие и выбор KV-состояний. StreamingLLM сохраняет attention sinks и недавние токены [1]; "
            "H2O оставляет heavy hitters по накопленному вниманию [2]. SnapKV отбирает состояния префикса по "
            "observation-окну [3], а PyramidKV дополнительно распределяет бюджет неравномерно между слоями [4]. "
            "Эти методы сравниваются численно в едином цикле декодирования. KIVI квантует K и V [5], а Quest "
            "выбирает релевантные страницы кэша во время attention [6]; оба метода остаются качественными "
            "аналогами, поскольку честное сравнение латентности требует соответствующих оптимизированных ядер.",
            "1.2 Текстовая и агентная память. RAG возвращает найденные фрагменты внешнего источника во вход "
            "модели [7]. LongLLMLingua вместо этого сжимает уже имеющийся промпт с учётом вопроса [8]. MemGPT "
            "организует иерархию хранения и функции управления [9]; MemoryBank добавляет профиль пользователя и "
            "забывание [10]; A-MEM связывает и изменяет структурированные заметки [11]; Mem0 извлекает и "
            "консолидирует факты, включая графовое представление [12]. Наш текстовый банк проще этих систем и "
            "не выполняет отдельных LLM-вызовов для извлечения или редактирования фактов.",
            "1.3 Латентная K/V-память. Memorizing Transformers сохраняют K/V прошлых сегментов во внешней "
            "пополняемой памяти и выполняют kNN-augmented attention [13]. Memory³ также использует явную "
            "память в форме разреженных attention K/V [30]. LongMem вводит residual side-network [14], RMT — "
            "memory-токены между сегментами [15], Landmark Attention — обучаемые якоря [16], Infini-attention — "
            "сжатое состояние и линейное attention [17]. Memory Layers используют разреженные обучаемые таблицы "
            "памяти [18, 19]. Эти работы показывают, что общий принцип латентной памяти и её подключения к "
            "attention не является новым.",
            "M+ и MemoryLLM поддерживают самообновляемый латентный пул скрытых состояний [33], а MemDefrag "
            "консолидирует per-layer скрытые состояния и удаляет нерелевантные фрагменты [34]. Отличие нашей "
            "системы — не само существование памяти, а конкретная training-free связка: общий банк исходных "
            "средних активаций, TTL и EMA, динамический косинусный гейт и повторное использование замороженных "
            "$W_K/W_V$ сразу в нескольких full-attention слоях. Мы не утверждаем абсолютную новизну каждого "
            "компонента и рассматриваем новизну комбинации как кандидатную, а не исчерпывающе доказанную.",
            "1.4 Геометрия эмбеддингов. Центрирование и удаление общих направлений являются известными приёмами "
            "борьбы с анизотропией представлений [31, 32]. Наш running centroid не следует позиционировать как "
            "новый алгоритм центрирования. Предмет эмпирического интереса — его роль как неявно обновляемой "
            "темы диалога и отличие этого бесплатного gist-эффекта от отдельного TTL-gist-банка.",
            "1.5 Ограничения длинного контекста и гибридной модели. Даже доступный полный контекст не гарантирует "
            "одинакового использования сведений в разных позициях [20]. Qwen3.5 сочетает линейные и full-attention "
            "слои; связанные модели состояния включают Mamba и Gated DeltaNet [21, 22]. Поэтому выводы о слоях и "
            "инъекции нельзя автоматически переносить на чисто attention-архитектуры. Таблица 1 разделяет аналоги "
            "по уровню вмешательства и роли в настоящей работе.",
        ]),
        ("2 МАТЕРИАЛЫ И МЕТОДЫ", [
            "2.1 Общая архитектура. Система содержит независимые текстовый и латентный каналы. Текстовый банк "
            "хранит дословные пользовательские реплики и их эмбеддинги. Латентный банк хранит скрытые состояния "
            "средних full-attention слоёв. В режиме RAG-only активен только первый канал; в inject-only — только "
            "второй; в двухканальном режиме оба действуют одновременно. Веса базовой модели не изменяются.",
            "2.2 Эмбеддинги текстового банка. Для реплики $x$ длиной $T$ токенов извлекаются скрытые состояния "
            "$h_i^{(l)}$ выбранного слоя и усредняются по позициям: $e_l(x)=T^{-1}\\sum_{i=1}^{T}h_i^{(l)}$. "
            "Вариант last использует финальный слой, mid — один средний слой, mean — сначала среднее всех слоёв "
            "кроме embedding, затем среднее по токенам. Реплика кодируется изолированно, не более 128 токенов; "
            "отдельный энкодер не используется.",
            "Raw-контроль нормализует исходный эмбеддинг. Centered-вариант вычитает running mean ранее принятых "
            "реплик и затем выполняет $L_2$-нормирование. Запись фиксируется по среднему, ещё не содержащему "
            "саму новую реплику, и позднее не пересчитывается:",
            _math(r"c_t=\frac{1}{t}\sum_{j=1}^{t}e_j,\qquad u_j=\frac{e_j-c_{j-1}}{\Vert e_j-c_{j-1}\Vert_2},\qquad c_0=0,"),
            "где $t$ — число принятых реплик; $e_j$ — эмбеддинг $j$-й записи; $c_t$ — бегущее среднее; $u_j$ — "
            "сохранённый нормированный ключ. Среднее включает позднее заменённые и вытесненные записи. Запрос "
            "центрируется текущим средним. Поскольку запрос и записи имеют единичную норму, косинусная близость "
            "сводится к скалярному произведению:",
            _math(r"\operatorname{sim}(u,q)=\frac{u\cdot q}{\Vert u\Vert_2\,\Vert q\Vert_2},"),
            "где $u$ и $q$ — ключ записи и запрос. В промпт возвращаются тексты top-4 записей. Реплики, "
            "оканчивающиеся на «?», не записываются; эта эвристика не распознаёт все вопросы. При сходстве не "
            "менее 0,95 выполняется upsert, при переполнении банка на 256 записей вытесняется старейшая запись.",
            "2.3 Живой банк активаций. Банк содержит 512 слотов и является общим для инструментированных слоёв. "
            "Новый вектор сливается с ближайшим занятым слотом при косинусной близости не менее 0,95; занимает "
            "свободный слот при близости к ближайшему не более 0,4 и прохождении гейта активности; иначе "
            "отбрасывается. Слияние выполняется экспоненциальным скользящим средним:",
            _math(r"a_j\leftarrow\lambda a_j+(1-\lambda)h,\qquad \lambda=0{,}75,"),
            "где $a_j$ — хранимый вектор; $h$ — новое скрытое состояние. Новый слот получает TTL 5; извлечение "
            "увеличивает TTL на 2 до потолка 6, слияние поднимает его как минимум до 5, а каждый вызов "
            "инструментированного слоя уменьшает TTL на 1. Слот с неположительным TTL вытесняется. Гейт "
            "активности требует, чтобы сумма хотя бы одного неперекрывающегося 32-мерного блока превышала 0,5.",
            "2.4 Динамический отбор. Сначала CorrectionLayer возвращает до 256 кандидатов по максимальной "
            "косинусной близости слота к токенам текущего скрытого состояния. Для динамической конфигурации "
            "дополнительно вычисляется близость каждого кандидата $a_j$ к среднему текущему состоянию "
            "$\\bar h_t$. Если кандидаты упорядочены исходным retrieval, их число после гейта равно",
            _math(r"n_t=\min\!\left(k_{\max},\sum_{j=1}^{m_t}\mathbb{1}\!\left[\cos(a_j,\bar h_t)\geq\tau\right]\right),\qquad \tau=0{,}5,\quad k_{\max}=32,"),
            "где $m_t$ — число кандидатов; $n_t$ — число инжектируемых векторов. В attention подаются первые "
            "$n_t$ векторов этого упорядоченного списка. Допускается $n_t=0$. Фиксированный k32 вместо этого "
            "использует $n_t=\\min(m_t,32)$ независимо от второго косинусного гейта.",
            "2.5 K/V-инъекция. Если $A_t$ содержит $n_t$ выбранных активаций размерности 2560, дополнительные "
            "ключи и значения вычисляются проекциями текущего слоя и конкатенируются с реальными состояниями:",
            _math(r"K_m=R_K(W_KA_t),\qquad V_m=W_VA_t,\qquad \widetilde K=[K;K_m],\qquad \widetilde V=[V;V_m],"),
            "где $R_K$ — QK-Norm слоя; $W_K,W_V$ — замороженные проекции. RoPE к памяти не применяется. "
            "Расширенное attention записывается как",
            _math(r"O=\operatorname{softmax}\!\left(\frac{Q\widetilde K^{\top}}{\sqrt{d_k}}+B\right)\widetilde V,"),
            "где $B$ — аддитивная маска, дополненная нулевыми столбцами памяти. Причинная маска реальных "
            "токенов сохраняется, включая multi-token prefill. Память видима всем реальным токенам, но не "
            "попадает в последовательный KV-кэш. Инъекция выполняется в слои 11, 15, 19 и 23 — среднюю половину "
            "восьми full-attention слоёв Qwen3.5-4B. Группировка query/KV-голов, сигмоидный гейт и выходная "
            "проекция $W_O$ остаются базовыми.",
            "2.6 Порядок работы. Текстовый retrieval выполняется до добавления текущей реплики в банк; найденные "
            "заметки объединяются с последними десятью парами «пользователь—ответ». Во время prefill и decoding "
            "латентный банк пополняется и одновременно предоставляет выбранные K/V. Поскольку вычисление "
            "эмбеддингов также проходит через инструментированные слои, гибрид способен влиять на собственные "
            "будущие ключи. Эта самоиндукция рассматривается как возможный источник негативного взаимодействия, "
            "но не считается доказанной причиной без отдельного механистического теста.",
            "2.7 Центроид как неявный gist. Running mean текстового банка меняется с историей, поэтому centered "
            "запрос представляет отклонение текущей реплики от накопленной темы. Это отличается от явного "
            "TTL-взвешенного gist, прибавляемого к запросу. Первый механизм встроен в центрирование; второй "
            "тестируется отдельно и не должен смешиваться с ним при интерпретации результатов.",
            "2.8 Модель и вычислительная среда. Используется Qwen/Qwen3.5-4B bf16 [29]: 32 слоя, скрытая "
            "размерность 2560, 16 query-голов и 4 KV-головы, 8 full-attention слоёв. Вычисления выполнены на "
            "AMD RX 7900 XT с 20 GiB VRAM, PyTorch 2.9.1+ROCm 6.3. Генерация greedy, до 32 новых токенов.",
        ]),
        ("3 ЭКСПЕРИМЕНТАЛЬНАЯ УСТАНОВКА", [
            "3.1 Серии экспериментов. Основные выводы опираются на post-fix серии latent-v1, full-core-v1 и "
            "full-rag-v1. Более ранний battery100 используется только как исторический предварительный тест: до "
            "исправления маски латентный канал был инертен в multi-token prefill. Поэтому его прежнее преимущество "
            "двухканальной системы не рассматривается как актуальное доказательство. Состав серий приведён в "
            "таблице 2.",
            "3.2 Main-корпус. Полный main содержит 48 диалогов: 24 комбинации темы, персоны и сценария на длинах "
            "100 и 200 ходов. Текущий срез core включает пять первых 100-ходовых диалогов; RAG — четыре. Каждый "
            "диалог содержит факты, прямые и переформулированные вопросы, multi-hop, temporal и unanswerable-пробы "
            "в lag-зонах 0–5, 6–12, 13–20 и 21+. Сценарии включают baseline, updates, adversarial, distractors и "
            "multifact. Тема, персона и сценарий меняются совместно, поэтому срезы не являются независимыми "
            "факториальными условиями.",
            "3.3 Сравниваемые конфигурации. Core сравнивает Full KV, Sliding, TopK, SnapKV, PyramidKV при среднем "
            "бюджете 128, фиксированную живую память k32 и dynamic. RAG-набор содержит полную матрицу слоёв "
            "last/mid/mean × raw/centered × RAG-only/+K/V, а также bank key и bank deviation. Гибриды используют "
            "фиксированные k32; dynamic в них не применяется. Параметры приведены в таблице 3.",
            "3.4 Протокол латентности. Каждый main-батч выполняет один warmup и пять повторов с ротацией порядка "
            "стратегий. End-to-end время включает prefill, decoding, retrieval и embedding. Отдельно сохраняются "
            "prompt/generated/embed-токены, максимальная длина кэша и torch.max_memory_allocated. Последняя "
            "метрика не включает всю reserved-память долгоживущего процесса. Старый battery100 измерял более "
            "узкий путь генерации и поэтому не смешивается с main-латентностью.",
            "3.5 Метрики качества. Keyword recall равен доле ожидаемых строк, найденных в ответе без учёта регистра:",
            _math(r"R=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\exists k\in y_i:\ \operatorname{contains}(\operatorname{lower}(a_i),k)\right],"),
            "где $N$ — число проб; $y_i$ — допустимые ключевые строки; $a_i$ — ответ. Multi-hop допускает "
            "частичный балл 0,5. Answerable-only исключает unanswerable-пробы. Исходная unanswerable-метрика "
            "завышена: отсутствие forbidden-строки не всегда означает настоящий отказ. Для обновлений основной "
            "показатель — stale_updated_rate, поскольку простое упоминание старого значения не обязательно "
            "является ошибкой.",
            "Дополнительные диагностики включают tone как косинус к репликам банка и repetition rate по словесным "
            "триграммам:",
            _math(r"D=1-\frac{U}{G},"),
            "где $G$ — общее число триграмм; $U$ — число уникальных. Обе метрики не заменяют человеческую "
            "оценку связности или естественности диалога.",
            "3.6 Метрики attention и статистика. Decode-side attention-прокси суммирует среднюю длину кэша по "
            "$L$ full-attention слоям на $S$ шагах:",
            _math(r"A=\sum_{s=1}^{S}\frac{1}{L}\sum_{l=1}^{L}\left|K_{l,s}\right|,\qquad E_A=\frac{A_{\mathrm{full}}}{A_{\mathrm{method}}},"),
            "где $A$ не является прямым замером FLOPs и не описывает квадратичную стоимость prefill. Основные "
            "статистические выводы используют диалог, а не вопрос, как единицу анализа. Доверительные интервалы "
            "получены percentile bootstrap; парные сравнения — paired bootstrap по общим batch_id с поправкой "
            "Холма внутри заранее определённого семейства сравнений. Незначимая разница при малом $n$ не "
            "интерпретируется как доказанная эквивалентность.",
        ]),
        ("4 РЕЗУЛЬТАТЫ", [
            f"4.1 Живая память без текстового RAG. На пяти core-диалогах dynamic получил recall "
            f"{_number(dynamic['recall_mean'])}, k32 — {_number(fixed['recall_mean'])}, PyramidKV — "
            f"{_number(pyramid['recall_mean'])}, SnapKV — {_number(snap['recall_mean'])}, Full KV — "
            f"{_number(ref['recall_mean'])}. Dynamic лучше k32 после поправки Холма "
            f"(p={_number(core_p['v4_kv_mid_dyn vs v4_kv_mid'], 3)}), тогда как различия с PyramidKV и SnapKV "
            "не значимы. При этом численный средний recall PyramidKV выше; поэтому нельзя утверждать ни "
            "превосходство, ни доказанный паритет dynamic с ним.",
            f"Dynamic подключает в среднем {_number(dynamic['slots_per_step'], 2)} слота/шаг против "
            f"{_number(fixed['slots_per_step'], 2)} у k32. Относительно PyramidKV его E2E ниже на "
            f"{_number(speed, 1)}%, средний peak allocated VRAM ниже на {_number(vram, 1)}%, prompt-объём меньше "
            f"в {_number(prompt_factor, 2)} раза. Attention-прокси, напротив, выше примерно в "
            f"{_number(attention_factor, 2)} раза: ресурсное преимущество относится к измеренным латентности, "
            f"памяти и входу, но не ко всем вычислительным осям. Результаты приведены в таблице 4 и на "
            "рисунках 2–4.",
            f"4.2 Текстовый RAG без инъекций. Лучшие конфигурации — mean centered "
            f"({_number(rag['rag_mean_centered']['recall_mean'])}), last centered "
            f"({_number(rag['rag_last_centered']['recall_mean'])}) и mid centered "
            f"({_number(rag['rag_mid_centered']['recall_mean'])}). Их доверительные интервалы перекрываются; "
            "однозначного победителя нет. Bank key и bank deviation получают лишь "
            f"{_number(rag['rag_bank_key']['recall_mean'])} и {_number(rag['rag_bank_dev']['recall_mean'])}.",
            f"Центрирование last увеличивает recall с {_number(rag['rag_text_only']['recall_mean'])} до "
            f"{_number(rag['rag_last_centered']['recall_mean'])}; сдвиг положителен на всех четырёх диалогах, "
            f"p<0,001 после Холма. Для mid и mean приросты составляют около +0,034 и не значимы "
            f"(p={_number(center_p['rag_mid_centered vs rag_latent_mid'], 3)} и "
            f"{_number(center_p['rag_mean_centered vs rag_mean'], 3)}). Raw mid/mean сильнее raw last, но после "
            "центрирования различия между слоями почти исчезают. Полные данные даны в таблице 5 и на рисунках "
            "5–8.",
            f"4.3 Добавление K/V к RAG. На текущих четырёх диалогах лучший гибрид — {best_hybrid['dual']} с "
            f"recall {_number(best_hybrid['dual_recall'])}, что ниже лучших чистых RAG. Средний сдвиг по шести "
            "парам варьируется от +0,007 до −0,146, а E2E увеличивается примерно на 12–14%. Значимое после "
            "Холма ухудшение получено у last centered (Δ=−0,146, p=0,042); остальные пары не показывают "
            "значимого преимущества. Отрицательные сдвиги сохраняются и без unanswerable-проб. Парные данные "
            "приведены в таблице 6 и на рисунке 9.",
            "Ранний срез из двух диалогов показывал противоположную картину: raw mid/mean-гибриды лидировали. "
            "Добавление health/adversarial и особенно finance/distractors отменило этот вывод. На finance все "
            "шесть гибридов хуже соответствующих чистых каналов. Это демонстрирует сценарную зависимость и "
            "опасность формирования научного вывода по первым благоприятным диалогам.",
            "4.4 Пилотные серии и перенос слоя. В post-fix latent-v1 на четырёх pilot-диалогах rag_latent_mid "
            "получил 0,851 против 0,775 у raw last, а inject-only k32 и PyramidKV имели практически одинаковый "
            "recall 0,533 и 0,532. Это раннее доказательство ресурсного компромисса живой памяти согласуется с "
            "core, но не отменяет более позднего численного отставания dynamic от PyramidKV на main. Преимущество "
            "mid-ключей также не универсально: на thread_coherence last был лучше mid. Поэтому выбор слоя "
            "остаётся задачно-зависимым.",
            "4.5 Дополнительные механизмы. Явный TTL-gist, merge-цель и K/V-контекстуализация не дали робастного "
            "преимущества в специализированных тестах. На thread_coherence чистый centered RAG имел confusion "
            "0,000 против 0,286 у K/V после исправления маски. В goal_dialogue merge ухудшил confusion, в "
            "interlocutor метрика оказалась насыщена, а слепая оценка не выявила консенсуса о преимуществе "
            "живой памяти. Эти результаты не имеют статистической мощности основного corpus и используются как "
            "диагностические, а не как самостоятельные доказательства общего null-эффекта. Сводка дана в таблице 7.",
        ]),
        ("5 ОБСУЖДЕНИЕ", [
            "5.1 Сильные стороны самостоятельной живой памяти. Механизм работает без текстового retrieval и "
            "без обучения, а dynamic улучшает фиксированную k32 при более чем десятикратном сокращении среднего "
            "числа инъекций. По сравнению с PyramidKV он уменьшает измеренную латентность, allocated VRAM и "
            "prompt-объём. Это делает самостоятельный канал содержательным направлением, несмотря на то что "
            "его recall остаётся ниже полного контекста и не превосходит PyramidKV по текущему среднему.",
            "5.2 Слабые стороны гибрида. Добавление тех же латентных состояний к текстовому RAG не эквивалентно "
            "суммированию их полезности. На distractors гибрид особенно слаб, а centered last теряет качество "
            "статистически значимо. Возможные причины включают конкуренцию текстовых и латентных сигналов, "
            "лишние инъекции и самоискажение ключей при prefill. Настоящие данные различают эффект по сценариям, "
            "но не позволяют выбрать одну причину. Динамический гейт в гибридах ещё не тестировался, поэтому "
            "перенос отрицательного результата k32 на dynamic-гибрид не допускается.",
            "5.3 Что работает в текстовом канале. Наиболее убедительный компонентный результат — last centering "
            "относительно собственного raw-контроля. Running centroid можно интерпретировать как обновляемую "
            "тему, относительно которой запрос и записи кодируются отклонением. Явный TTL-gist при этом не нужен "
            "или вредит в проведённых тестах. Однако центрирование само по себе известно; вкладом является его "
            "эмпирическое разделение с K/V и gist-компонентами в конкретной диалоговой системе.",
            "5.4 Новизна и связь с аналогами. Нельзя заявлять новизну общего принципа «латентная память + "
            "инъекция в attention»: он представлен в Memorizing Transformers, Memory³, MemoryLLM/M+ и других "
            "работах. Вместе с тем эти системы не воспроизводят дословно нашу training-free конфигурацию с "
            "общим TTL/EMA-банком исходных скрытых состояний, динамическим активационным гейтом и frozen "
            "проекциями в нескольких средних слоях. Мы представляем её как предложенный новый вариант, а не как "
            "доказанное первое появление всего класса. Научная ценность результата дополнительно состоит в "
            "проверке того, где этот вариант помогает и где ухудшает систему.",
            "5.5 Угрозы валидности. Main-корпус не завершён: core содержит 5/48, RAG — 4/48 диалогов, все "
            "длиной 100 ходов. Малое $n$ ограничивает мощность парных тестов; percentile bootstrap на таком "
            "срезе имеет дискретное и потенциально оптимистичное покрытие. Сценарий конфаундирован темой, "
            "персоной и seed. Keyword recall измеряет восстановление заданных строк, а не всё качество ответа; "
            "unanswerable-метрика завышена. Политики KV реализованы в проекте по публикациям, а не взяты из "
            "официальных оптимизированных реализаций [23–28]. Наконец, выводы получены на одной гибридной "
            "Qwen3.5-4B и одной GPU-конфигурации.",
            "5.6 Следующие эксперименты. Приоритетно завершение общих main-диалогов core и RAG, включая "
            "200-ходовые. Для гибрида требуется отдельная dynamic-версия и механистический тест самоискажения "
            "банка. Для живой памяти важны temporal-пробы и дальние lag-зоны: текущий temporal recall k32 и "
            "dynamic равен нулю. Сравнение с KIVI/Quest требует оптимизированных ядер, а выводы на других "
            "моделях и языках должны быть проверены независимо.",
        ]),
        ("ЗАКЛЮЧЕНИЕ", [
            "Предложена и реализована training-free живая память с изменяемым банком скрытых состояний, "
            "TTL/EMA-динамикой и динамической K/V-инъекцией в средние attention-слои. Её отличие от известных "
            "классов латентной памяти описано без приписывания новизны общему принципу инъекции.",
            f"Без текстового RAG dynamic получил recall {_number(dynamic['recall_mean'])} и превзошёл k32 "
            f"({_number(fixed['recall_mean'])}, p=0,027 после Холма), сократив среднее число инъекций с "
            f"{_number(fixed['slots_per_step'], 2)} до {_number(dynamic['slots_per_step'], 2)}. Относительно "
            f"PyramidKV он быстрее и легче по измеренным ресурсам, но его качество пока численно ниже "
            f"({_number(pyramid['recall_mean'])} у PyramidKV), а паритет не доказан.",
            f"Текстовый centered RAG достиг {_number(rag['rag_mean_centered']['recall_mean'])}; last centering "
            "дал наиболее устойчивый внутренний эффект. В гибридах фиксированная K/V-инъекция общего "
            "преимущества не показала и в среднем ухудшала качество. Таким образом, сильная сторона работы — "
            "работоспособный и ресурсно-эффективный самостоятельный латентный канал, а слабая — его текущее "
            "взаимодействие с текстовым retrieval и temporal-удержанием. Завершение полного корпуса необходимо "
            "для окончательной оценки.",
        ]),
        ("ДОСТУПНОСТЬ ДАННЫХ И ВОСПРОИЗВОДИМОСТЬ", [
            "Первичные результаты сохранены в results/corpus/full-core-v1/batches/ и "
            "results/corpus/full-rag-v1/batches/. Промежуточные отчёты автоматически пересчитываются из "
            "манифестов и JSON. Исторический battery100 находится в "
            "results/qwen35_comparison/qwen35-4b_battery100.json и помечен как pre-fix. Текст статьи "
            "генерируется scripts/paper_text.py; таблицы, рисунки и PDF — scripts/report_data.py, "
            "scripts/paper_figures.py и scripts/build_paper.py. Проверка статьи против кода и архивов "
            "выполняется scripts/verify_paper.py.",
        ]),
    ]
