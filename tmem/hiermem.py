"""Иерархическая память кода: дерево файл → класс → метод → вызовы.

Заказ 14.11: узел-объявление (заголовок + тело) получает ключ как среднее
активаций модели по блоку и контекстуализированное значение — пул по
токенам чанка в контексте родителя, взятый на ЛАТЕНТНОМ (среднем) слое
смыслов, а не на выходном; от объявления
древовидно идут связи к использованиям функции. Ключи центрируются НЕ
глобальным бегущим средним, а эмбеддингом родителя по дереву: метод —
отклонение от класса, вызов — отклонение от объявления. Это снимает общую
тему ветки и разводит почти одинаковые куски кода (провал плоского банка
на 1305 чанках: top-4 забивался близнецами by_region).

Поиск двухстадийный (отклонения разных веток несравнимы напрямую):
1. объявления ранжируются по абсолютным ключам с глобальным центрированием
   (бегущее среднее, как в ChunkBank) — выбираются top-ветки;
2. внутри ветки запрос проецируется в residual-пространство родителя
   (q − emb_родителя) и ранжирует детей по их отклонениям.
Так top-k расходится по разным веткам (диверсификация), а выдача содержит
и определение, и места использования.

Дерево строится по ast: containment (файл → класс → метод, файл → функция)
плюс usage-рёбра (функция → вызывающие её функции). Вызовы разрешаются по
имени: Name → модульная функция, Attribute (self.x()) → метод того же
класса. На функцию не более max_usages использований (равномерно по файлу).
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field

import torch

from .chunkmem import embed_block_key, embed_chunk
from .latentrag import latent_mean
from .memory import Memory


@dataclass
class CodeNode:
    nid: int
    kind: str                      # file | class | function | usage
    name: str
    text: str
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    key_raw: torch.Tensor | None = None   # абсолютный ключ (mean активаций)
    key_res: torch.Tensor | None = None   # нормированное отклонение от родителя
    val: torch.Tensor | None = None       # контекстуализированное значение
    # компоненты ключа (вариант 3): хранятся для дешёвого rekey(β, γ)
    _qrec: torch.Tensor | None = None     # mid-mean сигнатуры+docstring (запрос)
    _a: torch.Tensor | None = None        # mid-span тела в контексте родителя (latent)
    _content: torch.Tensor | None = None  # изолированный mid-mean тела (содержание)


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(v.norm(), min=1e-8)


def build_code_tree(text: str, max_usages: int = 4) -> list[CodeNode]:
    """Parse a Python file into the containment+usage node tree (no embeddings)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    def add(kind: str, name: str, body: str, parent: int | None) -> CodeNode:
        n = CodeNode(nid=len(nodes), kind=kind, name=name, text=body, parent=parent)
        nodes.append(n)
        if parent is not None:
            nodes[parent].children.append(n.nid)
        return n

    header = "".join(lines[:40])
    file_node = add("file", "<module>", header, None)

    func_nodes: dict[str, CodeNode] = {}      # модульные функции по имени
    class_methods: dict[str, dict[str, CodeNode]] = {}
    func_asts: dict[int, ast.AST] = {}

    def func_source(node) -> str:
        return ast.get_source_segment(text, node) or ""

    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            doc = ast.get_docstring(top) or ""
            cls = add("class", top.name, f"class {top.name}:\n    {doc}\n", file_node.nid)
            methods: dict[str, CodeNode] = {}
            for sub in top.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m = add("function", sub.name, func_source(sub), cls.nid)
                    methods[sub.name] = m
                    func_asts[m.nid] = sub
            class_methods[top.name] = methods
        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f = add("function", top.name, func_source(top), file_node.nid)
            func_nodes[top.name] = f
            func_asts[f.nid] = top

    # usage-рёбра: кто вызывает каждую функцию
    callers: dict[int, list[int]] = {}        # callee nid -> caller nids
    for caller in nodes:
        if caller.kind != "function":
            continue
        body = func_asts.get(caller.nid)
        if body is None:
            continue
        seen: set[int] = set()
        for call in ast.walk(body):
            if not isinstance(call, ast.Call):
                continue
            callee = None
            if isinstance(call.func, ast.Name):
                callee = func_nodes.get(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                # self.method() внутри класса → метод того же класса
                cls_name = None
                for cn, methods in class_methods.items():
                    if caller.name in methods and methods[caller.name].nid == caller.nid:
                        cls_name = cn
                        break
                if cls_name:
                    callee = class_methods[cls_name].get(call.func.attr)
            if callee is not None and callee.nid != caller.nid and callee.nid not in seen:
                seen.add(callee.nid)
                callers.setdefault(callee.nid, []).append(caller.nid)

    for callee_nid, caller_nids in callers.items():
        if len(caller_nids) > max_usages:  # равномерно по файлу, не только первые
            step = len(caller_nids) / max_usages
            caller_nids = [caller_nids[int(i * step)] for i in range(max_usages)]
        for caller_nid in caller_nids:
            caller = nodes[caller_nid]
            add("usage", f"{nodes[callee_nid].name}@{caller.name}", caller.text, callee_nid)
    return nodes


class HierBank:
    """Two-stage hierarchical retrieval over the code tree.

    Stage 1 ranks declarations by globally-centred absolute keys; stage 2
    descends into the top branches and ranks children by parent-relative
    residual keys against the correspondingly projected query.
    """

    def __init__(self, nodes: list[CodeNode], device: torch.device | str = "cpu") -> None:
        self.nodes = nodes
        self.device = torch.device(device)
        self.mean = None
        self._n = 0
        self.stats = {"ingested": 0, "declarations": 0, "usages": 0}

    @torch.no_grad()
    def ingest(self, bundle, max_tokens: int = 256, context_tokens: int = 512) -> None:
        """Embed every node: absolute key, parent-relative residual, span value."""
        for node in self.nodes:  # родители идут раньше детей (порядок построения)
            key, _ = embed_block_key(bundle, node.text, layers="mean", max_tokens=max_tokens)
            parent = self.nodes[node.parent] if node.parent is not None else None
            context = parent.text if parent is not None else ""
            # значение контекстуализируется активациями ЛАТЕНТНОГО (среднего)
            # слоя — там токены сходятся в смыслы (полоса инжекции памяти v4)
            val, _, _ = embed_chunk(bundle, node.text, context=context, mode="span",
                                    max_tokens=max_tokens, context_tokens=context_tokens,
                                    layer="mid")
            node.key_raw = key.float().to(self.device)
            node.val = val.float().to(self.device)
            node.key_res = _norm(node.key_raw - parent.key_raw) if parent is not None \
                else _norm(node.key_raw)
            if node.kind == "function":
                self._n += 1
                m = node.key_raw if self.mean is None else self.mean
                self.mean = (m * (self._n - 1) + node.key_raw) / self._n
                self.stats["declarations"] += 1
            if node.kind == "usage":
                self.stats["usages"] += 1
            self.stats["ingested"] += 1

    def _center(self, v: torch.Tensor) -> torch.Tensor:
        return _norm(v - self.mean) if self.mean is not None else _norm(v)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k_branches: int = 3,
                 k_children: int = 2) -> list[tuple[CodeNode, float, str]]:
        """[(node, score, stage)] — объявления + использования из разных веток."""
        q = query.float().to(self.device)
        decls = [n for n in self.nodes if n.kind == "function" and n.key_raw is not None]
        if not decls:
            return []
        qc = self._center(q)
        scored = sorted(((float(self._center(n.key_raw) @ qc), n) for n in decls),
                        key=lambda t: -t[0])
        out: list[tuple[CodeNode, float, str]] = []
        for s1, branch in scored[:k_branches]:
            out.append((branch, s1, "declaration"))
            kids = [self.nodes[c] for c in branch.children
                    if self.nodes[c].key_res is not None]
            if not kids:
                continue
            q_res = _norm(q - branch.key_raw)
            scored_kids = sorted(((float(k.key_res @ q_res), k) for k in kids),
                                 key=lambda t: -t[0])
            out += [(k, s2, "usage") for s2, k in scored_kids[:k_children]]
        return out


def build_project_tree(root, max_usages: int = 6) -> list[CodeNode]:
    """Дерево по проекту: проект → файл → класс → метод + usage-рёбра.

    В отличие от :func:`build_code_tree` (один файл), обходит каталог *.py,
    строит узлы на каждый файл/класс/функцию и разрешает межфайловые вызовы
    через импорты (from .utils import retry → utils.retry). Usage-рёбра:
    функция → вызывающие её функции. Разрешаются Name-вызовы (импорты и
   .same-module) и self.method() того же класса; прочие attribute-вызовы
    (self._store.x) пропускаются без вывода типов.
    """
    import pathlib

    root = pathlib.Path(root)
    files = sorted(root.rglob("*.py"))
    nodes: list[CodeNode] = []

    def add(kind, name, text, parent):
        n = CodeNode(nid=len(nodes), kind=kind, name=name, text=text, parent=parent)
        nodes.append(n)
        if parent is not None:
            nodes[parent].children.append(n.nid)
        return n

    proj = add("file", "<project>", "project root", None)
    func_nodes: dict[str, CodeNode] = {}
    func_asts: dict[int, ast.AST] = {}
    node_file: dict[int, int] = {}
    node_module: dict[int, str] = {}
    node_class: dict[int, str | None] = {}
    file_imports: dict[int, dict[str, str]] = {}

    for f in files:
        rel = f.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src)
        header = ast.get_docstring(tree) or ""
        fnode = add("file", str(rel), f'"""{header}"""\n', proj.nid)
        node_file[fnode.nid] = fnode.nid
        node_module[fnode.nid] = module

        imports: dict[str, str] = {}
        for imp in ast.walk(tree):
            if isinstance(imp, ast.ImportFrom) and imp.module:
                base = imp.module if imp.level == 0 else imp.module
                for a in imp.names:
                    imports[a.asname or a.name] = f"{base}.{a.name}"
            elif isinstance(imp, ast.Import):
                for a in imp.names:
                    imports[a.asname or a.name.split(".")[0]] = a.name
        file_imports[fnode.nid] = imports

        for top in tree.body:
            if isinstance(top, ast.ClassDef):
                doc = ast.get_docstring(top) or ""
                cls = add("class", top.name, f"class {top.name}:\n    {doc}\n", fnode.nid)
                node_file[cls.nid] = fnode.nid
                node_module[cls.nid] = module
                for sub in top.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m = add("function", sub.name, ast.get_source_segment(src, sub) or "", cls.nid)
                        q = f"{module}.{top.name}.{sub.name}"
                        func_nodes[q] = m
                        func_asts[m.nid] = sub
                        node_file[m.nid] = fnode.nid
                        node_module[m.nid] = module
                        node_class[m.nid] = top.name
            elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = add("function", top.name, ast.get_source_segment(src, top) or "", fnode.nid)
                q = f"{module}.{top.name}"
                func_nodes[q] = fn
                func_asts[fn.nid] = top
                node_file[fn.nid] = fnode.nid
                node_module[fn.nid] = module
                node_class[fn.nid] = None

    callers: dict[int, list[int]] = {}
    for caller in nodes:
        if caller.kind != "function":
            continue
        body = func_asts.get(caller.nid)
        if body is None:
            continue
        fimports = file_imports[node_file[caller.nid]]
        module = node_module[caller.nid]
        seen: set[int] = set()
        for call in ast.walk(body):
            if not isinstance(call, ast.Call):
                continue
            callee = None
            if isinstance(call.func, ast.Name):
                q = fimports.get(call.func.id) or f"{module}.{call.func.id}"
                callee = func_nodes.get(q)
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) \
                    and call.func.value.id == "self" and node_class.get(caller.nid):
                q = f"{module}.{node_class[caller.nid]}.{call.func.attr}"
                callee = func_nodes.get(q)
            if callee is not None and callee.nid != caller.nid and callee.nid not in seen:
                seen.add(callee.nid)
                callers.setdefault(callee.nid, []).append(caller.nid)

    for callee_nid, caller_nids in callers.items():
        if len(caller_nids) > max_usages:
            step = len(caller_nids) / max_usages
            caller_nids = [caller_nids[int(i * step)] for i in range(max_usages)]
        for caller_nid in caller_nids:
            add("usage", f"{nodes[callee_nid].name}@{nodes[caller_nid].name}",
                nodes[caller_nid].text, callee_nid)
    return nodes


def request_proxy(node: CodeNode) -> str:
    """Signature + docstring as the 'task' proxy for a function node."""
    try:
        doc = ast.get_docstring(ast.parse(textwrap.dedent(node.text)))
    except SyntaxError:
        doc = None
    first = node.text.strip().splitlines()[0]
    return first + (" — " + doc if doc else "")


class HierLatentBank:
    """Целевая память для кода: дерево + латентный слой + TTL-gist.

    Комбинирует ортогональные слои:
    * структура дерева (файл→класс→метод→вызовы) из :func:`build_code_tree`
      даёт организацию, диверсификацию по веткам и связи использований;
    * латентная схема эмбеддингов: ключ объявления = norm(q_сигнатура +
      beta·mean_активаций_тела) на ОДНОМ латентном слое; ключ usage/класса =
      латентный mean его текста; центрирование отклонением от родителя;
    * запрос = латентный mean + alpha·TTL-gist слот-банка (живая тема);
    * поиск двухстадийный: объявления по глобально-центрированным ключам →
      спуск к детям по residual-ключам (q − ключ_ветки).

    Ингестия «после ответа»: тело функции — это ответ, сигнатура+docstring —
    запрос; слот-банк наполняется латентными per-token состояниями тел.
    """

    def __init__(self, nodes: list[CodeNode], beta: float = 1.0, gamma: float = 0.0,
                 slot_size: int = 512, device: torch.device | str = "cpu") -> None:
        self.nodes = nodes
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)
        self.slots = None  # создаётся в ingest, когда известен d_model
        self._slot_size = slot_size
        self.file_of: dict[int, int] = {}  # nid узла → nid его файла-модуля
        self.mean = None
        self._n = 0
        self.stats = {"ingested": 0, "declarations": 0, "usages": 0, "gist_used": 0}

    @torch.no_grad()
    def ingest(self, bundle, max_tokens: int = 256) -> None:
        d_model = bundle.hidden_size
        self.slots = Memory(d_model, size=self._slot_size, merge_k=0.25, device=self.device)
        for node in self.nodes:  # родители раньше детей
            parent = self.nodes[node.parent] if node.parent is not None else None
            ctx = parent.text if parent is not None else ""
            if node.kind == "function":
                q_rec = latent_mean(bundle, request_proxy(node), max_tokens)
                a, a_toks, _ = embed_chunk(bundle, node.text, context=ctx, mode="late",
                                           layer="mid", max_tokens=max_tokens)
                content = latent_mean(bundle, node.text, max_tokens)  # вариант 3: содержание
                if a_toks is not None and a_toks.shape[0] > 0:
                    self.slots.update(a_toks.to(self.device), crit=0.95,
                                      max_to_new=1.0, window_crit=-1e9)
                node._qrec, node._a, node._content = q_rec, a, content
                raw = _norm(q_rec + self.beta * a + self.gamma * content)
                self._n += 1
                m = raw if self.mean is None else self.mean
                self.mean = (m * (self._n - 1) + raw) / self._n
                self.stats["declarations"] += 1
            else:
                raw = latent_mean(bundle, node.text, max_tokens)
                if node.kind == "usage":
                    self.stats["usages"] += 1
            node.key_raw = raw.float().to(self.device)
            node.key_res = _norm(node.key_raw - parent.key_raw) if parent is not None \
                else _norm(node.key_raw)
            p = node
            while p.parent is not None and p.kind != "file":
                p = self.nodes[p.parent]
            self.file_of[node.nid] = p.nid if (p.kind == "file" and p.parent is not None) else node.nid
            self.stats["ingested"] += 1

    def rekey(self, beta: float, gamma: float) -> None:
        """Пересобрать ключи из сохранённых компонент (без embed-форвардов).

        key = norm(q_rec + beta·a + gamma·content); пересчитывает бегущее
        среднее и residual-ключи. Позволяет свипать beta/gamma после одной
        ингестии. gamma=0 — базовый ключ (без content-вектора).
        """
        self.beta, self.gamma = beta, gamma
        self.mean = None
        self._n = 0
        for node in self.nodes:  # родители раньше детей
            if node.kind == "function" and node._qrec is not None:
                raw = _norm(node._qrec + beta * node._a + gamma * node._content)
                node.key_raw = raw.to(self.device)
                self._n += 1
                m = raw if self.mean is None else self.mean
                self.mean = (m * (self._n - 1) + raw) / self._n
            parent = self.nodes[node.parent] if node.parent is not None else None
            node.key_res = _norm(node.key_raw - parent.key_raw) if parent is not None \
                else _norm(node.key_raw)

    def _center(self, v: torch.Tensor) -> torch.Tensor:
        return _norm(v - self.mean) if self.mean is not None else _norm(v)

    @staticmethod
    def _merge_rankings(descent: list[tuple[CodeNode, float]],
                        flat: list[tuple[float, CodeNode]],
                        k: int) -> list[tuple[CodeNode, float]]:
        """Union двух ранжирований (спуск + плоское) с нормировкой шкал.

        Path-score спуска и косинус плоского поиска — разные шкалы; каждую
        нормируем в [0, 1] внутри своего списка и берём max по узлу: функция,
        найденная любым из способов, получает кредит. Дедуп по nid.
        """
        def norm_scores(pairs):
            if not pairs:
                return {}
            vals = [s for _, s in pairs]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            return {n.nid: (n, (s - lo) / span) for n, s in pairs}

        d = norm_scores([(n, s) for n, s in descent])
        f = norm_scores([(n, s) for s, n in flat])
        merged: dict[int, tuple[CodeNode, float]] = {}
        for nid, (n, s) in {**d, **f}.items():
            best = max(d.get(nid, (n, -1.0))[1], f.get(nid, (n, -1.0))[1])
            merged[nid] = (n, best)
        return sorted(merged.values(), key=lambda t: -t[1])[:k]

    def _shallow(self, flat: list[tuple[float, CodeNode]],
                 k: int) -> list[tuple[CodeNode, float]]:
        """Минимальное дерево файл→класс + плоский поиск внутри (cluster-then-rank).

        Плоский скор всех функций → группировка по файлу-модулю → top-k файлов
        по лучшему члену → внутри отобранных файлов плоский top-k. Структура
        даёт диверсификацию по файлам, точный поиск делает плоский ранжировщик;
        модульные хелперы не теряются (внутри файла ранжируются все функции).
        """
        files: dict[int, list[tuple[CodeNode, float]]] = {}
        for s, n in flat:
            files.setdefault(self.file_of.get(n.nid, n.nid), []).append((n, s))
        file_best = [(fid, max(s for _, s in members)) for fid, members in files.items()]
        top_files = sorted(file_best, key=lambda t: -t[1])[:k]
        pool = [ns for fid, _ in top_files for ns in files[fid]]
        return sorted(pool, key=lambda t: -t[1])[:k]

    @torch.no_grad()
    def retrieve(self, bundle, query: str, k_branches: int = 4,
                 k_children: int = 2, alpha: float = 0.5,
                 max_tokens: int = 256, max_depth: int = 3,
                 mode: str = "union") -> list[tuple[CodeNode, float, str]]:
        """Двойной поиск: ветка по контекстному эмбеддингу + residual-спуск.

        Запрос ``q`` = латентный mean контекста + alpha·TTL-gist. Beam-спуск
        от корня по дереву: на каждом узле запрос перецентрируется
        ``q_local = norm(q − key_узла)`` и матчит детей по их residual-ключам
        (отклонениям от этого узла) — так вектор каждый раз в нужном
        пространстве. Path-score накапливается вдоль пути; функции
        собираются на любой глубине (top-level или метод). К top-функциям
        добавляются их использования (usage) как контекст.
        """
        q = latent_mean(bundle, query, max_tokens)
        self.slots.get_relevant(q.view(1, 1, -1))
        g = self.slots.ttl_gist()
        if g is not None and alpha > 0:
            q = q + alpha * g.to(q.device)
            self.stats["gist_used"] += 1
        q = q.to(self.device)

        root = self.nodes[0]
        beam: list[tuple[CodeNode, float]] = [(root, 0.0)]
        candidates: dict[int, tuple[CodeNode, float]] = {}
        for _ in range(max_depth):
            nxt: list[tuple[CodeNode, float]] = []
            for node, ps in beam:
                if node.key_raw is None:
                    continue
                q_local = _norm(q - node.key_raw)
                kids = [self.nodes[c] for c in node.children
                        if self.nodes[c].key_res is not None]
                scored = sorted(((float(k.key_res @ q_local), k) for k in kids),
                                key=lambda t: -t[0])
                for s, k in scored[:k_branches]:
                    nps = ps + s
                    if k.kind == "function":
                        prev = candidates.get(k.nid)
                        if prev is None or nps > prev[1]:
                            candidates[k.nid] = (k, nps)
                    nxt.append((k, nps))
            nxt.sort(key=lambda t: -t[1])
            beam = nxt[:k_branches]
            if not beam:
                break

        top = sorted(candidates.values(), key=lambda t: -t[1])[:k_branches]
        # Плоское латентное ранжирование всех объявлений (находит отдельные
        # модульные хелперы, которые жадный beam-спуск отсекает).
        qc = self._center(q)
        decls = [n for n in self.nodes if n.kind == "function" and n.key_raw is not None]
        flat = [(float(self._center(n.key_raw) @ qc), n) for n in decls]
        if mode == "descent":
            selected = top
        elif mode == "flat":
            selected = [(n, s) for s, n in sorted(flat, key=lambda t: -t[0])[:k_branches]]
        elif mode == "shallow":
            selected = self._shallow(flat, k_branches)
        else:  # union: спуск ∪ плоское, нормировка шкал, max по узлу
            selected = self._merge_rankings(top, flat, k_branches)
        out: list[tuple[CodeNode, float, str]] = [(n, s, "declaration") for n, s in selected]
        for func, _ in selected:
            kids = [self.nodes[c] for c in func.children
                    if self.nodes[c].kind == "usage" and self.nodes[c].key_res is not None]
            if not kids:
                continue
            q_local = _norm(q - func.key_raw)
            scored = sorted(((float(k.key_res @ q_local), k) for k in kids),
                            key=lambda t: -t[0])
            out += [(k, s, "usage") for s, k in scored[:k_children]]
        return out
