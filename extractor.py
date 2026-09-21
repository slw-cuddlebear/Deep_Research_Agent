"""
模块 3：信息提取器（Extractor）
输入：搜索结果 → 抓取网页正文
输出：结构化事实（事实 + 原文摘录 + 来源 + 置信度 + 来源可信度）
要点：正文抓取（trafilatura / BeautifulSoup）、LLM 提炼、事实去重与冲突检测。
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import config
import llm as llm_module
import prompts
import search_tools
from events import Timer, log_event
from observability import inherit_context, traced_node
from state import Fact, to_dicts

_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
_URL_RE = re.compile(r"https?://\S+")
_PUNCT_RE = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()\[\]【】\-—_/\\|]+")


# ========== 文本相似度与去重 ==========
def _normalize_claim(text: str) -> str:
    """事实归一化：去标点、去空白、转小写，便于比较"""
    return _PUNCT_RE.sub("", (text or "").lower())


def claim_similarity(a: str, b: str) -> float:
    """两条事实的文本相似度（0~1），基于序列匹配，无需额外模型"""
    na, nb = _normalize_claim(a), _normalize_claim(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # 长度差异过大时先快速排除，避免无谓计算
    if min(len(na), len(nb)) / max(len(na), len(nb)) < 0.4:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _strip_source_noise(text: str) -> str:
    """去掉 URL：域名或路径里的数字（如 page-3）不是事实数据，不能参与冲突判断"""
    return _URL_RE.sub(" ", text or "").strip()


def _mask_numbers(text: str) -> str:
    """把数字替换成占位符后归一化，用于判断两句话是否"说的是同一件事" """
    return _normalize_claim(_NUM_RE.sub("<NUM>", text))


def dedupe_facts(new_facts: list, existing_facts: list, threshold: float = None) -> list:
    """
    事实去重：与已有事实（或本次新提取的事实）高度相似时丢弃。
    相似但不完全相同时，保留来源可信度更高、置信度更高的那条。
    """
    threshold = threshold or config.DEDUP_SIMILARITY
    kept = [dict(f) for f in to_dicts(existing_facts)]

    for fact in to_dicts(new_facts):
        claim = fact.get("claim", "")
        duplicate = False
        for old in kept:
            if claim_similarity(claim, old.get("claim", "")) >= threshold:
                duplicate = True
                winner = max(
                    (old, fact),
                    key=lambda f: (f.get("credibility", 0.5), f.get("confidence", 0.5)),
                )
                if winner is fact:
                    old.update(fact)
                break
        if not duplicate:
            kept.append(dict(fact))

    # 只返回本次新增的部分（已有事实由状态图自身累积，不能重复写回）
    existing_keys = {_normalize_claim(f.get("claim", "")) for f in to_dicts(existing_facts)}
    merged = [f for f in kept if _normalize_claim(f.get("claim", "")) not in existing_keys]
    return merged


def detect_conflicts(facts: list) -> list:
    """
    冲突检测：找出"同一句话、数字却不一样"的事实对，返回 [(id_a, id_b, 数字a, 数字b)]。

    判定条件是严格的——**把数字抹掉之后两句话必须完全一致**，只是数字不同。
    这样才叫矛盾（"营收 100 亿" vs "营收 120 亿"）；而"同一来源的两条不同事实"
    或者"URL 里带 page-3 / page-4"这类噪声不会被误判成冲突。

    这一点很重要：早期版本只要求"文本相似 + 数字不同"，结果把域名里的编号当成了
    互相矛盾的数据，一次运行误报 25 组冲突。
    """
    items = to_dicts(facts)
    conflicts = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            claim_a, claim_b = a.get("claim", ""), b.get("claim", "")

            # 去掉 URL 等噪声：域名/路径里的数字不是事实数据
            text_a, text_b = _strip_source_noise(claim_a), _strip_source_noise(claim_b)
            if not text_a or not text_b:
                continue

            # 数字抹掉后必须完全一致，才算"说的是同一件事"
            if _mask_numbers(text_a) != _mask_numbers(text_b):
                continue

            nums_a = set(_NUM_RE.findall(text_a))
            nums_b = set(_NUM_RE.findall(text_b))
            if nums_a and nums_b and nums_a != nums_b:
                conflicts.append((a.get("id"), b.get("id"), sorted(nums_a), sorted(nums_b)))
    return conflicts


# ========== LLM 提炼 ==========
def extract_facts_from_page(page: dict, question: str, llm=None) -> list:
    """从单个网页正文里提炼事实"""
    content = (page.get("text") or "")[: config.MAX_PAGE_CHARS]
    if len(content) < 120:
        return []

    prompt = prompts.EXTRACTOR_PROMPT.format(
        question=question,
        title=page.get("title") or page.get("url", ""),
        content=content,
    )
    try:
        data = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_EXTRACTOR, llm=llm)
    except Exception:
        return []

    rows = data.get("facts") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []

    credibility = search_tools.score_credibility(page.get("url", ""), page.get("title", ""))
    facts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        claim = str(row.get("claim") or row.get("fact") or "").strip()
        if not claim:
            continue
        try:
            confidence = float(row.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        facts.append(
            Fact(
                claim=claim,
                quote=str(row.get("quote") or row.get("evidence") or "").strip()[:400],
                source_url=page.get("url", ""),
                source_title=page.get("title", ""),
                confidence=min(max(confidence, 0.0), 1.0),
                credibility=credibility,
            ).model_dump()
        )
    return facts


# ========== 节点 ==========
def _select_targets(state: dict) -> list:
    """
    挑出本轮需要抓取的网页：
      - 跳过已抓过的 URL（本地片段按"文档+片段号"判重）
      - 按可信度与相关度排序
      - **来源配额**：给本地知识库保留一定比例的名额，避免被数量占优的网页结果挤光
      - 把"这条结果归属的子问题"挂到 focus 上：提炼时以子问题为焦点，比拿整个研究问题去问更精准
    """
    done_urls = {
        search_tools.result_key(p.get("url", "")) for p in to_dicts(state.get("pages"))
    }
    sub_question_by_id = {
        p.get("id"): p.get("question", "") for p in to_dicts(state.get("plan"))
    }
    fallback = state.get("question", "")

    web, local = [], []
    for raw in to_dicts(state.get("search_results")):
        item = dict(raw)
        url = item.get("url", "")
        if not url or search_tools.result_key(url) in done_urls:
            continue
        item["focus"] = sub_question_by_id.get(item.get("sub_question_id")) or fallback
        score = item.get("score", 0.0) + search_tools.score_credibility(url) * 0.4
        (local if item.get("engine") == "local" else web).append((score, item))

    web.sort(key=lambda pair: pair[0], reverse=True)
    local.sort(key=lambda pair: pair[0], reverse=True)

    limit = config.MAX_PAGES_PER_ROUND

    def key_of(item):
        return search_tools.result_key(item.get("url", ""))

    web_items = [item for _, item in web]
    local_items = [item for _, item in local]

    # 先按融合分数统一竞争名额
    picked = sorted(web_items + local_items,
                    key=lambda it: it.get("score", 0.0), reverse=True)[:limit]

    # 再保证本地知识库的**最低**名额：
    # 本地片段数量天然少于网页，纯按分数排序时容易被整体挤掉。
    # 但只保底，不设硬性比例——Phase 8 消融实验证明，固定 40% 的配额在
    # 语料较小时会白白占掉网页阅读名额（T1 的网页阅读量从 29 篇掉到 19 篇），
    # 而本地引用只贡献了 6%。
    if local_items and config.LOCAL_MIN_SLOTS > 0:
        picked_keys = {key_of(it) for it in picked}
        local_in = len([it for it in picked if it.get("engine") == "local"])
        missing = min(len(local_items), config.LOCAL_MIN_SLOTS) - local_in
        if missing > 0:
            spare = [it for it in local_items if key_of(it) not in picked_keys]
            droppable = sorted([it for it in picked if it.get("engine") != "local"],
                               key=lambda it: it.get("score", 0.0))
            while missing > 0 and spare and droppable:
                picked.remove(droppable.pop(0))
                keep = spare.pop(0)
                picked.append(keep)
                picked_keys.add(key_of(keep))
                missing -= 1

    picked.sort(key=lambda it: it.get("score", 0.0), reverse=True)
    return picked[:limit]


def _local_pages(targets: list) -> tuple:
    """
    把本地知识库片段直接转成 PageContent —— 它们自带正文，不需要"抓取"。
    返回 (本地页面列表, 需要联网抓取的网页目标)。
    """
    from state import PageContent

    local_pages, web_targets = [], []
    for item in targets:
        if item.get("engine") == "local":
            text = item.get("content") or item.get("snippet") or ""
            local_pages.append(
                PageContent(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    text=text,
                    ok=True,
                    chars=len(text),
                    source="local",
                ).model_dump()
            )
        else:
            web_targets.append(item)
    return local_pages, web_targets


@traced_node("extractor")
def extractor_node(state: dict) -> dict:
    """LangGraph 节点：抓正文（本地片段直读）→ LLM 提炼事实 → 去重 → 冲突检测"""
    question = state.get("question", "")
    targets = _select_targets(state)
    if not targets:
        return {
            "events": [log_event(state, "extractor", "没有新的网页需要阅读",
                                 status="warn", iteration=state.get("iteration", 1))],
        }

    events = []
    with Timer() as timer:
        local_pages, web_targets = _local_pages(targets)
        pages = list(local_pages)
        failed = 0
        if web_targets:
            fetched = search_tools.fetch_pages([t["url"] for t in web_targets])
            pages.extend(fetched)
            failed = len([p for p in fetched if not p.get("ok")])
        ok_pages = [p for p in pages if p.get("ok")]

        # 并发调用 LLM 提炼（IO 密集，线程池能明显提速）
        # inherit_context 让子线程继承调用线程的追踪上下文，
        # 这样并发的 LLM 调用也能正确挂在"提取器"这个父 span 下面。
        url_meta = {t["url"]: t for t in targets}
        raw_facts = []
        with ThreadPoolExecutor(max_workers=min(config.MAX_CONCURRENCY, max(len(ok_pages), 1))) as pool:
            futures = {}
            for page in ok_pages:
                meta = url_meta.get(page.get("url"), {})
                # 以"归属子问题"为提炼焦点，拿不到时退回整个研究问题
                focus = meta.get("focus") or question
                futures[pool.submit(inherit_context(extract_facts_from_page), page, focus)] = (page, meta)
            for future in as_completed(futures):
                page, meta = futures[future]
                try:
                    facts = future.result()
                except Exception:
                    facts = []
                for fact in facts:
                    fact["sub_question_id"] = meta.get("sub_question_id", "")
                raw_facts.extend(facts)

    # 丢弃低置信度事实，再做去重
    raw_facts = [f for f in raw_facts if f.get("confidence", 0) >= config.MIN_FACT_CONFIDENCE]
    before = len(raw_facts)
    fresh_facts = dedupe_facts(raw_facts, state.get("facts") or [])

    # 统一编号：延续已有事实的序号
    start = len(to_dicts(state.get("facts")))
    for offset, fact in enumerate(fresh_facts, start=1):
        fact["id"] = f"F{start + offset}"

    all_facts = to_dicts(state.get("facts")) + fresh_facts
    # 冲突只做"报告"不做"改写"：结论交给写作阶段权衡，避免静默删掉可能有价值的信息
    conflicts = detect_conflicts(all_facts)

    stats = dict(state.get("stats") or {})
    stats["pages_fetched"] = stats.get("pages_fetched", 0) + len(ok_pages)
    stats["pages_failed"] = stats.get("pages_failed", 0) + failed
    stats["conflicts"] = conflicts
    stats["facts_total"] = len(all_facts)
    # 供反思决策器判断"再搜一轮还有没有意义"
    stats["facts_gained_last_round"] = len(fresh_facts)

    events.append(
        log_event(
            state, "extractor",
            f"阅读 {len(ok_pages)} 篇网页 → 提炼 {len(fresh_facts)} 条新事实",
            f"抓取失败 {failed} 篇；去重丢弃 {before - len(fresh_facts)} 条；累计 {len(all_facts)} 条",
            elapsed=timer.elapsed, iteration=state.get("iteration", 1),
            pages=len(ok_pages), facts=len(fresh_facts),
        )
    )
    if conflicts:
        events.append(
            log_event(
                state, "extractor", f"检测到 {len(conflicts)} 组互相矛盾的数据",
                "；".join(f"{a} vs {b}" for a, b, _, _ in conflicts[:3]),
                status="warn", iteration=state.get("iteration", 1),
            )
        )

    return {"pages": pages, "facts": fresh_facts, "stats": stats, "events": events}
