"""
模块 2：搜索执行器（Searcher）
输入：子问题（含建议检索词）
输出：去重排序后的搜索结果列表（标题 + 摘要 + URL）
要点：多源（Tavily / DuckDuckGo / 本地知识库）、并发搜索、跨轮去重、结果融合排序。

Phase 5 起是**混合搜索**：同一轮里既查网页，也查本地知识库，
再把两路结果归一化到可比的分数区间后合并。
"""
from collections import defaultdict

import config
import llm as llm_module
import local_kb
import prompts
import search_tools
from events import Timer, log_event
from observability import traced_node
from state import to_dicts


def build_query_plan(plan: list, pending_queries: list = None, llm=None) -> list:
    """
    把研究计划翻译成"检索词 → 子问题"的映射列表。
    pending_queries（反思补充词）独享一组，不归属任何子问题。
    """
    tasks = []
    for sub in to_dicts(plan):
        queries = sub.get("search_queries") or []
        if not queries:
            queries = [sub.get("question", "")]
        for query in queries:
            query = str(query).strip()
            if query:
                tasks.append({"query": query, "sub_question_id": sub.get("id", "")})

    for query in pending_queries or []:
        query = str(query).strip()
        if query:
            tasks.append({"query": query, "sub_question_id": ""})

    return tasks


def expand_query(sub_question: str, llm=None) -> list:
    """为单个子问题生成互补检索词（搜索结果过少时调用）"""
    try:
        prompt = prompts.QUERY_EXPANSION_PROMPT.format(question=sub_question)
        data = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_ANALYST, llm=llm)
        queries = data.get("queries") if isinstance(data, dict) else data
        return [str(q).strip() for q in (queries or []) if str(q).strip()][:2]
    except Exception:
        return []


# ========== 混合搜索：本地知识库 ==========
def search_local_many(plan: list, pending_queries: list = None, max_workers: int = None) -> tuple:
    """
    并发查询本地知识库。
    检索词用"子问题本身"（而不是衍生关键词）——本地库是语义检索，
    用完整问句比用短关键词召回更准。
    """
    tasks = []
    for sub in to_dicts(plan):
        question = (sub.get("question") or "").strip()
        if question:
            tasks.append((question, sub.get("id", "")))
    for query in pending_queries or []:
        query = str(query).strip()
        if query:
            tasks.append((query, ""))
    if not tasks:
        return [], []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = max_workers or config.MAX_CONCURRENCY
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {
            pool.submit(local_kb.search_local, question, None, sub_id): question
            for question, sub_id in tasks
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                errors.append(f"[本地库:{question[:20]}] {type(exc).__name__}: {exc}")
    return results, errors


def normalize_scores(items: list, key: str = "score") -> list:
    """
    把一批结果的分数线性归一化到 [0.3, 1.0]。

    ⚠️ 只用它处理**同一来源内部**的分数。跨来源比较请用 `normalize_union`。
    """
    if not items:
        return items
    values = [float(it.get(key) or 0.0) for it in items]
    low, high = min(values), max(values)
    span = high - low
    for item, value in zip(items, values):
        # 全部同分时统一给 0.6，避免除零
        item[key] = 0.6 if span <= 1e-9 else round(0.3 + 0.7 * (value - low) / span, 4)
    return items


def normalize_union(web_items: list, local_items: list, key: str = "score") -> tuple:
    """
    用**两路合起来的统一尺度**归一化，保留跨来源的相对强弱。

    为什么不能用两次 `normalize_scores`：那样每一路都会被各自拉伸到满量程，
    本地库哪怕只召回到一堆弱相关片段，其中最强的那个也会被拉成 1.0，
    进而压过所有网页结果。这正是 Phase 8 消融实验暴露出来的问题——
    本地引用只占 6%，却吃掉了 40% 的阅读名额，把网页阅读量从 29 篇压到 19 篇。
    """
    combined = [float(it.get(key) or 0.0) for it in list(web_items) + list(local_items)]
    if not combined:
        return web_items, local_items
    low, high = min(combined), max(combined)
    span = high - low

    def apply(items):
        for item in items:
            value = float(item.get(key) or 0.0)
            item[key] = 0.6 if span <= 1e-9 else round(0.3 + 0.7 * (value - low) / span, 4)
        return items

    return apply(web_items), apply(local_items)


def fuse_results(web_results: list, local_results: list) -> list:
    """
    融合网页与本地库结果：
      1. 两路用**统一尺度**归一化分数，保留"谁更相关"的可比性；
      2. 本地结果乘一个轻微加权（内部资料通常更贴身）；
      3. 网页结果按 URL 去重，本地片段按 (文档, 片段号) 去重。
    最低保障名额由提取器在"决定读哪些页面"时执行（见 extractor._select_targets）。
    """
    web = [dict(r) for r in (web_results or [])]
    local = [dict(r) for r in (local_results or [])]
    web, local = normalize_union(web, local)

    for item in local:
        item["score"] = round(min(item.get("score", 0.0) * config.LOCAL_SCORE_BOOST, 1.0), 4)
        item["engine"] = "local"

    seen_web, seen_local = set(), set()
    merged = []
    for item in web:
        key = search_tools.normalize_url(item.get("url", ""))
        if not key or key in seen_web:
            continue
        seen_web.add(key)
        merged.append(item)
    for item in local:
        key = item.get("url", "")
        if not key or key in seen_local:
            continue
        seen_local.add(key)
        merged.append(item)

    merged.sort(key=lambda it: it.get("score", 0.0), reverse=True)
    return merged


@traced_node("searcher")
def searcher_node(state: dict) -> dict:
    """LangGraph 节点：并发执行网页检索 + 本地库检索，返回融合后的搜索结果"""
    plan = state.get("plan") or []
    pending = state.get("pending_queries") or []
    iteration = state.get("iteration", 1)

    # 已经检索过的词不再重复跑（补充搜索轮次尤其重要）
    searched = set(state.get("stats", {}).get("searched_queries") or [])
    tasks = [t for t in build_query_plan(plan, pending) if t["query"] not in searched]

    use_local = config.USE_LOCAL_KB and local_kb.list_source_files()
    if not tasks and not use_local:
        return {
            "events": [log_event(state, "searcher", "没有新的检索词，跳过搜索",
                                 status="warn", iteration=iteration)],
        }

    events = []
    with Timer() as timer:
        # ---- 网页检索 ----
        results, per_query, errors = [], {}, []
        if tasks:
            queries = [t["query"] for t in tasks]
            results, per_query, errors = search_tools.search_many(queries)
            # 命中过少时，给对应子问题做一次查询扩展再补一轮
            thin = [t for t in tasks if per_query.get(t["query"], 0) == 0]
            if thin:
                extra = []
                for task in thin[:2]:
                    sub = next((p for p in to_dicts(plan) if p.get("id") == task["sub_question_id"]), None)
                    if not sub:
                        continue
                    for query in expand_query(sub.get("question", ""), llm=None):
                        if query not in searched and query not in queries:
                            extra.append({"query": query, "sub_question_id": sub.get("id", "")})
                if extra:
                    extra_results, extra_per, extra_errors = search_tools.search_many(
                        [t["query"] for t in extra]
                    )
                    results = search_tools.dedupe_results(results + extra_results)
                    tasks.extend(extra)
                    per_query.update(extra_per)
                    errors.extend(extra_errors)

        # ---- 本地知识库检索 ----
        local_results, local_errors = [], []
        if use_local:
            local_results, local_errors = search_local_many(plan, pending)

    # 跨轮去重：已经在状态里的 URL / 本地片段不再重复加入
    known_urls = {
        search_tools.result_key(r.get("url", "")) for r in to_dicts(state.get("search_results"))
    }
    query_to_sub = defaultdict(list)
    for task in tasks:
        query_to_sub[task["query"]].append(task["sub_question_id"])

    if use_local:
        # 本地片段按 (文档, 片段号) 去重，网页结果按规范化 URL 去重
        local_results = [
            item for item in search_tools.dedupe_by_key(local_results)
            if search_tools.result_key(item.get("url", "")) not in known_urls
        ]
        web_deduped = search_tools.dedupe_results(results)
        merged = fuse_results(web_deduped, local_results)
        before = len(merged)
        merged = [
            item for item in merged
            if item.get("engine") == "local"
            or search_tools.result_key(item.get("url", "")) not in known_urls
        ]
        skipped = before - len(merged)
    else:
        merged, skipped = [], 0
        for item in search_tools.dedupe_results(results):
            if search_tools.result_key(item.get("url", "")) in known_urls:
                skipped += 1
                continue
            owners = query_to_sub.get(item.get("query")) or [""]
            item["sub_question_id"] = next((o for o in owners if o), owners[0])
            merged.append(item)

    # 给融合结果补上归属子问题（本地结果已自带）
    for item in merged:
        if item.get("sub_question_id"):
            continue
        owners = query_to_sub.get(item.get("query")) or [""]
        item["sub_question_id"] = next((o for o in owners if o), owners[0])

    engine = search_tools.resolve_engine() if tasks else "local-only"
    stats = dict(state.get("stats") or {})
    stats["searched_queries"] = sorted(searched | set(per_query.keys()))
    stats["searches"] = stats.get("searches", 0) + len(per_query)
    stats["engine"] = engine
    stats["local_used"] = bool(use_local)
    stats["local_hits"] = stats.get("local_hits", 0) + len([m for m in merged
                                                            if m.get("engine") == "local"])

    local_count = len([m for m in merged if m.get("engine") == "local"])
    events.append(
        log_event(
            state, "searcher",
            f"第 {iteration} 轮检索：{len(per_query)} 个关键词 → {len(merged)} 条新结果"
            + (f"（含本地库 {local_count} 条）" if use_local else ""),
            f"网页引擎：{engine}；跨轮去重丢弃 {skipped} 条"
            + ("；本地知识库已启用" if use_local else "；本地知识库未启用"),
            elapsed=timer.elapsed, iteration=iteration,
            queries=list(per_query.keys()), engine=engine, local_hits=local_count,
        )
    )
    all_errors = errors + local_errors
    if all_errors:
        events.append(
            log_event(state, "searcher", f"{len(all_errors)} 个检索任务失败",
                      "；".join(all_errors[:3]), status="warn", iteration=iteration)
        )

    return {"search_results": merged, "stats": stats, "events": events}
