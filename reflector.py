"""
模块 4：反思决策器（Reflector）
输入：当前已收集的全部信息（事实 + 已检索词 + 计划完成度）
输出：是否需要补充搜索 + 补充搜索的关键词

设计要点：
  1. LLM 只做"判断"，不做"检索"；判断结果一律经过规则层二次校验，避免死循环。
  2. 四道刹车：最大轮数、无新增事实、补充词为空、补充词与历史重复。
  3. 每轮反思都留档到 state["reflections"]，供前端展示与评估统计。
"""
import re

import config
import llm as llm_module
import prompts
from events import Timer, log_event
from observability import traced_node
from state import ReflectionDecision, to_dicts

# 每轮最多接受几条补充检索词
MAX_NEW_QUERIES = 3

# 包含关系判重时，短词至少要有长词的这么长，防止 "AI" 命中一切
MIN_CONTAINMENT_RATIO = 0.6

_NORM_RE = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()\[\]【】\-—_/\\|]+")


def normalize_query(query: str) -> str:
    """检索词归一化，用于判断"这个词是不是已经搜过了\""""
    return _NORM_RE.sub("", (query or "").lower())


def is_repeated(query: str, history: list, threshold: float = 0.90) -> bool:
    """
    判断检索词是否与历史重复。判定条件（满足其一即算重复）：
      1. 归一化后完全相同；
      2. 包含关系，且短的那条不短于长的那条的 60%（避免 "AI" 命中一切）；
      3. 归一化后相似度 >= threshold。

    阈值取 0.90 而不是更低的经验值：中文检索词常有共同前缀（如都以"DeepSeek "开头），
    阈值过低会把"补充证据"和"数据"这种真正不同的检索词误判成重复而丢掉。
    """
    from difflib import SequenceMatcher

    target = normalize_query(query)
    if not target:
        return True
    for old in history:
        old_norm = normalize_query(old)
        if not old_norm:
            continue
        if target == old_norm:
            return True

        shorter, longer = sorted((target, old_norm), key=len)
        if shorter in longer and len(shorter) / len(longer) >= MIN_CONTAINMENT_RATIO:
            return True

        if SequenceMatcher(None, target, old_norm).ratio() >= threshold:
            return True
    return False


# ========== 事实摘要（喂给 LLM 前先压缩，控制 Token）==========
def build_facts_digest(facts: list, plan: list = None) -> str:
    """
    把事实按子问题分组压缩成清单文本。
    每条事实只保留编号 + 陈述 + 来源域名，避免把整段原文塞进 Prompt。
    """
    items = to_dicts(facts)
    if not items:
        return "（暂无）"

    from urllib.parse import urlparse

    grouped = {}
    for fact in items:
        grouped.setdefault(fact.get("sub_question_id") or "未分类", []).append(fact)

    lines = []
    for sub_id, group in grouped.items():
        lines.append(f"【{sub_id}】共 {len(group)} 条")
        for fact in group[:12]:  # 单个子问题最多展示 12 条，防止 Prompt 过长
            domain = urlparse(fact.get("source_url", "")).netloc
            lines.append(
                f"  {fact.get('id')}. {fact.get('claim', '')[:110]}"
                f"（来源 {domain or '未知'}，置信度 {fact.get('confidence', 0):.1f}）"
            )
        if len(group) > 12:
            lines.append(f"  …另有 {len(group) - 12} 条")
    return "\n".join(lines)


def build_plan_digest(plan: list) -> str:
    """把研究计划压成清单文本"""
    items = to_dicts(plan)
    if not items:
        return "（无）"
    return "\n".join(f"{p.get('id')}. {p.get('question', '')}" for p in items)


# ========== LLM 反思 ==========
def reflect(question: str, plan: list, facts: list, searched_queries: list,
            iteration: int = 1, llm=None) -> dict:
    """调用 LLM 判断信息充分性，返回 ReflectionDecision 的 dict"""
    prompt = prompts.REFLECTOR_PROMPT.format(
        question=question,
        plan=build_plan_digest(plan),
        fact_count=len(to_dicts(facts)),
        facts=build_facts_digest(facts, plan),
        searched="、".join(searched_queries[-20:]) or "（无）",
    )
    raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_ANALYST, llm=llm)
    if not isinstance(raw, dict):
        raw = {"sufficient": True, "reason": "反思输出格式异常，按充分处理", "new_queries": []}

    queries = raw.get("new_queries") or raw.get("queries") or []
    if isinstance(queries, str):
        queries = [queries]
    gaps = raw.get("gaps") or []
    if isinstance(gaps, str):
        gaps = [gaps]

    try:
        coverage = float(raw.get("coverage", 0.0))
    except (TypeError, ValueError):
        coverage = 0.0

    return ReflectionDecision(
        sufficient=bool(raw.get("sufficient", True)),
        reason=str(raw.get("reason") or "").strip(),
        gaps=[str(g).strip() for g in gaps if str(g).strip()],
        new_queries=[str(q).strip() for q in queries if str(q).strip()],
        coverage=min(max(coverage, 0.0), 1.0),
        iteration=iteration,
    ).model_dump()


# ========== 决策裁决（规则层兜底）==========
def decide(decision: dict, state: dict) -> tuple:
    """
    规则层二次裁决：LLM 说"要继续"也未必真的继续。
    返回 (是否继续搜索, 最终采用的补充检索词, 中止原因)。
    """
    searched = list(state.get("stats", {}).get("searched_queries") or [])
    iteration = state.get("iteration", 1)
    max_iterations = state.get("stats", {}).get("max_iterations") or config.MAX_ITERATIONS

    # 刹车 1：达到最大轮数
    if iteration >= max_iterations:
        return False, [], f"已达最大搜索轮数（{max_iterations}）"

    if decision.get("sufficient"):
        return False, [], ""

    # 刹车 2：LLM 说要继续但没给出新检索词
    unique_candidates = []
    for query in decision.get("new_queries", []):
        # 既不能和历史检索词重复，新词之间也不能互相重复
        if is_repeated(query, searched) or is_repeated(query, unique_candidates):
            continue
        unique_candidates.append(query)
    candidates = unique_candidates[:MAX_NEW_QUERIES]
    if not candidates:
        return False, [], "反思认为信息不足，但未能给出全新的补充检索词"

    # 刹车 3：上一轮没有新增事实，再搜也大概率是重复内容
    if iteration >= 2:
        gained = state.get("stats", {}).get("facts_gained_last_round", 0)
        if gained == 0:
            return False, [], "上一轮补充搜索未带来任何新事实，停止继续搜索"

    return True, candidates, ""


# ========== 节点 ==========
@traced_node("reflector")
def reflector_node(state: dict) -> dict:
    """LangGraph 节点：判断信息是否充分，不充分则给出补充检索词"""
    question = state.get("question", "")
    plan = state.get("plan") or []
    facts = to_dicts(state.get("facts"))
    iteration = state.get("iteration", 1)
    stats = dict(state.get("stats") or {})
    searched = list(stats.get("searched_queries") or [])

    with Timer() as timer:
        try:
            decision = reflect(question, plan, facts, searched, iteration)
        except Exception as exc:
            # 反思失败不应该让整条链路崩掉，直接按"信息充分"处理并留下告警
            decision = ReflectionDecision(
                sufficient=True,
                reason=f"反思调用失败（{type(exc).__name__}），为保证链路继续按信息充分处理",
                iteration=iteration,
            ).model_dump()

    continue_search, new_queries, stop_reason = decide(decision, state)

    if stop_reason:
        decision["reason"] = (decision.get("reason") + "；" if decision.get("reason") else "") + stop_reason
    decision["sufficient"] = not continue_search

    stats["reflect_rounds"] = stats.get("reflect_rounds", 0) + 1
    if continue_search:
        stats["reflection_triggered"] = stats.get("reflection_triggered", 0) + 1
        stats["max_iterations"] = stats.get("max_iterations") or config.MAX_ITERATIONS

    events = [
        log_event(
            state, "reflector",
            f"第 {iteration} 轮反思：{'信息充分，可以写作' if not continue_search else f'信息不足，补充 {len(new_queries)} 个检索词'}",
            decision.get("reason", ""),
            status="done" if not continue_search else "warn",
            elapsed=timer.elapsed, iteration=iteration,
            coverage=decision.get("coverage", 0.0),
            gaps=decision.get("gaps", []),
            new_queries=new_queries,
        )
    ]
    if decision.get("gaps"):
        events.append(
            log_event(
                state, "reflector", f"识别出 {len(decision['gaps'])} 个信息缺口",
                "；".join(decision["gaps"][:3]),
                elapsed=0.0, iteration=iteration,
            )
        )

    update = {
        "reflections": [decision],
        "stats": stats,
        "events": events,
        "pending_queries": new_queries,
    }
    if continue_search:
        update["iteration"] = iteration + 1
    return update


def route_after_reflection(state: dict) -> str:
    """
    条件边：反思之后往哪走。
    Phase 2 阶段是"回到搜索执行器 / 结束"；Phase 3 会把"结束"改成"进入报告生成器"。
    """
    return "search_again" if (state.get("pending_queries") or []) else "finish"
