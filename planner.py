"""
模块 1：任务规划器（Planner）
输入：用户研究问题
输出：子问题列表 + 研究计划
要点：用 LLM 做任务分解；子问题之间有逻辑递进；支持用户手动调整计划。

**子问题必须自带主语**：真实出现过子问题被写成"背景是什么？""现状如何？""风险在哪？"
的情况——这类问题脱离原研究问题后完全不知道在问什么，拿去检索只会搜回无关内容，
进而污染整份报告。这里设了两道防线：
  ① Prompt 中明确给出反例与正例；
  ② 规则层再检查一遍，检出"无主语小标题"就用一次补救 Prompt 改写，仍不合格则程序化兜底。
"""
import re

import config
import llm as llm_module
import prompts
from events import Timer, log_event
from observability import traced_node
from state import SubQuestion, to_dicts

# 典型的"无主语小标题"：整句就是一个泛化名词 + 疑问词，没有任何研究对象
_GENERIC_HEADING_RE = re.compile(
    r"^(当前|目前|整体|总体)?"
    r"(背景|现状|定义|概念|概述|意义|趋势|前景|风险|挑战|影响|优势|劣势|优缺点|结论|总结|对策|建议|方案)"
    r"[^，。；：]{0,8}[？?]?$"
)
_MIN_SUBQUESTION_LEN = 10


def is_self_contained(question: str) -> bool:
    """
    判断子问题是否是"能独立看懂、带研究主体"的完整问句。

    判定刻意保守，只拦真正的"无主语句子"：
      - 长度不足 10 字的判为不合格；
      - 整句匹配 `_GENERIC_HEADING_RE`（如"背景是什么？""风险在哪？"）的判为不合格。
    不做"必须与原问题有词重叠"这类检查——那会误杀
    "MoE 架构对推理成本的影响"这种引入新概念但完全正当的子问题。
    """
    text = (question or "").strip()
    if len(text) < _MIN_SUBQUESTION_LEN:
        return False
    return not _GENERIC_HEADING_RE.match(text)


def _repair_subquestion(question: str, bad: str, llm=None) -> tuple:
    """
    补救一个无主语子问题。
    先用一次 LLM 改写；失败或仍不合格则程序化兜底：把研究问题作为主语前缀拼接。
    :return: (改写后的子问题, 检索词列表)
    """
    prompt = prompts.REWRITE_SUBQUESTION_PROMPT.format(question=question, bad_question=bad)
    try:
        raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_ANALYST, llm=llm)
    except Exception:
        raw = None

    if isinstance(raw, dict):
        new_question = str(raw.get("question") or "").strip()
        if is_self_contained(new_question):
            queries = raw.get("search_queries") or raw.get("queries") or []
            if isinstance(queries, str):
                queries = [queries]
            queries = [str(q).strip() for q in queries if str(q).strip()]
            return new_question, queries or [new_question]

    # 程序化兜底：把研究主体接到前面，保证"能看懂、能检索"
    subject = question.rstrip("？?。.").strip()
    suffix = bad.rstrip("？?。.").strip()
    fallback = f"{subject}——{suffix}是怎样的？" if suffix else subject
    return fallback, [fallback]


def _normalize_plan(raw: dict, question: str, llm=None) -> tuple:
    """
    把 LLM 返回的原始 JSON 规整成 (plan, plan_summary)。
    做五件事：字段补全、编号重排、数量裁剪、去重，以及**无主语子问题的检出与补救**。
    """
    summary = str(raw.get("plan_summary") or "").strip()
    raw_items = raw.get("sub_questions") or raw.get("subQuestions") or []

    plan, seen = [], set()
    for item in raw_items:
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("question") or item.get("sub_question") or "").strip()
        if not text:
            continue
        key = text.replace(" ", "")
        if key in seen:  # 同一子问题重复出现，丢弃
            continue

        queries = item.get("search_queries") or item.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        queries = [str(q).strip() for q in queries if str(q).strip()]

        # 防线②：检出无主语的"背景是什么？"这类子问题并补救
        if not is_self_contained(text):
            text, queries = _repair_subquestion(question, text, llm=llm)

        # 检索词也要带主体：全是"背景/现状/风险"这类泛词时换成子问题本身
        if not queries or all(_GENERIC_HEADING_RE.match(q) or len(q) <= 3 for q in queries):
            queries = [text]

        seen.add(key)
        plan.append(
            SubQuestion(
                id=f"Q{len(plan) + 1}",
                question=text,
                rationale=str(item.get("rationale") or "").strip(),
                search_queries=queries[:3],
            )
        )
        if len(plan) >= config.MAX_SUB_QUESTIONS:
            break

    if len(plan) < config.MIN_SUB_QUESTIONS and plan:
        # 数量偏少时兜底补充一个"补充核查"型子问题，保证研究面不至于太窄
        plan.append(
            SubQuestion(
                id=f"Q{len(plan) + 1}",
                question=f"围绕「{question}」还有哪些关键数据或反对意见需要核实？",
                rationale="补齐可能被忽略的反面证据与关键数据",
                search_queries=[f"{question} 争议", f"{question} 数据"],
            )
        )
    return to_dicts(plan), summary


def plan_research(question: str, llm=None) -> dict:
    """调用 LLM 生成研究计划，返回 {'plan': [...], 'plan_summary': '...'}"""
    prompt = prompts.PLANNER_PROMPT.format(
        question=question,
        min_n=config.MIN_SUB_QUESTIONS,
        max_n=config.MAX_SUB_QUESTIONS,
    )
    raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_ANALYST, llm=llm)
    if not isinstance(raw, dict):
        raw = {"sub_questions": raw}
    plan, summary = _normalize_plan(raw, question, llm=llm)
    if not plan:
        raise ValueError("规划器未能产出任何子问题，请检查 Prompt 或模型输出")
    return {"plan": plan, "plan_summary": summary}


def revise_plan(question: str, current_plan: list, feedback: str, llm=None) -> dict:
    """
    根据用户反馈修订已有研究计划（对应"支持用户手动调整研究计划"）。
    feedback 例如："第三个子问题改成关注成本" / "去掉 Q2，加上安全性对比"。
    """
    current_text = "\n".join(
        f"{p.get('id')}. {p.get('question')}" for p in to_dicts(current_plan)
    )
    prompt = (
        f"{prompts.PLANNER_PROMPT.format(question=question, min_n=config.MIN_SUB_QUESTIONS, max_n=config.MAX_SUB_QUESTIONS)}\n\n"
        f"【当前计划】\n{current_text}\n\n"
        f"【用户修改意见】\n{feedback}\n\n"
        "请在保留用户意见的前提下重写完整计划，仍然按上面的 JSON 格式输出。"
    )
    raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_ANALYST, llm=llm)
    if not isinstance(raw, dict):
        raw = {"sub_questions": raw}
    plan, summary = _normalize_plan(raw, question, llm=llm)
    return {"plan": plan, "plan_summary": summary}


@traced_node("planner")
def planner_node(state: dict) -> dict:
    """LangGraph 节点：把研究问题拆解成子问题列表"""
    question = state.get("question", "")
    existing = state.get("plan") or []

    if existing and state.get("plan_approved"):
        # 用户已经确认过计划（或上一轮已规划），不重复消耗 Token
        return {
            "plan": existing,
            "events": [log_event(state, "planner", "沿用已确认的研究计划",
                                 f"共 {len(existing)} 个子问题", status="done")],
        }

    with Timer() as timer:
        result = plan_research(question)
    plan = result["plan"]

    return {
        "plan": plan,
        "plan_summary": result["plan_summary"],
        "events": [
            log_event(
                state, "planner", f"拆解出 {len(plan)} 个子问题",
                result["plan_summary"], elapsed=timer.elapsed,
                sub_questions=[p["question"] for p in plan],
            )
        ],
    }
