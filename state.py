"""
状态定义模块：Agent 的数据模型（Pydantic）与 LangGraph 状态（TypedDict）。
设计原则：
  - 对外用 Pydantic 模型做校验，进入状态图后统一转成 dict，方便序列化与归档。
  - 会跨节点累积的字段（搜索结果 / 网页 / 事实 / 事件）挂 operator.add reducer，
    这样并行分支写回时会自动合并而不是互相覆盖。
"""
import operator
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

import config


def now_str() -> str:
    """统一时间戳格式"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str = "") -> str:
    """生成短 id"""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ========== 数据模型 ==========
class SubQuestion(BaseModel):
    """子问题（研究计划的最小单元）"""

    id: str = Field(default="", description="编号，如 Q1")
    question: str = Field(description="子问题本身")
    rationale: str = Field(default="", description="为什么要研究它 / 它在整体中的位置")
    search_queries: list[str] = Field(default_factory=list, description="建议的检索关键词")


class SearchResult(BaseModel):
    """一条搜索结果（标题 + 摘要 + URL）"""

    id: str = Field(default="")
    query: str = Field(default="", description="触发这条结果的检索词")
    title: str = Field(default="")
    url: str = Field(default="")
    snippet: str = Field(default="", description="搜索引擎给出的摘要")
    content: str = Field(default="", description="本地知识库片段自带正文，无需再抓取")
    engine: str = Field(default="", description="tavily / ddg / local")
    score: float = Field(default=0.0, description="归一化后的相关性打分")
    sub_question_id: str = Field(default="", description="归属的子问题编号")
    fetched: bool = Field(default=False, description="正文是否已抓取")


class PageContent(BaseModel):
    """抓取到的网页正文"""

    url: str = Field(default="")
    title: str = Field(default="")
    text: str = Field(default="")
    ok: bool = Field(default=True)
    error: str = Field(default="")
    chars: int = Field(default=0)
    source: str = Field(default="web", description="web / local")


class Fact(BaseModel):
    """从网页中提炼出的结构化事实"""

    id: str = Field(default="")
    claim: str = Field(description="事实陈述")
    quote: str = Field(default="", description="支撑该事实的原文摘录")
    source_url: str = Field(default="")
    source_title: str = Field(default="")
    sub_question_id: str = Field(default="")
    confidence: float = Field(default=0.5, description="LLM 自评置信度 0~1")
    credibility: float = Field(default=0.5, description="来源可信度 0~1")
    conflict_with: list[str] = Field(default_factory=list, description="与之冲突的事实编号")
    created_at: str = Field(default_factory=now_str)


class ReflectionDecision(BaseModel):
    """反思决策器的输出"""

    sufficient: bool = Field(default=False, description="现有信息是否足以作答")
    reason: str = Field(default="", description="判断理由")
    gaps: list[str] = Field(default_factory=list, description="仍缺失的信息点")
    new_queries: list[str] = Field(default_factory=list, description="补充搜索关键词")
    coverage: float = Field(default=0.0, description="信息覆盖率估计 0~1")
    iteration: int = Field(default=1)


class Citation(BaseModel):
    """报告中的引用条目"""

    index: int = Field(default=0, description="引用编号，如 [1]")
    title: str = Field(default="")
    url: str = Field(default="")
    source_type: str = Field(default="web", description="web 网络来源 / local 本地知识库")
    used_by: list[str] = Field(default_factory=list, description="引用了它的章节")


class ResearchReport(BaseModel):
    """结构化研究报告"""

    title: str = Field(default="")
    question: str = Field(default="")
    summary: str = Field(default="", description="摘要")
    sections: list[dict] = Field(default_factory=list, description="[{heading, content}]")
    conclusion: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
    markdown: str = Field(default="", description="最终 Markdown 全文")
    created_at: str = Field(default_factory=now_str)


class AgentEvent(BaseModel):
    """过程可视化用的事件（思考 / 搜索 / 反思 / 写作）"""

    step: str = Field(default="", description="planner / searcher / extractor / ...")
    title: str = Field(default="", description="一句话说明")
    detail: str = Field(default="")
    status: str = Field(default="done", description="start / done / warn / error")
    elapsed: float = Field(default=0.0, description="本步耗时（秒）")
    ts: str = Field(default_factory=now_str)
    data: dict = Field(default_factory=dict, description="结构化附加信息")


# ========== LangGraph 状态 ==========
class ResearchState(TypedDict, total=False):
    """贯穿整张状态图的研究状态"""

    # ---- 输入 ----
    question: str

    # ---- 规划 ----
    plan: list[dict]              # [SubQuestion]
    plan_summary: str
    plan_approved: bool           # 用户是否已确认计划（True 则跳过重新规划）

    # ---- 检索（累积）----
    search_results: Annotated[list[dict], operator.add]   # [SearchResult]

    # ---- 阅读（累积）----
    pages: Annotated[list[dict], operator.add]            # [PageContent]

    # ---- 事实（累积）----
    facts: Annotated[list[dict], operator.add]            # [Fact]

    # ---- 反思（累积）----
    reflections: Annotated[list[dict], operator.add]      # [ReflectionDecision]
    pending_queries: list[str]    # 本轮反思给出的补充检索词
    iteration: int                # 当前搜索轮次，从 1 开始

    # ---- 输出 ----
    report: dict                  # ResearchReport
    report_markdown: str

    # ---- 过程可视化 / 统计 ----
    events: Annotated[list[dict], operator.add]           # [AgentEvent]
    stats: dict
    error: str


def to_dicts(items) -> list:
    """把 Pydantic 模型列表转成 dict 列表（已是 dict 的原样返回）"""
    out = []
    for item in items or []:
        if isinstance(item, BaseModel):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(item)
        else:
            out.append(dict(item))
    return out


def initial_state(question: str, max_iterations: int = None) -> dict:
    """构造一次研究的初始状态"""
    return {
        "question": question.strip(),
        "plan": [],
        "plan_summary": "",
        "plan_approved": False,
        "search_results": [],
        "pages": [],
        "facts": [],
        "reflections": [],
        "pending_queries": [],
        "iteration": 1,
        "report": {},
        "report_markdown": "",
        "events": [],
        "stats": {
            "max_iterations": (
                max_iterations if max_iterations is not None else config.MAX_ITERATIONS
            ),
            "started_at": now_str(),
            "searches": 0,
            "pages_fetched": 0,
            "engine": "",
        },
        "error": "",
    }
