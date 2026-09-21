"""
顶层门面模块：DeepResearchAgent。
对标 RAG_DocQA 的 rag_core.py——app.py / main.py / eval_*.py 都通过它调用，
保证"一次研究"的流程、配置与产物落盘方式完全一致。
"""
import json
import time
from datetime import datetime
from pathlib import Path

import config
from events import bus
from graph import RECURSION_LIMIT, build_graph, describe_graph
from llm import get_usage, reset_usage, use_llm
from planner import plan_research, revise_plan
from state import initial_state, to_dicts
from writer import render_report_markdown


class DeepResearchAgent:
    """深度研究 Agent：封装状态图，对外只暴露 run / plan 两个动作"""

    def __init__(self, max_iterations: int = None, engine: str = None,
                 max_pages_per_round: int = None, llm=None, use_local_kb: bool = None):
        """
        :param max_iterations: 最大搜索轮数（含首轮），None 用配置默认值
        :param engine: 搜索后端 auto / tavily / ddg，None 用配置默认值
        :param max_pages_per_round: 单轮最多阅读网页数
        :param llm: 注入的 LLM。**只在本次 plan()/run() 的作用域内生效**，
                    不会污染进程级全局（早期版本用 set_llm 注入，导致 Streamlit
                    常驻进程里"离线假模型"泄漏到后续的"真实运行"，是真 Bug）。
        :param use_local_kb: 是否启用本地知识库（None 用配置默认值）
        """
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.engine = engine or config.SEARCH_ENGINE
        self.max_pages_per_round = max_pages_per_round or config.MAX_PAGES_PER_ROUND
        self._llm_override = llm

        if engine:
            config.SEARCH_ENGINE = engine
        if max_pages_per_round:
            config.MAX_PAGES_PER_ROUND = max_pages_per_round
        if max_iterations:
            # 同步回配置，这样 --doctor / 环境表展示的是**本次实际生效**的轮数，
            # 而不是配置文件的默认值（否则 `--max-iter 1` 时仍显示 3，具有误导性）
            config.MAX_ITERATIONS = self.max_iterations
        if use_local_kb is not None:
            config.USE_LOCAL_KB = use_local_kb

        self.graph = build_graph()

    # ---------- 本地知识库 ----------
    @staticmethod
    def kb_status(rebuild: bool = False) -> dict:
        """本地知识库状态（文档数 / 片段数 / 是否已建索引）"""
        import local_kb

        return local_kb.kb_status(rebuild=rebuild)

    # ---------- 计划相关（支持人工调整）----------
    def plan(self, question: str) -> dict:
        """只跑规划器，返回研究计划（供用户确认/修改）"""
        with use_llm(self._llm_override):
            return plan_research(question)

    def revise_plan(self, question: str, current_plan: list, feedback: str) -> dict:
        """按用户意见修订研究计划"""
        with use_llm(self._llm_override):
            return revise_plan(question, current_plan, feedback)

    # ---------- 主流程 ----------
    def run(self, question: str, on_event=None, plan: list = None,
            plan_summary: str = "", max_iterations: int = None) -> dict:
        """
        执行一次完整研究。

        :param question: 研究问题
        :param on_event: 事件回调 callable(event: dict)，用于实时展示思考过程
        :param plan: 已确认的研究计划（传了就跳过自动规划）
        :param plan_summary: 计划说明
        :param max_iterations: 覆盖最大搜索轮数
        :return: 最终状态 dict
        """
        state = initial_state(question, max_iterations or self.max_iterations)
        if plan:
            state["plan"] = to_dicts(plan)
            state["plan_summary"] = plan_summary
            state["plan_approved"] = True

        unsubscribe = bus.subscribe(on_event) if on_event else None
        reset_usage()
        started = time.perf_counter()

        # ---- 可观测性：整张图的运行被一个 tracer 覆盖 ----
        from observability import Tracer, get_langfuse_client, use_tracer

        tracer = Tracer(
            name="deep_research",
            question=question,
            langfuse_client=get_langfuse_client() if config.TRACING_ENABLED else None,
            enabled=config.TRACING_ENABLED,
        )
        try:
            # use_llm 只在本次运行的作用域内生效，退出即恢复，
            # 不会把注入的假模型留在进程里污染后续的真实运行。
            with use_llm(self._llm_override), use_tracer(tracer):
                # 必须有一个根 span：否则每个节点都会各自成为一棵独立的 trace，
                # 在 LangFuse 里会看到 5 条互不相干的记录，而不是一次完整的研究。
                with tracer.span("research", kind="span",
                                 question=question[:120],
                                 max_iterations=max_iterations or self.max_iterations,
                                 local_kb=config.USE_LOCAL_KB):
                    final_state = self.graph.invoke(
                        state, config={"recursion_limit": RECURSION_LIMIT}
                    )
        finally:
            if unsubscribe:
                unsubscribe()

        trace = tracer.finish()
        final_state["trace"] = trace

        final_state["stats"]["elapsed"] = round(time.perf_counter() - started, 2)
        final_state["stats"]["llm_usage"] = get_usage()
        final_state["stats"]["cost_usd"] = trace.get("cost_usd", 0.0)
        final_state["stats"]["cost_cny"] = trace.get("cost_cny", 0.0)
        final_state["stats"]["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 报告里的"附：研究过程"含耗时与 Token 统计，而这些数据要等整张图跑完才有，
        # 所以这里必须**重渲染一次**，否则报告尾部会写着"LLM 调用 0 次 / 耗时 0 秒"。
        if final_state.get("report"):
            final_state["report_markdown"] = render_report_markdown(
                final_state["report"], final_state
            )
            final_state["stats"]["report_chars"] = len(final_state["report_markdown"])
        return final_state

    # ---------- 产物落盘 ----------
    @staticmethod
    def save_run(state: dict, out_dir: Path = None,
                 formats: tuple = ("md", "html", "pdf")) -> dict:
        """
        把一次运行的完整产物写入 outputs/<时间戳>/：
          state.json   全量状态（可复现）
          events.json  过程事件
          report.md / report.html / report.pdf  报告成品

        :return: {"dir": 目录, "paths": {名称: 路径}, "warnings": [提示]}
        """
        out_dir = Path(out_dir or config.OUTPUT_DIR)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        paths, warnings = {}, []
        state_path = run_dir / "state.json"
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(state), fh, ensure_ascii=False, indent=2)
        paths["state"] = str(state_path)

        events = to_dicts(state.get("events"))
        if events:
            events_path = run_dir / "events.json"
            with open(events_path, "w", encoding="utf-8") as fh:
                json.dump(events, fh, ensure_ascii=False, indent=2)
            paths["events"] = str(events_path)

        report_md = state.get("report_markdown") or ""
        if report_md:
            from report import export_report

            exported = export_report(report_md, run_dir, stem="report", formats=formats)
            paths.update(exported["paths"])
            warnings.extend(exported["warnings"])

        trace = state.get("trace")
        if trace:
            trace_path = run_dir / "trace.json"
            with open(trace_path, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(trace), fh, ensure_ascii=False, indent=2)
            paths["trace"] = str(trace_path)
            if trace.get("langfuse_error"):
                warnings.append(f"LangFuse 上报异常：{trace['langfuse_error']}")

        return {"dir": str(run_dir), "paths": paths, "warnings": warnings}

    @staticmethod
    def graph_mermaid() -> str:
        """状态图的 Mermaid 文本"""
        return describe_graph()


def _jsonable(obj):
    """把状态里的非 JSON 类型（Path / set / Pydantic）递归转成可序列化对象"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump())
    return obj


def summarize(state: dict) -> dict:
    """汇总一次运行的关键指标，供 CLI / 前端 / 评估脚本使用"""
    stats = dict(state.get("stats") or {})
    facts = to_dicts(state.get("facts"))
    pages = to_dicts(state.get("pages"))
    results = to_dicts(state.get("search_results"))
    reflections = to_dicts(state.get("reflections"))
    plan = to_dicts(state.get("plan"))

    # 子问题覆盖率：有事实支撑的子问题占比
    covered = {f.get("sub_question_id") for f in facts if f.get("sub_question_id")}
    plan_ids = {p.get("id") for p in plan}
    coverage = len(covered & plan_ids) / len(plan_ids) if plan_ids else 0.0

    local_pages = len([p for p in pages if p.get("source") == "local"])

    return {
        "研究问题": state.get("question", ""),
        "子问题数": len(plan),
        "子问题覆盖率": f"{coverage:.0%}",
        "检索词数": len(stats.get("searched_queries") or []),
        "搜索结果数": len(results),
        "其中本地库结果": stats.get("local_hits", 0),
        "成功抓取网页数": len([p for p in pages if p.get("ok")]) - local_pages,
        "读取本地片段数": local_pages,
        "事实数": len(facts),
        "矛盾数据组数": len(stats.get("conflicts") or []),
        "搜索轮数": state.get("iteration", 1),
        "反思轮数": len(reflections),
        "反思触发补充搜索": stats.get("reflection_triggered", 0),
        "报告章节数": stats.get("report_sections", 0),
        "引用来源数": stats.get("citations", 0),
        "其中本地文档引用": len([c for c in (state.get("report") or {}).get("citations") or []
                                 if c.get("source_type") == "local"]),
        "越界引用数": stats.get("invalid_citation_count", 0),
        "报告字数": stats.get("report_chars", 0),
        "耗时(秒)": stats.get("elapsed", 0),
        "Token 消耗": stats.get("llm_usage", {}).get("total_tokens", 0),
        "成本(CNY)": f"¥{stats.get('cost_cny', 0.0):.4f}",
    }
