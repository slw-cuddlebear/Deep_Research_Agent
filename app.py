"""
Streamlit 前端（Phase 6）：把 Agent 的思考过程实时展示出来。
对应规划里的"模块 6：过程可视化"。

设计要点：
  - **边跑边看**：Agent 每发生一个事件就通过事件总线推给回调，前端立刻追加一行，
    而不是等 5 分钟后一次性看到结果。
  - **计划可改**：研究计划先用 data_editor 展示，用户可以直接改子问题和检索词再执行。
  - **离线演示**：没有 API Key 也能点开看看界面长什么样（假模型 + 假搜索）。
  - 报告支持在线预览与 Markdown / HTML / PDF 下载。
"""
import warnings
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import streamlit as st

import config
from deep_research import DeepResearchAgent, summarize
from state import to_dicts

# ========== 页面配置 ==========
st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    .fixed-header {
        position: sticky; top: 0; z-index: 999;
        background-color: var(--background-color);
        padding: 0.5rem 0; border-bottom: 1px solid rgba(128, 128, 255, 0.2);
    }
    .metric-note { color: #888; font-size: 0.82rem; }
    .event-line { font-family: ui-monospace, Consolas, monospace; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
st.markdown("## 🔎 Deep Research Agent")
st.markdown(
    '<span class="metric-note">拆解问题 → 多源检索 → 阅读提炼 → 反思补充 → 生成带引用的研究报告</span>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

EVENT_ICONS = {"start": "▶", "done": "✔", "warn": "⚠", "error": "✖"}
# 示例问题下拉框的占位选项。不用"（不选）"这种说法——
# 它和上面的研究问题输入框上下相邻，容易被误认成"第二个问题框"。
EXAMPLE_PLACEHOLDER = "测试问题示例（点击展开选择）"
STEP_NAMES = {
    "planner": "任务规划",
    "searcher": "检索",
    "extractor": "阅读提炼",
    "reflector": "反思",
    "writer": "写作",
}


# ========== 会话状态 ==========
def init_state() -> None:
    for key, default in [
        ("question", ""),
        ("plan", []),
        ("plan_summary", ""),
        ("final_state", None),
        ("events", []),
        ("offline", False),
        ("error", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


init_state()


# ========== 侧边栏 ==========
def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ 运行设置")

        mode = st.radio(
            "运行模式",
            ["真实运行（需 API Key）", "离线演示（假数据）"],
            index=1 if st.session_state.offline else 0,
            help="离线演示使用假模型与假搜索，不联网、不消耗额度，用于查看界面与完整链路。",
        )
        st.session_state.offline = mode.startswith("离线")

        with st.expander("检索参数", expanded=True):
            max_iter = st.slider(
                "最大搜索轮数", 1, 5, config.MAX_ITERATIONS,
                help="反思后最多再补搜几轮。轮数越多越全面，但耗时与 Token 线性增长。",
            )
            max_pages = st.slider(
                "单轮最多阅读网页数", 4, 24, config.MAX_PAGES_PER_ROUND,
                help="先按融合分数统一竞争名额，再给本地知识库保留下面的保底条数。",
            )
            engine = st.selectbox(
                "网页搜索后端", ["auto", "tavily", "ddg"], index=0,
                help="auto：有 TAVILY_API_KEY 用 Tavily，否则用免费的 DuckDuckGo。",
            )
            use_local = st.checkbox(
                "启用本地知识库", value=config.USE_LOCAL_KB,
                help=f"检索 {config.LOCAL_KB_DIR.name}/ 下的文档，与网页结果融合。",
            )
            local_slots = st.slider(
                "本地库最低保障条数", 0, 8, config.LOCAL_MIN_SLOTS,
                disabled=not use_local,
                help="本地片段天然少于网页，容易被整体挤掉，所以保底几条；"
                     "其余名额由网页与本地按融合分数自由竞争。",
            )

        with st.expander("导出格式", expanded=False):
            want_md = st.checkbox("Markdown", value=True, disabled=True)
            want_html = st.checkbox("HTML", value=True)
            want_pdf = st.checkbox("PDF", value=True)
            save_to_disk = st.checkbox(
                "同时保存到 outputs/", value=True,
                help="每次研究会在 outputs/<时间戳>/ 下落盘完整状态与报告。",
            )

        st.divider()
        st.header("🧪 环境状态")
        env = config.describe_env()
        for key, value in env.items():
            icon = "✔" if "未配置" not in str(value) else "✖"
            st.markdown(f"{icon} **{key}**：{value}")

        st.divider()
        st.header("📚 本地知识库")
        try:
            import local_kb

            status = local_kb.kb_status()
            if status.get("documents"):
                st.success(f"{status['documents']} 篇文档 / {status.get('chunks') or 0} 个片段")
                with st.expander("文档列表"):
                    for name in status.get("files", []):
                        st.markdown(f"- {name}")
            else:
                st.warning(status.get("message", "语料为空"))
            if st.button("🔄 强制重建索引", use_container_width=True):
                with st.spinner("正在重建向量索引…"):
                    info = local_kb.build_index(force=True)
                st.success(info["message"])
        except Exception as exc:
            st.error(f"本地知识库不可用：{type(exc).__name__}: {exc}")

        st.divider()
        with st.expander("ℹ️ 使用说明"):
            st.markdown(
                "1. 输入研究问题，点 **生成研究计划**\n"
                "2. 在表格里按需修改子问题与检索词\n"
                "3. 点 **开始研究**，右侧实时显示思考过程\n"
                "4. 完成后可预览并下载报告\n\n"
                f"提示：一次完整研究通常需要 **4~7 分钟**、约 **10 万 Token**。"
            )

    return {
        "max_iterations": max_iter,
        "max_pages": max_pages,
        "engine": engine,
        "use_local_kb": use_local,
        "local_min_slots": local_slots,
        "formats": tuple(
            name for name, on in (("md", want_md), ("html", want_html), ("pdf", want_pdf)) if on
        ) or ("md",),
        "save": save_to_disk,
    }


# ========== 实时事件 ==========
def make_event_callback(placeholder, counter: dict):
    """
    返回一个事件回调：每个事件把它渲染成一行追加到 placeholder 里。
    这就是"实时展示思考过程"的实现方式——不等整条链路跑完。
    """
    lines = []

    def callback(event: dict) -> None:
        icon = EVENT_ICONS.get(event.get("status", "done"), "·")
        step = STEP_NAMES.get(event.get("step", ""), event.get("step", ""))
        elapsed = f" `{event['elapsed']}s`" if event.get("elapsed") else ""
        lines.append(f"{icon} **{step}** {event.get('title', '')}{elapsed}")
        detail = event.get("detail")
        if detail:
            lines.append(f"　　<span class='metric-note'>{str(detail)[:200]}</span>")
        counter["count"] = len(lines)
        # 只渲染最近 60 行，避免长任务把页面撑爆
        placeholder.markdown("\n\n".join(lines[-60:]), unsafe_allow_html=True)

    return callback


# ========== 计划编辑 ==========
def render_plan_editor() -> None:
    plan = st.session_state.plan
    if not plan:
        return

    st.subheader("📋 研究计划")
    if st.session_state.plan_summary:
        st.info(st.session_state.plan_summary)
    st.caption("可以直接在表格里修改子问题与检索词，修改后的计划会用于本次研究。")

    frame = pd.DataFrame([
        {
            "编号": item.get("id", ""),
            "子问题": item.get("question", ""),
            "检索词（用 / 分隔）": " / ".join(item.get("search_queries") or []),
        }
        for item in to_dicts(plan)
    ])
    edited = st.data_editor(
        frame, use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "编号": st.column_config.TextColumn(width="small", disabled=True),
            "子问题": st.column_config.TextColumn(width="large"),
            "检索词（用 / 分隔）": st.column_config.TextColumn(width="medium"),
        },
        key="plan_editor",
    )

    # 把编辑结果写回会话状态
    new_plan = []
    for index, row in edited.iterrows():
        question = str(row.get("子问题") or "").strip()
        if not question:
            continue
        queries = [q.strip() for q in str(row.get("检索词（用 / 分隔）") or "").split("/") if q.strip()]
        new_plan.append({
            "id": str(row.get("编号") or f"Q{len(new_plan) + 1}"),
            "question": question,
            "rationale": "",
            "search_queries": queries or [question],
        })
    st.session_state.plan = new_plan


# ========== 执行研究 ==========
def run_research(settings: dict) -> None:
    question = (st.session_state.get("question") or "").strip()
    if not question:
        st.warning("请先输入研究问题")
        return

    if not st.session_state.offline and not config.has_llm_key():
        st.error("未配置 DEEPSEEK_API_KEY。请填写 .env，或切换到「离线演示」模式。")
        return

    st.session_state.events = []
    st.session_state.final_state = None
    st.session_state.error = ""

    st.subheader("🧠 思考过程")
    progress = st.progress(0.0, text="准备中…")
    log_placeholder = st.empty()
    counter = {"count": 0}

    plan = st.session_state.plan or None
    plan_summary = st.session_state.plan_summary

    try:
        with st.spinner("研究进行中，请勿关闭页面…"):
            if st.session_state.offline:
                from offline_stubs import OfflineLLM, offline_search

                with offline_search(with_local=settings["use_local_kb"]) as backend:
                    agent = DeepResearchAgent(
                        llm=OfflineLLM(reflect_rounds=1),
                        max_iterations=settings["max_iterations"],
                        engine=settings["engine"],
                        max_pages_per_round=settings["max_pages"],
                        use_local_kb=settings["use_local_kb"],
                    )
                    config.LOCAL_MIN_SLOTS = settings["local_min_slots"]
                    final = agent.run(question, on_event=make_event_callback(log_placeholder, counter),
                                      plan=plan, plan_summary=plan_summary)
                st.caption(f"离线搜索被调用 {backend.search_calls} 次，本地库 {backend.local_calls} 次")
            else:
                # 二次防护：Streamlit 进程常驻，若此前有代码污染过进程级 LLM 默认值，
                # 这里强制清掉，确保真实模式一定走真实模型。
                # （模型注入本身已改为作用域级 use_llm，不会污染全局）
                from llm import reset_llm

                reset_llm()
                agent = DeepResearchAgent(
                    max_iterations=settings["max_iterations"],
                    engine=settings["engine"],
                    max_pages_per_round=settings["max_pages"],
                    use_local_kb=settings["use_local_kb"],
                )
                config.LOCAL_MIN_SLOTS = settings["local_min_slots"]
                final = agent.run(question, on_event=make_event_callback(log_placeholder, counter),
                                  plan=plan, plan_summary=plan_summary)

        progress.progress(1.0, text="完成")
        st.session_state.final_state = final
        st.session_state.events = final.get("events") or []

        if settings["save"]:
            result = DeepResearchAgent.save_run(final, formats=settings["formats"])
            st.success(f"产物已保存到 `{result['dir']}`")
            for warning in result["warnings"]:
                st.warning(warning)

    except Exception as exc:
        st.session_state.error = f"{type(exc).__name__}: {exc}"
        st.error(f"研究失败：{st.session_state.error}")
    finally:
        progress.empty()


# ========== 结果展示 ==========
def render_metrics(state: dict) -> None:
    metrics = summarize(state)
    keys = list(metrics.keys())
    # 每行 4 个指标
    for start in range(0, len(keys), 4):
        columns = st.columns(4)
        for column, key in zip(columns, keys[start:start + 4]):
            with column:
                st.metric(key, str(metrics[key]))


def render_timeline(events: list) -> None:
    """按轮次展示检索轨迹"""
    if not events:
        st.info("没有事件记录")
        return

    for step in ("planner", "searcher", "extractor", "reflector", "writer"):
        step_events = [e for e in to_dicts(events) if e.get("step") == step]
        if not step_events:
            continue
        with st.expander(f"{STEP_NAMES.get(step, step)}（{len(step_events)} 个事件）", expanded=step == "searcher"):
            for event in step_events:
                icon = EVENT_ICONS.get(event.get("status", "done"), "·")
                st.markdown(f"{icon} **{event.get('title', '')}**")
                if event.get("detail"):
                    st.caption(event["detail"])
                data = event.get("data") or {}
                if data.get("queries"):
                    st.code("\n".join(data["queries"]), language=None)


def render_reflections(reflections: list) -> None:
    if not reflections:
        st.info("没有反思记录")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "轮次": item.get("iteration"),
                "结论": "信息充分" if item.get("sufficient") else "需补充搜索",
                "覆盖率": f"{item.get('coverage', 0):.0%}",
                "补充检索词": " / ".join(item.get("new_queries") or []),
                "判断理由": str(item.get("reason", ""))[:200],
            }
            for item in to_dicts(reflections)
        ]),
        use_container_width=True, hide_index=True,
    )


def render_facts(facts: list) -> None:
    if not facts:
        st.info("没有提炼到事实")
        return
    frame = pd.DataFrame([
        {
            "编号": item.get("id"),
            "子问题": item.get("sub_question_id") or "-",
            "事实": item.get("claim", ""),
            "置信度": round(float(item.get("confidence", 0)), 2),
            "来源可信度": round(float(item.get("credibility", 0)), 2),
            "来源": item.get("source_url", ""),
        }
        for item in to_dicts(facts)
    ])
    st.caption(f"共 {len(frame)} 条事实；按来源可信度排序")
    st.dataframe(
        frame.sort_values("来源可信度", ascending=False),
        use_container_width=True, hide_index=True, height=460,
    )


def render_sources(citations: list) -> None:
    if not citations:
        st.info("没有引用来源")
        return
    web = [c for c in to_dicts(citations) if c.get("source_type") != "local"]
    local = [c for c in to_dicts(citations) if c.get("source_type") == "local"]

    if web:
        st.markdown(f"**网络来源（{len(web)} 条）**")
        st.dataframe(
            pd.DataFrame([{"编号": f"[{c['index']}]", "标题": c.get("title", ""),
                           "链接": c.get("url", "")} for c in web]),
            use_container_width=True, hide_index=True,
            column_config={"链接": st.column_config.LinkColumn(display_text="打开")},
        )
    if local:
        st.markdown(f"**本地知识库文档（{len(local)} 条）**")
        st.dataframe(
            pd.DataFrame([{"编号": f"[{c['index']}]", "文档": c.get("title", "")} for c in local]),
            use_container_width=True, hide_index=True,
        )


def render_downloads(state: dict, formats: tuple) -> None:
    markdown = state.get("report_markdown") or ""
    if not markdown:
        return

    from report import export_report, find_cjk_font, markdown_to_html

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    columns = st.columns(len(formats) + 1)
    index = 0

    if "md" in formats:
        with columns[index]:
            st.download_button("⬇️ 下载 Markdown", data=markdown.encode("utf-8"),
                               file_name=f"report_{stamp}.md", mime="text/markdown",
                               use_container_width=True)
        index += 1

    if "html" in formats:
        font_path, _ = find_cjk_font()
        html = markdown_to_html(markdown, title="研究报告", font_path=font_path,
                                font_family="ReportCJK" if font_path else None)
        with columns[index]:
            st.download_button("⬇️ 下载 HTML", data=html.encode("utf-8"),
                               file_name=f"report_{stamp}.html", mime="text/html",
                               use_container_width=True)
        index += 1

    if "pdf" in formats:
        with columns[index]:
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                result = export_report(markdown, tmp, stem="report", formats=("pdf",))
                pdf_path = result["paths"].get("pdf")
                if pdf_path:
                    payload = Path(pdf_path).read_bytes()
                    st.download_button("⬇️ 下载 PDF", data=payload,
                                       file_name=f"report_{stamp}.pdf", mime="application/pdf",
                                       use_container_width=True)
                else:
                    st.button("PDF 不可用", disabled=True, use_container_width=True,
                              help="；".join(result["warnings"]) or "未找到中文字体")
        index += 1


def render_trace(state: dict) -> None:
    """展示可观测性信息：成本核算 + span 时间线"""
    trace = state.get("trace") or {}
    if not trace:
        st.info("本次运行没有 trace 记录（可能被 TRACING_ENABLED 关闭）")
        return

    from observability import slowest_spans, trace_summary

    summary = trace_summary(trace)
    keys = list(summary.keys())
    for start in range(0, len(keys), 4):
        columns = st.columns(4)
        for column, key in zip(columns, keys[start:start + 4]):
            with column:
                st.metric(key, str(summary[key]))

    if trace.get("langfuse_enabled"):
        st.success("已上报 LangFuse" + (f"（异常：{trace['langfuse_error']}）"
                                     if trace.get("langfuse_error") else ""))
    else:
        st.caption("未配置 LangFuse Key → 仅本地追踪。配置后会自动上报，方法见 README。")

    st.divider()
    st.markdown("**最慢的步骤**（定位性能瓶颈）")
    slow = slowest_spans(trace, limit=15)
    if slow:
        st.dataframe(
            pd.DataFrame([
                {
                    "步骤": span.get("name", ""),
                    "类型": span.get("kind", ""),
                    "耗时(秒)": round(span.get("duration", 0), 2),
                    "模型": span.get("model", ""),
                    "Token": (span.get("usage") or {}).get("total_tokens", ""),
                    "成本(USD)": (span.get("cost") or {}).get("usd", ""),
                }
                for span in slow
            ]),
            use_container_width=True, hide_index=True,
        )

    with st.expander(f"完整 span 列表（{len(trace.get('spans') or [])} 条）"):
        st.dataframe(
            pd.DataFrame([
                {
                    "步骤": span.get("name", ""),
                    "类型": span.get("kind", ""),
                    "父节点": span.get("parent", "") or "（根）",
                    "耗时(秒)": round(span.get("duration", 0), 3),
                }
                for span in (trace.get("spans") or [])
            ]),
            use_container_width=True, hide_index=True, height=380,
        )


def render_result(state: dict, formats: tuple) -> None:
    st.divider()
    st.subheader("📊 运行指标")
    render_metrics(state)

    st.divider()
    tabs = st.tabs(["📄 报告", "🔍 检索轨迹", "🔁 反思轨迹", "🧾 事实清单",
                    "🔗 引用来源", "⏱️ 可观测性"])

    with tabs[0]:
        markdown = state.get("report_markdown") or ""
        if markdown:
            render_downloads(state, formats)
            st.divider()
            st.markdown(markdown)
        else:
            st.warning("本次研究没有产出报告")

    with tabs[1]:
        render_timeline(state.get("events") or [])

    with tabs[2]:
        render_reflections(state.get("reflections") or [])

    with tabs[3]:
        render_facts(state.get("facts") or [])

    with tabs[4]:
        render_sources((state.get("report") or {}).get("citations") or [])

    with tabs[5]:
        render_trace(state)


# ========== 主流程 ==========
def main() -> None:
    settings = render_sidebar()

    st.subheader("❓ 研究问题")
    # 注意：输入框必须用 key 绑定会话状态，且"示例问题"通过 on_change 回调写入。
    # 如果直接在主流程里 `st.session_state.question = pick`，赋值发生在输入框渲染之后，
    # 用户会看到"选了示例但输入框没变"，要再点一次才生效（实测踩过）。
    st.text_area(
        "输入一个需要多源检索、多步推理的复杂问题（也可以直接用下面的示例）",
        key="question",
        height=90,
        placeholder="在此输入研究问题，例如：2024 年以来国产大模型的推理成本下降了大约多少，"
                    "主要靠哪些技术手段实现？",
        label_visibility="collapsed",
    )

    examples = ["2024 年以来国产大模型的推理成本下降了大约多少，主要靠哪些技术手段实现？",
                "RAG 与长上下文大模型在处理超长文档问答时各有什么优劣？",
                "向量数据库选型（Chroma / Milvus / Qdrant / pgvector）在中小规模 RAG 场景下应如何权衡？"]
    st.selectbox(
        "或者从示例问题里挑一个（选中后会自动填入上面的输入框）",
        [EXAMPLE_PLACEHOLDER] + examples,
        key="example_pick",
        on_change=_apply_example,
        label_visibility="collapsed",
        help="选中任意一条示例，它会自动填入上面的研究问题输入框，可直接开始研究。",
    )

    st.divider()
    col_a, col_b, col_c = st.columns([1, 1, 3])

    with col_a:
        if st.button("① 生成研究计划", use_container_width=True):
            question = (st.session_state.question or "").strip()
            if not question:
                st.warning("请先输入研究问题")
            else:
                with st.spinner("正在拆解问题…"):
                    try:
                        agent = DeepResearchAgent(llm=_offline_llm_if_needed())
                        payload = agent.plan(question)
                        st.session_state.plan = payload["plan"]
                        st.session_state.plan_summary = payload.get("plan_summary", "")
                        st.session_state.final_state = None
                    except Exception as exc:
                        st.error(f"规划失败：{type(exc).__name__}: {exc}")

    with col_b:
        run_clicked = st.button("② 开始研究", type="primary", use_container_width=True)

    with col_c:
        if st.session_state.plan:
            if st.button("清空计划，让 Agent 自己规划"):
                st.session_state.plan = []
                st.session_state.plan_summary = ""

    if st.session_state.plan:
        render_plan_editor()
    else:
        st.caption("提示：也可以跳过第 ① 步，直接点「开始研究」，Agent 会一次性完成规划与检索。")

    if run_clicked:
        run_research(settings)

    if st.session_state.final_state:
        render_result(st.session_state.final_state, settings["formats"])
    else:
        st.info("研究尚未开始。点击「开始研究」后会在这里实时显示思考过程。")


def _apply_example() -> None:
    """
    下拉框的 on_change 回调：把选中的示例写进研究问题输入框。
    回调在脚本重跑**之前**执行，所以输入框渲染时就能拿到新值。
    """
    pick = st.session_state.get("example_pick", EXAMPLE_PLACEHOLDER)
    if pick and pick != EXAMPLE_PLACEHOLDER:
        st.session_state["question"] = pick


def _offline_llm_if_needed():
    """离线模式下规划也要用假模型，否则会因为没有 Key 而失败"""
    if st.session_state.get("offline"):
        from offline_stubs import OfflineLLM

        return OfflineLLM()
    return None


if __name__ == "__main__":
    main()
