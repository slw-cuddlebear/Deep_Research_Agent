"""
模块 5：报告生成器（Writer）
输入：全部结构化事实
输出：结构化 Markdown 研究报告（摘要 / 正文 / 结论 / 参考文献）

引用溯源的设计（这是本模块最关键的部分）：
  1. **按来源编号，而不是按事实编号**——同一网页提炼出的多条事实共用一个引用号，
     这样参考文献里不会出现同一个链接重复列 5 次。
  2. 写作时把"引用号"直接喂给 LLM，要求它写 [1]、[2]，而不是让模型自由发挥。
  3. 写完后**重新编号**：只保留正文真正引用过的来源，按首次出现顺序连续编号 1..K，
     避免出现 [1][3][7] 这种断号，同时天然剔除"提取了却没用上"的来源。
  4. 越界引用号（模型编造的 [99]）会被剔除并记录，按**去重后的编号个数**计入
     `invalid_citation_count`（同一个编造编号在多章出现只算一个）。
     注意：判断"越界"必须拿**全局**白名单，不能只看本章白名单——跨章引用是合法的。
"""
import re
from urllib.parse import urlparse

import config
import llm as llm_module
import prompts
from events import Timer, log_event
from observability import traced_node
from state import Citation, ResearchReport, to_dicts, now_str

_CITE_RE = re.compile(r"\[(\d{1,3})\]")
# 同时匹配半角与全角方括号的引用写法
_CITE_LOOSE_RE = re.compile(r"[\[【](\d{1,3})[\]】]")


# ========== 引用编号 ==========
def citation_key(url: str) -> str:
    """
    引用分组键。

    本地知识库的 source_url 形如 `local://文档.md#chunk-3`，
    如果按整串分组，**同一篇文档的每个片段都会各自拿到一个引用号**，
    参考文献里就会出现三条一模一样的记录（实测踩过）。
    所以本地来源要按"文档"归组，去掉片段号。
    """
    url = (url or "").strip()
    if url.startswith("local://"):
        return url.split("#")[0]
    return url


def build_citations(facts: list) -> tuple:
    """
    按来源分组分配引用编号。

    :return: (citations 列表, fact_id -> 引用编号 的映射)
    """
    items = to_dicts(facts)
    groups = {}
    for fact in items:
        url = (fact.get("source_url") or "").strip()
        key = citation_key(url) or f"__no_source__{fact.get('id')}"
        groups.setdefault(key, []).append(fact)

    # 可信度高、置信度高、事实条数多的来源排在前面
    def rank(group):
        return (
            max(f.get("credibility", 0.5) for f in group),
            max(f.get("confidence", 0.5) for f in group),
            len(group),
        )

    ordered = sorted(groups.values(), key=rank, reverse=True)[: config.MAX_CITATIONS]

    citations, fact_to_cite = [], {}
    for index, group in enumerate(ordered, start=1):
        head = group[0]
        url = head.get("source_url") or ""
        title = head.get("source_title") or (urlparse(url).netloc if url else "未知来源")
        is_local = url.startswith("local://")
        citations.append(
            Citation(
                index=index,
                title=_clean_local_title(title) if is_local else title.strip(),
                url=url,
                source_type="local" if is_local else "web",
                used_by=[],
            ).model_dump()
        )
        for fact in group:
            fact_to_cite[fact.get("id")] = index
    return citations, fact_to_cite


def _clean_local_title(title: str) -> str:
    """本地片段标题去掉"（本地知识库 第 N 段）"后缀，参考文献里只留文档名"""
    return re.sub(r"（本地知识库.*?）\s*$", "", (title or "").strip()).strip() or title


def local_file_name(url: str) -> str:
    """从 local:// 链接里取出文件名"""
    return (url or "").replace("local://", "").split("#")[0]


def format_citation_entry(citation: dict) -> str:
    """
    渲染一条参考文献。
    网络来源给可点击链接；本地文档给文件名——把 local:// 暴露给读者没有意义。
    """
    index = citation.get("index")
    title = citation.get("title") or "未命名来源"
    url = citation.get("url") or ""
    if citation.get("source_type") == "local" or url.startswith("local://"):
        name = local_file_name(url)
        if not name or name == title:
            return f"[{index}] {name or title}（本地知识库文档）"
        return f"[{index}] {title}（本地知识库：{name}）"
    return f"[{index}] {title} — {url}" if url else f"[{index}] {title}"


def render_facts_for_prompt(facts: list, fact_to_cite: dict) -> str:
    """把事实渲染成 [引用号] 陈述 的清单，供大纲与写作 Prompt 使用"""
    lines = []
    for fact in to_dicts(facts):
        cite = fact_to_cite.get(fact.get("id"))
        tag = f"[{cite}]" if cite else "[未编号]"
        credibility = fact.get("credibility", 0.5)
        lines.append(
            f"{tag} ({fact.get('id')}) {fact.get('claim', '')}"
            f"　｜来源可信度 {credibility:.1f}"
        )
    return "\n".join(lines) or "（无可用事实）"


def render_cited_facts(facts: list, fact_to_cite: dict) -> str:
    """写作章节时用的事实清单：只带引用号与陈述，减少干扰信息"""
    lines = []
    for fact in to_dicts(facts):
        cite = fact_to_cite.get(fact.get("id"))
        if not cite:
            continue
        lines.append(f"[{cite}] {fact.get('claim', '')}")
    return "\n".join(lines) or "（无可用事实）"


# ========== 引用校验与重编号 ==========
def find_citations(text: str) -> list:
    """按出现顺序提取正文里的引用号"""
    return [int(m.group(1)) for m in _CITE_RE.finditer(text or "")]


def sanitize_citations(text: str, allowed: set) -> tuple:
    """
    剔除越界引用号（模型编造的编号）。
    :return: (清理后的文本, 被剔除的编号列表)
    """
    removed = []

    def replace(match):
        number = int(match.group(1))
        if number in allowed:
            return match.group(0)
        removed.append(number)
        return ""

    return _CITE_LOOSE_RE.sub(replace, text or ""), removed


def renumber_citations(texts: list, allowed: set) -> tuple:
    """
    按"首次出现顺序"把引用号重排成连续的 1..K。
    :return: (重排后的文本列表, {旧号: 新号}, 实际用到的旧号顺序)
    """
    order = []
    for text in texts:
        for number in find_citations(text):
            if number in allowed and number not in order:
                order.append(number)

    mapping = {old: new for new, old in enumerate(order, start=1)}

    def replace(match):
        number = int(match.group(1))
        return f"[{mapping[number]}]" if number in mapping else ""

    return [_CITE_RE.sub(replace, text or "") for text in texts], mapping, order


# ========== LLM 调用 ==========
def plan_outline(question: str, plan_summary: str, facts: list, fact_to_cite: dict,
                 llm=None) -> dict:
    """生成报告大纲"""
    prompt = prompts.OUTLINE_PROMPT.format(
        question=question,
        plan_summary=plan_summary or "（无）",
        facts_brief=render_facts_for_prompt(facts, fact_to_cite),
        min_sections=config.MIN_SECTIONS,
        max_sections=config.MAX_SECTIONS,
    )
    raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_WRITER, llm=llm)
    if not isinstance(raw, dict):
        raw = {"sections": raw}
    return raw


def normalize_outline(raw: dict, facts: list, question: str) -> dict:
    """
    规整大纲，并保证"事实被章节覆盖"：
      - 章节标题为空 / 章节数不足 → 补默认章节
      - fact_ids 不存在 → 丢弃；某章节没分到事实 → 从尚未分配的事实里补齐
      - 有事实没被任何章节引用 → 追加一节"补充发现"
    """
    items = to_dicts(facts)
    valid_ids = [f.get("id") for f in items if f.get("id")]

    title = str(raw.get("title") or "").strip() or f"研究报告：{question[:40]}"
    points = raw.get("summary_points") or []
    if isinstance(points, str):
        points = [points]

    raw_sections = raw.get("sections") or raw.get("outline") or []
    sections, assigned = [], set()
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or item.get("title") or "").strip()
        if not heading:
            continue
        ids = [str(i) for i in (item.get("fact_ids") or []) if str(i) in valid_ids]
        sections.append({
            "heading": heading,
            "focus": str(item.get("focus") or "").strip(),
            "fact_ids": [i for i in ids if i not in assigned],
        })
        assigned.update(sections[-1]["fact_ids"])
        if len(sections) >= config.MAX_SECTIONS:
            break

    # 章节数不足时补齐：把剩余事实**尽量均匀**分给新章节。
    # 注意：宁可少写一章，也不造一个没有任何事实支撑的空章节。
    leftovers = [i for i in valid_ids if i not in assigned]
    while len(sections) < config.MIN_SECTIONS and leftovers:
        sections_needed = config.MIN_SECTIONS - len(sections)
        per_section = max(1, len(leftovers) // sections_needed)
        take = leftovers[:per_section]
        sections.append({
            "heading": f"补充发现（{len(sections) + 1}）",
            "focus": "汇总尚未纳入正文的其余发现",
            "fact_ids": take,
        })
        assigned.update(take)
        leftovers = [i for i in leftovers if i not in assigned]

    # 仍未被任何章节引用的事实 → 追加"补充发现"章节，避免提炼出的事实被浪费
    leftovers = [i for i in valid_ids if i not in assigned]
    if leftovers:
        sections.append({
            "heading": "补充发现",
            "focus": "正文之外但值得记录的其余发现",
            "fact_ids": leftovers,
        })

    # 有章节没分到事实 → 从剩余事实里尽量补
    if valid_ids and sections:
        pool = [i for i in valid_ids if i not in assigned]
        for section in sections:
            if not section["fact_ids"] and pool:
                take = pool[: config.MIN_FACTS_PER_SECTION]
                section["fact_ids"] = take
                assigned.update(take)
                pool = [i for i in pool if i not in assigned]

    # 极端情况：一条事实都没有（或全被过滤），至少要有一个章节承接"资料不足"的说明
    if not sections:
        sections.append({
            "heading": "现有资料不足",
            "focus": "说明为何无法形成结论",
            "fact_ids": [],
        })

    return {
        "title": title,
        "summary_points": [str(p).strip() for p in points if str(p).strip()],
        "sections": sections,
    }


def demote_headings(text: str) -> str:
    """
    把章节正文里的标题整体降一级。
    模型写正文时常用 "## 一、xxx" 做小标题，而章节标题本身已经是 "## N. xxx"，
    不降级就会出现"同级标题套同级标题"，目录结构会乱掉。
    代码块里的内容不动（# 在代码里可能是有意义的注释）。
    """
    lines, in_fence = [], False
    for line in (text or "").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        if not in_fence and re.match(r"^#{1,5}\s", line):
            line = "#" + line
        lines.append(line)
    return "\n".join(lines)


def write_section(question: str, title: str, section: dict, facts: list,
                  allowed_ids: set, llm=None) -> str:
    """撰写单章正文"""
    prompt = prompts.WRITER_SECTION_PROMPT.format(
        question=question,
        title=title,
        heading=section.get("heading", ""),
        focus=section.get("focus") or "围绕本章标题展开",
        facts=render_cited_facts(facts, {f.get("id"): f.get("_cite") for f in to_dicts(facts)}),
        allowed_ids="、".join(f"[{i}]" for i in sorted(allowed_ids)) or "（无）",
    )
    try:
        raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_WRITER, llm=llm)
        content = raw.get("content") if isinstance(raw, dict) else raw
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = str(content or "").strip()
    except Exception:
        content = ""

    if not content:
        # 写作失败时退化为"事实清单"，至少保证引用可溯源，不产生编造内容
        lines = ["现有资料尚不足以完整回答该问题，以下为已核实的事实：", ""]
        for fact in to_dicts(facts):
            cite = fact.get("_cite")
            tag = f"[{cite}]" if cite else ""
            lines.append(f"- {fact.get('claim', '')} {tag}")
        content = "\n".join(lines)
    return demote_headings(content)


def write_summary(question: str, title: str, body: str, allowed_ids: set, llm=None) -> dict:
    """撰写摘要与结论"""
    prompt = prompts.SUMMARY_PROMPT.format(
        question=question,
        title=title,
        body=body[:6000],
        allowed_ids="、".join(f"[{i}]" for i in sorted(allowed_ids)) or "（无）",
    )
    try:
        raw = llm_module.call_llm_json(prompt, system=prompts.SYSTEM_WRITER, llm=llm)
        if isinstance(raw, dict):
            return {
                "summary": str(raw.get("summary") or "").strip(),
                "conclusion": str(raw.get("conclusion") or "").strip(),
            }
    except Exception:
        pass
    return {"summary": "", "conclusion": ""}


# ========== Markdown 渲染 ==========
def render_report_markdown(report: dict, state: dict = None, include_process: bool = True) -> str:
    """把结构化报告渲染成 Markdown 全文"""
    lines = [f"# {report.get('title', '研究报告')}", ""]
    lines.append(f"> **研究问题**：{report.get('question', '')}")
    lines.append(f"> **生成时间**：{report.get('created_at') or now_str()}")

    if state:
        stats = state.get("stats") or {}
        pages = to_dicts(state.get("pages"))
        web_pages = len([p for p in pages if p.get("source") != "local"])
        local_pages = len([p for p in pages if p.get("source") == "local"])
        scale = (f"{state.get('iteration', 1)} 轮检索 ｜ "
                 f"{len(stats.get('searched_queries') or [])} 个关键词 ｜ "
                 f"{web_pages} 篇网页")
        if local_pages:
            scale += f" + {local_pages} 段本地知识库"
        scale += f" ｜ {len(to_dicts(state.get('facts')))} 条事实"
        lines.append(f"> **检索规模**：{scale}")
    lines.append("")

    if report.get("summary"):
        lines += ["## 摘要", "", report["summary"], ""]

    for idx, section in enumerate(report.get("sections") or [], start=1):
        lines += [f"## {idx}. {section.get('heading', '')}", "", section.get("content", ""), ""]

    if report.get("conclusion"):
        lines += ["## 结论", "", report["conclusion"], ""]

    citations = report.get("citations") or []
    if citations:
        web_cites = [c for c in citations if c.get("source_type") != "local"]
        local_cites = [c for c in citations if c.get("source_type") == "local"]
        lines += ["## 参考文献", ""]
        if web_cites and local_cites:
            # 两类来源分开列，但引用编号保持全局连续
            lines += ["### 网络来源", ""]
            lines += [format_citation_entry(c) for c in web_cites]
            lines += ["", "### 本地知识库文档", ""]
            lines += [format_citation_entry(c) for c in local_cites]
        else:
            lines += [format_citation_entry(c) for c in citations]
        lines.append("")

    if include_process and state:
        lines += _render_process(state)

    return "\n".join(lines).rstrip() + "\n"


def _render_process(state: dict) -> list:
    """渲染"附：研究过程"，让读者能看到 Agent 是怎么查的"""
    stats = state.get("stats") or {}
    lines = ["---", "", "## 附：研究过程", "", "**研究计划**", ""]
    for item in to_dicts(state.get("plan")):
        lines.append(f"- {item.get('id')}：{item.get('question', '')}")

    lines += ["", "**检索轨迹**", ""]
    for idx, reflection in enumerate(to_dicts(state.get("reflections")), start=1):
        verdict = "信息充分" if reflection.get("sufficient") else "需补充搜索"
        lines.append(f"- 第 {idx} 轮反思：{verdict}（覆盖率 {reflection.get('coverage', 0):.0%}）"
                     f" — {reflection.get('reason', '')}")
        for gap in reflection.get("gaps") or []:
            lines.append(f"  - 缺口：{gap}")

    usage = stats.get("llm_usage") or {}
    lines += [
        "",
        "**执行统计**",
        "",
        f"- 搜索轮数：{state.get('iteration', 1)}",
        f"- 检索关键词：{len(stats.get('searched_queries') or [])} 个",
        f"- 抓取网页：{stats.get('pages_fetched', 0)} 篇成功 / {stats.get('pages_failed', 0)} 篇失败",
        f"- 提炼事实：{len(to_dicts(state.get('facts')))} 条",
        f"- 识别矛盾：{len(stats.get('conflicts') or [])} 组",
        f"- LLM 调用：{usage.get('calls', 0)} 次，Token {usage.get('total_tokens', 0)}",
        f"- 端到端耗时：{stats.get('elapsed', 0)} 秒",
    ]
    return lines


# ========== 节点 ==========
def build_report(state: dict, llm=None) -> dict:
    """
    完整的报告生成流程：编号 → 大纲 → 逐章写作 → 摘要结论 → 重编号 → 渲染 Markdown。
    返回 {"report": {...}, "report_markdown": "...", "stats": {...}, "events": [...]}
    """
    question = state.get("question", "")
    facts = to_dicts(state.get("facts"))
    stats = dict(state.get("stats") or {})
    events = []

    # ---- 无事实兜底：如实说明，不编造 ----
    if not facts:
        report = ResearchReport(
            title=f"研究报告：{question[:40]}",
            question=question,
            summary="本次研究未能获取到可用的事实信息，因此无法形成结论。",
            sections=[{
                "heading": "研究未能完成",
                "content": "所有检索与网页抓取均未产出可用事实，可能原因：\n"
                           "- 搜索后端不可用（未配置 TAVILY_API_KEY 且 DuckDuckGo 不可达）\n"
                           "- 网页抓取被目标站点拦截\n"
                           "- 检索词过于生僻\n\n"
                           "建议检查 `python main.py --doctor` 的输出后重试。",
            }],
            conclusion="研究未能完成，无结论可给出。",
            citations=[],
        ).model_dump()
        markdown = render_report_markdown(report, state)
        return {
            "report": report, "report_markdown": markdown, "stats": stats,
            "events": [log_event(state, "writer", "没有可用事实，生成失败说明报告", status="warn")],
        }

    # ---- 1. 分配引用号 ----
    citations, fact_to_cite = build_citations(facts)
    for fact in facts:
        fact["_cite"] = fact_to_cite.get(fact.get("id"))
    allowed = {c["index"] for c in citations}

    with Timer() as timer:
        # ---- 2. 大纲 ----
        try:
            outline = normalize_outline(plan_outline(question, state.get("plan_summary", ""),
                                                     facts, fact_to_cite, llm=llm), facts, question)
        except Exception as exc:
            outline = normalize_outline({}, facts, question)
            events.append(log_event(state, "writer", "大纲生成失败，改用默认章节结构",
                                    str(exc)[:200], status="warn"))

        # ---- 3. 逐章写作 ----
        sections = []
        for section in outline["sections"]:
            section_facts = [f for f in facts if f.get("id") in set(section["fact_ids"])]
            section_allowed = {fact_to_cite[f.get("id")] for f in section_facts
                               if f.get("id") in fact_to_cite}
            content = write_section(question, outline["title"], section, section_facts,
                                    section_allowed, llm=llm)
            # section_allowed 只用来压缩**下发给模型的事实清单**（让 Prompt 短一些）；
            # 校验必须用全局白名单 allowed。早期这里把 section_allowed 也传给了
            # sanitize_citations（现已改为下方传 allowed），导致模型跨章引用另一条
            # 同样合法的来源时，被当成"越界编号"从正文学掉并误计入越界引用数。
            content, removed = sanitize_citations(content, allowed)
            if removed:
                stats.setdefault("invalid_citations", []).extend(removed)
            sections.append({"heading": section["heading"], "content": content})

        body = "\n\n".join(f"## {s['heading']}\n\n{s['content']}" for s in sections)
        summary_payload = write_summary(question, outline["title"], body, allowed, llm=llm)

    summary, _ = sanitize_citations(summary_payload.get("summary", ""), allowed)
    conclusion, _ = sanitize_citations(summary_payload.get("conclusion", ""), allowed)

    # ---- 4. 重编号：只保留正文真正引用过的来源，连续编号 1..K ----
    texts = [summary] + [s["content"] for s in sections] + [conclusion]
    (summary, *rest), mapping, order = renumber_citations(texts, allowed)
    for section, content in zip(sections, rest[:-1]):
        section["content"] = content
    conclusion = rest[-1]

    final_citations = []
    for citation in citations:
        new_index = mapping.get(citation["index"])
        if not new_index:
            continue
        item = dict(citation)
        item["index"] = new_index
        final_citations.append(item)
    final_citations.sort(key=lambda c: c["index"])

    # 完全没用上任何引用时，保留全部来源，避免参考文献为空
    if not final_citations:
        final_citations = citations
        events.append(log_event(state, "writer", "正文未产生任何引用标记，已保留全部来源",
                                status="warn"))

    report = ResearchReport(
        title=outline["title"],
        question=question,
        summary=summary,
        sections=sections,
        conclusion=conclusion,
        citations=[Citation(**c) for c in final_citations],
    ).model_dump()
    markdown = render_report_markdown(report, state)

    # ---- 5. 统计 ----
    cited_indexes = set(find_citations(markdown.split("## 参考文献")[0]))
    stats["report_sections"] = len(sections)
    stats["citations"] = len(final_citations)
    stats["citations_used"] = len(cited_indexes)
    stats["citation_coverage"] = (
        round(len(cited_indexes) / len(final_citations), 3) if final_citations else 0.0
    )
    stats["invalid_citation_count"] = len(set(stats.get("invalid_citations") or []))
    stats["report_chars"] = len(markdown)

    events.append(
        log_event(
            state, "writer",
            f"生成报告：{len(sections)} 章 / {len(final_citations)} 条引用",
            f"正文 {len(markdown)} 字符；越界引用 {stats['invalid_citation_count']} 处",
            elapsed=timer.elapsed,
            citations=len(final_citations), sections=len(sections),
        )
    )
    return {"report": report, "report_markdown": markdown, "stats": stats, "events": events}


@traced_node("writer")
def writer_node(state: dict) -> dict:
    """LangGraph 节点：整合事实，生成带引用的结构化报告"""
    return build_report(state)
