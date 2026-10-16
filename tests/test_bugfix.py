"""
Bug 修复回归测试（对应真实使用中暴露的缺陷）。

    python tests/test_bugfix.py

已收录的 Bug：
  A. 假模型泄漏：`DeepResearchAgent(llm=假模型)` 早期用 set_llm 改写了**进程级全局**，
     Streamlit 常驻进程里"先跑离线演示、再切真实运行"会继续用假模型，
     造成"真搜索 + 假报告"的混合产物。
  B. 乱码：抓取 PDF / 二进制时 requests 把字节当文本解码，得到上万字符的乱码正文，
     一路流进事实 → 报告 → PDF。
  C. 子问题没有主语："背景是什么？""现状如何？""风险在哪？"这类问题脱离原问题后
     无法理解，拿去检索必然搜回无关内容。
  D. PDF 中文不折行：xhtml2pdf 只按空格断词，整段中文被画成单行冲出页面裁掉。
  F. 引用白名单传错作用域：逐章校验时误用"本章"白名单，把合法的跨章引用当成
     越界编号删掉，并让"越界引用数"指标失真。
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


# ========== Bug A：假模型不得泄漏到后续真实运行 ==========
def test_llm_scope_isolation() -> None:
    print("\n[A] 假模型的作用域隔离（曾经的严重 Bug）")
    import config
    import llm as llm_module
    from offline_stubs import OfflineLLM, offline_search
    from deep_research import DeepResearchAgent

    class FakeLLM:
        def invoke(self, messages):
            class R:
                content = '{"sub_questions": []}'
                usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return R()

    llm_module.reset_llm()
    real_before = None
    try:
        real_before = llm_module.get_llm()
    except Exception:
        real_before = "no-key"

    # 用假模型跑一次（模拟用户点了一次「离线演示」）
    with offline_search():
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0), max_iterations=1)
        agent.run("离线自检：作用域隔离")

    after = llm_module.get_llm()
    check("跑完离线研究后，进程级 LLM 没有被替换成假模型",
          not isinstance(after, OfflineLLM),
          f"进程级 LLM 变成了 {type(after).__name__}（这就是泄漏）")
    check("进程级 LLM 仍是原来的真实模型",
          type(after) is type(real_before) if real_before != "no-key" else True,
          f"{type(real_before).__name__} → {type(after).__name__}")

    # 作用域外必须拿不到假模型
    override_inside = None
    with llm_module.use_llm(FakeLLM()):
        override_inside = type(llm_module.get_llm()).__name__
    check("use_llm 作用域内生效", override_inside == "FakeLLM", str(override_inside))
    check("use_llm 作用域外自动失效",
          type(llm_module.get_llm()).__name__ != "FakeLLM",
          f"退出作用域后仍拿到 {type(llm_module.get_llm()).__name__}")

    # 嵌套/并发使用不应互相污染
    from concurrent.futures import ThreadPoolExecutor

    with llm_module.use_llm(FakeLLM()):
        with ThreadPoolExecutor(max_workers=1) as pool:
            inside_thread = pool.submit(lambda: type(llm_module.get_llm()).__name__).result()
        # 线程默认不继承 contextvar，这里只要求不崩、不污染主线程外
    check("use_llm 支持嵌套后正确恢复",
          type(llm_module.get_llm()).__name__ != "FakeLLM",
          "退出后未恢复")

    # reset_llm 能强制回到真实模型
    llm_module.set_llm(FakeLLM())
    check("set_llm 确实会改写进程级默认值（用于测试夹具）",
          type(llm_module.get_llm()).__name__ == "FakeLLM")
    llm_module.reset_llm()
    check("reset_llm 能清掉被污染的值",
          type(llm_module.get_llm()).__name__ != "FakeLLM",
          "reset_llm 没起作用")

    # DeepResearchAgent 不应调用 set_llm
    import inspect
    src = inspect.getsource(DeepResearchAgent)
    check("DeepResearchAgent 不再调用 set_llm（改用 use_llm）",
          "set_llm(" not in src and "use_llm(" in src,
          "仍在使用会污染全局的 set_llm")


# ========== Bug B：二进制 / PDF 内容必须被拦下 ==========
def test_binary_content_rejected() -> None:
    print("\n[B] 乱码来源：PDF 二进制被当作网页正文")
    from search_tools import fetch_page, looks_like_binary

    # 真实故障样本特征：大量替换字符
    garbage = "".join(chr(0xFFFD) if i % 2 else chr(0x10 + i % 80) for i in range(600))
    check("高比例替换字符被判为二进制", looks_like_binary(garbage) is True,
          "没识别出二进制内容")
    check("正常中文正文不误判",
          looks_like_binary("这是一段正常的中文网页正文，用来验证判定逻辑不会误伤。") is False)
    check("英文正文不误判",
          looks_like_binary("This is a normal English paragraph used to test the detector.") is False)
    check("空文本安全", looks_like_binary("") is False)
    check("控制字符过多被判为二进制",
          looks_like_binary("".join(chr(i % 31) for i in range(500))) is True)

    # .pdf 之类的下载链接应当直接跳过，不发请求
    for url in ["https://example.com/paper.pdf",
                "https://example.com/a.pdf?x=1",
                "https://example.com/file.docx",
                "https://example.com/archive.zip"]:
        result = fetch_page(url)
        check(f"跳过下载类链接 {url.rsplit('/', 1)[-1]}",
              result["ok"] is False and "文件下载" in result.get("error", ""),
              str(result.get("error"))[:80])

    # _clean_text 必须剔除替换字符与控制字符
    from search_tools import _clean_text
    cleaned = _clean_text("正常文字\ufffd\ufffd\x01\x02更多正常文字")
    check("清洗后不含替换字符", "\ufffd" not in cleaned, repr(cleaned))
    check("清洗后不含控制字符",
          all(ord(c) >= 32 or c in "\t\n\r" for c in cleaned), repr(cleaned))

    # _fetch_with_bs4 必须检查 Content-Type
    import inspect
    from search_tools import _fetch_with_bs4
    src = inspect.getsource(_fetch_with_bs4)
    check("兜底抓取器检查了 Content-Type", "Content-Type" in src and "content_type" in src)
    check("兜底抓取器检查了 PDF 文件头", b"%PDF".decode() in src or "%PDF" in src)


# ========== Bug C：子问题必须自带主语 ==========
def test_subquestion_self_contained() -> None:
    print("\n[C] 子问题必须自带主语")
    from planner import _GENERIC_HEADING_RE, is_self_contained

    bad_examples = [
        "背景是什么？", "现状如何？", "风险在哪？", "有哪些优势？",
        "发展趋势如何？", "影响是什么？", "优缺点", "挑战有哪些？",
        "当前背景是什么？", "整体趋势如何？", "结论是什么？",
    ]
    for text in bad_examples:
        check(f"检出无主语子问题「{text}」", is_self_contained(text) is False,
              "被判为合格（应拦截）")

    good_examples = [
        "开源大模型与闭源大模型在主流评测上的性能差距目前有多大？",
        "开源大模型在数据隐私与自主可控方面相比闭源模型有哪些优势？",
        "采用开源大模型在部署成本、运维与合规上分别面临哪些风险？",
        "MoE 架构对推理成本的影响有多大？",
        "向量数据库在中小规模 RAG 场景下应如何选型？",
    ]
    for text in good_examples:
        check(f"合格子问题通过「{text[:20]}…」", is_self_contained(text) is True,
              "被误判为不合格")

    # 程序化兜底：无主语子问题被补救后必须带上研究主体
    import llm as llm_module
    from planner import _repair_subquestion, _normalize_plan

    class NoLLM:
        def invoke(self, messages):
            raise RuntimeError("故意失败，走程序化兜底")

    with llm_module.use_llm(NoLLM()):
        fixed, queries = _repair_subquestion("开源LLM相比闭源LLM有什么优缺点", "背景是什么？")
    check("兜底修复后的子问题含研究主体",
          "开源LLM" in fixed or "闭源LLM" in fixed, fixed)
    check("兜底修复后仍可读", len(fixed) >= 10, fixed)
    check("兜底修复后给了检索词", bool(queries), str(queries))

    # 整条 _normalize_plan 链路：喂进"背景是什么？"应被自动救回
    with llm_module.use_llm(NoLLM()):
        plan, _ = _normalize_plan(
            {"sub_questions": [
                {"id": "Q1", "question": "背景是什么？", "search_queries": ["背景"]},
                {"id": "Q2", "question": "开源与闭源模型在性能上的差距有多大？",
                 "search_queries": ["开源 闭源 性能 对比"]},
                {"id": "Q3", "question": "现状如何？", "search_queries": ["现状"]},
            ]},
            "开源LLM相比闭源LLM有什么优缺点",
            llm=None,
        )
    check("整条规划链路会把无主语子问题全部救回",
          all(is_self_contained(p["question"]) for p in plan),
          str([p["question"] for p in plan]))
    check("救回后检索词不再是泛词",
          all(len(q) > 3 and not _GENERIC_HEADING_RE.match(q)
              for p in plan for q in p["search_queries"]),
          str([p["search_queries"] for p in plan]))


# ========== Bug D：PDF 中文不折行 ==========
def test_pdf_cjk_wrapping() -> None:
    """
    xhtml2pdf 不会给中文自动折行，段落会画成一整行冲出页面右边界被裁掉。
    这里用"最长行估算宽度"来验证修复：正常一行不应超过页面可用宽度。
    """
    print("\n[D] PDF 中文折行（曾经的排版 Bug）")
    import tempfile
    from pathlib import Path

    from pypdf import PdfReader

    from report import (_FRAME_WIDTH_PT, _TAG_FONT_PT, _break_cjk_text,
                        markdown_to_html, wrap_cjk_for_pdf)

    # --- 单元层：断行函数 ---
    long_cn = ("这是一段没有任何换行符也没有空格的中文长文本用来测试渲染引擎是否会自动折行"
               "如果不折行文字就会一直向右延伸直到超出页面边界并被裁掉")
    lines = _break_cjk_text(long_cn, 44)
    check("长中文被切成多行", len(lines) > 1, str(len(lines)))
    check("每行宽度都在上限内",
          all(sum(1.0 if ord(c) > 0x2E80 else 0.5 for c in ln) <= 44.5 for ln in lines),
          str([round(sum(1.0 if ord(c) > 0x2E80 else 0.5 for c in ln), 1) for ln in lines]))
    check("断行后不丢字", "".join(lines).replace(" ", "") == long_cn.replace(" ", ""),
          f"{len(''.join(lines))} vs {len(long_cn)}")
    check("行首不出现中文标点",
          not any(ln[:1] in "，。；：！？、）】》」』…" for ln in lines),
          str([ln[:1] for ln in lines]))
    check("英文单词不被拦腰截断",
          all(" " in ln or len(ln) <= 44 for ln in lines), str(lines))
    check("短文本不被切分", len(_break_cjk_text("很短的一句话。", 44)) == 1)

    # 英文为主的段落不该被手工断行（引擎自己能折）
    html_en = "<p>This is a long English paragraph that the renderer can wrap on its own at spaces.</p>"
    check("英文段落不被手工断行", wrap_cjk_for_pdf(html_en) == html_en)

    # 中文段落应被插入 <br/>
    html_cn = f"<p>{long_cn}</p>"
    wrapped = wrap_cjk_for_pdf(html_cn)
    check("中文段落被插入 <br/>", "<br/>" in wrapped, wrapped[:120])
    check("断行不破坏标签结构",
          wrapped.count("<p>") == 1 and wrapped.count("</p>") == 1, wrapped[:120])

    # 混合内容：中文段落断行、英文段落保留
    mixed = wrap_cjk_for_pdf(f"<p>{long_cn}</p><p>Short English only.</p>")
    check("混合内容只断中文段", mixed.count("<br/>") > 0 and mixed.count("<p>") == 2)

    # --- 端到端：真的渲染一份 PDF 并测量 ---
    report_md = ("# 排版回归测试\n\n## 摘要\n\n" + long_cn + "\n\n"
                 "## 1. 第一节\n\n" + long_cn.replace("渲染引擎", "排版引擎") + "\n\n"
                 "- 列表项：" + long_cn[:60] + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        from report import export_report

        result = export_report(report_md, Path(tmp), stem="t", formats=("pdf",))
        pdf_path = result["paths"].get("pdf")
        check("能生成 PDF", bool(pdf_path), str(result["warnings"]))
        if pdf_path:
            reader = PdfReader(pdf_path)
            limit_units = _FRAME_WIDTH_PT / _TAG_FONT_PT["p"]
            over = []
            for page in reader.pages:
                layout = page.extract_text(extraction_mode="layout") or ""
                for line in layout.splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    cjk = sum(1 for c in text if ord(c) > 0x2E80)
                    units = cjk + (len(text) - cjk) * 0.5
                    if units > limit_units * 1.05:
                        over.append((round(units, 1), text[:60]))
            check("渲染结果里没有超出页面宽度的行", not over,
                  f"超宽行 {len(over)} 条：{over[:2]}")
            text_all = "".join((p.extract_text() or "") for p in reader.pages)
            check("长段落内容完整未被裁掉",
                  "超出页面边界并被裁掉" in text_all.replace("\n", ""),
                  "关键片段在 PDF 里找不到了")
            check("生成 PDF 不报字体告警", not result["warnings"], str(result["warnings"]))

            # 字体覆盖校验必须能識别"样本缺失"，而不是只看第一个样本是否命中
            import report as report_module

            html_ok = report_module.markdown_to_html(
                report_md, title="排版回归测试", font_path=report_module.find_cjk_font()[0],
                font_family=report_module.PDF_FONT_NAME, for_pdf=True,
            )
            check("字符齐全时校验通过",
                  report_module._verify_cjk_text(pdf_path, html_ok) == "",
                  "正常文档被判为缺字")
            # 把正文里出现最多的一段中文整体换成 PDF 里不存在的生僻字，
            # 制造"大面积缺字"，校验必须报错（早期只看单个样本命中，会漏判）
            tampered = html_ok.replace("这是一段没有任何换行符", "齉龘靐齉龘齉龘靐齉龘齉龘")
            check("篡改确实命中了正文", tampered != html_ok, "替换目标不在文档里，用例本身无效")
            check("样本大面积缺失时校验能报错",
                  report_module._verify_cjk_text(pdf_path, tampered) != "",
                  "缺字样本未被检出，PDF 方框字会漏判")


# ========== Bug F：引用白名单传错作用域（合法跨章引用被删除）==========
class _CrossCiteLLM:
    """
    故意让"第一章"的正文**跨章引用**第二章来源的编号 [2]，并额外编造一个 [99]。
    用于验证：合法编号必须保留，只有编造编号才被剔除。
    """

    def __init__(self, facts_by_id: dict):
        self.facts_by_id = facts_by_id
        self.section_calls = 0

    def invoke(self, messages):
        text = messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1])

        class _R:
            def __init__(self, content: str):
                self.content = content
                self.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        if '"sections"' in text:
            return _R(json.dumps({
                "title": "跨章引用回归",
                "summary_points": ["要点一"],
                "sections": [
                    {"heading": "第一章", "focus": "先讲 A", "fact_ids": ["F1"]},
                    {"heading": "第二章", "focus": "再讲 B", "fact_ids": ["F2"]},
                ],
            }, ensure_ascii=False))

        if '"content"' in text:
            self.section_calls += 1
            # 本章只分到 1 条事实，但正文故意同时引用两个**全局合法**编号 + 一个编造编号
            return _R(json.dumps({
                "content": "本章结论引用本章来源[1]，并跨章引用[2]；此处另有一个编造编号[99]。"
            }, ensure_ascii=False))

        if '"conclusion"' in text:
            return _R(json.dumps({"summary": "摘要[1]", "conclusion": "结论[2]"},
                                 ensure_ascii=False))

        return _R("{}")


def test_citation_scope_is_correct() -> None:
    """
    曾经的问题：逐章写作时把**本章**的引用白名单传给了 sanitize_citations，
    于是模型跨章引用另一条同样合法的来源时，该引用被当作"越界编号"从正文里删掉，
    还被计入"越界引用数"。因为写作 Prompt 恰好只下发本章编号，
    这个问题长期没有被真实运行触发，指标恒为 0。
    """
    print("\n[F] 引用白名单必须用全局范围（跨章引用是合法的）")
    import writer

    facts = [
        {"id": "F1", "claim": "事实一", "source_url": "https://a.example.com/1",
         "source_title": "来源A", "confidence": 0.9, "credibility": 0.9},
        {"id": "F2", "claim": "事实二", "source_url": "https://b.example.com/2",
         "source_title": "来源B", "confidence": 0.9, "credibility": 0.6},
        {"id": "F3", "claim": "事实三", "source_url": "https://c.example.com/3",
         "source_title": "来源C", "confidence": 0.8, "credibility": 0.5},
    ]
    citations, fact_to_cite = writer.build_citations(facts)
    check("三个来源各拿到一个引用号",
          sorted(c["index"] for c in citations) == [1, 2, 3],
          str(citations))

    state = {"question": "跨章引用回归问题", "stats": {}, "facts": facts,
             "plan_summary": "", "plan": []}
    stub = _CrossCiteLLM({f["id"]: f for f in facts})
    out = writer.build_report(state, llm=stub)
    report, stats = out["report"], out["stats"]
    body = "\n".join(s["content"] for s in report["sections"])

    check("第一章确实跨章引用了 [2]", "[2]" in body, body)
    check("跨章引用的合法编号未被删除", "[2]" in body and "[1]" in body, body)
    check("编造的 [99] 被剔除", "[99]" not in body, body)
    check("越界引用数只统计编造编号（去重后为 1）",
          stats["invalid_citation_count"] == 1, str(stats.get("invalid_citations")))
    check("越界明细指向编造的编号 99",
          set(stats.get("invalid_citations") or []) == {99},
          str(stats.get("invalid_citations")))
    check("保留下来的来源编号连续无断号",
          [c["index"] for c in report["citations"]] == [1, 2],
          str([c["index"] for c in report["citations"]]))
    check("未被正文引用的 F3 来源被剔除（重编号的副作用）",
          all("c.example.com" not in c["url"] for c in report["citations"]),
          str([c["url"] for c in report["citations"]]))
    check("正文引用号都在参考文献范围内",
          set(writer.find_citations(body)) <= {c["index"] for c in report["citations"]},
          body)


# ========== 端到端：离线链路仍然正常 ==========
def test_end_to_end_still_works() -> None:
    print("\n[E] 修复后端到端仍然正常")
    from deep_research import DeepResearchAgent
    from offline_stubs import OfflineLLM, offline_search
    from planner import is_self_contained

    with offline_search():
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=1), max_iterations=2)
        final = agent.run("开源LLM相比闭源LLM有什么优缺点？")

    plan = final.get("plan") or []
    check("离线链路仍能产出计划", len(plan) >= 3, str(len(plan)))
    check("离线假计划也带主语",
          all(is_self_contained(p["question"]) for p in plan),
          str([p["question"] for p in plan]))
    check("离线假计划里含研究主体",
          any("开源" in p["question"] for p in plan),
          str([p["question"] for p in plan]))
    check("仍能产出报告", bool(final.get("report_markdown")))
    check("报告里不出现旧的假子问题文案",
          "离线自检问题一：背景是什么" not in (final.get("report_markdown") or ""))


def main() -> int:
    print("=" * 64)
    print("Bug 修复回归测试：假模型泄漏 / 乱码 / 子问题无主语 / 跨章引用")
    print("=" * 64)
    test_llm_scope_isolation()
    test_binary_content_rejected()
    test_subquestion_self_contained()
    test_pdf_cjk_wrapping()
    test_citation_scope_is_correct()
    test_end_to_end_still_works()

    print("\n" + "=" * 64)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 64)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
