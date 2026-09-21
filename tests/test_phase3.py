"""
Phase 3 自检脚本（不需要 API Key，也不联网）。
运行方式：
    python tests/test_phase3.py

覆盖内容：
    1. 引用编号：按来源（而非按事实）分组分配
    2. 引用校验与重编号：越界引用剔除、断号压缩成连续 1..K
    3. 大纲规整：章节数补齐、事实全覆盖、无事实章节补料
    4. 报告 Markdown 渲染结构
    5. 导出：Markdown / HTML / PDF（PDF 依赖系统中文字体，探测不到则跳过并提示）
    6. 离线端到端：跑完反思循环后确实产出带引用的报告
"""
import os
import re
import sys
import tempfile

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


def make_facts() -> list:
    """三条事实，分布在两个来源上（用于验证"按来源编号"）"""
    return [
        {"id": "F1", "claim": "事实一：成本下降了 30%。", "source_url": "https://www.nature.com/a",
         "source_title": "Nature 文章", "confidence": 0.9, "credibility": 0.9,
         "sub_question_id": "Q1"},
        {"id": "F2", "claim": "事实二：该方案已被广泛部署。", "source_url": "https://www.nature.com/a",
         "source_title": "Nature 文章", "confidence": 0.8, "credibility": 0.9,
         "sub_question_id": "Q1"},
        {"id": "F3", "claim": "事实三：延迟与吞吐存在权衡。", "source_url": "https://blog.example.com/b",
         "source_title": "技术博客", "confidence": 0.7, "credibility": 0.5,
         "sub_question_id": "Q2"},
    ]


# ========== 1. 引用编号 ==========
def test_citation_build() -> None:
    print("\n[1] 引用编号分配")
    from writer import build_citations, render_cited_facts, render_facts_for_prompt

    citations, fact_to_cite = build_citations(make_facts())
    check("按来源分组：3 条事实 → 2 条引用", len(citations) == 2, f"实际 {len(citations)}")
    check("同一来源的两条事实共用引用号",
          fact_to_cite.get("F1") == fact_to_cite.get("F2"), str(fact_to_cite))
    check("不同来源引用号不同", fact_to_cite.get("F1") != fact_to_cite.get("F3"), str(fact_to_cite))
    check("高可信来源排在前（Nature 拿 [1]）", fact_to_cite.get("F1") == 1, str(fact_to_cite))
    check("引用条目带标题与链接",
          citations[0]["title"] == "Nature 文章" and citations[0]["url"].startswith("https://"))
    check("引用编号从 1 开始连续", [c["index"] for c in citations] == [1, 2], str(citations))

    prompt_text = render_facts_for_prompt(make_facts(), fact_to_cite)
    check("大纲事实清单带引用号与事实号",
          "[1]" in prompt_text and "(F1)" in prompt_text, prompt_text)
    cited = render_cited_facts(make_facts(), fact_to_cite)
    check("写作事实清单只保留引用号与陈述",
          cited.count("[") == 3 and "F1" not in cited, cited)

    # 没有来源链接的事实也要能编号，不能丢
    orphan = [{"id": "F9", "claim": "无来源事实", "confidence": 0.5, "credibility": 0.5}]
    orphan_citations, orphan_map = build_citations(orphan)
    check("无来源事实也能分配引用号",
          len(orphan_citations) == 1 and orphan_map.get("F9") == 1, str(orphan_map))


# ========== 2. 引用校验与重编号 ==========
def test_citation_sanitize() -> None:
    print("\n[2] 引用校验与重编号")
    from writer import find_citations, renumber_citations, sanitize_citations

    text = "结论一[1]，结论二[2]，结论三[9]。"
    cleaned, removed = sanitize_citations(text, {1, 2, 3})
    check("越界引用号被剔除", "[9]" not in cleaned, cleaned)
    check("被剔除的编号被记录", removed == [9], str(removed))
    check("合法引用号保留", "[1]" in cleaned and "[2]" in cleaned, cleaned)

    full_width, _ = sanitize_citations("全角引用【1】和【7】", {1})
    check("全角方括号引用也能识别与剔除",
          "【1】" in full_width and "【7】" not in full_width, full_width)

    check("按出现顺序提取引用号", find_citations("a[3]b[1]c[3]") == [3, 1, 3],
          str(find_citations("a[3]b[1]c[3]")))

    # 断号压缩：[3][7] → [1][2]
    texts = ["甲[3]乙", "丙[7]丁[3]"]
    new_texts, mapping, order = renumber_citations(texts, {3, 7})
    check("引用号按首次出现顺序重排", new_texts == ["甲[1]乙", "丙[2]丁[1]"], str(new_texts))
    check("重编号映射正确", mapping == {3: 1, 7: 2}, str(mapping))
    check("使用顺序被记录", order == [3, 7], str(order))

    # 未使用的引用号不应出现在结果里
    _, unused_map, _ = renumber_citations(["只有[5]"], {5, 6})
    check("未使用的引用号不参与重编号", unused_map == {5: 1}, str(unused_map))


# ========== 3. 大纲规整 ==========
def test_outline_normalize() -> None:
    print("\n[3] 大纲规整")
    from writer import normalize_outline

    facts = make_facts()

    # 章节数不足 + fact_ids 不存在 + 有事实未被覆盖
    raw = {"title": "", "sections": [{"heading": "第一章", "fact_ids": ["F1", "F99"]}]}
    outline = normalize_outline(raw, facts, "测试问题")
    headings = [s["heading"] for s in outline["sections"]]
    check("标题为空时自动补标题", outline["title"].startswith("研究报告"), outline["title"])
    check("章节数补齐到下限", len(outline["sections"]) >= 3, str(headings))
    check("不存在的事实编号被丢弃",
          all("F99" not in s["fact_ids"] for s in outline["sections"]), str(outline["sections"]))

    covered = {i for s in outline["sections"] for i in s["fact_ids"]}
    check("所有事实都被至少一个章节覆盖", covered == {"F1", "F2", "F3"}, str(covered))

    # 每个章节都不该是空的
    check("每个章节都分到了事实",
          all(s["fact_ids"] for s in outline["sections"]), str(outline["sections"]))

    # fact_ids 重复出现在多个章节时应只归第一个
    raw2 = {"sections": [{"heading": "A", "fact_ids": ["F1", "F2"]},
                         {"heading": "B", "fact_ids": ["F1", "F2", "F3"]},
                         {"heading": "C", "fact_ids": []}]}
    outline2 = normalize_outline(raw2, facts, "测试问题")
    first_two = outline2["sections"][:2]
    check("事实编号不会重复分配到两个章节",
          not (set(first_two[0]["fact_ids"]) & set(first_two[1]["fact_ids"])), str(first_two))

    # 空输入也不能崩
    empty = normalize_outline({}, facts, "测试问题")
    check("空大纲也能生成可用结构", len(empty["sections"]) >= 3, str(empty["sections"]))


# ========== 4. 报告渲染 ==========
def test_render() -> None:
    print("\n[4] 报告 Markdown 渲染")
    import json
    from writer import render_report_markdown

    report = {
        "title": "测试报告", "question": "测试问题", "created_at": "2025-01-01 00:00:00",
        "summary": "这是摘要[1]。",
        "sections": [{"heading": "背景", "content": "正文内容[1]。"},
                     {"heading": "发现", "content": "发现内容[2]。"}],
        "conclusion": "这是结论[1]。",
        "citations": [{"index": 1, "title": "来源一", "url": "https://a.com"},
                      {"index": 2, "title": "来源二", "url": "https://b.com"}],
    }
    state = {
        "plan": [{"id": "Q1", "question": "子问题一"}],
        "reflections": [{"sufficient": True, "coverage": 0.9, "reason": "信息充分", "gaps": []}],
        "iteration": 1, "facts": make_facts(), "pages": [{"ok": True}],
        "stats": {"searched_queries": ["q1"], "pages_fetched": 1, "pages_failed": 0,
                  "conflicts": [], "elapsed": 1.2, "llm_usage": {"calls": 3, "total_tokens": 100}},
    }
    markdown = render_report_markdown(report, state)

    check("含一级标题", markdown.startswith("# 测试报告"), markdown[:40])
    check("含研究问题与生成时间", "研究问题" in markdown and "生成时间" in markdown)
    check("含摘要章节", "## 摘要" in markdown)
    check("正文章节自动编号", "## 1. 背景" in markdown and "## 2. 发现" in markdown)
    check("含结论章节", "## 结论" in markdown)
    check("含参考文献且带链接",
          "## 参考文献" in markdown and "[1] 来源一 — https://a.com" in markdown, markdown[-400:])
    check("含研究过程附录", "## 附：研究过程" in markdown and "检索轨迹" in markdown)
    check("附录含 Token 统计", "Token 100" in markdown)

    no_process = render_report_markdown(report, state, include_process=False)
    check("可关闭研究过程附录", "## 附：研究过程" not in no_process)

    empty_render = render_report_markdown({"title": "空", "question": "q"}, None)
    check("空报告也能渲染不崩", empty_render.startswith("# 空"), empty_render[:40])


# ========== 5. 导出 ==========
def test_export() -> None:
    print("\n[5] 报告导出")
    from report import export_report, find_cjk_font, markdown_to_html

    markdown = "# 测试报告\n\n## 摘要\n\n这是中文摘要，用于验证中文字体渲染。[1]\n\n" \
               "## 1. 正文\n\n- 要点一\n- 要点二\n\n## 参考文献\n\n[1] 来源 — https://a.com\n"

    html = markdown_to_html(markdown, title="测试报告")
    check("HTML 含标题与正文", "<h1>" in html and "测试报告" in html)
    check("HTML 列出列表项", "<li>" in html)
    check("HTML 声明 UTF-8", 'charset="utf-8"' in html)

    font_path, font_error = find_cjk_font()
    print(f"       (中文字体：{font_path or '未找到'} {('— ' + font_error[:60]) if font_error else ''})")

    if font_path:
        html_with_font = markdown_to_html(markdown, title="测试报告", font_path=font_path,
                                         font_family="ReportCJK")
        check("HTML 内嵌 @font-face", "@font-face" in html_with_font and "file:///" in html_with_font,
              html_with_font[:600])

    with tempfile.TemporaryDirectory() as tmp:
        result = export_report(markdown, tmp, stem="report", formats=("md", "html", "pdf"))
        paths = result["paths"]
        check("导出 Markdown", "md" in paths and os.path.getsize(paths["md"]) > 0)
        check("导出 HTML", "html" in paths and os.path.getsize(paths["html"]) > 0)

        if font_path:
            check("导出 PDF", "pdf" in paths, str(result["warnings"]))
            if "pdf" in paths:
                with open(paths["pdf"], "rb") as handle:
                    magic = handle.read(5)
                check("PDF 文件头正确", magic == b"%PDF-", str(magic))
                check("PDF 无导出告警", not result["warnings"], str(result["warnings"]))

                # 关键校验：把 PDF 里的文字抽回来，确认中文真的渲染进去了，
                # 而不是静默降级成 Helvetica 输出一堆方框。
                from pypdf import PdfReader

                extracted = "".join(
                    (page.extract_text() or "") for page in PdfReader(paths["pdf"]).pages
                ).replace("\n", "")
                check("PDF 内确实含中文（字体未降级）", "测试报告" in extracted,
                      f"抽取到的文本：{extracted[:120]}")
                check("PDF 内含引用编号", "[1]" in extracted, f"抽取到的文本：{extracted[:120]}")
                check("PDF 体积合理（内嵌字体后 > 8KB）",
                      os.path.getsize(paths["pdf"]) > 8192,
                      f"{os.path.getsize(paths['pdf'])} bytes")
        else:
            check("缺中文字体时跳过 PDF 并给出提示",
                  "pdf" not in paths and result["warnings"], str(result["warnings"]))


# ========== 6. 离线端到端 ==========
def test_demote_headings() -> None:
    print("\n[6] 章节内标题降级")
    from writer import demote_headings

    check("二级标题降为三级", demote_headings("## 一、背景") == "### 一、背景",
          demote_headings("## 一、背景"))
    check("一级标题降为二级", demote_headings("# 大标题") == "## 大标题")
    check("普通文本不受影响", demote_headings("正文没有标题") == "正文没有标题")
    check("代码块里的 # 不动",
          demote_headings("```\n# 这是注释\n```") == "```\n# 这是注释\n```",
          demote_headings("```\n# 这是注释\n```"))
    check("混合内容只动标题",
          demote_headings("正文\n## 小标题\n更多正文") == "正文\n### 小标题\n更多正文")


def test_end_to_end() -> None:
    print("\n[7] 离线端到端：完整报告生成")
    from deep_research import DeepResearchAgent, summarize
    from offline_stubs import OfflineLLM, offline_search

    with offline_search():
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=1), max_iterations=3)
        final = agent.run("离线自检：能否产出带引用的研究报告？")

    markdown = final.get("report_markdown") or ""
    report = final.get("report") or {}

    check("状态里有结构化报告", bool(report) and report.get("title"), str(report)[:80])
    check("产出 Markdown 全文", len(markdown) > 300, f"{len(markdown)} 字符")
    check("报告含摘要", "## 摘要" in markdown)
    check("报告含正文多章", markdown.count("\n## ") >= 3, str(markdown.count("\n## ")))
    check("报告含结论", "## 结论" in markdown)
    check("报告含参考文献", "## 参考文献" in markdown)
    check("有引用条目", len(report.get("citations") or []) > 0, str(report.get("citations")))
    check("参考文献带真实链接",
          all(c.get("url") for c in report.get("citations") or []), str(report.get("citations")))

    # 引用编号必须连续 1..K，正文里不能出现断号或越界号
    indexes = [c["index"] for c in report.get("citations") or []]
    check("引用编号连续无断号", indexes == list(range(1, len(indexes) + 1)), str(indexes))

    body = markdown.split("## 参考文献")[0]
    used = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    check("正文引用号都在参考文献范围内", used <= set(indexes), f"used={sorted(used)}, refs={indexes}")
    check("参考文献没有未被引用的死条目", used == set(indexes), f"used={sorted(used)}, refs={indexes}")

    # 回归测试：报告尾部的统计要在整张图跑完后重渲染一次，
    # 否则会写着"LLM 调用 0 次 / 耗时 0 秒"（第一次真实运行的实测 Bug）。
    check("附录统计不是 0 耗时", "端到端耗时：0 秒" not in markdown,
          markdown[markdown.find("执行统计"):][:200])
    check("附录统计含真实 Token 字段", "Token" in markdown)

    metrics = summarize(final)
    check("指标含报告章节数", metrics.get("报告章节数", 0) >= 2, str(metrics))
    check("指标含引用来源数", metrics.get("引用来源数", 0) > 0, str(metrics))
    check("指标含越界引用数", metrics.get("越界引用数", 0) == 0, str(metrics))
    check("报告字数与指标一致",
          metrics.get("报告字数") == len(final.get("report_markdown") or ""),
          f"{metrics.get('报告字数')} vs {len(final.get('report_markdown') or '')}")

    # 无事实时也要能出一份"如实说明"的报告，而不是崩掉
    from writer import build_report
    empty = build_report({"question": "无事实问题", "facts": [], "stats": {}, "plan": []})
    check("无事实时生成失败说明报告", "未能" in empty["report_markdown"], empty["report_markdown"][:80])
    check("无事实报告不含参考文献", "## 参考文献" not in empty["report_markdown"])


def main() -> int:
    print("=" * 60)
    print("Phase 3 自检：报告生成与引用溯源")
    print("=" * 60)
    test_citation_build()
    test_citation_sanitize()
    test_outline_normalize()
    test_render()
    test_export()
    test_demote_headings()
    test_end_to_end()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
