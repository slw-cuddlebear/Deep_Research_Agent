"""
Phase 5 自检脚本（不需要 API Key，也不联网、不加载 embedding 模型）。
运行方式：
    python tests/test_phase5.py

覆盖内容：
    1. 结果唯一键：本地片段必须保留片段号（否则同文档所有片段会被误判成同一条）
    2. 分数归一化：不同量纲的网页分与本地库分变得可比
    3. 结果融合：两路合并、各自去重、本地加权
    4. 来源配额：本地文档不会被数量占优的网页挤光
    5. 本地片段直读：不需要联网抓取
    6. 引用渲染：本地来源显示为文档名，且与网络来源分组列出
    7. 语料与索引：本地语料可加载、可切片
    8. 离线端到端（含本地库）
"""
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


# ========== 1. 结果唯一键 ==========
def test_result_key() -> None:
    print("\n[1] 结果唯一键")
    import config
    from search_tools import normalize_url, result_key, score_credibility

    check("本地片段保留片段号",
          result_key("local://doc.md#chunk-3") != result_key("local://doc.md#chunk-7"),
          "两个不同片段被当成了同一条结果")
    check("同一本地片段键稳定",
          result_key("local://doc.md#chunk-3") == result_key("local://doc.md#chunk-3"))
    check("网页仍走 URL 规范化",
          result_key("https://a.com/x?utm_source=y") == normalize_url("https://a.com/x?utm_source=y"))
    check("本地来源可信度高于普通网页",
          score_credibility("local://doc.md#chunk-1") > score_credibility("https://blog.example.com/a"),
          f"{score_credibility('local://doc.md#chunk-1')} vs {score_credibility('https://blog.example.com/a')}")
    check("本地来源可信度等于配置值",
          abs(score_credibility("local://doc.md#chunk-1") - config.LOCAL_CREDIBILITY) < 1e-9)


# ========== 2. 分数归一化 ==========
def test_normalize() -> None:
    print("\n[2] 分数归一化")
    from searcher import normalize_scores

    items = [{"score": 0.9}, {"score": 0.5}, {"score": 0.1}]
    normalize_scores(items)
    check("最高分归一化到 1.0", items[0]["score"] == 1.0, str(items))
    check("最低分归一化到 0.3", items[2]["score"] == 0.3, str(items))
    check("分数保持单调", items[0]["score"] > items[1]["score"] > items[2]["score"], str(items))

    same = [{"score": 0.6}, {"score": 0.6}]
    normalize_scores(same)
    check("全部同分时不除零", all(i["score"] == 0.6 for i in same), str(same))

    empty = normalize_scores([])
    check("空列表安全", empty == [])

    # 不同量纲归一化后才可比：Tavily 的 0.9 与 BGE 的 0.72 不该直接比大小
    web = normalize_scores([{"score": 0.9}, {"score": 0.3}])
    local = normalize_scores([{"score": 0.72}])
    check("单元素批次归一化到 0.6", local[0]["score"] == 0.6, str(local))
    check("网页批次内部可比", web[0]["score"] > web[1]["score"], str(web))


# ========== 3. 结果融合 ==========
def test_fuse() -> None:
    print("\n[3] 结果融合")
    import config
    from searcher import fuse_results

    web = [
        {"url": "https://a.com/1", "score": 0.9, "engine": "tavily", "title": "A1"},
        {"url": "https://a.com/2", "score": 0.7, "engine": "tavily", "title": "A2"},
        {"url": "https://a.com/1?utm_source=x", "score": 0.5, "engine": "tavily", "title": "重复"},
    ]
    local = [
        {"url": "local://doc.md#chunk-1", "score": 0.72, "engine": "local", "title": "L1"},
        {"url": "local://doc.md#chunk-1", "score": 0.71, "engine": "local", "title": "重复片段"},
        {"url": "local://doc.md#chunk-2", "score": 0.68, "engine": "local", "title": "L2"},
    ]
    merged = fuse_results(web, local)

    check("重复网页被去重", len([m for m in merged if m["url"].startswith("https")]) == 2,
          str([m["url"] for m in merged]))
    check("重复本地片段被去重", len([m for m in merged if m["url"].startswith("local")]) == 2,
          str([m["url"] for m in merged]))
    check("同文档不同片段都保留",
          len({m["url"] for m in merged if m["url"].startswith("local")}) == 2,
          str([m["url"] for m in merged]))
    check("融合结果按分数降序",
          [m["score"] for m in merged] == sorted([m["score"] for m in merged], reverse=True),
          str([m["score"] for m in merged]))
    check("本地结果分数不超过 1.0",
          all(m["score"] <= 1.0 for m in merged if m["url"].startswith("local")),
          str([m["score"] for m in merged]))
    check("融合权重来自配置",
          abs(config.LOCAL_SCORE_BOOST - 1.15) < 1e-9, str(config.LOCAL_SCORE_BOOST))

    only_web = fuse_results(web, [])
    check("无本地结果时也能融合", len(only_web) == 2, str(len(only_web)))
    check("空输入安全", fuse_results([], []) == [])


def test_union_normalization() -> None:
    """
    回归测试：跨来源必须用**统一尺度**归一化。

    早期实现对网页和本地库**各自**做 min-max 归一化，导致本地库哪怕只召回到
    一堆弱相关片段，其中最强的那个也会被拉满到 1.0，从而压过所有网页结果。
    Phase 8 消融实验实测的后果：本地引用只贡献 6%，却吃掉 40% 的阅读名额，
    T1 的网页阅读量从 29 篇被压到 19 篇。
    """
    print("\n[3b] 跨来源统一尺度归一化（Phase 8 回归）")
    from searcher import fuse_results, normalize_union

    # 网页明显更相关（0.9/0.8），本地库只是勉强及格（0.46/0.45）
    web = [{"url": "https://a.com/1", "score": 0.9, "engine": "tavily"},
           {"url": "https://a.com/2", "score": 0.8, "engine": "tavily"}]
    local = [{"url": "local://d.md#chunk-1", "score": 0.46, "engine": "local"},
             {"url": "local://d.md#chunk-2", "score": 0.45, "engine": "local"}]

    norm_web, norm_local = normalize_union([dict(w) for w in web], [dict(l) for l in local])
    best_web = max(w["score"] for w in norm_web)
    best_local = max(l["score"] for l in norm_local)
    check("统一尺度下强相关网页分数高于弱相关本地片段",
          best_web > best_local,
          f"网页最高 {best_web} vs 本地最高 {best_local}（各自归一化会让它们都变成 1.0）")

    merged = fuse_results([dict(w) for w in web], [dict(l) for l in local])
    top_two = [m["url"] for m in merged[:2]]
    check("融合后网页结果排在前面",
          all(u.startswith("https") for u in top_two), str(top_two))

    # 反过来：本地明显更相关时应当排前面
    hot_local = [{"url": "local://d.md#chunk-1", "score": 0.95, "engine": "local"}]
    cold_web = [{"url": "https://a.com/1", "score": 0.35, "engine": "tavily"}]
    merged2 = fuse_results(cold_web, hot_local)
    check("本地明显更相关时排在最前", merged2[0]["url"].startswith("local"),
          str([(m["url"], m["score"]) for m in merged2]))

    check("空输入不崩", normalize_union([], []) == ([], []))


# ========== 4. 来源保障名额 ==========
def test_source_quota() -> None:
    print("\n[4] 本地库保障名额（保底 + 按分数竞争）")
    import config
    from extractor import _select_targets

    config.MAX_PAGES_PER_ROUND = 10
    config.LOCAL_MIN_SLOTS = 2

    # 20 条高分网页 + 5 条低分本地片段
    state = {
        "question": "测试问题",
        "plan": [{"id": "Q1", "question": "子问题一"}],
        "pages": [],
        "search_results": (
            [{"url": f"https://web.com/{i}", "score": 0.95, "engine": "tavily",
              "sub_question_id": "Q1"} for i in range(20)]
            + [{"url": f"local://doc.md#chunk-{i}", "score": 0.4, "engine": "local",
                "sub_question_id": "Q1"} for i in range(5)]
        ),
    }
    picked = _select_targets(state)
    local_picked = [p for p in picked if p.get("engine") == "local"]
    web_picked = [p for p in picked if p.get("engine") != "local"]

    check("总数不超过单轮上限", len(picked) <= 10, str(len(picked)))
    check("本地库拿到保底名额", len(local_picked) >= 2,
          f"本地只入选 {len(local_picked)} 条（低分网页把它挤掉了）")
    check("保底之外不再多占名额（低分不该挤掉高分网页）", len(local_picked) == 2,
          f"本地占了 {len(local_picked)} 条，超过保底 {config.LOCAL_MIN_SLOTS} 条")
    check("高分网页仍占绝大多数名额", len(web_picked) == 8, str(len(web_picked)))

    # 本地分数更高时应当靠竞争拿到更多名额（保底不是上限）
    state["search_results"] = (
        [{"url": f"https://web.com/{i}", "score": 0.3, "engine": "tavily",
          "sub_question_id": "Q1"} for i in range(20)]
        + [{"url": f"local://doc.md#chunk-{i}", "score": 0.9, "engine": "local",
            "sub_question_id": "Q1"} for i in range(5)]
    )
    picked_high = _select_targets(state)
    local_high = [p for p in picked_high if p.get("engine") == "local"]
    check("本地高分时能靠竞争拿到更多名额", len(local_high) == 5, str(len(local_high)))

    # 只有本地结果时应全部入选
    state["search_results"] = [
        {"url": f"local://doc.md#chunk-{i}", "score": 0.7 - i * 0.01, "engine": "local",
         "sub_question_id": "Q1"} for i in range(8)
    ]
    picked3 = _select_targets(state)
    check("只有本地结果时全部入选", len(picked3) == 8, str(len(picked3)))

    # 已读过的本地片段不该重复入选（片段号不同要能区分）
    state["pages"] = [{"url": "local://doc.md#chunk-0", "ok": True, "source": "local"}]
    state["search_results"] = [
        {"url": "local://doc.md#chunk-0", "score": 0.9, "engine": "local", "sub_question_id": "Q1"},
        {"url": "local://doc.md#chunk-1", "score": 0.8, "engine": "local", "sub_question_id": "Q1"},
    ]
    picked4 = _select_targets(state)
    urls = {p["url"] for p in picked4}
    check("已读片段被跳过", "local://doc.md#chunk-0" not in urls, str(urls))
    check("同文档未读片段仍可入选", "local://doc.md#chunk-1" in urls, str(urls))

    # 保底设为 0 时纯按分数竞争
    config.LOCAL_MIN_SLOTS = 0
    state["pages"] = []
    state["search_results"] = (
        [{"url": f"https://web.com/{i}", "score": 0.95, "engine": "tavily",
          "sub_question_id": "Q1"} for i in range(20)]
        + [{"url": f"local://doc.md#chunk-{i}", "score": 0.4, "engine": "local",
            "sub_question_id": "Q1"} for i in range(5)]
    )
    check("保底为 0 时本地不占名额",
          len([p for p in _select_targets(state) if p.get("engine") == "local"]) == 0)
    config.LOCAL_MIN_SLOTS = 2


# ========== 5. 本地片段直读 ==========
def test_local_pages_skip_fetch() -> None:
    print("\n[5] 本地片段直读（无需联网抓取）")
    from extractor import _local_pages

    targets = [
        {"url": "local://doc.md#chunk-1", "engine": "local", "title": "本地文档",
         "content": "本地片段正文内容" * 20},
        {"url": "https://web.com/a", "engine": "tavily", "title": "网页"},
    ]
    local_pages, web_targets = _local_pages(targets)

    check("本地片段转成 PageContent", len(local_pages) == 1, str(local_pages))
    check("本地页面标记为 local", local_pages[0]["source"] == "local")
    check("本地页面自带正文", len(local_pages[0]["text"]) > 50, str(len(local_pages[0]["text"])))
    check("本地页面视为抓取成功", local_pages[0]["ok"] is True)
    check("网页目标单独返回待抓取", len(web_targets) == 1 and
          web_targets[0]["url"] == "https://web.com/a", str(web_targets))

    # 没有 content 时退回 snippet，不能产生空页面
    fallback, _ = _local_pages([
        {"url": "local://doc.md#chunk-9", "engine": "local", "snippet": "摘要兜底"}
    ])
    check("无正文时退回摘要", fallback[0]["text"] == "摘要兜底", str(fallback))


# ========== 6. 引用渲染 ==========
def test_citation_render() -> None:
    print("\n[6] 本地来源的引用渲染")
    from writer import build_citations, format_citation_entry, render_report_markdown

    facts = [
        {"id": "F1", "claim": "网络事实", "source_url": "https://a.com/x", "source_title": "网页标题",
         "confidence": 0.8, "credibility": 0.9, "sub_question_id": "Q1"},
        {"id": "F2", "claim": "本地事实", "source_url": "local://doc.md#chunk-2",
         "source_title": "内部文档.md（本地知识库 第 3 段）",
         "confidence": 0.8, "credibility": 0.85, "sub_question_id": "Q1"},
    ]
    citations, fact_to_cite = build_citations(facts)
    check("本地与网络各占一个引用号", len(citations) == 2, str(citations))

    local_cite = next(c for c in citations if c["source_type"] == "local")
    check("本地引用被标记为 local", local_cite["source_type"] == "local")
    check("本地引用标题去掉片段后缀", "本地知识库" not in local_cite["title"], local_cite["title"])
    check("本地引用标题保留文档名", local_cite["title"] == "内部文档.md", local_cite["title"])

    entry = format_citation_entry(local_cite)
    check("本地引用不暴露 local:// 协议", "local://" not in entry, entry)
    check("本地引用标注为本地知识库", "本地知识库" in entry and "内部文档.md" in entry, entry)

    web_cite = next(c for c in citations if c["source_type"] == "web")
    check("网络引用仍给出可点击链接",
          "https://a.com/x" in format_citation_entry(web_cite), format_citation_entry(web_cite))
    check("本地引用不重复文件名",
          format_citation_entry(local_cite).count("内部文档.md") == 1,
          format_citation_entry(local_cite))

    # 回归测试：同一篇本地文档的多个片段必须共用一个引用号，
    # 否则参考文献里会出现三条一模一样的记录（T5 真实运行实测到）。
    from writer import citation_key
    multi = [
        {"id": "F1", "claim": "片段一", "source_url": "local://doc.md#chunk-1",
         "source_title": "doc.md（本地知识库 第 1 段）", "confidence": 0.8, "credibility": 0.85},
        {"id": "F2", "claim": "片段二", "source_url": "local://doc.md#chunk-5",
         "source_title": "doc.md（本地知识库 第 6 段）", "confidence": 0.8, "credibility": 0.85},
        {"id": "F3", "claim": "片段三", "source_url": "local://doc.md#chunk-9",
         "source_title": "doc.md（本地知识库 第 10 段）", "confidence": 0.8, "credibility": 0.85},
    ]
    multi_citations, multi_map = build_citations(multi)
    check("同文档不同片段共用一个引用号", len(multi_citations) == 1, str(multi_citations))
    check("片段映射到同一引用号", len(set(multi_map.values())) == 1, str(multi_map))
    check("引用键忽略片段号",
          citation_key("local://doc.md#chunk-1") == citation_key("local://doc.md#chunk-9"))
    check("不同本地文档仍是不同引用号",
          citation_key("local://a.md#chunk-1") != citation_key("local://b.md#chunk-1"))
    check("网页引用键不受影响",
          citation_key("https://a.com/x") == "https://a.com/x")

    report = {
        "title": "混合来源报告", "question": "q", "created_at": "2025-01-01 00:00:00",
        "summary": "摘要[1][2]。", "sections": [{"heading": "正文", "content": "内容[1][2]。"}],
        "conclusion": "结论[1]。", "citations": citations,
    }
    markdown = render_report_markdown(report, None)
    check("参考文献分组列出网络来源", "### 网络来源" in markdown, markdown[-500:])
    check("参考文献分组列出本地文档", "### 本地知识库文档" in markdown, markdown[-500:])
    check("引用编号跨组保持连续", "[1]" in markdown and "[2]" in markdown)

    # 只有网络来源时不分组，避免多余标题
    web_only = dict(report, citations=[web_cite])
    markdown2 = render_report_markdown(web_only, None)
    check("纯网络来源时不出现分组标题", "### 网络来源" not in markdown2, markdown2[-300:])


# ========== 7. 语料与索引 ==========
def test_corpus() -> None:
    print("\n[7] 语料加载与切片")
    import config
    import local_kb

    files = local_kb.list_source_files()
    check("演示语料存在", len(files) >= 3, f"实际 {len(files)} 篇：{[f.name for f in files]}")
    check("语料都在 knowledge_base 目录",
          all(config.LOCAL_KB_DIR in f.parents for f in files), str(files))

    documents = local_kb.load_documents()
    check("文档可加载", len(documents) >= 3, f"实际 {len(documents)} 篇")
    check("文档带来源文件名", all(d.metadata.get("source_name") for d in documents))

    chunks = local_kb.split_documents(documents)
    check("文档可切片", len(chunks) > 0, str(len(chunks)))
    check("切片带片段号", all("chunk_index" in c.metadata for c in chunks))
    check("切片长度受控", max(len(c.page_content) for c in chunks) <= config.LOCAL_CHUNK_SIZE + 200,
          f"最长 {max(len(c.page_content) for c in chunks)}")
    check("语料指纹可计算", len(local_kb._fingerprint(files)) == 16,
          local_kb._fingerprint(files))

    # 索引状态不应抛异常
    status = local_kb.kb_status()
    check("知识库状态可查询", isinstance(status, dict) and status.get("documents", 0) >= 3,
          str(status))

    # 相关度闸门：本地库只回答"它真正知道的事"
    check("配置了本地召回相关度下限", config.LOCAL_MIN_SCORE > 0, str(config.LOCAL_MIN_SCORE))
    check("相关度下限处于合理区间", 0.35 <= config.LOCAL_MIN_SCORE <= 0.60,
          f"{config.LOCAL_MIN_SCORE}（实测标定：相关查询 0.59~0.77，无关查询 0.33~0.42）")

    # 闸门要真的生效：无关问题不应召回任何片段
    irrelevant = local_kb.search_local("今天天气怎么样适合钓鱼吗", top_k=5)
    check("无关问题不召回本地片段", len(irrelevant) == 0,
          f"无关问题召回了 {len(irrelevant)} 条：{[i['url'] for i in irrelevant]}")
    relevant = local_kb.search_local("向量数据库选型 Chroma Milvus Qdrant 对比", top_k=3)
    check("相关问题能召回本地片段", len(relevant) > 0, "相关问题召回为空")

    # 分数必须落在 [0,1]（用余弦距离自己算，避免 Chroma relevance_score 越界）
    check("召回分数都在 [0,1] 内",
          all(0.0 <= h["score"] <= 1.0 for h in relevant),
          str([h["score"] for h in relevant]))
    check("召回结果按分数降序",
          [h["score"] for h in relevant] == sorted([h["score"] for h in relevant], reverse=True),
          str([h["score"] for h in relevant]))


# ========== 8. 并发安全 ==========
def test_concurrency() -> None:
    """
    并发回归测试。

    真实运行中曾出现"3 个检索任务失败"：5 个子问题并发查询本地库，
    而 Chroma 的持久化客户端不允许同进程对同一目录重复实例化，
    索引重建也不能并发执行。修复方式是在 local_kb 里加一把可重入锁。
    """
    print("\n[8] 并发安全（曾在真实运行中暴露）")
    from concurrent.futures import ThreadPoolExecutor

    import local_kb

    local_kb.reset_cache()

    queries = [
        "向量数据库选型 Chroma Milvus Qdrant",
        "RAG 与长上下文超长文档问答",
        "为什么用 LangGraph 编排",
        "向量检索的内存占用怎么估算",
        "分层的性价比如何",
        "过滤条件下的检索性能",
        "切片粒度对检索的影响",
        "冷启动时间的影响",
    ]

    def search(query):
        try:
            return ("ok", len(local_kb.search_local(query, top_k=3)))
        except Exception as exc:
            return ("fail", f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(search, queries))

    failures = [detail for status, detail in results if status == "fail"]
    check("并发首次检索零失败", not failures, f"{len(failures)} 个失败：{failures[:2]}")
    check("并发检索都有返回", all(status == "ok" for status, _ in results), str(results))
    check("并发后向量库单例已建立", local_kb._vectorstore is not None)

    # 清空缓存后再并发一次，验证"冷启动并发"这条最容易出错的路径
    local_kb.reset_cache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results2 = list(pool.map(search, queries[:4]))
    failures2 = [detail for status, detail in results2 if status == "fail"]
    check("清缓存后并发重建零失败", not failures2, f"{len(failures2)} 个失败：{failures2[:2]}")


# ========== 9. 离线端到端（含本地库）==========
def test_end_to_end_with_local() -> None:
    print("\n[9] 离线端到端：混合检索")
    import config
    from deep_research import DeepResearchAgent, summarize
    from offline_stubs import OfflineLLM, offline_search

    original_use_local = config.USE_LOCAL_KB
    config.USE_LOCAL_KB = True
    try:
        with offline_search(with_local=True) as backend:
            agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0), max_iterations=2,
                                      use_local_kb=True)
            final = agent.run("离线自检：混合检索是否把本地文档也纳入？")
    finally:
        config.USE_LOCAL_KB = original_use_local

    markdown = final.get("report_markdown") or ""
    report = final.get("report") or {}
    citations = report.get("citations") or []
    pages = final.get("pages") or []

    check("本地库被调用", backend.local_calls > 0, str(backend.local_calls))
    check("搜索结果含本地片段",
          any(r.get("engine") == "local" for r in final.get("search_results") or []),
          str([r.get("engine") for r in final.get("search_results") or []]))
    check("本地片段被读取", len([p for p in pages if p.get("source") == "local"]) > 0,
          str([p.get("source") for p in pages]))
    check("本地片段进入引用列表",
          any(c.get("source_type") == "local" for c in citations), str(citations))
    check("报告标注本地知识库分组", "### 本地知识库文档" in markdown, markdown[-400:])
    check("报告不暴露 local:// 协议", "local://" not in markdown, "本地协议泄漏到报告中")

    metrics = summarize(final)
    check("指标含本地库结果数", metrics.get("其中本地库结果", 0) > 0, str(metrics))
    check("指标含本地文档引用数", metrics.get("其中本地文档引用", 0) > 0, str(metrics))
    check("指标含读取本地片段数", metrics.get("读取本地片段数", 0) > 0, str(metrics))

    # 关闭本地库时应完全不调用
    original_use_local = config.USE_LOCAL_KB
    config.USE_LOCAL_KB = False
    try:
        with offline_search(with_local=True) as backend2:
            agent2 = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0), max_iterations=1)
            final2 = agent2.run("离线自检：关闭本地库时不应调用它")
    finally:
        config.USE_LOCAL_KB = original_use_local

    check("关闭后不调用本地库", backend2.local_calls == 0, str(backend2.local_calls))
    check("关闭后无本地引用",
          not any(c.get("source_type") == "local" for c in (final2.get("report") or {}).get("citations") or []))


def main() -> int:
    print("=" * 60)
    print("Phase 5 自检：本地知识库与混合检索")
    print("=" * 60)
    test_result_key()
    test_normalize()
    test_fuse()
    test_union_normalization()
    test_source_quota()
    test_local_pages_skip_fetch()
    test_citation_render()
    test_corpus()
    test_concurrency()
    test_end_to_end_with_local()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
