"""
Phase 1 自检脚本（不需要 API Key，也不联网）。
运行方式：
    python tests/test_phase1.py

覆盖内容：
    1. 模块导入是否健康
    2. 工具函数：URL 规范化 / 结果去重 / 事实去重 / JSON 稳健解析
    3. LangGraph 状态图能否编译
    4. 用假模型 + 假搜索跑通"规划 → 搜索 → 提取"最小链路
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    """极简断言：记录通过/失败，不中断整个脚本"""
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


# ========== 1. 模块导入 ==========
def test_imports() -> None:
    print("\n[1] 模块导入")
    import config  # noqa: F401
    import extractor
    import graph
    import llm
    import planner
    import prompts  # noqa: F401
    import search_tools
    import searcher
    import state

    check("config 可导入", True)
    check("state 数据模型可用", hasattr(state, "ResearchState") and hasattr(state, "Fact"))
    check("graph 可导入", hasattr(graph, "build_graph"))
    check("节点模块齐备", all(hasattr(m, "planner_node") for m in [planner]))
    check("searcher/extractor 节点存在",
          hasattr(searcher, "searcher_node") and hasattr(extractor, "extractor_node"))
    check("llm 模块可导入", hasattr(llm, "call_llm_json"))


# ========== 2. 工具函数 ==========
def test_utils() -> None:
    print("\n[2] 工具函数")
    from extractor import claim_similarity, dedupe_facts, detect_conflicts
    from llm import extract_json
    from search_tools import dedupe_results, normalize_url, score_credibility

    # URL 规范化
    check("追踪参数被剥离",
          normalize_url("https://a.com/x?utm_source=weibo&id=1") == "https://a.com/x?id=1",
          normalize_url("https://a.com/x?utm_source=weibo&id=1"))
    check("末尾斜杠与锚点被统一",
          normalize_url("https://A.com/x/#top") == "https://a.com/x",
          normalize_url("https://A.com/x/#top"))

    # 结果去重
    rows = [
        {"id": "S1", "url": "https://a.com/p?utm_source=x", "title": "A", "snippet": "短", "score": 0.5},
        {"id": "S2", "url": "https://a.com/p", "title": "A", "snippet": "更长的摘要内容", "score": 0.9},
        {"id": "S3", "url": "https://b.com/q", "title": "B", "snippet": "B 摘要", "score": 0.7},
    ]
    deduped = dedupe_results(rows)
    check("重复 URL 被合并", len(deduped) == 2, f"实际 {len(deduped)}")
    check("重复 URL 保留更长摘要",
          any(r.get("snippet") == "更长的摘要内容" for r in deduped))
    check("高可信域名得分更高",
          score_credibility("https://www.nature.com/articles/x")
          > score_credibility("https://weibo.com/x"))

    # JSON 稳健解析
    check("裸 JSON 可解析", extract_json('{"a": 1}') == {"a": 1})
    check("代码围栏 JSON 可解析", extract_json('```json\n{"a": 2}\n```') == {"a": 2})
    check("夹带解释文字也能解析",
          extract_json('好的，结果如下：\n{"a": 3}\n以上。') == {"a": 3})

    # 事实去重与冲突检测
    check("相同事实相似度为 1", claim_similarity("营收增长了 20%", "营收增长了20%。") > 0.95)
    check("无关事实相似度低", claim_similarity("营收增长了 20%", "今天天气很好") < 0.3)

    existing = [{"id": "F1", "claim": "2024 年该公司营收为 100 亿元。", "confidence": 0.8,
                 "credibility": 0.9}]
    new = [
        {"claim": "2024年该公司营收为100亿元", "confidence": 0.7, "credibility": 0.5},
        {"claim": "该公司员工总数为 5000 人。", "confidence": 0.6, "credibility": 0.5},
    ]
    kept = dedupe_facts(new, existing)
    check("近似事实被去重", len(kept) == 1, f"实际 {len(kept)}")
    check("保留的是新事实", kept and "员工总数" in kept[0]["claim"])

    conflicts = detect_conflicts([
        {"id": "F1", "claim": "2024 年该公司营收为 100 亿元"},
        {"id": "F2", "claim": "2024 年该公司营收为 120 亿元"},
        {"id": "F3", "claim": "该公司总部位于北京"},
    ])
    check("数字矛盾被识别", len(conflicts) == 1, f"实际 {len(conflicts)}")

    # 回归测试：曾经因为只看"文本相似 + 数字不同"，把 URL 里的 page-3 / page-4
    # 当成互相矛盾的数据，一次运行误报 25 组冲突。
    noise_conflicts = detect_conflicts([
        {"id": "F1", "claim": "该来源给出的结论是：这条路线在成本控制上具备明显优势。"
                             "（离线自检来源 https://offline.example.com/page-3）"},
        {"id": "F2", "claim": "该来源指出：截至 2024 年，相关核心指标已提升约 30%。"
                             "（离线自检来源 https://offline.example.com/page-4）"},
    ])
    check("URL 中的编号不产生误报冲突", len(noise_conflicts) == 0, str(noise_conflicts))

    unchanged = detect_conflicts([
        {"id": "F1", "claim": "2024 年该公司营收为 100 亿元"},
        {"id": "F2", "claim": "2024 年该公司营收为 100 亿元。"},
    ])
    check("数字相同不判为冲突", len(unchanged) == 0, str(unchanged))


# ========== 3. 状态图 ==========
def test_graph() -> None:
    print("\n[3] LangGraph 状态图")
    from graph import build_graph, describe_graph

    app = build_graph()
    check("状态图可编译", app is not None)
    check("状态图含 planner/searcher/extractor 节点",
          all(n in app.get_graph().nodes for n in ("planner", "searcher", "extractor")),
          str(list(app.get_graph().nodes)))
    check("Mermaid 描述非空", "planner" in describe_graph())


# ========== 4. 离线端到端 ==========
def test_offline_pipeline() -> None:
    print("\n[4] 离线端到端（假模型 + 假搜索）")
    from deep_research import DeepResearchAgent, summarize
    from offline_stubs import OfflineLLM, offline_search

    events = []
    with offline_search():
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0))
        final = agent.run("离线自检：Deep Research Agent 链路是否通畅？", on_event=events.append)

    check("产出研究计划", len(final.get("plan") or []) == 3, str(final.get("plan")))
    check("产出搜索结果", len(final.get("search_results") or []) > 0)
    check("抓取到网页", len(final.get("pages") or []) > 0)
    check("提炼出事实", len(final.get("facts") or []) > 0)
    check("事实带来源链接", all(f.get("source_url") for f in final.get("facts") or []))
    check("事实编号连续", [f["id"] for f in final["facts"]] ==
          [f"F{i}" for i in range(1, len(final["facts"]) + 1)])
    check("事件回调收到事件", len(events) >= 3, f"实际 {len(events)}")
    check("状态内也留了事件", len(final.get("events") or []) >= 3)

    metrics = summarize(final)
    check("指标汇总含事实数", metrics.get("事实数", 0) > 0, str(metrics))
    check("指标汇总含子问题覆盖率", "子问题覆盖率" in metrics, str(metrics))


def main() -> int:
    print("=" * 60)
    print("Phase 1 自检：规划 → 搜索 → 提取 最小链路")
    print("=" * 60)
    test_imports()
    test_utils()
    test_graph()
    test_offline_pipeline()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
