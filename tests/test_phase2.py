"""
Phase 2 自检脚本（不需要 API Key，也不联网）。
运行方式：
    python tests/test_phase2.py

覆盖内容：
    1. 反思辅助函数：检索词归一化 / 重复判定 / 事实摘要压缩
    2. 规则层刹车：最大轮数、无新检索词、检索词重复、无新增事实
    3. 条件边路由
    4. 状态图结构（reflector 节点与回边）
    5. 离线端到端：反思确实触发了补充搜索，且能收敛退出（不死循环）
"""
import os
import re
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


def make_state(**overrides) -> dict:
    """构造一个用于测试 decide() 的最小状态"""
    state = {
        "question": "测试问题",
        "plan": [{"id": "Q1", "question": "子问题一"}],
        "facts": [],
        "iteration": 1,
        "stats": {"max_iterations": 3, "searched_queries": [], "facts_gained_last_round": 5},
    }
    state.update(overrides)
    return state


# ========== 1. 辅助函数 ==========
def test_helpers() -> None:
    print("\n[1] 反思辅助函数")
    from reflector import build_facts_digest, build_plan_digest, is_repeated, normalize_query

    check("检索词归一化去标点", normalize_query("DeepSeek R1，评测！") == "deepseekr1评测",
          normalize_query("DeepSeek R1，评测！"))
    check("空检索词视为重复", is_repeated("", ["任意"]) is True)

    history = ["DeepSeek 推理成本"]
    check("完全相同的检索词判为重复", is_repeated("DeepSeek 推理成本", history) is True)
    check("仅大小写/标点不同也判为重复", is_repeated("deepseek推理成本！", history) is True)
    check("包含关系判为重复", is_repeated("DeepSeek 推理成本 下降", history) is True)
    check("全新检索词不判为重复", is_repeated("OpenAI o1 数学能力", history) is False)

    # 回归测试：曾经因为阈值 0.75 过低，把两个真正不同的补充检索词误判成重复而丢掉一个。
    # "离线自检 补充证据1" 与 "离线自检 数据1" 的相似度恰好是 0.750。
    similar_prefix = ["离线自检 补充证据1", "离线自检 数据1"]
    check("共同前缀但内容不同的检索词不算重复", is_repeated("离线自检 数据1", similar_prefix[:1]) is False,
          "相似度 0.750 曾被误判为重复")
    check("过短的包含关系不算重复（AI 不应命中一切）",
          is_repeated("AI", ["AI 芯片市场规模预测"]) is False)
    check("足够长的包含关系算重复",
          is_repeated("AI 芯片市场规模", ["AI 芯片市场规模预测"]) is True)

    facts = [
        {"id": "F1", "claim": "事实一", "source_url": "https://www.nature.com/a", "confidence": 0.9,
         "sub_question_id": "Q1"},
        {"id": "F2", "claim": "事实二", "source_url": "https://x.com/b", "confidence": 0.5,
         "sub_question_id": "Q2"},
    ]
    digest = build_facts_digest(facts)
    check("事实摘要按子问题分组", "【Q1】" in digest and "【Q2】" in digest, digest)
    check("事实摘要带来源域名", "nature.com" in digest, digest)
    check("空事实返回占位文本", build_facts_digest([]) == "（暂无）")
    check("计划摘要含子问题", "子问题一" in build_plan_digest([{"id": "Q1", "question": "子问题一"}]))


# ========== 2. 规则层刹车 ==========
def test_brakes() -> None:
    print("\n[2] 规则层刹车（防死循环）")
    from reflector import decide

    # 刹车 1：达到最大轮数
    state = make_state(iteration=3)
    ok, queries, reason = decide({"sufficient": False, "new_queries": ["全新检索词"]}, state)
    check("达到最大轮数时强制停止", ok is False and not queries, f"ok={ok}, reason={reason}")
    check("停止原因说明是最大轮数", "最大搜索轮数" in reason, reason)

    # LLM 说充分 → 停止
    state = make_state()
    ok, queries, reason = decide({"sufficient": True, "new_queries": []}, state)
    check("LLM 判定充分时停止", ok is False and queries == [] and reason == "", f"{ok}, {queries}, {reason}")

    # 刹车 2：说要继续却没给新词
    ok, queries, reason = decide({"sufficient": False, "new_queries": []}, state)
    check("无补充检索词时停止", ok is False and not queries, reason)

    # 刹车 2 变体：给的词全都搜过了
    state = make_state(stats={"max_iterations": 3, "searched_queries": ["老检索词"],
                              "facts_gained_last_round": 5})
    ok, queries, reason = decide({"sufficient": False, "new_queries": ["老检索词"]}, state)
    check("补充检索词全部重复时停止", ok is False and not queries, reason)

    # 正常继续
    state = make_state(stats={"max_iterations": 3, "searched_queries": ["老检索词"],
                              "facts_gained_last_round": 5})
    ok, queries, reason = decide(
        {"sufficient": False, "new_queries": ["新词一", "老检索词", "新词二"]}, state)
    check("有全新检索词时继续搜索", ok is True and queries == ["新词一", "新词二"], f"{ok}, {queries}")

    # 新词之间互相重复也要去重
    ok, queries, _ = decide({"sufficient": False, "new_queries": ["新词一", "新词一！", "新词二"]}, state)
    check("补充检索词互相去重", queries == ["新词一", "新词二"], str(queries))

    # 刹车 3：上一轮没有新增事实
    state = make_state(iteration=2, stats={"max_iterations": 5, "searched_queries": [],
                                           "facts_gained_last_round": 0})
    ok, queries, reason = decide({"sufficient": False, "new_queries": ["新词"]}, state)
    check("上一轮零新增事实时停止", ok is False and not queries, reason)
    check("停止原因说明零新增", "未带来任何新事实" in reason, reason)

    # 最多只接受 3 条补充词
    state = make_state(stats={"max_iterations": 3, "searched_queries": [],
                              "facts_gained_last_round": 5})
    ok, queries, _ = decide(
        {"sufficient": False, "new_queries": ["甲1", "乙2", "丙3", "丁4", "戊5"]}, state)
    check("补充检索词上限为 3 条", len(queries) == 3, str(queries))


# ========== 3. 路由 ==========
def test_routing() -> None:
    print("\n[3] 条件边路由")
    from reflector import route_after_reflection

    check("有待搜检索词 → 回到搜索", route_after_reflection({"pending_queries": ["x"]}) == "search_again")
    check("无待搜检索词 → 结束", route_after_reflection({"pending_queries": []}) == "finish")
    check("字段缺失也安全结束", route_after_reflection({}) == "finish")


# ========== 4. 状态图结构 ==========
def test_graph_shape() -> None:
    print("\n[4] 状态图结构")
    from graph import BRANCH_FINISH, BRANCH_SEARCH_AGAIN, build_graph, describe_graph

    app = build_graph()
    nodes = set(app.get_graph().nodes)
    check("包含 reflector 节点", "reflector" in nodes, str(sorted(nodes)))
    check("四个节点齐全", {"planner", "searcher", "extractor", "reflector"} <= nodes, str(sorted(nodes)))

    edges = {(e.source, e.target) for e in app.get_graph().edges}
    check("extractor → reflector 连通", ("extractor", "reflector") in edges, str(sorted(edges)))

    # 条件边的两个分支：回边与进入报告生成
    conditional = [e for e in app.get_graph().edges if e.conditional]
    check("reflector 有两条条件分支", len(conditional) == 2, str([(e.source, e.target) for e in conditional]))
    targets = {e.target for e in conditional}
    check("条件分支指向 searcher 与下游节点", targets == {"searcher", "writer"}, str(targets))
    check("分支名常量被用到", BRANCH_SEARCH_AGAIN == "search_again" and BRANCH_FINISH == "finish")
    check("Mermaid 描述含反思循环", "reflector" in describe_graph() and "信息不足" in describe_graph())


# ========== 5. 离线端到端 ==========
def test_loop_end_to_end() -> None:
    print("\n[5] 离线端到端：反思循环")
    from deep_research import DeepResearchAgent, summarize
    from offline_stubs import OfflineLLM, offline_search

    # --- 场景 A：第一轮反思说"不足"，应触发第二轮检索，第二轮后收敛 ---
    with offline_search() as backend:
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=1), max_iterations=3)
        final = agent.run("离线自检：反思循环是否生效？")

    reflections = final.get("reflections") or []
    check("产生了多轮反思", len(reflections) >= 2, f"实际 {len(reflections)}")
    check("第 1 轮反思判定为不充分", reflections[0]["sufficient"] is False, str(reflections[0]))
    check("最后一轮反思判定为充分", reflections[-1]["sufficient"] is True, str(reflections[-1]))
    check("反思触发了补充搜索", final["stats"].get("reflection_triggered", 0) >= 1,
          str(final["stats"].get("reflection_triggered")))
    check("搜索被调用了多轮", backend.search_calls >= 2, f"实际 {backend.search_calls}")
    check("检索词数多于首轮", len(final["stats"].get("searched_queries") or []) > 3,
          str(final["stats"].get("searched_queries")))
    check("补充搜索结果被累积", len(final.get("search_results") or []) > 3,
          f"实际 {len(final.get('search_results') or [])}")
    check("事实跨轮累积", len(final.get("facts") or []) > 3, f"实际 {len(final.get('facts') or [])}")
    check("有反思事件", any(e.get("step") == "reflector" for e in final.get("events") or []))
    check("收敛后待搜检索词清空", not final.get("pending_queries"), str(final.get("pending_queries")))

    metrics = summarize(final)
    check("指标含反思轮数", metrics.get("反思轮数", 0) >= 2, str(metrics))
    check("指标含反思触发次数", metrics.get("反思触发补充搜索", 0) >= 1, str(metrics))

    # --- 场景 B：第一轮就充分，不该有多余搜索 ---
    with offline_search() as backend_b:
        agent_b = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0), max_iterations=3)
        final_b = agent_b.run("离线自检：信息充分时是否立即收敛？")
    check("信息充分时只搜一轮", backend_b.search_calls == 1, f"实际 {backend_b.search_calls}")
    check("信息充分时只反思一轮", len(final_b.get("reflections") or []) == 1)
    check("信息充分时不触发补充搜索", final_b["stats"].get("reflection_triggered", 0) == 0)

    # --- 场景 C：LLM 一直说不足，也必须被最大轮数刹住 ---
    with offline_search() as backend_c:
        agent_c = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=99), max_iterations=2)
        final_c = agent_c.run("离线自检：一直说信息不足会不会死循环？")
    check("最大轮数生效，未死循环", len(final_c.get("reflections") or []) == 2,
          f"实际 {len(final_c.get('reflections') or [])}")
    check("最大轮数下搜索次数受限", backend_c.search_calls <= 2, f"实际 {backend_c.search_calls}")
    check("末轮反思写明被最大轮数中止",
          "最大搜索轮数" in (final_c["reflections"][-1].get("reason") or ""),
          str(final_c["reflections"][-1]))

    # --- 场景 D：补充词与历史重复时不该重复消耗搜索 ---
    class RepeatQueryLLM(OfflineLLM):
        """
        每轮都原样回一个"已经搜过的词"，规则层应识别为重复并立刻停止。

        这里从 Prompt 里把已检索词读回来，而不是硬编码某个词——
        否则一旦规划器产出的检索词变了（例如加上研究主体），测试就会失效。
        """

        def _dispatch(self, text: str) -> dict:
            if '"sufficient"' in text:
                match = re.search(r"【已经检索过的关键词】\s*\n([^\n]+)", text)
                already = [q.strip() for q in (match.group(1).split("、") if match else [])
                           if q.strip()]
                return {"sufficient": False, "reason": "重复补充词测试", "coverage": 0.4,
                        "gaps": ["缺数据"], "new_queries": already[:1] or ["兜底检索词"]}
            return super()._dispatch(text)

    with offline_search() as backend_d:
        agent_d = DeepResearchAgent(llm=RepeatQueryLLM(), max_iterations=5)
        final_d = agent_d.run("离线自检：重复补充词会不会导致死循环？")
    check("重复补充词被识别并停止", backend_d.search_calls == 1, f"实际 {backend_d.search_calls}")
    check("重复补充词场景仅反思一轮", len(final_d.get("reflections") or []) == 1)
    check("停止原因写明未给出全新检索词",
          "全新的补充检索词" in (final_d["reflections"][-1].get("reason") or ""),
          str(final_d["reflections"][-1]))


def main() -> int:
    print("=" * 60)
    print("Phase 2 自检：反思循环与补充搜索")
    print("=" * 60)
    test_helpers()
    test_brakes()
    test_routing()
    test_graph_shape()
    test_loop_end_to_end()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
