"""
LangGraph 状态图编排模块。
节点 = 处理步骤，边 = 流转（反思后条件跳转决定"继续搜索"还是"进入写作"）。

Phase 1 最小链路：
    planner → searcher → extractor → END

Phase 2 加入反思循环：
    planner → searcher → extractor → reflector ─┬─(信息不足)→ searcher
                                                 └─(信息充分)→ END

Phase 3 接入报告生成：
    planner → searcher → extractor → reflector ─┬─(信息不足)→ searcher
                                                 └─(信息充分)→ writer → END
"""
from langgraph.graph import END, StateGraph

from extractor import extractor_node
from planner import planner_node
from reflector import reflector_node, route_after_reflection
from searcher import searcher_node
from state import ResearchState
from writer import writer_node

# 递归上限：每轮循环约 3 个节点，留足余量给最大轮数
RECURSION_LIMIT = 80

# 条件边的分支名（graph 与 reflector 共用同一组常量，避免写错字符串）
BRANCH_SEARCH_AGAIN = "search_again"
BRANCH_FINISH = "finish"


def build_graph():
    """构建并编译研究状态图"""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "extractor")
    graph.add_edge("extractor", "reflector")

    # 条件边：信息不足就回到搜索执行器，充分则进入报告生成
    graph.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            BRANCH_SEARCH_AGAIN: "searcher",
            BRANCH_FINISH: "writer",
        },
    )
    graph.add_edge("writer", END)

    return graph.compile()


def describe_graph() -> str:
    """返回状态图的 Mermaid 描述，供 README 与前端展示"""
    return (
        "flowchart TD\n"
        "    START([用户研究问题]) --> planner[Planner 任务规划器]\n"
        "    planner --> searcher[Searcher 搜索执行器]\n"
        "    searcher --> extractor[Extractor 信息提取器]\n"
        "    extractor --> reflector{Reflector 反思决策器}\n"
        "    reflector -->|信息不足：补充搜索| searcher\n"
        "    reflector -->|信息充分| writer[Writer 报告生成器]\n"
        "    writer --> OUT([输出研究报告 + 引用来源])"
    )
