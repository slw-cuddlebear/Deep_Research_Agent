"""
命令行入口：一次深度研究的最简调用方式。

用法示例：
    python main.py "2024 年国内大模型推理成本下降了多少"
    python main.py "问题" --max-iter 3 --engine tavily
    python main.py "问题" --interactive      # 先看研究计划，可手动修改再执行
    python main.py "问题" --plan-only        # 只生成研究计划
    python main.py "问题" --offline          # 离线自检（用假模型 + 假搜索，不消耗 API）
    python main.py --doctor                  # 环境体检：依赖 / Key / 网络 / 连通性
"""
import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from deep_research import DeepResearchAgent, summarize
from state import to_dicts

console = Console()


# ========== 展示辅助 ==========
def print_env() -> None:
    """打印当前环境配置"""
    table = Table(title="运行环境", show_header=False, box=None)
    for key, value in config.describe_env().items():
        table.add_row(f"[cyan]{key}[/cyan]", str(value))
    console.print(table)


def print_plan(plan: list, plan_summary: str = "") -> None:
    """打印研究计划"""
    if plan_summary:
        console.print(Panel(plan_summary, title="研究思路", border_style="blue"))
    table = Table(title="研究计划", show_lines=False)
    table.add_column("编号", style="cyan", width=6)
    table.add_column("子问题", style="white")
    table.add_column("检索词", style="dim")
    for item in to_dicts(plan):
        table.add_row(
            item.get("id", ""),
            item.get("question", ""),
            " / ".join(item.get("search_queries") or []),
        )
    console.print(table)


def print_facts(facts: list, limit: int = 15) -> None:
    """打印提炼出的事实"""
    facts = to_dicts(facts)
    if not facts:
        console.print("[yellow]未提炼到任何事实[/yellow]")
        return
    table = Table(title=f"提炼事实（共 {len(facts)} 条，展示前 {min(limit, len(facts))} 条）")
    table.add_column("编号", style="cyan", width=6)
    table.add_column("事实", style="white")
    table.add_column("置信度", width=8)
    table.add_column("可信度", width=8)
    table.add_column("来源", style="dim", overflow="fold")
    for item in facts[:limit]:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("claim", ""))[:80],
            f"{item.get('confidence', 0):.2f}",
            f"{item.get('credibility', 0):.2f}",
            str(item.get("source_url", ""))[:60],
        )
    console.print(table)


def print_reflections(reflections: list) -> None:
    """打印反思决策轨迹"""
    reflections = to_dicts(reflections)
    if not reflections:
        return
    table = Table(title=f"反思轨迹（共 {len(reflections)} 轮）")
    table.add_column("轮次", width=6, style="cyan")
    table.add_column("结论", width=14)
    table.add_column("覆盖率", width=8)
    table.add_column("判断理由", style="white", overflow="fold")
    for item in reflections:
        sufficient = item.get("sufficient")
        verdict = "[green]信息充分[/green]" if sufficient else "[yellow]需补充搜索[/yellow]"
        table.add_row(
            str(item.get("iteration", "")),
            verdict,
            f"{item.get('coverage', 0):.0%}",
            str(item.get("reason", ""))[:80],
        )
    console.print(table)


def print_trace(state: dict, limit: int = 10) -> None:
    """打印可观测性信息：成本汇总 + 最慢的若干步骤"""
    trace = state.get("trace") or {}
    if not trace:
        return

    from observability import slowest_spans, trace_summary

    table = Table(title="可观测性 / 成本核算", show_header=False, box=None)
    for key, value in trace_summary(trace).items():
        table.add_row(f"[cyan]{key}[/cyan]", str(value))
    console.print(table)

    slow = slowest_spans(trace, limit=limit)
    if slow:
        detail = Table(title="最慢的步骤")
        detail.add_column("步骤", style="cyan")
        detail.add_column("类型", width=10)
        detail.add_column("耗时(秒)", justify="right", width=10)
        detail.add_column("Token", justify="right", width=9)
        for span in slow:
            detail.add_row(
                str(span.get("name", "")),
                str(span.get("kind", "")),
                f"{span.get('duration', 0):.2f}",
                str((span.get("usage") or {}).get("total_tokens", "")),
            )
        console.print(detail)

    if trace.get("langfuse_enabled"):
        console.print("[green]✔ 已上报 LangFuse[/green]" + (
            f" [red]（有异常：{trace.get('langfuse_error')}）[/red]"
            if trace.get("langfuse_error") else ""
        ))
    elif config.LANGFUSE_PUBLIC_KEY:
        console.print("[yellow]⚠ LangFuse Key 不完整，未上报[/yellow]")
    else:
        console.print("[dim]· 未配置 LangFuse Key，仅本地追踪[/dim]")


def make_event_printer(verbose: bool = False):
    """构造实时打印事件的回调"""
    icons = {"start": "▶", "done": "✔", "warn": "⚠", "error": "✖"}

    def printer(event: dict) -> None:
        icon = icons.get(event.get("status", "done"), "·")
        color = {"warn": "yellow", "error": "red"}.get(event.get("status", "done"), "green")
        line = f"[{color}]{icon}[/{color}] [bold]{event.get('step', '')}[/bold] {event.get('title', '')}"
        if event.get("elapsed"):
            line += f" [dim]({event['elapsed']}s)[/dim]"
        console.print(line)
        if verbose and event.get("detail"):
            console.print(f"    [dim]{event['detail']}[/dim]")

    return printer


# ========== 离线自检 ==========
def run_offline(question: str, reflect_rounds: int = 1, save: bool = True,
                formats: tuple = ("md", "html", "pdf")) -> int:
    """
    不联网、不消耗 API 的链路自检。
    默认让反思节点先回答一次"信息不足"，以便把反思循环也走一遍。
    """
    from offline_stubs import OfflineLLM, offline_search

    console.print(Panel(
        "离线自检模式：使用假模型与假搜索结果，仅验证状态图链路（不联网、不消耗额度）",
        style="yellow",
    ))

    with offline_search() as backend:
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=reflect_rounds))
        final = agent.run(question, on_event=make_event_printer())

    print_plan(final.get("plan"), final.get("plan_summary"))
    print_facts(final.get("facts"))
    print_reflections(final.get("reflections"))
    if final.get("report_markdown"):
        console.rule("[bold]研究报告（离线假数据）[/bold]")
        console.print(final["report_markdown"])
    print_summary(final)
    console.print(f"[dim]离线搜索被调用 {backend.search_calls} 次[/dim]")

    if save:
        result = DeepResearchAgent.save_run(final, formats=formats)
        console.print(f"[green]产物已写入[/green] {result['dir']}")
        for name, path in result["paths"].items():
            console.print(f"  · [cyan]{name}[/cyan] → {path}")
        for warning in result["warnings"]:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")
    return 0


# ========== 结果汇总 ==========
def print_summary(state: dict) -> None:
    """打印运行指标"""
    table = Table(title="运行指标", show_header=False, box=None)
    for key, value in summarize(state).items():
        table.add_row(f"[cyan]{key}[/cyan]", str(value))
    console.print(table)


# ========== 主入口 ==========
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep Research Agent —— 输入一个研究问题，输出带引用来源的结构化报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="?", help="研究问题")
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS,
                        help=f"最大搜索轮数，默认 {config.MAX_ITERATIONS}")
    parser.add_argument("--engine", choices=["auto", "tavily", "ddg"], default="auto",
                        help="网页搜索后端，默认 auto（有 Tavily Key 用 Tavily，否则用 DuckDuckGo）")
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES_PER_ROUND,
                        help=f"单轮最多阅读网页数，默认 {config.MAX_PAGES_PER_ROUND}")
    parser.add_argument("--interactive", action="store_true", help="先展示研究计划，可手动修改后再执行")
    parser.add_argument("--plan-only", action="store_true", help="只生成研究计划")
    parser.add_argument("--offline", action="store_true", help="离线自检（假模型 + 假搜索，不消耗 API）")
    parser.add_argument("--doctor", action="store_true", help="环境体检：依赖 / Key / 网络 / 连通性一次查清")
    parser.add_argument("--formats", nargs="+", choices=["md", "html", "pdf"], default=["md", "html", "pdf"],
                        help="报告导出格式，默认全部导出（md/html/pdf）")
    parser.add_argument("--no-local", action="store_true",
                        help="关闭本地知识库，只用网页搜索")
    parser.add_argument("--rebuild-kb", action="store_true",
                        help="强制重建本地知识库索引后退出")
    parser.add_argument("--no-save", action="store_true", help="不把运行产物写入 outputs/")
    parser.add_argument("--verbose", action="store_true", help="打印事件详情")
    return parser


def print_kb_status(status: dict) -> None:
    """打印本地知识库状态"""
    if status.get("documents"):
        chunks = status.get("chunks") or 0
        console.print(
            f"[cyan]本地知识库[/cyan] {status['documents']} 篇文档 / {chunks} 个片段"
            f" [dim]（{status.get('message', '')}）[/dim]"
        )
    else:
        console.print(f"[yellow]本地知识库[/yellow] {status.get('message', '未启用')}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.doctor:
        from doctor import run_doctor

        return run_doctor()

    if args.rebuild_kb:
        from local_kb import build_index

        info = build_index(force=True)
        console.print(f"[green]{info['message']}[/green]")
        return 0

    question = (args.question or "").strip()
    if not question:
        question = input("请输入研究问题：").strip()
    if not question:
        console.print("[red]研究问题不能为空[/red]")
        return 2

    if args.offline:
        return run_offline(question, save=not args.no_save, formats=tuple(args.formats))

    print_env()
    print_kb_status(DeepResearchAgent.kb_status())
    if not config.has_llm_key():
        console.print(Panel(
            "未检测到 DEEPSEEK_API_KEY。\n"
            "请复制 .env.example 为 .env 并填入你的 Key（申请地址：https://platform.deepseek.com），\n"
            "或者先用 `python main.py \"问题\" --offline` 做离线链路自检。",
            title="缺少 API Key", border_style="red",
        ))
        return 3

    agent = DeepResearchAgent(
        max_iterations=args.max_iter, engine=args.engine, max_pages_per_round=args.max_pages,
        use_local_kb=not args.no_local,
    )

    plan_payload = None
    if args.interactive:
        console.print("[bold]正在生成研究计划…[/bold]")
        plan_payload = agent.plan(question)
        print_plan(plan_payload["plan"], plan_payload.get("plan_summary", ""))
        while True:
            feedback = input("回车确认并开始研究，或输入修改意见后回车：").strip()
            if not feedback:
                break
            console.print("[bold]正在按你的意见修订计划…[/bold]")
            plan_payload = agent.revise_plan(question, plan_payload["plan"], feedback)
            print_plan(plan_payload["plan"], plan_payload.get("plan_summary", ""))

    if args.plan_only:
        plan_payload = plan_payload or agent.plan(question)
        print_plan(plan_payload["plan"], plan_payload.get("plan_summary", ""))
        return 0

    console.rule("[bold]开始研究[/bold]")
    final_state = agent.run(
        question,
        on_event=make_event_printer(args.verbose),
        plan=plan_payload["plan"] if plan_payload else None,
        plan_summary=plan_payload.get("plan_summary", "") if plan_payload else "",
    )

    if not plan_payload:
        print_plan(final_state.get("plan"), final_state.get("plan_summary"))
    print_facts(final_state.get("facts"))
    print_reflections(final_state.get("reflections"))

    if final_state.get("report_markdown"):
        console.rule("[bold]研究报告[/bold]")
        console.print(final_state["report_markdown"])

    print_summary(final_state)
    print_trace(final_state)

    if not args.no_save:
        result = DeepResearchAgent.save_run(final_state, formats=tuple(args.formats))
        console.rule("[bold]产物落盘[/bold]")
        console.print(f"输出目录：[green]{result['dir']}[/green]")
        for name, path in result["paths"].items():
            console.print(f"  · [cyan]{name}[/cyan] → {path}")
        for warning in result["warnings"]:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
