"""
Phase 4：端到端批量测试脚本。
把 data/research_questions.json 里的研究问题逐个跑完整链路，汇总成指标表。

用法：
    python run_tests.py                    # 跑全部问题
    python run_tests.py --limit 1          # 只跑第 1 个
    python run_tests.py --ids T1 T3        # 只跑指定问题
    python run_tests.py --no-pdf           # 不导出 PDF（快一点）
    python run_tests.py --no-check-links   # 跳过引用链接可访问性检查

产物：
    outputs/batch_<时间戳>/<问题ID>/        每次研究的完整产物
    results/phase4_<时间戳>.csv            逐题指标
    results/phase4_<时间戳>_summary.json   汇总指标
"""
import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config
from deep_research import DeepResearchAgent, summarize
from state import to_dicts

console = Console()

QUESTIONS_FILE = config.DATA_DIR / "research_questions.json"
LINK_CHECK_TIMEOUT = 12
LINK_CHECK_WORKERS = 5


# ========== 引用链接可访问性 ==========
# 区分"链接失效"和"站点反爬"很重要：403/429 是对方拒绝我们的请求，
# 不代表链接不存在。而且能进入参考文献的 URL，在检索阶段其实已经被成功抓取过，
# 所以把反爬当成"引用错误"会显著低估真实准确率。
BLOCKED_STATUS = {401, 402, 403, 405, 406, 407, 429, 503}
DEAD_STATUS = {404, 410, 451, 400}


def check_url(url: str) -> tuple:
    """
    检查一个引用链接是否真实可访问。
    返回 (url, 状态分类, 详情)；状态分类为 ok / blocked / dead / error / local。
    HEAD 被拒时退回 GET（很多站点不支持 HEAD）。
    """
    import requests

    if not url:
        return url, "error", "无链接"
    # 本地知识库来源不是网络链接，用 HTTP 去"访问"它没有意义
    if url.startswith("local://"):
        return url, "local", "本地知识库文档"

    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    last = "未知"
    for method in ("head", "get"):
        try:
            response = requests.request(
                method, url, headers=headers, timeout=LINK_CHECK_TIMEOUT,
                allow_redirects=True, stream=(method == "get"),
            )
            status = response.status_code
            if method == "get":
                response.close()
            if status < 400:
                return url, "ok", status
            if status in BLOCKED_STATUS:
                last = f"{status}（站点反爬，链接未必失效）"
                continue
            if status in DEAD_STATUS:
                return url, "dead", status
            last = status
        except Exception as exc:
            last = f"{type(exc).__name__}"
    return url, "blocked" if isinstance(last, str) and "反爬" in last else "error", last


def check_citations(citations: list) -> dict:
    """
    并发检查全部引用链接，返回分类统计。

    **本地知识库来源单独计数**：它们不是网络链接，混进分母会把可访问率算成错的
    （实测过：21 条引用里 5 条是本地文档，直接把 100% 拉低到 71%）。
    """
    urls = [c.get("url") for c in to_dicts(citations) if c.get("url")]
    if not urls:
        return {"total": 0, "ok": 0, "blocked": 0, "dead": 0, "error": 0, "local": 0,
                "web_total": 0, "rate": 0.0, "strict_rate": 0.0, "exist_rate": 0.0, "details": []}

    results = []
    with ThreadPoolExecutor(max_workers=LINK_CHECK_WORKERS) as pool:
        futures = {pool.submit(check_url, url): url for url in urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append((futures[future], "error", str(exc)))

    counts = {"ok": 0, "blocked": 0, "dead": 0, "error": 0, "local": 0}
    for _, category, _ in results:
        counts[category] = counts.get(category, 0) + 1

    total = len(urls)
    web_total = total - counts["local"]
    strict = counts["ok"] / web_total if web_total else 0.0
    exists = (counts["ok"] + counts["blocked"]) / web_total if web_total else 0.0
    return {
        "total": total,
        "web_total": web_total,
        "ok": counts["ok"],
        "blocked": counts["blocked"],
        "dead": counts["dead"],
        "error": counts["error"],
        "local": counts["local"],
        "rate": round(strict, 3),
        "strict_rate": round(strict, 3),
        "exist_rate": round(exists, 3),
        "details": [{"url": u, "category": c, "status": s} for u, c, s in results],
    }


# ========== 单题执行 ==========
def run_one(item: dict, out_root: Path, formats: tuple, check_links: bool) -> dict:
    """跑一个研究问题，返回本题的指标记录"""
    question_id = item.get("id") or "Q"
    question = item["question"]
    console.rule(f"[bold]{question_id}[/bold] {question}")

    started = time.perf_counter()
    agent = DeepResearchAgent()
    try:
        final_state = agent.run(question, on_event=_headline_printer())
        error = ""
    except Exception as exc:
        console.print(f"[red]✖ 运行失败：{type(exc).__name__}: {exc}[/red]")
        return {
            "id": question_id, "question": question, "type": item.get("type", ""),
            "完成": False, "错误": f"{type(exc).__name__}: {exc}"[:300],
            "耗时(秒)": round(time.perf_counter() - started, 1),
        }

    paths = DeepResearchAgent.save_run(final_state, out_dir=out_root / question_id, formats=formats)
    metrics = summarize(final_state)
    report = final_state.get("report") or {}
    citations = report.get("citations") or []
    markdown = final_state.get("report_markdown") or ""

    link_stats = {"total": len(citations), "ok": 0, "blocked": 0, "dead": 0,
                  "error": 0, "local": 0, "rate": 0.0, "exist_rate": 0.0}
    if check_links and citations:
        link_stats = check_citations(citations)

    record = {
        "id": question_id,
        "question": question,
        "type": item.get("type", ""),
        "完成": bool(markdown) and len(citations) > 0,
        "错误": error,
        **metrics,
        "网络引用数": link_stats.get("web_total", 0),
        "引用可访问数": link_stats["ok"],
        "引用严格可访问率": link_stats["rate"],
        "引用反爬拦截数": link_stats.get("blocked", 0),
        "引用链接存在率": link_stats.get("exist_rate", 0.0),
        "引用疑似失效数": link_stats.get("dead", 0),
        "本地文档引用数": link_stats.get("local", 0),
        "产出目录": paths["dir"],
    }

    if link_stats.get("details"):
        detail_path = Path(paths["dir"]) / "citation_check.json"
        detail_path.write_text(
            json.dumps(link_stats["details"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        record["引用校验明细"] = str(detail_path)

    console.print(
        f"[green]✔ {question_id} 完成[/green] "
        f"事实 {metrics.get('事实数')} 条 ｜ 引用 {len(citations)} 条 ｜ "
        f"严格可访问 {link_stats['rate']:.0%} / 链接存在 {link_stats.get('exist_rate', 0):.0%} ｜ "
        f"{metrics.get('耗时(秒)')}s ｜ Token {metrics.get('Token 消耗')}"
    )
    return record


def _headline_printer():
    """只打印关键事件，避免批量测试刷屏"""
    def printer(event: dict) -> None:
        step = event.get("step", "")
        if step in ("planner", "searcher", "reflector", "writer") or event.get("status") == "error":
            icon = {"warn": "⚠", "error": "✖"}.get(event.get("status"), "·")
            console.print(f"  {icon} [dim]{step}[/dim] {event.get('title', '')}")
    return printer


# ========== 汇总 ==========
def load_questions(ids: list = None, limit: int = None) -> list:
    """读取测试问题集"""
    if not QUESTIONS_FILE.exists():
        console.print(f"[red]问题集不存在：{QUESTIONS_FILE}[/red]")
        return []
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    items = data.get("questions") or data
    if ids:
        wanted = {i.upper() for i in ids}
        items = [q for q in items if str(q.get("id", "")).upper() in wanted]
    if limit:
        items = items[:limit]
    return items


def summarize_records(records: list) -> dict:
    """计算整体指标（对应规划"七、量化指标"）"""
    done = [r for r in records if r.get("完成")]

    def avg(key, rows=None):
        rows = rows if rows is not None else records
        values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(values) / len(values), 2) if values else 0

    def total(key):
        return sum(r.get(key) for r in records if isinstance(r.get(key), (int, float)))

    # 分母只用**网络引用**：本地知识库来源不是网络链接，混进来会把可访问率算错
    total_web = total("网络引用数") or total("引用来源数")
    strict_ok = total("引用可访问数")
    blocked = total("引用反爬拦截数")
    dead = total("引用疑似失效数")

    return {
        "题目数": len(records),
        "任务完成数": len(done),
        "任务完成率": round(len(done) / len(records), 3) if records else 0.0,
        "平均搜索步数": avg("搜索轮数"),
        "平均反思轮数": avg("反思轮数"),
        "反思触发率": round(
            sum(1 for r in records if (r.get("反思触发补充搜索") or 0) > 0) / len(records), 3
        ) if records else 0.0,
        "平均抓取网页数": avg("成功抓取网页数"),
        "平均事实数": avg("事实数"),
        "平均引用来源数": avg("引用来源数"),
        "平均本地引用数": avg("本地文档引用数"),
        "引用准确率(严格可访问)": round(strict_ok / total_web, 3) if total_web else 0.0,
        "引用准确率(链接存在)": round((strict_ok + blocked) / total_web, 3) if total_web else 0.0,
        "引用反爬拦截数": blocked,
        "引用疑似失效数": dead,
        "平均耗时(秒)": avg("耗时(秒)"),
        "平均Token": avg("Token 消耗"),
        "总Token": total("Token 消耗"),
    }


def write_csv(records: list, path: Path) -> None:
    """把逐题指标写成 CSV"""
    if not records:
        return
    columns = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def print_summary_table(summary: dict) -> None:
    table = Table(title="Phase 4 端到端测试汇总")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="white")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)


# ========== 仅重查引用链接 ==========
def _find_state_files(paths: list) -> list:
    """把输入的目录/文件展开成 state.json 路径列表"""
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "state.json":
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("state.json")))
    return found


def _derive_question_id(state_path: Path) -> str:
    """从 <...>/<题目ID>/<时间戳>/state.json 里推出题目 ID"""
    parts = state_path.parts
    if len(parts) >= 3 and re.fullmatch(r"[A-Za-z]\d+", parts[-3]):
        return parts[-3]
    return ""


def recheck_citations(paths: list, csv_path: Path = None) -> int:
    """
    只重新核对已有运行的引用链接，不重跑研究（省时省钱）。
    用法：python run_tests.py --recheck outputs/batch_xxx
    """
    state_files = _find_state_files(paths)
    if not state_files:
        console.print("[red]没有找到任何 state.json[/red]")
        return 2

    records = []
    for state_path in state_files:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[yellow]跳过 {state_path}：{exc}[/yellow]")
            continue

        question_id = _derive_question_id(state_path) or state_path.parent.name
        citations = (state.get("report") or {}).get("citations") or []
        stats = state.get("stats") or {}
        metrics = summarize(state)
        link_stats = check_citations(citations)

        record = {
            "id": question_id,
            "question": state.get("question", ""),
            "完成": bool(state.get("report_markdown")) and len(citations) > 0,
            "引用来源数": len(citations),
            "网络引用数": link_stats.get("web_total", 0),
            "引用可访问数": link_stats["ok"],
            "引用严格可访问率": link_stats["rate"],
            "引用反爬拦截数": link_stats["blocked"],
            "引用链接存在率": link_stats["exist_rate"],
            "引用疑似失效数": link_stats["dead"],
            "本地文档引用数": link_stats.get("local", 0),
            "搜索轮数": state.get("iteration", 1),
            "反思轮数": len(to_dicts(state.get("reflections"))),
            "反思触发补充搜索": stats.get("reflection_triggered", 0),
            "成功抓取网页数": len([p for p in to_dicts(state.get("pages"))
                                   if p.get("ok") and p.get("source") != "local"]),
            "读取本地片段数": len([p for p in to_dicts(state.get("pages"))
                                   if p.get("source") == "local"]),
            "事实数": len(to_dicts(state.get("facts"))),
            "耗时(秒)": stats.get("elapsed", 0),
            "Token 消耗": (stats.get("llm_usage") or {}).get("total_tokens", 0),
            "子问题覆盖率": metrics.get("子问题覆盖率", ""),
            "产出目录": str(state_path.parent),
        }
        records.append(record)

        console.print(
            f"[green]✔ {question_id}[/green] 引用 {len(citations)} 条 ｜ "
            f"严格可访问 {link_stats['rate']:.0%} ｜ 反爬拦截 {link_stats['blocked']} ｜ "
            f"疑似失效 {link_stats['dead']} ｜ 链接存在率 {link_stats['exist_rate']:.0%}"
        )
        detail_path = state_path.parent / "citation_check.json"
        detail_path.write_text(
            json.dumps(link_stats["details"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = summarize_records(records)
    print_summary_table(summary)

    csv_path = Path(csv_path) if csv_path else config.RESULTS_DIR / "phase4_citation_recheck.csv"
    write_csv(records, csv_path)
    console.print(f"[green]结果已写入[/green] {csv_path}")
    return 0


# ========== 主入口 ==========
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deep Research Agent 端到端批量测试")
    parser.add_argument("--ids", nargs="+", help="只跑指定问题 ID，如 T1 T3")
    parser.add_argument("--limit", type=int, help="只跑前 N 个问题")
    parser.add_argument("--no-pdf", action="store_true", help="不导出 PDF")
    parser.add_argument("--no-check-links", action="store_true", help="跳过引用链接可访问性检查")
    parser.add_argument("--recheck", nargs="+", metavar="PATH",
                        help="只重新核对已有运行的引用链接（不重跑研究），可给目录或 state.json")
    parser.add_argument("--csv", help="--recheck 模式下结果 CSV 的输出路径")
    args = parser.parse_args(argv)

    if args.recheck:
        return recheck_citations(args.recheck, args.csv)

    if not config.has_llm_key() or not config.has_tavily_key():
        console.print("[red]缺少 DEEPSEEK_API_KEY 或 TAVILY_API_KEY，请先运行 python main.py --doctor[/red]")
        return 3

    questions = load_questions(args.ids, args.limit)
    if not questions:
        console.print("[red]没有可运行的问题[/red]")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = config.OUTPUT_DIR / f"batch_{stamp}"
    formats = ("md", "html") if args.no_pdf else ("md", "html", "pdf")

    console.print(f"[bold]共 {len(questions)} 个研究问题[/bold]，产出目录：{out_root}")
    records = []
    for index, item in enumerate(questions, start=1):
        console.print(f"\n[bold cyan]({index}/{len(questions)})[/bold cyan]")
        records.append(run_one(item, out_root, formats, not args.no_check_links))

    summary = summarize_records(records)
    print_summary_table(summary)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / f"phase4_{stamp}.csv"
    json_path = config.RESULTS_DIR / f"phase4_{stamp}_summary.json"
    write_csv(records, csv_path)
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"\n[green]逐题指标[/green] {csv_path}")
    console.print(f"[green]汇总结果[/green] {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
