"""
Phase 8：量化评估脚本 —— 策略消融 + 自研 LLM 裁判。

用法：
    python eval_research.py                    # 跑完整矩阵（3 题 × 3 配置）
    python eval_research.py --offline          # 离线自检：假模型 + 假搜索，不消耗额度
    python eval_research.py --ids T1 T3        # 指定题目
    python eval_research.py --configs full no_local
    python eval_research.py --no-judge         # 跳过 LLM 裁判，只跑客观指标

评估两类指标：
  A. **客观指标**（零额外成本，来自已有的插桩）：任务完成率、事实数、引用数、
     引用可访问率、耗时、Token、成本、反思触发率。
  B. **报告质量**（自研 LLM 裁判，每题一次调用）：忠实度、相关性、覆盖度、结构性。
     忠实度是拿"报告里的事实清单"去核对报告正文，而不是让裁判凭空判断。

为什么不用 RAGAS：RAGAS 面向的是"问答 + 检索上下文"场景，而这里的产物是一份
带引用的长篇研究报告，RAGAS 的四个指标并不直接适用；而且它内部会多次调用 LLM
当裁判，成本是自研裁判的 2~3 倍。自研裁判的 rubric 可以针对研究报告定制，
也更可解释。

**同源偏见说明**：生成报告和评判报告用的是同一个模型（DeepSeek），存在"自己批改
自己"的倾向。本脚本通过"给出具体扣分点"的 rubric 降低主观性，但无法完全消除。
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config
import llm as llm_module
from deep_research import DeepResearchAgent, summarize
from state import to_dicts

console = Console()

QUESTIONS_FILE = config.DATA_DIR / "research_questions.json"

# ========== 消融配置 ==========
ABLATIONS = [
    {
        "key": "full",
        "label": "完整版（反思循环 + 本地知识库）",
        "overrides": {},
    },
    {
        "key": "no_reflect",
        "label": "无补充搜索（最大搜索轮数=1）",
        "overrides": {"max_iterations": 1},
    },
    {
        "key": "no_local",
        "label": "无本地知识库（只用网页搜索）",
        "overrides": {"use_local_kb": False},
    },
]

# 裁判 rubric：每个维度都给了 1.0 / 0.5 / 0.0 的锚点，减少打分随意性
JUDGE_PROMPT = """你是严格的研究报告评审，只依据下面给出的材料打分，不要自行补充外部知识。

【研究问题】
{question}

【研究计划（子问题）】
{plan}

【报告正文】
{report}

【报告中引用的事实清单】（编号 + 陈述 + 来源）
{facts}

【评分维度】每项 0.0~1.0：
1. faithfulness（忠实度）：报告里的**事实性陈述**是否都能在上面的事实清单里找到依据？
   - 1.0：事实性陈述基本都能对应到清单，且标注了引用编号
   - 0.5：有明显无依据的断言，或引用编号与内容对不上
   - 0.0：大量编造内容，或几乎没有引用
   注意：你只能比对"报告 vs 事实清单"，**不要判断事实清单本身对错**。
2. relevance（相关性）：报告是否直接回答了研究问题，而不是泛泛而谈？
3. coverage（覆盖度）：研究计划里的各个子问题是否都被实质性回答？
4. structure（结构性）：摘要 / 正文 / 结论 / 参考文献是否完整，章节是否逻辑递进？

【打分纪律】
- 不要因为报告"看起来长"就给高分，长而空泛要扣分。
- 报告里出现"现有资料尚不足以回答"这类**诚实的局限说明不扣分**，这恰恰说明没有编造。
- 必须写出具体扣分点（引用原文片段），不允许只写"整体不错"。
- 四个维度不要给成同一个分数，要能区分开。

【输出格式】只输出 JSON，不要解释文字：
{{"faithfulness": 0.8, "relevance": 0.7, "coverage": 0.6, "structure": 0.9,
  "deductions": "具体扣分点，引用报告里的原文片段",
  "strengths": "做得好的地方"}}"""

JUDGE_DIMENSIONS = ["faithfulness", "relevance", "coverage", "structure"]
JUDGE_LABELS = {
    "faithfulness": "忠实度",
    "relevance": "相关性",
    "coverage": "覆盖度",
    "structure": "结构性",
}


# ========== LLM 裁判 ==========
def render_facts_for_judge(facts: list, limit: int = 200) -> str:
    """把事实清单压成裁判可读的文本（控制长度，避免 Prompt 爆炸）"""
    lines = []
    for fact in to_dicts(facts)[:limit]:
        lines.append(
            f"{fact.get('id')}. {str(fact.get('claim', ''))[:120]} "
            f"｜来源 {fact.get('source_url', '')[:80]}"
        )
    return "\n".join(lines) or "（报告没有提炼出任何事实）"


def judge_report(question: str, plan: list, report_markdown: str, facts: list,
                 llm=None) -> dict:
    """用 LLM 给报告质量打分，返回 {维度: 分数} + 评语"""
    plan_text = "\n".join(
        f"{p.get('id')}. {p.get('question', '')}" for p in to_dicts(plan)
    ) or "（无计划）"

    prompt = JUDGE_PROMPT.format(
        question=question,
        plan=plan_text,
        report=report_markdown[:12000],
        facts=render_facts_for_judge(facts),
    )
    try:
        raw = llm_module.call_llm_json(prompt, llm=llm, retries=1)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", **{d: None for d in JUDGE_DIMENSIONS}}

    if not isinstance(raw, dict):
        return {"error": "裁判返回格式异常", **{d: None for d in JUDGE_DIMENSIONS}}

    result = {}
    for dimension in JUDGE_DIMENSIONS:
        try:
            value = float(raw.get(dimension))
        except (TypeError, ValueError):
            value = None
        result[dimension] = None if value is None else round(min(max(value, 0.0), 1.0), 3)
    result["deductions"] = str(raw.get("deductions") or "")[:600]
    result["strengths"] = str(raw.get("strengths") or "")[:400]
    return result


# ========== 单次运行 ==========
def run_one(question_item: dict, ablation: dict, out_root: Path, offline: bool,
            judge: bool) -> dict:
    """按某个消融配置跑一个研究问题，返回一行评估记录"""
    question_id = question_item.get("id", "Q")
    question = question_item["question"]
    overrides = dict(ablation.get("overrides") or {})

    console.print(f"\n[bold cyan]▶ {ablation['key']} × {question_id}[/bold cyan] "
                  f"[dim]{ablation['label']}[/dim]")

    # 每个组合都显式重置配置，避免上一组的影响串到下一组
    config.MAX_ITERATIONS = int(overrides.get("max_iterations", config.DEFAULT_MAX_ITERATIONS))
    config.USE_LOCAL_KB = bool(overrides.get("use_local_kb", config.DEFAULT_USE_LOCAL_KB))
    config.MAX_PAGES_PER_ROUND = int(overrides.get("max_pages", config.DEFAULT_MAX_PAGES))

    # 离线模式下必须**每次新建**假模型：它内部记录了"第几轮反思"的计数器，
    # 复用同一个实例会让后面的运行一上来就判"信息充分"，把矩阵结果污染掉。
    run_llm = None
    if offline:
        from offline_stubs import OfflineLLM

        run_llm = OfflineLLM(reflect_rounds=1)

    started = time.perf_counter()
    error = ""
    final_state = {}
    try:
        agent = DeepResearchAgent(
            max_iterations=config.MAX_ITERATIONS,
            use_local_kb=config.USE_LOCAL_KB,
            max_pages_per_round=config.MAX_PAGES_PER_ROUND,
            llm=run_llm,
        )
        final_state = agent.run(question)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        console.print(f"[red]✖ 运行失败：{error}[/red]")

    record = {
        "config": ablation["key"],
        "config_label": ablation["label"],
        "question_id": question_id,
        "question": question,
        "完成": bool(final_state.get("report_markdown")),
        "错误": error,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    if not final_state:
        return record

    metrics = summarize(final_state)
    record.update({
        "搜索轮数": final_state.get("iteration", 1),
        "反思轮数": len(to_dicts(final_state.get("reflections"))),
        "反思触发补充搜索": (final_state.get("stats") or {}).get("reflection_triggered", 0),
        "成功抓取网页数": metrics.get("成功抓取网页数", 0),
        "读取本地片段数": metrics.get("读取本地片段数", 0),
        "事实数": metrics.get("事实数", 0),
        "报告章节数": metrics.get("报告章节数", 0),
        "引用来源数": metrics.get("引用来源数", 0),
        "本地文档引用数": metrics.get("其中本地文档引用", 0),
        "越界引用数": metrics.get("越界引用数", 0),
        "报告字数": metrics.get("报告字数", 0),
        "Token 消耗": metrics.get("Token 消耗", 0),
        "成本(CNY)": float((final_state.get("stats") or {}).get("cost_cny", 0.0)),
    })

    report = final_state.get("report") or {}
    citations = report.get("citations") or []
    web_citations = [c for c in to_dicts(citations) if c.get("source_type") != "local"]
    record["网络引用数"] = len(web_citations)

    # 引用可访问性（跳过本地来源）
    if web_citations and not offline:
        try:
            from run_tests import check_citations

            stats = check_citations(web_citations)
            record["引用可访问率"] = stats["rate"]
            record["引用存在率"] = stats["exist_rate"]
        except Exception:
            record["引用可访问率"] = None
    else:
        record["引用可访问率"] = None

    # ---- LLM 裁判 ----
    if judge and final_state.get("report_markdown"):
        verdict = judge_report(question, final_state.get("plan"), final_state["report_markdown"],
                               final_state.get("facts"), llm=run_llm)
        for dimension in JUDGE_DIMENSIONS:
            record[f"judge_{dimension}"] = verdict.get(dimension)
        record["judge_deductions"] = verdict.get("deductions", "")
        if verdict.get("error"):
            record["judge_error"] = verdict["error"]
        scores = [verdict.get(d) for d in JUDGE_DIMENSIONS if isinstance(verdict.get(d), (int, float))]
        record["judge_avg"] = round(sum(scores) / len(scores), 3) if scores else None

    # 落盘单次产物
    try:
        paths = DeepResearchAgent.save_run(
            final_state, out_dir=out_root / ablation["key"] / question_id, formats=("md",)
        )
        record["产出目录"] = paths["dir"]
    except Exception as exc:
        record["产出目录"] = ""
        console.print(f"[yellow]产物落盘失败：{exc}[/yellow]")

    console.print(
        f"✔ {ablation['key']} × {question_id} 完成 ｜ "
        f"事实 {record['事实数']} ｜ 引用 {record['引用来源数']} ｜ "
        f"轮数 {record['搜索轮数']} ｜ {record['wall_seconds']}s ｜ "
        f"¥{record['成本(CNY)']:.4f}"
        + (f" ｜ 裁判均分 {record.get('judge_avg')}" if record.get("judge_avg") else "")
    )
    return record


# ========== 汇总 ==========
def _avg(values: list) -> float:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 3) if numbers else 0.0


def aggregate(records: list) -> dict:
    """按消融配置汇总，并给出与完整版的对比"""
    by_config = {}
    for record in records:
        by_config.setdefault(record["config"], []).append(record)

    summary = {}
    for key, rows in by_config.items():
        done = [r for r in rows if r.get("完成")]
        summary[key] = {
            "label": rows[0].get("config_label", key),
            "题目数": len(rows),
            "任务完成率": round(len(done) / len(rows), 3) if rows else 0.0,
            "平均搜索轮数": _avg([r.get("搜索轮数") for r in rows]),
            "平均事实数": _avg([r.get("事实数") for r in rows]),
            "平均引用来源数": _avg([r.get("引用来源数") for r in rows]),
            "平均本地引用数": _avg([r.get("本地文档引用数") for r in rows]),
            "平均报告字数": _avg([r.get("报告字数") for r in rows]),
            "引用可访问率": _avg([r.get("引用可访问率") for r in rows]),
            "平均耗时(秒)": _avg([r.get("wall_seconds") for r in rows]),
            "平均Token": _avg([r.get("Token 消耗") for r in rows]),
            "平均成本(CNY)": round(_avg([r.get("成本(CNY)") for r in rows]), 4),
            "总成本(CNY)": round(sum(r.get("成本(CNY)") or 0 for r in rows), 4),
            "裁判均分": _avg([r.get("judge_avg") for r in rows]),
        }
        for dimension in JUDGE_DIMENSIONS:
            summary[key][f"裁判_{JUDGE_LABELS[dimension]}"] = _avg(
                [r.get(f"judge_{dimension}") for r in rows]
            )

    # 与完整版的差值
    baseline = summary.get("full")
    if baseline:
        for key, data in summary.items():
            if key == "full":
                continue
            data["对比完整版"] = {
                "事实数": round(data["平均事实数"] - baseline["平均事实数"], 2),
                "引用来源数": round(data["平均引用来源数"] - baseline["平均引用来源数"], 2),
                "报告字数": round(data["平均报告字数"] - baseline["平均报告字数"], 1),
                "裁判均分": round(data["裁判均分"] - baseline["裁判均分"], 3),
                "成本(CNY)": round(data["平均成本(CNY)"] - baseline["平均成本(CNY)"], 4),
                "耗时(秒)": round(data["平均耗时(秒)"] - baseline["平均耗时(秒)"], 1),
            }
    return summary


def print_summary_table(summary: dict) -> None:
    table = Table(title="策略消融对比")
    table.add_column("配置", style="cyan")
    table.add_column("完成率", justify="right")
    table.add_column("轮数", justify="right")
    table.add_column("事实", justify="right")
    table.add_column("引用", justify="right")
    table.add_column("字数", justify="right")
    table.add_column("裁判均分", justify="right")
    table.add_column("Token", justify="right")
    table.add_column("¥", justify="right")
    table.add_column("耗时(s)", justify="right")
    for key, data in summary.items():
        table.add_row(
            f"{key}\n[dim]{data['label'][:22]}[/dim]",
            f"{data['任务完成率']:.0%}",
            f"{data['平均搜索轮数']:.1f}",
            f"{data['平均事实数']:.0f}",
            f"{data['平均引用来源数']:.1f}",
            f"{data['平均报告字数']:.0f}",
            f"{data['裁判均分']:.3f}" if data["裁判均分"] else "-",
            f"{data['平均Token']:.0f}",
            f"{data['平均成本(CNY)']:.4f}",
            f"{data['平均耗时(秒)']:.0f}",
        )
    console.print(table)


# ========== Markdown 报告 ==========
def render_markdown_report(summary: dict, records: list, stamp: str) -> str:
    """生成可直接放进 README 的评估报告"""
    lines = [
        "# Phase 8 量化评估：策略消融实验",
        "",
        f"> 生成时间：{stamp} ｜ 题目数：{len({r['question_id'] for r in records})} ｜ "
        f"配置数：{len(summary)} ｜ 总运行次数：{len(records)}",
        "",
        "## 一、总体对比",
        "",
        "| 配置 | 完成率 | 平均轮数 | 平均事实 | 平均引用 | 报告字数 | 裁判均分 | 平均 Token | 平均成本 | 平均耗时 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, data in summary.items():
        lines.append(
            f"| **{key}**<br>{data['label']} | {data['任务完成率']:.0%} | "
            f"{data['平均搜索轮数']:.1f} | {data['平均事实数']:.0f} | "
            f"{data['平均引用来源数']:.1f} | {data['平均报告字数']:.0f} | "
            f"{data['裁判均分']:.3f} | {data['平均Token']:.0f} | "
            f"¥{data['平均成本(CNY)']:.4f} | {data['平均耗时(秒)']:.0f}s |"
        )

    lines += ["", "## 二、报告质量四维度", "",
              "| 配置 | 忠实度 | 相关性 | 覆盖度 | 结构性 | 均分 |", "|---|---|---|---|---|---|"]
    for key, data in summary.items():
        lines.append(
            f"| {key} | {data['裁判_忠实度']:.3f} | {data['裁判_相关性']:.3f} | "
            f"{data['裁判_覆盖度']:.3f} | {data['裁判_结构性']:.3f} | {data['裁判均分']:.3f} |"
        )

    comparisons = {k: v["对比完整版"] for k, v in summary.items() if "对比完整版" in v}
    if comparisons:
        lines += ["", "## 三、相对完整版的差异", "",
                  "| 配置 | Δ事实 | Δ引用 | Δ字数 | Δ裁判均分 | Δ成本 | Δ耗时 |",
                  "|---|---|---|---|---|---|---|"]
        for key, diff in comparisons.items():
            lines.append(
                f"| {key} | {diff['事实数']:+.0f} | {diff['引用来源数']:+.1f} | "
                f"{diff['报告字数']:+.0f} | {diff['裁判均分']:+.3f} | "
                f"¥{diff['成本(CNY)']:+.4f} | {diff['耗时(秒)']:+.0f}s |"
            )

    lines += ["", "## 四、逐题明细", "",
              "| 配置 | 题目 | 轮数 | 事实 | 引用 | 裁判均分 | Token | 成本 |",
              "|---|---|---|---|---|---|---|---|"]
    for record in records:
        lines.append(
            f"| {record['config']} | {record['question_id']} | {record.get('搜索轮数', '-')} | "
            f"{record.get('事实数', '-')} | {record.get('引用来源数', '-')} | "
            f"{record.get('judge_avg') or '-'} | {record.get('Token 消耗', '-')} | "
            f"¥{record.get('成本(CNY)', 0):.4f} |"
        )

    deductions = [r for r in records if r.get("judge_deductions")]
    if deductions:
        lines += ["", "## 五、裁判给出的主要扣分点", ""]
        for record in deductions:
            lines.append(f"- **{record['config']} × {record['question_id']}**："
                         f"{record['judge_deductions'][:300]}")

    lines += [
        "",
        "## 六、方法与局限",
        "",
        "- **客观指标**来自程序内置插桩，零额外成本：完成率、轮数、事实数、引用数、字数、Token、成本、耗时。",
        "- **报告质量**由自研 LLM 裁判打分，四个维度各 0~1：忠实度、相关性、覆盖度、结构性。",
        "  忠实度的判定方式是「把报告正文与它自己引用的事实清单逐条比对」，而不是让裁判凭空判断。",
        "- 未使用 RAGAS：RAGAS 面向「问答 + 检索上下文」场景，其四指标不直接适配带引用的长篇研究报告，",
        "  且内部多次调用 LLM 当裁判，成本是自研裁判的 2~3 倍。",
        "- **同源偏见**：生成报告与评判报告用的是同一个模型，存在「自己批改自己」的倾向。",
        "  本脚本用「必须给出具体扣分点」的 rubric 降低主观性，但无法完全消除。",
        "- 样本量偏小（每题每配置 1 次），结论应视为**趋势**而非统计显著结论。",
        "",
        "### 复现方式",
        "",
        "```bash",
        "python eval_research.py              # 完整矩阵",
        "python eval_research.py --offline    # 离线自检，不消耗额度",
        "python eval_research.py --no-judge   # 只跑客观指标",
        "```",
        "",
    ]
    return "\n".join(lines)


# ========== 主流程 ==========
def load_questions(ids: list = None, limit: int = None) -> list:
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


def write_csv(records: list, path: Path) -> None:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deep Research Agent 策略消融评估")
    parser.add_argument("--ids", nargs="+", help="只跑指定题目 ID，如 T1 T3")
    parser.add_argument("--limit", type=int, help="只跑前 N 题")
    parser.add_argument("--configs", nargs="+", help="只跑指定配置，如 full no_local")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判")
    parser.add_argument("--offline", action="store_true",
                        help="离线自检：假模型 + 假搜索，不联网、不消耗额度")
    args = parser.parse_args(argv)

    if not args.offline and (not config.has_llm_key() or not config.has_tavily_key()):
        console.print("[red]缺少 API Key，请先运行 python main.py --doctor"
                      "（或用 --offline 做离线自检）[/red]")
        return 3

    questions = load_questions(args.ids, args.limit)
    if not questions:
        console.print("[red]没有可评估的问题[/red]")
        return 2

    ablations = ABLATIONS
    if args.configs:
        wanted = {c.lower() for c in args.configs}
        ablations = [a for a in ABLATIONS if a["key"] in wanted]
        if not ablations:
            console.print(f"[red]未知配置，可选：{[a['key'] for a in ABLATIONS]}[/red]")
            return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = config.OUTPUT_DIR / f"eval_{stamp}"
    total = len(questions) * len(ablations)
    console.print(f"[bold]评估矩阵：{len(questions)} 题 × {len(ablations)} 配置 = {total} 次研究[/bold]")
    console.print(f"产出目录：{out_root}\n")

    # 消融实验要反复改写全局配置，跑完必须还原——
    # 否则同进程里后续代码会莫名其妙地继承"最后一组配置"（实测踩过：
    # 跑完评估后 config.MAX_ITERATIONS 被留在 1）。
    original_config = {
        "MAX_ITERATIONS": config.MAX_ITERATIONS,
        "USE_LOCAL_KB": config.USE_LOCAL_KB,
        "MAX_PAGES_PER_ROUND": config.MAX_PAGES_PER_ROUND,
    }

    # 离线模式的桩必须覆盖整个矩阵，所以在这里统一装好
    if args.offline:
        from offline_stubs import offline_search

        manager = offline_search(with_local=False)
        manager.__enter__()
    else:
        manager = None

    records = []
    try:
        for question_item in questions:
            for ablation in ablations:
                records.append(run_one(question_item, ablation, out_root, args.offline,
                                       not args.no_judge))
    finally:
        if manager is not None:
            manager.__exit__(None, None, None)
        for key, value in original_config.items():
            setattr(config, key, value)

    summary = aggregate(records)
    console.print()
    print_summary_table(summary)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / f"phase8_{stamp}.csv"
    json_path = config.RESULTS_DIR / f"phase8_{stamp}.json"
    md_path = config.RESULTS_DIR / f"phase8_{stamp}_report.md"
    write_csv(records, csv_path)
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown_report(summary, records, stamp), encoding="utf-8")

    console.print(f"\n[green]逐次明细[/green] {csv_path}")
    console.print(f"[green]结构化结果[/green] {json_path}")
    console.print(f"[green]评估报告[/green] {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
