"""
Phase 8 自检脚本（不需要 API Key、不联网，全部用离线桩）。
运行方式：
    python tests/test_phase8.py

覆盖内容：
    1. 消融矩阵定义：配置齐全、默认值快照可用
    2. 裁判输出解析：正常 JSON、夹带文字、越界分数裁剪、异常降级
    3. 事实清单渲染：长度受控
    4. 汇总聚合：均值、完成率、与完整版的差值
    5. 报告渲染：Markdown 表格结构完整
    6. 离线端到端：真跑一遍矩阵（假模型 + 假搜索），验证产物齐全
    7. 配置隔离：跑完一组后配置能还原，不会串到下一组
"""
import json
import os
import sys
import tempfile
from pathlib import Path

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


def make_record(config_key: str, question_id: str, **overrides) -> dict:
    """构造一条评估记录"""
    record = {
        "config": config_key,
        "config_label": config_key,
        "question_id": question_id,
        "question": "测试问题",
        "完成": True,
        "错误": "",
        "wall_seconds": 100.0,
        "搜索轮数": 3,
        "反思轮数": 3,
        "反思触发补充搜索": 2,
        "成功抓取网页数": 20,
        "读取本地片段数": 0,
        "事实数": 100,
        "报告章节数": 5,
        "引用来源数": 15,
        "本地文档引用数": 0,
        "越界引用数": 0,
        "报告字数": 8000,
        "Token 消耗": 100000,
        "成本(CNY)": 0.2,
        "网络引用数": 15,
        "引用可访问率": 0.95,
        "judge_faithfulness": 0.8,
        "judge_relevance": 0.9,
        "judge_coverage": 0.7,
        "judge_structure": 0.85,
        "judge_avg": 0.8125,
        "judge_deductions": "扣分点示例",
    }
    record.update(overrides)
    return record


# ========== 1. 消融配置 ==========
def test_ablations() -> None:
    print("\n[1] 消融配置定义")
    import config
    from eval_research import ABLATIONS, JUDGE_DIMENSIONS

    keys = [a["key"] for a in ABLATIONS]
    check("包含完整版配置", "full" in keys, str(keys))
    check("包含无反思配置", "no_reflect" in keys, str(keys))
    check("包含无本地库配置", "no_local" in keys, str(keys))
    check("每个配置都有中文说明",
          all(a.get("label") for a in ABLATIONS), str([a.get("label") for a in ABLATIONS]))

    check("完整版没有覆盖项", ABLATIONS[0]["overrides"] == {}, str(ABLATIONS[0]))
    no_reflect = next(a for a in ABLATIONS if a["key"] == "no_reflect")
    check("无反思配置把轮数压到 1", no_reflect["overrides"].get("max_iterations") == 1,
          str(no_reflect))
    no_local = next(a for a in ABLATIONS if a["key"] == "no_local")
    check("无本地库配置关闭本地检索", no_local["overrides"].get("use_local_kb") is False,
          str(no_local))

    check("裁判有 4 个维度", len(JUDGE_DIMENSIONS) == 4, str(JUDGE_DIMENSIONS))
    check("默认值快照可用",
          config.DEFAULT_MAX_ITERATIONS >= 1 and config.DEFAULT_MAX_PAGES >= 1
          and isinstance(config.DEFAULT_USE_LOCAL_KB, bool),
          f"{config.DEFAULT_MAX_ITERATIONS}, {config.DEFAULT_MAX_PAGES}, {config.DEFAULT_USE_LOCAL_KB}")


# ========== 2. 裁判解析 ==========
def test_judge_parsing() -> None:
    print("\n[2] 裁判输出解析")
    from eval_research import JUDGE_DIMENSIONS, judge_report, render_facts_for_judge

    class FakeJudge:
        def __init__(self, payload):
            self.payload = payload

        class _Resp:
            def __init__(self, content):
                self.content = content
                self.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        def invoke(self, messages):
            return self._Resp(self.payload)

    # 正常 JSON
    verdict = judge_report("问题", [{"id": "Q1", "question": "子问题"}], "报告正文",
                           [{"id": "F1", "claim": "事实"}],
                           llm=FakeJudge('{"faithfulness":0.8,"relevance":0.9,'
                                         '"coverage":0.7,"structure":0.85,'
                                         '"deductions":"扣分","strengths":"优点"}'))
    check("四个维度都被解析",
          all(isinstance(verdict.get(d), float) for d in JUDGE_DIMENSIONS), str(verdict))
    check("数值正确", verdict["faithfulness"] == 0.8, str(verdict))
    check("评语被保留", verdict["deductions"] == "扣分", str(verdict))

    # 越界分数要被裁剪到 [0,1]
    clipped = judge_report("问题", [], "报告", [],
                           llm=FakeJudge('{"faithfulness":1.8,"relevance":-0.5,'
                                         '"coverage":0.5,"structure":0.5}'))
    check("超过 1 的分数被裁到 1", clipped["faithfulness"] == 1.0, str(clipped))
    check("低于 0 的分数被裁到 0", clipped["relevance"] == 0.0, str(clipped))

    # 维度缺失 → None 而不是崩溃
    missing = judge_report("问题", [], "报告", [],
                           llm=FakeJudge('{"faithfulness":0.8}'))
    check("缺失维度记为 None", missing["relevance"] is None and missing["faithfulness"] == 0.8,
          str(missing))

    # 非数值 → None
    bad_type = judge_report("问题", [], "报告", [],
                            llm=FakeJudge('{"faithfulness":"很好","relevance":0.5,'
                                          '"coverage":0.5,"structure":0.5}'))
    check("非数值维度记为 None", bad_type["faithfulness"] is None, str(bad_type))

    # 完全无法解析 → 返回 error 且不抛异常
    broken = judge_report("问题", [], "报告", [], llm=FakeJudge("这不是 JSON"))
    check("解析失败时返回 error 而不是抛异常", bool(broken.get("error")), str(broken)[:120])

    # LLM 抛异常也要兜住
    class ExplodingJudge:
        def invoke(self, messages):
            raise ConnectionError("模拟网络不可用")

    exploded = judge_report("问题", [], "报告", [], llm=ExplodingJudge())
    check("LLM 异常时返回 error", "ConnectionError" in str(exploded.get("error")),
          str(exploded)[:120])

    # 事实清单渲染要限长
    many = [{"id": f"F{i}", "claim": "很长的陈述" * 40, "source_url": "https://x.com/" + "a" * 50}
            for i in range(300)]
    text = render_facts_for_judge(many)
    check("事实清单有条数上限", text.count("\n") <= 200, str(text.count("\n")))
    check("单条事实被截断", max(len(line) for line in text.splitlines()) < 260,
          str(max(len(line) for line in text.splitlines())))
    check("空事实有占位文本", render_facts_for_judge([]) == "（报告没有提炼出任何事实）")


# ========== 3. 汇总聚合 ==========
def test_aggregate() -> None:
    print("\n[3] 汇总聚合")
    from eval_research import aggregate

    records = [
        make_record("full", "T1"),
        make_record("full", "T2", 事实数=120, **{"成本(CNY)": 0.3}),
        make_record("no_reflect", "T1", 搜索轮数=1, 事实数=60, 报告字数=5000,
                    judge_avg=0.6, **{"成本(CNY)": 0.1}),
        make_record("no_local", "T1", 本地文档引用数=0, 事实数=80,
                    judge_avg=0.75, **{"成本(CNY)": 0.15}),
    ]
    summary = aggregate(records)

    check("按配置分组", set(summary.keys()) == {"full", "no_reflect", "no_local"},
          str(list(summary.keys())))
    check("完整版事实数取平均", summary["full"]["平均事实数"] == 110.0,
          str(summary["full"]["平均事实数"]))
    check("完成率计算正确", summary["full"]["任务完成率"] == 1.0, str(summary["full"]))
    check("总成本累加正确", abs(summary["full"]["总成本(CNY)"] - 0.5) < 1e-6,
          str(summary["full"]["总成本(CNY)"]))
    check("裁判维度被翻译成中文",
          "裁判_忠实度" in summary["full"], str(list(summary["full"].keys())))

    diff = summary["no_reflect"].get("对比完整版")
    check("给出与完整版的差值", diff is not None, str(summary["no_reflect"].keys()))
    if diff:
        check("差值方向正确（事实变少为负）", diff["事实数"] < 0, str(diff))
        check("差值包含成本维度", "成本(CNY)" in diff, str(diff))

    # 完整版自己不该有"对比完整版"
    check("完整版不与自己对比", "对比完整版" not in summary["full"])

    # 未完成的运行不能让聚合崩掉
    failed = aggregate([make_record("x", "T1", 完成=False, 事实数=None,
                                    judge_avg=None, **{"成本(CNY)": None})])
    check("含失败记录时聚合仍可用", failed["x"]["任务完成率"] == 0.0, str(failed["x"]))
    check("None 值被安全忽略", failed["x"]["平均事实数"] == 0.0, str(failed["x"]))


# ========== 4. 报告渲染 ==========
def test_markdown_report() -> None:
    print("\n[4] 评估报告渲染")
    from eval_research import aggregate, render_markdown_report

    records = [make_record("full", "T1"), make_record("no_reflect", "T1", 事实数=60,
                                                      judge_avg=0.6)]
    summary = aggregate(records)
    markdown = render_markdown_report(summary, records, "20250101_000000")

    check("含标题", markdown.startswith("# Phase 8 量化评估"), markdown[:60])
    check("含总体对比表", "## 一、总体对比" in markdown)
    check("含四维度表", "## 二、报告质量四维度" in markdown)
    check("含差异表", "## 三、相对完整版的差异" in markdown)
    check("含逐题明细", "## 四、逐题明细" in markdown)
    check("含方法与局限", "## 六、方法与局限" in markdown)
    check("局限里写明同源偏见", "同源偏见" in markdown)
    check("局限里写明样本量小", "样本量偏小" in markdown)
    check("表格分隔行数量合理", markdown.count("|---") >= 4, str(markdown.count("|---")))
    check("不含未替换的占位符", "{" not in markdown.replace("{{", ""), "存在残留的格式化占位符")

    # 有扣分点时要列出来
    with_deduction = render_markdown_report(
        summary, [make_record("full", "T1", judge_deductions="第 3 章引用与内容对不上")], "x")
    check("列出裁判扣分点", "第 3 章引用与内容对不上" in with_deduction)


# ========== 5. 离线端到端 ==========
def test_offline_matrix() -> None:
    print("\n[5] 离线端到端：真跑一遍矩阵")
    import config
    from eval_research import aggregate, main as eval_main

    # 记住运行前的配置，稍后验证隔离性
    before = (config.MAX_ITERATIONS, config.USE_LOCAL_KB, config.MAX_PAGES_PER_ROUND)

    with tempfile.TemporaryDirectory() as tmp:
        # 把产物目录挪到临时目录，避免污染项目 outputs/
        original_output = config.OUTPUT_DIR
        original_results = config.RESULTS_DIR
        config.OUTPUT_DIR = Path(tmp) / "outputs"
        config.RESULTS_DIR = Path(tmp) / "results"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            exit_code = eval_main(["--offline", "--ids", "T1", "--configs",
                                   "full", "no_reflect", "--limit", "1"])
            check("离线矩阵正常结束", exit_code == 0, f"exit={exit_code}")

            csvs = list(config.RESULTS_DIR.glob("phase8_*.csv"))
            jsons = list(config.RESULTS_DIR.glob("phase8_*.json"))
            mds = list(config.RESULTS_DIR.glob("phase8_*_report.md"))
            check("产出 CSV", len(csvs) == 1, str(csvs))
            check("产出 JSON", len(jsons) == 1, str(jsons))
            check("产出评估报告 Markdown", len(mds) == 1, str(mds))

            if jsons:
                payload = json.loads(jsons[0].read_text(encoding="utf-8"))
                check("JSON 含 summary", "summary" in payload, str(list(payload.keys())))
                check("JSON 含 records", len(payload.get("records") or []) == 2,
                      str(len(payload.get("records") or [])))
                check("两种配置都被记录",
                      {r["config"] for r in payload["records"]} == {"full", "no_reflect"},
                      str({r["config"] for r in payload["records"]}))

                # 关键：无反思配置的轮数必须真的比完整版少
                by_config = {r["config"]: r for r in payload["records"]}
                check("完整版触发了补充搜索",
                      by_config["full"]["搜索轮数"] > by_config["no_reflect"]["搜索轮数"],
                      f"full={by_config['full']['搜索轮数']} "
                      f"no_reflect={by_config['no_reflect']['搜索轮数']}")
                check("无反思配置确实只跑 1 轮",
                      by_config["no_reflect"]["搜索轮数"] == 1,
                      str(by_config["no_reflect"]["搜索轮数"]))
                check("离线运行有裁判分数",
                      all(r.get("judge_avg") for r in payload["records"]),
                      str([r.get("judge_avg") for r in payload["records"]]))
                check("离线运行成本为 0",
                      all(r.get("成本(CNY)") == 0.0 for r in payload["records"]),
                      str([r.get("成本(CNY)") for r in payload["records"]]))
        finally:
            config.OUTPUT_DIR = original_output
            config.RESULTS_DIR = original_results

    after = (config.MAX_ITERATIONS, config.USE_LOCAL_KB, config.MAX_PAGES_PER_ROUND)
    check("评估跑完后配置被还原（不污染后续运行）", before == after,
          f"运行前 {before} → 运行后 {after}")


def main() -> int:
    print("=" * 60)
    print("Phase 8 自检：量化评估与策略消融")
    print("=" * 60)
    test_ablations()
    test_judge_parsing()
    test_aggregate()
    test_markdown_report()
    test_offline_matrix()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
