"""
Phase 7 自检脚本（不需要 API Key、不需要 LangFuse 账号、不联网）。
运行方式：
    python tests/test_phase7.py

覆盖内容：
    1. 成本核算：价格表、高峰/错峰计价、缓存命中折扣、汇率换算
    2. Span 树：父子关系、耗时、并发下的上下文继承
    3. 并发正确性：用**真实的 OpenTelemetry 内存导出器**验证
       子线程里的 span 确实挂在调用线程的父 span 下
       （LangFuse v4 就是基于 OTEL 的，所以这条等价于验证上报结构正确）
    4. LangFuse 接入：未配置 Key 时优雅降级；配置后用**假客户端**验证
       确实发出了正确形状的 observation / 事件
    5. 观测不能拖垮主流程：假客户端抛异常时主流程仍然跑完
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

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


# ========== 1. 成本核算 ==========
def test_cost() -> None:
    print("\n[1] 成本核算")
    import config
    from observability import estimate_cost, is_peak, price_for

    price = price_for("deepseek-chat")
    check("价格表命中 deepseek-chat", price["input"] == 0.15 and price["output"] == 0.60,
          str(price))
    check("未知模型有兜底价格", price_for("某个不存在的模型")["input"] > 0)

    # 错峰时段：2025-01-08（周三）12:00 UTC
    off_peak_at = datetime(2025, 1, 8, 12, 0, tzinfo=timezone.utc)
    peak_at = datetime(2025, 1, 8, 2, 0, tzinfo=timezone.utc)       # 周三 02:00 UTC → 高峰
    weekend_at = datetime(2025, 1, 11, 2, 0, tzinfo=timezone.utc)   # 周六 02:00 UTC → 错峰

    check("周三 12:00 UTC 判为错峰", is_peak(off_peak_at) is False)
    check("周三 02:00 UTC 判为高峰", is_peak(peak_at) is True)
    check("周六 02:00 UTC 判为错峰", is_peak(weekend_at) is False)
    check("周三 07:00 UTC 判为高峰",
          is_peak(datetime(2025, 1, 8, 7, 0, tzinfo=timezone.utc)) is True)
    check("周三 05:00 UTC 判为错峰（高峰间隙）",
          is_peak(datetime(2025, 1, 8, 5, 0, tzinfo=timezone.utc)) is False)

    # 1M 输入 + 1M 输出，错峰：0.15 + 0.60 = 0.75 USD
    cost = estimate_cost("deepseek-chat", 1_000_000, 1_000_000, at=off_peak_at)
    check("错峰计价正确", abs(cost["usd"] - 0.75) < 1e-6, str(cost))
    check("高峰是错峰的 2 倍", abs(cost["multiplier"] - 1.0) < 1e-9, str(cost))

    peak_cost = estimate_cost("deepseek-chat", 1_000_000, 1_000_000, at=peak_at)
    check("高峰计价翻倍", abs(peak_cost["usd"] - 1.50) < 1e-6, str(peak_cost))
    check("高峰标记正确", peak_cost["peak"] is True)
    check("汇率换算正确",
          abs(cost["cny"] - cost["usd"] * config.USD_TO_CNY) < 1e-6, str(cost))

    # 缓存命中：1M 输入全部命中 → 0.003；未命中部分按 0.15
    hit_cost = estimate_cost("deepseek-chat", 1_000_000, 0, cache_hit_tokens=1_000_000,
                             at=off_peak_at)
    check("缓存命中按低价计", abs(hit_cost["usd"] - 0.003) < 1e-6, str(hit_cost))
    partial = estimate_cost("deepseek-chat", 1_000_000, 0, cache_hit_tokens=500_000,
                            at=off_peak_at)
    check("部分命中按比例计", abs(partial["usd"] - (0.5 * 0.15 + 0.5 * 0.003)) < 1e-6,
          str(partial))

    # 缓存命中数超过输入数时不能算成负成本
    weird = estimate_cost("deepseek-chat", 100, 0, cache_hit_tokens=999, at=off_peak_at)
    check("缓存命中数越界不产生负成本", weird["usd"] >= 0, str(weird))


# ========== 2. Span 树 ==========
def test_span_tree() -> None:
    print("\n[2] Span 树与耗时")
    from observability import Tracer, use_tracer

    tracer = Tracer(name="test")
    with use_tracer(tracer):
        with tracer.span("outer", kind="span") as outer:
            time.sleep(0.02)
            with tracer.span("inner", kind="tool") as inner:
                time.sleep(0.02)
            tracer.record_generation("llm.call", "deepseek-chat", "提示词", "输出",
                                     usage={"prompt_tokens": 100, "completion_tokens": 50,
                                            "cache_hit_tokens": 0, "total_tokens": 150},
                                     duration=0.01)

    trace = tracer.finish()
    names = [s["name"] for s in trace["spans"]]
    check("三个 span 都被记录", set(names) == {"outer", "inner", "llm.call"}, str(names))

    outer_span = next(s for s in trace["spans"] if s["name"] == "outer")
    inner_span = next(s for s in trace["spans"] if s["name"] == "inner")
    gen_span = next(s for s in trace["spans"] if s["name"] == "llm.call")

    check("inner 的父是 outer", inner_span["parent"] == outer_span["span_id"],
          f"{inner_span['parent']} vs {outer_span['span_id']}")
    check("llm.call 的父也是 outer", gen_span["parent"] == outer_span["span_id"])
    check("outer 没有父", outer_span["parent"] == "")
    check("span 有真实耗时", outer_span["duration"] >= 0.03, str(outer_span["duration"]))
    check("generation 被识别为 generation 类型", gen_span["kind"] == "generation")

    check("LLM 调用次数统计正确", trace["llm_calls"] == 1, str(trace["llm_calls"]))
    check("Token 汇总正确", trace["total_tokens"] == 150, str(trace["total_tokens"]))
    check("成本已计算", trace["cost_usd"] > 0, str(trace["cost_usd"]))
    check("总耗时已记录", trace["total_duration"] > 0, str(trace["total_duration"]))
    check("节点耗时按名字汇总", "outer" in trace["node_durations"], str(trace["node_durations"]))
    check("未启用 LangFuse 时标记正确", trace["langfuse_enabled"] is False)

    # 关闭追踪后不应记录任何东西
    disabled = Tracer(name="off", enabled=False)
    with use_tracer(disabled):
        with disabled.span("should-not-appear"):
            pass
    check("关闭追踪后不记录 span", disabled.finish()["spans"] == [])


# ========== 3. 并发上下文继承（真实 OTEL 验证）==========
def test_context_inheritance_with_otel() -> None:
    """
    这是本 Phase 最关键的一条验证。

    LangFuse v4 基于 OpenTelemetry。子线程不会自动继承调用线程的 contextvars，
    所以并发 LLM 调用的 span 有可能"掉到 trace 外面"。这里用**真实的内存导出器**
    验证：子线程里创建的 OTEL span，父节点确实是调用线程的 span。
    """
    print("\n[3] 并发上下文继承（真实 OpenTelemetry 验证）")
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError as exc:
        check("opentelemetry 可用（LangFuse 依赖）", False, str(exc))
        return

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_tracer = provider.get_tracer("test")

    from observability import inherit_context

    def worker(task_id):
        with otel_tracer.start_as_current_span(f"worker-{task_id}"):
            time.sleep(0.01)
        return task_id

    # 父 span 在调用线程里开着，子线程应该挂在它下面
    with otel_tracer.start_as_current_span("parent") as parent:
        parent_context = parent.get_span_context()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(inherit_context(worker), range(4)))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    check("父 span 与 4 个子 span 都被导出", len(spans) == 5, str(sorted(spans)))

    workers_linked = []
    for index in range(4):
        span = spans.get(f"worker-{index}")
        if span is None:
            continue
        workers_linked.append(
            span.parent is not None
            and span.parent.span_id == parent_context.span_id
        )
    check("并发子 span 的父节点都是调用线程的父 span",
          len(workers_linked) == 4 and all(workers_linked),
          f"4 个子 span 中只有 {sum(workers_linked)} 个正确挂载")

    # 不继承上下文时应当挂不上（证明 inherit_context 确实起了作用）
    exporter.clear()
    with otel_tracer.start_as_current_span("parent2"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, range(2)))
    orphaned = [s for s in exporter.get_finished_spans() if s.name.startswith("worker-")]
    check("不继承上下文时子 span 会脱离父节点（反证）",
          all(s.parent is None for s in orphaned),
          "没有复现出脱离现象，说明这个测试没能证明 inherit_context 的必要性")

    # 同时验证本模块自己的 span 栈也能跨线程继承
    from observability import Tracer, use_tracer

    tracer = Tracer(name="concurrent")
    with use_tracer(tracer):
        with tracer.span("extractor", kind="span"):
            def inner_worker(index):
                from observability import current_tracer

                active = current_tracer()
                if active is None:
                    return None
                own = active.current_span()
                return own.span_id if own else None

            with ThreadPoolExecutor(max_workers=4) as pool:
                parent_ids = list(pool.map(inherit_context(inner_worker), range(4)))

    trace = tracer.finish()
    extractor_span = next(s for s in trace["spans"] if s["name"] == "extractor")
    check("本模块的 span 栈也能跨线程继承",
          parent_ids and all(pid == extractor_span["span_id"] for pid in parent_ids),
          f"子线程拿到的父 span：{parent_ids}，期望 {extractor_span['span_id']}")


# ========== 4. LangFuse 接入（假客户端）==========
class FakeObservation:
    """模拟 LangFuse 的 observation 上下文管理器"""

    def __init__(self, client, name, as_type, **kwargs):
        self.client = client
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.updates = []
        self.exited = False

    def __enter__(self):
        self.client.active.append(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        if self.client.fail_on_exit:
            raise RuntimeError("模拟 LangFuse 退出失败")
        if self.client.active and self.client.active[-1] == self.name:
            self.client.active.pop()
        return False


class FakeLangfuse:
    """记录调用形状的假 LangFuse 客户端，用于验证接入代码发出的结构是否正确"""

    def __init__(self, fail_on_exit: bool = False):
        self.observations = []
        self.observation_objects = []
        self.updates = []
        self.events = []
        self.active = []
        self.flushed = 0
        self.fail_on_exit = fail_on_exit

    def start_as_current_observation(self, name, as_type="span", **kwargs):
        obs = FakeObservation(self, name, as_type, **kwargs)
        self.observations.append({"name": name, "as_type": as_type, "depth": len(self.active)})
        self.observation_objects.append(obs)
        return obs

    def update_current_span(self, **kwargs):
        self.updates.append(("span", kwargs))

    def update_current_generation(self, **kwargs):
        self.updates.append(("generation", kwargs))

    def create_event(self, **kwargs):
        self.events.append(kwargs)

    def flush(self):
        self.flushed += 1


def test_langfuse_integration() -> None:
    print("\n[4] LangFuse 接入（假客户端）")
    import config
    from observability import Tracer, langfuse_enabled, use_tracer

    check("未配置 Key 时 langfuse_enabled 为假", langfuse_enabled() is False
          or bool(config.LANGFUSE_PUBLIC_KEY), f"未预期的配置状态：{config.LANGFUSE_PUBLIC_KEY!r}")

    fake = FakeLangfuse()
    tracer = Tracer(name="with-langfuse", langfuse_client=fake)
    with use_tracer(tracer):
        with tracer.span("planner", kind="span"):
            with tracer.span("llm.call", kind="generation") as span:
                span.model = "deepseek-chat"
                span.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                span.cost = {"usd": 0.001}
            tracer.record_generation("并发调用", "deepseek-chat", "p", "o",
                                     usage={"prompt_tokens": 1, "completion_tokens": 1,
                                            "total_tokens": 2}, duration=0.01)

    trace = tracer.finish()

    names = [(o["name"], o["as_type"]) for o in fake.observations]
    check("为每个 span 开了 observation", len(fake.observations) == 2, str(names))
    check("generation 类型被正确传递",
          ("llm.call", "generation") in names, str(names))
    check("嵌套深度正确（planner 在外、llm.call 在内）",
          fake.observations[0]["depth"] == 0 and fake.observations[1]["depth"] == 1,
          str([(o["name"], o["depth"]) for o in fake.observations]))
    check("所有 observation 都被正常关闭",
          len(fake.observation_objects) == 2
          and all(o.exited for o in fake.observation_objects),
          str([(o.name, o.exited) for o in fake.observation_objects]))
    check("更新了 generation 的模型与用量",
          any(kind == "generation" and kw.get("model") == "deepseek-chat"
              for kind, kw in fake.updates), str([k for k, _ in fake.updates]))
    check("更新内容里带上了真实耗时",
          any("duration_seconds" in (kw.get("metadata") or {}) for _, kw in fake.updates),
          str([kw.get("metadata") for _, kw in fake.updates]))
    check("并发调用用事件方式补记（避免上下文错乱）",
          len(fake.events) == 1 and "cost_usd" in fake.events[0]["metadata"],
          str(fake.events))
    check("结束时 flush 了一次", fake.flushed == 1, str(fake.flushed))
    check("trace 标记 LangFuse 已启用", trace["langfuse_enabled"] is True)
    check("上报无异常", not trace["langfuse_error"], trace["langfuse_error"])


def test_langfuse_failure_tolerance() -> None:
    print("\n[5] 观测失败不能拖垮主流程")
    from observability import Tracer, use_tracer

    fake = FakeLangfuse(fail_on_exit=True)
    tracer = Tracer(name="broken", langfuse_client=fake)

    completed = False
    with use_tracer(tracer):
        with tracer.span("step"):
            completed = True

    trace = tracer.finish()
    check("LangFuse 退出抛异常时主流程仍跑完", completed is True)
    check("异常被记录而不是抛出", bool(trace["langfuse_error"]), trace["langfuse_error"])
    check("本地 trace 仍然完整", len(trace["spans"]) == 1, str(len(trace["spans"])))

    # 客户端本身抛异常（例如网络不可用）也不能影响主流程
    class ExplodingClient:
        def start_as_current_observation(self, **kwargs):
            raise ConnectionError("模拟网络不可用")

        def flush(self):
            raise ConnectionError("模拟网络不可用")

    tracer2 = Tracer(name="explode", langfuse_client=ExplodingClient())
    ok = False
    with use_tracer(tracer2):
        with tracer2.span("step"):
            ok = True
    trace2 = tracer2.finish()
    check("客户端创建 span 失败时主流程仍跑完", ok is True)
    check("失败原因被记录（且没被 flush 异常覆盖）",
          "ConnectionError" in trace2["langfuse_error"]
          and "模拟网络不可用" in trace2["langfuse_error"],
          trace2["langfuse_error"])
    check("本地 trace 仍然完整", len(trace2["spans"]) == 1)


# ========== 6. 与 LLM 层、端到端串联 ==========
def test_end_to_end_tracing() -> None:
    print("\n[6] 端到端：追踪真的挂上了")
    import config
    from deep_research import DeepResearchAgent, summarize
    from offline_stubs import OfflineLLM, offline_search

    with offline_search():
        agent = DeepResearchAgent(llm=OfflineLLM(reflect_rounds=0), max_iterations=1)
        final = agent.run("离线自检：追踪是否覆盖了所有节点？")

    trace = final.get("trace") or {}
    check("最终状态里带 trace", bool(trace), "trace 为空")
    check("trace 有 span 列表", len(trace.get("spans") or []) > 0,
          str(len(trace.get("spans") or [])))

    names = {s["name"] for s in trace.get("spans", [])}
    for node in ("planner", "searcher", "extractor", "reflector", "writer"):
        check(f"覆盖节点 {node}", node in names, f"实际 span：{sorted(names)}")

    kinds = {s["kind"] for s in trace.get("spans", [])}
    check("含 generation 类型（LLM 调用）", "generation" in kinds, str(kinds))

    node_spans = [s for s in trace["spans"] if s["kind"] == "span"]
    root_spans = [s for s in node_spans if not s["parent"]]
    check("只有一个根 span（整次研究一个 trace）", len(root_spans) == 1,
          f"实际 {len(root_spans)} 个根 span：{[s['name'] for s in root_spans]}"
          "（多个根 span 会在 LangFuse 里裂成多条 trace）")
    check("根 span 叫 research", root_spans and root_spans[0]["name"] == "research",
          str([s["name"] for s in root_spans]))

    root_id = root_spans[0]["span_id"] if root_spans else ""
    non_root_nodes = [s for s in node_spans if s["parent"]]
    check("其余节点都是根 span 的直接子节点",
          all(s["parent"] == root_id for s in non_root_nodes),
          str([(s["name"], s["parent"]) for s in non_root_nodes]))

    metrics = summarize(final)
    check("指标里含成本", "成本(CNY)" in metrics, str(list(metrics.keys())))
    stats = final.get("stats") or {}
    check("stats 里含 cost_usd", "cost_usd" in stats, str(list(stats.keys())))
    check("离线假模型成本为 0", stats.get("cost_usd", 1) == 0.0, str(stats.get("cost_usd")))
    check("trace 与 stats 的 Token 一致",
          stats.get("llm_usage", {}).get("total_tokens", 0) == trace.get("total_tokens", 0),
          f"stats={stats.get('llm_usage', {}).get('total_tokens')} trace={trace.get('total_tokens')}")


def main() -> int:
    print("=" * 60)
    print("Phase 7 自检：可观测性与成本核算")
    print("=" * 60)
    test_cost()
    test_span_tree()
    test_context_inheritance_with_otel()
    test_langfuse_integration()
    test_langfuse_failure_tolerance()
    test_end_to_end_tracing()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
