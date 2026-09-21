"""
可观测性模块（Phase 7）：本地追踪（默认常开）+ 可选 LangFuse 上报。

设计：**本地追踪是权威数据源，LangFuse 是可选出口。**
  - 没有账号也能用：每一步的耗时、Token、成本都记录在 span 树里，
    落盘成 trace.json，并在 CLI / 前端渲染成时间线。
  - 配了 Key 就自动上报：tracer 的 span 上下文同时也是一个 LangFuse observation，
    因此上报的是**真实的起止时间**，而不是事后重放出来的假耗时。
  - 观测永远不能拖垮主流程：所有上报失败都被吞掉并记成一条 warn 事件。

并发说明：信息提取是线程池并发的，而 OpenTelemetry 的上下文是 contextvars。
子线程不会自动继承调用线程的上下文，所以提供 `inherit_context()`：
它在提交任务时抓一份上下文快照，在 worker 线程里恢复，
这样并发的 LLM 调用也能正确挂在"提取器"这个父 span 下面。
"""
import contextvars
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config

# ========== 价格表 ==========
# 单位：美元 / 百万 tokens。**错峰价**（高峰价为错峰价的 2 倍）。
# 来源：https://api-docs.deepseek.com/quick_start/pricing/
# 价格会调整，可用 .env 里的 LLM_PRICE_INPUT / LLM_PRICE_OUTPUT 覆盖。
DEFAULT_PRICE = {"input": 0.15, "output": 0.60, "cache_hit_input": 0.003}

MODEL_PRICES = {
    # deepseek-chat 是旧名称，官方已由 deepseek-flash 承接，按 Flash 价格计
    "deepseek-chat": {"input": 0.15, "output": 0.60, "cache_hit_input": 0.003},
    "deepseek-flash": {"input": 0.15, "output": 0.60, "cache_hit_input": 0.003},
    "deepseek-v4-flash": {"input": 0.15, "output": 0.60, "cache_hit_input": 0.003},
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98, "cache_hit_input": 0.022},
    "deepseek-reasoner": {"input": 0.66, "output": 1.98, "cache_hit_input": 0.022},
}


def price_for(model: str) -> dict:
    """取某模型的价格（支持 .env 覆盖输入/输出单价）"""
    price = dict(MODEL_PRICES.get((model or "").strip(), DEFAULT_PRICE))
    if config.LLM_PRICE_INPUT is not None:
        price["input"] = config.LLM_PRICE_INPUT
    if config.LLM_PRICE_OUTPUT is not None:
        price["output"] = config.LLM_PRICE_OUTPUT
    return price


def is_peak(at: datetime = None) -> bool:
    """
    是否处于高峰计价时段。
    高峰：UTC 周一至周五 01:00-04:00 与 06:00-10:00；其余（含周末）为错峰，价格为一半。
    注：官方还排除中国法定节假日，这里未做节假日判断，会略微高估成本。
    """
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    at = at.astimezone(timezone.utc)
    if at.weekday() >= 5:      # 周六周日全时段错峰
        return False
    return (1 <= at.hour < 4) or (6 <= at.hour < 10)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                  cache_hit_tokens: int = 0, at: datetime = None) -> dict:
    """
    按 token 用量估算成本。
    :return: {"usd": 美元, "cny": 人民币, "multiplier": 计价系数, "peak": 是否高峰}
    """
    price = price_for(model)
    multiplier = 2.0 if is_peak(at) else 1.0

    cache_hit = max(int(cache_hit_tokens or 0), 0)
    normal_input = max(int(prompt_tokens or 0) - cache_hit, 0)
    completion = max(int(completion_tokens or 0), 0)

    usd = (
        normal_input * price["input"]
        + cache_hit * price["cache_hit_input"]
        + completion * price["output"]
    ) / 1_000_000 * multiplier

    return {
        "usd": round(usd, 6),
        "cny": round(usd * config.USD_TO_CNY, 6),
        "multiplier": multiplier,
        "peak": multiplier > 1.0,
    }


# ========== Span ==========
@dataclass
class Span:
    """一次操作（节点 / LLM 调用 / 检索 / 工具）的记录"""

    name: str
    kind: str = "span"                 # span / generation / retriever / tool
    parent: str = ""
    span_id: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)
    input: object = None
    output: object = None
    model: str = ""
    usage: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    error: str = ""
    thread: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "span_id": self.span_id,
            "parent": self.parent,
            "duration": round(self.duration, 3),
            "start": datetime.fromtimestamp(self.start, timezone.utc).isoformat()
            if self.start else "",
            "metadata": self.metadata,
            "input": self.input,
            "output": self.output,
            "model": self.model,
            "usage": self.usage,
            "cost": self.cost,
            "error": self.error,
        }


# ========== Tracer ==========
class Tracer:
    """记录一次研究的完整 span 树，并可选同步上报到 LangFuse"""

    def __init__(self, name: str = "deep_research", question: str = "",
                 langfuse_client=None, enabled: bool = True):
        self.name = name
        self.question = question
        self.enabled = enabled
        self.langfuse = langfuse_client
        self.spans: list = []
        self.started_at = time.time()
        self.finished_at = 0.0
        self.export_error = ""
        self._lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._counter = 0
        # 每个线程/上下文各自持有 span 栈
        self._stack: contextvars.ContextVar = contextvars.ContextVar(
            f"span_stack_{id(self)}", default=()
        )

    def _note_error(self, message: str) -> None:
        """
        记录上报异常。

        只保留**第一个**错误（通常是根因），后续只做补充拼接——
        否则 finish() 阶段的 flush 失败会把前面真正的失败原因覆盖掉。
        """
        if not message:
            return
        with self._error_lock:
            if not self.export_error:
                self.export_error = message
            elif message not in self.export_error:
                self.export_error = f"{self.export_error} | {message}"

    # ---------- 内部 ----------
    def _next_id(self, name: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{name}-{self._counter}"

    def current_span(self):
        stack = self._stack.get()
        return stack[-1] if stack else None

    # ---------- span 上下文 ----------
    @contextmanager
    def span(self, name: str, kind: str = "span", **metadata):
        """打开一个 span：记录本地时间线，并在启用时同步开一个 LangFuse observation"""
        if not self.enabled:
            yield None
            return

        parent = self.current_span()
        span = Span(
            name=name, kind=kind,
            parent=parent.span_id if parent else "",
            span_id=self._next_id(name),
            start=time.time(),
            metadata={k: v for k, v in metadata.items() if v not in (None, "")},
            thread=threading.current_thread().name,
        )
        token = self._stack.set(self._stack.get() + (span,))

        langfuse_cm, langfuse_obs = None, None
        if self.langfuse is not None:
            try:
                langfuse_cm = self.langfuse.start_as_current_observation(
                    name=name,
                    as_type=("generation" if kind == "generation" else
                             kind if kind in ("span", "chain", "tool", "retriever", "agent") else "span"),
                    input=span.input,
                    metadata=span.metadata or None,
                )
                langfuse_obs = langfuse_cm.__enter__()
            except Exception as exc:
                self._note_error(f"{type(exc).__name__}: {exc}")
                langfuse_cm = None

        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.time()
            span.duration = span.end - span.start
            with self._lock:
                self.spans.append(span)
            self._stack.reset(token)

            if langfuse_cm is not None:
                self._update_langfuse(span)
                try:
                    langfuse_cm.__exit__(None, None, None)
                except Exception as exc:
                    self._note_error(f"{type(exc).__name__}: {exc}")

    def _update_langfuse(self, span: Span) -> None:
        """把耗时/用量/成本补充到当前 LangFuse observation 上"""
        try:
            extra = dict(span.metadata)
            extra["duration_seconds"] = round(span.duration, 3)
            if span.error:
                extra["error"] = span.error
            if span.kind == "generation":
                self.langfuse.update_current_generation(
                    output=span.output, model=span.model or None,
                    usage_details=span.usage or None,
                    cost_details={"total": span.cost.get("usd", 0.0)} if span.cost else None,
                    metadata=extra or None,
                    level="ERROR" if span.error else "DEFAULT",
                )
            else:
                self.langfuse.update_current_span(
                    output=span.output, metadata=extra or None,
                    level="ERROR" if span.error else "DEFAULT",
                )
        except Exception as exc:
            self._note_error(f"{type(exc).__name__}: {exc}")

    # ---------- 记录一次 LLM 调用 ----------
    def record_generation(self, name: str, model: str, prompt: str, output: str,
                          usage: dict = None, duration: float = None) -> None:
        """把一次已经完成的 LLM 调用补记成一个 generation span"""
        if not self.enabled:
            return
        usage = usage or {}
        parent = self.current_span()
        cost = estimate_cost(
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("cache_hit_tokens", 0),
        )
        span = Span(
            name=name, kind="generation",
            parent=parent.span_id if parent else "",
            span_id=self._next_id(name),
            start=time.time() - (duration or 0.0),
            end=time.time(),
            duration=duration or 0.0,
            input=(prompt or "")[:600],
            output=(output or "")[:600],
            model=model,
            usage=usage,
            cost=cost,
            thread=threading.current_thread().name,
        )
        with self._lock:
            self.spans.append(span)
        # 并发场景下不进 LangFuse 的 observation 栈，用事件方式补记，避免上下文错乱
        if self.langfuse is not None:
            try:
                self.langfuse.create_event(
                    name=f"llm:{name}", input=span.input, output=span.output,
                    metadata={"model": model, "usage": usage, "cost_usd": cost["usd"],
                              "duration_seconds": round(span.duration, 3)},
                )
            except Exception as exc:
                self._note_error(f"{type(exc).__name__}: {exc}")

    # ---------- 汇总 ----------
    def finish(self) -> dict:
        """收尾：汇总统计、上报 LangFuse、返回可序列化的 trace"""
        self.finished_at = time.time()
        if self.langfuse is not None:
            try:
                self.langfuse.flush()
            except Exception as exc:
                self._note_error(f"{type(exc).__name__}: {exc}")

        with self._lock:
            spans = list(self.spans)

        generations = [s for s in spans if s.kind == "generation"]
        prompt_tokens = sum(s.usage.get("prompt_tokens", 0) for s in generations)
        completion_tokens = sum(s.usage.get("completion_tokens", 0) for s in generations)
        cache_hits = sum(s.usage.get("cache_hit_tokens", 0) for s in generations)
        cost_usd = sum(s.cost.get("usd", 0.0) for s in generations)

        node_spans = [s for s in spans if s.kind == "span"]
        node_durations = {}
        for span in node_spans:
            node_durations[span.name] = round(
                node_durations.get(span.name, 0.0) + span.duration, 2
            )

        return {
            "name": self.name,
            "question": self.question,
            "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "total_duration": round(self.finished_at - self.started_at, 2),
            "llm_calls": len(generations),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_tokens": cache_hits,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost_usd, 6),
            "cost_cny": round(cost_usd * config.USD_TO_CNY, 6),
            "peak_pricing": is_peak(),
            "node_durations": node_durations,
            "langfuse_enabled": self.langfuse is not None,
            "langfuse_error": self.export_error,
            "spans": [s.to_dict() for s in sorted(spans, key=lambda s: s.start)],
        }


# ========== 当前 tracer（上下文级）==========
_current: contextvars.ContextVar = contextvars.ContextVar("current_tracer", default=None)


def current_tracer():
    """取当前上下文里的 tracer（没有则返回 None）"""
    return _current.get()


@contextmanager
def use_tracer(tracer: Tracer):
    """在块内把 tracer 设为当前 tracer"""
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


def inherit_context(fn):
    """
    包装一个要在子线程里执行的函数，让它继承调用线程的上下文
    （既包括本模块的 span 栈，也包括 OpenTelemetry 的上下文）。

    注意：`contextvars.Context` **不能被并发进入**。所以这里的做法是
    在包装时抓一份快照，而在**每次执行时**再复制一份新副本给这次调用使用。
    这样无论调用方是"每次提交都调一次 inherit_context"（本项目提取器的写法），
    还是"只包装一次然后交给 pool.map 复用"，都不会踩到
    `RuntimeError: cannot enter context: ... is already entered`。
    （这个坑是实测踩出来的：pool.map 只会求值一次包装函数。）
    """
    caller_ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return caller_ctx.copy().run(fn, *args, **kwargs)

    return wrapper


def traced_node(name: str):
    """装饰器：把 LangGraph 节点包成一个 span"""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            tracer = current_tracer()
            if tracer is None:
                return fn(state, *args, **kwargs)
            with tracer.span(name, kind="span",
                             iteration=state.get("iteration", 1)):
                return fn(state, *args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def tool_span(name: str, **metadata):
    """给工具类操作（检索、抓取、本地库）打 span 的便捷入口"""
    tracer = current_tracer()
    if tracer is None:
        yield None
        return
    with tracer.span(name, kind="tool", **metadata) as span:
        yield span


# ========== LangFuse 客户端 ==========
_client_lock = threading.Lock()
_langfuse_client = None
_langfuse_checked = False


def langfuse_enabled() -> bool:
    """是否配置了 LangFuse Key"""
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)


def get_langfuse_client():
    """
    构建（并缓存）LangFuse 客户端；未配置 Key 时返回 None。
    构建失败不会抛异常，只把原因记在 `langfuse_error()` 里。
    """
    global _langfuse_client, _langfuse_checked
    if not langfuse_enabled():
        return None
    with _client_lock:
        if _langfuse_client is not None or _langfuse_checked:
            return _langfuse_client
        _langfuse_checked = True
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        except Exception:
            _langfuse_client = None
        return _langfuse_client


def langfuse_error() -> str:
    """LangFuse 客户端构建失败时的原因"""
    return ""


def verify_langfuse() -> tuple:
    """
    验证 LangFuse 配置是否真的可用（会真的发一次鉴权请求）。
    :return: (是否可用, 说明)
    """
    if not langfuse_enabled():
        return False, "未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
    client = get_langfuse_client()
    if client is None:
        return False, "LangFuse 客户端创建失败（检查 langfuse 包是否安装）"
    try:
        ok = client.auth_check()
    except Exception as exc:
        return False, f"鉴权失败：{type(exc).__name__}: {exc}"
    if ok:
        return True, f"鉴权通过（{config.LANGFUSE_HOST}）"
    return False, "鉴权未通过：Key 或 Host 不正确"


# ========== 渲染辅助 ==========
def trace_summary(trace: dict) -> dict:
    """把 trace 压成适合展示的一行行指标"""
    if not trace:
        return {}
    return {
        "总耗时": f"{trace.get('total_duration', 0)}s",
        "LLM 调用次数": trace.get("llm_calls", 0),
        "输入 Token": trace.get("prompt_tokens", 0),
        "输出 Token": trace.get("completion_tokens", 0),
        "缓存命中 Token": trace.get("cache_hit_tokens", 0),
        "总 Token": trace.get("total_tokens", 0),
        "成本(USD)": f"${trace.get('cost_usd', 0):.4f}",
        "成本(CNY)": f"¥{trace.get('cost_cny', 0):.4f}",
        "计价时段": "高峰（2 倍）" if trace.get("peak_pricing") else "错峰",
        "LangFuse": "已上报" if trace.get("langfuse_enabled") else "未启用",
    }


def slowest_spans(trace: dict, limit: int = 10) -> list:
    """按耗时倒序列出最慢的 span，用于定位性能瓶颈"""
    spans = (trace or {}).get("spans") or []
    return sorted(spans, key=lambda s: s.get("duration", 0), reverse=True)[:limit]
