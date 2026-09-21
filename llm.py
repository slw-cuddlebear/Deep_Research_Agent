"""
大模型封装：DeepSeek（兼容 OpenAI 接口）的创建、JSON 调用与 Token 统计。
节点里统一通过 get_llm() 取模型。

**LLM 注入有两层，务必区分**（这里踩过一个严重 Bug）：
  - `use_llm(x)`：**作用域内**的临时覆盖（contextvar），退出即失效。给指定这次研究用假模型就该用它。
  - `set_llm(x)`：改写**进程级**全局默认值。只适合测试夹具，绝不能在跑一次研究的路径上调用。

历史教训：早期 `DeepResearchAgent.__init__` 直接调用 `set_llm(llm)` 注入假模型，
而 Streamlit 进程是常驻的——用户先跑了一次「离线演示」，之后切到「真实运行」并不会重置这个全局，
于是**真搜索 + 假模型**混着跑：计划是假模型编的、报告是假模型的占位文本、引用却来自真实搜索结果。
"""
import contextvars
import json
import re
import threading
import time
from contextlib import contextmanager

from tenacity import retry, stop_after_attempt, wait_exponential

import config

# ========== 进程内单例与用量统计 ==========
_lock = threading.Lock()
_llm = None
_usage = {
    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
    "cache_hit_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "cost_cny": 0.0,
}

# 作用域级的 LLM 覆盖（跨线程由 observability.inherit_context 一并继承）
_llm_override: contextvars.ContextVar = contextvars.ContextVar("llm_override", default=None)


def get_llm():
    """取当前生效的 LLM：优先作用域覆盖，其次进程级默认（懒加载真实模型）"""
    override = _llm_override.get()
    if override is not None:
        return override

    global _llm
    with _lock:
        if _llm is None:
            from langchain_openai import ChatOpenAI

            if not config.has_llm_key():
                raise RuntimeError(
                    "未检测到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入你的 Key，"
                    "申请地址：https://platform.deepseek.com"
                )
            _llm = ChatOpenAI(
                model=config.DEEPSEEK_MODEL,
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                temperature=config.TEMPERATURE,
                timeout=120,
                max_retries=2,
            )
        return _llm


@contextmanager
def use_llm(llm):
    """
    在块内使用指定 LLM；传 None 表示"回退到真实模型"。
    退出后自动恢复，不会影响进程里的其他任务。
    """
    token = _llm_override.set(llm)
    try:
        yield llm
    finally:
        _llm_override.reset(token)


def set_llm(llm) -> None:
    """
    设置**进程级默认** LLM。仅用于测试夹具。

    ⚠️ 不要在处理单次研究的代码路径里调用它——应该用 `use_llm()`。
    """
    global _llm
    with _lock:
        _llm = llm


def reset_llm() -> None:
    """清掉进程级默认 LLM，让下次 get_llm() 重新构建真实模型"""
    global _llm
    with _lock:
        _llm = None


def reset_usage() -> None:
    """重置 Token 统计"""
    with _lock:
        for k in _usage:
            _usage[k] = 0


def get_usage() -> dict:
    """取累计 Token 统计"""
    with _lock:
        return dict(_usage)


def _record_usage(response) -> dict:
    """从 LangChain 响应里累计 Token 用量，返回本次用量明细"""
    meta = getattr(response, "usage_metadata", None) or {}
    if not meta:
        meta = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})

    prompt_tokens = int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0)
    completion_tokens = int(meta.get("output_tokens") or meta.get("completion_tokens") or 0)
    # 缓存命中的输入 token 单价低得多，单独统计才能算准成本
    details = meta.get("input_token_details") or {}
    cache_hit = int(details.get("cache_read") or meta.get("prompt_cache_hit_tokens") or 0)

    from observability import estimate_cost

    cost = estimate_cost(config.DEEPSEEK_MODEL, prompt_tokens, completion_tokens, cache_hit)

    with _lock:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += prompt_tokens
        _usage["completion_tokens"] += completion_tokens
        _usage["cache_hit_tokens"] += cache_hit
        _usage["total_tokens"] += int(meta.get("total_tokens") or 0) or (
            prompt_tokens + completion_tokens
        )
        _usage["cost_usd"] += cost["usd"]
        _usage["cost_cny"] += cost["cny"]

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ========== 基础调用 ==========
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def _invoke(messages):
    return get_llm().invoke(messages)


def call_llm(prompt: str, system: str = None, llm=None, span_name: str = "llm.call") -> str:
    """调用 LLM 返回纯文本（同时记录一次 generation span）"""
    started = time.perf_counter()
    if llm is not None:
        response = llm.invoke(_build_messages(prompt, system))
    else:
        response = _invoke(_build_messages(prompt, system))
    elapsed = time.perf_counter() - started

    usage = _record_usage(response)
    text = _to_text(response)

    # 记进当前 tracer（没有 tracer 时是空操作，不影响主流程）
    try:
        from observability import current_tracer

        tracer = current_tracer()
        if tracer is not None:
            tracer.record_generation(
                span_name, config.DEEPSEEK_MODEL, prompt, text, usage=usage, duration=elapsed
            )
    except Exception:
        pass

    return text


def _build_messages(prompt: str, system: str = None):
    if system:
        return [("system", system), ("human", prompt)]
    return [("human", prompt)]


def _to_text(response) -> str:
    """兼容 str / AIMessage / 自定义假模型 三种返回形态"""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # 多模态分片
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


# ========== JSON 调用 ==========
def extract_json(text: str):
    """
    从 LLM 输出里稳健地抽出 JSON。
    依次尝试：原文解析 → 去代码围栏解析 → 括号配对截取解析。
    """
    if not text:
        raise ValueError("LLM 返回为空，无法解析 JSON")

    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates.extend([f.strip() for f in fenced])

    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        s = text.find(start_ch)
        e = text.rfind(end_ch)
        if s != -1 and e != -1 and e > s:
            candidates.append(text[s : e + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError(f"无法从 LLM 输出中解析出 JSON：{text[:300]}")


def call_llm_json(prompt: str, system: str = None, llm=None, retries: int = 2):
    """
    调用 LLM 并解析为 Python 对象（dict / list）。
    解析失败时把错误反馈给模型要求重写，最多重试 retries 次。
    """
    last_error = None
    current_prompt = prompt
    for attempt in range(retries + 1):
        text = call_llm(current_prompt, system=system, llm=llm)
        try:
            return extract_json(text)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            current_prompt = (
                f"{prompt}\n\n"
                f"【上一次输出无法解析】错误：{exc}\n"
                "请只输出合法的 JSON，不要任何解释文字，不要 Markdown 代码围栏。"
            )
    raise ValueError(f"LLM 连续 {retries + 1} 次未能返回合法 JSON：{last_error}")
