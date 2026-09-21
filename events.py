"""
事件总线：把 Agent 的每一步动作实时广播出去。
用途：
  - CLI 实时打印进度；
  - Streamlit 前端实时渲染思考过程（Phase 6）；
  - 同时把事件写回状态的 events 字段，便于事后复盘与归档。
"""
import threading
import time

from state import AgentEvent, to_dicts


class EventBus:
    """极简发布-订阅总线，支持多订阅者与线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []
        self._history = []

    def subscribe(self, callback):
        """注册订阅者，返回取消订阅的函数"""
        with self._lock:
            self._subscribers.append(callback)
        return lambda: self.unsubscribe(callback)

    def unsubscribe(self, callback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def clear_subscribers(self) -> None:
        with self._lock:
            self._subscribers.clear()

    def publish(self, event) -> None:
        """发布事件（dict 或 AgentEvent）"""
        if isinstance(event, AgentEvent):
            event = event.model_dump()
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # 订阅者异常不能影响主流程
                pass

    def history(self) -> list:
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


bus = EventBus()


def log_event(state: dict, step: str, title: str, detail: str = "",
              status: str = "done", elapsed: float = 0.0, **data) -> dict:
    """
    记录一个过程事件：既广播给订阅者，又返回给节点写回状态。
    节点用法： return {"events": [log_event(state, "planner", "拆解完成")], ...}
    """
    event = AgentEvent(
        step=step,
        title=title,
        detail=detail,
        status=status,
        elapsed=round(elapsed, 3),
        data=data,
    ).model_dump()
    bus.publish(event)
    return event


class Timer:
    """上下文管理器，用于统计单步耗时"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        return False


def get_events(state: dict) -> list:
    """从状态里读取事件列表"""
    return to_dicts(state.get("events") or [])
