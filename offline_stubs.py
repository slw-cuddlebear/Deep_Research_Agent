"""
离线桩模块（Offline Stubs）：不联网、不消耗 API 额度也能跑通整张状态图。
用途：
  - `python main.py "问题" --offline` 离线演示与链路自检；
  - 单元测试注入假模型与假搜索。

设计：假模型按 Prompt 里的 JSON 结构标记分派响应，因此即使 Prompt 措辞改动，
只要输出格式约定不变，桩就不会失效。
"""
import json
import re
import threading

import search_tools

_URL_IN_TITLE_RE = re.compile(r"离线网页\s+(\S+)")
_PAGE_NO_RE = re.compile(r"page-(\d+)")
# 从写作 Prompt 里解析出"允许使用的引用编号"，让离线报告也能真实走一遍引用重编号
_ALLOWED_CITES_RE = re.compile(r"只能从下面允许的编号里选：(.+)")

# 从规划 Prompt 里解析出研究问题，让离线假计划也能带上主语（而不是"背景是什么？"）
_QUESTION_RE = re.compile(r"【研究问题】\s*\n?\s*(.+?)\s*\n")


def _extract_question(prompt_text: str) -> str:
    """从 Prompt 中抽取研究问题，用于给假计划拼出带主语的子问题"""
    match = _QUESTION_RE.search(prompt_text or "")
    if not match:
        return ""
    question = match.group(1).strip()
    return question[:40]

# 用一批措辞差异明显的模板轮换，模拟"不同网页给出不同信息"，
# 避免假事实因为措辞雷同而被去重逻辑合理地合并掉。
_CLAIM_TEMPLATES = [
    "该来源给出的结论是：这条技术路线在成本控制上具备明显优势。",
    "该来源指出：截至 2024 年，相关核心指标已提升约 30%。",
    "该来源的调研显示：主流厂商已开始规模化部署该方案。",
    "该来源的研究认为：这一结论目前仍存在争议，需要更多实证支撑。",
    "该来源提到：监管与合规要求是影响落地的关键变量。",
    "该来源强调：工程实现的复杂度往往被显著低估。",
    "该来源对比后认为：不同规模场景下的最优选型并不相同。",
    "该来源的实测数据表明：延迟与吞吐之间存在明确的权衡关系。",
]


class OfflineLLM:
    """
    离线假模型：返回与真实 Prompt 约定一致的合法 JSON。
    - 反思节点第一次回答"信息不足"，第二次回答"信息充分"，用于验证反思循环能真正触发并收敛。
    """

    class _Response:
        def __init__(self, content: str):
            self.content = content
            self.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __init__(self, reflect_rounds: int = 1):
        """:param reflect_rounds: 前几轮反思回答"信息不足"，用于控制循环次数"""
        self.reflect_rounds = reflect_rounds
        self.reflect_calls = 0
        self.extract_calls = 0
        self._lock = threading.Lock()

    # ---------- 分派 ----------
    def invoke(self, messages):
        text = messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1])
        return self._Response(json.dumps(self._dispatch(text), ensure_ascii=False))

    def _dispatch(self, text: str) -> dict:
        if '"sub_questions"' in text:  # 模块 1：规划
            # 刻意产出**带主语**的合格子问题。早期版本返回"离线自检问题一：背景是什么？"
            # 这类无主语句子，既让离线演示看起来像坏的，也掩盖了真实存在的规划缺陷。
            subject = _extract_question(text) or "该研究主题"
            return {
                "plan_summary": "离线自检：先界定范围与现状 → 再比对关键维度 → 最后归纳风险与限制",
                "sub_questions": [
                    {"id": "Q1",
                     "question": f"{subject}——它的背景与当前发展现状是怎样的？",
                     "rationale": "先界定范围与现状",
                     "search_queries": [f"{subject} 背景 现状"]},
                    {"id": "Q2",
                     "question": f"{subject}——涉及哪些关键技术方案与对比维度？",
                     "rationale": "再比对关键维度",
                     "search_queries": [f"{subject} 技术方案 对比"]},
                    {"id": "Q3",
                     "question": f"{subject}——在实践中存在哪些风险与限制？",
                     "rationale": "最后归纳风险与限制",
                     "search_queries": [f"{subject} 风险 限制"]},
                ],
            }

        if '"facts"' in text:  # 模块 3：提炼
            with self._lock:
                self.extract_calls += 1
                call_no = self.extract_calls
            match = _URL_IN_TITLE_RE.search(text)
            tag = match.group(1) if match else f"page-{call_no}"
            page_match = _PAGE_NO_RE.search(tag)
            page_no = int(page_match.group(1)) if page_match else call_no
            first = _CLAIM_TEMPLATES[(page_no * 2) % len(_CLAIM_TEMPLATES)]
            second = _CLAIM_TEMPLATES[(page_no * 2 + 1) % len(_CLAIM_TEMPLATES)]
            return {
                "facts": [
                    {"claim": f"{first.rstrip('。')}（离线自检来源 {tag}）",
                     "quote": "离线自检", "confidence": 0.8},
                    {"claim": f"{second.rstrip('。')}（离线自检来源 {tag}）",
                     "quote": "离线自检", "confidence": 0.6},
                ]
            }

        if '"sufficient"' in text:  # 模块 4：反思
            with self._lock:
                self.reflect_calls += 1
                round_no = self.reflect_calls
            if round_no <= self.reflect_rounds:
                return {
                    "sufficient": False,
                    "reason": f"离线自检：第 {round_no} 轮判断信息不足，需要补充搜索",
                    "coverage": 0.5,
                    "gaps": ["缺少具体的量化数据", "缺少反面证据"],
                    "new_queries": [f"离线自检 补充证据{round_no}", f"离线自检 数据{round_no}"],
                }
            return {
                "sufficient": True,
                "reason": "离线自检：各子问题均已有事实支撑，信息充分",
                "coverage": 0.85,
                "gaps": [],
                "new_queries": [],
            }

        if '"queries"' in text:  # 查询扩展
            return {"queries": ["离线自检 扩展检索词"]}

        if '"faithfulness"' in text:  # Phase 8：LLM 裁判
            return {
                "faithfulness": 0.82,
                "relevance": 0.90,
                "coverage": 0.75,
                "structure": 0.88,
                "deductions": "离线自检扣分点：第 3 章存在少量无引用的概括性表述。",
                "strengths": "离线自检优点：章节结构清晰，结论明确标注了局限。",
            }

        if '"summary_points"' in text:  # 模块 5：大纲（Phase 3）
            return {
                "title": "离线自检研究报告",
                "summary_points": ["离线自检要点一", "离线自检要点二"],
                "sections": [
                    {"heading": "研究背景与现状", "focus": "交代问题背景",
                     "fact_ids": ["F1", "F2", "F3"]},
                    {"heading": "关键发现", "focus": "汇总核心事实",
                     "fact_ids": ["F4", "F5", "F6"]},
                ],
            }

        if '"conclusion"' in text:  # 模块 5：摘要与结论（Phase 3）
            cites = _allowed_cites(text)
            tags = "".join(f"[{c}]" for c in cites[:2])
            return {
                "summary": f"离线自检摘要：本报告用于验证状态图链路是否通畅，所有内容均为假数据。{tags}",
                "conclusion": f"离线自检结论：链路通畅，可进入真实运行。{tags[:3]}",
            }

        if '"content"' in text:  # 模块 5：章节正文（Phase 3）
            cites = _allowed_cites(text)
            tags = "".join(f"[{c}]" for c in cites[:2]) or "[1]"
            return {
                "content": (
                    "离线自检章节正文。此处用于验证写作节点的解析逻辑与引用重编号。"
                    f"第一条可用结论来自相关来源{tags}，可用于支撑本章论点。"
                )
            }

        return {"ok": True}


def _allowed_cites(prompt_text: str) -> list:
    """从写作 Prompt 中解析出允许使用的引用编号列表"""
    match = _ALLOWED_CITES_RE.search(prompt_text or "")
    if not match:
        return []
    return [int(n) for n in re.findall(r"\[(\d+)\]", match.group(1))]


class OfflineSearchBackend:
    """
    离线假搜索：每调用一次就生成全新的 URL，保证补充搜索轮次能拿到"新网页"，
    从而真正走完"反思 → 重新搜索 → 重新提炼"的循环。

    本地知识库：默认返回空（避免加载 embedding 模型拖慢测试）；
    `with_local=True` 时返回带 local:// 链接的假片段，用于验证混合检索链路。
    """

    def __init__(self, with_local: bool = False):
        self._lock = threading.Lock()
        self._counter = 0
        self.search_calls = 0
        self.local_calls = 0
        self.with_local = with_local

    def _next_url(self) -> str:
        with self._lock:
            self._counter += 1
            return f"https://offline.example.com/page-{self._counter}"

    def search_many(self, queries, max_results=None, engine=None, max_workers=None):
        with self._lock:
            self.search_calls += 1
        rows, per_query = [], {}
        for query in queries:
            per_query[query] = 1
            rows.append({
                "id": f"S{self._counter}", "query": query,
                "title": f"离线结果：{query}", "url": self._next_url(),
                "snippet": "离线自检摘要", "engine": "offline",
                "score": round(0.5 + 0.4 * ((self._counter % 5) / 5), 3),
                "sub_question_id": "",
            })
        return search_tools.dedupe_results(rows), per_query, []

    def search_local(self, query, top_k=None, sub_question_id=""):
        """假的本地知识库检索：返回自带正文的 local:// 片段"""
        with self._lock:
            self.local_calls += 1
            serial = self.local_calls
        if not self.with_local:
            return []
        return [{
            "id": f"L{serial}",
            "query": query,
            "title": f"离线本地文档.md（本地知识库 第 {serial} 段）",
            "url": f"local://离线本地文档.md#chunk-{serial}",
            "snippet": f"离线本地片段 {serial}：这是本地知识库提供的内部信息。",
            "content": f"离线本地片段 {serial}：这是本地知识库提供的内部信息。" * 6,
            "engine": "local",
            "score": 0.72,
            "sub_question_id": sub_question_id,
            "fetched": True,
        }]

    def fetch_pages(self, urls, **kwargs):
        return [{
            "url": url, "title": f"离线网页 {url}",
            "text": f"这是离线自检正文（{url}）。" * 30,
            "ok": True, "error": "", "chars": 600, "source": "web",
        } for url in urls]


class offline_search:
    """
    上下文管理器：把搜索 / 抓取 / 本地库检索临时替换成离线桩。
    用法：
        with offline_search():
            agent.run(question)
        with offline_search(with_local=True):   # 额外验证混合检索
            agent.run(question)
    """

    def __init__(self, backend: OfflineSearchBackend = None, with_local: bool = False):
        self.backend = backend or OfflineSearchBackend(with_local=with_local)

    def __enter__(self) -> OfflineSearchBackend:
        import local_kb

        self._orig_search = search_tools.search_many
        self._orig_fetch = search_tools.fetch_pages
        self._orig_local = local_kb.search_local
        search_tools.search_many = self.backend.search_many
        search_tools.fetch_pages = self.backend.fetch_pages
        local_kb.search_local = self.backend.search_local
        return self.backend

    def __exit__(self, exc_type, exc_val, exc_tb):
        import local_kb

        search_tools.search_many = self._orig_search
        search_tools.fetch_pages = self._orig_fetch
        local_kb.search_local = self._orig_local
        return False
