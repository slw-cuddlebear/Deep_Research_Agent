"""
搜索工具模块：网页搜索（Tavily 优先 + DuckDuckGo 免费兜底）、正文抓取、去重与可信度打分。
对外只暴露 search_web / fetch_pages 两个主函数，上层节点不关心具体引擎。
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import config
from state import PageContent, SearchResult, new_id

# ========== 来源可信度规则 ==========
# 规则刻意保持简单可解释（不引入额外模型）：官方/学术/权威媒体 0.9，行业媒体与专业社区 0.6，
# 聚合站与 UGC 0.4，其余 0.5。中文权威站点单列，否则新华社、人民网这类会被误判成 0.5。
HIGH_TRUST_DOMAINS = (
    # 政府 / 教育 / 国际组织
    ".gov.cn", ".edu.cn", ".gov", ".edu", ".org.cn", ".ac.cn",
    "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org",
    # 学术与出版
    "arxiv.org", "nature.com", "science.org", "acm.org", "ieee.org", "springer.com",
    "sciencedirect.com", "aclanthology.org", "openreview.net", "nih.gov", "doi.org",
    # 官方 / 权威中文媒体与通讯社
    "news.cn", "xinhuanet.com", "people.com.cn", "cctv.com", "chinadaily.com.cn",
    "gmw.cn", "qstheory.cn", "ce.cn", "chinanews.com.cn", "stcn.com", "yicai.com",
    "caixin.com", "thepaper.cn", "bjnews.com.cn", "21jingji.com",
    # 机构与厂商官方
    "wikipedia.org", "openai.com", "deepmind.google", "anthropic.com", "microsoft.com",
    "google.com", "github.com", "deepseek.com", "aliyun.com", "huawei.com",
    "hikunpeng.com", "nvidia.com", "intel.com", "amd.com",
    # 权威研究机构
    "idc.com", "gartner.com", "mckinsey.com", "statista.com", "iresearch.com.cn",
    "cnnic.net.cn", "caict.ac.cn",
)
MEDIUM_TRUST_DOMAINS = (
    # 科技/财经行业媒体
    "36kr.com", "tmtpost.com", "cls.cn", "zhidx.com", "wallstreetcn.com", "infoq.cn",
    "oschina.net", "sspai.com", "leiphone.com", "jiqizhixin.com",
    "chyxx.com", "iimedia.cn", "fxbaogao.com", "qianzhan.com", "askci.com",
    # 技术社区与博客
    "zhihu.com", "juejin.cn", "csdn.net", "cnblogs.com", "segmentfault.com",
    "medium.com", "towardsdatascience.com", "substack.com", "dev.to",
)
LOW_TRUST_DOMAINS = ("baidu.com", "so.com", "toutiao.com", "weibo.com", "douban.com")


def score_credibility(url: str, title: str = "") -> float:
    """
    按域名给来源可信度打分（0~1），供事实排序与报告加权使用。
    规则简单可解释，避免引入额外模型。

    本地知识库（local://）走单独一档：它是自建/受控语料，
    对"内部事实"而言比任何网页都权威。
    """
    if not url:
        return 0.3
    if url.startswith("local://"):
        return config.LOCAL_CREDIBILITY
    domain = (urlparse(url).netloc or "").lower()
    if any(d in domain for d in HIGH_TRUST_DOMAINS):
        return 0.9
    if any(d in domain for d in MEDIUM_TRUST_DOMAINS):
        return 0.6
    if any(d in domain for d in LOW_TRUST_DOMAINS):
        return 0.4
    return 0.5


# ========== URL 规范化与去重 ==========
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "spm", "from", "share_token", "ref", "fbclid", "gclid"}


def normalize_url(url: str) -> str:
    """去掉追踪参数、锚点与末尾斜杠，用于结果去重"""
    if not url:
        return ""
    try:
        parts = urlparse(url.strip())
    except ValueError:
        return url.strip()
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() not in _TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    return urlunparse((parts.scheme.lower(), parts.netloc.lower(), path, "", urlencode(query), ""))


def result_key(url: str) -> str:
    """
    结果的唯一键。

    本地知识库片段必须**保留片段号**：`local://doc.md#chunk-3` 与 `#chunk-7` 是两段不同内容。
    若直接套用 normalize_url（它会丢掉 #fragment），同一文档的所有片段会被当成同一条结果，
    补充搜索轮次里新命中的片段会被误判为"已读过"而丢弃。
    """
    if not url:
        return ""
    if url.startswith("local://"):
        return url.strip()
    return normalize_url(url)


def dedupe_by_key(results: list) -> list:
    """按 result_key 去重（本地片段保留片段号）"""
    best = {}
    for item in results:
        data = item.model_dump() if isinstance(item, SearchResult) else dict(item)
        key = result_key(data.get("url", ""))
        if not key:
            continue
        if key not in best:
            best[key] = data
        elif len(data.get("snippet", "")) > len(best[key].get("snippet", "")):
            best[key] = data
    return list(best.values())


def dedupe_results(results: list) -> list:
    """
    按规范化 URL 去重；同一 URL 命中多次时保留信息更全的一条并累加命中次数。
    返回去重并按 (引擎打分 + 可信度) 排序后的结果。
    """
    best = {}
    for item in results:
        data = item.model_dump() if isinstance(item, SearchResult) else dict(item)
        key = normalize_url(data.get("url", ""))
        if not key:
            continue
        data["url"] = data.get("url", "")
        hit = best.get(key)
        if hit is None:
            data["hits"] = 1
            best[key] = data
        else:
            hit["hits"] = hit.get("hits", 1) + 1
            if len(data.get("snippet", "")) > len(hit.get("snippet", "")):
                hit["snippet"] = data["snippet"]
            if not hit.get("title") and data.get("title"):
                hit["title"] = data["title"]
            hit["score"] = max(hit.get("score", 0.0), data.get("score", 0.0))

    ordered = sorted(
        best.values(),
        key=lambda d: (d.get("score", 0.0) + score_credibility(d.get("url", "")) * 0.5
                       + min(d.get("hits", 1) - 1, 2) * 0.1),
        reverse=True,
    )
    return ordered


# ========== Tavily 搜索 ==========
def _search_tavily(query: str, max_results: int) -> list:
    """调用 Tavily API 搜索"""
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results, search_depth="basic")
    out = []
    for item in response.get("results", []):
        out.append(
            SearchResult(
                id=new_id("S"),
                query=query,
                title=(item.get("title") or "").strip(),
                url=item.get("url") or "",
                snippet=(item.get("content") or "").strip(),
                engine="tavily",
                score=float(item.get("score") or 0.0),
            ).model_dump()
        )
    return out


# ========== DuckDuckGo 搜索（免 Key 兜底）==========
def _ddgs_client():
    """兼容 ddgs（新包名）与 duckduckgo_search（旧包名）"""
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS

        return DDGS


def _search_ddg(query: str, max_results: int) -> list:
    """调用 DuckDuckGo 搜索"""
    DDGS = _ddgs_client()
    out = []
    with DDGS() as client:
        try:
            rows = client.text(query, region=config.DDG_REGION, max_results=max_results)
        except TypeError:  # 部分版本不支持 region 参数
            rows = client.text(query, max_results=max_results)
    for rank, item in enumerate(rows or []):
        out.append(
            SearchResult(
                id=new_id("S"),
                query=query,
                title=(item.get("title") or "").strip(),
                url=item.get("href") or item.get("url") or "",
                snippet=(item.get("body") or "").strip(),
                engine="ddg",
                # DDG 不返回分数，用排名折算一个递减分数
                score=round(max(0.2, 1.0 - rank * 0.12), 3),
            ).model_dump()
        )
    return out


# ========== 统一入口 ==========
def resolve_engine(engine: str = None) -> str:
    """决定用哪个引擎：显式指定 > 有 Tavily Key 用 Tavily > 否则 DuckDuckGo"""
    engine = (engine or config.SEARCH_ENGINE or "auto").lower()
    if engine != "auto":
        return engine
    return "tavily" if config.has_tavily_key() else "ddg"


# 没有可用搜索后端时的提示语（DuckDuckGo 在部分网络环境不可达）
NO_ENGINE_HINT = (
    "没有可用的网页搜索后端。\n"
    "  原因：未配置 TAVILY_API_KEY，而免费兜底的 DuckDuckGo 在当前网络不可达。\n"
    "  解决：到 https://tavily.com 注册免费账号（每月 1000 次检索额度），"
    "把 Key 填进 .env 的 TAVILY_API_KEY 后重试。\n"
    "  也可以先运行 `python main.py \"问题\" --offline` 做离线链路自检。"
)


def search_one(query: str, max_results: int = None, engine: str = None) -> list:
    """
    执行单条检索词，返回 SearchResult 列表。
    主引擎失败时自动降级到另一个引擎，保证链路不中断。
    """
    max_results = max_results or config.SEARCH_RESULTS_PER_QUERY
    engine = resolve_engine(engine)
    order = [engine] + [e for e in ("tavily", "ddg") if e != engine]

    last_error = None
    tried = []
    for name in order:
        if name == "tavily" and not config.has_tavily_key():
            continue
        tried.append(name)
        try:
            if name == "tavily":
                return _search_tavily(query, max_results)
            return _search_ddg(query, max_results)
        except Exception as exc:  # 网络异常 / 限流 / 缺依赖
            last_error = f"{name}: {exc}"
            time.sleep(1)

    if not tried:
        raise RuntimeError(NO_ENGINE_HINT)
    if "ddg" in tried and not config.has_tavily_key():
        raise RuntimeError(f"{NO_ENGINE_HINT}\n原始错误：{last_error}")
    raise RuntimeError(f"检索失败（已尝试 {'/'.join(tried)}）: {last_error}")


def search_many(queries: list, max_results: int = None, engine: str = None,
                max_workers: int = None) -> tuple:
    """
    并发执行多条检索词。
    返回 (去重后的结果列表, 每条 query 的原始条数统计, 错误信息列表)。
    """
    queries = [q.strip() for q in (queries or []) if q and q.strip()]
    if not queries:
        return [], {}, []

    max_workers = max_workers or config.MAX_CONCURRENCY
    raw, errors, per_query = [], [], {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(queries))) as pool:
        futures = {pool.submit(search_one, q, max_results, engine): q for q in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                rows = future.result()
                per_query[query] = len(rows)
                raw.extend(rows)
            except Exception as exc:
                per_query[query] = 0
                errors.append(f"[{query}] {exc}")

    return dedupe_results(raw), per_query, errors


# ========== 正文抓取 ==========
_BOILERPLATE_PATTERNS = (
    r"登录|注册|扫码|下载APP|关注我们|版权所有|免责声明|点击查看全文|上一篇|下一篇",
)

# 这些 Content-Type 一定不是"网页正文"，抓到只会得到二进制乱码
_BINARY_CONTENT_TYPES = (
    "application/pdf", "application/octet-stream", "application/zip",
    "application/x-zip", "application/x-rar", "application/msword",
    "application/vnd.openxmlformats", "application/vnd.ms-",
    "image/", "audio/", "video/", "font/",
)

# 明显是文件下载而不是网页的 URL（Content-Type 检查之外的廉价前置过滤）
_FILE_URL_RE = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|7z|gz|tar|exe|dmg|apk)(\?|#|$)", re.I
)

# 二进制被当作文本读取时会出现大量替换字符（U+FFFD）与控制字符
_BINARY_TEXT_THRESHOLD = 0.01   # 异常字符占比超过 1% 即判定为二进制


def looks_like_binary(text: str, threshold: float = _BINARY_TEXT_THRESHOLD) -> bool:
    """
    判断一段文本是不是"把二进制文件当文本读"的产物。

    实测：抓取期刊与政府网站的 PDF 时，requests 会把 PDF 字节按文本解码，
    得到长度上万、替换字符占比 45%~58% 的"正文"。若不拦截，这些乱码会一路流进
    事实提炼 → 报告 → PDF，表现为满屏乱码。
    """
    if not text:
        return False
    sample = text[:4000]
    bad = 0
    for ch in sample:
        code = ord(ch)
        if ch == "\ufffd" or (code < 32 and ch not in "\t\n\r"):
            bad += 1
    return bad / len(sample) > threshold


def _clean_text(text: str) -> str:
    """压缩空白、剔除替换字符与不可打印字符、去掉明显的模板噪声行"""
    if not text:
        return ""
    # 注意：U+FFFD（0xFFFD）本身 >= 32，只按码位过滤是拦不住它的，必须显式排除。
    text = "".join(
        ch for ch in text
        if ch != "\ufffd" and (ch in "\t\n\r" or ord(ch) >= 32)
    )
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) < 6 and not re.search(r"[\u4e00-\u9fa5]{2,}", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _fetch_with_trafilatura(url: str) -> tuple:
    """用 trafilatura 抓正文（效果最好）"""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "", ""
    if isinstance(downloaded, bytes) and downloaded[:4] == b"%PDF":
        raise ValueError("目标是 PDF 文件，不是网页")
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True, favor_recall=True
    ) or ""
    title = ""
    meta = trafilatura.extract_metadata(downloaded)
    if meta and getattr(meta, "title", None):
        title = meta.title
    return text, title


def _fetch_with_bs4(url: str) -> tuple:
    """trafilatura 失败时用 requests + BeautifulSoup 兜底"""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        url, timeout=config.FETCH_TIMEOUT,
        headers={"User-Agent": config.USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
    )
    response.raise_for_status()

    # 关键：先看 Content-Type。PDF / 压缩包 / 图片被当文本解码就是乱码的源头。
    content_type = (response.headers.get("Content-Type") or "").lower()
    if any(bad in content_type for bad in _BINARY_CONTENT_TYPES):
        raise ValueError(f"非网页内容（Content-Type: {content_type.split(';')[0]}）")

    # 有些站点不返回 Content-Type，用文件头兜底
    if response.content[:4] == b"%PDF":
        raise ValueError("目标是 PDF 文件，不是网页")

    if not response.encoding or response.encoding.lower() in ("iso-8859-1",):
        response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return main.get_text("\n", strip=True), title


def fetch_page(url: str, timeout: int = None) -> dict:
    """
    抓取单页正文，返回 PageContent 的 dict。
    两个抓取器依次尝试，都失败则返回 ok=False 的结果（不抛异常，避免拖垮整条链路）。
    """
    timeout = timeout or config.FETCH_TIMEOUT

    # 廉价前置过滤：明显是文件下载的 URL 直接跳过，省一次网络请求
    if _FILE_URL_RE.search(url or ""):
        return PageContent(url=url, ok=False, chars=0,
                           error="URL 指向文件下载而非网页，已跳过").model_dump()

    errors = []
    for fetcher in (_fetch_with_trafilatura, _fetch_with_bs4):
        try:
            text, title = fetcher(url)
            text = _clean_text(text or "")
            if len(text) < 200:
                errors.append(f"{fetcher.__name__}: 正文过短({len(text)})")
                continue
            # 关键：内容像二进制就丢弃，绝不让乱码进入提炼环节
            if looks_like_binary(text):
                errors.append(f"{fetcher.__name__}: 内容疑似二进制/编码错误，已丢弃")
                continue
            return PageContent(
                url=url, title=title.strip(), text=text[: config.MAX_PAGE_CHARS * 2],
                ok=True, chars=len(text),
            ).model_dump()
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {exc}")
    return PageContent(url=url, ok=False, error="; ".join(errors), chars=0).model_dump()


def fetch_pages(urls: list, max_workers: int = None, timeout: int = None,
                max_pages: int = None) -> list:
    """并发抓取多个 URL，保持输入顺序返回"""
    urls = [u for u in dict.fromkeys(urls or []) if u]
    if max_pages:
        urls = urls[:max_pages]
    if not urls:
        return []

    max_workers = max_workers or config.MAX_CONCURRENCY
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as pool:
        futures = {pool.submit(fetch_page, url, timeout): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                results[url] = PageContent(url=url, ok=False, error=str(exc)).model_dump()
    return [results[u] for u in urls if u in results]
