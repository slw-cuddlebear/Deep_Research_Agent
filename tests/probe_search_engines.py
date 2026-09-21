"""
搜索引擎可用性探测脚本。
不同网络环境（尤其是国内网络）下可访问的搜索引擎差别很大，本脚本逐个实测并打印结果。

运行： python tests/probe_search_engines.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

QUERY = "DeepSeek R1 评测"

TARGETS = [
    ("bing", "https://www.bing.com/search?q={q}&setlang=zh-CN",
     ["li.b_algo", "#b_results > li.b_algo", ".b_algo"]),
    ("sogou", "https://www.sogou.com/web?query={q}",
     ["div.vrwrap", "div.rb", ".results .vrwrap", ".vrwrap"]),
    ("so360", "https://www.so.com/s?q={q}",
     ["li.res-list", ".result", ".res-list"]),
    ("mojeek", "https://www.mojeek.com/search?q={q}",
     ["ul.results-standard li", "li.result"]),
    ("baidu", "https://www.baidu.com/s?wd={q}",
     ["div.result", "div.c-container", ".result.c-container"]),
    ("duckduckgo", "https://html.duckduckgo.com/html/?q={q}",
     ["div.result", "div.web-result"]),
]


def probe(name: str, url_template: str, selectors: list) -> bool:
    """探测单个引擎，成功返回 True"""
    url = url_template.format(q=requests.utils.quote(QUERY))
    started = time.perf_counter()
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
    except Exception as exc:
        print(f"[{name:11s}] ✖ 网络不可达  {type(exc).__name__}: {str(exc)[:70]}")
        return False

    elapsed = time.perf_counter() - started
    soup = BeautifulSoup(response.text, "lxml")
    for selector in selectors:
        items = soup.select(selector)
        if items:
            print(f"[{name:11s}] ✔ HTTP {response.status_code}  {len(items)} 条  {elapsed:.2f}s  ({selector})")
            for item in items[:2]:
                link = item.find("a")
                title = link.get_text(" ", strip=True) if link else item.get_text(" ", strip=True)
                href = link.get("href", "") if link else ""
                print(f"             · {title[:56]}")
                print(f"               {href[:78]}")
            return True
    print(f"[{name:11s}] ⚠ HTTP {response.status_code} 但未匹配到结果节点（页面长度 {len(response.text)}）")
    return False


def main() -> int:
    print(f"探测查询词：{QUERY}\n")
    available = [name for name, url, sel in TARGETS if probe(name, url, sel)]
    print("\n可用引擎：", available or "无")
    return 0 if available else 1


if __name__ == "__main__":
    sys.exit(main())
