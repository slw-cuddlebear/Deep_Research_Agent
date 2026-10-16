"""
环境体检模块（Environment Doctor）。
用途：把"跑不起来"的原因一次性查清楚——依赖是否装全、Key 是否填了、网络是否通、Key 是否有效。

运行： python main.py --doctor
"""
import importlib
import socket
import sys
import time

from rich.console import Console
from rich.table import Table

import config

console = Console()

# 逐个 import 名 → 展示名
REQUIRED_PACKAGES = [
    ("langgraph", "langgraph"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_core", "langchain-core"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("langchain_chroma", "langchain-chroma"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("trafilatura", "trafilatura"),
    ("bs4", "beautifulsoup4"),
    ("requests", "requests"),
    ("tavily", "tavily-python"),
    ("ddgs", "ddgs"),
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("langfuse", "langfuse"),
    ("markdown", "markdown"),
    ("xhtml2pdf", "xhtml2pdf"),
    ("pypdf", "pypdf"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("tenacity", "tenacity"),
    ("rich", "rich"),
]

# 需要联网的地址：(展示名, 主机, 端口, 是否必需)
HOSTS = [
    ("DeepSeek API", "api.deepseek.com", 443, True),
    ("Tavily API", "api.tavily.com", 443, True),
    ("HuggingFace 镜像", "hf-mirror.com", 443, False),
    ("DuckDuckGo（免费兜底）", "html.duckduckgo.com", 443, False),
]


def check_local_kb() -> bool:
    """本地知识库检查：语料是否存在、索引是否就绪、embedding 能否加载"""
    console.print("\n[bold]本地知识库检查[/bold]")
    try:
        import local_kb
    except Exception as exc:
        console.print(f"  ✖ local_kb 模块不可用：{type(exc).__name__}: {exc}")
        return False

    files = local_kb.list_source_files()
    if not files:
        console.print(f"  ⚠ {config.LOCAL_KB_DIR} 下没有文档，本地知识库将被跳过")
        console.print("    [dim]把 .md / .txt / .pdf / .docx 放进该目录即可启用[/dim]")
        return False

    console.print(f"  ✔ 语料目录：{config.LOCAL_KB_DIR}（{len(files)} 篇文档）")
    for path in files[:6]:
        console.print(f"    · [dim]{path.name}[/dim]")
    if len(files) > 6:
        console.print(f"    · [dim]…另有 {len(files) - 6} 篇[/dim]")

    manifest = local_kb._read_manifest()
    ready = local_kb.is_index_ready()
    if ready:
        console.print(f"  ✔ 向量索引：已就绪（{manifest.get('chunks', 0)} 个片段）")
    else:
        console.print("  ⚠ 向量索引：尚未构建（首次检索时会自动构建，需加载 embedding 模型）")
    console.print(f"  [dim]Embedding 模型：{config.EMBEDDING_MODEL}（CPU）[/dim]")
    return True


def check_observability() -> bool:
    """可观测性检查：本地追踪是否可用、LangFuse 是否配置且能鉴权"""
    console.print("\n[bold]可观测性检查[/bold]")
    try:
        import observability
    except Exception as exc:
        console.print(f"  ✖ observability 模块不可用：{type(exc).__name__}: {exc}")
        return False

    if config.TRACING_ENABLED:
        console.print("  ✔ 本地追踪：已启用（每步耗时 / Token / 成本 → trace.json）")
    else:
        console.print("  ⚠ 本地追踪：已被 TRACING_ENABLED 关闭")

    if not observability.langfuse_enabled():
        console.print("  · LangFuse：未配置（可选）。到 https://cloud.langfuse.com 注册后填写 "
                      "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 即可自动上报")
        return True

    console.print(f"  · LangFuse Host：{config.LANGFUSE_HOST}")
    ok, message = observability.verify_langfuse()
    console.print(f"  {'✔' if ok else '✖'} LangFuse 鉴权：{message}")
    return ok


def check_pdf_font() -> bool:
    """PDF 导出所需的中文字体检查"""
    console.print("\n[bold]PDF 导出检查[/bold]")
    try:
        from report import find_cjk_font, platform_hint
    except Exception as exc:
        console.print(f"  ✖ report 模块不可用：{type(exc).__name__}: {exc}")
        return False

    font_path, error = find_cjk_font()
    if font_path:
        console.print(f"  ✔ 中文字体：{font_path}")
        console.print(f"  [dim]{platform_hint()}[/dim]")
        return True
    console.print("  ⚠ 未找到中文字体，将跳过 PDF 导出（Markdown / HTML 不受影响）")
    console.print(f"  [dim]{error}[/dim]")
    return False


def check_python() -> bool:
    """Python 版本检查（README 与技术栈声明 Python 3.11；3.10 起可运行）"""
    version = sys.version_info
    ok = version >= (3, 10)
    console.print(
        f"{'✔' if ok else '✖'} Python 版本：{sys.version.split()[0]}"
        + ("" if ok else "（需要 3.10 及以上；项目声明并实测的是 3.11）")
    )
    return ok


def check_packages() -> list:
    """依赖安装检查，返回缺失的包"""
    console.print("\n[bold]依赖检查[/bold]")
    missing = []
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            console.print(f"  ✔ {display_name}")
        except Exception as exc:
            missing.append(display_name)
            console.print(f"  ✖ {display_name}  [dim]({type(exc).__name__}: {str(exc)[:60]})[/dim]")
    return missing


def check_env() -> dict:
    """环境变量检查"""
    console.print("\n[bold]环境变量检查[/bold]")
    env_path = config.BASE_DIR / ".env"
    console.print(f"  {'✔' if env_path.exists() else '✖'} .env 文件：{'已存在' if env_path.exists() else '不存在（请复制 .env.example 为 .env）'}")

    llm_ok, tavily_ok = config.has_llm_key(), config.has_tavily_key()
    console.print(f"  {'✔' if llm_ok else '✖'} DEEPSEEK_API_KEY："
                  + ("已配置" if llm_ok else "未配置 → https://platform.deepseek.com"))
    console.print(f"  {'✔' if tavily_ok else '✖'} TAVILY_API_KEY："
                  + ("已配置" if tavily_ok else "未配置 → https://tavily.com（免费 1000 次/月）"))
    return {"llm": llm_ok, "tavily": tavily_ok}


def check_network() -> dict:
    """网络连通性检查"""
    console.print("\n[bold]网络连通性检查[/bold]")
    results = {}
    for name, host, port, required in HOSTS:
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=6):
                results[name] = True
                console.print(f"  ✔ {name}  [dim]({host}:{port}, {(time.perf_counter() - started) * 1000:.0f}ms)[/dim]")
        except Exception as exc:
            results[name] = False
            tag = "必需" if required else "可选"
            console.print(f"  ✖ {name}  [dim][{tag}] {host}:{port} 不可达（{type(exc).__name__}）[/dim]")
    return results


def check_llm_live() -> bool:
    """真实调用一次 DeepSeek，验证 Key 是否可用"""
    if not config.has_llm_key():
        console.print("\n[dim]跳过 DeepSeek 连通测试（Key 未配置）[/dim]")
        return False
    console.print("\n[bold]DeepSeek 调用测试[/bold]")
    try:
        import llm

        llm.set_llm(None)
        reply = llm.call_llm("只回复两个字：正常")
        console.print(f"  ✔ 调用成功，模型回复：{reply.strip()[:40]}")
        usage = llm.get_usage()
        console.print(f"  [dim]Token 用量：{usage['total_tokens']}[/dim]")
        return True
    except Exception as exc:
        console.print(f"  ✖ 调用失败：{type(exc).__name__}: {str(exc)[:200]}")
        return False


def check_tavily_live() -> bool:
    """真实调用一次 Tavily，验证 Key 是否可用"""
    if not config.has_tavily_key():
        console.print("\n[dim]跳过 Tavily 连通测试（Key 未配置）[/dim]")
        return False
    console.print("\n[bold]Tavily 调用测试[/bold]")
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query="DeepSeek", max_results=2, search_depth="basic")
        count = len(response.get("results", []))
        console.print(f"  ✔ 调用成功，返回 {count} 条结果")
        return count > 0
    except Exception as exc:
        console.print(f"  ✖ 调用失败：{type(exc).__name__}: {str(exc)[:200]}")
        return False


def run_doctor() -> int:
    """执行全部体检项，返回 0 表示核心链路可用"""
    console.rule("[bold]Deep Research Agent 环境体检[/bold]")
    console.print(f"项目目录：{config.BASE_DIR}\n")

    python_ok = check_python()
    missing = check_packages()
    env = check_env()
    network = check_network()
    check_local_kb()
    check_pdf_font()
    check_observability()

    check_llm_live()
    check_tavily_live()

    # ---------- 结论 ----------
    console.rule("[bold]体检结论[/bold]")
    blockers = []
    if not python_ok:
        blockers.append("Python 版本过低（需 3.10+）")
    if missing:
        blockers.append(f"缺少依赖：{', '.join(missing)} → pip install -r requirements.txt")
    if not env["llm"]:
        blockers.append("DEEPSEEK_API_KEY 未配置 → 到 https://platform.deepseek.com 申请后填入 .env")
    if not env["tavily"] and not network.get("DuckDuckGo（免费兜底）"):
        blockers.append(
            "网页搜索不可用：未配置 TAVILY_API_KEY，且 DuckDuckGo 在当前网络不可达 "
            "→ 到 https://tavily.com 申请免费 Key 后填入 .env"
        )

    if blockers:
        console.print("[yellow]还需你手动完成以下事项：[/yellow]")
        for idx, item in enumerate(blockers, 1):
            console.print(f"  {idx}. {item}")
    else:
        console.print("[green]全部就绪，可以运行：python main.py \"你的研究问题\"[/green]")

    table = Table(title="一句话状态", show_header=False, box=None)
    table.add_row("[cyan]离线链路自检[/cyan]", "python main.py \"问题\" --offline （不需要任何 Key）")
    table.add_row("[cyan]完整研究[/cyan]", "python main.py \"问题\" （需要 DeepSeek + Tavily）")
    table.add_row("[cyan]计划可人工调整[/cyan]", "python main.py \"问题\" --interactive")
    console.print(table)

    return 0 if not blockers else 1
