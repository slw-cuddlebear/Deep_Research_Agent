"""
Phase 9 自检脚本：Docker 配置的静态校验 + 仓库卫生检查。

运行方式：
    python tests/test_phase9.py

为什么是"静态校验"：本机 **未安装 Docker**，而且实测
desktop.docker.com / download.docker.com / hub.docker.com 全部被重置连接，
连 WSL2 内核包（wslstorestorage.blob.core.windows.net）与 aka.ms 也不可达，
所以镜像构建无法在本机验证。

既然构建跑不了，就把"能在本地验证的部分"验到位：
Dockerfile 指令合法性、COPY 的源文件是否真实存在、依赖是否装全、
中文字体包是否与代码里的候选路径对得上、.dockerignore 有没有误伤必需文件、
compose 是不是合法 YAML、.gitignore 有没有漏掉密钥或误伤证据文件、
以及仓库里有没有混进密钥和大文件。
"""
import os
import re
import sys
import tomllib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


def read(relative: str) -> str:
    path = BASE_DIR / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ========== 1. Dockerfile ==========
def test_dockerfile() -> None:
    print("\n[1] Dockerfile")
    text = read("Dockerfile")
    check("Dockerfile 存在且非空", bool(text.strip()))

    upper = text.upper()
    for instruction in ("FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD", "HEALTHCHECK"):
        check(f"包含 {instruction} 指令", re.search(rf"^\s*{instruction}\b", upper, re.M) is not None,
              f"缺少 {instruction}")

    check("基础镜像带版本（没有用 latest）",
          bool(re.search(r"^FROM\s+\S+:\S+", text, re.M)), "FROM 行缺少 tag")
    check("Python 版本与项目要求一致（3.11）",
          "python:3.11" in text, "基础镜像不是 python:3.11")

    # 构建时用的 pip 源要能从国内访问
    check("默认使用国内 pip 源",
          "pypi.tuna.tsinghua.edu.cn" in text, "没有配置国内 pip 镜像")
    check("pip 源可被覆盖（构建参数）",
          "ARG PIP_INDEX_URL" in text, "缺少 PIP_INDEX_URL 构建参数")

    # 中文字体：必须与 report.py 里的 Linux 候选路径对得上
    from report import FONT_CANDIDATES
    wqy_paths = [p for p, _ in FONT_CANDIDATES if p.startswith("/usr/share/fonts/truetype/wqy/")]
    check("代码里有 Linux 文泉驿字体候选路径", bool(wqy_paths), str(wqy_paths[:2]))
    check("Dockerfile 安装了 fonts-wqy-zenhei",
          "fonts-wqy-zenhei" in text,
          "没装字体的话 PDF 导出会被静默跳过")
    check("字体包路径与代码候选一致",
          any("wqy-zenhei" in p for p in wqy_paths),
          f"代码里找的是 {wqy_paths}，但装了 fonts-wqy-zenhei 会装到 "
          f"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

    # apt 源：国内构建
    check("apt 换成了国内镜像", "mirrors.tuna.tsinghua.edu.cn" in text,
          "deb.debian.org 在国内很慢")
    check("安装后清理 apt 缓存", "rm -rf /var/lib/apt/lists" in text)
    check("apt 使用 --no-install-recommends", "--no-install-recommends" in text)

    # 依赖安装要在 COPY . . 之前，才能吃到层缓存
    dep_pos = text.find("requirements.txt ./")
    copy_all_pos = text.find("COPY . .")
    check("先装依赖再拷贝代码（利用层缓存）",
          0 < dep_pos < copy_all_pos, f"requirements 位置={dep_pos}, COPY . . 位置={copy_all_pos}")

    # COPY 的源必须在仓库里真实存在
    copy_sources = re.findall(r"^COPY\s+(?!--from)(.+?)\s+\S+\s*$", text, re.M)
    missing = []
    for source in copy_sources:
        for token in source.split():
            if token.startswith("--"):
                continue
            if not (BASE_DIR / token).exists():
                missing.append(token)
    check("COPY 的源文件都存在", not missing, f"缺失：{missing}")

    check("暴露 8501 端口", "EXPOSE 8501" in text)
    check("健康检查打到 Streamlit 健康端点",
          "_stcore/health" in text, "健康检查路径不对")
    check("启动命令是 streamlit run app.py",
          "streamlit" in text and "app.py" in text)
    check("容器内监听 0.0.0.0（否则宿主机访问不到）",
          "0.0.0.0" in text, "缺少 --server.address=0.0.0.0")
    check("设置了时区", "TZ=Asia/Shanghai" in text)
    check("HF 走国内镜像",
          "hf-mirror.com" in text, "模型下载会很慢")
    check("容器的入口文件存在", (BASE_DIR / "app.py").exists())

    # 镜像体积提示：torch 很大，README 要说清楚
    requirements = read("requirements.txt")
    heavy = [p for p in ("sentence-transformers", "torch") if p in requirements]
    check("识别出重量级依赖（用于文档说明）", bool(heavy), str(heavy))


# ========== 2. .dockerignore ==========
def test_dockerignore() -> None:
    print("\n[2] .dockerignore")
    text = read(".dockerignore")
    check(".dockerignore 存在", bool(text.strip()))

    lines = [line.strip() for line in text.splitlines()]
    rules = [line for line in lines if line and not line.startswith("#")]

    for must in (".env", ".venv/", "outputs/", "chroma_db/", "__pycache__/", ".git/"):
        check(f"忽略了 {must}", must in rules, f"规则列表：{rules}")

    check("没有忽略 .env 以外的必需目录（data/）",
          "data/" not in rules, "data/ 里有测试问题集与知识库语料，不能忽略")
    check("没有忽略 tests/（容器内可跑自检）",
          "tests/" not in rules, "tests/ 被忽略了")
    check("没有忽略 requirements.txt",
          "requirements.txt" not in rules)
    check("没有忽略 app.py / main.py",
          "app.py" not in rules and "main.py" not in rules)


# ========== 3. docker-compose.yml ==========
def test_compose() -> None:
    print("\n[3] docker-compose.yml")
    text = read("docker-compose.yml")
    check("docker-compose.yml 存在", bool(text.strip()))

    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception as exc:
        check("YAML 语法合法", False, f"{type(exc).__name__}: {exc}")
        return

    check("YAML 语法合法", isinstance(data, dict), str(type(data)))
    services = data.get("services") or {}
    check("定义了 services", bool(services), str(list(services.keys())))
    check("包含 web 服务", "web" in services, str(list(services.keys())))
    check("包含 cli 服务", "cli" in services)
    check("包含 eval 服务", "eval" in services)

    web = services.get("web") or {}
    check("web 有 build 配置", "build" in web, str(list(web.keys())))
    check("web 用 env_file 注入密钥（而不是写进镜像）",
          ".env" in (web.get("env_file") or []), str(web.get("env_file")))
    check("web 映射了 8501 端口",
          any("8501" in str(p) for p in (web.get("ports") or [])), str(web.get("ports")))
    check("web 设置了重启策略", bool(web.get("restart")), str(web.get("restart")))
    check("web 定义了健康检查", "healthcheck" in web)

    volumes = web.get("volumes") or []
    volume_text = " ".join(str(v) for v in volumes)
    for must in ("outputs", "chroma_db", "data"):
        check(f"web 挂载了 {must}", must in volume_text, volume_text)
    check("模型缓存用具名卷持久化",
          "hf_cache" in volume_text, volume_text)
    check("声明了 hf_cache 具名卷",
          "hf_cache" in (data.get("volumes") or {}), str(data.get("volumes")))

    # cli / eval 放 profile，避免 up 时被顺带启动
    for name in ("cli", "eval"):
        check(f"{name} 在 profile 里（不会随 up 启动）",
              bool((services.get(name) or {}).get("profiles")),
              str((services.get(name) or {}).get("profiles")))
        check(f"{name} 有 entrypoint",
              bool((services.get(name) or {}).get("entrypoint")),
              str((services.get(name) or {}).get("entrypoint")))

    check("cli 的入口是 main.py",
          "main.py" in str((services.get("cli") or {}).get("entrypoint")), "")
    check("eval 的入口是 eval_research.py",
          "eval_research.py" in str((services.get("eval") or {}).get("entrypoint")), "")
    check("compose 里说明了国内镜像加速（否则拉不到基础镜像）",
          "1panel" in text or "registry-mirrors" in text or "加速" in text,
          "没有提示如何解决 registry 不可达")


# ========== 4. Streamlit 配置 ==========
def test_streamlit_config() -> None:
    print("\n[4] Streamlit 配置")
    path = BASE_DIR / ".streamlit" / "config.toml"
    check(".streamlit/config.toml 存在", path.exists())
    if not path.exists():
        return
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        check("TOML 语法合法", False, f"{type(exc).__name__}: {exc}")
        return
    check("TOML 语法合法", isinstance(data, dict))
    check("开启 headless（服务器环境必需）",
          data.get("server", {}).get("headless") is True, str(data.get("server")))
    check("端口与 Dockerfile 一致（8501）",
          data.get("server", {}).get("port") == 8501, str(data.get("server", {}).get("port")))
    check("监听地址与 Dockerfile 一致",
          data.get("server", {}).get("address") == "0.0.0.0",
          str(data.get("server", {}).get("address")))
    check("关闭使用统计上报",
          data.get("browser", {}).get("gatherUsageStats") is False)


# ========== 5. 仓库卫生 ==========
def test_repo_hygiene() -> None:
    print("\n[5] 仓库卫生（密钥 / 大文件 / 忽略规则）")
    gitignore = read(".gitignore")
    rules = [line.strip() for line in gitignore.splitlines()
             if line.strip() and not line.startswith("#")]

    for must in (".env", ".venv/", "chroma_db/", "outputs/", "__pycache__/"):
        check(f".gitignore 覆盖 {must}", must in rules, str(rules))
    check(".gitignore 没有忽略 results/（评估证据要提交）",
          "results/" not in rules, "results/ 被忽略了，评估结论会丢失")
    check(".gitignore 忽略 Streamlit 机密文件",
          ".streamlit/secrets.toml" in rules, str(rules))

    # 全仓库扫描：不该出现真实密钥
    secret_patterns = [
        (re.compile(r"sk-[0-9a-f]{20,}"), "DeepSeek Key"),
        (re.compile(r"tvly-[A-Za-z0-9-]{15,}"), "Tavily Key"),
        (re.compile(r"pk-lf-[A-Za-z0-9-]{10,}"), "LangFuse Public Key"),
        (re.compile(r"sk-lf-[A-Za-z0-9-]{10,}"), "LangFuse Secret Key"),
    ]
    skip_dirs = {".venv", ".git", "__pycache__", "outputs", "cache", "chroma_db", ".idea"}
    leaks = []
    for path in BASE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name == ".env":          # 本地密钥文件本身（已被 gitignore）
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".yml", ".yaml",
                                       ".toml", ".example", ".cfg", ".ini"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern, label in secret_patterns:
            if pattern.search(content):
                leaks.append(f"{path.relative_to(BASE_DIR)} 含疑似 {label}")
    check("仓库里没有混入真实密钥", not leaks, "；".join(leaks[:5]))

    # .env.example 必须是占位符
    example = read(".env.example")
    check(".env.example 里没有真实 Key",
          "sk-自行配置" in example and not re.search(r"sk-[0-9a-f]{20,}", example),
          "示例文件里出现了疑似真实 Key")

    # 大文件检查（排除已被忽略的目录）
    big = []
    for path in BASE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            big.append(f"{path.relative_to(BASE_DIR)} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
    check("没有被跟踪的超大文件（>2MB）", not big, "；".join(big[:5]))

    # 运行产物不应出现在受跟踪范围
    check("outputs/ 目录被忽略（内含 15MB 运行产物）",
          "outputs/" in rules, str(rules))


# ========== 6. README 完整性 ==========
def test_readme() -> None:
    print("\n[6] README 完整性")
    text = read("README.md")
    check("README 存在且足够长", len(text) > 6000, f"{len(text)} 字符")

    required_sections = [
        "快速开始", "系统架构", "技术栈", "项目结构", "量化评估",
        "可观测性", "前端", "License",
    ]
    for section in required_sections:
        check(f"含「{section}」章节", section in text, f"缺少 {section}")

    check("含 Docker 部署说明", "docker" in text.lower(), "没有 Docker 说明")
    check("含开发日志（9 个 Phase）",
          text.count("**Phase ") >= 9, f"只找到 {text.count('**Phase ')} 条")
    check("含实测数据", "实测" in text, "没有实测结果")
    check("说明了局限与同源偏见", "同源偏见" in text or "局限" in text)
    check("含 GitHub 仓库地址",
          "github.com/slw-cuddlebear" in text, "没有仓库地址")


def test_markdown_json_blocks() -> None:
    """
    回归检查：Markdown 里标注为 ```json 的代码块必须是**严格合法 JSON**。

    起因：README 中曾写成
        // Docker Desktop → Settings → Docker Engine
        { "registry-mirrors": [...] }
    编辑器按 JSON 语法检查会标红（JSON 标准不允许注释）。这不影响文档阅读，
    但 `daemon.json` 本身就是严格 JSON，读者整段复制进 `daemon.json` 会导致
    Docker 守护进程启动失败——因此说明文字必须写在代码块之外。
    """
    print("\n[6c] Markdown 里的 JSON 代码块必须是严格 JSON")
    import json
    import re

    blocks_checked, offenders, commented = 0, [], []
    # 注意：不能简单用 `"//" in block` 判断注释——URL 里的 `https://` 也含 `//`。
    # 只把「行首（允许前置空白）以 // 开头」视为注释。
    comment_line = re.compile(r"^\s*//", re.M)

    for path in list(BASE_DIR.glob("*.md")) + list((BASE_DIR / "results").glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for block in re.findall(r"```json\s*\n(.*?)```", text, re.S):
            blocks_checked += 1
            if comment_line.search(block):
                commented.append(path.name)
            try:
                json.loads(block)
            except Exception as exc:
                offenders.append(f"{path.name}: {exc}")

    check("仓库里存在 JSON 示例代码块", blocks_checked > 0, "一个都没找到，检查被跳过")
    check("所有 ```json 代码块都是严格合法 JSON", not offenders, "；".join(offenders[:3]))
    check("JSON 代码块里不含 // 注释行", not commented, str(sorted(set(commented))))
    check("注释检查不会误伤 URL（https:// 不算注释）",
          not comment_line.search('{"u": "https://example.com"}'),
          "正则把 URL 当成注释了")


def test_standard_files() -> None:
    print("\n[6b] 标准工程文件")
    required = {
        "LICENSE": "MIT License",
        "README.md": "Deep Research Agent",
        "requirements.txt": "streamlit",
        ".gitignore": "outputs/",
        ".env.example": "DEEPSEEK_API_KEY",
        "Dockerfile": "FROM",
        "docker-compose.yml": "services:",
        ".dockerignore": ".env",
    }
    for name, must_contain in required.items():
        path = BASE_DIR / name
        exists = path.exists()
        check(f"{name} 存在", exists)
        if exists and must_contain:
            check(f"{name} 内容合理（含「{must_contain}」）",
                  must_contain in path.read_text(encoding="utf-8", errors="ignore"),
                  f"没找到「{must_contain}」")

    license_text = read("LICENSE")
    check("LICENSE 是 MIT", license_text.startswith("MIT License"), license_text[:30])
    check("LICENSE 有版权人", "Copyright (c)" in license_text)
    check("README 声明的协议与 LICENSE 一致",
          "MIT" in read("README.md"), "README 里没写 MIT")
    check("results/ 有结果索引说明", (BASE_DIR / "results" / "README.md").exists(),
          "评估结果缺少索引，容易看混")


# ========== 7. 代码可编译 ==========
def test_code_health() -> None:
    print("\n[7] 代码健康度")
    import py_compile

    entry_files = ["app.py", "main.py", "run_tests.py", "eval_research.py",
                   "deep_research.py", "observability.py", "local_kb.py", "report.py"]
    broken = []
    for name in entry_files:
        path = BASE_DIR / name
        if not path.exists():
            broken.append(f"{name} 缺失")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            broken.append(f"{name}: {exc}")
    check("核心模块都能编译", not broken, "；".join(broken[:3]))

    check("所有入口文件都存在",
          all((BASE_DIR / n).exists() for n in entry_files),
          str([n for n in entry_files if not (BASE_DIR / n).exists()]))

    # requirements 与 doctor 的依赖清单要一致
    from doctor import REQUIRED_PACKAGES
    requirements = read("requirements.txt").lower()
    missing = [display for _, display in REQUIRED_PACKAGES
               if display.lower().split("-")[0] not in requirements]
    check("doctor 检查的依赖都在 requirements.txt 里", not missing, str(missing))


def main() -> int:
    print("=" * 60)
    print("Phase 9 自检：Docker 配置（静态校验）与仓库卫生")
    print("=" * 60)
    print("\n注：本机未安装 Docker，且 Docker 官方源不可达，因此这里只做静态校验。")
    test_dockerfile()
    test_dockerignore()
    test_compose()
    test_streamlit_config()
    test_repo_hygiene()
    test_readme()
    test_markdown_json_blocks()
    test_standard_files()
    test_code_health()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
