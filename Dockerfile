# syntax=docker/dockerfile:1
#
# Deep Research Agent 镜像
#
# 构建：
#   docker build -t deep-research-agent:latest .
#
# 运行（推荐用 docker compose）：
#   docker compose up -d
#   浏览器打开 http://localhost:8501
#
# 国内网络提示：
#   本机实测 download.docker.com / hub.docker.com / desktop.docker.com 均不可达，
#   基础镜像需要走国内加速站才能拉到，例如：
#     docker pull docker.1panel.live/library/python:3.11-slim
#     docker tag  docker.1panel.live/library/python:3.11-slim python:3.11-slim
#   或把加速站配进 Docker 的 daemon.json 的 registry-mirrors。

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Deep Research Agent" \
      org.opencontainers.image.description="拆解问题→多源检索→阅读提炼→反思补充→生成带引用的研究报告" \
      org.opencontainers.image.source="https://github.com/slw-cuddlebear/Deep_Research_Agent" \
      org.opencontainers.image.licenses="MIT"

# 允许在构建时覆盖 pip 源（国内默认走清华镜像）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# ---- CPU 版 PyTorch ----
# 项目只在 CPU 上跑 embedding，而 PyPI 上 Linux 的 torch 默认是 CUDA 版，
# 会连带拉约 3GB 的 nvidia-* / triton 依赖，明显撑大镜像。
# 所以先从国内镜像单独装 CPU 版，后面装 requirements 时 pip 会认为 torch 已满足。
# （实测最终镜像体积约 3.17GB，见 README「工程验证记录」。）
#
# 镜像源实测（同一个 torch-2.6.0+cpu 轮子）：
#   上交大 mirror.sjtu.edu.cn  11.9 MB/s  ✅
#   阿里云 mirrors.aliyun.com   0.08 MB/s  ❌（会卡到超时）
#
# 上交大的 /cpu/ 是**标准 PEP 503 包索引**（按包名分子目录，torch/ 子目录里就是各
# 版本的轮子），因此可以直接当 index 用，比硬编码某个直链更稳（直链会随上游清理失效）。
#
# ⚠️ requirements.txt 里也写了 torch==2.6.0，两处版本必须一致：否则
# sentence-transformers 会把已装的 CPU 版**升级成 CUDA 版**，白白多出 3GB 依赖
# （这是实际踩过的回归，见 README「踩坑与解决」第 16 条）。

ARG TORCH_CPU_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
ARG TORCH_SPEC=torch==2.6.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ---- 系统层：换成国内 Debian 源，并装中文字体 ----
# 中文字体是 PDF 导出必需的：report.py 的 FONT_CANDIDATES 里列了
# /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc，fonts-wqy-zenhei 正好装在这里。
# 不装的话导出 PDF 会被自动跳过（不会报错，但会少一种格式）。
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        fonts-wqy-zenhei \
        curl \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 依赖层：单独 COPY requirements.txt，代码改动时不必重装依赖 ----
COPY requirements.txt ./
# 第 0 步：先升级 pip。基础镜像自带的 pip 24.0 太旧，在内存受限的 WSL2 里解析本项目
#   近 200 个依赖时会发生**段错误**（实测 exit 139 / 偶发 UnboundLocalError，
#   崩在不同包上、不可稳定复现，是典型的内存损坏）。升级到 pip 26 后同一命令稳定通过。
# 第 1 步：装 CPU 版 torch（PEP 503 索引）；requirements.txt 里同样锁了 torch==2.6.0，
#   否则后续会被 sentence-transformers 升级成 CUDA 版并多拉约 3GB 的 nvidia-*。
# 第 2 步：装其余依赖；此时 torch 已满足，pip 不会再拉 CUDA 版。
#
# 关于 --mount=type=cache：把 pip 的下载缓存挂进来，重复构建不必重下（首次仍要下）。
# 该语法需要 BuildKit；若你的 Docker 只有 legacy builder，去掉这两个 --mount 即可，
# 其余指令与两种构建器都兼容（详见 README「国内网络环境部署说明」）。
RUN --mount=type=cache,target=/root/.cache/pip set -eux; \
    pip install --upgrade -i "${PIP_INDEX_URL}" pip; \
    pip install -i "${PIP_INDEX_URL}" \
        --extra-index-url "${TORCH_CPU_INDEX}" "${TORCH_SPEC}"; \
    pip install -i "${PIP_INDEX_URL}" -r requirements.txt; \
    python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"; \
    python -c "import torch; assert 'cpu' in torch.__version__ and not torch.cuda.is_available(), 'torch 不是 CPU 版：' + torch.__version__"; \
    ! pip list 2>/dev/null | grep -qi '^nvidia-'

# ---- 代码层 ----
COPY . .

# 运行时目录（挂载卷时会覆盖，这里保证镜像内也能直接跑）
RUN mkdir -p outputs results cache chroma_db data/knowledge_base

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
