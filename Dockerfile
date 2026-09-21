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
# 会连带拉约 2GB 的 nvidia-* 依赖，把镜像从 ~1.2GB 撑到 ~4GB。
# 所以先从国内镜像单独装 CPU 版，后面装 requirements 时 pip 会认为 torch 已满足。
#
# 镜像源实测（同一个 torch-2.6.0+cpu 轮子）：
#   上交大 mirror.sjtu.edu.cn  11.9 MB/s  ✅
#   阿里云 mirrors.aliyun.com   0.08 MB/s  ❌（会卡到超时）
# 注意：上交大的 /cpu/ 是 PEP 503 包索引（按包名分子目录），
# 不能当 --find-links 用，只能走直链或用它当 index。
ARG TORCH_WHEEL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl
ARG TORCH_CPU_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
ARG TORCH_SPEC=torch==2.6.0+cpu

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
# 第一步：装 CPU 版 torch（直链优先，失败则回退到上交大的包索引）
# 第二步：装其余依赖；此时 torch 已满足，pip 不会再拉 CUDA 版
RUN set -eux; \
    pip install --no-cache-dir -i "${PIP_INDEX_URL}" "${TORCH_WHEEL}" \
      || pip install --no-cache-dir -i "${PIP_INDEX_URL}" \
             --extra-index-url "${TORCH_CPU_INDEX}" "${TORCH_SPEC}"; \
    pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r requirements.txt; \
    python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

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
