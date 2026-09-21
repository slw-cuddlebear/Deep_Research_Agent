"""
全局配置模块：路径、默认参数、模型与环境变量。
所有模块统一从这里取配置，保证 app.py / main.py / eval_*.py 行为一致。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ========== 路径 ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"

load_dotenv(BASE_DIR / ".env")

for _d in (DATA_DIR, OUTPUT_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ========== 大模型 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
TEMPERATURE = 0.3

# ========== 规划器 ==========
MIN_SUB_QUESTIONS = 3          # 子问题数量下限
MAX_SUB_QUESTIONS = 5          # 子问题数量上限

# ========== 搜索与抓取 ==========
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SEARCH_ENGINE = "auto"         # auto / tavily / ddg
SEARCH_RESULTS_PER_QUERY = 5   # 单条 query 取回的结果数
DDG_REGION = "cn-zh"           # DuckDuckGo 区域，中文优先
MAX_CONCURRENCY = 5            # 并发搜索 / 并发抓取的线程数
FETCH_TIMEOUT = 20             # 单页抓取超时（秒）
MAX_PAGE_CHARS = 6000          # 单页正文送入 LLM 的最大字符数
MAX_PAGES_PER_ROUND = 12       # 单轮反思最多阅读的网页数
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0 Safari/537.36"
)

# ========== 反思循环 ==========
MAX_ITERATIONS = 3             # 最大搜索轮数，防止死循环

# ========== 事实与引用 ==========
MIN_FACT_CONFIDENCE = 0.3      # 低于该置信度的事实直接丢弃
DEDUP_SIMILARITY = 0.90        # 事实去重的相似度阈值

# ========== 报告生成 ==========
MIN_SECTIONS = 3               # 报告正文章节数下限
MAX_SECTIONS = 5               # 报告正文章节数上限
MAX_CITATIONS = 40             # 参考文献条目上限（超出部分不再分配引用号）
MIN_FACTS_PER_SECTION = 2      # 每章至少分到几条事实

# ========== 本地知识库（Phase 5）==========
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
LOCAL_KB_DIR = DATA_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOCAL_TOP_K = 5                # 每个子问题从本地库召回几个片段
# 本地召回的相关度下限。实测标定（BGE 余弦相似度，12 个片段的小语料）：
#   相关子问题最高分 0.59~0.77（均值 0.67）
#   无关问题最高分   0.33~0.42（均值 0.40），只有"python 怎么读写 excel"这类
#   技术性无关问题会冲到 0.60 形成难例
# 取 0.45：相关查询 8/8 全部召回，无关查询仅 1/8 误召。
# 故意偏向召回——漏掉内部知识比多读一段废话代价更大。
LOCAL_MIN_SCORE = 0.45
LOCAL_CHUNK_SIZE = 700          # 本地文档切片长度
LOCAL_CHUNK_OVERLAP = 120       # 切片重叠
USE_LOCAL_KB = True             # 是否启用本地知识库（可用 --no-local 关闭）
# 本地知识库的**最低**阅读名额（不是硬性比例）。
# 本地片段数量天然少于网页，纯按分数排序容易被整体挤掉，所以保底几条；
# 剩下的名额由网页与本地按融合分数自由竞争。
# 为什么不做固定比例配额：Phase 8 消融实验实测，固定 40% 配额在小语料下
# 会白白占掉网页阅读名额（T1 网页阅读量 29 篇 → 19 篇），而本地引用只占 6%。
LOCAL_MIN_SLOTS = 2
LOCAL_CREDIBILITY = 0.85        # 本地知识库来源的可信度（自建/受控语料，高于普通网页）
LOCAL_SCORE_BOOST = 1.15        # 融合排序时给本地结果的加权（内部资料通常更贴身）

# ========== 可观测性（Phase 7）==========
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
TRACING_ENABLED = os.getenv("TRACING_ENABLED", "1").strip() not in ("0", "false", "False")

# 成本核算：美元 → 人民币汇率；模型单价可在 .env 里覆盖（美元 / 百万 tokens）
USD_TO_CNY = float(os.getenv("USD_TO_CNY", "7.1") or 7.1)
LLM_PRICE_INPUT = float(os.environ["LLM_PRICE_INPUT"]) if os.getenv("LLM_PRICE_INPUT") else None
LLM_PRICE_OUTPUT = float(os.environ["LLM_PRICE_OUTPUT"]) if os.getenv("LLM_PRICE_OUTPUT") else None


# ========== 初始默认值的快照 ==========
# 评估脚本会**临时改写**上面的配置来跑消融实验，每跑一组都要能还原回默认值。
# 放在文件末尾是为了确保引用的常量都已经定义好了。
DEFAULT_MAX_ITERATIONS = MAX_ITERATIONS
DEFAULT_MAX_PAGES = MAX_PAGES_PER_ROUND
DEFAULT_USE_LOCAL_KB = USE_LOCAL_KB
DEFAULT_LOCAL_TOP_K = LOCAL_TOP_K
DEFAULT_LOCAL_MIN_SLOTS = LOCAL_MIN_SLOTS


def has_llm_key() -> bool:
    """是否配置了可用的 DeepSeek Key"""
    return bool(DEEPSEEK_API_KEY) and not DEEPSEEK_API_KEY.startswith("sk-自行配置")


def has_tavily_key() -> bool:
    """是否配置了 Tavily Key（没配就走免费的 DuckDuckGo）"""
    return bool(TAVILY_API_KEY)


def describe_env() -> dict:
    """返回当前环境配置摘要，供 CLI / 前端展示"""
    if has_tavily_key():
        engine = "Tavily"
    else:
        engine = "DuckDuckGo（免费兜底，部分网络不可达，建议配置 TAVILY_API_KEY）"

    local_files = []
    if LOCAL_KB_DIR.exists():
        local_files = [p for p in LOCAL_KB_DIR.rglob("*")
                       if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf", ".docx")]
    local_desc = (f"已启用（{len(local_files)} 篇文档）"
                  if USE_LOCAL_KB and local_files else
                  "已启用但语料为空" if USE_LOCAL_KB else "已关闭")

    return {
        "DeepSeek Key": "已配置" if has_llm_key() else "未配置",
        "模型": DEEPSEEK_MODEL,
        "网页搜索": engine,
        "本地知识库": local_desc,
        "Embedding": EMBEDDING_MODEL,
        "最大搜索轮数": MAX_ITERATIONS,
    }
