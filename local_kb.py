"""
本地知识库模块（Phase 5）。
把 data/knowledge_base/ 下的文档切片、向量化、存入 Chroma，并提供检索接口。

设计要点：
  1. **指纹复用索引**：用文件指纹（文件名 + 大小 + 内容哈希）判断语料是否变化，
     没变就直接复用已有索引，避免每次启动都重新向量化；
     指纹变了则清空并全量重建（不是只更新变化的那几个文档）。
  2. **文档即来源**：本地片段用 `local://<文件名>#chunk-<n>` 作为 URL，
     与网页 URL 走同一套引用编号与参考文献渲染逻辑（report.py 会识别 local:// 前缀）。
  3. **正文随结果返回**：本地片段自带正文，提取器无需再"抓取"，省一次网络请求。
  4. 模型复用 RAG 项目的 BGE 中文小模型，CPU 可跑，无需 GPU。
"""
import hashlib
import json
import shutil
import threading

import config
from state import new_id

# 懒加载单例
_embeddings = None
_vectorstore = None
_kb_cache = {"fingerprint": None, "stats": {}}

# 本地库检索是并发调用的（每个子问题一个线程），而 Chroma 的持久化客户端
# **不允许同一进程对同一目录重复实例化**，索引重建更是不能并发执行。
# 少了这把锁，5 个并发子问题会出现"3 个检索任务失败"这种偶发错误。
# 用 RLock 是因为 build_index 内部会再次进入受锁保护的代码。
_lock = threading.RLock()

LOCAL_URL_PREFIX = "local://"
MANIFEST_NAME = "kb_manifest.json"

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx")


# ========== 模型 ==========
def get_embeddings():
    """取 BGE 中文 embedding 模型（进程内单例）"""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ========== 语料读取 ==========
def list_source_files() -> list:
    """列出知识库目录下支持的文件"""
    if not config.LOCAL_KB_DIR.exists():
        return []
    return sorted(
        p for p in config.LOCAL_KB_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _fingerprint(files: list) -> str:
    """语料指纹：文件名 + 大小 + 内容哈希，任一变化都会让指纹改变"""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("utf-8"))
    return digest.hexdigest()[:16]


def load_documents() -> list:
    """加载知识库里的全部文档"""
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    documents = []
    for path in list_source_files():
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        except Exception as exc:
            print(f"[local_kb] 跳过 {path.name}：{type(exc).__name__}: {exc}")
            continue

        for doc in docs:
            doc.metadata["source_name"] = path.name
            doc.metadata["source_path"] = str(path)
        documents.extend(docs)
    return documents


def split_documents(documents: list) -> list:
    """按语义段落切片（中文按标点切，比按字符硬切更完整）"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.LOCAL_CHUNK_SIZE,
        chunk_overlap=config.LOCAL_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


# ========== 索引 ==========
def _manifest_path():
    return config.CHROMA_DIR / MANIFEST_NAME


def _read_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(data: dict) -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def is_index_ready() -> bool:
    """已有索引且语料未变化时返回 True"""
    files = list_source_files()
    if not files:
        return False
    manifest = _read_manifest()
    return (
        manifest.get("fingerprint") == _fingerprint(files)
        and config.CHROMA_DIR.exists()
        and (config.CHROMA_DIR / "chroma.sqlite3").exists()
    )


def build_index(force: bool = False) -> dict:
    """
    构建（或复用）本地知识库索引。线程安全。

    注意语义：**语料指纹未变化时直接复用**已有索引（省去重复向量化）；
    一旦指纹变化（增删改任一文档）就会清空向量库并**全量重建**，
    并非只对变化的文档做增量更新。
    :return: {"built": bool, "documents": n, "chunks": n, "model": str, "message": str}
    """
    global _vectorstore

    with _lock:
        return _build_index_locked(force)


def _build_index_locked(force: bool) -> dict:
    global _vectorstore

    files = list_source_files()
    if not files:
        return {"built": False, "documents": 0, "chunks": 0, "model": config.EMBEDDING_MODEL,
                "message": f"{config.LOCAL_KB_DIR} 下没有找到文档（支持 "
                           f"{'/'.join(SUPPORTED_SUFFIXES)}），本地知识库已跳过"}

    fingerprint = _fingerprint(files)
    manifest = _read_manifest()

    if not force and is_index_ready() and manifest.get("fingerprint") == fingerprint:
        return {"built": False, "documents": manifest.get("documents", 0),
                "chunks": manifest.get("chunks", 0), "model": config.EMBEDDING_MODEL,
                "message": f"复用已有索引（{manifest.get('chunks', 0)} 个片段，"
                           f"{manifest.get('documents', 0)} 篇文档）"}

    # 语料变了：先释放旧客户端再清目录，避免新旧文档串库 / 句柄占用
    _vectorstore = None
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()
    chunks = split_documents(documents)

    from langchain_chroma import Chroma

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="local_knowledge_base",
        persist_directory=str(config.CHROMA_DIR),
        # 显式指定余弦空间：BGE 输出是归一化向量，余弦距离才是正确的度量。
        # 不指定的话 Chroma 默认用 L2，其 relevance_score 可能落到 [0,1] 之外
        # （实测出现过 -0.02 这样的负分），相关度闸门就没法定阈值了。
        collection_metadata={"hnsw:space": "cosine"},
    )

    stats = {
        "fingerprint": fingerprint,
        "documents": len(files),
        "chunks": len(chunks),
        "model": config.EMBEDDING_MODEL,
        "files": [p.name for p in files],
    }
    _write_manifest(stats)
    return {"built": True, "documents": len(files), "chunks": len(chunks),
            "model": config.EMBEDDING_MODEL,
            "message": f"新建索引：{len(files)} 篇文档 → {len(chunks)} 个片段"}


def get_vectorstore():
    """取得（必要时构建）向量库。线程安全，且保证全进程只有一个客户端实例。"""
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            _build_index_locked(force=False)
            if not is_index_ready():
                return None
            from langchain_chroma import Chroma

            _vectorstore = Chroma(
                collection_name="local_knowledge_base",
                embedding_function=get_embeddings(),
                persist_directory=str(config.CHROMA_DIR),
            )
        return _vectorstore


def reset_cache() -> None:
    """清空进程内缓存（测试用）"""
    global _vectorstore, _embeddings
    with _lock:
        _vectorstore = None
        _kb_cache["fingerprint"] = None
        _kb_cache["stats"] = {}


# ========== 检索 ==========
def search_local(query: str, top_k: int = None, sub_question_id: str = "") -> list:
    """
    在本地知识库中检索，返回与 SearchResult 同构的 dict 列表：
    engine="local"、url="local://<文件名>#chunk-<n>"，并**自带正文**（content 字段），
    提取器拿到后不需要再抓取。

    低于 `config.LOCAL_MIN_SCORE`（默认 0.45）的片段会被丢弃——本地库只应该回答
    "它真正知道的事"。该阈值基于 BGE 余弦相似度实测标定（12 个片段的小语料）：
      相关子问题最高分 0.59~0.77（均值 0.67）
      无关问题最高分   0.33~0.42（均值 0.40）
    取 0.45 时相关查询 8/8 全部召回、无关查询仅 1/8 误召；标定细节见 config.py 注释。
    """
    top_k = top_k or config.LOCAL_TOP_K
    query = (query or "").strip()
    if not query:
        return []

    store = get_vectorstore()
    if store is None:
        return []

    # 用余弦距离自己算相似度，而不是让 Chroma 给 relevance_score：
    # 后者的取值范围不保证落在 [0,1]（实测出现过负数），相关度闸门会失效。
    try:
        pairs = store.similarity_search_with_score(query, k=top_k)
    except Exception:
        return []

    results = []
    scored = []
    for doc, distance in pairs:
        try:
            similarity = 1.0 - float(distance)
        except (TypeError, ValueError):
            similarity = 0.0
        scored.append((doc, min(max(similarity, 0.0), 1.0)))

    for doc, score in scored:
        if score < config.LOCAL_MIN_SCORE:
            continue
        meta = doc.metadata or {}
        name = meta.get("source_name") or "未知文档"
        chunk_index = meta.get("chunk_index", 0)
        text = (doc.page_content or "").strip()
        if not text:
            continue
        results.append({
            "id": new_id("L"),
            "query": query,
            "title": f"{name}（本地知识库 第 {chunk_index + 1} 段）",
            "url": f"{LOCAL_URL_PREFIX}{name}#chunk-{chunk_index}",
            "snippet": text[:500],
            "content": text,               # 自带正文，无需抓取
            "engine": "local",
            "score": round(score, 4),
            "sub_question_id": sub_question_id,
            "fetched": True,
        })
    return results


def kb_status(rebuild: bool = False) -> dict:
    """返回知识库状态，供 CLI / doctor / 前端展示"""
    files = list_source_files()
    if not files:
        return {"ready": False, "documents": 0, "chunks": 0,
                "message": f"未找到文档（{config.LOCAL_KB_DIR}）"}
    if rebuild:
        info = build_index(force=True)
    else:
        manifest = _read_manifest()
        info = {
            "documents": manifest.get("documents", len(files)),
            "chunks": manifest.get("chunks", 0),
            "message": "索引已就绪" if is_index_ready() else "索引尚未构建（首次检索时自动构建）",
        }
    return {
        "ready": is_index_ready(),
        "documents": info.get("documents", len(files)),
        "chunks": info.get("chunks", 0),
        "model": config.EMBEDDING_MODEL,
        "files": [p.name for p in files],
        "message": info.get("message", ""),
    }
