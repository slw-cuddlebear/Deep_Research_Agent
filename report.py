"""
报告导出模块：Markdown → HTML → PDF。
对应规划里的"支持导出 PDF / Markdown"。

设计取舍：
  - PDF 采用 xhtml2pdf（纯 Python），无需 GTK / 浏览器内核，装上就能用。
  - 中文字体**运行时自动探测**系统字体（Windows/macOS/Linux 各列了候选），
    探测不到时**跳过 PDF 并把原因说清楚**，而不是抛异常或输出一堆方块字。
  - xhtml2pdf 只支持 CSS 2.1 子集，所以样式写得比较克制。
"""
import logging
import re
import sys
from pathlib import Path

import config

# ========== 中文字体探测 ==========
# (路径, ttc 子字体序号)；.ttf 优先，因为 reportlab 对 ttc 的处理依赖子字体索引
FONT_CANDIDATES = [
    # Windows
    ("C:/Windows/Fonts/Deng.ttf", 0),          # 等线
    ("C:/Windows/Fonts/simhei.ttf", 0),        # 黑体
    ("C:/Windows/Fonts/msyh.ttc", 0),          # 微软雅黑
    ("C:/Windows/Fonts/simsun.ttc", 0),        # 宋体
    # macOS
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0),
    # Linux（Docker 镜像里会装 Noto CJK）
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
]

PDF_FONT_NAME = "ReportCJK"
# 仅用于验证字体文件能否被 reportlab 解析，不参与 PDF 渲染
_PROBE_FONT_NAME = "_CJKProbe"

_font_cache = {"resolved": False, "path": None, "index": 0, "error": ""}


def find_cjk_font() -> tuple:
    """
    探测可用的中文字体文件，并用 reportlab 试解析一次确认它真的能用。
    :return: (字体路径, 说明) —— 找不到时路径为 None
    """
    if _font_cache["resolved"]:
        return _font_cache["path"], _font_cache["error"]

    _font_cache["resolved"] = True
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:
        _font_cache["error"] = f"reportlab 不可用：{exc}"
        return None, _font_cache["error"]

    tried = []
    for path, index in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            # 试解析：能建出 TTFont 对象才说明这个文件真的可用于渲染
            pdfmetrics.registerFont(TTFont(_PROBE_FONT_NAME, path, subfontIndex=index))
            _font_cache.update({"path": path, "index": index})
            return path, ""
        except Exception as exc:
            tried.append(f"{path}: {type(exc).__name__}")

    _font_cache["error"] = (
        "未找到可用于 PDF 的中文字体，已跳过 PDF 导出（Markdown / HTML 不受影响）。"
        + (f" 试解析失败：{'; '.join(tried[:3])}" if tried else "")
        + " 如需 PDF：Windows/macOS 自带中文字体，Linux 请安装 fonts-wqy-zenhei 或 fonts-noto-cjk。"
    )
    return None, _font_cache["error"]


def _font_uri(font_path: str) -> str:
    """把本地字体路径转成 xhtml2pdf 能识别的 file:// URI"""
    return Path(font_path).resolve().as_uri()


# ========== Markdown → HTML ==========
_FONT_FACE_BLOCK = """  @font-face {{
    font-family: "{family}";
    src: url("{uri}");
  }}
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
{font_face}  @page {{ size: A4; margin: 2cm 1.8cm; }}
  /* 全局兜底：连 <title> 这类不在 body 内的元素也能拿到中文字体，
     否则 xhtml2pdf 会对标题里的汉字报"缺字"并退回 Helvetica。 */
  * {{ font-family: "{font}", "Helvetica"; }}
  body {{ font-family: "{font}", "Helvetica"; font-size: 10.5pt; line-height: 1.65; color: #1a1a1a; }}
  h1 {{ font-size: 19pt; color: #111; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px; }}
  h2 {{ font-size: 14pt; color: #2b6cb0; margin-top: 18px; }}
  h3 {{ font-size: 12pt; color: #2c5282; }}
  p {{ margin: 6px 0; text-align: justify; }}
  ul, ol {{ margin: 6px 0 6px 18px; }}
  li {{ margin: 3px 0; }}
  blockquote {{ margin: 8px 0; padding: 6px 10px; background: #f5f8fc;
                border-left: 3px solid #90b4de; color: #333; font-size: 9.5pt; }}
  code {{ font-family: "Courier New"; background: #f4f4f4; font-size: 9.5pt; }}
  pre {{ background: #f6f8fa; padding: 8px; font-size: 9pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #cbd5e0; padding: 5px 7px; }}
  th {{ background: #eef2f7; }}
  hr {{ border: none; border-top: 1px solid #cbd5e0; margin: 14px 0; }}
  a {{ color: #2b6cb0; word-wrap: break-word; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_html(markdown_text: str, title: str = "研究报告",
                     font_path: str = None, font_family: str = None,
                     for_pdf: bool = False) -> str:
    """
    把 Markdown 全文转成带样式的 HTML。

    :param font_path: 中文字体文件路径；给了就内嵌 @font-face（xhtml2pdf 只认这种方式）
    :param for_pdf: 是否用于 PDF 渲染。为 True 时会**手工给中文断行**，原因见下。

    为什么 PDF 需要手工断行：
        xhtml2pdf 0.2.20 **不会对中文自动折行**（它靠空格断行，而中文没有空格），
        它自带的 CJK 开关 `-pdf-word-wrap: CJK` 又存在 bytes/str 混用的 Bug，
        一开启就抛 TypeError。实测：108 字的中文段落会被画成一整行冲出页面右边界，
        而同样长度的英文段落能正常折行。
        因此这里按字号估算每行可容纳的字数，主动插入 `<br/>`。
        浏览器与 Markdown 导出不受影响——浏览器本身就会给中文折行。
    """
    try:
        import markdown as md_lib

        body = md_lib.markdown(
            markdown_text,
            extensions=["extra", "tables", "fenced_code", "sane_lists", "nl2br"],
        )
    except ImportError:
        # markdown 库不可用时退化为 <pre>，至少 HTML 还能看
        import html as html_lib

        body = f"<pre>{html_lib.escape(markdown_text)}</pre>"

    if for_pdf:
        body = wrap_cjk_for_pdf(body)

    font_face = ""
    family = font_family or "sans-serif"
    if font_path:
        font_face = _FONT_FACE_BLOCK.format(family=font_family or PDF_FONT_NAME,
                                            uri=_font_uri(font_path))
    return _HTML_TEMPLATE.format(
        title=title, body=body, font=family, font_face=font_face
    )


# ========== 中文手工断行（xhtml2pdf 不支持 CJK 自动折行）==========
# A4 宽 21cm，左右各 1.8cm 页边距 → 正文可用宽度约 493pt
_FRAME_WIDTH_PT = 493.0
# 各标签字号（pt），与 _HTML_TEMPLATE 里的 CSS 保持一致
_TAG_FONT_PT = {
    "h1": 19.0, "h2": 14.0, "h3": 12.0, "h4": 10.5, "h5": 10.5, "h6": 10.5,
    "p": 10.5, "li": 10.5, "blockquote": 9.5, "td": 9.5, "th": 9.5, "dd": 10.5,
}
# 这些标签自带缩进或只是表格单元格，可用宽度要打折
_TAG_WIDTH_FACTOR = {"li": 0.93, "blockquote": 0.90, "td": 0.45, "th": 0.45}
# 安全余量：估算宽度与渲染器真实字宽有偏差，留 6% 富余避免个别行仍然溢出
_WIDTH_SAFETY = 0.94
_BLOCK_TAGS = tuple(_TAG_FONT_PT)
# 适合作为断点的中文标点
_CJK_PUNCT = "，。；：！？、）】》」』…"
# 不宜出现在行首的标点：断行时把它们留在上一行
_NO_LINE_START = "，。；：！？、）】》」』…％%"


def _char_width(ch: str) -> float:
    """按字符估算宽度（以"一个全角汉字"为 1 个单位）"""
    code = ord(ch)
    if code > 0x2E80:          # CJK 汉字与全角标点
        return 1.0
    if ch == "\t":
        return 2.0
    return 0.5                  # 半角字母数字与符号


def _break_cjk_text(text: str, max_units: float) -> list:
    """
    把一段文本按 max_units（以全角字为单位）切成多行。
    断点优先级：中文标点后 > 任意汉字后 > 空格后；**绝不在英文单词中间断开**。
    """
    if max_units <= 2 or not text:
        return [text]

    lines, current, width = [], [], 0.0
    for ch in text:
        current.append(ch)
        width += _char_width(ch)
        if width < max_units:
            continue

        cut = len(current)
        # 优先断在中文标点之后（且保证标点不落到下一行行首）
        for back in (1, 2):
            if cut - back <= 0:
                break
            prev = current[cut - back - 1]
            nxt = current[cut - back] if cut - back < len(current) else ""
            if prev in _CJK_PUNCT and nxt not in _NO_LINE_START:
                cut -= back
                break
        else:
            # 回退到最近的"汉字后"或"空格后"，避免劈开英文单词
            for idx in range(len(current) - 1, max(len(current) - 20, 1), -1):
                prev = current[idx - 1]
                if prev == " " or _char_width(prev) == 1.0:
                    cut = idx
                    break

        lines.append("".join(current[:cut]).rstrip())
        current = current[cut:]
        width = sum(_char_width(c) for c in current)

    if current:
        lines.append("".join(current).rstrip())
    return [ln for ln in lines if ln] or [text]


def wrap_cjk_for_pdf(html: str) -> str:
    """
    给 HTML 里以中文为主的块级文本主动插入 `<br/>`，让 xhtml2pdf 正确折行。
    已存在的 `<br/>` 视为硬换行（每段独立计算宽度）。
    """
    try:
        from bs4 import BeautifulSoup
        from bs4.element import NavigableString
    except ImportError:
        return html

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return html

    changed = False
    for tag_name in _BLOCK_TAGS:
        for element in soup.find_all(tag_name):
            font_pt = _TAG_FONT_PT.get(tag_name, 10.5)
            max_units = (_FRAME_WIDTH_PT * _TAG_WIDTH_FACTOR.get(tag_name, 1.0)
                         * _WIDTH_SAFETY) / font_pt

            for node in list(element.find_all(string=True)):
                if not isinstance(node, NavigableString):
                    continue
                text = str(node)
                if len(text) < 12:
                    continue
                # 只有中文占多数时才手工断行，英文段落交给引擎自己折（它能正常折）
                cjk = sum(1 for c in text if ord(c) > 0x2E80)
                if cjk / max(len(text), 1) < 0.5:
                    continue

                lines = _break_cjk_text(text, max_units)
                if len(lines) <= 1:
                    continue

                parent = node.parent
                if parent is None:
                    continue
                # 在文本节点位置依次插入 行文本 + <br/>
                cursor = node
                for index, line in enumerate(lines):
                    if index:
                        br = soup.new_tag("br")
                        cursor.insert_after(br)
                        cursor = br
                    piece = NavigableString(line)
                    cursor.insert_after(piece)
                    cursor = piece
                node.extract()
                changed = True

    return str(soup) if changed else html


# ========== HTML → PDF ==========
# xhtml2pdf 在字体没生效时只会往日志里写一句 warning，PDF 照样生成、
# 中文全变方框。这里把它的日志抓下来，把"静默降级"变成"明确失败"。
#
# 注意：只把"字体族根本没解析出来"的日志当致命错误。
# "No font this document has can draw 'X'" 这类**逐字缺字**告警不作数——
# 实测发现它会被 <title> 里的汉字触发，而正文其实渲染得好好的；
# 正文是否真的有中文，交给后面 pypdf 抽文本校验来判定，那才是硬证据。
_FONT_FAILURE_MARKERS = (
    "could not read",
    "is used instead",
    "not known to this document",
)


class _WarningCollector(logging.Handler):
    """收集渲染期间 xhtml2pdf 抛出的 warning"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:  # 日志格式化失败不能影响渲染
            pass


def _build_resource_policy(font_path: str):
    """
    xhtml2pdf 0.2.20 起引入了资源沙箱：文档只能读取"自己的目录"，因此
    C:\\Windows\\Fonts 下的中文字体会被判为越权读取而拒绝（现象就是中文全变方框）。

    这里按最小授权原则，只额外把**字体所在目录**加进白名单，
    保留默认的 SSRF 防护（不允许访问内网地址），而不是图省事放开全部限制。
    """
    try:
        from xhtml2pdf.config.resources import ResourceAccessPolicy
    except ImportError:  # 老版本没有沙箱，直接返回 None
        return None
    try:
        return ResourceAccessPolicy(
            extra_roots=(Path(font_path).resolve().parent,),
            max_resource_bytes=64 * 1024 * 1024,   # 中文字体动辄 20MB，默认 20MB 上限偏紧
        )
    except Exception:
        return None


def html_to_pdf(html: str, out_path) -> tuple:
    """
    用 xhtml2pdf 渲染 PDF，并校验中文字体确实生效。
    :return: (是否成功, 成功=文件路径 / 失败=原因)
    """
    font_path, font_error = find_cjk_font()
    if not font_path:
        return False, font_error

    try:
        from xhtml2pdf import pisa
    except ImportError as exc:
        return False, f"xhtml2pdf 不可用，已跳过 PDF 导出：{exc}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collector = _WarningCollector()
    xhtml_logger = logging.getLogger("xhtml2pdf")
    root_logger = logging.getLogger()
    xhtml_logger.addHandler(collector)
    root_logger.addHandler(collector)
    try:
        policy = _build_resource_policy(font_path)
        with open(out_path, "wb") as handle:
            result = pisa.CreatePDF(html, dest=handle, encoding="utf-8",
                                    resource_policy=policy)
    except TypeError:
        # 极端情况下该版本不接受 resource_policy 参数，退回不带策略的调用
        try:
            with open(out_path, "wb") as handle:
                result = pisa.CreatePDF(html, dest=handle, encoding="utf-8")
        except Exception as exc:
            return False, f"PDF 渲染失败：{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"PDF 渲染失败：{type(exc).__name__}: {exc}"
    finally:
        xhtml_logger.removeHandler(collector)
        root_logger.removeHandler(collector)

    if getattr(result, "err", 0):
        return False, f"PDF 渲染过程中出现 {result.err} 个错误：{out_path}"

    font_problems = [m for m in collector.messages
                     if any(marker in m for marker in _FONT_FAILURE_MARKERS)]
    if font_problems:
        return False, ("中文字体未能嵌入 PDF（中文会显示为空白或方框），已删除该文件："
                       + font_problems[0][:160])

    # 最后一道保险：把 PDF 里的文字抽回来，确认中文真的写进去了
    missing = _verify_cjk_text(out_path, html)
    if missing:
        try:
            out_path.unlink()
        except OSError:
            pass
        return False, f"PDF 中未能检索到中文内容（缺失示例：{missing}），已删除该文件"

    return True, str(out_path)


def _verify_cjk_text(pdf_path, html: str) -> str:
    """
    从生成好的 PDF 里抽文字，确认正文中文确实写进去了。
    返回空字符串表示校验通过，否则返回第一个没找到的中文片段。

    抽样时会**剔除 <title> 标签的内容**——标题走的是元数据路径，
    不能拿它来证明正文渲染正常。

    判定口径：均匀抽取正文里的中文片段，要求**绝大多数**（>75%）能在 PDF 文本层找到。
    早期版本只要求"任意一个片段命中"，抽 20 个样本只看第一个字面命中就放过，
    遇到字体只覆盖部分汉字（其余渲染成方框）时会漏判。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""  # 没有 pypdf 就不校验，不阻断导出

    try:
        reader = PdfReader(str(pdf_path))
        extracted = "".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""

    if not extracted.strip():
        return ""

    # 只取**正文可见文本**作样本：<style>/<script> 里的 CSS 注释也含大量汉字，
    # 但它们不会出现在 PDF 文本层里，拿来当样本会造成误报。
    body_html = re.sub(r"<title>.*?</title>", " ", html or "", flags=re.S)
    body_html = re.sub(r"<(style|script)\b.*?</\1>", " ", body_html, flags=re.S | re.I)
    samples = re.findall(r"[\u4e00-\u9fa5]{3,}", _strip_html(body_html))
    if not samples:
        return ""

    # 去重后均匀取样：涵盖报告首尾，避免只看开头的几段
    unique = list(dict.fromkeys(samples))
    if len(unique) > 20:
        step = len(unique) / 20
        unique = [unique[int(i * step)] for i in range(20)]

    flat = extracted.replace("\n", "")
    missing = [s for s in unique if s[:3] not in flat]
    # 允许个别片段因为 PDF 文本抽取的换行/空格差异而没命中；
    # 但只要缺失比例超过 1/4，就说明字体覆盖不全（部分汉字会渲染成方框）。
    if len(missing) > max(1, len(unique) // 4):
        return missing[0][:12]
    return ""


def _strip_html(html: str) -> str:
    """粗粒度去标签，用于提取中文样本"""
    return re.sub(r"<[^>]+>", " ", html or "")


# ========== 统一导出入口 ==========
def export_report(markdown_text: str, out_dir, stem: str = "report",
                  formats: tuple = ("md", "html", "pdf")) -> dict:
    """
    导出报告，返回 {"paths": {格式: 路径}, "warnings": [提示], "font": 字体路径}。
    任何一步失败都不会抛异常，只记进 warnings —— 报告本身不能因为导出失败而丢失。
    """
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, warnings = {}, []
    title = _extract_title(markdown_text) or "研究报告"

    if "md" in formats:
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        paths["md"] = str(md_path)

    font_path, font_error = find_cjk_font() if "pdf" in formats else (None, "")

    if "html" in formats:
        html_path = out_dir / f"{stem}.html"
        html = markdown_to_html(markdown_text, title=title, font_path=font_path,
                               font_family=PDF_FONT_NAME if font_path else None)
        html_path.write_text(html, encoding="utf-8")
        paths["html"] = str(html_path)

    if "pdf" in formats:
        if not font_path:
            warnings.append(font_error)
        else:
            # for_pdf=True：给中文手工断行（xhtml2pdf 不会自动折 CJK）
            html = markdown_to_html(markdown_text, title=title, font_path=font_path,
                                    font_family=PDF_FONT_NAME, for_pdf=True)
            ok, message = html_to_pdf(html, out_dir / f"{stem}.pdf")
            if ok:
                paths["pdf"] = message
            else:
                warnings.append(message)

    return {"paths": paths, "warnings": warnings, "font": font_path}


def _extract_title(markdown_text: str) -> str:
    """从 Markdown 里取一级标题"""
    for line in (markdown_text or "").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def platform_hint() -> str:
    """给用户看的平台字体提示"""
    if sys.platform.startswith("win"):
        return "Windows：使用系统自带的中文字体（等线/黑体/微软雅黑）"
    if sys.platform == "darwin":
        return "macOS：使用系统自带的苹方/宋体"
    return "Linux：需自行安装中文字体，例如 apt-get install fonts-noto-cjk"
