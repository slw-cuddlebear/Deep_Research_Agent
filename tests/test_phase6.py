"""
Phase 6 自检脚本：用 Streamlit 官方的 AppTest 真跑一遍前端。
运行方式：
    python tests/test_phase6.py

为什么用 AppTest：它会把 app.py 当成真实 Streamlit 脚本执行一遍，
可以模拟"点按钮 / 填输入框"，并检查页面上的报错、指标、下载按钮。
比"能 import 就算过"强得多。

覆盖内容：
    1. 页面能正常渲染，无异常
    2. 侧边栏设置项齐全
    3. 示例问题选择能写入输入框
    4. 离线模式下能生成研究计划，且计划进入可编辑表格
    5. 离线模式下能跑完整研究，页面出现报告与下载按钮
    6. 缺 Key 时真实模式给出明确提示而不是崩溃
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASSED, FAILED = [], []

APP = os.path.join(BASE_DIR, "app.py")


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


def fresh_app(timeout: int = 120):
    """启动一个干净的 AppTest 实例"""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=timeout)
    return app


def render_text(app) -> str:
    """把页面上所有 markdown / 标题文本拼起来，便于断言"""
    chunks = []
    for collection in ("markdown", "subheader", "header", "caption", "info", "success", "warning"):
        try:
            for element in getattr(app, collection):
                chunks.append(str(getattr(element, "value", "") or ""))
        except Exception:
            pass
    return "\n".join(chunks)


# ========== 1. 基础渲染 ==========
def test_render() -> None:
    print("\n[1] 页面基础渲染")
    app = fresh_app().run()

    check("脚本无异常", not app.exception, str([e.value for e in app.exception][:2]))
    check("没有 error 提示", len(app.error) == 0, str([e.value for e in app.error][:2]))

    text = render_text(app)
    check("页面显示标题", "Deep Research Agent" in text, text[:120])
    check("存在研究问题输入框", len(app.text_area) >= 1, str(len(app.text_area)))
    check("存在研究问题标签", "研究问题" in text, text[:200])


# ========== 2. 侧边栏设置 ==========
def test_sidebar() -> None:
    print("\n[2] 侧边栏设置项")
    app = fresh_app().run()

    radio_labels = [r.label for r in app.radio]
    check("有运行模式切换（真实/离线）", any("运行模式" in label for label in radio_labels),
          str(radio_labels))

    slider_labels = [s.label for s in app.slider]
    check("有最大搜索轮数滑块", any("最大搜索轮数" in label for label in slider_labels),
          str(slider_labels))
    check("有单轮阅读数滑块", any("阅读网页数" in label for label in slider_labels),
          str(slider_labels))

    selectbox_labels = [s.label for s in app.selectbox]
    check("有搜索后端选择", any("搜索后端" in label for label in selectbox_labels),
          str(selectbox_labels))

    checkbox_labels = [c.label for c in app.checkbox]
    check("有本地知识库开关", any("本地知识库" in label for label in checkbox_labels),
          str(checkbox_labels))

    text = render_text(app)
    check("侧边栏展示环境状态", "DeepSeek Key" in text, text[:200])
    check("侧边栏展示本地知识库状态", "本地知识库" in text)

    check("有导出格式选项",
          sum(1 for c in app.checkbox if "HTML" in c.label or "PDF" in c.label) >= 2,
          str([c.label for c in app.checkbox]))


# ========== 3. 示例问题 ==========
def test_example_pick() -> None:
    print("\n[3] 示例问题选择")
    app = fresh_app().run()

    options = app.selectbox[0].options if app.selectbox else []
    check("示例问题下拉框存在", len(options) >= 4, str(options[:2]))

    app.selectbox[0].select(options[1]).run()
    check("选中示例后写入输入框",
          options[1] in app.text_area[0].value,
          f"输入框内容：{app.text_area[0].value[:60]}")


# ========== 4. 离线生成计划 ==========
def _switch_to_offline(app):
    """把运行模式切到离线演示"""
    radio = next(r for r in app.radio if "运行模式" in r.label)
    radio.set_value("离线演示（假数据）")
    return app.run()


def test_offline_plan() -> None:
    print("\n[4] 离线模式：生成研究计划")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=180).run()
    app = _switch_to_offline(app)
    check("切换到离线模式无异常", not app.exception, str([e.value for e in app.exception][:2]))

    app.text_area[0].input("离线自检：前端能否生成研究计划？")
    app.run()

    plan_button = next((b for b in app.button if "生成研究计划" in b.label), None)
    check("存在「生成研究计划」按钮", plan_button is not None,
          str([b.label for b in app.button]))

    if plan_button:
        plan_button.click().run()
        check("生成计划无异常", not app.exception, str([e.value for e in app.exception][:2]))
        plan = app.session_state.get("plan") or []
        check("计划写入会话状态", len(plan) >= 3, f"实际 {len(plan)} 个子问题")
        check("计划含子问题文本",
              all(item.get("question") for item in plan), str(plan)[:200])
        check("计划含检索词",
              all(item.get("search_queries") for item in plan), str(plan)[:200])
        check("页面显示研究计划区", "研究计划" in render_text(app))


# ========== 5. 离线跑完整研究 ==========
def test_offline_run() -> None:
    print("\n[5] 离线模式：完整研究 + 报告与下载")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=300).run()
    app = _switch_to_offline(app)
    app.text_area[0].input("离线自检：前端能否跑完整研究并给出报告？")
    app.run()

    run_button = next((b for b in app.button if "开始研究" in b.label), None)
    check("存在「开始研究」按钮", run_button is not None, str([b.label for b in app.button]))
    if not run_button:
        return

    run_button.click().run()

    check("运行无异常", not app.exception, str([e.value for e in app.exception][:2]))
    state = app.session_state.get("final_state")
    check("产出最终状态", bool(state), "final_state 为空")

    if state:
        check("产出报告 Markdown", bool(state.get("report_markdown")),
              f"{len(state.get('report_markdown') or '')} 字符")
        check("报告含参考文献", "## 参考文献" in (state.get("report_markdown") or ""))
        check("产出事实清单", len(state.get("facts") or []) > 0,
              str(len(state.get("facts") or [])))
        check("产出反思记录", len(state.get("reflections") or []) > 0,
              str(len(state.get("reflections") or [])))
        check("产出事件流", len(state.get("events") or []) >= 3,
              str(len(state.get("events") or [])))

    # 指标卡片
    metric_labels = [m.label for m in app.metric]
    check("页面显示运行指标卡片", len(metric_labels) >= 8, str(metric_labels[:6]))
    check("指标含事实数", any("事实数" in label for label in metric_labels), str(metric_labels))
    check("指标含反思轮数", any("反思" in label for label in metric_labels), str(metric_labels))

    # 下载按钮
    downloads = [d.label for d in app.download_button]
    check("提供 Markdown 下载", any("Markdown" in label for label in downloads), str(downloads))
    check("提供 HTML 下载", any("HTML" in label for label in downloads), str(downloads))
    check("提供 PDF 下载", any("PDF" in label for label in downloads), str(downloads))

    # 分页签
    tab_labels = [t.label for t in app.tabs] if hasattr(app, "tabs") else []
    text = render_text(app)
    check("页面出现报告内容", "参考文献" in text or "摘要" in text, text[-200:])


# ========== 6. 缺 Key 提示 ==========
def test_missing_key_hint() -> None:
    print("\n[6] 真实模式缺 Key 时的提示")
    import config
    from streamlit.testing.v1 import AppTest

    original_has_key = config.has_llm_key
    config.has_llm_key = lambda: False
    try:
        app = AppTest.from_file(APP, default_timeout=180).run()
        app.text_area[0].input("这个研究问题在没有 Key 时应该给出提示")
        app.run()
        run_button = next((b for b in app.button if "开始研究" in b.label), None)
        if run_button:
            run_button.click().run()
        errors = [e.value for e in app.error]
        check("缺 Key 时给出明确报错而不是崩溃", any("DEEPSEEK_API_KEY" in str(e) for e in errors),
              str(errors[:2]))
        check("缺 Key 时不产生最终状态",
              not app.session_state.get("final_state"), "竟然产出了报告")
    finally:
        config.has_llm_key = original_has_key


def main() -> int:
    print("=" * 60)
    print("Phase 6 自检：Streamlit 前端")
    print("=" * 60)
    test_render()
    test_sidebar()
    test_example_pick()
    test_offline_plan()
    test_offline_run()
    test_missing_key_hint()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
