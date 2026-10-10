from phase_agent.runtime.studio_reply_presentation import format_turn_reply


def test_initial_status_has_advice():
    text = format_turn_reply("当前：无待运行任务，等待下一步指令。\n下一步：说继续生成方案。", {})
    assert text.splitlines() == ["当前状态：尚未开始搜索。", "下一步：说继续生成方案。"]


def test_approval_details_and_limits_are_preserved():
    state = {"pending_execution_policies": {"p": {}}}
    text = format_turn_reply("下一步：生成初始结构\n最多12个；成本上限726。\n回复同意执行。", state)
    assert text.startswith("当前状态：")
    assert text.splitlines()[1] == "下一步：生成初始结构"
    assert "成本上限726" in text and "同意" in text


def test_failure_is_not_reported_as_success():
    text = format_turn_reply("本轮未执行：缺少母结构", {})
    assert "下一步：先处理上述问题" in text
    assert "本轮未执行：缺少母结构" in text


def test_configuring_is_not_search_started():
    text = format_turn_reply("请设置母结构", {}, configuring=True)
    assert "配置修订中" in text
    assert "下一步：" in text
