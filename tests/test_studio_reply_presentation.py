from phase_agent.runtime.studio_reply_presentation import format_turn_reply


def test_initial_status_has_advice():
    text = format_turn_reply("当前：无待运行任务，等待下一步指令。\n下一步：说继续生成方案。", {})
    assert text.splitlines() == ["当前状态：尚未开始搜索。", "下一步：说继续生成方案。"]


def test_approval_details_and_limits_are_preserved():
    state = {"pending_execution_policies": {"p": {}}}
    text = format_turn_reply("下一步：生成初始结构\n最多12个；成本上限726。\n回复同意执行。", state)
    assert text.startswith("当前状态：")
    assert text.splitlines()[1] == "下一步：生成初始结构；回复同意执行。"
    assert "成本上限726" in text and "同意" in text


def test_short_receipt_keeps_explicit_approval_instruction():
    from phase_agent.runtime.interaction_reports import short_reply

    state = {"pending_execution_policies": {"p": {}}}
    for approval in ("回复“同意”执行，或提出修改。", "回复‘同意’执行，或提出修改。"):
        rendered = format_turn_reply("建议：准备计算输入文件\n" + approval, state)
        assert "下一步：准备计算输入文件；" + approval in short_reply(rendered)


def test_blocked_plan_overrides_old_approval_instruction(monkeypatch):
    from phase_agent.runtime.interaction_reports import short_reply

    monkeypatch.setattr(
        "phase_agent.runtime.plan_queries.pending_plan_reply",
        lambda state: "暂不可批准：旧任务未核对。\n下一步：先核对执行回执。",
    )
    state = {"pending_execution_policies": {"p": {}}}
    rendered = format_turn_reply("建议：准备计算输入文件\n回复“同意”执行。", state)
    assert "先核对执行回执" in short_reply(rendered)
    assert "同意" not in rendered


def test_failure_is_not_reported_as_success():
    text = format_turn_reply("本轮未执行：缺少母结构", {})
    assert "下一步：先处理上述问题" in text
    assert "本轮未执行：缺少母结构" in text


def test_configuring_is_not_search_started():
    text = format_turn_reply("请设置母结构", {}, configuring=True)
    assert "配置修订中" in text
    assert "下一步：" in text


def test_recovery_gate_is_not_overlaid_by_saved_plan():
    from phase_agent.runtime.chat_state_presentation import format_epoch_reply

    gate = (
        "检测到可回收的返回结果或待核对任务；本轮尚未回收、未入账、未分析。"
        "请明确回复‘同意回收’执行回收，或回复‘拒绝回收’保持当前状态。"
    )
    state = {"pending_execution_policies": {"stale": {}}}
    rendered = format_epoch_reply(format_turn_reply(gate, state), state)
    assert rendered == gate
    assert "输入准备方案" not in rendered
