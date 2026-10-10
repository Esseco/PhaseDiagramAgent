from unittest.mock import patch

from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply, _compact_post_dft_reply


ASSESSMENT = "\n".join([
    "结果回收与分析：epoch0（mace-mh-1）",
    "本轮 DFT：回收 31/34，合格配对 27。",
    "成功 31、失败 0；未回传且不再等待 3。",
    "原模型：能量 MAE/RMSE 9.6777/11.208 meV/atom；受力 119.76/192.17 meV/Å。",
    "数据：`synthetic/training`。",
    "受力检查提醒：尺度小于10%，原因未确定。",
    "DFT 相图：`synthetic/dft.csv`。",
    "DFT 校正综合相图：partial；未校正结构 201。 CSV：`synthetic/combined.csv`。",
])
REVIEW = "\n".join([
    "本轮总结：", "首选建议：微调。", "本轮发现：稳定相6条。",
    "微调取舍：建议微调，须核查同帧接口。", "补DFT取舍：暂缓。",
    "新branch取舍：暂缓。", "收敛与停止：未收敛。",
    "数据局限：独立测试归属未确认。",
])


def test_compact_shows_metrics_and_action_without_full_report():
    text = _compact_post_dft_reply(ASSESSMENT, "磁矩：合格27，异常4。", REVIEW, "回复同意执行。")
    assert "31/34" in text and "9.6777/11.208" in text
    assert "回复同意执行。" in text
    assert "synthetic/" not in text and "<details>" not in text
    assert len(text.splitlines()) <= 6


def test_confirmation_reports_status_without_repeating_science():
    result = {"status": "confirmation_required", "reason": "synthetic approval gate"}
    with patch("phase_agent.runtime.post_dft_presentation.post_dft_lines", side_effect=AssertionError("must not repeat")):
        text = format_workflow_reply(result, "synthetic/state.json")
    assert "仍需确认" in text and "synthetic approval gate" in text
    assert "synthetic/state.json" in text and "本轮总结" not in text


def test_followup_failure_preserves_actual_reason():
    text = format_workflow_reply({"status": "failed", "reason": "synthetic missing structure"}, "synthetic/state.json")
    assert text == "本轮失败：synthetic missing structure"
    assert "本轮总结" not in text


def test_verbose_keeps_full_plain_report():
    with patch("phase_agent.runtime.post_dft_presentation.post_dft_lines", return_value=ASSESSMENT),\
         patch("phase_agent.runtime.post_dft_presentation.post_dft_review_lines", return_value=REVIEW):
        text = format_workflow_reply({"status": "completed"}, "synthetic/state.json", verbose=True)
    assert ASSESSMENT in text and REVIEW in text
    assert "<details>" not in text
