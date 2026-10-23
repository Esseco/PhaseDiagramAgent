from copy import deepcopy
from pathlib import Path

from phase_agent.runtime.chat_state_presentation import brief_chat_state, format_epoch_reply
from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.runtime.interaction_reports import save_interaction, short_reply
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.waiting_files_presentation import waiting_file_lines


def partial_state():
    return {
        "active_model_version": "m",
        "tasks": [
            {
                "task_id": f"mc-{i}",
                "stage": "deep_search",
                "status": "pending",
                "relax_mc_bundle_id": "b",
            }
            for i in range(5)
        ],
        "relax_mc_batches": {
            "b": {
                "model_version": "m",
                "import_checksum": "verified",
                "result_summary": {"status": "failed", "reason": "squeue --jobs 3743912 failed"},
            }
        },
    }


def test_status_receipt_keeps_missing_evidence_cause_and_owner(tmp_path):
    state = partial_state()
    before = deepcopy(state)
    full = "\n".join([brief_chat_state(state), *waiting_file_lines(state)])
    display = format_epoch_reply(format_turn_reply(concise_body(full, state), state), state)
    receipt = save_interaction(tmp_path, "现在什么状态", full, display, state)
    assert "5 项结果待核验" in receipt
    assert "仍缺 5 项结果的核验凭据" in receipt
    assert "Slurm 队列查询失败" in receipt and "不能证明计算失败" in receipt
    assert "请回传原作业的 sacct 状态" in receipt and "Agent 据此核对这 5 项" in receipt
    assert "已检查当前项目状态" not in receipt and "本次未提供新的计算预算" not in receipt
    assert "[本轮交互记录]" not in receipt
    assert len(receipt.splitlines()) <= 5
    assert all(line.endswith("  ") for line in receipt.splitlines()[:-1])
    report = Path(receipt.split("交互记录地址：", 1)[1])
    assert report.is_file() and "squeue --jobs 3743912 failed" in report.read_text(encoding="utf-8")
    assert state == before


def test_result_wins_over_generic_this_turn_and_approval_keeps_budget():
    reply = short_reply(
        "本次：已读取状态。\n结果：仍缺5项凭据。\n原因：日志未回传。\n下一步：请回传日志。"
    )
    assert "结果：仍缺5项凭据" in reply and "已读取状态" not in reply
    approval = short_reply(
        "结果：方案待审。\n下一步：核对后回复同意。确认范围：只准备输入。\n预算：2个GPU作业。"
    )
    assert "同意" in approval and "只准备输入" in approval and "2个GPU" in approval


def test_plain_answer_and_input_handoff_survive_shortening():
    assert "参数表示采样次数" in short_reply("当前项目 · epoch1 · 配置修订中\n参数表示采样次数。")
    text = "当前项目 · epoch1 · 等待提交\n上传整个文件夹：`E:/input`\n提交：`sbatch submit_all.sh`\n回传：`E:/output`"
    reply = short_reply(text)
    assert "E:/input" in reply and "sbatch submit_all.sh" in reply and "E:/output" in reply


def test_state_query_has_no_invented_action_or_budget():
    reply = concise_body("当前：等待回传。", {})
    assert reply == "当前：等待回传。"
    assert "预算：3个GPU作业。" in concise_body("结果：方案待审。\n预算：3个GPU作业。", {})


def test_unimported_results_are_not_called_verified():
    state = partial_state()
    state["relax_mc_batches"]["b"].pop("import_checksum")
    assert "结果待核验" not in brief_chat_state(state)
