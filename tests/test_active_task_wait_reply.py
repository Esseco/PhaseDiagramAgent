from phase_agent.graphs.dialogue.saved_control import active_task_wait_reply


def test_active_wait_reply_lists_only_current_dft_paths(monkeypatch):
    monkeypatch.setattr(
        "phase_agent.runtime.chat_state_presentation.brief_chat_state",
        lambda facts: "当前：DFT 运行中。",
    )
    facts = {
        "active_model_version": "model-new",
        "tasks": [{"task_id": "DFT-1", "stage": "dft_single_point", "status": "pending"}],
        "relax_mc_batches": {
            "relax-mc-1": {
                "model_version": "model-new",
                "status": "completed",
                "upload_folder": r"E:\project\RelaxMC_input",
                "script_path": r"E:\project\RelaxMC_input\submit_all.sh",
                "return_directory": r"E:\project\RelaxMC_return",
            }
        },
        "action_records": [
            {
                "status": "prepared",
                "final_action": {"tool": "select_dft_candidates"},
            }
        ],
        "slurm_batches": [
            {
                "task_ids": ["DFT-1"],
                "handoff": {
                    "folder": r"E:\project\DFT_input",
                    "submit_script": r"E:\project\DFT_input\submit_all.sh",
                    "return_directory": r"E:\project\DFT_return",
                },
            }
        ],
    }

    reply = active_task_wait_reply(facts)

    assert "DFT 输入地址：`E:\\project\\DFT_input`" in reply
    assert "DFT 回传地址：`E:\\project\\DFT_return`" in reply
    assert "Relax+MC 输入地址" not in reply


def test_active_wait_reply_lists_only_current_relax_mc_paths(monkeypatch):
    monkeypatch.setattr(
        "phase_agent.runtime.chat_state_presentation.brief_chat_state",
        lambda facts: "当前：Relax+MC 运行中。",
    )
    facts = {
        "active_model_version": "model-new",
        "tasks": [
            {
                "task_id": "MC-1",
                "stage": "deep_search",
                "status": "pending",
                "relax_mc_bundle_id": "relax-mc-1",
            }
        ],
        "relax_mc_batches": {
            "relax-mc-1": {
                "model_version": "model-new",
                "status": "prepared",
                "upload_folder": r"E:\project\RelaxMC_input",
                "script_path": r"E:\project\RelaxMC_input\submit_all.sh",
                "return_directory": r"E:\project\RelaxMC_return",
            }
        },
        "slurm_batches": [],
    }

    reply = active_task_wait_reply(facts)

    assert "Relax+MC 输入地址：`E:\\project\\RelaxMC_input`" in reply
    assert "Relax+MC 提交脚本：`E:\\project\\RelaxMC_input\\submit_all.sh`" in reply
    assert "Relax+MC 回收地址：`E:\\project\\RelaxMC_return`" in reply
    assert "DFT 输入地址" not in reply
