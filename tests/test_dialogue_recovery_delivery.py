"""Real handler/graphs/ledger/feedback/approval; only the LLM response is simulated."""

from pathlib import Path
from copy import deepcopy
from collections import Counter
import hashlib
import json
import shutil
import zipfile

import pytest

from tests.test_legacy_delivery_recovery import setup_original, relocated
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.analysis.feedback.dft_comparison_tables import (
    build_comparison_tables,
    comparison_metrics,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Deep real HPC batch names plus pytest's full test name exceed Windows
    # MAX_PATH during atomic extraction. Keep the private root short, as in
    # the existing actual-project delivery fixture; don't alter archive data.
    return tmp_path_factory.mktemp("p")


@pytest.mark.parametrize(
    "project,expected", [("O3-1ele-NaCoO2", 30), ("O3-1ele-NaCrO2", 12), ("O3-2ele-NaFeMn", 0)]
)
def test_true_dialogue_notice_analysis_proposal_approval_delivery(
    tmp_path, monkeypatch, project, expected
):
    root = Path("E:/0-PhaseDiagram") / project
    original = root / "workflow_state/state.json"
    ledger = root / "workflow_state/ledgers/phase_data.json"
    session_file = root / "parameters/config_session.json"
    protected = {str(p): digest(p) for p in [original, ledger, session_file]}
    saved = json.loads(original.read_text(encoding="utf-8"))
    local = tmp_path / "local"
    if expected:
        _, h = setup_original(tmp_path, project)
        for seq in ["02", "03", "04"]:
            for suffix in ["input", "output"]:
                p = (
                    Path("E:/0-PhaseDiagram")
                    / project
                    / "submissions/archives/01-upload"
                    / ("epoch0_DFTSP_Batch01_Recovery" + seq + "_" + suffix)
                )
                shutil.copytree(p, Path(h["folder"]).with_name(p.name))
    from tests.project_snapshot_files import copy_history_files

    print("Private history files", copy_history_files(saved, root, local), flush=True)
    state = relocated(saved, root, local)
    references = root / "structures/reference_structures"
    if references.is_dir():
        shutil.copytree(references, local / "structures/reference_structures", dirs_exist_ok=True)
    session = relocated(json.loads(session_file.read_text(encoding="utf-8")), root, local)
    config = session["confirmed_snapshot"]["config"]
    # Physical paths are private; scientific parameters and approval versions
    # are those in the actual confirmed project session.
    paths = {
        "state_path": local / "workflow_state/state.json",
        "ledger_path": local / "workflow_state/ledgers/phase_data.json",
        "phase_diagram_directory": local / "analysis",
        "branch_energy_pool_ledger_path": local / "workflow_state/ledgers/energy_pool.json",
        "upload_batches_directory": local / "submissions",
        "work_directory": local / "work",
        "approval_directory": local / "approvals",
        "local_action_directory": local / "approved_batches",
        "structure_directory": local / "structures",
    }
    run_config = {**config, **{k: str(v) for k, v in paths.items()}}
    state_path = paths["state_path"]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manager = PhaseDataManager.load(ledger)
    # Keep existing structure/reference paths readable. manager.save uses only
    # the explicitly supplied private ledger destination.
    manager.save(paths["ledger_path"])
    from phase_agent.configuration.session.phase_cache_location import phase_cache_location

    old_cache = phase_cache_location(original)
    new_cache = phase_cache_location(state_path)
    if old_cache.is_file():
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_cache, new_cache)
    refs = deepcopy((config.get("system") or {}).get("phase_references") or {})
    requests = []

    def client(payload):
        print(project, "model protocol", payload.get("mode"), flush=True)
        requests.append({"mode": payload.get("mode"), "keys": list(payload)})
        requests[-1]["context_keys"] = list((payload.get("decision_context") or {}))
        (tmp_path / "model-requests.json").write_text(
            json.dumps(requests, indent=2), encoding="utf-8"
        )
        if not expected:
            return {
                "kind": "answer",
                "answer": "当前没有新的合格DFT训练数据，本次不准备微调输入，也不提交计算。",
            }
        brief = (payload.get("decision_context") or {}).get("scientific_decision_brief")
        if not brief:
            if payload.get("mode") == "saved_work_control":
                return {"kind": "advance"}
            if "dialogue_outcome" in (payload.get("output_contracts") or {}) or (
                payload.get("decision_context") or {}
            ).get("conversation_controls"):
                return {"kind": "advance"}
            return {
                "kind": "inspect_evidence",
                "topics": ["configuration", "tasks", "model", "budget", "dft", "convergence"],
            }
        action = {
            "tool": "update_mlip",
            "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
            "target_ids": [],
            "budget": 0.0,
            "reason": "利用本批合格DFT标注准备微调输入。",
            "expected_purpose": "生成超算微调与交叉验证文件，等待用户手动提交。",
            "round_budget_review": {
                "choice": "finetune",
                "reason": "先使用已有合格标注，避免在模型误差较大时追加计算。",
                "uncertainty": "新增搜索或DFT的边际收益尚无可靠数值估计。",
                "revisit_when": "微调模型验证完成后重新比较。",
                "alternatives": [
                    {
                        "path": path,
                        "expected_benefit": "可能改善稳定相判断，但目前已有标注可先利用。",
                        "benefit_status": "hypothesis",
                        "expected_relative_cost": None,
                        "cost_basis": "未运行新增任务，无可靠成本估计；计入后续验证和训练成本。",
                    }
                    for path in ("supplement_dft", "new_search")
                ],
            },
            "scientific_decision": {
                "point": "round_direction",
                "config_version": brief["config_version"],
                "model_version": brief["model_version"],
                "bottleneck": "原模型误差尚高于目标。",
                "rationale": "优先利用现有合格DFT标注改进模型。",
                "alternatives": [
                    {"route": "继续搜索", "reason": "搜索可能有收益，但先利用已有标注更合理。"}
                ],
                "expected_outcome": "生成训练提交输入。",
                "success_criterion": "文件与数据校验通过，训练后另行评估。",
                "limitations": "只模拟本次模型决策，不推断实际新模型性能。",
            },
            "post_dft_review": {
                "choice": "finetune",
                "stop_status": "continue",
                "finetune_recommendation": "now",
                "error_assessment": "使用实际原模型同帧DFT误差，未达到停止目标。",
                "dft_assessment": "已有合格DFT标注，不重算原失败点。",
                "convergence_assessment": "不能据当前误差声明相图收敛。",
                "coverage_assessment": "沿用已记录结构与组成覆盖。",
                "finetune_assessment": "优先用已有标注准备微调输入，批准不代表执行训练。",
                "search_assessment": "暂缓进一步搜索，先利用现有数据。",
                "reference_assessment": "使用原冻结模型配对，未混用微调后指标。",
                "round_findings": "已有合格数据满足输入准备。",
                "limitations": "本次未真实训练或提交。",
            },
        }
        from phase_agent.decisions.agent.proposal_validation import proposal_errors

        assert not proposal_errors(action, payload)
        return action

    def forbidden(*a, **k):
        raise AssertionError("No real inference, training, Slurm or service subprocess")

    # Windows asyncio defines a Popen subclass at import time. Load the real
    # class definitions before forbidding process creation during this test.
    __import__("asyncio")
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    # Use the service's real on-disk runtime composition, including its draft.
    runtime_file = local / "agent_runtime.json"
    runtime_settings = relocated(
        json.loads((root / "agent_runtime.json").read_text(encoding="utf-8")), root, local
    )
    runtime_file.write_text(json.dumps(runtime_settings), encoding="utf-8")
    local_session = local / "parameters/config_session.json"
    local_session.parent.mkdir(parents=True, exist_ok=True)
    local_session.write_text(json.dumps(session), encoding="utf-8")
    editable = root / "parameters/search_config.project.json"
    if editable.is_file():
        target = local / "parameters/search_config.project.json"
        target.write_text(
            json.dumps(relocated(json.loads(editable.read_text(encoding="utf-8")), root, local)),
            encoding="utf-8",
        )
    from phase_agent.runtime.runtime_composition import compose_runtime

    monkeypatch.setattr(
        "phase_agent.runtime.runtime_clients.create_optional_runtime_client", lambda *a, **k: client
    )
    handler = compose_runtime(
        runtime_file,
        chat_handler_factory=RunWorkflowChatHandler,
        model_switcher_factory=lambda *a, **k: forbidden,
        runtime_factory=forbidden,
    )
    from phase_agent.runtime import studio_runtime

    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(runtime_file))
    monkeypatch.setattr(studio_runtime, "_handler", handler)

    def send(text):
        return studio_runtime.send_project_message(text, "copy-project")

    print(project, "entering actual Studio message entry", flush=True)
    before_status = digest(state_path)
    status_reply = send("现在什么状态")
    assert digest(state_path) == before_status, "Status must not collect or execute"
    assert not requests, "Status must not call a model"
    waiting_dft = any(
        row.get("stage") in {"dft_single_point", "dft_relax"}
        and row.get("status") in {"pending", "running", "submitted", "unknown"}
        for row in saved.get("tasks") or []
    )
    if expected and waiting_dft:
        assert "回传包" in status_reply and "待本地" in status_reply, status_reply
        assert "回传包" in status_reply.splitlines()[0], status_reply
        assert "仅回收旧任务：上传" not in status_reply, status_reply
        assert "执行 `sbatch" not in status_reply, status_reply
    (tmp_path / "status-reply.txt").write_text(status_reply, encoding="utf-8")
    if expected and not waiting_dft and saved.get("pending_execution_policies"):
        # A live project may already have collected this batch while the
        # developer is checking another project. Do not erase its saved plan or
        # manufacture a fresh approval merely to exercise the input generator.
        assert "方案待确认" in status_reply
        current = json.loads(state_path.read_text(encoding="utf-8"))
        assert set(current["pending_execution_policies"]) == set(
            saved["pending_execution_policies"]
        )
        assert (
            sum(row.get("status") == "completed" for row in current.get("dft_mlip_comparisons", []))
            == expected
        )
        (tmp_path / "dialogue-evidence.json").write_text(
            json.dumps(
                {
                    "project": project,
                    "scope": "already_recovered_saved_plan_status",
                    "paired": expected,
                    "status_reply": status_reply,
                }
            ),
            encoding="utf-8",
        )
        for p, sha in protected.items():
            assert digest(p) == sha
        return
    reply = send("结果已回传，请回收分析，并准备下一步微调方案。")
    observed = json.loads(state_path.read_text(encoding="utf-8"))
    evidence = {
        "project": project,
        "first_reply": reply,
        "requests": requests,
        "status": handler.last_turn_outcome,
    }
    (tmp_path / "dialogue-evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    if not expected:
        assert set(observed.get("remote_finetune_jobs", {})) == set(
            saved.get("remote_finetune_jobs", {})
        )
        for p, sha in protected.items():
            assert digest(p) == sha
        return
    assert (
        sum(r["status"] == "completed" for r in observed.get("dft_mlip_comparisons", []))
        == expected
    ), evidence
    assert not observed.get("remote_finetune_jobs"), "No inputs before concrete approval"
    assert len(observed.get("pending_execution_policies", {})) == 1, evidence
    before_approve = len(requests)
    reply2 = send("同意")
    final = json.loads(state_path.read_text(encoding="utf-8"))
    evidence.update(
        approval_reply=reply2,
        approval_status=handler.last_turn_outcome,
        model_calls_at_approval=len(requests) - before_approve,
    )
    (tmp_path / "dialogue-evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert len(requests) == before_approve, "Approval must reuse the stored scientific plan"
    assert len(final.get("remote_finetune_jobs", {})) == 1, evidence
    job = next(iter(final["remote_finetune_jobs"].values()))
    assert job["submitted"] is False and job["activated"] is False
    assert len(final["new_dft_records"]) == expected
    folders = list((local / "submissions").rglob("epoch0_Finetune_Batch01_input"))
    assert len(folders) == 1
    folder = folders[0]
    assert {p.name for p in folder.iterdir()} == {
        "inputs.zip",
        "submit_all.sh",
        "dispatch_tasks.py",
    }
    output = folder.with_name("epoch0_Finetune_Batch01_output")
    assert output.is_dir() and not list(output.iterdir())
    with zipfile.ZipFile(folder / "inputs.zip") as archive:
        assert archive.testzip() is None
        for row in json.loads(archive.read("transfer_manifest.json"))["files"]:
            assert hashlib.sha256(archive.read(row["path"])).hexdigest() == row["sha256"]
    energies, forces = build_comparison_tables(
        final["dft_dataset_records"], final["dft_mlip_comparisons"], round_name="Recovery04"
    )
    metrics = comparison_metrics(energies, forces, {"test": "real-dialogue"})
    assert metrics["matched_structures"] == expected
    evidence.update(
        paired=expected,
        input=str(folder),
        output=str(output),
        metrics=metrics,
        committee_members=len(job["report"]["committees"]),
        cv_folds=len(job["report"]["main_model"]["fold_jobs"]),
    )
    calls_before_repeat = len(requests)
    repeated_reply = send("同意")
    repeated = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(repeated.get("remote_finetune_jobs", {})) == 1
    assert len(repeated["new_dft_records"]) == expected
    assert len(requests) == calls_before_repeat
    evidence["repeat_approval_reply"] = repeated_reply
    for p, sha in protected.items():
        assert digest(p) == sha
    (tmp_path / "dialogue-evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(
        project,
        "PASS real dialogue notice -> errors -> proposal -> approval -> input/output; pairs",
        expected,
        flush=True,
    )
