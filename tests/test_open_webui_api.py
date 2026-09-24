import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from run.open_webui_api import (
    RunWorkflowChatHandler, classify_user_decision, create_open_webui_runtime,
    handle_chat_request,
)
from run.configuration_chat import ConfigurationChatHandler
from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
def runtime_adapter_fixture():
    return "adapter-value"


class OpenWebUIAPITest(unittest.TestCase):
    def _runtime_files(self, root, *, history=False, session=True):
        from data_layer.ledger.phase_data_manager import PhaseDataManager

        root = Path(root)
        boundary = {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1}}
        confirmed = {"system": {"boundary": boundary}, "agent": {"allowed_tools": ["pause_search"]}, "budgets": {}}
        config_session = {"status": "confirmed", "confirmed_snapshot": {
            "config_version": "config-test", "config_hash": "hash", "config": confirmed,
        }}
        state = {"confirmed_config": confirmed, "config_version": "config-test"}
        if history:
            state["action_records"] = [{"status": "completed"}]
        (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
        manager = PhaseDataManager(boundary)
        if history:
            manager.add_branch(P="O3", H=boundary["H"]["O3"][0], x=1, T=1)
        manager.save(root / "ledger.json")
        (root / "phase.json").write_text("{}", encoding="utf-8")
        if session:
            (root / "session.json").write_text(json.dumps(config_session), encoding="utf-8")
        settings = {"state_path": "state.json", "ledger_path": "ledger.json",
                    "phase_references_path": "phase.json", "new_runs_directory": "runs"}
        if session:
            settings["config_session_path"] = "session.json"
        runtime = root / "runtime.json"
        runtime.write_text(json.dumps(settings), encoding="utf-8")
        return runtime

    def test_only_final_user_line_can_approve(self):
        self.assertEqual(classify_user_decision("请解释预算\n同意"), "approve")
        self.assertEqual(classify_user_decision("请解释预算"), "comment")
        self.assertEqual(classify_user_decision("不同意"), "comment")
        self.assertEqual(classify_user_decision("拒绝"), "reject")

    def test_openai_request_passes_real_user_messages(self):
        seen = {}
        response = handle_chat_request(
            {"model": "phase-search-agent", "user": "chat-1", "messages": [
                {"role": "assistant", "content": "approve"},
                {"role": "user", "content": "请先解释结果"},
            ]},
            lambda messages, *, conversation_id: seen.update(
                messages=messages, conversation_id=conversation_id
            ) or "awaiting approval",
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "awaiting approval")
        self.assertEqual(seen["conversation_id"], "chat-1")
        self.assertEqual(seen["messages"][-1]["role"], "user")

    def test_pending_proposal_uses_actual_user_turn_and_stable_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({"pending_execution_policies": {
                "action-7": {"revision": 1, "agent_proposal": {"raw_action": {"tool": "pause_search"}}}
            }}), encoding="utf-8")
            calls = []

            def workflow(**kwargs):
                calls.append(kwargs)
                return {"status": "awaiting_approval", "agent_proposal": {
                    "current_state_analysis": "test", "recommended_action": "pause_search",
                    "action_parameters": {}, "reason": "test", "estimated_cost": {},
                    "calculation_plan": {}, "expected_purpose": "test",
                }}

            handler = RunWorkflowChatHandler({
                "state_path": str(state_path),
                "config_session": {"status": "confirmed"},
            }, workflow=workflow)
            reply = handler([{"role": "user", "content": "调整预算后再看"}])
            self.assertIn("Agent action proposal", reply)
            self.assertEqual(calls[0]["invocation_id"], "action-7")
            self.assertEqual(calls[0]["execution_mode"], "interactive")
            self.assertEqual(calls[0]["human_feedback"], {
                "decision": "comment", "comment": "调整预算后再看",
            })

    def test_unpending_user_instruction_reaches_project_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            observed = {}

            def workflow(**kwargs):
                observed.update(kwargs["agent_client"]({"state": "snapshot"}))
                return {"status": "planned_only"}

            handler = RunWorkflowChatHandler({
                "state_path": str(state_path),
                "agent_client": lambda payload: payload,
            }, workflow=workflow)
            handler([{"role": "user", "content": "优先评估最近 DFT 误差"}])
            self.assertEqual(observed["user_instruction"], "优先评估最近 DFT 误差")

    def test_status_command_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({"tasks": [], "slurm_batches": []}), encoding="utf-8")
            calls = []
            handler = RunWorkflowChatHandler({"state_path": str(state_path)}, workflow=lambda **kwargs: calls.append(kwargs))
            reply = handler([{"role": "user", "content": "查看状态"}])
            self.assertIn("当前项目状态", reply)
            self.assertEqual(calls, [])

    def test_missing_user_message_is_rejected(self):
        with self.assertRaises(ValueError):
            handle_chat_request({"messages": [{"role": "assistant", "content": "hello"}]}, lambda *_a, **_k: "x")

    def test_builtin_factory_composes_runtime_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "local-test-key"}):
            runtime = self._runtime_files(directory)
            settings = json.loads(runtime.read_text(encoding="utf-8"))
            settings["runtime_adapters"] = {"example_adapter": "tests.test_open_webui_api:runtime_adapter_fixture"}
            runtime.write_text(json.dumps(settings), encoding="utf-8")
            handler = create_open_webui_runtime(runtime)
            self.assertEqual(type(handler.workflow_kwargs["manager"]).__name__, "PhaseDataManager")
            self.assertEqual(handler.workflow_kwargs["example_adapter"], "adapter-value")
            self.assertEqual(handler.workflow_kwargs["config_session"]["confirmed_snapshot"]["config_version"], "config-test")

    def test_factory_recovers_confirmed_config_from_old_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "local-test-key"}):
            runtime = self._runtime_files(directory, session=False)
            handler = create_open_webui_runtime(runtime)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "confirmed")
            self.assertEqual(handler.workflow_kwargs["config_session"]["confirmed_snapshot"]["config_version"], "config-test")

    def test_history_requires_choice_before_workflow_and_continue_restores_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            proposal = {"current_state_analysis": "old", "recommended_action": "pause_search",
                        "action_parameters": {}, "reason": "old", "estimated_cost": {},
                        "calculation_plan": {}, "expected_purpose": "inspect"}
            state_path.write_text(json.dumps({"pending_execution_policies": {
                "old-action": {"agent_proposal": proposal}
            }}), encoding="utf-8")
            calls = []
            handler = RunWorkflowChatHandler({"state_path": str(state_path)}, workflow=lambda **kw: calls.append(kw), history_prompt=True)
            first = handler([{"role": "user", "content": "查看一下"}], conversation_id="chat-a")
            self.assertIn("继续", first)
            self.assertEqual(calls, [])
            continued = handler([{"role": "user", "content": "继续"}], conversation_id="chat-a")
            self.assertIn("Agent action proposal", continued)
            self.assertEqual(calls, [])

    def test_new_run_uses_unique_paths_and_keeps_old_ledger(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "local-test-key"}):
            runtime = self._runtime_files(directory, history=True)
            old_ledger = Path(directory) / "ledger.json"
            before = old_ledger.read_bytes()
            first = create_open_webui_runtime(runtime)
            second = create_open_webui_runtime(runtime)
            first([{"role": "user", "content": "新建"}], conversation_id="one")
            second([{"role": "user", "content": "新建"}], conversation_id="two")
            self.assertNotEqual(first.state_path, second.state_path)
            self.assertTrue(first.state_path.exists())
            self.assertTrue((first.state_path.parent / "phase_data.json").exists())
            self.assertEqual(old_ledger.read_bytes(), before)

    def test_no_history_runs_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            calls = []
            handler = RunWorkflowChatHandler({"state_path": str(state_path)},
                                             workflow=lambda **kw: calls.append(kw) or {"status": "planned_only"})
            handler([{"role": "user", "content": "开始"}], conversation_id="chat")
            self.assertEqual(len(calls), 1)

    def test_incomplete_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.json"
            runtime.write_text(json.dumps({"state_path": "state.json"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少"):
                create_open_webui_runtime(runtime)

    def test_first_start_enters_persistent_config_only_mode(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            root = Path(directory)
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps({
                "config_session_path": "session.json", "state_path": "state.json",
                "ledger_path": "ledger.json", "phase_references_path": "phase.json",
            }), encoding="utf-8")
            handler = create_open_webui_runtime(runtime)
            self.assertIsInstance(handler, ConfigurationChatHandler)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "draft")
            self.assertEqual(
                handler.workflow_kwargs["config_session"]["setup_stage"],
                "awaiting_storage_path",
            )
            self.assertTrue((root / "session.json").is_file())
            self.assertIsNone(handler.editable_config_path)
            self.assertFalse((root / "search_config.draft.json").exists())
            self.assertFalse((root / "state.json").exists())
            self.assertFalse((root / "ledger.json").exists())
            workspace = root / "my phase workspace"
            path_reply = handler(
                [{"role": "user", "content": str(workspace)}], conversation_id="setup"
            )
            self.assertIn("确认存储路径", path_reply)
            self.assertIn(str(workspace.resolve()), path_reply)
            self.assertFalse(workspace.exists())
            self.assertFalse((root / "search_config.draft.json").exists())

            resumed = create_open_webui_runtime(runtime)
            self.assertEqual(
                resumed.workflow_kwargs["config_session"]["setup_stage"],
                "awaiting_storage_confirmation",
            )
            self.assertFalse(workspace.exists())
            confirm_path = resumed(
                [{"role": "user", "content": "确认存储路径"}], conversation_id="setup"
            )
            draft = workspace / "search_config.draft.json"
            self.assertIn(str(draft), confirm_path)
            self.assertIn("读取配置 JSON", confirm_path)
            self.assertTrue(draft.is_file())
            self.assertEqual(
                resumed.workflow_kwargs["config_session"]["config"]["storage"]["workspace_root"],
                str(workspace.resolve()),
            )
            draft.write_text("user edit", encoding="utf-8")
            ready_handler = create_open_webui_runtime(runtime)
            self.assertEqual(draft.read_text(encoding="utf-8"), "user edit")
            reply = ready_handler([{"role": "user", "content": "开始配置"}], conversation_id="setup")
            self.assertIn("DeepSeek API", reply)
            self.assertFalse((root / "state.json").exists())

    def test_legacy_draft_is_migrated_to_path_first_without_overwriting_json(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft
        from config_layer.session.save_config_session import save_config_session

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            root = Path(directory)
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps({
                "config_session_path": "session.json",
                "editable_config_draft_path": "search_config.draft.json",
                "state_path": "state.json",
                "ledger_path": "ledger.json",
            }), encoding="utf-8")
            session_path = root / "session.json"
            save_config_session(create_config_draft(default_layered_search_config()), session_path)
            old_draft = root / "search_config.draft.json"
            old_draft.write_text("preserve prior user file", encoding="utf-8")

            handler = create_open_webui_runtime(runtime)
            self.assertEqual(
                handler.workflow_kwargs["config_session"]["setup_stage"],
                "awaiting_storage_path",
            )
            self.assertIsNone(handler.editable_config_path)
            self.assertEqual(old_draft.read_text(encoding="utf-8"), "preserve prior user file")

            handler([{"role": "user", "content": str(root)}], conversation_id="setup")
            reply = handler([{"role": "user", "content": "确认存储路径"}], conversation_id="setup")
            self.assertIn("已有设置文件，未覆盖", reply)
            self.assertEqual(old_draft.read_text(encoding="utf-8"), "preserve prior user file")
            self.assertFalse((root / "state.json").exists())

    def test_configuration_agent_only_edits_draft_and_cannot_start_workflow(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = root / "session.json"
            state_path = root / "state.json"
            session = create_config_draft(default_layered_search_config())
            calls = []
            handler = ConfigurationChatHandler(
                {"state_path": str(state_path), "config_session": session},
                config_session_path=session_path, base_directory=root,
                agent_client=lambda _payload: {
                    "reply": "已记录模型版本候选。", "patch": {
                        "calculation.mlip_version": "mace-mh-1"
                    }, "reasons": {"calculation.mlip_version": "用户明确提供的模型信息。"},
                    "questions": [],
                },
                runtime_factory=lambda: calls.append("workflow"),
            )
            reply = handler([{"role": "user", "content": "模型使用 mh-1"}], conversation_id="setup")
            self.assertIn("calculation.mlip_version", reply)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "draft")
            self.assertFalse(state_path.exists())
            self.assertEqual(calls, [])
            blocked = handler([{"role": "user", "content": "确认配置"}], conversation_id="setup")
            self.assertIn("还不能确认", blocked)
            self.assertEqual(calls, [])

    def test_explicit_config_confirmation_saves_snapshot_but_does_not_start_workflow(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "O3.vasp"
            reference.write_text("structure placeholder", encoding="utf-8")
            boundary = {
                "P": ["O3"],
                "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
                "TM_ratio": {"Fe": 1, "Mn": 1},
            }
            config = default_layered_search_config(
                boundary=boundary, phase_references={"O3": str(reference)}
            )
            config["calculation"]["mlip_version"] = "mace-mh-1"
            config["mlip"]["model_path"] = "/cluster/models/mace-mh-1.model"
            session = create_config_draft(config)
            delegate_calls = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                runtime_factory=lambda: (
                    lambda _messages, *, conversation_id=None:
                    delegate_calls.append(conversation_id) or "search mode"
                ),
            )
            confirmation = handler(
                [{"role": "user", "content": "确认配置"}], conversation_id="setup"
            )
            self.assertIn("不会提交或启动计算", confirmation)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "confirmed")
            confirmed = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
            saved = root / "config_snapshots" / f"{confirmed['config_version']}.json"
            self.assertTrue(saved.is_file())
            self.assertIn(str(saved), confirmation)
            self.assertFalse((root / "state.json").exists())
            self.assertEqual(delegate_calls, [])
            next_turn = handler(
                [{"role": "user", "content": "继续"}], conversation_id="setup"
            )
            self.assertEqual(next_turn, "search mode")
            self.assertEqual(delegate_calls, ["setup"])

    def test_manual_upload_runner_writes_portable_review_bundle_without_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ManualUploadBatchRunner(
                Path(directory) / "batches",
                worker_command=["python3", "-m", "worker"],
                stage_profiles={"mc": {"stages": ["deep_search"],
                                       "slurm_options": {"partition": "REVIEW_ME"}}},
            )
            task = {"task_id": "T1", "task_key": "K1", "stage": "deep_search", "status": "pending"}
            result = runner.prepare({
                "confirmed_config_version": "c1", "tasks": [task], "pending_tasks": [task],
                "budget_reservations": {"K1": {"status": "reserved", "reserved_cost": 1}},
            })
            batch = result["batch"]
            root = Path(batch["upload_directory"])
            self.assertEqual(result["status"], "prepared")
            self.assertTrue((root / "manifest.json").is_file())
            self.assertTrue((root / "SHA256SUMS").is_file())
            self.assertIn("Nothing has been submitted", (root / "UPLOAD_AND_SUBMIT.md").read_text())
            self.assertNotIn("scheduler_job_id", batch)


if __name__ == "__main__":
    unittest.main()
