import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from run.open_webui_api import (
    RunWorkflowChatHandler, classify_user_decision, create_open_webui_runtime,
    handle_chat_request, _workspace_path_clarification_reply, format_workflow_reply,
    _is_config_migration_approval,
)
from run.configuration_chat import ConfigurationChatHandler
from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
def runtime_adapter_fixture():
    return "adapter-value"


class OpenWebUIAPITest(unittest.TestCase):
    def test_workspace_correction_is_not_a_tool_proposal(self):
        state = {"confirmed_config": {"storage": {"workspace_root": r"E:\0-FM-PhaseDiagram"}}}
        reply = _workspace_path_clarification_reply(
            r"地址是E:\0-FM-PhaseDiagram 你写错地方了", state)
        self.assertIn("当前快照的工作区", reply)
        self.assertNotIn("建议：", reply)
        output_reply = _workspace_path_clarification_reply(
            r"输出目录：E:\0-FM-PhaseDiagram，帮我写入", state)
        self.assertIn("当前快照的工作区", output_reply)
        bad_state = {"confirmed_config": {"storage": {
            "workspace_root": r"E:\0-FM-PhaseDiagram     agent：V4.1flash"}}}
        recovery = _workspace_path_clarification_reply(
            r"输出目录：E:\0-FM-PhaseDiagram，帮我写入", bad_state)
        self.assertIn("已确认快照中的工作区路径", recovery)
        self.assertNotIn("建议：", recovery)

    def test_invalid_action_is_not_presented_for_approval(self):
        reply = format_workflow_reply(
            {"status": "awaiting_approval", "agent_proposal": {
                "recommended_action": None, "raw_action": {"tool": None}}}, "state.json")
        self.assertIn("无效动作", reply)
        self.assertNotIn("回复“同意”", reply)

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

    def test_config_migration_approval_is_routed_separately_from_action_approval(self):
        self.assertTrue(_is_config_migration_approval("批准迁移"))
        self.assertTrue(_is_config_migration_approval("请批准迁移"))
        self.assertFalse(_is_config_migration_approval("为什么批准迁移会失败"))
        self.assertFalse(_is_config_migration_approval("不批准迁移"))
        calls = []
        state_path = Path.cwd() / "migration-test-state-does-not-exist.json"

        def workflow(**kwargs):
            calls.append(kwargs)
            return {"status": "config_migrated", "from_config_version": "v1",
                    "config_version": "v2", "migration": {"budget_changes": [
                        {"field": "round_strategy.maximum_mc_budget", "from": 1000, "to": 5000},
                      ], "compatibility": {
                          "relax_task_count": 406,
                          "unrecorded_fields": ["mlip.relax_parameters.mace_default_dtype"],
                      }}}

        handler = RunWorkflowChatHandler(
            {"state_path": str(state_path)}, workflow=workflow, history_prompt=False,
        )
        reply = handler([{"role": "user", "content": "批准迁移"}], conversation_id="chat")
        self.assertTrue(calls[0]["approve_budget_extension"])
        self.assertIn("历史任务和结果仍保留原版本归属", reply)
        self.assertIn("maximum_mc_budget 1000→5000", reply)
        self.assertIn("已核对 406 条旧 Relax 结果", reply)
        self.assertIn("mace_default_dtype", reply)
        self.assertFalse(calls[0].get("execute_scientific_action", False))

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

    def test_simple_agreement_approves_current_non_sensitive_round(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            proposal = {"raw_action": {"tool": "pause_search"}}
            state_path.write_text(json.dumps({"pending_execution_policies": {
                "action-9": {"revision": 1, "agent_proposal": proposal}
            }}), encoding="utf-8")
            handler = RunWorkflowChatHandler({"state_path": str(state_path)})
            with patch.object(
                handler, "review_pending",
                return_value={"status": "completed", "result": {"status": "completed"}},
            ) as review:
                reply = handler([{"role": "user", "content": "同意"}])
            self.assertIn("completed", reply)
            self.assertEqual(review.call_args.args[0], "action-9")
            self.assertEqual(review.call_args.args[1], "approve")

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

    def test_confirmed_runtime_starts_without_deepseek_key_for_browser_setup(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("run.deepseek_credentials.load_deepseek_api_key", return_value=None):
            runtime = self._runtime_files(directory, session=False)
            handler = create_open_webui_runtime(runtime)
            self.assertIsNone(handler.workflow_kwargs["agent_client"])
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "confirmed")

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

    def test_continue_does_not_call_config_edit_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({"tasks": [{"task_id": "T1", "status": "completed"}]}),
                                  encoding="utf-8")
            handler = RunWorkflowChatHandler({"state_path": str(state_path)},
                                             workflow=lambda **_: {"status": "planned_only"},
                                             history_prompt=True)
            with patch("decision_layer.agent.classify_config_edit_intent.classify_config_edit_intent",
                       side_effect=AssertionError("纯继续不应分析为配置编辑")):
                handler([{"role": "user", "content": "继续"}])

    def test_navigation_commands_never_open_config_revision(self):
        for command in ("开始", "开始搜索", "下一步", "然后呢", "恢复运行"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                state_path = Path(directory) / "state.json"
                state_path.write_text("{}", encoding="utf-8")
                handler = RunWorkflowChatHandler(
                    {"state_path": str(state_path)},
                    workflow=lambda **_: {"status": "planned_only"},
                    config_intent_client=lambda _: (_ for _ in ()).throw(
                        AssertionError("流程指令不应调用配置分类器")),
                    config_revision_factory=lambda _: (_ for _ in ()).throw(
                        AssertionError("流程指令不应创建配置草稿")),
                )
                handler([{"role": "user", "content": command}])

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
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}), patch("run.deepseek_credentials.load_deepseek_api_key", return_value=None):
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

            workspace = root / "my phase workspace"
            path_reply = handler(
                [{"role": "user", "content": str(workspace)}], conversation_id="setup"
            )
            self.assertIn("回复“确认”", path_reply)
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
                [{"role": "user", "content": "确认"}], conversation_id="setup"
            )
            draft = workspace / "search_config.project.json"
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

    def test_workspace_path_rejects_multiline_prompt_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = root / "session.json"
            from config_layer.defaults.default_layered_search_config import default_layered_search_config
            from config_layer.session.create_config_draft import create_config_draft
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"),
                 "config_session": create_config_draft(
                     default_layered_search_config(), require_workspace_path=True
                 )},
                config_session_path=session_path, base_directory=root,
            )
            reply = handler([{"role": "user", "content": (
                str(root / "workspace") + "\\run\\### Task:\nSuggest follow-ups"
            )}], conversation_id="setup")
            self.assertIn("首次只需提供两项", reply)
            session = handler.workflow_kwargs["config_session"]
            self.assertEqual(session["setup_stage"], "awaiting_storage_path")
            self.assertNotIn("pending_workspace_root", session)
            self.assertFalse((root / "workspace").exists())

    def test_workspace_and_agent_model_can_be_confirmed_together_before_config(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            agent_calls = []
            switched_models = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"),
                 "config_session": create_config_draft(
                     default_layered_search_config(), require_workspace_path=True
                 )},
                config_session_path=root / "session.json", base_directory=root,
                current_deepseek_model="deepseek-v4-pro",
                agent_client=lambda payload: agent_calls.append(payload),
                deepseek_model_switcher=lambda model: (
                    switched_models.append(model) or model, lambda _payload: {}
                ),
            )

            preview = handler([{"role": "user", "content": (
                f"工作区根路径：{workspace}\nAgent版本：V4.1 Flash"
            )}], conversation_id="setup")
            self.assertIn(str(workspace.resolve()), preview)
            self.assertIn("deepseek-flash", preview)
            self.assertIn("回复“确认”", preview)
            self.assertFalse(workspace.exists())
            self.assertEqual(switched_models, [])

            confirmed = handler(
                [{"role": "user", "content": "确认"}], conversation_id="setup"
            )
            self.assertIn("必须补齐或解决", confirmed)
            self.assertIn("建议检查项", confirmed)
            self.assertIn("读取配置 JSON", confirmed)
            self.assertIn("读取配置 JSON 并继续", confirmed)
            self.assertTrue((workspace / "search_config.project.json").is_file())
            self.assertEqual(switched_models, ["deepseek-flash"])
            self.assertEqual(agent_calls, [])
            session = handler.workflow_kwargs["config_session"]
            self.assertEqual(session["setup_stage"], "json_ready")
            self.assertNotEqual(session.get("status"), "confirmed")
            self.assertFalse((root / "state.json").exists())

    def test_workspace_write_failure_does_not_commit_bad_session_state(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"),
                 "config_session": create_config_draft(
                     default_layered_search_config(), require_workspace_path=True
                 )},
                config_session_path=root / "session.json", base_directory=root,
            )
            handler([{"role": "user", "content": str(root / "workspace")}], conversation_id="setup")
            with patch(
                "config_layer.session.project_config_json.create_project_config_json",
                side_effect=OSError("permission denied"),
            ):
                reply = handler(
                    [{"role": "user", "content": "确认"}], conversation_id="setup"
                )
            self.assertIn("设置 JSON 写入失败", reply)
            session = handler.workflow_kwargs["config_session"]
            self.assertEqual(session["setup_stage"], "awaiting_storage_confirmation")
            self.assertNotIn("storage", session["config"])
            self.assertNotIn("editable_config_json_path", session)

    def test_corrupt_json_ready_workspace_session_recovers_to_path_prompt(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft
        from config_layer.session.save_config_session import save_config_session

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            root = Path(directory)
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps({
                "config_session_path": "session.json", "state_path": "state.json",
                "ledger_path": "ledger.json",
            }), encoding="utf-8")
            session = create_config_draft(default_layered_search_config(), require_workspace_path=True)
            session["setup_stage"] = "json_ready"
            session["config"]["storage"] = {
                "workspace_root": str(root / "run" / "### Task:\nmalformed prompt"),
                "paths": {},
            }
            session["editable_config_json_path"] = (
                str(root / "run" / "### Task:\nmalformed prompt" / "search_config.draft.json")
            )
            save_config_session(session, root / "session.json")

            handler = create_open_webui_runtime(runtime)
            recovered = handler.workflow_kwargs["config_session"]
            self.assertEqual(recovered["setup_stage"], "awaiting_storage_path")
            self.assertNotIn("editable_config_json_path", recovered)
            self.assertNotIn("storage", recovered["config"])
            self.assertIsNone(handler.editable_config_path)
            self.assertFalse((root / "run").exists())
            self.assertTrue(any(
                item.get("type") == "workspace_path_recovery"
                for item in recovered.get("dialogue", [])
            ))

    def test_startup_setup_stages_optional_flash_model_until_path_confirmation(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            switch_calls = []
            config_agent_calls = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"),
                 "config_session": create_config_draft(
                     default_layered_search_config(), require_workspace_path=True
                 )},
                config_session_path=root / "session.json", base_directory=root,
                current_deepseek_model="deepseek-v4-pro",
                agent_client=lambda payload: config_agent_calls.append(payload),
                deepseek_model_switcher=lambda model: (
                    switch_calls.append(model) or model, lambda _payload: {}
                ),
            )
            reply = handler(
                [{"role": "user", "content": "我想换成 DeepSeek V4 Flash"}],
                conversation_id="setup",
            )
            self.assertIn("请发送本地工作区根目录路径", reply)
            self.assertEqual(
                handler.workflow_kwargs["config_session"]["pending_deepseek_model"],
                "deepseek-flash",
            )
            self.assertEqual(switch_calls, [])
            workspace = root / "new workspace"
            preview = handler(
                [{"role": "user", "content": str(workspace)}], conversation_id="setup"
            )
            self.assertIn("DeepSeek V4.1 Flash", preview)
            self.assertEqual(switch_calls, [])
            confirmed = handler(
                [{"role": "user", "content": "确认"}], conversation_id="setup"
            )
            self.assertIn("DeepSeek V4.1 Flash", confirmed)
            self.assertEqual(switch_calls, ["deepseek-flash"])
            self.assertTrue((workspace / "search_config.project.json").is_file())
            self.assertEqual(config_agent_calls, [])

    def test_deepseek_model_request_resolves_current_flash_alias(self):
        from run.resolve_deepseek_model_request import resolve_deepseek_model_request

        self.assertEqual(
            resolve_deepseek_model_request("不是 Pro，我想换成 V4 Flash"),
            "deepseek-flash",
        )
        self.assertEqual(resolve_deepseek_model_request("V4 Flash"), "deepseek-flash")
        self.assertEqual(
            resolve_deepseek_model_request("切回 DeepSeek V4 Pro"),
            "deepseek-v4-pro",
        )
        self.assertIsNone(resolve_deepseek_model_request("V4 Flash 的价格是多少？"))

    def test_runtime_model_update_preserves_other_settings(self):
        from run.set_deepseek_runtime_model import set_deepseek_runtime_model

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.json"
            runtime.write_text(json.dumps({
                "deepseek": {"model": "deepseek-v4-pro", "base_url": "https://api.deepseek.com"},
                "state_path": "state.json",
            }), encoding="utf-8")
            selected = set_deepseek_runtime_model(runtime, "deepseek-flash")
            saved = json.loads(runtime.read_text(encoding="utf-8"))
            self.assertEqual(selected, "deepseek-flash")
            self.assertEqual(saved["deepseek"]["model"], "deepseek-flash")
            self.assertEqual(saved["deepseek"]["base_url"], "https://api.deepseek.com")
            self.assertEqual(saved["state_path"], "state.json")
            with self.assertRaisesRegex(ValueError, "不支持"):
                set_deepseek_runtime_model(runtime, "unknown-model")

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
            reply = handler([{"role": "user", "content": "确认"}], conversation_id="setup")
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
            self.assertIn("读取配置 JSON", blocked)
            self.assertEqual(calls, [])

    def test_config_agent_receives_confirmed_setup_and_recent_question_context(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            config = default_layered_search_config()
            config["calculation"]["mlip_version"] = None  # emulate an older draft
            config["bohb"]["scope"]["mlip_version"] = None
            config["storage"] = {"workspace_root": str(root), "paths": {}}
            session = create_config_draft(config)
            session["setup_stage"] = "json_ready"
            session["dialogue"].append({
                "type": "message", "role": "assistant",
                "message": (
                    "请确认 calculation.mlip_version 与 bohb.scope.mlip_version "
                    "是否都填 mace-mh-1？工作区 E:\\0-FM-PhaseDiagram 已确认。"
                ),
            })
            payloads = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                agent_client=lambda payload: (
                    payloads.append(payload) or {
                        "reply": "已按刚才确认的内容记录为 mace-mh-1。",
                        "patch": {
                            "calculation.mlip_version": "mace-mh-1",
                            "bohb.scope.mlip_version": "mace-mh-1",
                        },
                        "reasons": {
                            "calculation.mlip_version": "用户肯定回答了紧邻前一条版本确认问题。",
                            "bohb.scope.mlip_version": "用户肯定回答了紧邻前一条版本确认问题。",
                        },
                        "questions": [],
                    }
                ),
            )

            reply = handler([{"role": "user", "content": "是的"}], conversation_id="setup")

            self.assertIn("已按刚才确认", reply)
            payload = payloads[0]
            self.assertEqual(payload["setup_facts"]["workspace_root"], str(root))
            self.assertTrue(payload["setup_facts"]["workspace_root_confirmed"])
            self.assertEqual(payload["setup_facts"]["default_mlip_version"], "mace-mh-1")
            self.assertIn("是否都填 mace-mh-1", payload["conversation_context"][-1]["message"])
            updated = handler.workflow_kwargs["config_session"]["config"]
            self.assertEqual(updated["calculation"]["mlip_version"], "mace-mh-1")
            self.assertEqual(updated["bohb"]["scope"]["mlip_version"], "mace-mh-1")

    def test_default_mlip_version_is_mh1_for_search_and_bohb(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from run.configuration_chat import _fill_default_mlip_versions

        config = default_layered_search_config()
        self.assertEqual(config["calculation"]["mlip_version"], "mace-mh-1")
        self.assertEqual(config["mlip"]["name"], "mace-mh-1")
        self.assertEqual(config["bohb"]["scope"]["mlip_version"], "mace-mh-1")
        self.assertIsNone(config["mlip"]["model_path"])

        config["calculation"]["mlip_version"] = None
        config["bohb"]["scope"]["mlip_version"] = None
        migrated, filled = _fill_default_mlip_versions(config)
        self.assertEqual(migrated["calculation"]["mlip_version"], "mace-mh-1")
        self.assertEqual(migrated["bohb"]["scope"]["mlip_version"], "mace-mh-1")
        self.assertEqual(set(filled), {"calculation.mlip_version", "bohb.scope.mlip_version"})

    def test_config_cannot_start_without_agent_review(self):
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
            self.assertIn("Agent 审核通过后", confirmation)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "draft")
            self.assertFalse((root / "state.json").exists())
            self.assertEqual(delegate_calls, [])

    def test_agent_and_user_approval_enters_search_after_json_review(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft
        from config_layer.session.create_editable_config_json import create_editable_config_json
        from config_layer.session.resolve_workspace_paths import default_workspace_storage

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
            config["storage"] = default_workspace_storage(root)
            config_file = root / "search_config.draft.json"
            create_editable_config_json(
                config_file, config, workspace_defaults=config["storage"],
            )
            session = create_config_draft(config)
            session["setup_stage"] = "json_ready"
            session["editable_config_json_path"] = str(config_file)
            agent_reviews = []
            started = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                editable_config_path=config_file,
                agent_client=lambda payload: (
                    agent_reviews.append(payload) or {
                        "reply": "配置没有阻止搜索的问题。",
                        "patch": {}, "questions": [], "ready_for_search": True,
                    }
                ),
                runtime_factory=lambda: (
                    lambda messages, *, conversation_id=None:
                    started.append((messages[-1]["content"], conversation_id)) or "search proposal"
                ),
            )

            reviewed = handler(
                [{"role": "user", "content": "读取配置 JSON"}], conversation_id="setup"
            )
            self.assertIn("Agent 与程序检查均通过", reviewed)
            self.assertEqual(agent_reviews[0]["mode"], "configuration_json_review")
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "draft")
            self.assertFalse((root / "state.json").exists())
            self.assertEqual(started, [])

            result = handler(
                [{"role": "user", "content": "同意"}], conversation_id="setup"
            )
            self.assertIn("配置已确认并保存为版本", result)
            self.assertIn("搜索 Agent 已启动首轮分析", result)
            self.assertIn("search proposal", result)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "confirmed")
            self.assertEqual(started[0][1], "setup")
            self.assertIn("不得直接执行", started[0][0])

    def test_json_review_command_can_conditionally_continue_after_both_checks_pass(self):
        from config_layer.defaults.default_layered_search_config import default_layered_search_config
        from config_layer.session.create_config_draft import create_config_draft
        from config_layer.session.create_editable_config_json import create_editable_config_json
        from config_layer.session.resolve_workspace_paths import default_workspace_storage

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
            config["storage"] = default_workspace_storage(root)
            config_file = root / "search_config.draft.json"
            create_editable_config_json(config_file, config, workspace_defaults=config["storage"])
            session = create_config_draft(config)
            session["setup_stage"] = "json_ready"
            session["editable_config_json_path"] = str(config_file)
            started = []
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                editable_config_path=config_file,
                agent_client=lambda _payload: {
                    "reply": "Agent 检查通过。", "patch": {}, "questions": [],
                    "ready_for_search": True,
                },
                runtime_factory=lambda: (
                    lambda messages, *, conversation_id=None:
                    started.append((messages[-1]["content"], conversation_id)) or "search proposal"
                ),
            )

            reply = handler(
                [{"role": "user", "content": "读取配置 JSON 并继续"}],
                conversation_id="setup",
            )

            self.assertIn("配置已确认并保存为版本", reply)
            self.assertIn("搜索 Agent 已启动首轮分析", reply)
            self.assertEqual(handler.workflow_kwargs["config_session"]["status"], "confirmed")
            self.assertEqual(started[0][1], "setup")
            user_commands = [
                item for item in handler.workflow_kwargs["config_session"]["dialogue"]
                if item.get("role") == "user" and item.get("message") == "读取配置 JSON 并继续"
            ]
            self.assertEqual(len(user_commands), 1)

            blocked_config = default_layered_search_config(
                boundary=boundary, phase_references={"O3": str(reference)}
            )
            blocked_config["calculation"]["mlip_version"] = "mace-mh-1"
            blocked_config["mlip"]["model_path"] = None
            blocked_config["storage"] = default_workspace_storage(root)
            blocked_file = root / "blocked_config.json"
            create_editable_config_json(
                blocked_file, blocked_config, workspace_defaults=blocked_config["storage"]
            )
            blocked_session = create_config_draft(blocked_config)
            blocked_session["setup_stage"] = "json_ready"
            blocked_session["editable_config_json_path"] = str(blocked_file)
            blocked_started = []
            blocked_handler = ConfigurationChatHandler(
                {"state_path": str(root / "blocked_state.json"),
                 "config_session": blocked_session},
                config_session_path=root / "blocked_session.json", base_directory=root,
                editable_config_path=blocked_file,
                agent_client=lambda _payload: {
                    "reply": "Agent 检查通过。", "patch": {}, "questions": [],
                    "ready_for_search": True,
                },
                runtime_factory=lambda: blocked_started.append("created"),
            )
            blocked_reply = blocked_handler(
                [{"role": "user", "content": "读取配置 JSON 并继续"}],
                conversation_id="setup",
            )
            self.assertIn("当前仍有必填项", blocked_reply)
            self.assertEqual(blocked_handler.workflow_kwargs["config_session"]["status"], "draft")
            self.assertEqual(blocked_started, [])

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
