import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from analysis_layer.visualization.build_webui_charts import build_webui_charts
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.runtime.path_mapping import map_windows_to_linux, validate_path_mappings
from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.create_config_draft import create_config_draft
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from data_layer.memory.review_queue import propose_memory_update
from execution_layer.local.prepare_local_batch_files import prepare_local_batch_files
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.workflows.run_tool_step import run_tool_step
from execution_layer.policy.file_approval import proposal_hash
from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.file_protocol import write_json
from run.local_agent_control import LocalAgentControl
from run.open_webui_api import OpenWebUIRequestError, RunWorkflowChatHandler
from run.open_webui_api import format_workflow_reply


class OpenWebUILocalMVPTest(unittest.TestCase):
    def test_not_configured_reply_shows_nested_handler_reason(self):
        reply = format_workflow_reply({
            "status": "not_configured",
            "action": "prepare_local_batch_files",
            "execution": {"result": {"status": "not_configured",
                                      "reason": "remote_mlip_model_missing"}},
        }, Path("state.json"))
        self.assertIn("已确认的远端 MACE 模型路径", reply)
        self.assertNotIn("执行接口未配置", reply)

    def test_windows_linux_mapping_and_missing_or_conflicting_path(self):
        rows = validate_path_mappings([{"windows_local": r"E:\project\batches",
                                        "linux_remote": "/scratch/me/batches"}])
        self.assertEqual(map_windows_to_linux(r"E:\project\batches\B1\manifest.json", rows),
                         "/scratch/me/batches/B1/manifest.json")
        with self.assertRaisesRegex(ValueError, "absolute Windows"):
            validate_path_mappings([{"windows_local": "relative", "linux_remote": "/scratch/x"}])
        with self.assertRaisesRegex(ValueError, "no confirmed"):
            map_windows_to_linux(r"D:\other\x", rows)

    def test_dry_run_files_only_after_handler_call_conflict_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "batches"
            context = {"effective_config": {"local_action_directory": str(root)},
                       "config_version": "c1", "event_state": {"active_model_version": "m1"}}
            action = {"tool": "prepare_local_batch_files", "task_key": "T1",
                      "target_ids": ["S1"], "parameters": {"batch_id": "B1"}}
            self.assertFalse(root.exists())
            first = prepare_local_batch_files(action=action, context=context)
            self.assertEqual(first["status"], "prepared")
            self.assertTrue((root / "B1/manifest.json").is_file())
            second = prepare_local_batch_files(action=action, context=context)
            self.assertEqual(second["status"], "reused")
            conflicting = {**action, "task_key": "T2"}
            with self.assertRaises(FileExistsError):
                prepare_local_batch_files(action=conflicting, context=context)

    def test_registered_file_action_requires_approval_and_illegal_action_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = default_layered_search_config()
            session = create_config_draft(config)
            session = apply_config_revision(session, {"calculation.mlip_version": "m1",
                                                       "dft.parameters": {"encut": 520}})
            session = confirm_config_snapshot(session, user_confirmed=True)
            registry = create_tool_registry({"prepare_local_batch_files": prepare_local_batch_files})
            root = Path(directory) / "approved"
            agent = lambda _state: {"tool": "prepare_local_batch_files", "task_key": "LOCAL-1",
                "target_ids": ["S1"], "parameters": {"batch_id": "B1"}, "budget": 0,
                "reason": "prepare reviewed local files", "expected_purpose": "manual upload review",
                "_llm_usage": {"calls": 1, "input_tokens": 100, "output_tokens": 20, "cost": None}}
            proposed = run_tool_step({}, session, registry=registry, agent_client=agent,
                execution_mode="interactive", invocation_id="LOCAL-1",
                context={"effective_config": {"local_action_directory": str(root)}})
            self.assertEqual(proposed["status"], "awaiting_approval"); self.assertFalse(root.exists())
            self.assertEqual(proposed["state"]["budget_usage"]["llm"]["input_tokens"], 100)
            approved = run_tool_step(proposed["state"], session, registry=registry,
                execution_mode="interactive", invocation_id="LOCAL-1",
                human_feedback={"decision": "approve", "comment": "页面已核对\napprove"},
                context={"effective_config": {"local_action_directory": str(root)}})
            self.assertEqual(approved["execution"]["result"]["status"], "prepared")
            self.assertTrue((root / "B1/manifest.json").is_file())
            illegal = lambda _state: {"tool": "shell", "task_key": "BAD-1", "parameters": {},
                                      "budget": 0, "reason": "not allowed"}
            blocked = run_tool_step({}, session, registry=registry, agent_client=illegal,
                execution_mode="interactive", invocation_id="BAD-1")
            blocked = run_tool_step(blocked["state"], session, registry=registry,
                execution_mode="interactive", invocation_id="BAD-1",
                human_feedback={"decision": "approve", "comment": "approve"})
            self.assertEqual(blocked["status"], "rejected")
            self.assertIn("tool_not_allowed", blocked["validation"]["errors"])

    def test_simple_chat_agreement_approves_and_repeat_page_decision_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            proposal = {"recommended_action": "pause_search", "raw_action": {"tool": "pause_search"}}
            state = {"confirmed_config_version": "c1", "tasks": [],
                     "pending_execution_policies": {"P1": {"record_id": "P1", "agent_proposal": proposal}}}
            write_json(path, state); calls = []
            def workflow(**kwargs):
                calls.append(kwargs)
                updated = json.loads(path.read_text())
                updated["pending_execution_policies"].pop("P1")
                updated.setdefault("invocations", {})["P1"] = {"status": "completed"}
                write_json(path, updated)
                return {"status": "completed", "state": updated}
            handler = RunWorkflowChatHandler({"state_path": str(path)}, workflow=workflow)
            version = build_status_summary(state, config_version="c1")["summary_id"]
            with self.assertRaisesRegex(OpenWebUIRequestError, "stale"):
                handler.review_pending("P1", "approve", expected_state_version="old",
                                       expected_proposal_hash=proposal_hash(proposal))
            reply = handler([{"role": "user", "content": "同意"}], conversation_id="chat")
            self.assertIn("completed", reply)
            self.assertEqual(len(calls), 1)
            result = handler.review_pending("P1", "approve", expected_state_version=version,
                                            expected_proposal_hash=proposal_hash(proposal))
            self.assertEqual(result["status"], "already_processed")
            repeated = handler.review_pending("P1", "approve", expected_state_version=version,
                                              expected_proposal_hash=proposal_hash(proposal))
            self.assertEqual(repeated["status"], "already_processed")

    def test_draft_patch_confirm_and_memory_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); state_path = root / "state.json"; session_path = root / "session.json"
            session = create_config_draft(default_layered_search_config())
            session = apply_config_revision(session, {
                "calculation.mlip_version": "m1", "dft.parameters": {"encut": 520},
                "supercomputer.path_mappings": [{"windows_local": r"E:\batches", "linux_remote": "/scratch/batches"}],
            })
            write_json(session_path, session)
            proposed = propose_memory_update({}, category="search_rules", items=["keep endpoints"],
                                             system_id="sys", evidence_refs=["R1"])
            write_json(state_path, proposed["state"])
            fake = SimpleNamespace(state_path=state_path,
                workflow_kwargs={"config_session": session, "config_session_path": str(session_path)})
            control = LocalAgentControl(fake)
            changed = control.patch_config({"supercomputer.path_mappings": [
                {"windows_local": r"E:\batches", "linux_remote": "/scratch/new"}]},
                reasons={"supercomputer.path_mappings": "site path supplied"},
                impacts={"supercomputer.path_mappings": "future manifests only"})
            self.assertEqual(changed["status"], "draft_updated")
            confirmed = control.confirm_config(explicit=True)
            self.assertEqual(confirmed["status"], "confirmed")
            reviewed = control.review_memory(proposed["proposal"]["proposal_id"], approved=True)
            self.assertEqual(reviewed["status"], "approved")

    def test_validated_result_state_refreshes_summary_and_charts(self):
        state = {"confirmed_config_version": "c1", "active_model_version": "m1", "tasks": [],
                 "phase_diagrams": {"diagrams": {"dft": {"entries": []}}}}
        before = build_webui_charts(state)
        state["tasks"] = [{"task_id": "T1", "status": "completed", "stage": "dft_single_point"}]
        state["phase_diagrams"]["diagrams"]["dft"]["entries"] = [
            {"record_id": "R1", "composition": {"x": .5}, "ehull": 0.01}]
        state["recent_results"] = [{"result_id": "R1", "status": "validated"}]
        after = build_webui_charts(state)
        self.assertNotEqual(before["data_version"], after["data_version"])
        self.assertEqual(after["charts"]["convex_hull"]["data"][0]["record_id"], "R1")
        self.assertEqual(build_status_summary(state)["recent_results"][0]["result_id"], "R1")


if __name__ == "__main__":
    unittest.main()
