import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.session.create_config_draft import create_config_draft
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_editable_config_json import create_editable_config_json
from config_layer.session.load_editable_config_json import (
    _strip_jsonc_comments, config_leaf_patch, load_editable_config_json,
)
from config_layer.session.resolve_phase_reference_directory import resolve_phase_reference_directory
from config_layer.session.project_config_json import create_project_config_json, write_project_config_patch
from config_layer.session.project_config_json import expand_project_config
from config_layer.session.resolve_workspace_paths import (
    default_workspace_storage, resolve_workspace_paths,
)
from config_layer.session.validate_workspace_root import validate_workspace_root
from run.configuration_chat import ConfigurationChatHandler
from run.configuration_chat import _extract_workspace_path, _parse_workspace_setup_values


class EditableConfigJsonTests(unittest.TestCase):
    def test_reconfirm_unchanged_config_reuses_snapshot_version(self):
        draft = create_config_draft(default_layered_search_config())
        confirmed = confirm_config_snapshot(draft, user_confirmed=True)
        self.assertEqual(confirmed["status"], "confirmed")
        repeated = dict(confirmed, status="draft", draft_revision=confirmed["draft_revision"] + 1)
        same = confirm_config_snapshot(repeated, user_confirmed=True)
        self.assertEqual(same["confirmed_snapshot"]["config_version"],
                         confirmed["confirmed_snapshot"]["config_version"])
        changed = json.loads(json.dumps(repeated))
        changed["config"]["round_strategy"]["rule_default"]["mc_budget"] += 1
        different = confirm_config_snapshot(changed, user_confirmed=True)
        self.assertNotEqual(different["confirmed_snapshot"]["config_version"],
                            confirmed["confirmed_snapshot"]["config_version"])

    def test_start_reviews_current_file_instead_of_asking_about_old_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "search_config.project.json"
            create_project_config_json(path)
            session = create_config_draft(default_layered_search_config())
            session["setup_stage"] = "json_ready"
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                editable_config_path=path,
                agent_client=lambda _: self.fail("开始不应进入参数修改对话"),
            )
            with patch.object(handler, "_import_and_review_config_json", return_value="reviewed") as review:
                self.assertEqual(handler([{"role": "user", "content": "开始"}]), "reviewed")
            self.assertTrue(review.call_args.kwargs["continue_if_ready"])
            with patch.object(handler, "_import_and_review_config_json", return_value="reviewed") as review:
                self.assertEqual(handler([{"role": "user", "content": "继续"}]), "reviewed")
            self.assertFalse(review.call_args.kwargs["continue_if_ready"])

    def test_fixed_tm_project_config_is_reviewable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_config.project.json"
            create_project_config_json(path)
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            space = document["config"]["system"]["configuration_space"]
            space["roles"]["T"] = "fixed"
            space["fixed_T_source"] = "phase_reference"
            expanded = expand_project_config(document, source=path)
            self.assertEqual(expanded["system"]["branch_schema"]["fields"],
                             ["P", "H", "x"])

    def test_workspace_path_cannot_swallow_agent_model(self):
        self.assertIsNone(_extract_workspace_path(
            r"E:\0-FM-PhaseDiagram agent：V4.1flash"))
        self.assertEqual(_extract_workspace_path(
            r"地址是E:\0-FM-PhaseDiagram"), r"E:\0-FM-PhaseDiagram")
        with self.assertRaisesRegex(ValueError, "Agent 模型文字"):
            validate_workspace_root(r"E:\0-FM-PhaseDiagram     agent：V4.1flash")
        with self.assertRaisesRegex(ValueError, "Agent 模型文字"):
            resolve_workspace_paths({"storage": {
                "workspace_root": r"E:\0-FM-PhaseDiagram     agent：V4.1flash",
                "paths": default_workspace_storage("E:/valid")["paths"],
            }}, base_directory="E:/")
        self.assertEqual(_parse_workspace_setup_values(
            r"E:\0-FM-PhaseDiagram     agent：V4.1flash"),
            (r"E:\0-FM-PhaseDiagram", "deepseek-flash"))

    def test_agent_patch_writes_short_config_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_config.project.json"
            create_project_config_json(path)
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            changes = write_project_config_patch(
                path, {"system.H_generation.size_max": 12}, expected_hash=old_hash)
            self.assertEqual(changes[0]["new"], 12)
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            self.assertEqual(document["config"]["system"]["H_generation"]["size_max"], 12)
            with self.assertRaisesRegex(ValueError, "已被其他编辑修改"):
                write_project_config_patch(
                    path, {"system.H_generation.size_max": 16}, expected_hash=old_hash)

    def test_patch_updates_effective_override_not_shadowed_front_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_config.project.json"
            create_project_config_json(path)
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            document["config"]["run"]["initial_states_per_branch"] = 3
            document["overrides"]["run"] = {"initial_states_per_branch": 4}
            path.write_text(json.dumps(document), encoding="utf-8")
            changes = write_project_config_patch(
                path, {"run.initial_states_per_branch": 5})
            self.assertEqual(len(changes), 1)
            saved = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["overrides"]["run"]["initial_states_per_branch"], 5)
            self.assertEqual(expand_project_config(saved, source=path)["run"]["initial_states_per_branch"], 5)
            self.assertEqual(write_project_config_patch(
                path, {"run.initial_states_per_branch": 5}), [])

    def test_bare_write_command_applies_pending_agent_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "search_config.project.json"
            create_project_config_json(path)
            session = create_config_draft(default_layered_search_config())
            session["setup_stage"] = "json_ready"
            calls = []

            def agent(payload):
                calls.append(payload)
                return {
                    "reply": "已收到模型路径。",
                    "patch": {"mlip.model_path": "/remote/mace-mh-1.model"},
                    "reasons": {"mlip.model_path": "用户明确提供"},
                    "questions": [],
                }

            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                editable_config_path=path, agent_client=agent,
            )
            proposal = handler([{"role": "user", "content": "/remote/mace-mh-1.model"}])
            self.assertIn("回复“写入”", proposal)
            self.assertEqual(expand_project_config(
                json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8"))),
                source=path,
            )["mlip"]["model_path"],
                "/data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model")

            written = handler([{"role": "user", "content": "写入"}])
            self.assertIn("已把 1 项修改写入", written)
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            self.assertEqual(document["config"]["mlip"]["model_path"],
                             "/remote/mace-mh-1.model")
            self.assertGreaterEqual(len(calls), 1)

    def test_template_is_created_once_and_supports_new_boundary_and_phase_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_config.draft.json"
            config = default_layered_search_config()
            self.assertTrue(create_editable_config_json(
                path, config,
                bootstrap_hints={"local_initial_structure_directory": "E:/structures"},
            ))
            original = path.read_text(encoding="utf-8")
            self.assertIn("//", original)
            self.assertEqual(
                load_editable_config_json(path, config)["system"]["phase_reference_directory"],
                "E:/structures",
            )
            path.write_text(original + "\n", encoding="utf-8")
            self.assertFalse(create_editable_config_json(path, config))
            self.assertEqual(path.read_text(encoding="utf-8"), original + "\n")

            document = json.loads(_strip_jsonc_comments(original))
            self.assertEqual(document["config"]["storage"]["workspace_root"], str(path.parent.resolve()))
            self.assertEqual(document["config"]["system"]["boundary"]["P"]["at_x"],
                             {"0": ["P3"], "1": ["O3"]})
            self.assertEqual(document["config"]["system"]["H_generation"]["size_max"], 16)
            self.assertEqual(
                document["config"]["system"]["boundary"]["TM_ratio"],
                config["system"]["constraints"]["TM_ratio"],
            )
            document["config"]["system"]["boundary"] = {
                "P": ["O3"],
                "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
                "TM_ratio": {"Fe": 1, "Mn": 1},
            }
            document["config"]["system"]["H_generation"]["enabled"] = False
            document["config"]["system"]["phase_references"] = {"O3": "O3.vasp"}
            self.assertEqual(document["config"]["system"]["phase_reference_directory"], "E:/structures")
            edited = Path(directory) / "edited.json"
            edited.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_editable_config_json(edited, config)
            patch = config_leaf_patch(config, loaded)
            self.assertEqual(patch["system.boundary"]["P"], ["O3"])
            self.assertEqual(patch["system.phase_references"], {"O3": "O3.vasp"})

    def test_secret_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.json"
            create_editable_config_json(path, default_layered_search_config())
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            document["config"]["agent"]["api_key"] = "must-not-be-here"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "密钥类字段"):
                load_editable_config_json(path, default_layered_search_config())

    def test_single_reference_directory_maps_phase_names(self):
        config = default_layered_search_config()
        config["system"]["boundary"] = {"P": ["O3", "P3"]}
        config["system"]["phase_reference_directory"] = "E:/structures"
        resolved = resolve_phase_reference_directory(config)
        self.assertEqual(resolved["system"]["phase_references"], {
            "O3": str(Path("E:/structures") / "O3.vasp"),
            "P3": str(Path("E:/structures") / "P3.vasp"),
        })

    def test_workspace_paths_are_relative_to_one_root_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            config = {"storage": default_workspace_storage(root)}
            paths = resolve_workspace_paths(config, base_directory=directory)
            self.assertEqual(paths["state"], root.resolve() / "current/state.json")
            config["storage"]["paths"]["state"] = "../outside/state.json"
            with self.assertRaisesRegex(ValueError, "不能越出 workspace_root"):
                resolve_workspace_paths(config, base_directory=directory)

    def test_legacy_json_without_storage_receives_workspace_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.json"
            config = default_layered_search_config()
            create_editable_config_json(path, config)
            document = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
            document["config"].pop("storage")
            path.write_text(json.dumps(document), encoding="utf-8")
            defaults = default_workspace_storage(Path(directory) / "chosen")
            loaded = load_editable_config_json(path, config, default_storage=defaults)
            self.assertEqual(loaded["storage"], defaults)

    def test_import_updates_unconfirmed_draft_and_ignores_agent_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = default_layered_search_config()
            session = create_config_draft(config)
            session["pending_config_patch"] = {
                "patch": {"mlip.model_path": "/stale/model"},
                "source_hash": "stale-hash",
            }
            draft_path = root / "search_config.draft.json"
            create_editable_config_json(draft_path, config)
            document = json.loads(_strip_jsonc_comments(draft_path.read_text(encoding="utf-8")))
            document["config"]["calculation"]["mlip_version"] = "user-model-v2"
            document["config"]["system"]["H_generation"]["enabled"] = False
            draft_path.write_text(json.dumps(document), encoding="utf-8")
            handler = ConfigurationChatHandler(
                {"state_path": str(root / "state.json"), "config_session": session},
                config_session_path=root / "session.json", base_directory=root,
                editable_config_path=draft_path,
                agent_client=lambda _payload: {
                    "reply": "存在待补信息。", "patch": {"calculation.mlip_version": "agent-overwrite"},
                    "questions": ["请确认模型版本。"],
                },
            )

            reply = handler([{"role": "user", "content": "读取配置 JSON"}])
            updated = handler.workflow_kwargs["config_session"]
            self.assertIn(str(draft_path), reply)
            self.assertIn("不会自动确认", reply)
            self.assertEqual(updated["status"], "draft")
            self.assertEqual(updated["config"]["calculation"]["mlip_version"], "user-model-v2")
            self.assertTrue(any(item.get("author") == "user_json_draft" for item in updated["dialogue"]))
            self.assertNotIn("pending_config_patch", updated)
            self.assertFalse((root / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
