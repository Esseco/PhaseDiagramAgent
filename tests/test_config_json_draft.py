import json
import tempfile
import unittest
from pathlib import Path

from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.session.create_config_draft import create_config_draft
from config_layer.session.create_editable_config_json import create_editable_config_json
from config_layer.session.load_editable_config_json import (
    _strip_jsonc_comments, config_leaf_patch, load_editable_config_json,
)
from config_layer.session.resolve_phase_reference_directory import resolve_phase_reference_directory
from config_layer.session.resolve_workspace_paths import (
    default_workspace_storage, resolve_workspace_paths,
)
from run.configuration_chat import ConfigurationChatHandler


class EditableConfigJsonTests(unittest.TestCase):
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
            self.assertEqual(
                document["config"]["system"]["boundary"]["P"],
                config["system"]["constraints"]["phases"],
            )
            self.assertEqual(
                document["config"]["system"]["boundary"]["TM_ratio"],
                config["system"]["constraints"]["TM_ratio"],
            )
            document["config"]["system"]["boundary"] = {
                "P": ["O3"],
                "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
                "TM_ratio": {"Fe": 1, "Mn": 1},
            }
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
            draft_path = root / "search_config.draft.json"
            create_editable_config_json(draft_path, config)
            document = json.loads(_strip_jsonc_comments(draft_path.read_text(encoding="utf-8")))
            document["config"]["calculation"]["mlip_version"] = "user-model-v2"
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
            self.assertFalse((root / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
