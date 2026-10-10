"""Read-only transport/configuration validation; no scientific analysis."""

import argparse
import json
from pathlib import Path

from phase_agent.configuration.session.resolve_workspace_paths import resolve_workspace_paths


def verify_workspace_layout(workspace, *, backup=None):
    root = Path(workspace).resolve()
    settings = json.loads((root / "agent_runtime.json").read_text(encoding="utf-8"))
    storage = settings["runtime_storage_override"]
    paths = resolve_workspace_paths({"storage": storage}, base_directory=root)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    required = [
        paths["state"],
        paths["ledger"],
        root / settings["config_session_path"],
        root / settings["editable_config_draft_path"],
        paths["phase_diagrams"] / "output_index.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"必需文件缺失：{missing}")
    session = json.loads((root / settings["config_session_path"]).read_text(encoding="utf-8"))
    if settings.get("local_path_relocations"):
        from phase_agent.configuration.session.relocate_workspace_paths import (
            relocate_workspace_paths,
        )

        relocated = relocate_workspace_paths(
            (session.get("confirmed_snapshot") or {}).get("config") or {},
            settings["local_path_relocations"],
            root,
        )
        for phase, value in ((relocated.get("system") or {}).get("phase_references") or {}).items():
            if isinstance(value, str) and not Path(value).is_file():
                raise ValueError(f"搬迁后的母结构不存在：{phase} {value}")
    if state.get("confirmed_config_version") != (session.get("confirmed_snapshot") or {}).get(
        "config_version"
    ):
        raise ValueError("state与配置会话版本不同；不自动批准迁移或改科学设置")
    if backup:
        old = json.loads((Path(backup) / "metadata/current/state.json").read_text(encoding="utf-8"))
        for key in (
            "confirmed_config",
            "confirmed_config_version",
            "confirmed_config_hash",
            "active_model_version",
        ):
            if old.get(key) != state.get(key):
                raise ValueError(f"冻结配置或模型标识发生变化：{key}")
        for key in ("cost_history", "budget_usage", "budget_reservations"):
            if old.get(key) != state.get(key):
                raise ValueError(f"成本或预算记录发生变化：{key}")
        for key in ("tasks", "dft_dataset_records", "dft_training_records", "new_dft_records"):
            before, after = old.get(key) or [], state.get(key) or []
            if len(before) != len(after):
                raise ValueError(f"数据条目数量发生变化：{key}")
            for a, b in zip(before, after):
                for field in (
                    "task_id",
                    "task_key",
                    "status",
                    "energy",
                    "forces",
                    "stress",
                    "structure",
                    "magnetic_moments",
                    "spin_state_check",
                    "training_frames",
                ):
                    if a.get(field) != b.get(field):
                        raise ValueError(f"计算结果发生变化：{key}.{field}")
    return {
        "status": "verified",
        "state_path": str(paths["state"]),
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "scientific_data_recomputed": False,
        "missing_training_inputs": [
            job["directory"]
            for job in (state.get("remote_finetune_jobs") or {}).values()
            if not Path(job["directory"]).is_dir()
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--backup")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_workspace_layout(args.workspace, backup=args.backup),
            ensure_ascii=False,
            indent=2,
        )
    )
