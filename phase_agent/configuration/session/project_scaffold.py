"""Stage and read back a new data workspace before publishing owned files."""

from pathlib import Path
import json
import shutil
import tempfile

from phase_agent.configuration.runtime.project_template import (
    DEFAULT_PROJECT_SETTINGS,
    write_project_entry,
)


def _build(root, staged, execution_mode):
    target = staged / "agent_runtime.json"
    session_path = staged / DEFAULT_PROJECT_SETTINGS["config_session_path"]
    draft_path = staged / DEFAULT_PROJECT_SETTINGS["editable_config_draft_path"]
    settings = {
        **DEFAULT_PROJECT_SETTINGS,
        "execution_mode": execution_mode,
        "run_steps_per_click": 1,
    }
    target.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.configuration.session.apply_config_revision import apply_config_revision
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.project_config_json import create_project_config_json
    from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
    from phase_agent.configuration.session.save_config_session import save_config_session
    from phase_agent.configuration.runtime.project_template import BOOTSTRAP_HINTS

    config = default_layered_search_config()
    # New projects must not inherit the previous chemistry or an HPC model path.
    config["mlip"]["model_path"] = None
    config["system"]["constraints"]["TM_ratio"] = {}
    storage = default_workspace_storage(root)
    session = create_config_draft(config, require_workspace_path=True)
    session["bootstrap_hints"] = BOOTSTRAP_HINTS
    create_project_config_json(
        draft_path,
        config,
        bootstrap_hints={
            **BOOTSTRAP_HINTS,
            "local_initial_structure_directory": str(root / "structures" / "reference_structures"),
        },
        workspace_defaults=storage,
    )
    from phase_agent.configuration.session.split_project_config import split_project_file

    split_project_file(draft_path)
    session = apply_config_revision(
        session,
        {"storage": storage},
        reasons={"storage": "用户在本地启动器中选择了此独立项目工作区。"},
        author="user_selected_project_workspace",
    )
    session["setup_stage"] = "json_ready"
    session["editable_config_json_path"] = str(
        root / DEFAULT_PROJECT_SETTINGS["editable_config_draft_path"]
    )
    save_config_session(session, session_path)
    from phase_agent.analysis.feedback.workspace_guide import publish_workspace_guide

    publish_workspace_guide(staged)
    write_project_entry(staged)
    # Use the real readers to verify the generated split documents before publication.
    from phase_agent.configuration.session.split_project_config import (
        read_document,
        combine_project_documents,
    )

    document = read_document(draft_path)
    combine_project_documents(document, draft_path)
    assert json.loads(session_path.read_text(encoding="utf-8"))["status"] == "draft"


def create_project_scaffold(root, *, execution_mode="debug"):
    root = Path(root).resolve()
    created, created_dirs = [], []
    try:
        with tempfile.TemporaryDirectory(prefix=".project-", dir=root) as temporary:
            staged = Path(temporary)
            _build(root, staged, execution_mode)
            targets = [
                (source, root / source.relative_to(staged))
                for source in staged.rglob("*")
                if source.is_file()
            ]
            for _, target in targets:
                if not target.resolve().is_relative_to(root) or target.exists():
                    raise FileExistsError(
                        f"Project destination exists or escapes workspace: {target}"
                    )
            for source, target in targets:
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created_dirs.append(parent)
                with target.open("xb") as output:
                    created.append(target)
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, output)
            references = root / "structures/reference_structures"
            if not references.resolve().is_relative_to(root):
                raise ValueError("Reference directory escapes project workspace")
            references.mkdir(parents=True, exist_ok=True)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for directory in reversed(created_dirs):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise
