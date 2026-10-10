"""Backed-up layout-only migration; never recompute or change scientific data."""

from pathlib import Path
from datetime import datetime
import json
import shutil
from phase_agent.analysis.feedback.export_dft_products import dft_product_path


def migrate_analysis_outputs(workspace):
    workspace = Path(workspace).resolve()
    old = workspace / "current" / "phase_diagrams"
    target = workspace / "outputs"
    if not old.is_dir() or old.is_symlink() or target.exists():
        raise ValueError("migration requires original analysis directory and absent outputs target")
    paths = {}
    for source in old.rglob("*"):
        if not source.is_file():
            continue
        parts = source.relative_to(old).parts
        if parts[0] == "dft_results":
            destination = target / "legacy" / source.relative_to(old)
        elif "dft_rounds" in parts:
            i = parts.index("dft_rounds")
            root = target.joinpath(*parts[:i], *parts[i + 1 : -1])
            destination = dft_product_path(root, parts[-1])
        elif parts[-1].startswith("phase_diagram"):
            if len(parts) == 1:
                destination = target / "legacy" / parts[-1]
            else:
                sub = "history" if parts[-1] != "phase_diagram.csv" else ""
                destination = target / parts[0] / "phase_diagrams" / sub / parts[-1]
        else:
            destination = target.joinpath(*parts)
        if destination in paths.values():
            raise ValueError(f"duplicate migration destination: {destination}")
        paths[source] = destination
    for source in paths:
        if source.suffix == ".json":
            json.loads(source.read_text(encoding="utf-8"))
    backup = workspace / "backups" / "output_layout" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    shutil.copytree(old, backup / "phase_diagrams")
    references = [
        workspace / name
        for name in ("config_session.json", "search_config.project.json", "agent_runtime.json")
    ]
    references += [workspace / "current" / "state.json"]
    replacements = {str(a): str(b) for a, b in paths.items()}
    for source, destination in paths.items():
        if "dft_rounds" in source.parts:
            replacements[str(source.parent)] = str(destination.parent.parent)
    replacements[str(old)] = str(target)
    replacements["current/phase_diagrams"] = "outputs"

    def rewrite(value):
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for a, b in sorted(replacements.items(), key=lambda item: -len(item[0])):
                a = a.replace("\\", "/")
                if normalized == a or normalized.startswith(a + "/"):
                    return b.replace("\\", "/") + normalized[len(a) :]
        return value

    # Backup every reference before changes. Rewriting only strings preserves labels.
    for path in references:
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    for source, destination in paths.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    for path in [*references, *target.rglob("*.json")]:
        if path.is_file():
            original = json.loads(path.read_text(encoding="utf-8"))
            changed = rewrite(original)
            if original != changed:
                temporary = path.with_name(path.name + ".layout.tmp")
                temporary.write_text(
                    json.dumps(changed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(path)
    for directory in sorted(old.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if not any(old.iterdir()):
        old.rmdir()
    return {"files": len(paths), "backup": str(backup), "outputs": str(target)}
