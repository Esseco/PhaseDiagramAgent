"""Explicit offline migration of registered local inputs; no scientific runs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from execution_layer.local.migrate_workspace_layout import rewrite_active_paths, require_agent_offline


def migrate(root, *, apply=False):
    root = Path(root).resolve()
    state_path = root / "workflow_state/state.json"
    original = state_path.read_bytes()
    state = json.loads(original.decode("utf-8"))
    submission_root = (root / "submissions").resolve()
    moves, replacements, training = [], [], []
    for batch in state.get("slurm_batches") or []:
        source = Path(batch["upload_directory"]).resolve()
        if source.parent.name == "inputs" or not source.exists():
            continue
        destination = source.parent / "inputs" / source.name
        moves.append((source, destination))
        replacements.append((str(source), str(destination)))
    for job in (state.get("remote_finetune_jobs") or {}).values():
        directory = Path(job["directory"]).resolve()
        if directory in training or not directory.is_dir() or (directory / "inputs").exists():
            continue
        training.append(directory)
        for source in directory.iterdir():
            if source.name == "results":
                continue
            moves.append((source, directory / "inputs" / source.name))
            replacements.append((str(source), str(directory / "inputs" / source.name)))
    moves = list(dict.fromkeys(moves))
    for source, destination in moves:
        if not source.is_relative_to(submission_root) or not destination.is_relative_to(submission_root):
            raise ValueError(f"路径越界：{source}")
        if destination.exists():
            raise ValueError(f"目标已存在：{destination}")
        if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
               for path in [source, *source.parents, *(source.rglob("*") if source.is_dir() else [])]):
            raise ValueError(f"链接路径不能移动：{source}")
    report = {"moves": [{"from": str(a), "to": str(b)} for a, b in moves],
              "training_rounds": [str(path) for path in training], "apply": apply}
    if not apply or not moves:
        return report
    require_agent_offline(2024)
    require_agent_offline(8765)
    backup = root / "history_backups" / ("round-input-layout-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup.mkdir(parents=True)
    (backup / "state.json").write_bytes(original)
    (backup / "migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    changed_files, completed = [], []
    try:
        if state_path.read_bytes() != original:
            raise ValueError("state已变化，未移动")
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            completed.append((source, destination))
        for directory in training:
            collector = directory / "inputs/collect_training_results.py"
            if collector.exists():
                data = collector.read_bytes()
                changed_files.append((collector, data))
                (backup / (directory.name + "-collect_training_results.py")).write_bytes(data)
                template = Path(__file__).parents[1] / "remote/collect_training_results.py"
                text = template.read_text(encoding="utf-8")
                collector.write_text(text, encoding="utf-8")
            plan = directory / "inputs/training_plan.json"
            if plan.exists():
                data = plan.read_bytes()
                changed_files.append((plan, data))
                (backup / (directory.name + "-training_plan.json")).write_bytes(data)
                updated = rewrite_active_paths(json.loads(data.decode("utf-8")), replacements)
                plan.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
            (directory / "results").mkdir(exist_ok=True)
        updated = rewrite_active_paths(state, replacements)
        updated.setdefault("local_path_relocations", []).extend({"from": a, "to": b} for a, b in replacements)
        updated.setdefault("layout_migrations", []).append({"layout": "round-inputs-results-v1", "backup": str(backup)})
        temporary = state_path.with_suffix(".layout.tmp")
        temporary.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(state_path)
    except Exception:
        for path, data in reversed(changed_files):
            path.write_bytes(data)
        for source, destination in reversed(completed):
            destination.rename(source)
        state_path.write_bytes(original)
        raise
    report["backup"] = str(backup)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, apply=args.apply), ensure_ascii=False, indent=2))
