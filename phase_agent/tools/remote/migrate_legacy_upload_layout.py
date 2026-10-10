"""One-time, recoverable migration of flat legacy upload batch folders."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil

from phase_agent.tools.remote.build_upload_batch_directory import build_upload_batch_directory


def migrate_legacy_upload_layout(state_path, upload_root, *, dry_run=False):
    state_path = Path(state_path).resolve()
    root = Path(upload_root).resolve()
    original = json.loads(state_path.read_text(encoding="utf-8"))
    current = deepcopy(original)
    plans = _build_plans(current, root)
    if dry_run or not plans:
        return {
            "status": "planned" if plans else "already_migrated",
            "batch_count": len(plans),
            "plans": plans,
            "state": current,
        }

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = state_path.with_name(f"{state_path.stem}.before-readable-layout-{stamp}.json")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(state_path, backup)
    moved = []
    try:
        for plan in plans:
            source, target = Path(plan["source"]), Path(plan["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        replacements = [(row["source"], row["target"]) for row in plans]
        current = _replace_paths(current, replacements)
        by_id = {row.get("batch_id"): row for row in current.get("slurm_batches") or []}
        for plan in plans:
            by_id[plan["batch_id"]].update(plan["layout"])
        _atomic_write(state_path, current)
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        shutil.copy2(backup, state_path)
        raise

    from phase_agent.tools.remote.write_relax_upload_plan import write_relax_upload_plan

    write_relax_upload_plan(current, root)
    return {
        "status": "migrated",
        "batch_count": len(plans),
        "plans": plans,
        "backup_path": str(backup),
        "state": current,
    }


def _build_plans(state, root):
    plans = []
    for batch in state.get("slurm_batches") or []:
        source = Path(batch.get("upload_directory") or (root / batch["batch_id"])).resolve()
        if source.parent != root or not source.name.startswith("remote-"):
            continue
        if batch.get("job_id") or batch.get("status") in {"submitted", "running", "submitting"}:
            raise ValueError(f"active batch cannot be moved: {batch.get('batch_id')}")
        if not source.is_dir():
            raise FileNotFoundError(source)
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = {row.get("stage") for row in manifest}
        if len(stages) != 1:
            raise ValueError(f"batch has mixed or missing stages: {batch.get('batch_id')}")
        model_version = batch.get("model_version") or (
            manifest[0].get("model_version") if manifest else None
        )
        target, layout = build_upload_batch_directory(
            root,
            state,
            batch_id=batch["batch_id"],
            stage=next(iter(stages)),
            model_version=model_version,
        )
        target = target.resolve()
        if target.exists() or any(row["target"] == str(target) for row in plans):
            raise FileExistsError(target)
        batch.update(layout)
        plans.append(
            {
                "batch_id": batch["batch_id"],
                "source": str(source),
                "target": str(target),
                "layout": layout,
            }
        )
    return plans


def _replace_paths(value, replacements):
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    for old, new in replacements:
        for old_form, new_form in ((old, new), (Path(old).as_posix(), Path(new).as_posix())):
            if value.lower() == old_form.lower():
                return new_form
            prefix = old_form.rstrip("/\\")
            if value.lower().startswith(prefix.lower() + ("/" if "/" in value else "\\")):
                return new_form.rstrip("/\\") + value[len(prefix) :]
    return value


def _atomic_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--upload-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = migrate_legacy_upload_layout(args.state, args.upload_root, dry_run=args.dry_run)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "state"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
