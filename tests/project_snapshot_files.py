"""Read-only project fixtures: preserve real historical geometry and bindings."""

from pathlib import Path
import shutil


def copy_history_files(value, original, local):
    original, local = Path(original), Path(local)
    files = set()

    def add(value):
        path = Path(value)
        if path.is_relative_to(original) and path.is_file():
            files.add(path)

    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            name = item.replace("\\", "/").split("/")[-1]
            if item.lower().endswith((".vasp", ".poscar", ".cif", ".xyz")) or name in {
                "POSCAR",
                "CONTCAR",
                "task.json",
                "manifest.json",
                "batch.snapshot.json",
            }:
                add(item)

    visit(value)
    for batch in value.get("slurm_batches") or []:
        for directory in batch.get("task_directories") or []:
            add(Path(directory) / "task.json")
        if batch.get("upload_directory"):
            directory = Path(batch["upload_directory"])
            for name in ("manifest.json", "batch.snapshot.json"):
                add(directory / name)
            if directory.is_dir():
                for path in directory.glob("*/task.json"):
                    add(path)
    # Include the actual already-imported Relax/MC archive; otherwise its normal
    # inspection/no-op recovery cost and compatibility are absent from the test.
    from phase_agent.tools.relax_mc_batch.handoff import result_archive

    for batch in (value.get("relax_mc_batches") or {}).values():
        add(result_archive(batch))
    for path in files:
        target = local / path.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(files)
