"""Readable English handoff names; package fingerprints stay in machine records."""

import hashlib
import json
from pathlib import Path
import re
import zipfile


def handoff_label(directory, scripts):
    directory = Path(directory)
    project = next(
        (p.name for p in directory.parents if re.match(r"^(O3|P2|P3)-", p.name)), "Project"
    )
    material = re.sub(r"^Na|O2$", "", project.rsplit("-", 1)[-1])
    material = re.sub(r"[^A-Za-z0-9]+", "_", material).strip("_") or "Project"
    epoch = next(
        (
            re.match(r"epoch(\d+)", p.name)
            for p in [directory, *directory.parents]
            if re.match(r"epoch(\d+)", p.name)
        ),
        None,
    )
    round_number = int(epoch.group(1)) if epoch else 1
    counts = {}
    refresh = False
    training = False
    for script in scripts:
        path = directory / script
        text = str(path).lower()
        refresh = refresh or "model-refresh" in text
        training = training or (path.parent / "training_plan.json").is_file()
        if not refresh:
            match = re.search(r"(?:round-|dft-)(\d+)", text)
            if match:
                round_number = int(match.group(1))
        manifest = next(
            (p / "manifest.json" for p in path.parents if (p / "manifest.json").is_file()),
            path.parent / "manifest.json",
        )
        entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
        if not isinstance(entries, list):
            entries = []
        task_file = path.parent / "task.json"
        if task_file.is_file():
            task_id = json.loads(task_file.read_text(encoding="utf-8")).get("task_id")
            entries = [entry for entry in entries if entry.get("task_id") == task_id]
        for entry in entries:
            stage = entry.get("stage", "")
            if stage == "dft_single_point":
                kind = "DFTSP"
            elif stage == "dft_relax":
                kind = "DFTRelax"
            elif stage == "deep_search" or "mc-" in text or "mc_" in text:
                kind = "MC"
            elif "single-point" in text or "single_point" in text:
                kind = "SP"
            else:
                kind = "Relax"
            counts[kind] = counts.get(kind, 0) + 1
    operation = (
        "ModelRefresh"
        if refresh
        else "Finetune"
        if training
        else next(iter(counts))
        if len(counts) == 1
        else "Tasks"
    )
    suffix = "_".join(
        f"{kind}{counts[kind]}"
        for kind in ("SP", "Relax", "MC", "DFTSP", "DFTRelax")
        if counts.get(kind)
    )
    return f"{material}_{operation}_Round{round_number:02d}" + (f"_{suffix}" if suffix else "")


def package_folder(parent, label, fingerprint):
    """Reuse identical inputs; give different packages a visible batch sequence."""
    parent = Path(parent)
    for sequence in range(1, 10000):
        folder = parent / (label if sequence == 1 else f"{label}_Batch{sequence:02d}")
        if not folder.exists():
            return folder
        archive = folder / "inputs.zip"
        if archive.is_file():
            with zipfile.ZipFile(archive) as bundle:
                rows = json.loads(bundle.read("transfer_manifest.json"))["files"]
            identity = hashlib.sha256()
            for row in rows:
                identity.update(row["path"].encode())
                identity.update(bytes.fromhex(row["sha256"]))
            if identity.hexdigest()[:16] == fingerprint:
                return folder
    raise ValueError("Too many handoff batches for the same label")
