"""Portable batch export for explicit, manual supercomputer upload."""

from __future__ import annotations

import hashlib
from pathlib import Path

from execution_layer.remote.batch_runner import RemoteBatchRunner


class ManualUploadBatchRunner(RemoteBatchRunner):
    """Materialize immutable batches and instructions without submitting them."""

    def prepare(self, state):
        result = super().prepare(state)
        batch = result.get("batch")
        if result.get("status") != "prepared" or not batch:
            return result
        directory = Path(batch["manifest_path"]).parent
        checksums = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.name == "SHA256SUMS":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {path.relative_to(directory).as_posix()}")
        checksum_path = directory / "SHA256SUMS"
        checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
        guide_path = directory / "UPLOAD_AND_SUBMIT.md"
        guide_path.write_text(_guide(batch["batch_id"]), encoding="utf-8")
        batch.update({"upload_directory": str(directory),
                      "checksums_path": str(checksum_path), "upload_guide": str(guide_path)})
        result["batch"] = batch
        result["state"]["slurm_batches"][-1].update({
            "upload_directory": str(directory), "checksums_path": str(checksum_path),
            "upload_guide": str(guide_path),
        })
        return result


def _guide(batch_id):
    return f"""# Manual upload batch `{batch_id}`

This directory was generated locally. Nothing has been submitted.

1. Inspect `manifest.json`, every `task.json`, and `submit.sbatch`.
2. Confirm the model/input files and all site-specific Slurm settings.
3. Upload this entire directory without changing its relative layout.
4. On the remote system, run `sha256sum -c SHA256SUMS`.
5. Only after that manual review, submit with `sbatch submit.sbatch`.
6. Download result JSON, completion markers, final structures/checkpoints and logs
   according to the project's result whitelist, then run the local recovery flow.

The generated script is not evidence that executables, environments, paths,
pseudopotentials, permissions, or resource requests are correct for your site.
"""
