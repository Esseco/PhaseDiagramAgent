"""Offload CPU-heavy branch generation, electrostatic ranking, and deduplication to HPC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import create_bundle

BRANCH_GEN_SCHEMA = "remote-branch-generation-v1"


def prepare_remote_branch_generation_bundle(
    boundary: dict,
    system_config: dict,
    parameters: dict,
    parent_structures: list[dict],
    output_dir: str | Path,
) -> dict:
    """Package only boundary, parameters, and parent structures for CPU HPC execution."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    parents_dir = out / "parents"
    parents_dir.mkdir(exist_ok=True)
    packed_parents = []
    for idx, row in enumerate(parent_structures):
        src = Path(row["structure_path"])
        if not src.is_file():
            raise FileNotFoundError(f"parent structure missing: {src}")
        dst = parents_dir / f"parent_{idx:04d}.vasp"
        shutil.copy2(src, dst)
        packed_parents.append(
            {
                **{k: v for k, v in row.items() if k != "structure_path"},
                "path": f"parents/{dst.name}",
                "sha256": file_checksum(dst),
            }
        )
    manifest = {
        "schema": BRANCH_GEN_SCHEMA,
        "resource_type": "cpu",
        "requires_gpu": False,
        "boundary": boundary,
        "system_config": system_config,
        "parameters": parameters,
        "parents": packed_parents,
    }
    manifest_path = out / "branch_gen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    runner_path = out / "run_branch_gen.py"
    runner_path.write_text(
        "from pathlib import Path\n"
        "from phase_agent.tools.remote.remote_branch_generation import run_remote_branch_generation\n"
        "if __name__ == '__main__':\n"
        "    run_remote_branch_generation(Path('branch_gen_manifest.json'), Path('output'))\n",
        encoding="utf-8",
    )
    slurm_path = out / "submit_cpu.sh"
    slurm_path.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=branch-gen-cpu\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=16\n"
        "set -euo pipefail\n"
        'cd -- "${SLURM_SUBMIT_DIR:?}"\n'
        "python3 run_branch_gen.py\n",
        encoding="utf-8",
        newline="\n",
    )
    files = [manifest_path, runner_path, slurm_path, *(parents_dir.glob("*.vasp"))]
    archive = create_bundle(out, files, out / "inputs.zip", kind="inputs")
    return {
        "status": "prepared",
        "resource_type": "cpu",
        "requires_gpu": False,
        "directory": str(out),
        "upload_archive": str(archive),
        "script_path": str(slurm_path),
        "manifest_path": str(manifest_path),
    }


def run_remote_branch_generation(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    generate_fn=None,
) -> dict:
    """Execute branch enumeration, electrostatic ranking, and deduplication on HPC CPU."""
    mpath = Path(manifest_path).resolve()
    root = mpath.parent
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("schema") != BRANCH_GEN_SCHEMA:
        raise ValueError("remote_branch_gen_schema_invalid")
    for row in manifest.get("parents") or []:
        fpath = (root / row["path"]).resolve()
        if not fpath.is_file() or file_checksum(fpath) != row["sha256"]:
            raise ValueError("remote_branch_gen_parent_checksum_mismatch")

    out = Path(output_dir).resolve()
    struct_dir = out / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    if generate_fn is not None:
        generated = generate_fn(manifest, struct_dir)
    else:
        from phase_agent.science.structures.generate_branch_structures import (
            generate_branch_structures,
        )

        generated = generate_branch_structures(
            boundary=manifest["boundary"],
            system_config=manifest["system_config"],
            parameters=manifest["parameters"],
            output_dir=str(struct_dir),
        )

    exported = []
    files = []
    for idx, item in enumerate(generated.get("structures") or []):
        src = Path(item["structure_path"])
        dst = struct_dir / f"gen_{idx:05d}.vasp"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        files.append(dst)
        exported.append(
            {
                **{k: v for k, v in item.items() if k != "structure_path"},
                "structure_path": f"structures/{dst.name}",
                "structure_sha256": file_checksum(dst),
            }
        )

    result_payload = {
        "schema": BRANCH_GEN_SCHEMA,
        "resource_type": "cpu",
        "branches": generated.get("branches") or [],
        "structures": exported,
        "summary": generated.get("summary") or {"structure_count": len(exported)},
    }
    res_path = out / "generated_branches_result.json"
    res_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(res_path)
    archive = create_bundle(out, files, out / "results.zip", kind="results")
    return {
        "status": "completed",
        "resource_type": "cpu",
        "archive": str(archive),
        "structure_count": len(exported),
    }


def import_remote_branch_generation(
    archive_path: str | Path,
    destination: str | Path,
) -> dict:
    """Verify remote CPU branch generation results locally without rerunning Ewald/Matcher."""
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
    res_path = dest / "generated_branches_result.json"
    if not res_path.is_file():
        raise ValueError("remote_branch_gen_result_missing")
    data = json.loads(res_path.read_text(encoding="utf-8"))
    if data.get("schema") != BRANCH_GEN_SCHEMA:
        raise ValueError("remote_branch_gen_result_schema_invalid")
    verified_structures = []
    for row in data.get("structures") or []:
        fpath = (dest / row["structure_path"]).resolve()
        if not fpath.is_relative_to(dest) or not fpath.is_file():
            raise ValueError("remote_branch_gen_structure_missing")
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual != row.get("structure_sha256"):
            raise ValueError("remote_branch_gen_structure_checksum_mismatch")
        verified_structures.append({**row, "structure_path": str(fpath)})
    return {
        "status": "accepted",
        "branches": data.get("branches") or [],
        "structures": verified_structures,
        "summary": data.get("summary") or {},
    }
