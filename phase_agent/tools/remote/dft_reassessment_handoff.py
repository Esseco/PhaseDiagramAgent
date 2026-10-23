"""Portable GPU recovery using original DFT artifacts and existing pair generation."""

from pathlib import Path
import hashlib
import json
import shlex
import tempfile
import zipfile

from phase_agent.tools.remote.legacy_recovery import (
    recovery_plan,
    portable_transfer_source,
    _descriptor,
)
from phase_agent.tools.remote.transfer_bundle import create_bundle, validate_bundle
from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script


RUNTIME = (
    "tools/remote/dft_result_reassessment.py",
    "tools/remote/legacy_recovery_runner.py",
    "tools/remote/predict_dft_final_frame.py",
    "tools/remote/integrity.py",
    "tools/remote/dft_mlip_pair.py",
    "science/structures/load_result_structure.py",
    "science/mlip/run_mace_subprocess.py",
    "science/mlip/mace_worker.py",
    "science/mlip/predict_structure.py",
    "science/qbc/summarize_member_energies.py",
)


def prepare_reassessment_handoff(state, original, *, source_directory, returned_archive):
    """Add a new sibling only; no mutation of state or previously exported packages."""
    plan = recovery_plan(state, original)
    validate_bundle(returned_archive)
    options = {"schema": 1, "eligible_tasks": {}, "model": None}
    with (
        zipfile.ZipFile(original["upload_archive"]) as inputs,
        zipfile.ZipFile(returned_archive) as returned,
    ):
        for task in plan["tasks"]:
            name = task["result_directory"] + "/result.json"
            payload = returned.read(name)
            row = json.loads(payload)
            receipt = json.loads(returned.read(task["result_directory"] + "/task.finished.json"))
            digest = hashlib.sha256(payload).hexdigest()
            if receipt.get("result_checksum") != digest or receipt.get("status") != row.get(
                "status"
            ):
                raise ValueError("reassessment_source_receipt_invalid")
            if any(
                receipt.get(k) != v or row.get(k) != v
                for k, v in task["expected"].items()
                if v is not None
            ):
                raise ValueError("reassessment_source_task_identity_invalid")
            outputs = row.get("outputs") or {}
            if row.get("status") != "failed" or not all(
                outputs.get(k) is True
                for k in (
                    "xml_complete",
                    "electronic_converged",
                    "final_frame_valid",
                    "training_ready",
                )
            ):
                continue
            worker = task["input_directory"] + "/"
            config = json.loads(inputs.read(worker + "workflow.json"))
            if config.get("calculation") != "static" or config.get("previous_result"):
                raise ValueError("reassessment_only_standalone_static_supported")
            model_payload = inputs.read(worker + "comparison_model.json")
            model = json.loads(model_payload)
            if options["model"] is not None and options["model"] != model:
                raise ValueError("reassessment_mixed_original_models")
            options["model"] = model
            options["eligible_tasks"][row["task_id"]] = {
                "original_result_sha256": digest,
                "initial_structure_sha256": hashlib.sha256(
                    inputs.read(worker + config["structure_file"])
                ).hexdigest(),
                "comparison_model_sha256": hashlib.sha256(model_payload).hexdigest(),
            }
    if not options["eligible_tasks"]:
        raise ValueError("no_verified_final_frames_for_reassessment")
    if not Path(source_directory).is_absolute() and not str(source_directory).startswith("/"):
        raise ValueError("reassessment_source_directory_must_be_absolute")
    from Process_Vasp.workflows import atomate_runner

    with tempfile.TemporaryDirectory(prefix="reassess-input-") as temporary:
        root = Path(temporary)
        (root / "recovery_plan.json").write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        (root / "repair_options.json").write_text(
            json.dumps(options, sort_keys=True), encoding="utf-8"
        )
        (root / "_transfer.py").write_text(portable_transfer_source(), encoding="utf-8")
        app = Path(__file__).parents[2]
        for relative in RUNTIME:
            target = root / "runtime/phase_agent" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((app / relative).read_bytes())
        target = root / "runtime/Process_Vasp/workflows/atomate_runner.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(atomate_runner.__file__).read_bytes())
        for directory in list((root / "runtime").rglob("*")):
            if directory.is_dir():
                (directory / "__init__.py").write_text("", encoding="utf-8")
        payloads = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        extras = [p for p in payloads if p not in {"recovery_plan.json", "_transfer.py"}]
        runner = Path(__file__).with_name("legacy_recovery_runner.py").read_text(encoding="utf-8")
        runner = runner.rsplit('\nif __name__ == "__main__":', 1)[0]
        runner += """
def _reassessment_factory(work, plan):
    runtime = str((work / "runtime").resolve())
    sys.path.insert(0, runtime)
    os.environ["PYTHONPATH"] = runtime + os.pathsep + os.environ.get("PYTHONPATH", "")
    from phase_agent.tools.remote.dft_result_reassessment import reassessment_transform
    options = json.loads((work / "repair_options.json").read_text(encoding="utf-8"))
    expected = {task["expected"]["task_id"] for task in plan["tasks"]}
    if options.get("schema") != 1 or not set(options["eligible_tasks"]).issubset(expected):
        raise ValueError("reassessment_scope_mismatch")
    return reassessment_transform(work, options)

if __name__ == "__main__":
    raise SystemExit(main(extra_payloads=EXTRA_PAYLOADS, transform_factory=_reassessment_factory))
"""
        runner = "EXTRA_PAYLOADS = " + repr(tuple(extras)) + "\n" + runner
        template = Path(__file__).with_name("mlip_gpu_template.sh").read_text(encoding="utf-8")
        environment = options["model"].get("environment")
        if not environment:
            raise ValueError("reassessment_remote_mlip_environment_missing")
        activation = (
            "" if environment == "current" else "conda activate " + shlex.quote(environment)
        )
        command = (
            'set -euo pipefail\n: "${SLURM_JOB_ID:?Use sbatch submit_all.sh}"\n'
            'cd -- "${SLURM_SUBMIT_DIR:?}"\nexport PHASE_RECOVERY_SOURCE='
            + shlex.quote(str(source_directory))
            + "\npython3 recover_results.py"
        )
        script = (
            template.replace("__JOB_NAME__", "phase-dft-recover")
            .replace("conda activate mace", activation)
            .replace("python3 run_mlip_task.py", command)
        )
        # Stop if scientific environment activation fails, before calling Python.
        script = script.replace("export cuda_path=", "set -euo pipefail\nexport cuda_path=", 1)
        source = Path(original["folder"])
        for sequence in range(1, 100):
            folder = source.with_name(
                source.name.removesuffix("_input") + f"_Recovery{sequence:02d}_input"
            )
            output = folder.with_name(folder.name[:-6] + "_output")
            if folder.exists() or output.exists():
                continue
            create_bundle(
                root,
                [p for p in root.rglob("*") if p.is_file()],
                folder / "inputs.zip",
                kind="inputs",
            )
            (folder / "recover_results.py").write_text(runner, encoding="utf-8", newline="\n")
            write_unix_shell_script(folder / "submit_all.sh", script)
            output.mkdir()
            return _descriptor(original, folder, plan)
    raise ValueError("too_many_recovery_deliveries")
