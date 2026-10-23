"""Submit script, compute-node Python driver and one approved input archive."""

from pathlib import Path
import shutil


from phase_agent.tools.remote.transfer_limits import bootstrap_limits

BOOTSTRAP = (
    bootstrap_limits()
    + Path(__file__)
    .parents[1]
    .joinpath("remote/environment_preflight.py")
    .read_text(encoding="utf-8")
    + "\n"
    + Path(__file__).with_name("upload_driver.py").read_text(encoding="utf-8")
)


def input_directory(directory):
    root = Path(directory).resolve()
    return root.with_name(root.name + "_input")


def output_directory(directory):
    root = Path(directory).resolve()
    return root.with_name(root.name + "_output")


def result_archive(record):
    from phase_agent.tools.relax_mc_batch.recovery_receipt import selected_recovery

    recovery = selected_recovery(record)
    if recovery is not None:
        return recovery
    if not record.get("directory"):
        return Path(record["expected_results_archive"])
    named = output_directory(record["directory"]) / "results.zip"
    for candidate in (
        named,
        Path(record["expected_results_archive"]),
        Path(record["directory"]) / "returned/results.zip",
    ):
        if candidate.is_file():
            return candidate
    return Path(record["expected_results_archive"])


def compact_handoff(directory):
    """Publish a compact view without altering archived inputs or task records."""
    root = Path(directory).resolve()
    archive = root / "inputs.zip"
    original = (root / "submit_all.sh").read_text(encoding="utf-8")
    folder = input_directory(root)
    folder.mkdir(exist_ok=True)
    output_directory(root).mkdir(exist_ok=True)
    target = folder / "inputs.zip"
    if target.exists():
        from phase_agent.tools.remote.integrity import file_checksum

        if file_checksum(target) != file_checksum(archive):
            raise ValueError("Existing upload package differs; inspect before retry")
    else:
        shutil.copy2(archive, target)
    # Keep the coordinator's allocation and activation; replace only its command.
    prefix = original.split('cd -- "${SLURM_SUBMIT_DIR', 1)
    if len(prefix) != 2:
        raise ValueError("Unsupported coordinator script")
    guard = ': "${SLURM_JOB_ID:?Use sbatch submit_all.sh; Python runs on the compute node}"\n'
    header = prefix[0].replace("set -euo pipefail\n", "set -euo pipefail\n" + guard, 1)
    if guard not in header:
        raise ValueError("Coordinator script lacks strict shell setup")
    command = (
        'cd -- "${SLURM_SUBMIT_DIR:?Submit from the upload directory}"\npython3 run_upload.py\n'
    )
    (folder / "run_upload.py").write_text(BOOTSTRAP, encoding="utf-8", newline="\n")
    (folder / "submit_all.sh").write_text(header + command, encoding="utf-8", newline="\n")
    from phase_agent.tools.remote.input_validation import validate_handoff

    validate_handoff(folder, kind="relax_mc")
    return folder
