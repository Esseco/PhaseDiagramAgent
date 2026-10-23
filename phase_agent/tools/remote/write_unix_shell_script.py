"""Write Slurm shell scripts with LF endings on every host OS."""

from pathlib import Path


def write_unix_shell_script(path, content):
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if (
        b"#SBATCH" in data
        and Path(path).name != "submit_all.sh"
        and b"# GPU_USAGE_CLOCK" not in data
    ):
        text = data.decode("utf-8")
        lines = text.splitlines(keepends=True)
        index = 0
        while index < len(lines) and (
            not lines[index].strip() or lines[index].lstrip().startswith("#")
        ):
            index += 1
        hook = """# GPU_USAGE_CLOCK
_gpu_usage_directory="${SLURM_SUBMIT_DIR:-$PWD}"
_gpu_usage_started=$(date +%s)
_gpu_usage_finish() {
    _gpu_usage_exit=$?
    trap - EXIT
    python3 "$_gpu_usage_directory/_gpu_accounting.py" "$_gpu_usage_started" "$_gpu_usage_exit" || echo "GPU usage record unavailable" >&2
    exit "$_gpu_usage_exit"
}
trap _gpu_usage_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"""
        text = "".join(lines[:index]) + hook + "".join(lines[index:])
        data = text.encode("utf-8")
        directory = Path(path).parent
        directory.mkdir(parents=True, exist_ok=True)
        # Pure-stdlib helpers are portable; no phase_agent installation is needed.
        for source, target in [
            ("gpu_job_usage.py", "_gpu_accounting.py"),
            ("transfer_bundle.py", "_gpu_transfer_bundle.py"),
        ]:
            payload = Path(__file__).with_name(source).read_bytes()
            destination = directory / target
            if not destination.is_file() or destination.read_bytes() != payload:
                destination.write_bytes(payload)
    if not Path(path).is_file() or Path(path).read_bytes() != data:
        Path(path).write_bytes(data)
