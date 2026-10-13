import pytest
from phase_agent.tools.remote.submission_handoff import (
    dispatcher_gpu_directive,
    write_dispatch_entry,
)


@pytest.mark.parametrize(
    "resource,expected",
    [
        ("--gres=gpu:4", "--gres=gpu:1"),
        ("--gres=gpu:a100:2", "--gres=gpu:a100:1"),
        ("--gres=nic:1,gpu:v100:2", "--gres=gpu:v100:1"),
        ("--gpus=2", "--gres=gpu:1"),
        ("--gpus-per-node=a100:2", "--gres=gpu:a100:1"),
        ("-G 1", "--gres=gpu:1"),
    ],
)
def test_dispatch_reserves_one_gpu_of_required_kind(resource, expected):
    assert dispatcher_gpu_directive("#SBATCH " + resource) == "#SBATCH " + expected


def test_known_queue_policy_and_cpu_queue(tmp_path):
    template = "#SBATCH --partition=v100m3\n"
    script, _ = write_dispatch_entry(tmp_path, template)
    text = script.read_text()
    assert "#SBATCH --gres=gpu:1" in text
    assert text.index("#SBATCH --gres=gpu:1") < text.index("set -euo pipefail")
    assert "SLURM_JOB_ID" in text
    assert dispatcher_gpu_directive("#SBATCH --partition=cpu\n") == ""
