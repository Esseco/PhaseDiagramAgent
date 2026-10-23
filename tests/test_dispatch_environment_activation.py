import pytest

from phase_agent.tools.remote.submission_handoff import write_dispatch_entry


@pytest.mark.parametrize("environment", ["mace", "custom-mace"])
def test_dispatch_activates_same_conda_environment_as_child(tmp_path, environment):
    source = "/data/app/anaconda3/2024.10-1/bin/activate"
    template = (
        f"#!/bin/bash\n#SBATCH --partition=v100m3\n"
        f"source {source}\nconda activate {environment}\npython3 run_mlip_batch.py\n"
    )
    script, _ = write_dispatch_entry(tmp_path, template)
    content = script.read_text(encoding="utf-8")
    assert f"source {source}\nconda activate {environment}\n" in content
    assert content.index(f"conda activate {environment}") < content.index(
        "python3 dispatch_tasks.py"
    )
    assert "python3 run_mlip_batch.py" not in content


def test_dispatch_preserves_single_source_activation(tmp_path):
    script, _ = write_dispatch_entry(
        tmp_path, "#!/bin/bash\nsource /opt/venv/bin/activate\npython3 child.py\n"
    )
    assert "source /opt/venv/bin/activate\npython3 dispatch_tasks.py" in script.read_text(
        encoding="utf-8"
    )


def test_regenerating_entry_preserves_archive_and_submission_receipt(tmp_path):
    (tmp_path / "inputs.zip").write_bytes(b"immutable archive")
    (tmp_path / ".submitted_jobs.json").write_text(
        '{"jobs":{"old":{"status":"submitted","job_id":"123"}}}'
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    write_dispatch_entry(
        tmp_path, "#!/bin/bash\nsource /opt/conda/bin/activate\nconda activate mace\n"
    )
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
