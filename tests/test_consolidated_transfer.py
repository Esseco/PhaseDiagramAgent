from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import zipfile
from phase_agent.tools.remote.transfer_bundle import (
    result_bundle,
    checksum,
    import_registered_bundles,
    has_registered_return,
)


def completed(root, name):
    task = root / name
    task.mkdir(parents=True)
    payload = {"task_id": name, "task_key": name, "status": "completed", "outputs": {}}
    (task / "result.json").write_text(json.dumps(payload))
    (task / "task.finished.json").write_text(
        json.dumps({**payload, "result_checksum": checksum(task / "result.json")})
    )
    return task


def test_one_folder_one_archive_grows_and_is_automatically_read(tmp_path):
    remote = tmp_path / "remote/results"
    first = completed(remote, "A")
    result_bundle(remote, [first], "batch1")
    package = remote / "transfer/all.results.zip"
    assert [p.name for p in package.parent.iterdir()] == ["all.results.zip"]
    local = tmp_path / "local/results"
    (local / "transfer").mkdir(parents=True)
    shutil.copy2(package, local / "transfer")
    state = {"slurm_batches": [{"results_directory": str(local)}]}
    assert has_registered_return(state)
    assert import_registered_bundles([local]) == []
    assert (local / "A/result.json").is_file()
    assert not has_registered_return(state)
    second = completed(remote, "B")
    result_bundle(remote, [second], "batch2")
    shutil.copy2(package, local / "transfer")
    assert has_registered_return(state)
    assert import_registered_bundles([local]) == []
    assert (local / "B/result.json").is_file()
    assert import_registered_bundles([local]) == []
    assert not has_registered_return(state)
    with zipfile.ZipFile(package) as archive:
        assert {"A/result.json", "B/result.json"} <= set(archive.namelist())
        assert not any(name.endswith(".zip") for name in archive.namelist())


def test_parallel_finishers_publish_complete_valid_archive(tmp_path):
    root = tmp_path / "results"
    tasks = [completed(root, f"T{i}") for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(result_bundle, root, [task], task.name) for task in tasks
        ]
        for future in futures:
            assert future.result()
    with zipfile.ZipFile(root / "transfer/all.results.zip") as archive:
        assert archive.testzip() is None
        assert all(f"T{i}/task.finished.json" in archive.namelist() for i in range(4))
    assert not list((root / "transfer").glob("*.part"))


def test_consolidated_archive_also_works_when_dropped_in_results_root(tmp_path):
    remote = tmp_path / "remote"
    task = completed(remote, "A")
    result_bundle(remote, [task], "batch")
    local = tmp_path / "local"
    local.mkdir()
    shutil.copy2(remote / "transfer/all.results.zip", local)
    assert import_registered_bundles([local]) == []
    assert (local / "A/result.json").is_file()
