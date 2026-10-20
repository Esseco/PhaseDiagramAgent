from pathlib import Path
import hashlib, json, os, zipfile

import pytest

from phase_agent.tools.remote.transfer_bundle import import_bundle, MANIFEST


def extended(path):
    text = os.path.abspath(path)
    return (
        Path("\\\\?\\" + text) if os.name == "nt" and not text.startswith("\\\\?\\") else Path(text)
    )


def bundle(tmp_path, *, corrupt=False):
    name = "/".join(
        [
            "jobs",
            "epoch0_model_" + "m" * 45,
            "G0001_Relax_MC_round_0001",
            "inputs",
            "MC-sampling_remote_" + "a" * 35,
            "00000-MC-" + "b" * 24,
            "pool",
            "rank_001_step_0005.vasp",
        ]
    )
    payload = b"original scientific structure bytes"
    rows = [
        {"path": "result.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
        {"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
    ]
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("result.json", b"{}")
        zipped.writestr(name, b"corrupt" if corrupt else payload)
        zipped.writestr(MANIFEST, json.dumps({"schema": 1, "kind": "results", "files": rows}))
    return archive, name, payload


def test_long_return_extracts_and_is_idempotent(tmp_path):
    archive, name, payload = bundle(tmp_path)
    destination = tmp_path / "local" / "verified"
    target = extended(destination / name)
    assert len(str(target)) > 260
    assert import_bundle(archive, destination) == 2
    assert target.read_bytes() == payload
    assert import_bundle(archive, destination) == 2


def test_bad_long_result_does_not_publish_good_sibling(tmp_path):
    archive, name, payload = bundle(tmp_path, corrupt=True)
    destination = tmp_path / "local"
    with pytest.raises(ValueError, match="checksum"):
        import_bundle(archive, destination)
    assert not (destination / "result.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended path syntax")
def test_extended_drive_and_unc_are_idempotent_without_network_access():
    from phase_agent.tools.remote.transfer_bundle import local_path

    assert str(local_path(r"C:\science\results")) == r"\\?\C:\science\results"
    assert str(local_path(r"\\server\share\results")) == r"\\?\UNC\server\share\results"
    assert local_path(local_path(r"C:\science\results")) == local_path(r"C:\science\results")


def test_preflight_reports_staging_length_before_creating_destination(tmp_path):
    from phase_agent.tools.remote.transfer_bundle import check_return_paths

    archive, name, payload = bundle(tmp_path)
    destination = tmp_path / "not_created"
    report = check_return_paths(archive, destination)
    assert report["status"] == "passed"
    assert report["longest_staged_path_length"] >= 260
    assert report["longest_member"] == name
    assert report["legacy_limit_paths"] == 1
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows path restrictions")
@pytest.mark.parametrize("name", ["CON.vasp", "pool/" + "a" * 256 + ".vasp", "pool/bad?.vasp"])
def test_preflight_rejects_windows_incompatible_members_before_writes(tmp_path, name):
    archive = tmp_path / "invalid.zip"
    payload = b"structure"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, payload)
        zipped.writestr(
            MANIFEST,
            json.dumps(
                {
                    "schema": 1,
                    "kind": "results",
                    "files": [
                        {
                            "path": name,
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ],
                }
            ),
        )
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="阶段=解包前.*包内文件存在=是.*临时路径长度"):
        import_bundle(archive, destination)
    assert not destination.exists()


def test_extraction_os_error_names_stage_member_and_length(tmp_path, monkeypatch):
    archive, name, payload = bundle(tmp_path)
    original = Path.open

    def fail_payload(path, mode="r", *args, **kwargs):
        if path.name == "rank_001_step_0005.vasp" and mode == "wb":
            raise PermissionError(13, "access denied", str(path))
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_payload)
    destination = tmp_path / "verified"
    with pytest.raises(
        ValueError, match="阶段=解包与校验.*包内文件存在=是.*路径长度=.*系统错误码=13"
    ):
        import_bundle(archive, destination)
    assert not (destination / "result.json").exists()


def test_recovery_os_error_is_actionable_and_keeps_tasks_pending(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch import results

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"placeholder")
    state = {
        "tasks": [{"task_id": "t", "status": "pending"}],
        "relax_mc_batches": {"b": {"expected_results_archive": str(archive), "status": "prepared"}},
    }

    def denied(**kwargs):
        raise PermissionError(13, "access denied", str(tmp_path / "final.vasp"))

    monkeypatch.setattr(results, "import_results", denied)
    updated = results.recover_available(state, {})
    error = updated["relax_mc_batches"]["b"]["return_error"]
    assert "阶段=本地回收文件读写" in error and "路径长度=" in error and "系统错误码=13" in error
    assert "不要整批重投计算" in error and str(archive) in error
    assert updated["tasks"] == state["tasks"]
    assert "return_error" not in state["relax_mc_batches"]["b"]
