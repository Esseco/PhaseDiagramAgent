from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
import zipfile

import pytest

from phase_agent.tools.relax_mc_batch import plan
from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
from phase_agent.tools.remote.transfer_bundle import checksum
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_relax_mc_batch import fixture


def test_revise_unsent_delivery_keeps_task_identity_and_original_archive(tmp_path, monkeypatch):
    _, _, config, context, action = fixture(tmp_path, count=3)
    config["upload_batches_directory"] = str(tmp_path / "submissions")
    original_profiles = plan.profiles

    def legacy_profiles(config):
        profiles = original_profiles(config)
        profiles["mlip_gpu"].pop("partition_pool")
        return profiles

    monkeypatch.setattr(plan, "profiles", legacy_profiles)
    plan_action = plan.attach_preview(action, context["event_state"], context, config)
    result = prepare_batch(action=plan_action, context=context)
    assert result["status"] == "prepared"
    original = Path(result["upload_archive"])
    original_hash = checksum(original)
    path = tmp_path / "workflow_state" / "state.json"
    path.parent.mkdir()
    state = result["state"]
    tasks = deepcopy(state["tasks"])
    write_json(path, state)
    control = SimpleNamespace(state_path=path, chat_handler=SimpleNamespace(lock=RLock()))

    from phase_agent.tools.relax_mc_batch.revise_delivery import revise_registered_delivery

    args = {"bundle_id": result["bundle_id"], "source_sha256": original_hash}
    with pytest.raises(ValueError, match="remote_unsubmitted"):
        revise_registered_delivery(control, confirmed_unsubmitted=False, **args)
    with pytest.raises(ValueError, match="original_delivery_conflict"):
        revise_registered_delivery(
            control, confirmed_unsubmitted=True, **{**args, "source_sha256": "a" * 64}
        )
    state["tasks"][0]["job_id"] = "100"
    write_json(path, state)
    with pytest.raises(ValueError, match="original_pending_scope"):
        revise_registered_delivery(control, confirmed_unsubmitted=True, **args)
    state["tasks"] = tasks
    write_json(path, state)

    revision = revise_registered_delivery(control, confirmed_unsubmitted=True, **args)
    assert revision["status"] == "revised"
    updated = read_json(path)
    assert updated["tasks"] == tasks
    batch = updated["relax_mc_batches"][result["bundle_id"]]
    assert batch["plan_checksum"] == result["plan_checksum"]
    assert batch["expected_results_archive"] == result["expected_results_archive"]
    assert checksum(original) == original_hash
    with (
        zipfile.ZipFile(batch["upload_archive"]) as new_archive,
        zipfile.ZipFile(original) as old_archive,
    ):
        before = json.loads(old_archive.read("batch.json"))
        after = json.loads(new_archive.read("batch.json"))
        assert after["initial_state"] == before["initial_state"]
        assert after["config"] == before["config"]
        assert after["stage_profiles"]["mlip_gpu"]["partition_pool"] == ["v100m3", "v100g32"]
        assert all(
            new_archive.read(p) == old_archive.read(p)
            for p in old_archive.namelist()
            if p.startswith(("inputs/", "references/"))
        )
    assert (
        revise_registered_delivery(control, confirmed_unsubmitted=True, **args)["status"]
        == "already_revised"
    )


def test_minimal_revision_preserves_original_tasks_history_and_budget(tmp_path):
    _, _, config, context, action = fixture(tmp_path, count=3)
    config["upload_batches_directory"] = str(tmp_path / "submissions")
    plan_action = plan.attach_preview(action, context["event_state"], context, config)
    result = prepare_batch(action=plan_action, context=context)
    assert result["status"] == "prepared"
    original = Path(result["upload_archive"])
    original_hash = checksum(original)
    path = tmp_path / "workflow_state" / "state.json"
    path.parent.mkdir()
    state = result["state"]
    tasks = deepcopy(state["tasks"])
    write_json(path, state)
    control = SimpleNamespace(state_path=path, chat_handler=SimpleNamespace(lock=RLock()))
    from phase_agent.tools.relax_mc_batch.revise_delivery import minimize_registered_delivery

    args = {"bundle_id": result["bundle_id"], "source_sha256": original_hash}
    with pytest.raises(ValueError, match="remote_unsubmitted"):
        minimize_registered_delivery(control, confirmed_unsubmitted=False, **args)
    with pytest.raises(ValueError, match="original_delivery_conflict"):
        minimize_registered_delivery(
            control, confirmed_unsubmitted=True, **{**args, "source_sha256": "a" * 64}
        )
    revision = minimize_registered_delivery(control, confirmed_unsubmitted=True, **args)
    assert revision["status"] == "revised"
    updated = read_json(path)
    assert updated["tasks"] == tasks
    assert checksum(original) == original_hash
    batch = updated["relax_mc_batches"][result["bundle_id"]]
    with zipfile.ZipFile(original) as before, zipfile.ZipFile(batch["upload_archive"]) as after:
        a, b = json.loads(before.read("batch.json")), json.loads(after.read("batch.json"))
        assert (
            a["initial_state"]["budget_reservations"] == b["initial_state"]["budget_reservations"]
        )
        assert a["plan_checksum"] == b["plan_checksum"]
        assert [x["task_id"] for x in a["initial_state"]["tasks"]] == [
            x["task_id"] for x in b["initial_state"]["tasks"]
        ]
        for before_task, after_task in zip(
            a["initial_state"]["tasks"], b["initial_state"]["tasks"], strict=True
        ):
            assert before.read(before_task["structure_path"]) == after.read(
                after_task["structure_path"]
            )
    assert (
        minimize_registered_delivery(control, confirmed_unsubmitted=True, **args)["status"]
        == "already_revised"
    )


def test_minimal_revision_cleans_only_own_unregistered_output_on_state_write_failure(
    tmp_path, monkeypatch
):
    _, _, config, context, action = fixture(tmp_path, count=2)
    config["upload_batches_directory"] = str(tmp_path / "submissions")
    result = prepare_batch(
        action=plan.attach_preview(action, context["event_state"], context, config), context=context
    )
    original = Path(result["upload_archive"])
    original_hash = checksum(original)
    path = tmp_path / "workflow_state" / "state.json"
    path.parent.mkdir()
    write_json(path, result["state"])
    control = SimpleNamespace(state_path=path, chat_handler=SimpleNamespace(lock=RLock()))
    import phase_agent.tools.step_runner.file_protocol as protocol

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated_state_conflict")

    monkeypatch.setattr(protocol, "write_json_if_version", fail_write)
    from phase_agent.tools.relax_mc_batch.revise_delivery import minimize_registered_delivery

    with pytest.raises(RuntimeError, match="simulated_state_conflict"):
        minimize_registered_delivery(
            control,
            bundle_id=result["bundle_id"],
            source_sha256=original_hash,
            confirmed_unsubmitted=True,
        )
    assert checksum(original) == original_hash
    assert not Path(result["directory"]).with_name("RelaxMC_Batch01_Minimal_input").exists()
    assert not read_json(path)["relax_mc_batches"][result["bundle_id"]].get(
        "minimal_delivery_revision"
    )
