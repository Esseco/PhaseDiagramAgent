"""Navigation uses registered deliveries without changing scientific records."""

from copy import deepcopy
import json

import pytest
from pathlib import Path

from phase_agent.tools.remote.submission_layout import write_submission_index


def _delivery(root, label, *, model, task_id, task_status, waived=False):
    operation = root / f"epoch0/G0001_{label}"
    batch = operation / "inputs" / "batch"
    upload = root / f"epoch0/delivery/transfers/{label}_input"
    returned = upload.with_name(label + "_output")
    for folder in (batch, upload, returned):
        folder.mkdir(parents=True)
    script = upload / "submit_all.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    return (
        {
            "task_id": task_id,
            "model_version": model,
            "stage": "dft_single_point",
            "status": task_status,
            "recovery_wait_waived": waived,
        },
        {
            "task_ids": [task_id],
            "model_version": model,
            "stage": "dft_single_point",
            "status": "prepared",
            "upload_directory": str(batch),
            "script_path": str(batch / "GPU.sh"),
            "handoff": {
                "folder": str(upload),
                "submit_script": str(script),
                "command": "sbatch submit_all.sh",
                "return_directory": str(returned),
                "layout_version": 4,
            },
        },
    )


def test_registered_relax_mc_paths_are_shared_by_markdown_and_json(tmp_path):
    root = tmp_path / "submissions"
    internal = root / "epoch0/delivery/relax_mc/RelaxMC_Batch01"
    upload = root / "epoch0/delivery/relax_mc/Registered_input"
    returned = root / "epoch0/delivery/relax_mc/Registered_output"
    upload.mkdir(parents=True)
    returned.mkdir()
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "tasks": [
            {
                "task_id": "t1",
                "model_version": "m1",
                "status": "pending",
                "relax_mc_bundle_id": "b1",
            }
        ],
        "relax_mc_batches": {
            "b1": {
                "directory": str(internal),
                "upload_folder": str(upload),
                "script_path": str(upload / "submit_all.sh"),
                "expected_results_archive": str(returned / "results.zip"),
                "model_version": "m1",
                "status": "prepared",
            }
        },
    }
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    row = index["relax_mc_batches"][0]
    assert row["submission_path"]["path"] == str(upload)
    assert row["return_path"]["path"] == str(returned)
    assert "Registered_input" in text and "Registered_output" in text
    current = text.split("<details>", 1)[0]
    assert "RelaxMC_Batch01_input" not in current and "RelaxMC_Batch01_output" not in current
    assert state == before


def test_only_current_unfinished_tasks_precede_history_and_internal_archives(tmp_path):
    root = tmp_path / "submissions"
    specs = [
        ("Current", "m1", "pending", False),
        ("Completed", "m1", "completed", False),
        ("OldModel", "m0", "pending", False),
        ("Waived", "m1", "pending", True),
    ]
    pairs = [
        _delivery(root, label, model=model, task_id=label, task_status=status, waived=waived)
        for label, model, status, waived in specs
    ]
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 2, "m0": 1}},
        "tasks": [task for task, _ in pairs],
        "slurm_batches": [batch for _, batch in pairs],
    }
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    current, folded = text.split("<details>", 1)
    assert "Current_input" in current and "Current_output" in current
    assert "sbatch submit_all.sh" in current
    for label in ("Completed", "OldModel", "Waived"):
        assert label + "_input" not in current
        assert label + "_input" in folded
    assert "G0001_Current" not in current and "G0001_Current" in folded
    assert "历史交付" in folded and "内部计算档案" in text
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert len(index["handoffs"]) == len(index["operations"]) == 4
    assert index["handoffs"][0]["task_status_counts"]
    assert state == before


def test_current_training_delivery_is_visible_without_scientific_task_rows(tmp_path):
    root = tmp_path / "submissions"
    current = root / "epoch0/delivery/transfers/Finetune_Batch01_input"
    current.mkdir(parents=True)
    training = root / "epoch0/MLIP-finetune-round-0001"
    training.mkdir()
    state = {
        "active_model_version": "m1",
        "remote_finetune_jobs": {
            "train1": {
                "original_model_version": "m1",
                "status": "inputs_prepared",
                "directory": str(training),
                "handoff": {
                    "folder": str(current),
                    "submit_script": str(current / "submit_all.sh"),
                    "return_directory": str(current.with_name("Finetune_Batch01_output")),
                    "command": "sbatch submit_all.sh",
                    "layout_version": 4,
                },
            }
        },
    }
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    assert "Finetune_Batch01_input" in text.split("<details>", 1)[0]


def test_cancelled_relax_mc_wait_is_folded_without_losing_its_record(tmp_path):
    root = tmp_path / "submissions"
    directory = root / "epoch0/delivery/relax_mc/RelaxMC_Batch01"
    state = {
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": "t1",
                "status": "pending",
                "model_version": "m1",
                "relax_mc_bundle_id": "b1",
                "recovery_cancelled": True,
            }
        ],
        "relax_mc_batches": {
            "b1": {
                "directory": str(directory),
                "model_version": "m1",
                "status": "failed",
            }
        },
    }
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    current, folded = text.split("<details>", 1)
    assert "RelaxMC_Batch01_input" not in current
    assert "RelaxMC_Batch01_input" in folded and "停止等待 1 个 MC" in folded


def test_only_the_pending_training_followup_precedes_older_training_deliveries(tmp_path):
    root = tmp_path / "submissions"
    training = root / "epoch0/MLIP-finetune-round-0001"
    training.mkdir(parents=True)

    def delivery(name):
        folder = root / "epoch0/transfers" / (name + "_input")
        folder.mkdir(parents=True)
        return {
            "folder": str(folder),
            "submit_script": str(folder / "submit_all.sh"),
            "return_directory": str(folder.with_name(name + "_output")),
            "command": "sbatch submit_all.sh",
            "layout_version": 4,
        }

    original, manifest, validation = [
        delivery(name) for name in ("Training", "Manifest", "Validation")
    ]
    state = {
        "active_model_version": "m1",
        "remote_finetune_jobs": {
            "train1": {
                "original_model_version": "m1",
                "status": "recovered",
                "directory": str(training),
                "handoff": original,
                "transfer_handoffs": {"manifest": manifest, "validation": validation},
                "training_handoff": {"stage": "awaiting_validation", "handoff": validation},
            }
        },
    }
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    current, historical = text.split("<details>", 1)
    assert "Validation_input" in current
    assert "Training_input" not in current and "Manifest_input" not in current
    assert "Training_input" in historical and "Manifest_input" in historical


def test_discovered_collect_only_package_precedes_original_unrecovered_job(tmp_path, monkeypatch):
    root = tmp_path / "submissions"
    task, batch = _delivery(
        root,
        "Original",
        model="m1",
        task_id="t1",
        task_status="pending",
    )
    folder = root / "epoch0/delivery/transfers/Original_Recovery01_input"
    folder.mkdir(parents=True)
    returned = folder.with_name("Original_Recovery01_output")
    returned.mkdir()
    recovery = {
        "folder": str(folder),
        "submit_script": str(folder / "submit_all.sh"),
        "return_directory": str(returned),
        "task_ids": ["t1"],
        "layout_version": 4,
        "collect_only": True,
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.legacy_recovery.recovery_handoffs",
        lambda _state: iter([recovery]),
    )
    state = {
        "active_model_version": "m1",
        "tasks": [task],
        "slurm_batches": [batch],
    }
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    current, historical = text.split("<details>", 1)
    assert "Original_Recovery01_input" in current and "Original_Recovery01_output" in current
    assert "仅收集已有结果" in current and "原计算勿重投" in current
    assert "尚未提交或远端未运行" in current and "不构成提交授权" in current
    assert "transfers/Original_input" not in current
    assert "transfers/Original_input" in historical
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert len(index["operations"]) == 1 and len(index["handoffs"]) == 2
    assert state == before


@pytest.mark.parametrize("layout", ["returned", "prepared", "direct", "custom"])
def test_legacy_compact_return_path_preserves_paired_delivery_without_overriding_custom(
    tmp_path, layout
):
    root = tmp_path / "submissions"
    directory = root / "epoch0/delivery/relax_mc/RelaxMC_Batch01"
    directory.mkdir(parents=True)
    paired_input = directory.with_name(directory.name + "_input")
    paired_output = directory.with_name(directory.name + "_output")
    expected = directory / "returned/results.zip"
    if layout in {"returned", "prepared", "custom"}:
        paired_input.mkdir()
        paired_output.mkdir()
        (paired_input / "run_upload.py").write_text("# compact handoff driver", encoding="utf-8")
    if layout in {"returned", "custom"}:
        (paired_output / "results.zip").write_bytes(b"synthetic returned archive")
    if layout == "direct":
        expected.parent.mkdir()
        expected.write_bytes(b"synthetic direct CLI archive")
    if layout == "custom":
        expected = root / "custom_return/results.zip"
        expected.parent.mkdir()
    verified = root / "epoch0/evidence/verified_results/import-checksum"
    verified.mkdir(parents=True)
    record = {
        "directory": str(directory),
        "model_version": "m1",
        "status": "completed",
        "expected_results_archive": str(expected),
        "return_directory": str(verified),
    }
    if layout == "custom":
        record["upload_folder"] = str(root / "custom_input")
    state = {"active_model_version": "m1", "relax_mc_batches": {"b1": record}}
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    target = paired_output if layout in {"returned", "prepared"} else expected.parent
    assert index["relax_mc_batches"][0]["return_path"]["path"] == str(target)
    assert target.name in text and "import-checksum" not in text
    assert state == before


def test_moved_compact_input_defines_return_even_when_cancelled_output_never_existed(tmp_path):
    root = tmp_path / "submissions"
    directory = root / "epoch1/delivery/relax_mc/RelaxMC_Batch01"
    upload = directory.with_name(directory.name + "_input")
    upload.mkdir(parents=True)
    (upload / "run_upload.py").write_text("# paired output protocol", encoding="utf-8")
    expected = root / "relax_mc_batches/epoch1_RelaxMC_Batch01_output/results.zip"
    returned = upload.with_name("RelaxMC_Batch01_output")
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b1": {
                "directory": str(directory),
                "upload_folder": str(upload),
                "expected_results_archive": str(expected),
                "status": "cancelled",
                "model_version": "m1",
            }
        },
    }
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    row = index["relax_mc_batches"][0]
    assert row["return_path"]["path"] == str(returned)
    assert row["return_path"]["exists"] is False
    assert "RelaxMC_Batch01_output" not in text.split("<details>", 1)[0]
    assert not returned.exists() and state == before


def test_recovery_archive_stays_primary_and_old_inferred_path_is_explicit_legacy(tmp_path):
    root = tmp_path / "submissions"
    original = root / "epoch1/delivery/relax_mc/RelaxMC_Batch02"
    upload = original.with_name(original.name + "_input")
    upload.mkdir(parents=True)
    (upload / "run_upload.py").write_text("# original paired output protocol", encoding="utf-8")
    recovery = root / "recoveries/epoch1_RelaxMC_Batch02_collect_only"
    legacy_return = recovery.with_name(recovery.name + "_output")
    expected = root / "epoch1/evidence/recovery/RelaxMC_Batch02_collect_only_output/results.zip"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"synthetic verified collect-only archive")
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b2": {
                "directory": str(recovery),
                "upload_folder": str(upload),
                "expected_results_archive": str(expected),
                "status": "completed",
                "model_version": "m1",
                "recovery_original_directory": str(original),
                "recovery_original_archive": str(
                    original.with_name(original.name + "_output") / "results.zip"
                ),
            }
        },
    }
    before = deepcopy(state)
    text = write_submission_index(root, state).read_text(encoding="utf-8")
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    row = index["relax_mc_batches"][0]
    assert row["return_path"]["path"] == str(expected.parent)
    assert row["return_path"]["exists"] is True
    assert [path["path"] for path in row["legacy_return_paths"]] == [str(legacy_return)]
    assert row["legacy_return_paths"][0]["exists"] is False
    assert "历史回传路径" in text and "仅供核对" in text
    assert "recoveries/epoch1_RelaxMC_Batch02_collect_only_output" in text
    assert "RelaxMC_Batch02" not in text.split("<details>", 1)[0]
    assert not legacy_return.exists() and state == before


def test_submission_index_exposes_canonical_navigation_map(tmp_path):
    root = tmp_path / "submissions"
    text = write_submission_index(root, {}).read_text(encoding="utf-8")
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))

    navigation = index["navigation"]
    assert navigation["read_order"] == [
        "README.md",
        "submission_index.json",
        "epochN/README.md",
        "epochN/layout_manifest.json",
    ]
    assert [lane["id"] for lane in navigation["lanes"]] == [
        "delivery",
        "calculation",
        "evidence",
        "quarantine",
    ]
    assert "先读这里" in text
    assert "交付层" in text and "计算层" in text and "证据层" in text and "隔离层" in text


def test_epoch_submission_guide_points_back_to_root_quarantine(tmp_path):
    root = tmp_path / "submissions"
    (root / "epoch0").mkdir(parents=True)
    write_submission_index(
        root,
        {"upload_layout": {"model_rounds": {"m1": 1}}},
    )
    text = (root / "epoch0" / "README.md").read_text(encoding="utf-8")
    assert "../previously_used/<epoch>/" in text
    assert "`evidence/storage_archives/`（本轮存在时）" in text


def test_epoch_layout_manifest_classifies_registered_and_unclassified_children(tmp_path):
    root = tmp_path / "submissions"
    operation = root / "epoch0/calculations/G0001_DFT-0001_DFT-single-point"
    (operation / "inputs/batch").mkdir(parents=True)
    (operation / "results").mkdir()
    (root / "epoch0/delivery/transfers/Batch_input").mkdir(parents=True)
    (root / "epoch0/delivery/transfers/Batch_output").mkdir()
    (root / "epoch0/manual_notes").mkdir()
    write_submission_index(
        root,
        {
            "upload_layout": {"model_rounds": {"m1": 1}},
            "slurm_batches": [
                {
                    "upload_directory": str(operation / "inputs/batch"),
                    "results_directory": str(operation / "results"),
                    "task_ids": [],
                    "status": "completed",
                    "handoff": {
                        "folder": str(root / "epoch0/delivery/transfers/Batch_input"),
                        "return_directory": str(root / "epoch0/delivery/transfers/Batch_output"),
                        "layout_version": 4,
                    },
                }
            ],
        },
    )
    manifest = json.loads((root / "epoch0/layout_manifest.json").read_text(encoding="utf-8"))
    children = {row["name"]: row for row in manifest["children"]}
    calculations = next(row for row in manifest["children"] if row["name"] == "calculations")
    delivery = next(row for row in manifest["children"] if row["name"] == "delivery")
    assert calculations["lane"] == "calculation"
    assert delivery["lane"] == "delivery"
    assert any(row["name"] == operation.name for row in calculations["children"])
    assert "manual_notes" in manifest["unclassified"]
    epoch_readme = (root / "epoch0/README.md").read_text(encoding="utf-8")
    assert "layout_manifest.json" in epoch_readme
    assert "本轮直接子目录" in epoch_readme
    assert "manual_notes" in epoch_readme


def test_calculation_only_batch_is_not_listed_as_submission_handoff(tmp_path):
    root = tmp_path / "submissions"
    operation = root / "epoch0/calculations/G0001_DFT-0001_DFT-single-point"
    batch = operation / "inputs/batch"
    batch.mkdir(parents=True)
    (operation / "results").mkdir()
    state = {
        "upload_layout": {"model_rounds": {"m1": 1}},
        "active_model_version": "m1",
        "slurm_batches": [
            {
                "upload_directory": str(batch),
                "results_directory": str(operation / "results"),
                "task_ids": [],
                "status": "completed",
            }
        ],
    }
    write_submission_index(root, state)
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert len(index["operations"]) == 1
    assert index["handoffs"] == []


def test_relax_mc_index_prefers_paired_input_script_over_legacy_record(tmp_path):
    root = tmp_path / "submissions"
    legacy = root / "epoch0/delivery/relax_mc/RelaxMC_Batch01"
    upload = legacy.with_name(legacy.name + "_input")
    returned = upload.with_name(upload.name[:-6] + "_output")
    upload.mkdir(parents=True)
    returned.mkdir()
    (legacy / "submit_all.sh").parent.mkdir(parents=True)
    (legacy / "submit_all.sh").write_text("legacy", encoding="utf-8")
    (upload / "submit_all.sh").write_text("paired", encoding="utf-8")
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b1": {
                "directory": str(legacy),
                "upload_folder": str(upload),
                "script_path": str(legacy / "submit_all.sh"),
                "expected_results_archive": str(returned / "results.zip"),
                "model_version": "m1",
                "status": "prepared",
            }
        },
    }
    write_submission_index(root, state)
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert index["relax_mc_batches"][0]["submit_script"]["path"] == str(
        upload / "submit_all.sh"
    )


def test_epoch_manifest_marks_legacy_delivery_package_as_quarantine(tmp_path):
    root = tmp_path / "submissions"
    legacy = root / "epoch0/delivery/transfers/DFTSP_Batch01"
    paired = root / "epoch0/delivery/transfers/DFTSP_Batch02_input"
    legacy.mkdir(parents=True)
    paired.mkdir()
    write_submission_index(root, {"upload_layout": {"model_rounds": {"m1": 1}}})
    manifest = json.loads((root / "epoch0/layout_manifest.json").read_text(encoding="utf-8"))
    delivery = next(row for row in manifest["children"] if row["name"] == "delivery")
    transfers = next(row for row in delivery["children"] if row["name"] == "transfers")
    entries = {row["name"]: row for row in transfers.get("children", [])}
    assert entries["DFTSP_Batch01"]["lane"] == "quarantine"
    assert entries["DFTSP_Batch01"]["legacy"] is True
    assert entries["DFTSP_Batch02_input"]["lane"] == "delivery"
