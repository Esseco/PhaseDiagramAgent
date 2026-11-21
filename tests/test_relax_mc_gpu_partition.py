from pathlib import Path

import pytest
from phase_agent.tools.relax_mc_batch import run as mod


def test_two_partitions_balance_296_relax_and_105_mc():
    profiles = {"gpu": {"stages": ["relax_and_feature", "deep_search"],
                        "partition_pool": ["v100m3", "v100g32"],
                        "slurm_options": {"partition": "v100m3"}}}
    reservations = {"r": {"status": "reserved"}}
    tasks = [{"stage": "relax_and_feature", "status": "pending", "task_key": "r",
              "task_id": f"r{i}"} for i in range(296)]
    tasks += [{"stage": "deep_search", "status": "pending", "task_key": "r",
               "task_id": f"m{i}"} for i in range(105)]
    sizes = mod._balanced_batch_sizes({"tasks": tasks, "budget_reservations": reservations},
                                     profiles, {"relax_and_feature": 100, "deep_search": 20})
    assert sizes == {"relax_and_feature": 74, "deep_search": 18}
    for stage, total, size in [("relax_and_feature", 296, 74), ("deep_search", 105, 18)]:
        batches = []
        for start in range(0, total, size):
            profile = mod._wave_gpu_profile(profiles, stage, batches)
            part = profile["slurm_options"]["partition"]
            batches.append({"stage": stage, "gpu_partition": part,
                            "task_ids": list(range(start, min(start + size, total)))})
        counts = [sum(len(b["task_ids"]) for b in batches if b["gpu_partition"] == p)
                  for p in ("v100m3", "v100g32")]
        assert abs(counts[0] - counts[1]) <= size
        assert sum(counts) == total


def test_existing_frozen_profile_stays_single_partition():
    profiles = {"gpu": {"stages": ["relax_and_feature"],
                        "slurm_options": {"partition": "other"}}}
    assert mod._wave_gpu_profile(profiles, "relax_and_feature", [])["slurm_options"]["partition"] == "other"
    assert mod._balanced_batch_sizes({"tasks": []}, profiles, {"relax_and_feature": 100}) == {"relax_and_feature": 100}


def test_invalid_partition_pool_fails_closed():
    profiles = {"gpu": {"stages": ["deep_search"], "partition_pool": ["v100m3", "v100m3"],
                        "slurm_options": {"partition": "v100m3"}}}
    with pytest.raises(ValueError, match="gpu_partition_pool"):
        mod._wave_gpu_profile(profiles, "deep_search", [])


def test_partition_choice_counts_unfinished_batches_but_not_previous_waves():
    profiles = {"gpu": {"stages": ["deep_search"], "partition_pool": ["v100m3", "v100g32"],
                        "slurm_options": {"partition": "v100m3"}}}
    existing = [
        {"stage": "deep_search", "gpu_partition": "v100m3", "task_ids": ["a", "b"]},
        {"stage": "deep_search", "gpu_partition": "v100g32", "task_ids": ["old"] * 20,
         "relax_mc_wave_done": True},
    ]
    assert mod._wave_gpu_profile(profiles, "deep_search", existing)["slurm_options"]["partition"] == "v100g32"


def test_prepare_wave_writes_balanced_scripts(tmp_path, monkeypatch):
    profiles = {"gpu": {"stages": ["relax_and_feature", "deep_search"],
                        "partition_pool": ["v100m3", "v100g32"],
                        "slurm_options": {"partition": "v100m3", "gres": "gpu:1"}}}
    tasks = [{"stage": "relax_and_feature", "status": "pending", "task_key": "r",
              "task_id": f"r{i}"} for i in range(296)]
    state = {"tasks": tasks, "budget_reservations": {"r": {"status": "reserved"}}}
    class FakeRunner:
        def __init__(self, root, **kwargs):
            self.root = Path(root)
            self.size = kwargs["stage_batch_sizes"]["relax_and_feature"]
        def prepare(self, state, **kwargs):
            selected = [t for t in state["tasks"] if not t.get("slurm_batch_id")][:self.size]
            if not selected:
                return {"state": state, "status": "no_tasks"}
            index = len(state.get("slurm_batches", []))
            directory = self.root / str(index)
            directory.mkdir(parents=True)
            ids = [t["task_id"] for t in selected]
            for task in selected:
                task["slurm_batch_id"] = str(index)
            batch = {"batch_id": str(index), "task_ids": ids, "upload_directory": str(directory),
                     "script_path": str(directory / "job.sh")}
            state.setdefault("slurm_batches", []).append(batch)
            return {"state": state, "status": "prepared"}
    monkeypatch.setattr(mod, "ManualUploadBatchRunner", FakeRunner)
    monkeypatch.setattr(mod, "create_mlip_task_preparer", lambda *args: None)
    monkeypatch.setattr(mod, "checkpoint", lambda *args: None)
    monkeypatch.setattr(mod, "preflight_models", lambda *args: None)
    manifest = {"config": {"supercomputer": {"worker": {"command": []},
                                               "batch_sizes": {"relax_and_feature": 100}}},
                "stage_profiles": profiles, "phase_references": {}, "bundle_id": "b"}
    state, _ = mod.prepare_wave(tmp_path, state, None, manifest)
    batches = state["slurm_batches"]
    assert [len(b["task_ids"]) for b in batches] == [74] * 4
    assert [b["gpu_partition"] for b in batches] == ["v100m3", "v100g32"] * 2
    for batch in batches:
        assert f"#SBATCH --partition={batch['gpu_partition']}" in Path(batch["script_path"]).read_text()
