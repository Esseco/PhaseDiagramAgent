import pytest
from phase_agent.science.mlip import mace_worker
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.agents.reliability.extrapolation_round import _monitor_observation
from phase_agent.tools.workflows.direct_extrapolation_sampling import prepare_direct_sampling
from Process_MLIP.collapse_monitor import CollapseMonitor


class Atom:
    def __init__(self, score=0, force=0):
        self.score = score
        self.force = force

    def copy(self):
        return Atom(self.score, self.force)

    def __len__(self):
        return 1

    def get_forces(self):
        return [[self.force, 0, 0]]


def test_threshold_boundary_final_and_force_jump(monkeypatch):
    monkeypatch.setattr(
        mace_worker,
        "_extrapolation_monitor_factory",
        lambda: lambda a: CollapseMonitor(a, lambda p: [p.score], stop_on_extrapolation=False),
    )
    mon = mace_worker._threshold_monitor_factory(3)(Atom())
    mon(Atom(0, 0))
    mon(Atom(2, 2))
    assert mon.failures == []
    assert mon.check_final(Atom(3)) == 3 and mon.failures == []
    mon.check_final(Atom(3.01))
    assert [x["score"] for x in mon.failures] == [3.01]
    refresh = mace_worker._threshold_monitor_factory(1)(Atom())
    refresh.check_final(Atom(1.01))
    assert len(refresh.failures) == 1


def test_task_preparation_binds_policy_by_scope():
    class Manager:
        boundary = {}
        data = {
            "structures": {"s": {"branch_id": "b", "source_path": "fake.vasp"}},
            "branches": {"b": {}},
        }

    prep = create_mlip_task_preparer(Manager(), {}, {"mlip": {"model_path": "/m", "version": "v"}})
    base = {"stage": "relax_and_feature", "structure_id": "s"}
    batch = prep({**base, "relax_mc_bundle_id": "b"})
    refresh = prep(
        {**base, "model_refresh_id": "r", "parameters": {"model_refresh_operation": "relax"}}
    )
    assert batch["worker_job"]["parameters"]["feature_difference_threshold"] == 3
    assert refresh["worker_job"]["parameters"].get("feature_difference_threshold", 1) == 1


def test_threshold_three_is_valid_evidence_but_does_not_override_refresh():
    obs = {
        "structure_id": "s",
        "identity_scheme": "periodic_input_v1",
        "monitor_complete": True,
        "extrapolation_failure": False,
    }
    t = {
        "structure_id": "s",
        "outputs": {
            "relaxation_observation": obs,
            "extrapolation_monitor": {
                "final_feature_difference": 2.5,
                "feature_difference_threshold": 3.0,
                "extrapolation_failure": False,
                "failures": [],
            },
        },
    }
    assert _monitor_observation(t)["monitor_complete"] is True
    assert _monitor_observation(t, round_kind="epoch_refresh")["monitor_complete"] is False
    t["parameters"] = {"model_refresh_operation": "relax"}
    assert _monitor_observation(t)["monitor_complete"] is False


def test_direct_sampling_rejects_relax_mc_scope():
    with pytest.raises(ValueError, match="direct_recovery_requires_model_refresh_scope"):
        prepare_direct_sampling(
            action={"parameters": {"scope_id": "batch-id"}},
            context={"event_state": {"relax_mc_batches": {"batch-id": {}}}},
        )


def test_model_refresh_rejects_three_before_running(tmp_path):
    with pytest.raises(ValueError, match="model_refresh_feature_threshold_must_be_one"):
        mace_worker.run_mace_worker(
            {
                "operation": "relax",
                "output_directory": str(tmp_path),
                "parameters": {
                    "feature_difference_threshold": 3,
                    "model_refresh_operation": "relax",
                },
            }
        )
