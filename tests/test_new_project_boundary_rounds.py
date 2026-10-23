"""New-project boundaries through actual import, confirmation and region preview."""

from copy import deepcopy
from pathlib import Path
import json

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config
from phase_agent.configuration.session.project_config_json import write_project_config_patch
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.runtime.configuration_chat import ConfigurationChatHandler
from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.local_project_launcher import create_project
from phase_agent.tools.policy.region_sampling import preview


IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
DOUBLE = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
NEGATIVE = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]


def imported_project(tmp_path, matrices=None, tm_ratio=None):
    runtime_path = create_project(tmp_path / "new", registry=tmp_path / "registry.json")
    root = runtime_path.parent
    references = {}
    for phase in ("O3", "P3"):
        path = root / f"structures/reference_structures/{phase}.vasp"
        write(
            path,
            Atoms(
                "Na2FeMnO4",
                positions=[[index % 2, (index // 2) % 2, index // 4] for index in range(8)],
                cell=[6, 6, 6],
                pbc=True,
            ),
            format="vasp",
        )
        references[phase] = str(path)
    session_path = root / "parameters/config_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    boundary = {
        "P": {"at_x": {"0": ["P3"], "1": ["O3"]}, "intermediate": ["O3", "P3"]},
        "H": {
            phase: deepcopy(matrices if matrices is not None else [IDENTITY, DOUBLE])
            for phase in references
        },
        "TM_ratio": tm_ratio if tm_ratio is not None else {"Fe": 1, "Mn": 1},
    }
    source = root / "parameters/search_config.project.json"
    write_project_config_patch(
        source,
        {
            "system.boundary": boundary,
            "system.H_generation.enabled": False,
            "system.configuration_space.roles.T": "branch",
            "system.phase_references": references,
            "system.phase_reference_directory": str(root / "structures/reference_structures"),
            "mlip.model_path": "/remote/fake/m0.model",
            "python_environments.remote_python": "fake-python",
            "python_environments.remote_mlip": "fake-mlip",
        },
        baseline_config=session["config"],
    )
    session["setup_stage"] = "json_ready"
    review_calls = []

    def reviewer(payload):
        review_calls.append(deepcopy(payload))
        return {"reply": "已核对合成边界", "ready_for_search": True, "patch": {}, "questions": []}

    handler = ConfigurationChatHandler(
        {
            "state_path": str(root / "workflow_state/state.json"),
            "config_session": session,
            "config_session_path": str(session_path),
        },
        config_session_path=session_path,
        base_directory=root,
        editable_config_path=source,
        agent_client=reviewer,
    )
    reply = handler([{"role": "user", "content": "读取配置 JSON"}])
    return handler, boundary, references, review_calls, reply


def region(phase, low=0, high=1):
    return {
        "region_id": phase,
        "priority": 1,
        "strategy": "coverage",
        "phase": phase,
        "min_det_H": 1,
        "max_det_H": 2,
        "na_min": low,
        "na_max": high,
        "reason": "Synthetic boundary check without generating structures",
        "evidence_refs": [],
    }


def test_imported_boundary_survives_confirmation_and_distinct_runtime_calls(tmp_path):
    handler, boundary, references, calls, reply = imported_project(tmp_path)
    assert calls and calls[-1]["mode"] == "configuration_json_review", reply
    assert handler.configuration_readiness()["ready"], reply
    assert handler.workflow_kwargs["config_session"]["config"]["system"]["boundary"] == boundary
    system = handler.workflow_kwargs["config_session"]["config"]["system"]
    assert system["constraints"] == {"phases": ["O3", "P3"], "TM_ratio": {"Fe": 1, "Mn": 1}}
    assert system["species"]["substitutional"] == ["Fe", "Mn"]
    control = LocalAgentControl(handler)
    assert (
        control.confirm_config(explicit=False)["confirmation_error"]
        == "explicit_user_confirmation_required"
    )
    assert control.confirm_config(explicit=True)["status"] == "confirmed"
    session = handler.workflow_kwargs["config_session"]
    frozen = deepcopy(session["confirmed_snapshot"])
    manager = PhaseDataManager(boundary, system_config=frozen["config"]["system"])
    ledger_path = tmp_path / "ledger.json"
    expected_x = {"O3": {"1/4", "1/2", "3/4", "1"}, "P3": {"0", "1/4", "1/2", "3/4"}}

    # These are later configuration/preview calls, not completed scientific rounds.
    for invocation in ("first", "second", "third"):
        config = build_effective_run_config(
            session,
            {
                "system": {"boundary": {"P": ["O1"]}},
                "state_path": str(tmp_path / f"{invocation}.json"),
            },
        )
        assert config["system_config"]["boundary"] == boundary
        assert config["config_version"] == frozen["config_version"]
        assert "system" in config["ignored_runtime_keys"]
        result = preview(
            {"sampling_mode": "region_adaptive", "regions": [region("O3"), region("P3")]},
            manager,
            references,
            config,
            {"confirmed_config_version": frozen["config_version"], "active_model_version": "m0"},
        )
        for item in result["regions"]:
            assert {cell["x"] for cell in item["cells"]} == expected_x[item["region_id"]]
            assert {cell["det_H"] for cell in item["cells"]} == {1, 2}
        assert not manager.data["branches"] and not manager.data["structures"]
        manager.save(ledger_path)
        manager = PhaseDataManager.load(ledger_path)
        assert manager.boundary == boundary
        assert session["confirmed_snapshot"] == frozen
    assert not handler.state_path.exists()


def test_imported_tm_ratio_filters_incommensurate_supercells(tmp_path):
    triple = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    handler, boundary, references, _, reply = imported_project(
        tmp_path, [IDENTITY, triple], {"Fe": 2, "Mn": 1}
    )
    assert handler.configuration_readiness()["ready"], reply
    assert LocalAgentControl(handler).confirm_config(explicit=True)["status"] == "confirmed"
    config = build_effective_run_config(handler.workflow_kwargs["config_session"])
    manager = PhaseDataManager(boundary, system_config=config["system_config"])
    requested = {**region("O3"), "max_det_H": 3}
    result = preview(
        {"sampling_mode": "region_adaptive", "regions": [requested]},
        manager,
        references,
        config,
        {},
    )
    cells = result["regions"][0]["cells"]
    assert {cell["det_H"] for cell in cells} == {3}
    assert {cell["x"] for cell in cells} == {"1/6", "1/3", "1/2", "2/3", "5/6", "1"}
    assert not manager.data["branches"] and not manager.data["structures"]


@pytest.mark.parametrize(
    "matrices",
    [
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        [[[1.5, 0, 0], [0, 1, 0], [0, 0, 1]]],
        [[[1, 0, 0], [0, 1, 0]]],
        [[[10**400, 0], [0, 1]]],
        [],
    ],
    ids=["singular", "noninteger", "wrong-shape", "overflow", "missing"],
)
def test_invalid_explicit_supercell_cannot_be_confirmed(tmp_path, matrices):
    handler, _, _, _, reply = imported_project(tmp_path, matrices)
    readiness = handler.configuration_readiness()
    assert not readiness["ready"], (matrices, reply)
    result = LocalAgentControl(handler).confirm_config(explicit=True)
    assert result["status"] == "draft"
    assert result["confirmation_error"] == "configuration_not_ready"
    assert not handler.workflow_kwargs["config_session"].get("confirmed_snapshot")
    assert not handler.state_path.exists()


def test_negative_determinant_supercell_remains_legal(tmp_path):
    handler, boundary, references, _, reply = imported_project(tmp_path, [NEGATIVE])
    assert handler.configuration_readiness()["ready"], reply
    assert LocalAgentControl(handler).confirm_config(explicit=True)["status"] == "confirmed"
    session = handler.workflow_kwargs["config_session"]
    config = build_effective_run_config(session)
    manager = PhaseDataManager(boundary, system_config=config["system_config"])
    result = preview(
        {"sampling_mode": "region_adaptive", "regions": [region("O3", 1, 1)]},
        manager,
        references,
        config,
        {},
    )
    assert result["regions"][0]["cells"] == [
        {"H": NEGATIVE, "det_H": 1, "x": "1", "covered": False}
    ]


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[True, 0], [0, 1]],
        [[1.5, 0], [0, 1]],
        [[1, 0, 0], [0, 1, 0]],
    ],
)
def test_snapshot_api_rejects_invalid_branch_h_without_ui(tmp_path, matrix):
    from phase_agent.configuration.schema.validate_search_config import validate_search_config
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
    from tests.test_event_loop import _session

    config = deepcopy(_session()["confirmed_snapshot"]["config"])
    config["system"]["boundary"] = {"P": ["O3"], "H": {"O3": [matrix]}, "TM_ratio": {"Fe": 1}}
    draft = create_config_draft(config)
    audit = validate_search_config(config, stage="startup")
    assert not audit["valid"]
    assert any("system.boundary.H.O3[0]" in issue for issue in audit["conflicts"])
    result = confirm_config_snapshot(draft, user_confirmed=True)
    assert result["status"] == "draft" and not result.get("confirmed_snapshot")
    assert result["confirmation_error"] == "configuration_not_valid"


def test_bad_h_reimport_preserves_existing_confirmed_snapshot(tmp_path):
    handler, _, _, calls, _ = imported_project(tmp_path)
    assert LocalAgentControl(handler).confirm_config(explicit=True)["status"] == "confirmed"
    session = deepcopy(handler.workflow_kwargs["config_session"])
    count = len(calls)
    write_project_config_patch(
        handler.editable_config_path,
        {"system.boundary.H.O3": [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]},
        baseline_config=session["config"],
    )
    reply = handler([{"role": "user", "content": "读取配置 JSON"}])
    assert "配置 JSON 未导入" in reply and "行列式" in reply
    current = json.loads(Path(handler.config_session_path).read_text(encoding="utf-8"))
    assert current == handler.workflow_kwargs["config_session"]
    assert current["config"] == session["config"]
    assert current["confirmed_snapshot"] == session["confirmed_snapshot"]
    assert current["previous_confirmed_snapshot"] == session["confirmed_snapshot"]
    assert current["draft_revision"] == session["draft_revision"]
    assert current["last_imported_config_digest"] == session["last_imported_config_digest"]
    assert current["status"] == "draft" and not current.get("agent_reviewed_revision")
    assert len(calls) == count


def test_two_dimensional_supercells_still_import_and_confirm(tmp_path):
    handler, _, _, _, reply = imported_project(tmp_path, [[[1, 0], [0, 1]]])
    assert handler.configuration_readiness()["ready"], reply
    assert LocalAgentControl(handler).confirm_config(explicit=True)["status"] == "confirmed"
