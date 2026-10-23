"""Cold-start coverage through the real public workflow and proposal graph."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.runtime.local_project_launcher import create_project
from phase_agent.runtime.main import run_workflow
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.session.save_config_session import save_config_session
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


def forbidden(*args, **kwargs):
    pytest.fail("A new unapproved project must not prepare, execute or submit calculations")


def test_unconfirmed_project_cannot_start_science(tmp_path):
    config_path = create_project(tmp_path / "new", registry=tmp_path / "registry.json")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    session = json.loads(
        (config_path.parent / settings["config_session_path"]).read_text(encoding="utf-8")
    )
    result = run_workflow(
        None,
        {},
        {"state_path": str(config_path.parent / "workflow_state/state.json")},
        session,
        agent_client=forbidden,
        handlers={"generate_branches": forbidden},
    )
    assert result["status"] == "rejected" and result["reason"] == "configuration_not_confirmed"
    assert result["submitted"] is False


def test_empty_project_reaches_first_saved_approval_without_old_round_evidence(tmp_path):
    config_path = create_project(tmp_path / "new", registry=tmp_path / "registry.json")
    root = config_path.parent
    ref = root / "structures/reference_structures/O3.vasp"
    write(
        ref,
        Atoms(
            "NaFeO2",
            positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            cell=[6, 6, 6],
            pbc=True,
        ),
        format="vasp",
    )
    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Fe": 1},
    }
    draft = create_config_draft(
        default_layered_search_config(boundary=boundary, phase_references={"O3": str(ref)})
    )
    draft = apply_config_revision(
        draft,
        {
            "calculation.mlip_version": "m0",
            "mlip.name": "m0",
            "mlip.version": "m0",
            "mlip.model_path": "/remote/fake/m0.model",
            "dft.parameters": {"encut": 520},
            "convergence.hull_tolerance": 0.01,
            "convergence.minimum_dft_validations": 1,
            "convergence.coverage_threshold": 0.9,
            "agent.allowed_tools": ["generate_branches", "pause_search"],
        },
    )
    session = confirm_config_snapshot(draft, user_confirmed=True)
    assert session["status"] == "confirmed"
    save_config_session(session, root / "parameters/config_session.json")
    manager = PhaseDataManager(boundary)
    assert not manager.data["branches"] and not manager.data["structures"]
    model_calls = []

    def model(payload):
        model_calls.append(deepcopy(payload))
        return {
            "tool": "generate_branches",
            "task_key": "cold-first",
            "target_ids": [],
            "budget": 0,
            "reason": "First-round coverage is an exploration hypothesis",
            "expected_purpose": "Register structures only after approval",
            "parameters": {
                "sampling_mode": "region_adaptive",
                "initial_states_per_branch": 1,
                "regions": [
                    {
                        "region_id": "cold-r1",
                        "priority": 1,
                        "strategy": "coverage",
                        "phase": "O3",
                        "min_det_H": 1,
                        "max_det_H": 1,
                        "na_min": 0.5,
                        "na_max": 1.0,
                        "reason": "No energy evidence exists yet",
                        "evidence_refs": [],
                    }
                ],
            },
            "scientific_decision": {
                "point": "branch_sampling",
                "config_version": session["confirmed_snapshot"]["config_version"],
                "model_version": "m0",
                "bottleneck": "The initial candidate pool is empty",
                "rationale": "Explore the configured phase and legal composition interval",
                "alternatives": [
                    {"route": "Wait", "reason": "No calculations are awaiting return"}
                ],
                "expected_outcome": "Obtain initial structures for later screening",
                "success_criterion": "Registered structures have traceable branch identities",
                "limitations": "There is no energy, D, H or committee-error evidence yet",
                "selection_purposes": [],
            },
        }

    state_path = root / "workflow_state/state.json"
    options = dict(
        manager=manager,
        phase_references={"O3": str(ref)},
        run_config={"state_path": str(state_path)},
        config_session=session,
        handlers={"generate_branches": forbidden},
        agent_client=model,
        execution_mode="interactive",
        invocation_id="cold-first",
        state_path=state_path,
        approval_directory=root / "workflow_state/approvals",
        dispatcher=SimpleNamespace(dispatch=forbidden),
        scientific_decision_policy={"schema_version": 1},
    )
    result = run_workflow(**options)
    assert result["status"] == "awaiting_approval", result
    assert len(model_calls) == 1
    assert result["state"]["active_model_version"] == "m0"
    assert not result["state"].get("tasks") and not result["state"].get("qbc_candidates")
    assert not result["state"].get("generation_history") and not result["state"].get(
        "post_dft_assessment"
    )
    assert not manager.data["branches"] and not manager.data["structures"]
    assert result["state"]["pending_execution_policies"]
    assert (state_path.parent / "langgraph_approvals.sqlite").is_file()
    assert (state_path.parent / "langgraph_lifecycle.sqlite").is_file()
    saved = state_path.read_bytes()
    # The same checkpointed request only restores the original approval.
    restored = run_workflow(**{**options, "agent_client": forbidden})
    assert restored["status"] == "awaiting_approval"
    assert (
        restored["state"]["pending_execution_policies"]
        == result["state"]["pending_execution_policies"]
    )
    assert state_path.read_bytes() == saved
