import hashlib
import json
from types import SimpleNamespace

from phase_agent.policies.region_sampling import ledger_fingerprint
from phase_agent.tools.state.registered_branch_evidence import inspect_registered_branches
from phase_agent.tools.workflows.region_review import path_for


def test_frozen_region_proposal_is_not_output_with_unchanged_existing_ledger(tmp_path):
    root = tmp_path / "project"
    state_path = root / "workflow_state" / "state.json"
    state_path.parent.mkdir(parents=True)
    ledger_path = state_path.parent / "ledgers" / "phase_data.json"
    ledger_path.parent.mkdir()
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ledger = {
        "branches": {"old": {"P": "O3", "H": h, "x": "1", "T": ["Ni"]}},
        "structures": {"old_s": {"branch_id": "old"}},
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf8")
    state = {
        "workspace_layout": {"workspace_root": str(root)},
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "confirmed_config": {"storage": {"paths": {}}},
    }
    params = {
        "sampling_mode": "region_adaptive",
        "region_preview": {"ledger_fingerprint": ledger_fingerprint(SimpleNamespace(data=ledger))},
    }
    action = {"tool": "generate_branches", "task_key": "sample", "parameters": params}
    path = path_for(state, action)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "程序记录（可折叠）": {
                    "status": "awaiting_approval",
                    "task_key": "sample",
                    "model_version": "m",
                    "config_version": "c",
                }
            }
        ),
        encoding="utf8",
    )
    assert inspect_registered_branches(state, action, state_path) is None
    output = (
        root
        / "structures"
        / "candidate_structures"
        / ("RegionSampling-" + hashlib.sha256(b"sample").hexdigest()[:16])
    )
    output.mkdir(parents=True)
    assert inspect_registered_branches(state, action, state_path)["status"] == "invalid"
    output.rmdir()
    ledger["structures"]["new_s"] = {"branch_id": "old"}
    ledger_path.write_text(json.dumps(ledger), encoding="utf8")
    assert inspect_registered_branches(state, action, state_path)["status"] == "invalid"
