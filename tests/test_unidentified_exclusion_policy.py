from copy import deepcopy
import pytest
from tests.test_unidentified_mlip_hull import fixture
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
from phase_agent.analysis.phase.phase_boundary import assert_snapshot_boundary


def test_unknown_never_participates_in_standard_or_latest_hull(tmp_path):
    records, state = fixture(tmp_path)
    state["upload_layout"] = {"model_rounds": {"m": 1}}
    before = deepcopy(records)
    for latest in (False, True):
        result = update_phase_diagram(records, active_model_version="m", output_state=state,
                                      latest_per_structure=latest)
        hull = result["diagrams"]["mlip"]
        assert hull["status"] == "completed"
        assert {x["record_id"] for x in hull["entries"]} == {"left", "right"}
        assert all(x.get("phase") != "Unidentified" for x in hull["entries"])
    assert records == before


@pytest.mark.parametrize("row", [
    {"phase": "Unidentified", "phase_identification_status": "unknown", "unidentified_valid": True},
    {"phase": "O3", "phase_identification_status": "unknown"},
    {"phase": "X"},
])
def test_saved_snapshot_with_unknown_must_be_rebuilt(row):
    with pytest.raises(ValueError, match="未识别"):
        assert_snapshot_boundary({}, {"entries": [row]})


def test_legacy_identified_phase_without_status_remains_compatible():
    assert_snapshot_boundary({}, {"entries": [{"phase": "O3"}]})

def test_completed_record_with_explicit_unknown_is_not_admitted(tmp_path):
    records, state = fixture(tmp_path)
    records[-1]["status"] = "completed"
    records[-1]["phase"] = "O3"
    records[-1]["phase_identification_status"] = "unknown"
    diagram = update_phase_diagram(records, active_model_version="m", output_state=state)["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert all(x["record_id"] != "x" for x in diagram["entries"])
