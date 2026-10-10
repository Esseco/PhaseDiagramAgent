from phase_agent.analysis.state.summarize_phase_coverage import summarize_phase_coverage


def test_each_phase_has_own_representative():
    entries = [{"record_id": str(i), "phase": "O3", "ehull": 0,
                "x_Na_per_O2": .5} for i in range(15)]
    entries.append({"record_id": "p3", "phase": "P3", "ehull": .05,
                    "composition": {"Na": 1, "O": 2}})
    result = summarize_phase_coverage(entries)
    assert result["phases"][1]["lowest_ehull_entry"]["record_id"] == "p3"
    assert result["phases"][1]["observed_na_range"] == [1, 1]


def test_missing_evidence_is_not_zero():
    row = summarize_phase_coverage([{}])["phases"][0]
    assert row["phase"] == "unknown"
    assert row["observed_na_range"] is None
    assert row["missing_ehull_count"] == 1
    assert row["lowest_ehull_entry"] is None
