import os
import sqlite3

import pytest

from phase_agent.persistence.sqlite_paths import readonly_uri


@pytest.mark.parametrize("extended", [False, True])
def test_readonly_sqlite_handles_unicode_reserved_characters_and_extended_paths(tmp_path, extended):
    if extended and os.name != "nt":
        pytest.skip("Windows extended paths")
    path = tmp_path / "结构 # percent%.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE facts (value INTEGER)")
        connection.execute("INSERT INTO facts VALUES (1)")
    value = "\\\\?\\" + str(path) if extended else path
    with sqlite3.connect(readonly_uri(value), uri=True) as connection:
        assert connection.execute("SELECT value FROM facts").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO facts VALUES (2)")


@pytest.mark.skipif(os.name != "nt", reason="Windows extended paths")
def test_recovery_receipt_can_be_read_from_extended_path(tmp_path):
    from pathlib import Path
    from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report

    path = Path("\\\\?\\" + str(tmp_path)) / "state.json"
    identity = dict(
        invocation_id="a", config_version="c", action_hash="h", tool="generate_branches"
    )
    assert begin_execution(path, identity)["allowed"]
    assert recovery_report(path, {})["status"] == "reconciliation_required"
