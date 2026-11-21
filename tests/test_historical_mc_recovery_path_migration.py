import json
import shutil
from pathlib import Path

from tests.test_historical_mc_recovery import setup
from phase_agent.tools.relax_mc_batch.recovery_receipt import selected_recovery


def test_legacy_receipt_path_resolves_after_delivery_migration(tmp_path):
    record, state, archive = setup(tmp_path)
    directory = Path(record["directory"])
    root = directory.with_name(directory.name + "_recovery")
    migrated = root / "verified" / "migrated.results.zip"
    migrated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, migrated)
    receipt = {"archive": str(tmp_path / "old" / "relax_mc" / "recovered.results.zip"),
               "archive_checksum": __import__("hashlib").sha256(migrated.read_bytes()).hexdigest()}
    (directory / "recovery_return.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert selected_recovery(record) == migrated.resolve()
