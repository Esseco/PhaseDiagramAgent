"""Maintain a discoverable current CSV without touching unchanged files."""
from pathlib import Path
import shutil


def publish_current_csv(snapshot, archive_path):
    archive = Path(archive_path)
    current = archive.parent.parent / "phase_diagram.csv"
    if not current.is_file() or current.read_bytes() != archive.read_bytes():
        temporary = current.with_suffix(".csv.tmp")
        shutil.copy2(archive, temporary)
        temporary.replace(current)
    snapshot["archive_csv_path"] = str(archive)
    snapshot["csv_path"] = str(current)
    return current
