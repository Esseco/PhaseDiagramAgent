"""Mechanical CSV identity/path updates only; preserve every numerical cell string."""

import csv
import io

from phase_agent.tools.remote.migrate_legacy_upload_layout import _replace_paths


def rewrite_csv_metadata(path, replacements, epoch=None):
    original = path.read_bytes()
    text = original.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError(f"CSV列结构无法验证，不自动修改：{path}")
    changed = False
    identity = bool({"model_version", "hull_version", "task_id"}.intersection(fields))
    if epoch and identity and "epoch" not in fields and "model_epoch" not in fields:
        fields.insert(0, "epoch")
        changed = True
    for row in rows:
        if epoch and identity and "epoch" in fields and not row.get("epoch"):
            row["epoch"] = epoch
            changed = True
        for key in fields:
            if key.endswith("_path") and row.get(key):
                value = _replace_paths(row[key], replacements)
                if value != row[key]:
                    row[key] = value
                    changed = True
    if not changed:
        return original
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")
