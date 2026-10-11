"""Lossless column tables for repetitive scientific evidence sent to an LLM."""

from copy import deepcopy
import json


TABLE_FIELDS = {"available_branches", "qbc_candidates", "search_coverage", "evidence_catalog"}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def table_rows(rows):
    if not isinstance(rows, list) or len(rows) < 8 or not all(isinstance(r, dict) for r in rows):
        return rows
    # Different schemas stay as objects, preserving missing versus null fields.
    columns = list(rows[0])
    if not columns or any(set(row) != set(columns) for row in rows):
        return rows
    common = {key: rows[0][key] for key in columns if all(r[key] == rows[0][key] for r in rows)}
    columns = [key for key in columns if key not in common]
    table = {
        "format": "column_table_v1",
        "count": len(rows),
        "common": common,
        "columns": columns,
        "rows": [[row[key] for key in columns] for row in rows],
    }
    return table if _size(table) < _size(rows) else rows


def compact_evidence_tables(value):
    if isinstance(value, dict):
        return {
            key: table_rows(compact_evidence_tables(item))
            if key in TABLE_FIELDS
            else compact_evidence_tables(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [compact_evidence_tables(item) for item in value]
    return value


def expand_evidence_tables(value):
    """Decode transport evidence for offline consumers and equivalence checks."""
    if isinstance(value, dict):
        if value.get("format") == "column_table_v1":
            return [
                {
                    **deepcopy(value["common"]),
                    **dict(zip(value["columns"], expand_evidence_tables(row), strict=True)),
                }
                for row in value["rows"]
            ]
        return {key: expand_evidence_tables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_evidence_tables(item) for item in value]
    return deepcopy(value)


def evidence_row_count(value):
    if isinstance(value, dict) and value.get("format") == "column_table_v1":
        return value["count"]
    return len(value) if isinstance(value, list) else 0
