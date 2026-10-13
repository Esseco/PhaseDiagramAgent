"""Lossless column tables for repetitive scientific evidence sent to an LLM."""

from copy import deepcopy
import json


TABLE_FIELDS = {
    "available_branches",
    "qbc_candidates",
    "search_coverage",
    "evidence_catalog",
}


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
    best = table if _size(table) < _size(rows) else rows
    indexed = deepcopy(table)
    dictionaries = {}
    for index, column in enumerate(columns):
        values, indices, lookup = [], [], {}
        for row in indexed["rows"]:
            value = row[index]
            key = json.dumps(value, sort_keys=True, default=str)
            if key not in lookup:
                lookup[key] = len(values)
                values.append(value)
            indices.append(lookup[key])
        if _size(values) + _size(indices) + len(column) + 4 < _size(
            [row[index] for row in indexed["rows"]]
        ):
            dictionaries[column] = values
            for row, value_index in zip(indexed["rows"], indices, strict=True):
                row[index] = value_index
    if dictionaries:
        indexed.update(format="dictionary_column_table_v1", dictionaries=dictionaries)
        if _size(indexed) < _size(best):
            best = indexed
    return best


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
        if value.get("format") in {"column_table_v1", "dictionary_column_table_v1"}:
            dictionaries = value.get("dictionaries") or {}
            return [
                {
                    **deepcopy(value["common"]),
                    **{
                        key: expand_evidence_tables(
                            dictionaries[key][item] if key in dictionaries else item
                        )
                        for key, item in zip(value["columns"], row, strict=True)
                    },
                }
                for row in value["rows"]
            ]
        return {key: expand_evidence_tables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_evidence_tables(item) for item in value]
    return deepcopy(value)


def evidence_row_count(value):
    if isinstance(value, dict) and value.get("format") in {
        "column_table_v1",
        "dictionary_column_table_v1",
    }:
        return value["count"]
    return len(value) if isinstance(value, list) else 0
