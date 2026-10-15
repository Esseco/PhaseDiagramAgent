"""Lossless column tables for repetitive scientific evidence sent to an LLM."""

from copy import deepcopy
import json
from os.path import commonprefix


TABLE_FIELDS = {
    "available_branches",
    "qbc_candidates",
    "search_coverage",
    "evidence_catalog",
    "tasks",
    "pending_tasks",
    "segments",
    "entries",
    "stable_entries",
    "search_history",
    "recent_actions",
    "branch_candidates",
    "mc_pool",
    "coverage",
    "candidate_strata",
    "dft_candidate_costs",
}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def table_rows(rows):
    if not isinstance(rows, list) or len(rows) < 8 or not all(isinstance(r, dict) for r in rows):
        return rows
    # Group mixed schemas without inventing null values or losing row order.
    columns = list(rows[0])
    if not columns:
        return rows
    if any(set(row) != set(columns) for row in rows):
        groups = {}
        for index, row in enumerate(rows):
            group = groups.setdefault(tuple(sorted(row)), {"indices": [], "rows": []})
            group["indices"].append(index)
            group["rows"].append(row)
        table = {
            "format": "schema_group_table_v1",
            "count": len(rows),
            "groups": [
                {"indices": g["indices"], "table": table_rows(g["rows"])} for g in groups.values()
            ],
        }
        return table if _size(table) < _size(rows) else rows
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
    # Preserve full file paths, sharing their common prefix rather than repeating it.
    prefixed = deepcopy(indexed if dictionaries else table)
    prefixes = {}
    for index, column in enumerate(columns):
        if not column.endswith("_path"):
            continue
        if column in dictionaries:
            values = prefixed["dictionaries"][column]
        else:
            values = [row[index] for row in prefixed["rows"]]
        if not values or not all(isinstance(value, str) for value in values):
            continue
        prefix = commonprefix(values)
        if len(prefix) < 32:
            continue
        prefixes[column] = prefix
        suffixes = [value[len(prefix) :] for value in values]
        if column in dictionaries:
            prefixed["dictionaries"][column] = suffixes
        else:
            for row, suffix in zip(prefixed["rows"], suffixes, strict=True):
                row[index] = suffix
    if prefixes:
        prefixed.update(format="prefix_column_table_v1", prefixes=prefixes)
        if _size(prefixed) < _size(best):
            best = prefixed
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
    """Decode tables first, then resolve exact aliases against the decoded root."""
    from phase_agent.decisions.agent.transport_references import expand_transport_references

    return expand_transport_references(_expand_evidence_tables(value))


def _expand_evidence_tables(value):
    if isinstance(value, dict):
        if value.get("format") == "schema_group_table_v1":
            rows = [None] * value["count"]
            for group in value["groups"]:
                for index, row in zip(
                    group["indices"],
                    _expand_evidence_tables(group["table"]),
                    strict=True,
                ):
                    rows[index] = row
            if any(row is None for row in rows):
                raise ValueError("Incomplete grouped evidence table")
            return rows
        if value.get("format") in {
            "column_table_v1",
            "dictionary_column_table_v1",
            "prefix_column_table_v1",
        }:
            dictionaries = value.get("dictionaries") or {}
            return [
                {
                    **_expand_evidence_tables(value["common"]),
                    **{
                        key: _expand_evidence_tables(
                            value.get("prefixes", {}).get(key, "")
                            + (dictionaries[key][item] if key in dictionaries else item)
                            if key in value.get("prefixes", {})
                            else dictionaries[key][item]
                            if key in dictionaries
                            else item
                        )
                        for key, item in zip(value["columns"], row, strict=True)
                    },
                }
                for row in value["rows"]
            ]
        return {key: _expand_evidence_tables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_evidence_tables(item) for item in value]
    return deepcopy(value)


def evidence_row_count(value):
    if isinstance(value, dict) and value.get("format") == "evidence_ref_v1":
        return int(value.get("count", 0))
    if isinstance(value, dict) and value.get("format") in {
        "column_table_v1",
        "dictionary_column_table_v1",
        "schema_group_table_v1",
        "prefix_column_table_v1",
    }:
        return value["count"]
    return len(value) if isinstance(value, list) else 0
