"""Lossless column tables for repetitive scientific evidence sent to an LLM."""

from copy import deepcopy
import json
from os.path import commonprefix


TABLE_FIELDS = {
    "task_observations",
    "reports",
    "comparable_cohorts",
    "observed_budget_outcomes",
    "decision_feedback",
    "model_round_summary",
    "stages",
    "existing_branches",
    "available_allocations",
    "lowest_ehull_entries",
    "recent_action_history",
    "selection_purposes",
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
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(r, dict) for r in rows):
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
    # Encode each cell once, retaining canonical keys for only one column.
    # Size accounting uses the same character unit as _size, not token estimates.
    count = len(rows)
    common, dictionaries, indices_by_column = {}, {}, {}
    variable = []
    cell_size = 0
    for column in columns:
        values, indices, lookup = [], [], {}
        unique_size = column_size = 0
        for row in rows:
            value = row[column]
            key = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            )
            column_size += len(key)
            if key not in lookup:
                lookup[key] = len(values)
                values.append(value)
                unique_size += len(key)
            indices.append(lookup[key])
        cell_size += column_size
        if len(values) == 1:
            common[column] = values[0]
            continue
        variable.append(column)
        original_column_size = 2 + count - 1 + column_size
        dictionary_size = 2 + len(values) - 1 + unique_size
        index_size = 2 + count - 1 + sum(len(str(index)) for index in indices)
        if dictionary_size + index_size + len(column) + 4 < original_column_size:
            dictionaries[column] = values
            indices_by_column[column] = indices
    # Release the last column's encoded lookup before any candidate-size encoding.
    del values, indices, lookup, key
    # JSON object syntax and repeated keys, without serializing all input rows again.
    key_size = sum(len(json.dumps(key, ensure_ascii=False)) + 1 for key in columns)
    original_size = 2 + count - 1 + count * (2 + len(columns) - 1 + key_size) + cell_size
    table = {
        "format": "column_table_v1",
        "count": count,
        "common": common,
        "columns": variable,
        "rows": [[row[key] for key in variable] for row in rows],
    }
    table_size = _size(table)
    best, best_size = (table, table_size) if table_size < original_size else (rows, original_size)
    candidate = table
    if dictionaries:
        candidate = {
            **table,
            "format": "dictionary_column_table_v1",
            "dictionaries": dictionaries,
            "rows": [
                [
                    indices_by_column[key][index] if key in dictionaries else row[key]
                    for key in variable
                ]
                for index, row in enumerate(rows)
            ],
        }
        size = _size(candidate)
        if size < best_size:
            best, best_size = candidate, size
    prefixes, suffixes = {}, {}
    for index, column in enumerate(variable):
        if not column.endswith("_path"):
            continue
        values = (
            dictionaries[column]
            if column in dictionaries
            else [row[index] for row in candidate["rows"]]
        )
        if not values or not all(isinstance(value, str) for value in values):
            continue
        prefix = commonprefix(values)
        if len(prefix) >= 32:
            prefixes[column] = prefix
            suffixes[column] = [value[len(prefix) :] for value in values]
    if prefixes:
        # Only row-cell arrays change; nested evidence is never copied for size trials.
        prefixed = {
            **candidate,
            "format": "prefix_column_table_v1",
            "prefixes": prefixes,
            "rows": [list(row) for row in candidate["rows"]],
        }
        if dictionaries:
            prefixed["dictionaries"] = dict(dictionaries)
        for index, column in enumerate(variable):
            if column not in prefixes:
                continue
            if column in dictionaries:
                prefixed["dictionaries"][column] = suffixes[column]
            else:
                for row, suffix in zip(prefixed["rows"], suffixes[column], strict=True):
                    row[index] = suffix
        if _size(prefixed) < best_size:
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
    from phase_agent.models.transport_references import expand_transport_references

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
