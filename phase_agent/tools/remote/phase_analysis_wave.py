"""CPU-only phase analysis for a fully returned, frozen model-refresh wave."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from phase_agent.analysis.phase.identify_result_phase import identify_result_phase
from phase_agent.tools.remote.integrity import file_checksum, verify_result


def _within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("phase_analysis_path_outside_handoff")
    return path


GROUP_SIZE = 200


def analyze_wave(root, spec, *, classify=identify_result_phase, group_index=None):
    """Check every frozen receipt before writing any separate phase evidence."""
    root = Path(root).resolve()
    rows = spec.get("tasks") or []
    if not rows or len({row.get("task_id") for row in rows}) != len(rows):
        raise ValueError("phase_analysis_invalid_task_scope")
    from pymatgen.core import Structure

    references = {
        name: Structure.from_dict(value) for name, value in (spec.get("references") or {}).items()
    }
    data = deepcopy(spec["manager"])
    manager = SimpleNamespace(boundary=data["boundary"], data=data)
    ready = []
    for row in rows:
        result_path = _within(root, row["result_path"])
        checked = verify_result(result_path, result_path.with_name("task.finished.json"), row)
        if not checked["valid"]:
            raise ValueError(
                "phase_analysis_unverified_result:" + row["task_id"] + ":" + checked["reason"]
            )
        result = checked["result"]
        if result.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
            raise ValueError("phase_analysis_nonterminal_result:" + row["task_id"])
        if result.get("status") != "completed":
            continue
        outputs = result.get("outputs") or {}
        value = outputs.get("structure_path") or outputs.get("final_structure_path")
        path = result_path.parent / Path(str(value or "").replace("\\", "/")).name
        if not value or not path.is_file() or path.resolve().parent != result_path.parent.resolve():
            raise ValueError("phase_analysis_final_structure_missing:" + row["task_id"])
        if outputs.get("structure_checksum") != file_checksum(path):
            raise ValueError("phase_analysis_final_structure_mismatch:" + row["task_id"])
        ready.append((row, result, result_path, path))
    if group_index is not None:
        if type(group_index) is not int or group_index < 0 or group_index * GROUP_SIZE >= len(rows):
            raise ValueError("phase_analysis_invalid_group")
        selected = {
            row["task_id"]
            for row in rows[group_index * GROUP_SIZE : (group_index + 1) * GROUP_SIZE]
        }
        ready = [item for item in ready if item[0]["task_id"] in selected]
    written = []
    scope_cache = {}
    for row, result, result_path, path in ready:
        outputs = {**result.get("outputs", {}), "structure_path": str(path)}
        followup = outputs.get("refresh_force_followup") or {}
        is_single = (
            outputs.get("single_point_completed") is True and followup.get("relaxed") is not True
        )
        if is_single:
            from phase_agent.science.structures.load_result_structure import load_result_structure
            from phase_agent.analysis.phase.identify_result_phase import (
                _configured_phase,
                _classification_scope,
                _portable_classification_scope,
            )

            structure, digest = load_result_structure(outputs)
            if digest != row.get("input_structure_sha256"):
                raise ValueError("phase_analysis_single_point_input_mismatch:" + row["task_id"])
            configured = _configured_phase(structure, manager)
            scope = _classification_scope(manager, references, configured_phase=configured)
            portable = _portable_classification_scope(
                manager, references, configured_phase=configured
            )
            source = row.get("source_phase") or {}
            if (
                digest == row.get("input_structure_sha256")
                and source.get("structure_sha256") == digest
                and portable
                and source.get("portable_classification_scope") == portable
                and source.get("status") in {"identified", "outside_boundary", "unknown"}
            ):
                evidence = deepcopy(source)
            else:
                outputs.pop("phase_identification", None)
                classified = classify(
                    {**result, "branch_id": row.get("branch_id"), "outputs": outputs},
                    manager,
                    phase_references=references,
                    cache=None,
                    scope_cache=scope_cache,
                )
                evidence = (classified.get("outputs") or {}).get("phase_identification") or {}
        else:
            # A relaxation produces a new frame; classify it afresh.
            outputs.pop("phase_identification", None)
            classified = classify(
                {**result, "branch_id": row.get("branch_id"), "outputs": outputs},
                manager,
                phase_references=references,
                cache=None,
                scope_cache=scope_cache,
            )
            evidence = (classified.get("outputs") or {}).get("phase_identification") or {}
        if evidence.get("structure_sha256") != file_checksum(path) or not evidence.get(
            "portable_classification_scope"
        ):
            raise ValueError("phase_analysis_evidence_unbound:" + row["task_id"])
        evidence["origin"] = "remote_cpu"
        sidecar = {
            "schema": 1,
            "task_id": row["task_id"],
            "task_checksum": row.get("task_checksum"),
            "result_checksum": file_checksum(result_path),
            "phase_identification": evidence,
        }
        target = result_path.with_name("phase_identification.json")
        if target.is_file():
            if json.loads(target.read_text(encoding="utf8")) != sidecar:
                raise ValueError("phase_analysis_existing_evidence_conflict:" + row["task_id"])
            written.append(row["task_id"])
            continue
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(sidecar, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf8"
        )
        temporary.replace(target)
        written.append(row["task_id"])
    return {"status": "completed", "task_ids": written, "terminal_count": len(rows)}


def write_group_receipt(root, spec_path, index, *, classify=identify_result_phase):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    result = analyze_wave(root, spec, classify=classify, group_index=index)
    rows = spec["tasks"][index * GROUP_SIZE : (index + 1) * GROUP_SIZE]
    artifacts = {}
    for task_id in result["task_ids"]:
        row = next(item for item in rows if item["task_id"] == task_id)
        sidecar = _within(Path(root).resolve(), row["result_path"]).with_name(
            "phase_identification.json"
        )
        artifacts[task_id] = file_checksum(sidecar)
    receipt = {
        "schema": 1,
        "spec_checksum": file_checksum(spec_path),
        "group": index,
        "scope": [item["task_id"] for item in rows],
        "artifacts": artifacts,
    }
    target = spec_path.parent / f"phase-group-{index:05d}.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf8")) != receipt:
            raise ValueError("phase_analysis_group_receipt_conflict")
    else:
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, sort_keys=True), encoding="utf8")
        temp.replace(target)
    return receipt


def verify_groups(root, spec_path):
    spec_path = Path(spec_path).resolve()
    root = Path(root).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    rows = spec.get("tasks") or []
    if not rows or len({row.get("task_id") for row in rows}) != len(rows):
        raise ValueError("phase_analysis_invalid_task_scope")
    for index, offset in enumerate(range(0, len(rows), GROUP_SIZE)):
        group = rows[offset : offset + GROUP_SIZE]
        path = spec_path.parent / f"phase-group-{index:05d}.json"
        if not path.is_file():
            raise ValueError("phase_analysis_group_receipt_missing:" + str(index))
        receipt = json.loads(path.read_text(encoding="utf8"))
        if (
            receipt.get("schema") != 1
            or receipt.get("group") != index
            or receipt.get("spec_checksum") != file_checksum(spec_path)
            or receipt.get("scope") != [row["task_id"] for row in group]
        ):
            raise ValueError("phase_analysis_group_receipt_mismatch:" + str(index))
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("phase_analysis_group_artifacts_invalid")
        expected = set()
        for row in group:
            result_path = _within(root, row["result_path"])
            checked = verify_result(result_path, result_path.with_name("task.finished.json"), row)
            if not checked["valid"]:
                raise ValueError("phase_analysis_group_result_invalid:" + row["task_id"])
            status = checked["result"].get("status")
            if status not in {"completed", "failed", "timeout", "cancelled"}:
                raise ValueError("phase_analysis_group_nonterminal:" + row["task_id"])
            if status == "completed":
                expected.add(row["task_id"])
                sidecar = result_path.with_name("phase_identification.json")
                if not sidecar.is_file() or file_checksum(sidecar) != artifacts.get(row["task_id"]):
                    raise ValueError("phase_analysis_group_evidence_invalid:" + row["task_id"])
                evidence = json.loads(sidecar.read_text(encoding="utf8"))
                output = checked["result"].get("outputs") or {}
                value = output.get("structure_path") or output.get("final_structure_path")
                structure = result_path.parent / Path(str(value or "").replace("\\", "/")).name
                if (
                    not value
                    or not structure.is_file()
                    or structure.resolve().parent != result_path.parent.resolve()
                    or output.get("structure_checksum") != file_checksum(structure)
                    or evidence.get("task_id") != row["task_id"]
                    or evidence.get("task_checksum") != row.get("task_checksum")
                    or evidence.get("result_checksum") != file_checksum(result_path)
                    or (evidence.get("phase_identification") or {}).get("structure_sha256")
                    != file_checksum(structure)
                ):
                    raise ValueError("phase_analysis_group_evidence_invalid:" + row["task_id"])
        if set(artifacts) != expected:
            raise ValueError("phase_analysis_group_evidence_scope_mismatch")
    return {"status": "completed", "group_count": (len(rows) + GROUP_SIZE - 1) // GROUP_SIZE}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("root")
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--spec-checksum")
    args = parser.parse_args()
    if args.spec_checksum and file_checksum(args.spec) != args.spec_checksum:
        raise ValueError("phase_analysis_frozen_spec_changed")
    if args.finalize and args.group_index is not None:
        parser.error("finalize and group-index are exclusive")
    if args.finalize:
        result = verify_groups(args.root, args.spec)
    elif args.group_index is not None:
        result = write_group_receipt(args.root, args.spec, args.group_index)
    else:
        result = analyze_wave(args.root, json.loads(Path(args.spec).read_text(encoding="utf8")))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
