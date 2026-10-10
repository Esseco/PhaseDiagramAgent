"""为候选批次生成可追溯的逐模型 QBC 文件。"""

import json
from pathlib import Path

from phase_agent.science.qbc.evaluate_qbc import evaluate_qbc


def generate_qbc_file(
    candidates: list[dict], committee: dict, output_path, *, predictor, config=None
) -> dict:
    settings = {
        "minimum_models": 3,
        "energy_threshold": None,
        "force_rms_threshold": None,
        "force_max_threshold": None,
    }
    settings.update(config or {})
    rows = []
    for candidate in candidates:
        result = evaluate_qbc(candidate["structure"], committee, predictor=predictor)
        flags = []
        if result.get("status") == "completed":
            for field, threshold in (
                ("energy_per_atom_std", settings["energy_threshold"]),
                ("force_rms_disagreement", settings["force_rms_threshold"]),
                ("force_max_atom_disagreement", settings["force_max_threshold"]),
            ):
                if threshold is not None and result[field] >= threshold:
                    flags.append(field)
        rows.append(
            {
                "structure_id": candidate.get("structure_id"),
                "branch_id": candidate.get("branch_id"),
                "committee_id": committee.get("committee_id"),
                "qbc": result,
                "threshold_flags": flags,
            }
        )
    status = (
        "completed"
        if len(committee.get("loaded_members", [])) >= settings["minimum_models"]
        else "insufficient_committee"
    )
    report = {
        "status": status,
        "committee_id": committee.get("committee_id"),
        "config": settings,
        "count": len(rows),
        "results": rows,
        "interpretation": "committee_disagreement_not_true_error",
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)
    return report
