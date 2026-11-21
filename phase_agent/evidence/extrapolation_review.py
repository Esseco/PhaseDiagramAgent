"""Read the current refresh's bounded, provisional endpoint-screening evidence."""

import hashlib
import csv
import json
import math
from pathlib import Path


def _confirmed_endpoint_review(state, ids, root):
    summary = state.get('extrapolation_review_summary')
    if not summary:
        return None
    try:
        if summary['model_version'] != state['active_model_version']:
            raise ValueError('model mismatch')
        meta_path = Path(summary['source']).resolve()
        if not meta_path.is_relative_to(Path(root).resolve()):
            raise ValueError('evidence outside analysis root')
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        tasks = {t.get('task_id'): t for t in state.get('tasks') or [] if t.get('task_id') in ids}
        flagged = [t for t in tasks.values() if (t.get('extrapolation_review') or {}).get('status') == 'extrapolation_failure']
        if not flagged:
            raise ValueError('no bound failure rows')
        csv_path = Path(flagged[0]['extrapolation_review']['source']).resolve()
        if not csv_path.is_relative_to(Path(root).resolve()):
            raise ValueError('CSV outside analysis root')
        checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if checksum != meta['csv_sha256'] or checksum != summary['evidence_sha256']:
            raise ValueError('CSV checksum mismatch')
        returns = {Path(b['handoff']['return_directory']).resolve() for b in state.get('slurm_batches') or [] if set(b.get('task_ids') or []) & ids and (b.get('handoff') or {}).get('return_directory')}
        if len(returns) != 1:
            raise ValueError('ambiguous return directory')
        directory = next(iter(returns))
        for name, expected in meta['source_archives_sha256'].items():
            if Path(name).name != name:
                raise ValueError('invalid archive name')
            input_dir = directory.with_name(directory.name.removesuffix('_output') + '_input')
            path = input_dir / name if name == 'inputs.zip' else directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('archive checksum mismatch')
        with csv_path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != meta['screened'] or len({r['task_id'] for r in rows}) != len(rows):
            raise ValueError('screened count mismatch')
        failures = set()
        for row in rows:
            score = float(row['score'])
            task = tasks.get(row['task_id']) or {}
            if (not math.isfinite(score) or score < 0 or row['error'] or task.get('model_version') != summary['model_version']
                    or (score > 1) != (row['classification'] == 'extrapolation_failure')):
                raise ValueError('invalid row')
            if score > 1:
                failures.add(row['task_id'])
                evidence = task.get('extrapolation_review') or {}
                if evidence.get('status') != 'extrapolation_failure' or evidence.get('score') != score or evidence.get('csv_sha256') != checksum:
                    raise ValueError('failure binding mismatch')
        if (failures != {t['task_id'] for t in flagged} or len(failures) != summary['count']
                or summary['screened_count'] != len(rows) or summary['not_screened_count'] != len(ids)-len(rows)):
            raise ValueError('summary mismatch')
        return {'status': 'endpoint_classified_incomplete', 'failure_count': len(failures), 'suspected_count': None,
                'screened_count': len(rows), 'task_count': len(ids), 'coverage_complete': len(rows) == len(ids),
                'force_jump_evidence_available': False, 'source_path': str(meta_path),
                'instruction': '按用户规则，末态特征差大于1标记为外推失败；其余未评估任务不能判断，灾难性遗忘和微调仍需进一步证据。'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {'status': 'unverified', 'failure_count': None, 'suspected_count': None, 'coverage_complete': False,
                'instruction': '外推证据或任务绑定校验失败，不能将缺失证据视为零异常。'}


def _base_extrapolation_review(state):
    refresh = state.get("model_refresh") or {}
    ids = set(refresh.get("task_ids") or [])
    root = state.get("analysis_output_directory")
    if not ids or not root:
        return {"status": "not_available", "suspected_count": None}
    if state.get("active_model_version") != refresh.get("new_model_version"):
        return {"status": "unverified", "suspected_count": None, "coverage_complete": False,
                "instruction": "刷新模型与当前活动模型不一致，外推证据需重新核验。"}
    confirmed = _confirmed_endpoint_review(state, ids, root)
    if confirmed is not None:
        return confirmed
    candidates = sorted(Path(root).glob("epoch*/extrapolation_review/*_外推证据说明.json"))
    tasks = {t.get("task_id"): t for t in state.get("tasks") or [] if t.get("task_id") in ids}
    returns = {
        Path(batch["handoff"]["return_directory"])
        for batch in state.get("slurm_batches") or []
        if set(batch.get("task_ids") or []) & ids
        and (batch.get("handoff") or {}).get("return_directory")
    }
    valid = []
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            archived = report["source_archives_sha256"]
            if len(returns) != 1 or not isinstance(archived, dict) or not archived:
                continue
            directory = next(iter(returns))
            if any(Path(name).name != name for name in archived):
                continue
            if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
                   for name, digest in archived.items()):
                continue
            rows = report["screened"]
            if not isinstance(rows, list) or len(rows) != report.get("endpoint_screened"):
                continue
            selected = set()
            for row in rows:
                tid, score = row["task_id"], row["score"]
                task = tasks.get(tid) or {}
                if (tid in selected or tid not in ids or isinstance(score, bool)
                        or not isinstance(score, (int, float)) or not math.isfinite(score)
                        or score < 0 or task.get("model_version") != refresh.get("new_model_version")
                        or task.get("result_disposition") != "discarded_unconverged"
                        or (task.get("outputs") or {}).get("optimizer_converged") is not False
                        or row.get("monitor_force_jump_available") is not False):
                    raise ValueError("screening identity or evidence mismatch")
                selected.add(tid)
            count = sum(row["score"] > 1.0 for row in rows)
            if count != report.get("endpoint_difference_gt_1"):
                continue
            valid.append({
                "status": "provisional_incomplete", "suspected_count": count,
                "screened_count": len(rows), "task_count": len(ids),
                "coverage_complete": False, "force_jump_evidence_available": False,
                "source_path": str(path),
                "instruction": "末态特征差大于1仅为疑似外推失败；缺少逐步受力突增和全轮覆盖，不能判定崩塌比例、灾难性遗忘或自动触发微调。",
            })
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    if len(valid) == 1:
        return valid[0]
    return {"status": "unverified" if candidates else "not_available", "suspected_count": None,
            "coverage_complete": False,
            "instruction": "外推证据缺失、绑定不匹配或存在多个候选，不能当成零异常。"}


def extrapolation_review(state):
    """Combine independent endpoint evidence without claiming step monitoring."""
    from phase_agent.evidence.endpoint_supplement import supplement_facts

    base = _base_extrapolation_review(state)
    extra = supplement_facts(state)
    if extra is None:
        return base
    if extra.get("invalid") or base.get("status") != "endpoint_classified_incomplete":
        return {"status": "unverified", "failure_count": None, "coverage_complete": False,
                "force_jump_evidence_available": False, "instruction": "补充外推证据校验失败。"}
    try:
        flagged = next(t for t in state.get("tasks") or [] if (t.get("extrapolation_review") or {}).get("status") == "extrapolation_failure")
        with Path(flagged["extrapolation_review"]["source"]).open(encoding="utf-8-sig", newline="") as stream:
            old_ids = {row["task_id"] for row in csv.DictReader(stream)}
        scope = set((state.get("model_refresh") or {}).get("task_ids") or [])
        if len(old_ids) != base["screened_count"] or old_ids & extra["task_ids"] or not extra["task_ids"].issubset(scope):
            raise ValueError("endpoint_supplement_scope_invalid")
        total = len(old_ids | extra["task_ids"])
        return {**base, "failure_count": base["failure_count"] + extra["failure_count"],
                "screened_count": total, "coverage_complete": total == len(scope),
                "force_jump_evidence_available": False,
                "instruction": "末态筛查按任务和帧绑定；逐步受力证据缺失，不能据此自动判断灾难性遗忘。"}
    except (OSError, KeyError, ValueError, StopIteration):
        return {"status": "unverified", "failure_count": None, "coverage_complete": False,
                "force_jump_evidence_available": False, "instruction": "补充外推证据与原证据重叠或失效。"}
