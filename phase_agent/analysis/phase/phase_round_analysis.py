"""One derived plot/quality receipt per verified complete Relax/MC round."""

from hashlib import sha256
import json
from pathlib import Path
import re

from phase_agent.analysis.phase.phase_quality_reports import (
    get_phase_quality_report,
    review_markdown,
    _atomic_text,
    _digest,
    priority_group_representatives,
    analysis_layer_summaries,
)
from phase_agent.analysis.phase.phase_quality import RULE_VERSION
from phase_agent.analysis.phase.endpoint_reference import PLOT_RULE_VERSION
from phase_agent.tools.step_runner.file_protocol import write_json


def publish_relax_mc_analysis(state, bundle_id, *, directory=None):
    # Derived plotting/report failures must not turn verified imports into failures.
    try:
        return _publish(state, bundle_id, directory=directory)
    except Exception as error:
        return {
            "status": "diagnostic_unavailable",
            "reason": f"{type(error).__name__}: {error}",
            "submitted": False,
        }


def _publish(state, bundle_id, *, directory=None):
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    summary = batch.get("result_summary") or {}
    model = state.get("active_model_version")
    waiting = any(
        task.get("relax_mc_bundle_id") == bundle_id
        and task.get("status") not in {"completed", "failed", "timeout", "cancelled"}
        for task in state.get("tasks") or []
    )
    ready = (
        model
        and batch.get("model_version") == model
        and batch.get("status") == "completed"
        and batch.get("import_checksum")
        and not batch.get("return_error")
        and summary.get("status") == "completed"
        and summary.get("model_version") == model
        and (
            not batch.get("plan_checksum") or summary.get("plan_checksum") == batch["plan_checksum"]
        )
        and not waiting
    )
    if not ready:
        return {
            "status": "not_ready",
            "reason": "本轮回传尚未完整核验或模型/方案绑定不符",
            "submitted": False,
        }
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        snapshot.get("model_version") != model
        or snapshot.get("status") != "completed"
        or not snapshot.get("entries")
    ):
        return {"status": "not_ready", "reason": "尚无本轮当前模型的完整相图", "submitted": False}
    stored = snapshot.get("csv_path") or snapshot.get("path")
    if not stored:
        return {
            "status": "diagnostic_unavailable",
            "reason": "缺少已登记相图输出位置",
            "submitted": False,
        }
    folder = Path(stored).resolve().parent
    if folder.name == "history":
        folder = folder.parent
    identity = {
        "bundle_id": bundle_id,
        "model_version": model,
        "import_checksum": batch["import_checksum"],
    }
    token = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    receipt = folder / "quality" / "rounds" / f"relax_mc_{token}.json"
    if receipt.is_file():
        try:
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if all(previous.get(key) == value for key, value in identity.items()):
                artifacts = [
                    *(previous.get("plot_paths") or []),
                    previous.get("markdown_path"),
                    previous.get("review_report_path"),
                ]
                if (
                    previous.get("status") == "completed"
                    and previous.get("diagnostic_rule_version") == RULE_VERSION
                    and previous.get("plot_rule_version") == PLOT_RULE_VERSION
                    and previous.get("plot_paths")
                    and all(p and Path(p).is_file() for p in artifacts)
                ):
                    return previous
                if previous.get("hull_version") != snapshot.get("version"):
                    return {
                        "status": "diagnostic_unavailable",
                        "reason": "历史轮次图片缺失且当前相图已改变，不能冒充原轮次重画",
                        "submitted": False,
                    }
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    from phase_agent.runtime.phase_diagram_chat import phase_diagram_reply

    try:
        reply = phase_diagram_reply(state, operation="plot", method="mlip", directory=directory)
        paths = [
            p
            for p in re.findall(r"`([^`]+)`", reply)
            if Path(p).suffix.lower() in {".png", ".svg", ".pdf"}
        ]
        completed = bool(paths) and all(Path(p).is_file() for p in paths)
        report = get_phase_quality_report(state, write=True)
        report.update(**identity)
        review_path = receipt.with_suffix(".review.json")
        markdown_path = receipt.with_suffix(".md")
        # Bind the human-facing numbering to this immutable verified round.
        report["review_report_path"] = str(review_path)
        report.pop("report_checksum", None)
        report["report_checksum"] = _digest(report)
        write_json(review_path, report)
        _atomic_text(markdown_path, review_markdown(report, bundle_id=bundle_id, plot_paths=paths))
        result = {
            **identity,
            "status": "completed" if completed else "diagnostic_unavailable",
            "hull_version": snapshot.get("version"),
            "diagnostic_rule_version": RULE_VERSION,
            "plot_rule_version": PLOT_RULE_VERSION,
            "plot_paths": paths,
            "quality_evidence": {
                "status": report["status"],
                "decision_role": report["decision_role"],
                "analysis_layers": analysis_layer_summaries(report),
                "priority_structures": priority_group_representatives(report),
            },
            "markdown_path": str(markdown_path),
            "review_report_path": str(review_path),
            "reply": reply,
            "receipt_path": str(receipt),
            "submitted": False,
        }
        write_json(receipt, result)
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {
            **identity,
            "status": "diagnostic_unavailable",
            "reason": str(error),
            "submitted": False,
        }
