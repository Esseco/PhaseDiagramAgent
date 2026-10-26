"""Human report names from recorded epoch/stage; no history scan or business writes."""

from pathlib import Path
from functools import lru_cache
from itertools import count
import re


def report_directory(root, state):
    from phase_agent.analysis.state.model_epoch import model_epoch

    version = state.get("active_model_version")
    epoch = model_epoch(state, version) or ("epoch0" if not version else "unassigned")
    return Path(root).resolve() / "interaction_records" / epoch


def report_stem(state, record, now):
    from phase_agent.runtime.chat_state_presentation import epoch_stage_line

    _, stage = epoch_stage_line(state).split(" · ", 1)
    if (record or {}).get("failure_summary"):
        stage = "interaction_error"
    else:
        # Keep report paths ASCII so they are portable across shells and tools.
        for prefix, label in (
            ("补充 DFT 决策", "dft_selection"),
            ("DFT 输入方案", "dft_selection"),
            ("branch 生成方案", "structure_generation"),
            ("MC 分配方案", "mc_allocation"),
            ("新模型", "model_refresh"),
            ("微调", "model_training"),
            ("MC ", "mc_recovery"),
            ("Relax ", "relax_recovery"),
            ("DFT ", "dft_recovery"),
            ("无待运行任务", "workflow_check"),
        ):
            if stage.startswith(prefix):
                stage = label
                break
    stage = re.split(r"[，；。]", stage, maxsplit=1)[0]
    stage = re.sub(r"[^A-Za-z0-9_-]+", "_", stage).strip("_").lower()[:32] or "interaction"
    return f"{stage}_{now.astimezone().strftime('%Y%m%d')}"


@lru_cache(maxsize=64)
def _sequence(directory, stem):
    return count(1)


def reserve_report(directory, stem, record):
    """Reserve atomically across writers; a retry with the same record keeps its path."""
    existing = (record or {}).get("interaction_report")
    if existing:
        path = Path(existing).resolve()
        if path.parent == directory and path.suffix == ".md":
            return path, False
    for number in _sequence(str(directory), stem):
        path = directory / f"{stem}_{number:02d}.md"
        try:
            with path.open("x", encoding="utf-8"):
                pass
            return path, True
        except FileExistsError:
            continue
