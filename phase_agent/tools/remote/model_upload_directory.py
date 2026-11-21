"""One model epoch naming authority for uploads."""

from pathlib import Path


def model_upload_directory(root, state, model_version):
    label = str(model_version or "unversioned")
    rounds = state.setdefault("upload_layout", {}).setdefault("model_rounds", {})
    if label not in rounds:
        rounds[label] = max((int(v) for v in rounds.values()), default=0) + 1
    if int(rounds[label]) < 1 or any(
        other != label and int(number) == int(rounds[label]) for other, number in rounds.items()
    ):
        raise ValueError("Model epoch registration is ambiguous")
    from phase_agent.evidence.model_epoch import model_epoch

    epoch = model_epoch(state, label)
    if not epoch:
        raise ValueError("Model epoch registration is unavailable")
    return Path(root) / epoch
