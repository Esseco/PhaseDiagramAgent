"""One model epoch naming authority for uploads."""
from pathlib import Path
from execution_layer.remote.build_upload_batch_directory import _safe


def model_upload_directory(root, state, model_version):
    label = str(model_version or "unversioned")
    rounds = state.setdefault("upload_layout", {}).setdefault("model_rounds", {})
    if label not in rounds:
        rounds[label] = max((int(v) for v in rounds.values()), default=0) + 1
    return Path(root) / f"epoch{int(rounds[label])-1}_{_safe(label)}"
