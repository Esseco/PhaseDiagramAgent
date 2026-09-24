"""Create and persist a deterministic leakage-resistant group split."""

import hashlib
import json
import random
from copy import deepcopy


def split_branch_dataset(dataset: dict, *, config: dict) -> dict:
    settings = config.get("split") or {}
    fractions = [float(settings.get(name, value)) for name, value in (("train", .7), ("validation", .15), ("test", .15))]
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("train/validation/test 比例之和必须为 1")
    groups = {}
    for row in dataset["rows"]:
        key = _group_key(row, config.get("group_mode", "framework"))
        groups.setdefault(key, []).append(row["branch_id"])
        row["split_group"] = key
    keys = sorted(groups)
    random.Random(int(settings.get("seed", 0))).shuffle(keys)
    cut1, cut2 = round(len(keys) * fractions[0]), round(len(keys) * (fractions[0] + fractions[1]))
    membership = {key: "train" if i < cut1 else "validation" if i < cut2 else "test" for i, key in enumerate(keys)}
    output = deepcopy(dataset)
    for row in output["rows"]:
        row["split_group"] = _group_key(row, config.get("group_mode", "framework"))
        row["split"] = membership[row["split_group"]]
    output["split_manifest"] = {"seed": settings.get("seed", 0), "group_mode": config.get("group_mode", "framework"), "groups": membership}
    return output


def _group_key(row, mode):
    branch = row["branch"]
    if mode == "framework" and branch.get("framework_id"):
        payload = {"framework_id": branch["framework_id"]}
    else:
        payload = {"P": branch.get("P"), "H": branch.get("H")}
    return "G-" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
