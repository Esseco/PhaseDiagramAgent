"""从现有 branch 台账建立固定合法候选池。"""

from copy import deepcopy


def build_bohb_candidate_pool(manager, *, branch_ids=None, feature_builder=None, group_builder=None, region_builder=None) -> list[dict]:
    identifiers = branch_ids or sorted(manager.data["branches"])
    candidates = []
    for branch_id in identifiers:
        branch = deepcopy(manager.data["branches"][branch_id])
        branch["branch_id"] = branch_id
        if feature_builder:
            branch["bohb_features"] = feature_builder(branch)
        if group_builder:
            branch["composition_group"] = group_builder(branch)
        else:
            branch["composition_group"] = branch.get("x", str(branch.get("composition")))
        if region_builder:
            region = region_builder(branch)
            if isinstance(region, (list, tuple, set)):
                branch["region_ids"] = list(region)
            else:
                branch["region_id"] = region
        candidates.append(branch)
    return candidates
