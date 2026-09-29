"""Short layered-oxide project config backed by a pinned full default profile."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.session.resolve_workspace_paths import default_workspace_storage


FORMAT_ID = "phase-search-project-v2"
PROFILE_ID = "layered-oxide-v1"


def profile_defaults() -> dict:
    """The only source of inherited scientific defaults for this profile."""
    return default_layered_search_config()


def profile_digest() -> str:
    defaults = profile_defaults()
    # This additive problem-definition field must not invalidate existing drafts.
    defaults.get("system", {}).pop("configuration_space", None)
    # Remote path is an additive setup default; preserve existing pinned templates.
    defaults.get("mlip", {})["model_path"] = None
    raw = json.dumps(defaults, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def create_project_config_json(path, config: dict | None = None, *, bootstrap_hints=None,
                               workspace_defaults=None) -> bool:
    """Create a compact editable file once, without overwriting user edits."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    defaults = profile_defaults()
    source = deepcopy(config or defaults)
    source.setdefault("storage", deepcopy(workspace_defaults or default_workspace_storage(target.parent)))
    system = source.setdefault("system", {})
    system.setdefault("phase_reference_directory", (bootstrap_hints or {}).get(
        "local_initial_structure_directory", ""))
    system.setdefault("boundary", {
        "P": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3", "OP2"]},
        "H": {}, "TM_ratio": deepcopy(system.get("constraints", {}).get("TM_ratio", {})),
    })
    if system.get("system_id") == "layered_na_tm_oxide" and not system["boundary"].get("H"):
        from scientific_layer.structures.enumerate_layered_oxide_supercells import LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS
        system.setdefault("H_generation", {
            "enabled": True, "size_min": 4, "size_max": 16, "size_step": 2,
            "min_distance_angstrom": 2.0,
            "recommended_containment_matrices": [[list(row) for row in matrix]
                                                 for matrix in LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS],
            "selected_recommendation_indices": [0], "additional_containment_matrices": [],
        })
    project = {
        "system": {key: deepcopy(system[key]) for key in (
            "system_id", "configuration_space", "phase_reference_directory", "phase_references", "boundary", "H_generation"
        ) if key in system},
        "mlip": {"model_path": source.get("mlip", {}).get("model_path")},
        "budgets": {"total_relative_cost": source.get("budgets", {}).get("total_relative_cost")},
        "dft": {key: deepcopy(source.get("dft", {}).get(key)) for key in
                ("parameter_source", "parameters")},
        "convergence": deepcopy(source.get("convergence", {})),
        "run": {key: source.get("run", {}).get(key) for key in
                ("initial_states_per_branch", "seed")},
        "storage": deepcopy(source["storage"]),
    }
    # Preserve every non-default value not already represented in the short front section.
    remaining = _difference(defaults, source)
    _remove_covered(remaining, project)
    document = {
        "_format": FORMAT_ID, "profile": PROFILE_ID, "profile_digest": profile_digest(),
        "_help": "只改 config 中需要决定的项；其余继承固定层氧模板。高级参数写 overrides。不要写 API Key。修改后告诉 Agent：读取配置 JSON。",
        "config": project, "overrides": remaining,
    }
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("// 层氧短配置；profile_digest 锁定默认模板版本。旧长草稿不会被覆盖。\n")
            stream.write(json.dumps(document, ensure_ascii=False, indent=2))
            stream.write("\n")
    except FileExistsError:
        return False
    return True


def expand_project_config(document: dict, *, source: Path, baseline_config: dict | None = None) -> dict:
    """Expand a sparse project document into the existing complete config schema."""
    if document.get("profile") != PROFILE_ID:
        raise ValueError(f"配置体系 {document.get('profile')!r} 与当前 {PROFILE_ID!r} 不兼容。")
    profile_changed = document.get("profile_digest") != profile_digest()
    if profile_changed:
        if not _usable_baseline(baseline_config):
            raise ValueError(
                "默认模板版本已更新，且没有可用的已保存配置作为迁移基线；"
                "保留原文件并先恢复其 config_session，再读取或修改。"
            )
        # Rebase on the saved full configuration so a changed default never
        # silently replaces values inherited by an older short config.
        defaults = _fill_missing_defaults(baseline_config, profile_defaults())
    else:
        defaults = profile_defaults()
    project = document.get("config")
    overrides = document.get("overrides", {})
    if not isinstance(project, dict) or not isinstance(overrides, dict):
        raise ValueError("config 和 overrides 必须是 JSON 对象。")
    from config_layer.session.load_editable_config_json import _check_secrets
    _check_secrets(project)
    _check_secrets(overrides)
    allowed_optional = {"system": {"boundary", "H_generation", "phase_reference_directory"},
                        "run": {"generation_options"}, "root": {"storage"}}
    merged = _merge(defaults, project, "config", allowed_optional)
    merged = _merge(merged, overrides, "config", allowed_optional)
    system = merged.get("system", {})
    if ((system.get("configuration_space") or {}).get("roles") or {}).get("T") == "fixed":
        if (system.get("branch_schema") or {}).get("fields") == ["P", "H", "x", "T"]:
            system["branch_schema"]["fields"] = ["P", "H", "x"]
    boundary = system.get("boundary")
    if isinstance(boundary, dict):
        from scientific_layer.structures.boundary_utils import allowed_phases
        system["constraints"]["phases"] = sorted(allowed_phases(boundary.get("P", [])))
        system["constraints"]["TM_ratio"] = deepcopy(boundary.get("TM_ratio", {}))
    _validate_h_generation(system.get("H_generation"))
    if "storage" not in merged:
        merged["storage"] = default_workspace_storage(source.parent)
    from config_layer.session.resolve_workspace_paths import resolve_workspace_paths
    resolve_workspace_paths(merged, base_directory=source.parent)
    return merged


def write_project_config_patch(path, patch: dict, *, expected_hash: str | None = None,
                               baseline_config: dict | None = None) -> list[dict]:
    """Apply an explicitly authorized, non-secret patch to the short JSON file."""
    from config_layer.session.load_editable_config_json import _strip_jsonc_comments
    target = Path(path)
    source = target.read_bytes()
    if expected_hash is not None and hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError("配置文件已被其他编辑修改，请重新读取")
    document = json.loads(_strip_jsonc_comments(source.decode("utf-8")))
    if document.get("_format") != FORMAT_ID:
        raise ValueError("仅支持写入 phase-search-project-v2 短配置")
    profile_changed = document.get("profile_digest") != profile_digest()
    if profile_changed:
        effective = expand_project_config(
            document, source=target, baseline_config=baseline_config,
        )
        backup = _profile_migration_backup(target)
        with backup.open("xb") as stream:
            stream.write(source)
        document = _document_for_effective_config(document, effective)
        document["_migration"] = {
            "from_profile_digest": json.loads(_strip_jsonc_comments(source.decode("utf-8"))).get("profile_digest"),
            "to_profile_digest": profile_digest(),
            "baseline": "saved_config_session",
            "backup_file": backup.name,
        }
    changes = []
    for dotted_path, value in sorted(patch.items()):
        parts = dotted_path.split(".")
        # `overrides` is merged last. Editing `config` while the same leaf is
        # overridden would report success without changing the effective value.
        override_cursor = document["overrides"]
        overridden = True
        for part in parts:
            if not isinstance(override_cursor, dict) or part not in override_cursor:
                overridden = False
                break
            override_cursor = override_cursor[part]
        section = (document["overrides"] if overridden
                   or parts[0] not in document.get("config", {}) else document["config"])
        cursor = section
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                cursor[part] = {}
            elif not isinstance(current, dict):
                raise ValueError(f"配置路径不能向下展开：{dotted_path}")
            cursor = cursor[part]
        old = deepcopy(cursor.get(parts[-1]))
        if old == value:
            continue
        cursor[parts[-1]] = deepcopy(value)
        changes.append({"path": dotted_path, "old": old, "new": deepcopy(value)})
    # Check the resulting effective config before atomically replacing the file.
    effective = expand_project_config(document, source=target)
    for dotted_path, value in patch.items():
        actual = effective
        for part in dotted_path.split("."):
            actual = actual[part]
        if actual != value:
            raise ValueError(f"写入后生效值不一致：{dotted_path}")
    if not changes and not profile_changed:
        return []
    temporary = target.with_name(target.name + ".tmp")
    header = "// 层氧短配置；profile_digest 锁定默认模板版本。旧长草稿不会被覆盖。\n"
    temporary.write_text(header + json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(target)
    return changes


def _usable_baseline(config):
    return isinstance(config, dict) and all(
        isinstance(config.get(key), dict) for key in ("system", "budgets", "run")
    )


def _fill_missing_defaults(baseline, latest):
    """Keep saved values and add only fields introduced by the newer profile."""
    result = deepcopy(baseline)
    for key, value in latest.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _fill_missing_defaults(result[key], value)
    return result


def _document_for_effective_config(original, config):
    system = config.get("system") or {}
    project = {
        "system": {key: deepcopy(system[key]) for key in (
            "system_id", "configuration_space", "phase_reference_directory",
            "phase_references", "boundary", "H_generation",
        ) if key in system},
        "mlip": {"model_path": deepcopy((config.get("mlip") or {}).get("model_path"))},
        "budgets": {"total_relative_cost": deepcopy(
            (config.get("budgets") or {}).get("total_relative_cost"))},
        "dft": {key: deepcopy((config.get("dft") or {}).get(key))
                for key in ("parameter_source", "parameters")},
        "convergence": deepcopy(config.get("convergence") or {}),
        "run": {key: deepcopy((config.get("run") or {}).get(key))
                for key in ("initial_states_per_branch", "seed")
                if key in (config.get("run") or {})},
        "storage": deepcopy(config.get("storage") or {}),
    }
    remaining = _difference(profile_defaults(), config)
    _remove_covered(remaining, project)
    migrated = deepcopy(original)
    migrated.update({
        "_format": FORMAT_ID,
        "profile": PROFILE_ID,
        "profile_digest": profile_digest(),
        "config": project,
        "overrides": remaining,
    })
    return migrated


def _profile_migration_backup(target):
    candidate = target.with_name(target.name + ".pre-profile-migration.bak")
    index = 1
    while candidate.exists():
        candidate = target.with_name(target.name + f".pre-profile-migration-{index}.bak")
        index += 1
    return candidate


def _merge(base: dict, patch: dict, path: str, optional: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if not isinstance(key, str) or key.startswith("_") or "." in key:
            raise ValueError(f"字段名无效：{path}.{key}")
        if key not in result and key not in optional.get("root" if path == "config" else path.removeprefix("config."), set()):
            raise ValueError(f"未知配置字段：{path}.{key}")
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            # These dictionaries have user-defined keys; replace as one unit.
            if f"{path}.{key}" in {"config.system.boundary", "config.system.phase_references",
                                   "config.dft.parameters", "config.storage.paths",
                                   "config.round_strategy.rule_default.generation_quotas"}:
                result[key] = deepcopy(value) if f"{path}.{key}" != "config.system.boundary" else _merge_boundary(result[key], value)
            else:
                result[key] = _merge(result[key], value, f"{path}.{key}", optional)
        elif key in result and result[key] is not None and value is not None and not _compatible(result[key], value):
            raise ValueError(f"{path}.{key} 类型与默认模板不一致。")
        else:
            result[key] = deepcopy(value)
    return result


def _merge_boundary(base, patch):
    if not isinstance(patch, dict) or set(patch) - {"P", "H", "TM_ratio"}:
        raise ValueError("system.boundary 仅支持 P、H、TM_ratio。")
    return {**deepcopy(base), **deepcopy(patch)}


def _compatible(old, new):
    if isinstance(old, bool):
        return isinstance(new, bool)
    if isinstance(old, (int, float)):
        return isinstance(new, (int, float)) and not isinstance(new, bool)
    return isinstance(new, type(old))


def _validate_h_generation(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("system.H_generation 必须是 JSON 对象。")
    allowed = {"enabled", "size_min", "size_max", "size_step", "min_distance_angstrom",
               "recommended_containment_matrices", "selected_recommendation_indices",
               "additional_containment_matrices", "first_round_max_det_H"}
    if set(value) - allowed:
        raise ValueError("system.H_generation 含未知字段：" + ", ".join(sorted(set(value) - allowed)))
    for key in ("size_min", "size_max", "size_step", "first_round_max_det_H"):
        if key in value and (not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0):
            raise ValueError(f"system.H_generation.{key} 必须是正整数。")
    if ("size_min" in value and "size_max" in value
            and value["size_min"] > value["size_max"]):
        raise ValueError("H_generation.size_min 不能大于 size_max。")
    if ("first_round_max_det_H" in value and "size_max" in value
            and value["first_round_max_det_H"] > value["size_max"]):
        raise ValueError("H_generation.first_round_max_det_H 不能大于 size_max。")
    matrices = value.get("recommended_containment_matrices", [])
    if not isinstance(matrices, list):
        raise ValueError("recommended_containment_matrices 必须是矩阵列表。")
    for name, collection in (("recommended_containment_matrices", matrices),
                             ("additional_containment_matrices", value.get("additional_containment_matrices", []))):
        if not isinstance(collection, list):
            raise ValueError(f"{name} 必须是矩阵列表。")
        for index, matrix in enumerate(collection):
            if not isinstance(matrix, list) or len(matrix) not in (2, 3) or any(
                    not isinstance(row, list) or len(row) != len(matrix)
                    or any(not isinstance(number, int) for number in row) for row in matrix):
                raise ValueError(f"system.H_generation.{name}[{index}] 必须是 2×2 或 3×3 整数矩阵。")
    selected = value.get("selected_recommendation_indices", [])
    if not isinstance(selected, list) or any(not isinstance(index, int) or isinstance(index, bool)
                                             or index < 0 or index >= len(matrices) for index in selected):
        raise ValueError("selected_recommendation_indices 必须是推荐矩阵范围内的整数索引列表。")


def _difference(base, current):
    result = {}
    for key, value in current.items():
        if key not in base:
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(base[key], dict):
            child = _difference(base[key], value)
            if child:
                result[key] = child
        elif value != base[key]:
            result[key] = deepcopy(value)
    return result


def _remove_covered(changes, covered):
    for key, value in covered.items():
        if key not in changes:
            continue
        if isinstance(value, dict) and isinstance(changes[key], dict):
            _remove_covered(changes[key], value)
            if not changes[key]:
                changes.pop(key)
        else:
            changes.pop(key)
