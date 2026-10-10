"""Separate editable startup facts from later search policy overrides."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

INITIAL_ROOTS = {
    "python_environments",
    "system",
    "mlip",
    "dft",
    "supercomputer",
    "storage",
    "calculation",
    "frozen_parameters",
    "initial_mlip_health_check",
}
RUN_FORMAT = "phase-search-run-overrides-v1"


def read_document(path):
    from phase_agent.configuration.session.load_editable_config_json import (
        _strip_jsonc_comments,
        _reject_nonstandard_constant,
    )

    return json.loads(
        _strip_jsonc_comments(Path(path).read_text(encoding="utf-8-sig")),
        parse_constant=_reject_nonstandard_constant,
    )


def run_config_path(source, document):
    value = document.get("run_config_file")
    if not isinstance(value, str) or not value:
        raise ValueError("初始配置缺少run_config_file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("运行配置必须是同一配置目录下的相对JSON路径")
    target = (Path(source).parent / relative).resolve()
    if not target.is_relative_to(Path(source).parent.resolve()) or target == Path(source).resolve():
        raise ValueError("运行配置路径无效")
    return target


def combine_project_documents(document, source, *, run_document=None):
    if "run_config_file" not in document:
        return deepcopy(document)
    run = (
        run_document
        if run_document is not None
        else read_document(run_config_path(source, document))
    )
    if (
        not isinstance(run, dict)
        or run.get("_format") != RUN_FORMAT
        or not isinstance(run.get("config"), dict)
    ):
        raise ValueError("运行配置格式无效")
    if set(run["config"]) & INITIAL_ROOTS:
        raise ValueError("环境、边界、结构和计算设置请修改初始配置，不能放入运行配置")
    from phase_agent.configuration.session.load_editable_config_json import _check_secrets

    _check_secrets(run["config"])
    combined = deepcopy(document)
    combined.pop("run_config_file")
    # Root ownership is disjoint, so stale initial policy values cannot hide changes.
    for section in ("config", "overrides"):
        misplaced = set(combined.get(section, {})) - INITIAL_ROOTS
        if misplaced:
            raise ValueError("初始配置中混入运行参数：" + ", ".join(sorted(misplaced)))
    combined.setdefault("overrides", {}).update(deepcopy(run["config"]))
    return combined


def editable_project_hash(path):
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw)
    document = read_document(source)
    if isinstance(document, dict) and "run_config_file" in document:
        runtime = run_config_path(source, document)
        digest.update(b"\0run_config\0")
        digest.update(runtime.read_bytes())
    return digest.hexdigest()


def split_project_file(path, *, baseline_config=None):
    """Explicit migration; never overwrite an existing independent run config."""
    source = Path(path)
    document = read_document(source)
    if "run_config_file" in document:
        combine_project_documents(document, source)
        return run_config_path(source, document)
    from phase_agent.configuration.session.project_config_json import (
        FORMAT_ID,
        expand_project_config,
        profile_digest,
        _document_for_effective_config,
    )

    if document.get("_format") != FORMAT_ID:
        raise ValueError("只支持拆分project-v2配置；旧长草稿保持原样")
    runtime = source.with_name("run_config.project.json")
    if runtime.exists():
        raise FileExistsError("运行配置已存在，未覆盖：" + str(runtime))
    expanded = expand_project_config(document, source=source, baseline_config=baseline_config)
    if document.get("profile_digest") != profile_digest():
        document = _document_for_effective_config(document, expanded)
    initial = deepcopy(document)
    run_values = {}
    for section in ("config", "overrides"):
        current = initial.setdefault(section, {})
        for key in list(current):
            if key not in INITIAL_ROOTS:
                value = current.pop(key)
                if (
                    key in run_values
                    and isinstance(value, dict)
                    and isinstance(run_values[key], dict)
                ):
                    run_values[key] = merge_dict(run_values[key], value)
                else:
                    run_values[key] = value
    # Make environment and scheduler facts visible in the initial front section.
    for key in ("python_environments", "supercomputer"):
        if key in initial["overrides"]:
            initial["config"][key] = initial["overrides"].pop(key)
    for key in ("python_environments", "supercomputer"):
        initial["config"].setdefault(key, deepcopy(expanded[key]))
    initial["_scope"] = "initial"
    initial["run_config_file"] = runtime.name
    initial["_help"] = (
        "初始配置：本地/超算环境、边界、母结构、模型与计算设置。运行预算、采样、MC和微调参数见run_config_file。两份均可手改，保存后告诉Agent读取配置JSON；也可直接告诉Agent要改什么。"
    )
    run = {
        "_format": RUN_FORMAT,
        "_scope": "run",
        "_help": "运行中参数：预算、采样、MC、微调和验证。未填写的项继承锁定模板；后续阶段缺项到使用时再询问。不要在此覆盖初始环境、边界或结构。",
        "config": run_values,
    }
    # Check before publishing either file. Keep old bytes as an immutable migration backup.
    combined = combine_project_documents(initial, source, run_document=run)
    expand_project_config(combined, source=source)
    backup = source.with_name(source.name + ".before-split.bak")
    if not backup.exists():
        with backup.open("xb") as handle:
            handle.write(source.read_bytes())
    with runtime.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    write_document(source, initial)
    return runtime


def merge_dict(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = (
            merge_dict(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def write_document(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalized_patch_value(dotted_path, value, source):
    expected = deepcopy(value)

    def local_path(raw):
        path = Path(raw).expanduser()
        return str(path if path.is_absolute() else (Path(source).parent / path).resolve())

    if (
        dotted_path in {"remote_training_validation.data_path", "remote_training_job_template"}
        and expected
    ):
        expected = local_path(expected)
    elif (
        dotted_path == "remote_training_validation"
        and isinstance(expected, dict)
        and expected.get("data_path")
    ):
        expected["data_path"] = local_path(expected["data_path"])
    return expected


def patch_split_project(path, patch, *, expected_hash=None, baseline_config=None):
    source = Path(path)
    identity = editable_project_hash(source)
    if expected_hash is not None and identity != expected_hash:
        raise ValueError("初始或运行配置已被其他编辑修改，请重新读取")
    initial = read_document(source)
    runtime = run_config_path(source, initial)
    run = read_document(runtime)
    changes = []
    touched = set()
    for dotted_path, value in sorted(patch.items()):
        parts = dotted_path.split(".")
        if any(not part or part.startswith("_") for part in parts):
            raise ValueError("无效配置字段：" + dotted_path)
        initial_field = parts[0] in INITIAL_ROOTS
        document = initial if initial_field else run
        section = "config"
        # Existing advanced initial overrides take precedence over front fields.
        cursor = document.get("overrides", {})
        overridden = initial_field
        for part in parts:
            if not isinstance(cursor, dict) or part not in cursor:
                overridden = False
                break
            cursor = cursor[part]
        if overridden:
            section = "overrides"
        cursor = document.setdefault(section, {})
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError("字段不能向下展开：" + dotted_path)
        old = deepcopy(cursor.get(parts[-1]))
        if old != value:
            cursor[parts[-1]] = deepcopy(value)
            changes.append(
                {
                    "path": dotted_path,
                    "old": old,
                    "new": deepcopy(value),
                    "file": str(source if initial_field else runtime),
                }
            )
            touched.add("initial" if initial_field else "run")
    from phase_agent.configuration.session.project_config_json import (
        expand_project_config,
        profile_digest,
    )

    combined = combine_project_documents(initial, source, run_document=run)
    effective = expand_project_config(combined, source=source, baseline_config=baseline_config)
    # Expand first: stale templates require a usable saved full baseline.
    migrated = initial.get("profile_digest") != profile_digest()
    if migrated:
        from phase_agent.configuration.session.project_config_json import (
            _document_for_effective_config,
        )

        rebased = _document_for_effective_config(combined, effective)
        initial = deepcopy(initial)
        initial["profile_digest"] = rebased["profile_digest"]
        run = deepcopy(run)
        run_values = {}
        for section in ("config", "overrides"):
            initial[section] = {}
            for key, value in rebased.get(section, {}).items():
                if key in INITIAL_ROOTS:
                    initial[section][key] = deepcopy(value)
                elif (
                    key in run_values
                    and isinstance(value, dict)
                    and isinstance(run_values[key], dict)
                ):
                    run_values[key] = merge_dict(run_values[key], value)
                else:
                    run_values[key] = deepcopy(value)
        run["config"] = run_values
        checked = expand_project_config(
            combine_project_documents(initial, source, run_document=run), source=source
        )
        if checked != effective:
            raise ValueError("拆分模板迁移后的有效配置不一致，文件未修改")
        touched.update({"initial", "run"})
    for dotted_path, value in patch.items():
        actual = effective
        for part in dotted_path.split("."):
            actual = actual[part]
        if actual != normalized_patch_value(dotted_path, value, source):
            raise ValueError("写入后生效值不一致：" + dotted_path)
    if editable_project_hash(source) != identity:
        raise ValueError("配置在写入前发生变化，请重新读取")
    if migrated:
        from phase_agent.configuration.session.project_config_json import _profile_migration_backup

        for target in (source, runtime):
            backup = _profile_migration_backup(target)
            with backup.open("xb") as handle:
                handle.write(target.read_bytes())
    # A crash between replacements changes combined identity and invalidates review.
    if "run" in touched:
        write_document(runtime, run)
    if "initial" in touched:
        write_document(source, initial)
    return changes
