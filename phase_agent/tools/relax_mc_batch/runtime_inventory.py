"""Conservative static import closure for the portable Relax+MC runtime."""

import ast
from collections import deque
from pathlib import PurePosixPath


ROOTS = (
    "phase_agent.tools.relax_mc_batch.cli",
    "phase_agent.tools.relax_mc_batch.prepare",
    "phase_agent.tools.relax_mc_batch.upload_driver",
    "phase_agent.tools.slurm.run_slurm_array_task",
    "phase_agent.tools.remote.run_mlip_batch",
    "phase_agent.tools.remote.environment_preflight",
    "phase_agent.science.mlip.slurm_executor",
)


def runtime_python_inventory(sources, extra_roots=()):
    """Return a conservative source closure; reject missing requested executors.

    ``sources`` maps POSIX paths relative to the package parent to UTF-8 bytes.
    Dynamic project module strings are collected from every source, and all
    maintenance modules are retained for the known computed maintenance import.
    Non-Python data is handled separately and is not reduced by this function.
    """
    modules = {}
    for name in sources:
        path = PurePosixPath(name)
        if not name.startswith("phase_agent/") or path.suffix != ".py":
            continue
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = name
    for root in (*ROOTS, *extra_roots):
        if root not in modules:
            raise ValueError("required_runtime_module_missing:" + root)
    pending = deque((*ROOTS, *extra_roots))
    pending.extend(m for m in modules if m.startswith("phase_agent.graphs.dialogue.maintenance."))
    # Keep the whole scientific worker namespace: backend adapters and optional
    # calculator paths can be selected by job configuration at execution time.
    pending.extend(
        m
        for m in modules
        if m.startswith(
            (
                "phase_agent.science.mc.",
                "phase_agent.science.mlip.",
                "phase_agent.science.surrogate_models.",
                "phase_agent.tools.slurm.",
            )
        )
    )
    # Include literal dynamic executor references even when the caller is loaded
    # only on a conditional branch.
    for data in sources.values():
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="strict")
            for word in text.replace('"', " ").replace("'", " ").split():
                module = word.split(":", 1)[0].strip("(),;[]{}")
                if module.startswith("phase_agent.") and module in modules:
                    pending.append(module)
    selected = set()
    while pending:
        module = pending.popleft()
        if module in selected or module not in modules:
            continue
        selected.add(module)
        name = modules[module]
        package = module if PurePosixPath(name).name == "__init__.py" else module.rpartition(".")[0]
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        tree = ast.parse(sources[name].decode("utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    bits = package.split(".")
                    prefix = ".".join(bits[: len(bits) - node.level + 1])
                    target = prefix + (("." + node.module) if node.module else "")
                else:
                    target = node.module or ""
                pending.append(target)
                pending.extend(target + "." + alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.split(":", 1)[0]
                if value.startswith("phase_agent."):
                    pending.append(value)
                # Workers are sometimes copied by filename rather than imported.
                if value.endswith(".py") and "/" not in value and "\\" not in value:
                    sibling = str(PurePosixPath(name).parent / value)
                    pending.extend(m for m, path in modules.items() if path == sibling)
    return {modules[module] for module in selected}


def configured_worker_roots(worker):
    """Resolve configured project-owned executable module names explicitly."""
    roots = []
    for value in [*(worker.get("command") or []), worker.get("action_executor")]:
        if isinstance(value, str) and value.startswith("phase_agent."):
            roots.append(value.split(":", 1)[0])
    return roots


def copy_runtime(source, destination, worker):
    """Copy the conservative code closure and non-code runtime assets."""
    import shutil
    from pathlib import Path

    source = Path(source)
    sources = {
        "phase_agent/" + p.relative_to(source).as_posix(): p.read_bytes()
        for p in source.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    selected = runtime_python_inventory(sources, extra_roots=configured_worker_roots(worker))

    def ignored(directory, names):
        result = []
        for name in names:
            path = Path(directory) / name
            if (
                name == "__pycache__"
                or name == "README.md"
                or name.endswith((".pyc", ".pyw", ".code-workspace"))
            ):
                result.append(name)
            elif path.is_file() and path.suffix == ".py":
                relative = "phase_agent/" + path.relative_to(source).as_posix()
                if relative not in selected:
                    result.append(name)
        return result

    return shutil.copytree(source, destination, ignore=ignored)
