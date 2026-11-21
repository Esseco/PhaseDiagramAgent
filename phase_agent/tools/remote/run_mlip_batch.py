"""Run every task in one manually submitted MLIP job, preserving task results."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys

from phase_agent.tools.remote.export_batch_result import export_batch_result
from phase_agent.tools.cost.runtime_observation import start_timer, runtime_observation


def _worker_environment(batch, helper_directory=None):
    env = {**os.environ, "PHASE_DEFER_RESULT_ARCHIVE": "1"}
    if helper_directory is not None:
        env["PYTHONPATH"] = str(Path(helper_directory).resolve()) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    backend = Path(batch) / "mc_backend"
    if backend.is_dir():
        env["PYTHONPATH"] = str(backend.resolve()) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    return env


def run_mlip_batch(manifest_path="manifest.json", *, executor=None):
    if not executor:
        return _run_mlip_batch(manifest_path, executor=executor)
    manifest = Path(manifest_path).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [(manifest.parent / str(row["input_path"]).replace("\\", "/")) for row in entries]
    try:
        tasks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    except (OSError, ValueError, TypeError):
        return _run_mlip_batch(manifest, executor=executor)
    pure_predict = bool(executor and tasks) and all(
        (task.get("worker_job") or {}).get("operation") == "predict"
        and (task.get("parameters") or {}).get("model_refresh_operation") == "predict"
        for task in tasks
    )
    if not pure_predict:
        return _run_mlip_batch(manifest, executor=executor)

    from phase_agent.tools.remote.run_single_task import _load, run_single_task

    with ExitStack() as stack:
        task_directory = str(paths[0].parent)
        sys.path.insert(0, task_directory)
        stack.callback(sys.path.remove, task_directory)
        stack.enter_context(_reuse_calculators())
        previous = os.environ.get("PHASE_DEFER_RESULT_ARCHIVE")
        os.environ["PHASE_DEFER_RESULT_ARCHIVE"] = "1"

        def restore_environment():
            if previous is None:
                os.environ.pop("PHASE_DEFER_RESULT_ARCHIVE", None)
            else:
                os.environ["PHASE_DEFER_RESULT_ARCHIVE"] = previous

        stack.callback(restore_environment)
        execute = _load(executor)
        return _run_mlip_batch(manifest, executor=executor, inprocess=(run_single_task, execute))


@contextmanager
def _reuse_calculators():
    import inspect

    from phase_agent.science.mlip import model_interface

    original = model_interface.create_calculator
    cache = {}

    def cached(model, *, model_path=None):
        key = json.dumps([model, str(model_path)], sort_keys=True, default=str)
        if key not in cache:
            cache[key] = original(model, model_path=model_path)
        return cache[key]

    try:
        from Process_MLIP import calculators
    except ModuleNotFoundError as error:
        if error.name != "Process_MLIP":
            raise
        calculators = None
    if calculators is not None:
        mace_factory = calculators.create_mace_calculator
        signature = inspect.signature(mace_factory)
        mace_cache = {}

        def cached_mace(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            bound.arguments["model_path"] = str(bound.arguments["model_path"])
            key = json.dumps(bound.arguments, sort_keys=True, default=str)
            if key not in mace_cache:
                mace_cache[key] = mace_factory(*bound.args, **bound.kwargs)
            return mace_cache[key]

        calculators.create_mace_calculator = cached_mace
    model_interface.create_calculator = cached
    try:
        yield
    finally:
        model_interface.create_calculator = original
        if calculators is not None:
            calculators.create_mace_calculator = mace_factory


def run_slurm_predict_batch(manifest_path, executor):
    from phase_agent.tools.slurm.run_slurm_array_task import load_executor, run_slurm_array_task

    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    failures = []
    with _reuse_calculators():
        execute = load_executor(executor)
        for entry in entries:
            result = run_slurm_array_task(manifest_path, entry["array_index"], executor=execute)
            if result.get("status") != "completed":
                failures.append(entry["task_id"])
    return failures


def _run_mlip_batch(manifest_path="manifest.json", *, executor=None, inprocess=None):
    manifest = Path(manifest_path).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    batch = manifest.parent
    first_directory = (
        (batch / str(entries[0]["input_path"]).replace("\\", "/")).parent if entries else None
    )
    shared_helpers = (
        first_directory
        if first_directory is not None
        and (first_directory / "_handoff_refresh_executor.py").is_file()
        else None
    )
    results_directory = (
        batch.parent.parent if batch.parent.name == "inputs" else batch.parent
    ) / "results"
    results_directory.mkdir(exist_ok=True)
    exported = []
    failures = []
    timer = start_timer()
    executed = 0
    for entry in entries:
        # Batch manifests may have been generated on Windows and uploaded as-is.
        # Treat both slash styles as portable separators on the Linux worker.
        input_path = str(entry["input_path"]).replace("\\", "/")
        task_dir = (manifest.parent / input_path).parent
        marker = task_dir / "task.finished.json"
        if marker.is_file() and (task_dir / "result.json").is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed":
                    exported.append(
                        export_batch_result(task_dir, results_directory, create_archive=False)
                    )
                    continue
            except (OSError, ValueError) as error:
                print(f"已有结果汇集失败 {entry['task_id']}: {error}", file=sys.stderr)
                failures.append(entry["task_id"])
                continue
        with (
            (task_dir / "task.stdout.log").open("a", encoding="utf-8") as stdout,
            (task_dir / "task.stderr.log").open("a", encoding="utf-8") as stderr,
        ):
            command = (
                [
                    sys.executable,
                    "-m",
                    "phase_agent.tools.remote.run_single_task",
                    "--task",
                    "task.json",
                    "--executor",
                    executor,
                ]
                if executor
                else [sys.executable, "run_mlip_task.py"]
            )
            if inprocess is None:
                outcome = subprocess.run(
                    command,
                    cwd=task_dir,
                    env=_worker_environment(batch, shared_helpers),
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                failed = outcome.returncode != 0
            else:
                run_single_task, execute = inprocess
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = run_single_task(task_dir / "task.json", executor=execute)
                failed = result.get("status") != "completed"
            executed += 1
        if failed or not marker.is_file():
            failures.append(entry["task_id"])
        if marker.is_file() and (task_dir / "result.json").is_file():
            try:
                exported.append(
                    export_batch_result(task_dir, results_directory, create_archive=False)
                )
            except (OSError, ValueError) as error:
                print(f"结果汇集失败 {entry['task_id']}: {error}", file=sys.stderr)
                if entry["task_id"] not in failures:
                    failures.append(entry["task_id"])
    timing = runtime_observation(
        timer, {"batch_id": entries[0].get("batch_id") if entries else None}, {}
    )
    timing.update(
        {
            "allocation_basis": "whole_batch_do_not_add_to_task_slices",
            "executed_tasks": executed,
            "manifest_tasks": len(entries),
        }
    )
    batch_id = manifest.parent.name
    try:
        (results_directory / f"{batch_id}.runtime.json").write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as error:
        print(f"批次统计未保存（计算结果保持不变）: {error}", file=sys.stderr)
    from phase_agent.tools.remote.transfer_bundle import result_bundle

    result_bundle(results_directory, exported, batch_id)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--executor", default=None)
    parser.add_argument("--slurm-predict", action="store_true")
    args = parser.parse_args(argv)
    if args.slurm_predict and not args.executor:
        parser.error("--slurm-predict requires --executor")
    failures = (
        run_slurm_predict_batch(args.manifest, args.executor)
        if args.slurm_predict
        else run_mlip_batch(args.manifest, executor=args.executor)
    )
    if failures:
        print("未完成的 task: " + ", ".join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
