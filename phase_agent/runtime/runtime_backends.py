"""Wire calculation backends without preparing, submitting or collecting tasks."""

from phase_agent.runtime.runtime_config_io import _import_reference


def compose_runtime_backends(
    workflow_kwargs, settings, resolved_storage, *, reference_factory=_import_reference
):
    """Return configured dependencies, preserving caller-owned workflow arguments."""
    kwargs = dict(workflow_kwargs)
    for key in ("dispatcher", "task_runner"):
        reference = settings.get(f"{key}_factory")
        if reference:
            kwargs[key] = reference_factory(reference, f"{key}_factory")
    for key, reference in (settings.get("runtime_adapters") or {}).items():
        kwargs[key] = reference_factory(reference, f"runtime_adapters.{key}")
    manual = settings.get("manual_upload") or {}
    if manual.get("enabled"):
        if kwargs.get("task_runner") is not None:
            raise ValueError("manual_upload 与 task_runner_factory 不能同时配置")
        required = [key for key in ("batches_directory", "worker_command") if not manual.get(key)]
        if required:
            raise ValueError(f"manual_upload 缺少 {required}；不会猜测超算路径或命令")
        if manual.get("submit"):
            raise ValueError("manual_upload 只允许本地生成，submit 必须为 false")
        command = manual["worker_command"]
        if not isinstance(command, list) or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError("manual_upload.worker_command 必须是非空字符串数组")
        from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

        task_preparer = None
        if manual.get("task_preparer_factory"):
            task_preparer = reference_factory(
                manual["task_preparer_factory"], "manual_upload.task_preparer_factory"
            )
        kwargs["task_runner"] = ManualUploadBatchRunner(
            resolved_storage["upload_batches"],
            worker_command=command,
            dispatcher=kwargs.get("dispatcher"),
            stage_batch_sizes=manual.get("stage_batch_sizes"),
            stage_profiles=manual.get("stage_profiles"),
            task_preparer=task_preparer,
        )
    if kwargs.get("task_runner") is None and kwargs.get("result_collector") is None:
        # Read-only recovery for manually uploaded jobs; this never submits/prepares tasks.
        from phase_agent.tools.remote.batch_runner import RemoteBatchRunner

        kwargs["result_collector"] = RemoteBatchRunner(
            resolved_storage["upload_batches"],
            worker_command=[],
        )
    return kwargs
