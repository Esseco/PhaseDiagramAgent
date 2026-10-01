import pytest

from execution_layer.remote.batch_runner import RemoteBatchRunner
from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
from run.runtime_backends import compose_runtime_backends


def test_default_backend_is_only_a_collector_without_creating_files(tmp_path):
    original = {"state_path": "state.json"}
    output = compose_runtime_backends(original, {}, {"upload_batches": tmp_path / "upload"})
    assert isinstance(output["result_collector"], RemoteBatchRunner)
    assert original == {"state_path": "state.json"}
    assert not (tmp_path / "upload").exists()


def test_manual_backend_wires_without_preparing_tasks(tmp_path):
    settings = {"manual_upload": {"enabled": True, "batches_directory": "upload",
                                 "worker_command": ["worker"], "submit": False}}
    output = compose_runtime_backends({}, settings, {"upload_batches": tmp_path / "upload"})
    assert isinstance(output["task_runner"], ManualUploadBatchRunner)
    assert "result_collector" not in output
    assert not (tmp_path / "upload").exists()


@pytest.mark.parametrize("change,reason", [
    ({"submit": True}, "只允许本地生成"),
    ({"worker_command": "worker"}, "字符串数组"),
    ({"worker_command": []}, "缺少"),
])
def test_invalid_manual_settings_are_blocked(tmp_path, change, reason):
    settings = {"manual_upload": {"enabled": True, "batches_directory": "upload",
                                 "worker_command": ["worker"], **change}}
    with pytest.raises(ValueError, match=reason):
        compose_runtime_backends({}, settings, {"upload_batches": tmp_path / "upload"})
    assert not (tmp_path / "upload").exists()


def test_manual_and_configured_runner_conflict(tmp_path):
    with pytest.raises(ValueError, match="不能同时配置"):
        compose_runtime_backends({"task_runner": object()}, {"manual_upload": {"enabled": True}},
                                 {"upload_batches": tmp_path})


def test_explicit_collector_and_adapter_override_order_are_preserved(tmp_path):
    runner, collector = object(), object()
    objects = {"runner": runner, "collector": collector}
    settings = {"task_runner_factory": "runner",
                "runtime_adapters": {"task_runner": "collector", "result_collector": "collector"}}
    output = compose_runtime_backends({}, settings, {"upload_batches": tmp_path},
                                     reference_factory=lambda ref, label: objects[ref])
    assert output["task_runner"] is collector
    assert output["result_collector"] is collector
