from unittest.mock import Mock, patch
from phase_agent.runtime.local_project_launcher import _stop_owned_process, stop_project_service


def test_psutil_style_process_can_stop_without_poll():
    process = Mock(spec=["pid", "is_running", "terminate", "wait", "kill"])
    process.pid = 12
    process.is_running.side_effect = [True, False]
    with (
        patch("phase_agent.runtime.studio_run_lifecycle.interrupt_owned_studio"),
        patch("psutil.Process") as factory,
    ):
        factory.return_value.children.return_value = []
        assert _stop_owned_process(process)
    process.terminate.assert_called_once()


def test_explicit_stop_verifies_selected_project(tmp_path):
    path = tmp_path / "runtime.json"
    process = Mock()
    process.cmdline.return_value = [
        "python",
        "-m",
        "phase_agent.runtime.studio_service",
        "--runtime-config",
        str(path),
    ]
    with (
        patch("phase_agent.runtime.project_service.read_service", return_value={"pid": 12}),
        patch("psutil.Process", return_value=process),
        patch(
            "phase_agent.runtime.local_project_launcher._stop_owned_process", return_value=True
        ) as stop,
        patch("phase_agent.runtime.project_service.write_service") as write,
    ):
        assert stop_project_service(path)
        stop.assert_called_once_with(process)
        assert [call.args[1]["status"] for call in write.call_args_list] == ["stopping", "stopped"]
