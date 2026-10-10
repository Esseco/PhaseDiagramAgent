"""export operation adapter; invoked as a named dialogue graph node."""

from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.graphs.dialogue.commands import _phase_csv_request


def handle(self, user_message, state, messages, conversation_id):
    if _phase_csv_request(user_message):
        from phase_agent.analysis.phase.export_current_phase_diagram import (
            export_current_phase_diagram,
        )

        method = (
            "combined"
            if any(word in user_message.lower() for word in ("综合", "总相图", "校正", "combined"))
            else "dft"
            if "dft" in user_message.lower()
            else "mlip"
        )
        try:
            before = ((state.get("phase_diagrams") or {}).get(method) or {}).get("csv_path")
            path, count, version = export_current_phase_diagram(
                state, directory=self.workflow_kwargs.get("phase_diagram_directory"), method=method
            )
            if str(path) != before:
                write_json(self.state_path, state)
        except (OSError, ValueError) as error:
            return f"当前相图 CSV 未导出：{error}。未推进搜索或修改任务。"
        return f"当前 {method.upper()} 相图 CSV（版本 {version}，{count} 个结构）：`{path}`"
    return None
