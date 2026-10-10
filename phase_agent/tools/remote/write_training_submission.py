"""Render a manual training submission using the existing GPU template."""

from pathlib import Path
import shlex
import re

from phase_agent.tools.remote.manual_upload_runner import _render_job_name
from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script
from phase_agent.configuration.schema.python_environments import python_environment


def training_environment(config):
    value = (config.get("python_environments") or {}).get("remote_mlip")
    if value:
        return python_environment(config, "remote", mlip=True), "confirmed_config"
    # Legacy runs may lack host fields; reuse only the actual remote template declaration.
    script = Path(__file__).with_name("mlip_gpu_template.sh").read_text(encoding="utf-8")
    names = re.findall(r"^conda activate ([A-Za-z0-9_.-]+)\s*$", script, re.MULTILINE)
    if len(names) != 1:
        raise ValueError("remote_mlip缺失且远端GPU模板没有唯一明确环境，请确认超算环境")
    return names[0], "remote_gpu_template"


def write_training_submission(directory, environment, *, template=None):
    directory = Path(directory)
    script = (
        Path(template) if template else Path(__file__).with_name("mlip_gpu_template.sh")
    ).read_text(encoding="utf-8")
    command = "python3 run_mlip_task.py"
    if script.count(command) != 1:
        raise ValueError("训练GPU模板需恰有一处python3 run_mlip_task.py入口")
    if "conda activate mace" not in script:
        raise ValueError("训练GPU模板缺少可替换的mace环境入口")
    activation = "" if environment == "current" else f"conda activate {shlex.quote(environment)}"
    script = script.replace("conda activate mace", activation).replace(
        command, 'cd "${SLURM_SUBMIT_DIR:-.}"\nbash run_training.sh'
    )
    write_unix_shell_script(directory / "GPU.sh", _render_job_name(script, "mlip-finetune"))
    (directory / "UPLOAD_AND_SUBMIT.md").write_text(
        "上传完整训练轮目录（含inputs），在inputs内执行 sbatch GPU.sh，仅提交一次；results位于inputs的同级。\n"
        "脚本顺序训练4个committee成员和5个K折评估模型；主模型复用com_1，不重复训练。使用现有MLIP GPU资源模板。\n"
        "确认远端原模型文件存在且mace环境支持foundation_head等参数。\n"
        "正式committee全部数据参与训练，com_1同时为主模型；同文件validation仅训练监视，不是泛化误差。\n"
        "完成后只下载results中的K折指标/逐点能量与受力CSV、models.json和完成标记；不需要模型文件。\n"
        "模型保留在超算com_*训练目录；models.json记录绝对远端路径、大小和SHA256，后续任务在超算读取。不要移动或删除这些模型。\n"
        "泛化误差只参考main_cv_*留出数据；模型激活仍需独立验证与审批。\n",
        encoding="utf-8",
    )
