"""Render a manual training submission using the existing GPU template."""

from pathlib import Path
import json
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
    script = script.replace("conda activate mace", activation)
    plan = json.loads((directory / "training_plan.json").read_text(encoding="utf-8"))
    execution = plan["execution"]
    count = len(execution["training_models"])
    concurrency = min(count, execution["max_parallel_training_jobs"])
    worker = script.replace(command, 'cd "${SLURM_SUBMIT_DIR:-.}"\nbash run_training.sh')
    worker = worker.replace("#SBATCH --output=out", "#SBATCH --output=train-%A_%a.out").replace(
        "#SBATCH --error=err", "#SBATCH --error=train-%A_%a.err"
    )
    worker = worker.replace(
        "\n#SBATCH", f"\n#SBATCH --array=0-{count - 1}%{concurrency}\n#SBATCH", 1
    )
    write_unix_shell_script(directory / "GPU.sh", _render_job_name(worker, "mlip-finetune"))
    collector = (
        script.replace(command, 'cd "${SLURM_SUBMIT_DIR:-.}"\npython collect_training_results.py')
        .replace("#SBATCH --output=out", "#SBATCH --output=collect-%j.out")
        .replace("#SBATCH --error=err", "#SBATCH --error=collect-%j.err")
    )
    write_unix_shell_script(
        directory / "collect_results.sh", _render_job_name(collector, "mlip-training-review")
    )
    (directory / "UPLOAD_AND_SUBMIT.md").write_text(
        "上传本次登记的 *_input 文件夹，执行 sbatch submit_all.sh 一次；自动解包并派发训练，不另行提交内部 GPU.sh。\n"
        f"Slurm数组包含{count}个独立训练任务，每个使用1张GPU，最多同时运行{concurrency}个；主模型复用已指定committee成员，不重复训练。\n"
        "全部数组任务结束后，依赖作业collect_results.sh使用1张GPU汇总和评估；失败或缺少完成记录时明确报告，不激活不完整模型。\n"
        "确认远端原模型文件存在且mace环境支持foundation_head等参数。\n"
        f"正式committee全部数据参与训练，{plan['main_model']['final_model']}同时为主模型；同文件validation仅训练监视，不是泛化误差。\n"
        "完成后将对应 *_output 文件夹整体传回登记位置，说“已回传”并确认回收范围后批准回收；无需手动解包，不需要下载模型文件。\n"
        "模型保留在超算com_*训练目录；models.json记录绝对远端路径、大小和SHA256，后续任务在超算读取。不要移动或删除这些模型。\n"
        "泛化误差只参考main_cv_*留出数据；模型激活仍需独立验证与审批。\n",
        encoding="utf-8",
    )


def prepare_training_handoff(directory, inputs):
    """Rebuild the same dependency plan when a new package needs re-archiving."""
    from phase_agent.tools.remote.submission_handoff import prepare_handoff

    directory, inputs = Path(directory), Path(inputs)
    plan_path = inputs / "training_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    execution = plan.get("execution") or {}
    scripts = [inputs / "GPU.sh"]
    dependencies = {}
    if execution.get("mode") == "slurm_array":
        scripts.append(inputs / "collect_results.sh")
        dependencies = {
            scripts[1].relative_to(directory).as_posix(): [
                scripts[0].relative_to(directory).as_posix()
            ]
        }
    handoff = prepare_handoff(directory, scripts, afterany=dependencies)
    if dependencies:
        count = len(execution["training_models"])
        handoff.update(
            job_count=count + 1,
            training_job_count=count,
            collection_job_count=1,
            max_parallel_training_jobs=execution["max_parallel_training_jobs"],
        )
    return handoff
