"""Package training follow-ups through the same immutable batch handoff."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from phase_agent.tools.remote.submission_handoff import prepare_handoff


MANIFEST_ENTRY = """import importlib.util,json
from pathlib import Path
root=Path(__file__).resolve().parent
request=json.loads((root/'manifest_request.json').read_text())
plan=json.loads((root/'training_plan.json').read_text())
spec=importlib.util.spec_from_file_location('collector',root/'collect_training_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output=root.parent/'results';output.mkdir(exist_ok=True)
module.write_model_manifest(Path(request['remote_inputs']),plan,output)
(output/'models.json').replace(output/request['result_name'])
"""


def remote_training_inputs(job, config):
    supplied = config.get("remote_training_inputs_directory")
    context = Path(job["directory"]) / "results/remote_context.json"
    if not supplied and context.is_file():
        supplied = json.loads(context.read_text(encoding="utf-8")).get("input_directory")
    if supplied and PurePosixPath(supplied).is_absolute():
        return supplied
    # Old manifests sometimes have valid absolute paths but missing hashes.
    models = (job.get("returned_results") or {}).get("models") or []
    origins = set()
    for model in models:
        path = PurePosixPath(model.get("remote_model_path") or model.get("model_path") or "")
        if not path.is_absolute():
            continue
        parent = next((p.parent for p in path.parents if p.name == model.get("model_id")), None)
        if parent is not None:
            origins.add(str(parent))
    return next(iter(origins)) if len(origins) == 1 else None


def prepare_followup(job, script, names, kind, handoff):
    inputs = Path(job["directory"]) / "inputs"
    transfer = prepare_handoff(
        Path(job["directory"]),
        [script],
        payload_files=[inputs / name for name in names],
        operation_label=kind,
    )
    handoff.update(
        handoff=transfer,
        script=transfer["submit_script"],
        upload_folder=transfer["folder"],
        output_folder=transfer["return_directory"],
    )
    handoff["reason"] = (
        f"上传整个 {transfer['folder']}，执行 sbatch submit_all.sh 一次；"
        f"完成后将对应 _output 文件夹传回 {transfer['return_directory']}，再说“继续”。无需手动解包或重新训练。"
    )
    return handoff


def manifest_followup(job, config, script, handoff):
    inputs = Path(job["directory"]) / "inputs"
    remote = remote_training_inputs(job, config)
    if remote is None:
        handoff.update(
            script=None,
            reason="原训练的超算 inputs 绝对路径缺失；需补齐 remote_training_inputs_directory，才能封装可执行的清单核对批次。未重新训练或提交任务。",
        )
        return handoff
    token = hashlib.sha256(
        (remote + (inputs / "training_plan.json").read_text(encoding="utf-8")).encode()
    ).hexdigest()[:16]
    name = "models-" + token + ".json"
    (inputs / "manifest_request.json").write_text(
        json.dumps({"remote_inputs": remote, "result_name": name}), encoding="utf-8"
    )
    (inputs / "manifest_job.py").write_text(MANIFEST_ENTRY, encoding="utf-8")
    # Preserve the internal legacy script name; only the paired folder is public.
    path = Path(script)
    text = path.read_text(encoding="utf-8").replace(
        "python collect_training_results.py --manifest-only", "python manifest_job.py"
    )
    if path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")
    handoff["manifest_result"] = str(Path(job["directory"]) / "results" / name)
    return prepare_followup(
        job,
        script,
        [
            "GPU_manifest.sh",
            "training_plan.json",
            "manifest_request.json",
            "manifest_job.py",
            "collect_training_results.py",
        ],
        "ModelManifest",
        handoff,
    )
