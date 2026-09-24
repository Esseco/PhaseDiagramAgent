"""Create the same integrity envelope for independently executed VASP tasks."""
import argparse
import json
from pathlib import Path
from execution_layer.remote.integrity import file_checksum, payload_checksum
from scientific_layer.dft.parse_vasp_result import parse_vasp_result


def finalize_vasp(directory, *, exit_code=0):
    root = Path(directory); task = json.loads((root / "task.json").read_text(encoding="utf-8"))
    result = parse_vasp_result(root, exit_code=exit_code)
    keys = ("task_id", "task_key", "batch_id", "config_version", "model_version")
    result.update({key: task.get(key) for key in keys}); result["task_checksum"] = payload_checksum(task)
    result_path = root / "result.json"; _write(result_path, result)
    marker = {key: result.get(key) for key in (*keys, "task_checksum", "status")}
    marker.update({"result_file": result_path.name, "result_checksum": file_checksum(result_path)})
    _write(root / "task.finished.json", marker); return result


def _write(path, payload):
    temporary = Path(f"{path}.tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False,
        indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--directory", required=True)
    parser.add_argument("--exit-code", type=int, default=0); args = parser.parse_args(argv)
    finalize_vasp(args.directory, exit_code=args.exit_code); return 0


if __name__ == "__main__": raise SystemExit(main())
