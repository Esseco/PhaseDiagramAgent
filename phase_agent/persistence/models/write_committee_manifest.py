"""记录训练完成后的不同 MACE 模型及其文件指纹。"""

import hashlib
import json
from pathlib import Path


def write_committee_manifest(
    models: list[dict], output_path, *, committee_id: str, data_version: str, foundation_model: str
) -> dict:
    members, fingerprints = [], set()
    for model in models:
        path = Path(model["model_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in fingerprints:
            raise ValueError(f"委员会包含重复模型内容：{path}")
        fingerprints.add(digest)
        members.append({**model, "model_path": str(path.resolve()), "sha256": digest})
    manifest = {
        "committee_id": committee_id,
        "data_version": data_version,
        "foundation_model": foundation_model,
        "member_count": len(members),
        "members": members,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)
    return manifest
