"""Shared submission namespace contract and epoch manifests."""

from __future__ import annotations

import json
from pathlib import Path


_NAVIGATION = {
    "root_files": [
        "README.md",
        "submission_index.json",
        ".archive_index.json (optional archive registry)",
        ".layout-migration.json (optional migration receipt)",
    ],
    "read_order": [
        "README.md",
        "submission_index.json",
        "epochN/README.md",
        "epochN/layout_manifest.json",
    ],
    "lanes": [
        {
            "id": "delivery",
            "label": "交付层",
            "paths": ["epochN/delivery/transfers/", "epochN/delivery/relax_mc/"],
            "rule": "只上传登记的 *_input；远端 *_output 原样回传；目录中的无后缀旧包在 manifest 中标为 quarantine，不能作为当前入口。",
        },
        {
            "id": "calculation",
            "label": "计算层",
            "paths": [
                "epochN/calculations/<operation>/inputs/",
                "epochN/calculations/<operation>/results/",
            ],
            "rule": "本地计算工作目录和输入/结果留在原登记位置；不作为上传入口。",
        },
        {
            "id": "evidence",
            "label": "证据层",
            "paths": [
                "epochN/evidence/verified_results/",
                "epochN/evidence/recovery/",
                "epochN/evidence/storage_archives/",
            ],
            "rule": "已校验结果、回收凭据和完整归档只读保存；不要重复提交。",
        },
        {
            "id": "quarantine",
            "label": "隔离层",
            "paths": ["previously_used/"],
            "rule": "未登记或历史材料仅供追溯；禁止从这里提交或回收。",
        },
    ],
}

_FIXED_CHILD_LANES = {
    "transfers": "delivery",
    "relax_mc": "delivery",
    "recovery": "evidence",
    "verified_results": "evidence",
    "storage_archives": "evidence",
    ".transfer_receipts": "evidence",
    "previously_used": "quarantine",
}

# Every epoch has the same three physical buckets.  The bucket names are part
# of the on-disk contract so an Agent can understand an epoch from its tree
# before reading any task-specific names.
EPOCH_BUCKETS = {
    "calculations": {"lane": "calculation", "children": ()},
    "delivery": {"lane": "delivery", "children": ("transfers", "relax_mc")},
    "evidence": {
        "lane": "evidence",
        "children": ("recovery", "verified_results", "storage_archives", ".transfer_receipts"),
    },
}

LEGACY_EPOCH_CHILDREN = frozenset(_FIXED_CHILD_LANES)


def epoch_bucket(epoch_directory, bucket):
    """Return a canonical physical bucket inside an epoch."""
    if bucket not in EPOCH_BUCKETS:
        raise ValueError(f"Unknown submission epoch bucket: {bucket}")
    return Path(epoch_directory) / bucket


def epoch_delivery_directory(epoch_directory, kind):
    if kind not in {"transfers", "relax_mc"}:
        raise ValueError(f"Unknown delivery directory: {kind}")
    return epoch_bucket(epoch_directory, "delivery") / kind


def epoch_evidence_directory(epoch_directory, kind):
    if kind not in {"recovery", "verified_results", "storage_archives", ".transfer_receipts"}:
        raise ValueError(f"Unknown evidence directory: {kind}")
    return epoch_bucket(epoch_directory, "evidence") / kind


def submission_navigation():
    """Return a fresh copy of the stable root navigation contract."""
    return {
        "root_files": list(_NAVIGATION["root_files"]),
        "read_order": list(_NAVIGATION["read_order"]),
        "lanes": [dict(lane, paths=list(lane["paths"])) for lane in _NAVIGATION["lanes"]],
    }


def navigation_markdown():
    lines = [
        "## 先读这里",
        "",
        "1. 先看本页 `README.md`：只决定当前要交付/回传的目录。",
        "2. 再读 `submission_index.json`：读取精确路径、命令、状态和存在性；使用前重新核对文件。",
        "3. 最后看 `epochN/README.md` 与 `layout_manifest.json`：确认模型版本和该轮每个直接子目录的职责。",
        "",
        "| 层 | 入口 | 用途 |",
        "| --- | --- | --- |",
    ]
    for lane in _NAVIGATION["lanes"]:
        paths = "、".join(f"`{path}`" for path in lane["paths"])
        lines.append(f"| {lane['label']} | {paths} | {lane['rule']} |")
    lines.append("")
    return lines


def build_epoch_manifest(
    submission_root, epoch_directory, registered_directories, model_version=None
):
    """Classify an epoch's direct children without relocating any payload."""
    root = Path(submission_root).resolve()
    epoch = Path(epoch_directory).resolve()
    epoch.relative_to(root)
    registered = {Path(path).resolve() for path in registered_directories}
    children = []
    unclassified = []

    def entry(path, lane, registered_flag, nested=None):
        row = {
            "name": path.name,
            "path": path.relative_to(root).as_posix(),
            "entry_type": "directory" if path.is_dir() else "file",
            "lane": lane,
            "registered": registered_flag,
            "exists": path.exists(),
        }
        if nested is not None:
            row["children"] = nested
        return row

    for child in sorted(epoch.iterdir(), key=lambda path: path.name):
        if child.name in {"README.md", "layout_manifest.json"}:
            continue
        bucket = EPOCH_BUCKETS.get(child.name)
        if bucket is None:
            lane = _FIXED_CHILD_LANES.get(child.name)
            if lane is None:
                lane = "calculation" if child.resolve() in registered else "unclassified"
            if lane == "unclassified":
                unclassified.append(child.name)
            children.append(entry(child, lane, lane == "calculation"))
            continue
        nested = []
        for item in sorted(child.iterdir(), key=lambda path: path.name):
            item_lane = bucket["lane"]
            item_registered = False
            legacy = False
            if child.name == "calculations":
                item_registered = item.resolve() in registered
                if not item_registered:
                    item_lane = "unclassified"
                    unclassified.append(f"calculations/{item.name}")
            elif child.name == "delivery" and item.is_dir():
                # Delivery's first-level children are the stable `transfers`
                # and `relax_mc` namespaces.  Classify their package folders
                # one level deeper so the namespace itself stays delivery.
                packages = []
                for package in sorted(item.iterdir(), key=lambda path: path.name):
                    package_lane = "delivery"
                    package_legacy = package.is_dir() and not package.name.endswith(
                        ("_input", "_output")
                    )
                    if package_legacy:
                        package_lane = "quarantine"
                    package_row = entry(package, package_lane, False)
                    if package_legacy:
                        package_row["legacy"] = True
                    packages.append(package_row)
                row = entry(item, item_lane, item_registered, packages)
                nested.append(row)
                continue
            row = entry(item, item_lane, item_registered, None)
            if legacy:
                row["legacy"] = True
            nested.append(row)
        children.append(entry(child, bucket["lane"], False, nested))
    return {
        "schema_version": 2,
        "epoch": epoch.name,
        "model_version": model_version,
        "read_order": ["README.md", "layout_manifest.json", "../submission_index.json"],
        "children": children,
        "unclassified": unclassified,
    }


def write_epoch_manifests(submission_root, model_rounds, registered_directories):
    root = Path(submission_root).resolve()
    registered = {Path(path).resolve() for path in registered_directories}
    written = []
    for version, number in (model_rounds or {}).items():
        epoch = root / f"epoch{int(number) - 1}"
        if not epoch.is_dir():
            continue
        manifest = build_epoch_manifest(root, epoch, registered, model_version=version)
        path = epoch / "layout_manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written
