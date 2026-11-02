"""Export exactly the files consumed by Process_AL_MC ordering inspection."""

import json
from pathlib import Path, PurePosixPath
from phase_agent.tools.remote.transfer_bundle import create_bundle, checksum

ROOT_RESULTS = {
    "plan.json",
    "execution_binding.json",
    "result.json",
    "global_ranking.csv",
    "all_joint_records.csv",
    "diagnostics.csv",
    "top_structures.json",
    "result_manifest.json",
}


def allowed_result(name):
    path = PurePosixPath(name)
    return (
        name in ROOT_RESULTS
        or (
            len(path.parts) == 3
            and path.parts[0] == "matrices"
            and path.name in {"ranking.csv", "top_structures.json"}
        )
        or (
            len(path.parts) == 4
            and path.parts[0] == "matrices"
            and path.parts[2] == "structures"
            and path.suffix == ".vasp"
        )
    )


def export_ordering_result(directory, destination=None):
    root = Path(directory).resolve()
    manifest = json.loads((root / "result_manifest.json").read_text(encoding="utf-8"))
    names = set(manifest["files"]) | {"result_manifest.json"}
    if not ROOT_RESULTS.issubset(names):
        raise ValueError("ordering results are incomplete")
    for name in names:
        path = (root / name).resolve()
        path.relative_to(root)
        if not allowed_result(name) or not path.is_file():
            raise ValueError("unexpected ordering result: " + name)
        if name != "result_manifest.json" and checksum(path) != manifest["files"][name]:
            raise ValueError("ordering result checksum mismatch: " + name)
    return create_bundle(
        root,
        [root / name for name in names],
        destination or root.parent / (root.name + ".results.zip"),
        kind="results",
        binding={"plan_id": manifest["plan_id"]},
    )


if __name__ == "__main__":
    import sys

    print(export_ordering_result(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent))
