"""Preview MC after the verified Relax phase was imported."""

import argparse, json
from pathlib import Path
from phase_agent.runtime.codex.offline import CodexOffline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("preview_output_already_exists")
    data = CodexOffline(args.runtime_config).preview_imported_relax_mc(args.bundle_id)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in data.items() if k != "actions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
