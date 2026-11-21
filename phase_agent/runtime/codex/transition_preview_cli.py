"""Preview an authorized Relax–MC transition through the Codex Agent."""

import argparse
import json
from phase_agent.runtime.codex.offline import CodexOffline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = CodexOffline(args.runtime_config).preview_relax_mc_threshold_continuation(
        args.bundle_id
    )
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        if output.exists():
            raise ValueError("preview_output_already_exists")
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({k: v for k, v in result.items() if k != "actions"}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
