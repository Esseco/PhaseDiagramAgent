"""Import verified Relax phase through the Codex project Agent."""

import argparse
import json
from phase_agent.runtime.codex.offline import CodexOffline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--bundle-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            CodexOffline(args.runtime_config).import_relax_mc_phase(args.bundle_id),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
