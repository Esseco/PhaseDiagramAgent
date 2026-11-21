"""Record verified Relax–MC remote input evidence via the Codex Agent."""

import argparse
import json
from phase_agent.runtime.codex.offline import CodexOffline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runtime-config", "bundle-id", "attestation-path", "attestation-sha256"):
        parser.add_argument("--" + key, required=True)
    args = vars(parser.parse_args(argv))
    runtime = args.pop("runtime_config")
    result = CodexOffline(runtime).register_relax_mc_input_attestation(**args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
