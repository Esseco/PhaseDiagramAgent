"""Public remote runner, including integrity-aware DFT finalization."""
import shlex
from execution_layer.remote.batch_runner import DFT_STAGES, RemoteBatchRunner as _Base


class RemoteBatchRunner(_Base):
    def _script(self, batch_id, count, stage):
        if stage not in DFT_STAGES: return super()._script(batch_id, count, stage)
        profile = next((p for p in self.stage_profiles.values() if stage in set(p.get("stages") or [])), {})
        command = profile.get("vasp_command")
        if not command: raise ValueError(f"DFT stage {stage} lacks vasp_command")
        options = {"job-name": batch_id, "array": f"0-{count - 1}", **(profile.get("slurm_options") or {})}
        directives = "\n".join(f"#SBATCH --{key}={value}" for key, value in options.items())
        preamble = "\n".join(profile.get("shell_preamble") or [])
        body = ('batch_directory=$(cd "$(dirname "$0")" && pwd)\n'
                'task_prefix=$(printf "%s/%05d-" "$batch_directory" "$SLURM_ARRAY_TASK_ID")\n'
                'task_directory=$(compgen -G "${task_prefix}*" | head -n 1)\ncd "$task_directory"\n'
                f'set +e\n{command}\nexit_code=$?\nset -e\n'
                'python3 -m execution_layer.remote.finalize_vasp --directory . --exit-code "$exit_code"\n')
        return f"#!/usr/bin/env bash\nset -euo pipefail\n{directives}\n{preamble}\n{body}"
