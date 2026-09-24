#!/usr/bin/env bash
set -euo pipefail
#SBATCH --job-name=remote-000001
#SBATCH --array=0-0

batch_directory=$(cd "$(dirname "$0")" && pwd)
python3 -m execution_layer.remote.worker_cli --executor scientific_layer.mlip.slurm_executor:execute_mlip_task --manifest "$batch_directory/manifest.json" --array-index "$SLURM_ARRAY_TASK_ID"
