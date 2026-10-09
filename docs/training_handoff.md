# Training result handoff

The existing scientific graph collects training artifacts, then advances a durable training_handoff in each remote_finetune_job. State is saved before returning a manual wait. Repeated continue reads returned files and preserves identical generated scripts. No scheduler calls or local model inference are performed.

1. Missing paths/hashes: generates inputs/GPU_manifest.sh and refreshes collect_training_results.py. Upload both to the original HPC inputs directory; submit sbatch GPU_manifest.sh. Return results/models.json.
2. Complete manifest: checks independent validation prerequisites. It never uses K-fold results or the full training set as an independent test.
3. Configured independent validation: generates validation.xyz, validation_request.json, validate_remote_training.py and GPU_validation.sh. Upload to original inputs; submit sbatch GPU_validation.sh. Return the request-specific results/validation-<request_id>.json.
4. Recovery binds the report to the request, model manifest and dataset hashes, checks finite metrics, then registers candidate_models using the existing validate_mlip criteria. Failed validation/rejected candidates return control to downstream Agent decision-making.
5. Successful candidates require an explicit chat command: 激活候选 <candidate-version> 原因：<review reason>. 拒绝候选 <candidate-version> 原因：<reason> records rejection. Generic continue/approve never activates. Activation rechecks returned artifact fingerprints, uses existing governance and marks model-dependent results stale. Next continue enters existing model-refresh approval, followed by round-budget comparison.

Job scripts reuse execution_layer/remote/mlip_gpu_template.sh (the user's v100m3 template). Metadata operations also run on compute nodes. An optional confirmed-config remote_training_job_template may point to a compatible template with python3 run_mlip_task.py and conda activate mace placeholders.

Configure remote_training_validation in the confirmed scientific configuration (not only agent_runtime.json, whose unknown runtime keys are ignored):

```json
{
  "remote_training_validation": {
    "data_path": "E:/project/validation/independent.xyz",
    "data_version": "independent-v1",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "ranking_pairs": [[0, 1]],
    "criteria": {
      "max_energy_mae": null,
      "max_force_rmse": null,
      "max_critical_failure_fraction": null,
      "max_near_hull_ranking_reversals": null
    }
  }
}
```

Null thresholds are intentionally unconfigured; obtain scientifically justified user-confirmed limits. Energy MAE uses eV/atom and force RMSE uses eV/Angstrom. Ranking pairs are zero-based indices of independently labeled, same-composition near-hull structures, selected before seeing predictions. Active old-model metadata must include version, remote model_path and sha256 (mace_head if needed). Keep original inputs/_shared_data/train.xyz on HPC. The runner checks file hashes and rejects exact training-geometry overlap; the dataset owner must also exclude shared parent/trajectory provenance, as geometry comparison cannot establish that independence. Evaluation covers old and new main models, not a full committee calibration or dynamics-stability benchmark. Any failed/nonfinite prediction fails the job; no structures are silently dropped and no partial pass is returned.

Each handoff transition enters training_handoff_history and the existing evidence-verified memory-candidate pipeline. These are factual observations, not automatically approved durable advice or estimated scientific gains. Do not overwrite live state externally; restart the service and use continue to recover and persist transitions.

Validation: py1 tests cover scheduled metadata repair, repeated/restarted handoffs, prerequisite waits, request identity, nonfinite reports, candidate registration, explicit activation, changed-artifact rejection, memory extraction, reply rendering and mocked remote evaluation/overlap checks. Actual HPC inference remains to be verified by a submitted job.

## Native graph persistence

Training handoff now runs in the discoverable `scientific_lifecycle/training_lifecycle` subgraph. Collect, manifest checking, metadata job preparation, validation prerequisites, validation job preparation, validation recovery/candidate registration and transition recording are independent graph nodes. Waiting for manifest return, validation return, explicit activation review or missing configuration uses native LangGraph interrupt. The project-specific SQLite checkpoint database is `workflow_state/langgraph_training.sqlite`, with one thread per state-path/training-job identity and a file lock. Graph state contains serializable business/configuration snapshots, never manager/client/runtime objects.

Continue resumes the interrupted node using fresh canonical business state and configuration, then collects and checks current artifacts again. The checkpoint does not authorize activation or override state.json. A crash between nodes restarts reconciliation before preparing idempotent local artifacts. Explicit activation/rejection still uses the existing governance entry; the next continue reconciles that decision and leaves the subgraph for normal model refresh and round-budget review. Terminal rejected/failed validations do not force repeated waiting. The whole scientific graph is not globally checkpointed by this change; the training lifecycle has its own durable checkpoint boundary, while other stages retain their existing persistence.

Tests use actual SQLite saver close/reopen, native interrupt snapshots, fresh-state resume, changed files, validation wait to approval wait to explicit activation, and production Studio child discovery. HPC inference has not been executed locally.
