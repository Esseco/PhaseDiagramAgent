# Local legacy DFT failure return repair

`python -m phase_agent.runtime.repair_legacy_return --runtime-config <agent_runtime.json> --archive <registered-return-dir/legacy.results.zip> --receipt <downloaded-submitted-jobs.json> --task-id <DFT-id> --job-id <Slurm-id>`

This evidence-only command verifies the original immutable input inventory, submission receipt input hash and job ID, archive inventory, result identity, completion marker and result checksum. It re-exports only the selected failed/timeout task with the registered batch binding. The raw archive is preserved; it is never rewritten or treated as a valid bound return. Other unbound raw archives may still appear as rejected transfer diagnostics, independently of the verified failure import.

Import the verified archive through the existing runtime result collector and public result-recovery entry. Do not manually change task status. A failed task can then enter `restart_failed_task` through the normal Codex proposal/review gates.

For a complete legacy return, use `--all-terminal` instead of `--task-id`/`--job-id`. Each result is matched to its original task and dispatcher job; successful tasks additionally transport the checksum-verified training/MLIP/structure attachments through the existing artifact validator. The final scientific acceptance remains the runtime collector's responsibility. Repeated calls reuse only unchanged verified receipts and archives.

The optional explicit retry parameters `repair_profile=vasp_symmetry_single_gpu` and `prepare_inputs_after_approval=true` preserve the original workflow settings, override only the failed task's ISYM/SYMPREC and MPI rank count, and prepare its inputs through the existing manual DFT batch generator. No upload or scheduler submission is performed. Other pending tasks retain their recovery tracking. The generated package must be uploaded with its complete `_input` directory name to retain batch binding; do not rename it to `input`.

Validation: `tests/test_repair_legacy_return.py`, `tests/test_dft_failure_retry_profile.py`, and `tests/test_retry_admission_integrity.py`.
