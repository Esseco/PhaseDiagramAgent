# Initial and run configuration

`parameters/search_config.project.json` is the editable initial configuration. It contains local/remote Python environments, system boundaries and composition, mother-structure paths, the remote MLIP model, DFT/Slurm settings and storage paths. This existing filename is retained so project launch bindings remain valid.

`parameters/run_config.project.json` contains editable run overrides: budgets, branch selection/sampling, MC, fine-tuning, validation, convergence and later Agent policies. Missing values inherit the pinned profile; startup audit reports later-stage missing thresholds separately rather than demanding them before any search advice. Invalid supplied values and missing initial facts still require correction. Calculation stages retain their own checks; startup readiness is not permission to submit jobs or activate a model.

Both files are read together. Change either manually and say `读取配置 JSON` (review only), or `读取配置 JSON 并继续` (review and proceed if valid). `读取初始配置` and `读取运行配置` also check the combined configuration. During a running search, these commands open a new configuration revision without executing the old proposal. Agent edits route initial fields to the initial file and policy fields to the run file. Run overrides cannot replace initial environment/structure/boundary fields.

Every approved complete effective configuration still has an immutable version snapshot in `parameters/snapshots/`. Historical jobs keep their configuration versions. Editing a file never silently changes an executing job. Review identity includes the bytes of both files, plus the existing checks on imported effective configuration and mother structures, so a manual edit to either file requires a fresh review.

New projects generate both files automatically. The initial draft does not inherit another project's TM composition or HPC model path. Default local ordinary environment remains py1; remote environments must be supplied. Existing single-file project-v2 configurations remain supported. Explicit splitting preserves effective values and keeps `<initial-file>.before-split.bak`; existing live states and confirmed snapshots are not rewritten.

Later-stage independent validation uses `remote_training_validation` in the run file (data_path, data_version, ranking_pairs, criteria and optional label keys), as described in training_handoff.md. Initial local/remote environments and model paths remain in the initial file. DeepSeek credentials remain in the existing credential store/environment, not in editable scientific JSON.

The combined configuration is the effective next-run source; the initial file records current startup facts, while immutable snapshots preserve historical startup and run settings. Run overrides contain persistent user settings, not automatically promoted per-round Agent suggestions or memory. At runtime an Agent may still propose a one-round action without changing either file.

Verification: py1 regression tests cover splitting round trips, legacy compatibility, field routing, manual edit detection across both files, unknown/forbidden fields, deferred startup thresholds, running-chat import, configuration review, launcher, Studio, model handoff and refresh behavior. Actual searches or HPC computations were not launched by this change.
