# Exact MC occupation history

The project owns `workflow_state/cache/mc_occupancies.sqlite` (relative to the resolved state file). Branch IDs partition the history; MLIP version and MC round do not. Reference site ordering must match. Only Na/Vac occupation bitmaps are compared, without symmetry matching.

`Process_AL_MC` accepts optional `occupation_reference_file` (a compact snapshot) or `occupation_history_seed` (the same snapshot encoded as base64 for portable task JSON). Without either, only single-run visited states are used. The library does not manage a project database. A missing, invalid or incompatible explicitly provided file fails visibly.

New task preparation automatically merges recovered `mc_occupations.bin` deltas and passes the branch reference to the worker. Only the compact delta returns; database files stay local. Successful evaluations are recorded even when the MC move is rejected. Failed evaluations remain eligible. Initial seeds may be evaluated with the current model.

`max_duplicate_trials=100` limits consecutive duplicate proposals; stopping sets `duplicate_trial_limit`. Duplicate proposals do not consume effective MC steps or patience. This limit indicates failure to find a fresh proposal within the budget, not exhaustive enumeration.

Old tasks can backfill retained initial/final structures only. Unsaved historical trials cannot be reconstructed. Existing submission packages are unchanged; regenerated/new packages bundle the updated backend automatically. Concurrent jobs seeded before either returns do not share a live remote index.
