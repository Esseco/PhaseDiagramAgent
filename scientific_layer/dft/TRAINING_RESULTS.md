# Portable DFT training results

New converged VASP results embed final ionic-frame labels in `result.json.outputs`.
The existing result checksum and results-directory exporter cover this JSON; no
remote absolute path is needed to recover training structure or labels.

- `structure`: pymatgen Structure JSON (cell, species, coordinates).
- `training_energy`: raw frame `e_0_energy`, eV total, not Eform or Ehull.
- `forces`: N x 3, eV/angstrom.
- `stress`: optional 3 x 3, eV/angstrom^3, ASE tension-positive convention;
  converted from VASP compression-positive kbar.
- `training_frame_index`, `training_schema`, `training_ready`: provenance/status.

Recovery promotes these fields into `new_dft_records`; existing MACE preparation
converts the JSON structure and writes extxyz with REF_energy/REF_forces and,
when requested, REF_stress. Original summary energy remains in outputs for
phase-diagram use. Duplicate task recovery does not duplicate training records.

Missing/invalid labels mark training unavailable without fabricating values or
preventing valid energy-only phase analysis. Stress-required training rejects
missing stress. Nonconverged calculations are not training records.

This change does not retroactively modify old remote scripts or result files.
Existing jobs need the updated project parser available on the remote machine
and result re-extraction from their retained vasprun.xml (possibly compressed).
Energy/CONTCAR-only old results cannot reconstruct forces or stress.
Only the final frame is exported, not the entire relaxation trajectory.

## Official workflow integration

`run_workflow` now supplies a default same-frame evaluator through the existing
py-mace subprocess. It performs prediction only, never re-relaxes the DFT frame.
It uses the configured main model (committee main index when applicable), and
requires the DFT task model version to match. Missing model/environment or
prediction failures are explicit `not_evaluated` records, not fabricated errors.
An explicit runtime evaluator still overrides the default.

Scientific feedback retains portable records in `dft_dataset_records`, adds
training-ready data to `new_dft_records`, and keeps paired evidence in
`dft_mlip_comparisons`. Existing phase identification and hull refresh remain
authoritative. Phase identification is reused in dataset exports.

Under the configured phase-diagram directory, generated products are:
`dft_results/<MLIP-version>/DFT-round-<scope-hash>/training.json`,
`dft_records.json`, and `mlip_dft_metrics.json`. Scope uses the saved upload
operation/search group/parent Relax lineage. When no upload operation exists,
batch/task scope is retained instead of inventing a round association.
Unchanged files are not rewritten; duplicate recovery is not re-evaluated.

Energy MAE/RMSE are reported as total eV and per-atom eV/atom, equally weighted
per structure, without fitted offsets. Force MAE/RMSE average all atomic Cartesian
components. RMSE is sqrt(mean squared error), in the same units as MAE.
MSE is not exported. Incomplete prediction coverage is listed.
These are validation errors of the task's model, not committee disagreement.

Existing finalized result files are not automatically regenerated; old jobs still
need re-extraction or the independent recovery script. If the task model is no
longer configured, matching-version evaluation must be provided explicitly.
