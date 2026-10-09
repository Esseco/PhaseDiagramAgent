# Manual DFT preparation

Approved pending DFT tasks use prepare_local_batch_files mode dft_inputs. The
interface reuses Process_Vasp.generation.generate_atomate_input from Py-Code on
the local Python path. dft_relax uses RelaxMaker; dft_single_point uses one StaticMaker
directly, without preceding relaxation. Static overrides enforce NSW=0, IBRION=-1.

Parameters are stage-keyed incar_settings and kpoints_settings; relax/static keys
are allowed. GGA=None is forced in both input-set overrides (omit GGA tag).
All project-generated INCAR files use ALGO=Normal. This layered-oxide project
also enforces AMIX=0.2, BMIX=0.0001, AMIX_MAG=0.8, BMIX_MAG=0.0001 in both
Py-Code relax input-set overrides and legacy atomate materialized inputs.
The chosen candidate structure_path takes precedence over ledger source_path.

Upload the complete DFT allocation folder. Submit each task's GPU.sh once. The
remote Python environment must provide atomate2, jobflow, pymatgen, VASP command
configuration, and POTCAR data. VASP inputs are generated remotely by atomate2.
The workflow ends after the selected relaxation or single point, without follow-on jobs.
Final energy/structure and integrity marker are exported to the allocation results
folder through the existing finalize_vasp and export_batch_result protocol.

Preparation never submits or runs calculations. MC/QBC candidate construction is
not changed here. Existing confirmed single-point-first policy and relaxation fraction
cap are preserved; calculation types are never silently converted.

After a recorded second MC allocation is fully recovered, continuation no longer
requires another MC allocation. DFT candidate context is rebuilt from the latest
recovered segment per branch and unique identified current-model hull entries.
Candidate structure paths are retained; QBC is copied only from real outputs and
is explicitly not_configured when missing. This does not evaluate a committee or
authorize a DFT calculation. Existing policy/approval validation remains mandatory.

New interactive DFT proposals preview separate single-point/optimization counts, selection
reasons and configured nonzero cost. Approval registers child tasks and immediately
calls the same local upload preparer; no second input-preparation proposal is needed.
Empty, unsupported, partially rejected or uncosted plans cannot be approved as input
generation. Legacy pending DFT proposals without this preview must be rejected and
requested again. Existing sensitive-operation approval safeguards are unchanged.
