# Manual DFT preparation

Approved pending DFT tasks use prepare_local_batch_files mode dft_inputs. The
interface reuses Process_Vasp.generation.generate_atomate_input from Py-Code on
the local Python path. Only dft_relax is supported by this interface; single-point
tasks are rejected explicitly, never silently converted into relaxation.

Parameters are stage-keyed incar_settings and kpoints_settings; only relax is
allowed. GGA=None is forced in relax input-set overrides (omit GGA tag).
All project-generated INCAR files use ALGO=Normal. This layered-oxide project
also enforces AMIX=0.2, BMIX=0.0001, AMIX_MAG=0.8, BMIX_MAG=0.0001 in both
Py-Code relax input-set overrides and legacy atomate materialized inputs.
The chosen candidate structure_path takes precedence over ledger source_path.

Upload the complete DFT allocation folder. Submit each task's GPU.sh once. The
remote Python environment must provide atomate2, jobflow, pymatgen, VASP command
configuration, and POTCAR data. VASP inputs are generated remotely by atomate2.
The workflow ends after relaxation, without DOS/band/static follow-on jobs.
Final energy/structure and integrity marker are exported to the allocation results
folder through the existing finalize_vasp and export_batch_result protocol.

Preparation never submits or runs calculations. MC/QBC candidate construction is
not changed here. Existing confirmed single-point-first policies must be revised
and approved before proposing relaxation through this interface.
