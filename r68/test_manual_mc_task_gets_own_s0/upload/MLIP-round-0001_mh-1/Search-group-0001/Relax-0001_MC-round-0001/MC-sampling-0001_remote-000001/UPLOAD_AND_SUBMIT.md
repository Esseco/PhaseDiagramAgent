# Manual upload batch `remote-000001`

This directory was generated locally. Nothing has been submitted.

1. Keep every submission directory for this stage as a sibling under the same
   allocation folder under the MLIP-round stage. Inspect `manifest.json`, `task.json`, and `GPU.sh`.
2. Confirm the model/input files and all site-specific Slurm settings.
3. Upload the complete allocation folder, keeping its batch directories as siblings. Relax jobs
   contain up to 100 structures; MC jobs contain up to 10 compatible simulations,
   possibly from multiple branches.
4. Optionally verify its files with `sha256sum -c SHA256SUMS`.
5. For MLIP Relax/MC, submit the batch root's `GPU.sh` once; it runs each task
   subdirectory and collects results in this allocation folder's shared `results/`.
   For DFT, change into the only task subdirectory and submit its `GPU.sh` once.
   Never submit both levels.
6. After all jobs for this allocation finish, download the allocation folder's single
   `results/` directory into the matching local allocation folder. It includes result JSON,
   completion markers and required final structures.

The generated script is not evidence that executables, environments, paths,
pseudopotentials, permissions, or resource requests are correct for your site.
For DFT, the remote Python environment must import this project and pymatgen
to parse VASP output and publish `results/` after the calculation.
