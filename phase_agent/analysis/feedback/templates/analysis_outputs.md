# Analysis outputs

- `latest/`: current verified phase diagrams, voltage curve and DFT error, with explicit status. Missing or invalid results are never described as complete.
- `epochN/phase_diagram/`: numbered scientific analysis steps and the final complete publication of this epoch (`latest/`). Step order is never guessed or renumbered.
- `epochN/dft/`: Search-group/DFT-round products, comparisons, training records, magnetic analysis and diagnostics.
- `epochN/dft/summary/`: model-bound cumulative DFT error and status, alongside individual DFT rounds.
- `epochN/diagnostics/`: sampling, extrapolation and recovery evidence when present.
- `out_data/`: all current exported tables and their supporting metadata/logs.
- `history/`: every immutable version, internal snapshot, legacy phase diagram, selection artifact, failed-epoch archive and retained previous-latest output. No scientific files are deleted.

`README.md` and each epoch README list all categories. Full registered indexes and step metadata remain in `workflow_state/analysis/`.

Old directory paths are retained as verified hidden compatibility junctions on Windows; they reference the relocated files and do not duplicate data. Do not manually delete their targets or recursively copy both aliases and targets. Existing state, frozen approvals and checkpoint references remain unchanged. The Agent organizer validates destinations, hashes moved files and rolls back failed moves.
