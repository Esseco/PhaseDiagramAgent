# Analysis folder organization — 2026-11-10

## Scope
Preserve all scientific output kinds. Add complete root and per-epoch folder navigation; group newly generated DFT rounds under epochN/dft. Existing or registered DFT destinations remain unchanged, including missing registered destinations. Conflicting old/new destinations require review. Existing phase, DFT-error and versioned publication layouts remain intact. Explicit out_data and failed-epoch archive exceptions are now documented.

Navigation only lists directory names and does not read state or checkpoints. User text outside the generated navigation block is retained. Existing organization still owns publication validation and legacy migration; this change does not bypass it. Existing history directories were not forcibly relocated, and no scientific reference migration was attempted. This is not a complete physical rewrite of all historical folders.

## Verification
py1: 59 targeted tests passed across folder navigation, organization, output navigation, DFT products, out_data, publication, scaffold and recovered DFT entry. Ruff check of analysis/feedback passed. New navigation/path/export files were formatted. Synthetic tests cover retained product bytes, repeated navigation, legacy destinations, missing registered destinations and conflicts. No comprehensive performance benchmark was performed.

## Six real projects
After source verification, each was processed through the existing stdio MCP entry with --analysis-only and phase_plot(organize_analysis=true, refresh_analysis=false). No direct project file patching, model API, HPC action or persistent service restart was used. All six root navigation files exist and their links resolve. State SHA256 remained identical for all six; checkpoint database files were checked by size and modification time, not full content hash.

- NaNiO2-phase: epoch2 lacks a verified complete current MLIP hull; reported as incomplete.
- O3-1ele-NaCoO2: epoch1 Relax/MC formal publication is blocked; reported as incomplete.
- O3-1ele-NaCrO2, O3-2ele-NaFeMn, O3-NaCo-3phase: completed with retained legacy phase directories.
- O3-NiCo: latest/Historical_Reports contains a state-referenced structure and remains protected; reported as incomplete.

Detailed outcomes: analysis-folder-organization-results.json. Existing persistent services have not reloaded this source; the one-shot analysis entry used the updated source directly. Historical directory flattening remains intentionally limited by saved references, not silently declared complete.
