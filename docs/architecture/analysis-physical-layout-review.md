# Analysis physical layout

Six projects were organized through the existing analysis-only Agent entry. The organizer reports file hashes before and after each move, keeps hidden compatibility junctions, and checks the results on repeat. State SHA256 remained unchanged; checkpoint files were checked by size and modification time, not by content hash.

| Project | Directories moved | Files checked | Organization status |
| --- | ---: | ---: | --- |
| O3-NaCo-3phase | 13 | 563 | completed |
| NaNiO2-phase | 27 | 704 | incomplete |
| O3-1ele-NaCoO2 | 11 | 203 | incomplete |
| O3-2ele-NaFeMn | 17 | 334 | completed |
| O3-NiCo | 20 | 1230 | incomplete |
| O3-1ele-NaCrO2 | 35 | 1061 | completed |

Incomplete statuses reflect preexisting scientific publication or selection validation issues; see per-project analysis_organization.json. Cr MCP PID 52196 was stopped by explicit user authorization; it was not restarted. Full data: analysis-physical-layout-results.json.

Validation: 124 targeted tests passed before final boundary changes; subsequent 39 and 31 targeted tests passed. Ruff check of analysis/feedback and codex_team passed. Existing services must reload source separately.
