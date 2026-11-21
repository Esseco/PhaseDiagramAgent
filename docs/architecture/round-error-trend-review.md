# Round error trends and missing current-model pairs

Current selected MLIP predictions and DFT labels are distinct. NaCo-3phase has 32 current-model DFT single-point tasks with pending status; all 32 selected structure IDs occur in the current model phase diagram, but none of their registered result directories exists locally. This says nothing about remote completion. The other four projects with unavailable current-model comparisons have no registered current-model DFT task in the saved state. Local mlip_result.json searches in the five project submissions found only previous-model predictions; this search did not inspect remote HPC or every archive.

Trend charts use selected qbc task keys with candidate/object identity and input digest when present, plus registered parent-first selected task IDs with sampling evidence and model binding. They use only DFT/MLIP verified same-frame pairs and group by model/epoch; excluded selection provenance is counted. Plots use per-atom energy and pooled atomic Cartesian force components. Epochs may cover different structures; trends are not a fixed independent test. The existing per-round scatter figures are not filtered by this new selection provenance rule.

| Project | Selected matched structures per epoch |
| --- | --- |
| O3-NaCo-3phase | epoch0: 22 |
| O3-1ele-NaCoO2 | epoch0: 30 |
| O3-2ele-NaFeMn | epoch0: 27, epoch1: 12 |
| O3-NiCo | epoch0: 27, epoch1: 71 |
| NaNiO2-phase | epoch0: 21, epoch1: 17, epoch2: 60 |
| O3-1ele-NaCrO2 | epoch0: 12, epoch1: 12, epoch2: 15, epoch3: 17 |

63 targeted regression tests passed; after changing the automatic refresh trigger, the selected-provenance test passed and Ruff passed. Trend charts were visually inspected for Cr. The six figures were generated via the existing analysis-only Agent entry; no model inference, DFT calculation, submission or service restart was performed. Values and missing reasons are stored in each latest/Round_Error_MAE_by_epoch.csv and JSON.

No remote HPC state was inspected. Further completion requires the project Agent to inspect/recover returned results, or prepare a separately approved same-frame prediction when appropriate.
