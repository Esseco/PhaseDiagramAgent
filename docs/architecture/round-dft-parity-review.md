# DFT round and K-fold comparison figures

KFold_Error contains K-fold out-of-fold training prediction figures. Round_Error contains the verified same-frame DFT results and predictions from the frozen model bound to each DFT round. Relaxed DFT tasks use the verified final frame. Partial or absent predictions are not imputed; round_dft_parity.md lists paired/recovered counts and unavailable reasons.

Six authorized projects were refreshed through the existing Agent analysis-only entry: 28 round groups produced figures, one Cr group had no complete pair. All six latest/KFold_Error views have current-model K-fold results. Of the six latest/Round_Error views, FeMn has a verified current-model same-frame figure; the other five contain an explicit unavailability reason. Historical round figures remain available.

KFold_Error uses training round names; old training/<job-id> paths remain hidden compatible links. Historical images and evidence are retained. Latest Round_Error is the model-bound cumulative same-frame result and may include multiple DFT batches; the round index links individual groups. No real MLIP inference or HPC submission was performed. Some groups contain only a subset of recovered structures, explicitly counted in the index; the figures should not be treated as independent model tests without separate evidence.

Validation: 75 targeted tests on the initial implementation, 46 after latest view changes, and 23 after handling previously generated plots whose evidence becomes unavailable. Ruff check passed for touched production files. All six training and round indexes were checked for broken local links. The Ni analysis process reached approximately 2.7 GiB RSS during full refresh; no performance improvement is claimed.
