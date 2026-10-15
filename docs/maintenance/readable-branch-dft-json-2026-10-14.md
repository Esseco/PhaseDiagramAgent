# Readable branch and DFT review JSON — 2026-10-14

Applied the user-approved Chinese JSON example to branch and DFT selection review artifacts. Human-readable ordered sections come first; exact canonical program data remains in the final 程序记录（可折叠） section. Planned and actual quantities are separate. Missing scientific values and training membership remain explicitly unknown. Supercell matrices are deduplicated for display, and displayed cost rounding does not change canonical values.

Writers support legacy raw and wrapped records throughout proposal/preparation lifecycle. The shared atomic JSON writer retains its default sorting behavior; review writers explicitly preserve presentation order.

Deployment baseline hashes were checked and originals backed up. Existing FeMn branch_plan_a30bbc76c7c8681d.json was reformatted, with raw record equality verified. No current DFT review files were found in Co/Cr/FeMn. No scientific generation, submission, state approval or service restart was performed. Running services need a manual restart to load updated modules.

Final production verification: 15 passed, 1 deselected in 3.26s. Tests cover readable reviews, branch artifacts, DFT artifacts and compact transfer handoff; actual_coordinator excluded. Earlier Ruff checks passed. Existing artifact backup is under build/readable_branch_json/existing_artifact_backups.
