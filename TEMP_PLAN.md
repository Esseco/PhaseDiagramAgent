# Interaction Efficiency Implementation Plan

Goal: Preserve every chat and scientific decision boundary while reducing shared-path memory, I/O and provider traffic.

Architecture: Replace Studio PersistentDict checkpoint containers with lazy compressed SQLite mappings; retain original pickle files and migrate offline only while services are stopped. Keep business tasks/approvals/results untouched. Keep complete evidence accessible, transport compact views and prevent duplicate repair requests.

Constraints: py1; no real service start/stop; no live model calls or scientific workflow runs; no deletion of chat history; baseline hash check and source backups; all production service transitions remain user-controlled.

Tasks:
- [ ] Implement and test lazy nested Studio SQLite mappings, bounded value cache, atomic checked offline migration preserving original stores; integrate before first checkpoint creation and preserve borrowed history savers.
- [ ] Reduce repeated status JSON loads per turn and avoid full-state hashing allocations; retain independent mutable snapshots where execution relies on them.
- [ ] Reduce scientific request duplication using lossless transport tables, field-aware evidence reads and targeted response repair; retain complete candidate pools, required evidence and full original-fact validation.
- [ ] Run fake-model approvals/history/resume/configuration/status tests and memory/I/O/transport benchmarks. Install with baseline guard; migrate only stopped projects with verified backups and checksums. Never restart real services.

Review focus: nested checkpoint mutation and deletion; UUID keys and binary values; historical read paging and pending writes; source or service concurrency; candidate coverage and stale snapshot hashes; failures must not discard history or execute actions.




