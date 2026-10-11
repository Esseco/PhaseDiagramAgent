# Lifecycle wait boundary configuration fix

A new request at wait_for_project_input is the start of a new lifecycle iteration, not restoration of the old invocation. The runner now passes the existing reconcile signal into that wait boundary and lets initialize load the confirmed configuration. It does not pre-initialize or compare the previous request's configuration in that case.

The exception is restricted to the durable project thread, a distinct request ID, the exact native wait node and an interrupt. Same-request results remain cached; failures at in-flight read/analysis/action nodes still enforce configuration equality. Gateway message replay protection is unchanged.

Verified with 7 focused regressions and a read-only backup of the actual NaCoO2 checkpoint. The latter loaded config-0009-62857b28cdf6001e exactly once with fake callbacks and zero scientific actions. Source deployed with baseline hash and backup; idle NaCoO2 service restarted and healthy. Live checkpoints and receipts preserved. The failed old message stays replay-protected; send a new message to continue.

Whole-package ruff check passed; runner format check passed. Whole-package format check identifies an unrelated pre-existing difference in graphs/dialogue/nodes.py, not changed by this fix. An unrelated async Studio shutdown test hung in this Windows environment and was interrupted; the targeted suite was rerun independently and passed.
