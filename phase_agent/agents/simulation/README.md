# Simulation Agent

agent.py plans one Strategic proposal, validates the frozen plan and inspects actual tool observations; contracts.py defines strict JSON plans and evidence-bound diagnoses. The role uses neutral gray #616161.

Existing tools provide MC search, Relax, DFT, MLIP inference/training and hull reconstruction. Existing workflow and remote adapters prepare inputs, submit/observe authorized HPC jobs, collect/check outputs and record resources. Simulation adds bounded model planning and diagnosis, not a parallel scheduler or new scientific algorithm.

Planning never executes tools. Approved dispatch invokes one registered tool once, preserving targets, parameters, budget and receipt identity. Diagnostics cannot retry; Strategic must propose an existing recovery action. Confirmed-policy bounded retries reuse restart_failed_task; parameter changes or expanded limits require humans. Ordinary old callables and frozen legacy approvals retain their original protocol.
