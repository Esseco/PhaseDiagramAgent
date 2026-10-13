"""Shared conversational default; the LLM interprets explicit user overrides."""

RELAX_MC_PREFERENCE = """For the unified Branch search workflow, default to
the one-upload Relax+MLIP-MC package: operation prepare_relax_mc_batch, tool
prepare_local_batch_files, mode=relax_mc_inputs, with explicit recommended existing
branch IDs. Generate/register Branch first, then diagnose and recommend the useful
subset and show workload before existing approval. Do not prepare separate Relax
or allocate MC for the same batch by default. If the user explicitly asks to call
or allocate Relax and MLIP-MC separately, honor that stepwise preference, including
saved explicit human constraints, until the user changes it. A plain request to
continue does not switch the preference. Existing pending tasks, approved actions,
registered packages and recovered standalone Relax/MC work keep their original
recovery path; never convert them, reset MC rounds or duplicate tasks. A package
only prepares local files after approval; upload and one Slurm submission remain
manual. Default packaging does not authorize execution or submission. After return,
choose more Branch search or DFT using verified phase diagram/QBC evidence.
"""
