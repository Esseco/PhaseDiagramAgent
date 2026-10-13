# GPU grouping and Relax–MC progression

Configured defaults are 100 Relax structures and 20 MC tasks per GPU allocation. Incompatible models, resources or operation scopes remain separate; the last compatible group can be smaller. Batch capacity is configuration, with no hidden ten-task clamp. Manual/package and direct Slurm MLIP jobs execute task groups serially in one allocation; DFT task dispatch is unchanged.

After Relax results are verified, feedback identifies phases, updates the active-model formal hull and computes Ehull. The first MC requires that valid same-model hull. After each MC return the hull is rebuilt before the next allocation. At most three MC segments run under the approved continuation/stop rules. Missing hull endpoints, invalid model binding or failed results pause rather than inventing an Ehull or submitting duplicate work.

A package prepared before this change retains its frozen runtime; already submitted packages are not rewritten. New pending proposals are revised with updated grouping/checksums and remain unapproved. Services load changes on the user's manual restart.
