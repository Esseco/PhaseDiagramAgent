# DFT sampling and previews

Per search round, single points are capped at 100 structures and 3000 relative cost
unless explicitly configured otherwise. Registered same-model/same-generation
single points count toward both caps. Global/stage budgets remain separate.
Existing confirmed configuration and production task state are not edited by this
change; missing new per-round settings use these approved project defaults.

Recommendations reuse select_dft_candidates: hull score, available QBC, phase
representatives, Na composition spread and independent audit. Missing QBC is omitted,
never represented as zero uncertainty. Sufficient-size proposals missing available
phases require revision. Budget cannot guarantee every phase is affordable; missing
coverage requires an explicit revised plan, not silent selection of cheap candidates.

Previews display selected structure IDs, x_Na_per_O2, identified phase and current
Ehull eV/atom, then one QBC availability count. Candidate costs use actual atom counts
from diagram composition when the diagram lacks an explicit atom_count.

Agent context includes real per-candidate single-point/optimization costs. An
unapproved draft rejected for DFT budget may be replaced by a visibly labeled
budgeted single-point draft using the same phase/Na sampler. It is validated again,
requires fresh human approval, and never alters an approved plan or raises budgets.
Global, stage and same-round prior usage are deducted before draft sampling.
