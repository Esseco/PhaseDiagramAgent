# Structure tools

Structure generation and identification tools live here. They use the configured
boundary and phase references but do not silently modify confirmed search settings.

When `configuration_space.roles.T=fixed`, generation and Na/vacancy initialization copy the actual TM occupation of each user-supplied phase reference into its supercell. Random TM assignment and TM-ordering mutation are skipped. An ambiguous/disordered or ratio-incompatible reference raises an error instead of being repaired silently.

For a partially occupied Na composition, `initialize_branch_structures.py` now
keeps all candidate Na sites in the disordered template, ranks orderings
by Ewald electrostatic score, filters out structures with an empty Na layer,
and randomly selects up to three distinct structures from the ten lowest
valid orderings with a fixed seed. The score is stored as a ranking diagnostic,
not as an MLIP/DFT energy. Endpoints have no Na/vacancy ranking. A failed
ordering is reported per branch in `initialization_failures`; unscored legacy
intermediate structures are not silently treated as electrostatically ranked.
Relax preparation reuses those saved random picks without a second energy sort.
For every nonzero-Na branch, the same initialization step also saves a
`full_na_structure.vasp` template with identical P, H and T. MC task export copies
both `initial.vasp` and this full-Na site template into the task directory and uses
relative paths. Na0 endpoints are marked as having no Na/vacancy MC search variable.
If charge assignment or ordering fails, the same disordered Na-site template
is retried once with `Na=+1, O=-2, Fe=+3, Mn=+4`. The selected scheme is saved
as `electrostatic_charge_scheme`; scores from different schemes must not be
compared across branches. If both attempts fail, the branch and both errors
are recorded instead of selecting random structures.

`enumerate_legal_frameworks.py` lists frames already allowed by the boundary.
`enumerate_layered_oxide_supercells.py` is an upstream proposal tool for layered
oxides: it enumerates in-plane supercells from a phase mother structure, filters
them by selected `P_small` containment and minimum periodic distance, then returns
candidate `H` matrices and exact composition fractions. It only returns candidates;
the config importer can expand selected bounds into confirmed `boundary.H`.

The layered-oxide module exposes two recommended matrices through
`LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS`. Pass one, both, or additional custom
matrices using `p_small_list`. If multiple matrices are selected, every output
supercell must contain all of them.

```python
from scientific_layer.structures.enumerate_layered_oxide_supercells import (
    LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS as p_small_options,
    enumerate_layered_oxide_supercells,
)

candidates = enumerate_layered_oxide_supercells(
    phase="O3",
    mother_structure="/path/to/O3.vasp",
    sizes=[4, 6, 8, 10],  # supplied by the user
    p_small_list=[p_small_options[0]],
    min_distance=2.0,
)
```

Run this in `py1`, where `icet`, ASE, and pymatgen are available. Results include
`phase`, `S`, canonical 3×3 `H`, `det_H`, `R_cut` in Å, and `x_list` as exact
fraction strings. Calling the enumeration function alone does not alter
`boundary.H` or launch any scientific calculation. In an editable layered-oxide
config, set `boundary.P.at_x` (for example `0: P3`, `1: O3`) and
`boundary.P.intermediate`, then edit `H_generation.size_min`, `size_max`,
`size_step`, `min_distance_angstrom`, and at least one recommended index or
custom containment matrix. Importing the JSON expands H for each phase and
stores the explicit matrices in the confirmed snapshot. Old explicit P/H
configs remain supported with `H_generation.enabled=false`.
# MC 满 Na 位点模板

MC 的 `initial.vasp` 使用对应 branch 最低能的 Relax 结果；`full_na_structure.vasp` 使用相 `P` 的母结构按实际 `H` 扩胞，并落实相同的 `T`，不会从 Relax 结构补 Na。若该相母结构没有 Na 位点，可在已确认配置的 `system.mc_full_na_templates` 中按相提供带 Na 位点的参考文件路径，例如 `{"O1": ".../O1_Na_sites.vasp"}`。该文件须与 O1 母结构具有相同晶格及非 Na 框架位点；没有明确映射时拒绝准备 MC 输入。Na0 branch 不执行 Na/V MC。
