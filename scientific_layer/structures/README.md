# Structure tools

Structure generation and identification tools live here. They use the configured
boundary and phase references but do not silently modify confirmed search settings.

`enumerate_legal_frameworks.py` lists frames already allowed by the boundary.
`enumerate_layered_oxide_supercells.py` is an upstream proposal tool for layered
oxides: it enumerates in-plane supercells from a phase mother structure, filters
them by selected `P_small` containment and minimum periodic distance, then returns
candidate `H` matrices and exact composition fractions. It only returns candidates;
the user chooses which matrices, if any, to add to the confirmed `boundary.H`.

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
fraction strings. The generated candidates do not alter `boundary.H` or launch
any scientific calculation.
