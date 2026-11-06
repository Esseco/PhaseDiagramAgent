"""Compare transition-metal ordering under periodic cell deformation."""


def tm_order_preserved(original, final):
    import numpy as np
    from ase.geometry import get_distances
    from scipy.optimize import linear_sum_assignment

    non_tm = {"Na", "Li", "K", "Rb", "Cs", "O", "F", "S", "Se"}
    source_symbols = original.get_chemical_symbols()
    final_symbols = final.get_chemical_symbols()
    source_ids = [i for i, symbol in enumerate(source_symbols) if symbol not in non_tm]
    final_ids = [i for i, symbol in enumerate(final_symbols) if symbol not in non_tm]
    if len(source_ids) != len(final_ids):
        return False
    if len(source_ids) < 2:
        return all(source_symbols[i] == final_symbols[j] for i, j in zip(source_ids, final_ids))
    source_positions = original.get_scaled_positions(wrap=True)[source_ids] @ final.cell.array
    final_positions = final.get_scaled_positions(wrap=True)[final_ids] @ final.cell.array
    _, distances = get_distances(source_positions, final_positions, cell=final.cell, pbc=final.pbc)
    if not np.isfinite(distances).all():
        return False
    rows, columns = linear_sum_assignment(distances)
    return all(source_symbols[source_ids[i]] == final_symbols[final_ids[j]] for i, j in zip(rows, columns))
