"""Example manual feature: number of atoms."""


def feature_atom_count(structure, context=None) -> dict:
    return {"atom_count": float(len(structure))}


FEATURE_DEFINITION = {"name": "atom_count", "version": "1", "function": feature_atom_count}
