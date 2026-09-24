"""Example manual feature: relaxed cell volume per atom."""


def feature_volume_per_atom(structure, context=None) -> dict:
    if not len(structure):
        raise ValueError("empty structure")
    return {"volume_per_atom": float(structure.volume) / len(structure)}


FEATURE_DEFINITION = {"name": "volume_per_atom", "version": "1", "function": feature_volume_per_atom}
