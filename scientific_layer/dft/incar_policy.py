"""Confirmed INCAR overrides for the layered-oxide search project."""


def layered_oxide_incar(settings=None):
    return {**(settings or {}), "ALGO": "Normal", "AMIX": 0.2,
            "BMIX": 0.0001, "AMIX_MAG": 0.8, "BMIX_MAG": 0.0001}
