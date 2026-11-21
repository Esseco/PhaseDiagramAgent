"""Confirmed INCAR overrides for the layered-oxide search project."""


def layered_oxide_incar(settings=None, *, static=False):
    return {
        **(settings or {}),
        "NELM": (settings or {}).get("NELM", 200),
        **({"ENCUT": 520} if static else {}),
        "ALGO": "Normal",
        "AMIX": (settings or {}).get("AMIX", 0.2),
        "BMIX": (settings or {}).get("BMIX", 0.0001),
        "AMIX_MAG": (settings or {}).get("AMIX_MAG", 0.8),
        "BMIX_MAG": (settings or {}).get("BMIX_MAG", 0.0001),
    }
