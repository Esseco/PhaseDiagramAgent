"""Keep descriptive species and constraints consistent with the search boundary."""

from copy import deepcopy
from phase_agent.science.structures.boundary_utils import allowed_phases


def synchronize_system_scope(config):
    result = deepcopy(config)
    system = result.get("system") or {}
    boundary = system.get("boundary")
    if not isinstance(boundary, dict):
        return result
    ratio = boundary.get("TM_ratio")
    if not isinstance(ratio, dict) or not ratio:
        return result
    species = sorted(ratio)
    phases = sorted(allowed_phases(boundary.get("P", [])))
    system.setdefault("constraints", {}).update(TM_ratio=deepcopy(ratio), phases=phases)
    system.setdefault("species", {})["substitutional"] = species
    system.setdefault("occupancy_rules", {})["substitutional_sites"] = "/".join(species)
    inactive = set()
    roles = (system.get("configuration_space") or {}).get("roles") or {}
    if len(species) == 1 and roles and roles.get("T") is None:
        # A single species has a unique occupancy; there is no choice to ask for.
        roles["T"] = "fixed"
        system["configuration_space"]["fixed_T_source"] = "phase_reference"
    if roles.get("T") == "fixed" and (system.get("branch_schema") or {}).get("fields") == [
        "P",
        "H",
        "x",
        "T",
    ]:
        system["branch_schema"]["fields"] = ["P", "H", "x"]
    for variable, strategy in {
        "T": "tm_ordering",
        "P": "competing_phase",
        "x": "composition",
        "H": "periodic_extension",
    }.items():
        if roles.get(variable) == "fixed":
            inactive.add(strategy)
    if roles.get("T") == "fixed":
        inactive.add("tm_mutation")
    if len(species) == 1:
        inactive.update({"tm_ordering", "tm_mutation"})
    if len(phases) == 1:
        inactive.add("competing_phase")
    for section, key in (
        (system.get("generation") or {}, "enabled_strategies"),
        (result.get("generation_actions") or {}, "enabled"),
    ):
        if isinstance(section.get(key), list):
            section[key] = [name for name in section[key] if name not in inactive]
    return result
