"""Na-content formation-energy hull, following the project's cal_ehull notebook."""
from collections import defaultdict
from math import isclose
from pymatgen.core import Composition

from analysis_layer.phase.check_na_layer_uniformity import check_na_layer_uniformity


def build_composition_hull_entries(records):
    prepared, frameworks = [], set()
    for row in records:
        composition = Composition(row["composition"])
        oxygen = float(composition["O"])
        if oxygen <= 0:
            raise ValueError("相图要求含氧结构，无法归一化到 eV/O2")
        units = oxygen / 2
        framework = tuple(sorted((str(element), round(float(amount) / units, 9))
                                 for element, amount in composition.items()
                                 if str(element) not in {"Na", "O"}))
        frameworks.add(framework)
        x = float(composition["Na"]) / units
        prepared.append((row, composition, units, x,
                         float(row["normalized_total_energy"]) / units))
    if len(frameworks) != 1:
        raise ValueError("检测到 TM/O2 组分不一致；当前相图只允许 Na 含量变化，请分开处理")
    grouped = defaultdict(list)
    for item in prepared:
        grouped[round(item[3], 9)].append(item)
    xs = sorted(grouped)
    if len(xs) < 2:
        raise ValueError("至少需要两个不同 Na 含量才能确定相图端点")
    left, right = xs[0], xs[-1]
    e_left = min(item[4] for item in grouped[left])
    e_right = min(item[4] for item in grouped[right])
    def eform(x, energy):
        ratio = (x - left) / (right - left)
        return energy - (1 - ratio) * e_left - ratio * e_right
    lowest = {x: min(eform(x, item[4]) for item in grouped[x]) for x in xs}
    # Lower convex envelope of the lowest-energy structure at each Na content.
    hull = []
    for x in xs:
        point = (x, lowest[x])
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(point)
    entries = []
    for row, composition, units, x, energy_per_o2 in prepared:
        formed = eform(x, energy_per_o2)
        segment = next(((a, b) for a, b in zip(hull, hull[1:])
                        if a[0] - 1e-9 <= x <= b[0] + 1e-9), None)
        a, b = segment or (hull[-2], hull[-1])
        hull_formed = a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
        gap_per_o2 = max(0.0, formed - hull_formed)
        gap = gap_per_o2 * units / composition.num_atoms
        check = check_na_layer_uniformity(row.get("structure_path"), row["composition"])
        entries.append({"record_id": row.get("record_id"), "structure_id": row.get("structure_id"),
            "structure_path": row.get("structure_path"), "phase": row.get("phase"),
            "phase_identification_status": row.get("phase_identification_status"),
            "structure_sha256": row.get("structure_sha256"),
            "composition": row["composition"], "original_energy": row["original_energy"],
            "normalized_total_energy": row["normalized_total_energy"],
            "x_Na_per_O2": x, "endpoint_x_min": left, "endpoint_x_max": right,
            "endpoint_min_energy_per_O2": e_left, "endpoint_max_energy_per_O2": e_right,
            "eform_per_O2": formed, "hull_eform_per_O2": hull_formed,
            "ehull": gap, "ehull_unit": "eV/atom", "is_stable": gap_per_o2 <= 1e-8,
            "is_composition_ground_state": isclose(formed, lowest[round(x, 9)], rel_tol=0, abs_tol=1e-8),
            "energy_per_O2": energy_per_o2,
            "hull_energy_per_O2": energy_per_o2 - gap_per_o2,
            "ehull_per_O2": gap_per_o2,
            "source_version": row.get("source_version"),
            "na_layer_uniform": check["uniform"], "na_layer_status": check["status"],
            "na_layer_rule": check["rule"]})
    return entries
