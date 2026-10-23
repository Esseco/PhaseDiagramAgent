"""Bounded periodic geometry observations, never a structure acceptance filter."""

from hashlib import sha256
from pathlib import Path
from statistics import median

LAYERED_PHASES = {"O1", "O2", "O3", "P2", "P3", "OP2", "OP4"}


def inspect_geometry(row):
    import numpy as np
    from pymatgen.core import Structure

    path = Path(row["structure_path"]) if row.get("structure_path") else None
    accessible = path is not None and path.is_file()
    data = row.get("structure")
    if isinstance(data, dict):
        structure = Structure.from_dict(data)
    elif accessible:
        structure = Structure.from_file(str(path))
    else:
        raise ValueError("缺少可读取的最终结构；几何质量未知")
    if not structure.is_ordered:
        raise ValueError("无序占位结构尚未支持几何初筛")
    observed = structure.composition.as_dict()
    expected = row.get("composition") or {}
    composition_matches = all(
        abs(float(observed.get(k, 0)) - float(expected.get(k, 0))) < 1e-6
        for k in set(observed) | set(expected)
    )
    result = {"composition_matches": composition_matches, "file_accessible": accessible}
    digest = row.get("structure_sha256")
    if accessible and isinstance(digest, str) and len(digest) == 64:
        result["file_digest_matches"] = sha256(path.read_bytes()).hexdigest() == digest
    oxygen = float(observed.get("O", 0))
    if oxygen <= 0:
        raise ValueError("最终结构不含氧，无法按 O2 比较体积")
    result["volume_per_O2_A3"] = float(structure.volume / (oxygen / 2))
    symbols = [site.specie.symbol for site in structure]
    centers = [i for i, name in enumerate(symbols) if name not in {"Na", "Li", "K", "O"}]
    if not centers:
        raise ValueError("缺少可检查的氧化物骨架阳离子")
    # Radius is a neighbor-search bound, not an allowed/forbidden bond length.
    ci, ni, _, distances = structure.get_neighbor_list(4.0)
    shells = []
    for center in centers:
        neighbors = sorted(
            float(d)
            for c, n, d in zip(ci, ni, distances)
            if c == center and symbols[n] == "O" and d > 1e-8
        )
        if len(neighbors) < 6:
            raise ValueError("4 Å 搜索范围内氧近邻不足六个，局部环境需另行核查")
        shells.append(neighbors)
    sixth = median(shell[5] for shell in shells)
    cutoff = 1.15 * sixth
    counts = [sum(distance <= cutoff for distance in shell) for shell in shells]
    result.update(
        shortest_tm_o_A=min(shell[0] for shell in shells),
        median_nearest_tm_o_A=median(shell[0] for shell in shells),
        coordination_cutoff_A=cutoff,
        coordination_rule="1.15 * median sixth TM-O distance; periodic-image neighbors",
        expected_octahedral=row.get("phase") in LAYERED_PHASES,
        non_six_fraction=sum(count != 6 for count in counts) / len(counts),
        non_six_count=sum(count != 6 for count in counts),
        framework_cations=len(centers),
    )
    if not all(np.isfinite(v) for v in (sixth, cutoff, result["shortest_tm_o_A"])):
        raise ValueError("几何距离含非有限值")
    return result
