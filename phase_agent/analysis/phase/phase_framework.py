"""Conservative layered-framework topology evidence, independent of cell orientation."""

from hashlib import sha256
from pathlib import Path

from Process_AL_MC.structure_check import (
    RULE,
    LayeredFrameworkReference,
    periodic_framework,
)

__all__ = ["inspect_framework", "periodic_framework"]


def _read(path):
    from pymatgen.core import Structure

    source = Path(path)
    raw = source.read_bytes()
    structure = Structure.from_str(raw.decode("utf-8"), fmt="poscar")
    if not structure.is_ordered:
        raise ValueError("无序占位不支持自动骨架判定")
    return structure, sha256(raw).hexdigest()


def inspect_framework(row, initial_path):
    """A sufficient domain-rejection rule, never a general crystal validity classifier.

    Require an intact registered starting slab. A connected TM region must lose
    its 2D network or persistently lose its initial O6 coordination at three
    fixed cutoffs. Ambiguous changes and changed binding remain uncertain.
    """
    result = {
        "rule": RULE,
        "status": "uncertain",
        "auto_confirmed": False,
        "meaning": "是否保持本项目层状TM–O骨架；不判定该重构相的物理能量错误",
    }
    try:
        digest = row.get("structure_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("缺少最终结构的能量记录校验和")
        if not row.get("structure_path") or not initial_path:
            raise ValueError("缺少已绑定的初始或最终结构路径")
        final, final_hash = _read(row["structure_path"])
        initial, initial_hash = _read(initial_path)
        if final_hash != digest:
            raise ValueError("最终结构校验和与能量记录不符")
        if [str(site.specie) for site in initial] != [str(site.specie) for site in final]:
            raise ValueError("初末原子序号/元素无法对应，不能确认局部骨架变化")
        expected = row.get("composition") or {}
        if not expected or any(
            abs(float(s.composition.get(k, 0)) - float(expected.get(k, 0))) > 1e-6
            for s in (initial, final)
            for k in set(expected) | set(s.composition.as_dict())
        ):
            raise ValueError("初末结构组成与能量记录不符")
        reference = LayeredFrameworkReference(initial)
        result.update(reference.inspect(final))
        result.update(
            initial_structure_path=str(initial_path),
            initial_sha256=initial_hash,
            final_sha256=final_hash,
        )
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        result["reason"] = str(error)
    return result
