"""Branch、具体结构与计算历史台账；结构算法拆分在其他四个文件。"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from scientific_layer.structures.boundary_utils import (
    compact_json,
    det_H,
    json_safe,
    normalize_H,
    normalize_fraction,
    validate_boundary,
)


class PhaseDataManager:
    """只负责稳定编号、台账、计算结果和保存加载。"""

    SCHEMA_VERSION: ClassVar[int] = 2
    STAGES: ClassVar[tuple[str, ...]] = (
        "simple_check",
        "relax_and_feature",
        "deep_search",
        "dft_single_point",
        "dft_relax",
    )
    STAGE_LABELS: ClassVar[dict[str, str]] = {
        "simple_check": "简单检查",
        "relax_and_feature": "弛豫+特征识别",
        "deep_search": "深度搜索",
        "dft_single_point": "DFT 单点能",
        "dft_relax": "DFT 弛豫",
    }

    def __init__(self, boundary: dict[str, Any], system_config: dict[str, Any] | None = None) -> None:
        workflow = (system_config or {}).get("calculation_workflow", {})
        stage_specs = workflow.get("stages") or [
            {"name": name, "label": self.STAGE_LABELS[name]} for name in self.STAGES
        ]
        self.stages = tuple(item["name"] for item in stage_specs)
        self.stage_labels = {item["name"]: item.get("label", item["name"]) for item in stage_specs}
        self.data: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "boundary": validate_boundary(boundary),
            "system_config": json_safe(system_config),
            "branches": {},
            "structures": {},
        }

    @property
    def boundary(self) -> dict[str, Any]:
        return self.data["boundary"]

    def add_branch(
        self,
        *,
        P: str,
        H: Any,
        x: int | float | str | Fraction,
        T: Any,
        composition: Any = None,
    ) -> str:
        key = {
            "P": str(P).upper(),
            "H": normalize_H(H),
            "x": normalize_fraction(x),
            "T": json_safe(T),
        }
        return self.add_branch_record(key, composition=composition)

    def add_branch_record(self, parameters: dict[str, Any], *, composition: Any = None) -> str:
        """按体系配置中的 branch 字段登记；默认仍为 P/H/x/T。"""
        schema = ((self.data.get("system_config") or {}).get("branch_schema") or {}).get("fields", ["P", "H", "x", "T"])
        missing = set(schema) - parameters.keys()
        if missing:
            raise ValueError(f"branch 参数缺少字段：{sorted(missing)}")
        key = {name: json_safe(parameters[name]) for name in schema}
        branch_id = _stable_id("B", key)
        branches = self.data["branches"]
        if branch_id not in branches:
            branches[branch_id] = {
                "branch_id": branch_id,
                **key,
                "det_H": det_H(key["H"]) if "H" in key else None,
                "composition": json_safe(composition),
                "structure_ids": [],
            }
        else:
            existing = {name: branches[branch_id][name] for name in schema}
            if existing != key:
                raise RuntimeError(f"branch 编号冲突：{branch_id}")
            _fill_if_missing(branches[branch_id], "composition", composition)
        return branch_id

    def identify_branch(
        self,
        structure: Any,
        *,
        phase_references: dict[str, Any],
        register: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        """兼容入口；核心识别逻辑在 branch_identifier.py。"""
        from scientific_layer.structures.identify_branch import identify_branch_parameters

        result = identify_branch_parameters(
            structure, self.boundary, phase_references=phase_references, **options
        )
        result["branch_id"] = (
            self.add_branch(
                P=result["P"],
                H=result["H"],
                x=result["x"],
                T=result["T"],
                composition=result["composition"],
            )
            if register
            else None
        )
        return result

    def generate_branch_structure(self, **parameters: Any) -> Any:
        """兼容入口；核心生成逻辑在 structure_generator.py。"""
        from scientific_layer.structures.generate_branch_structure import generate_branch_structure

        return generate_branch_structure(self.boundary, **parameters)

    def add_structure(
        self,
        *,
        branch_id: str,
        arrangement: Any,
        composition: Any = None,
        source_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        branch = self._get_branch(branch_id)
        arrangement = json_safe(arrangement)
        structure_id = _stable_id(
            "S", {"branch_id": branch_id, "arrangement": arrangement}
        )
        structures = self.data["structures"]
        if structure_id not in structures:
            structures[structure_id] = {
                "structure_id": structure_id,
                "branch_id": branch_id,
                "arrangement": arrangement,
                "composition": json_safe(composition)
                if composition is not None
                else branch["composition"],
                "source_path": str(source_path) if source_path is not None else None,
                "metadata": json_safe(metadata),
                "config_features": None,
                "stage_history": {stage: [] for stage in self.stages},
            }
            branch["structure_ids"].append(structure_id)
            branch["structure_ids"].sort()
        else:
            structure = structures[structure_id]
            _fill_if_missing(structure, "composition", composition)
            _fill_if_missing(structure, "source_path", source_path)
            _fill_if_missing(structure, "metadata", metadata)
        return structure_id

    def record_result(
        self,
        *,
        structure_id: str,
        stage: str,
        converged: bool | None = None,
        convergence_info: dict[str, Any] | None = None,
        mlip_name: str | None = None,
        mlip_version: str | None = None,
        mlip_relax_version: str | None = None,
        mlip_energy: float | None = None,
        dft_code: str | None = None,
        dft_version: str | None = None,
        dft_settings: dict[str, Any] | None = None,
        dft_energy: float | None = None,
        energy_unit: str | None = None,
        ehull: float | None = None,
        ehull_unit: str | None = None,
        hull_reference_version: str | None = None,
        result_path: str | Path | None = None,
        recorded_at: str | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        structure = self._get_structure(structure_id)
        stage = self._normalize_stage(stage)
        result = {
            "stage": stage,
            "stage_label": self.stage_labels[stage],
            "converged": converged,
            "convergence_info": json_safe(convergence_info),
            "mlip_name": mlip_name,
            "mlip_version": mlip_version,
            "mlip_relax_version": mlip_relax_version,
            "mlip_energy": _finite(mlip_energy, "mlip_energy"),
            "dft_code": dft_code,
            "dft_version": dft_version,
            "dft_settings": json_safe(dft_settings),
            "dft_energy": _finite(dft_energy, "dft_energy"),
            "energy_unit": energy_unit,
            "ehull": _finite(ehull, "ehull"),
            "ehull_unit": ehull_unit,
            "hull_reference_version": hull_reference_version,
            "result_path": str(result_path) if result_path is not None else None,
            "recorded_at": recorded_at,
            "notes": notes,
            "metadata": json_safe(metadata),
        }
        result_id = _stable_id("R", {"structure_id": structure_id, **result})
        result["result_id"] = result_id
        history = structure["stage_history"][stage]
        if not any(item["result_id"] == result_id for item in history):
            history.append(result)
        return result_id

    def branch_table(self) -> list[dict[str, Any]]:
        from data_layer.ledger.coverage_report import branch_table

        return branch_table(self.data)

    def structure_table(self) -> list[dict[str, Any]]:
        from data_layer.ledger.coverage_report import structure_table

        return structure_table(self.data, self.stages, self.stage_labels)

    def coverage(self) -> list[dict[str, Any]]:
        from data_layer.ledger.coverage_report import coverage

        return coverage(self.data, self.stages, self.stage_labels)

    def save_coverage_csv(self, path: str | Path) -> Path:
        from data_layer.ledger.coverage_report import save_coverage_csv

        return save_coverage_csv(self.coverage(), path)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> PhaseDataManager:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(f"不支持的 schema_version：{data.get('schema_version')!r}")
        manager = cls(data["boundary"], system_config=data.get("system_config"))
        manager.data = data
        manager._validate_loaded_data()
        return manager

    def extract_config_features(self, structure_id: str) -> dict[str, Any]:
        self._get_structure(structure_id)
        raise NotImplementedError("自动构型特征尚未实现")

    def find_duplicates(self) -> list[list[str]]:
        raise NotImplementedError("结构去重尚未实现")

    def _get_branch(self, branch_id: str) -> dict[str, Any]:
        try:
            return self.data["branches"][branch_id]
        except KeyError as error:
            raise KeyError(f"未知 branch：{branch_id}") from error

    def _get_structure(self, structure_id: str) -> dict[str, Any]:
        try:
            return self.data["structures"][structure_id]
        except KeyError as error:
            raise KeyError(f"未知结构：{structure_id}") from error

    def _normalize_stage(self, stage: str) -> str:
        aliases = {
            **{item: item for item in self.stages},
            **{label: key for key, label in self.stage_labels.items()},
        }
        if stage not in aliases:
            raise ValueError(f"未知阶段 {stage!r}")
        return aliases[stage]

    def _validate_loaded_data(self) -> None:
        for branch in self.data["branches"].values():
            branch.setdefault("det_H", det_H(branch["H"]) if "H" in branch else None)
        for structure in self.data["structures"].values():
            self._get_branch(structure["branch_id"])
            structure.setdefault("config_features", None)
            for stage in self.stages:
                structure["stage_history"].setdefault(stage, [])


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def _fill_if_missing(record: dict[str, Any], field: str, value: Any) -> None:
    if record.get(field) is None and value is not None:
        record[field] = json_safe(value)
