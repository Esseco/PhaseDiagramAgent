"""Persistent, version-isolated ledger for relaxed branch energy pools."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


def energy_pool_identity(*, system_id: str, mlip_version: str, energy_basis: str) -> str:
    if not all((system_id, mlip_version, energy_basis)):
        raise ValueError("system_id, mlip_version and energy_basis are required")
    return f"{system_id}::{mlip_version}::{energy_basis}"


def load_branch_energy_pools(path, *, system_id=None, mlip_version=None, energy_basis=None) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    document = json.loads(source.read_text(encoding="utf-8"))
    pools = list((document.get("pools") or {}).values())
    if system_id is not None:
        pools = [pool for pool in pools if pool.get("system_id") == system_id]
    if mlip_version is not None:
        pools = [pool for pool in pools if pool.get("model_version") == mlip_version]
    if energy_basis is not None:
        pools = [pool for pool in pools if pool.get("energy_basis") == energy_basis]
    return deepcopy(pools)


def save_branch_energy_pool(pool: dict, path) -> str:
    identity = energy_pool_identity(system_id=pool.get("system_id"),
                                    mlip_version=pool.get("model_version"),
                                    energy_basis=pool.get("energy_basis"))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "pools": {}}
    if output.is_file():
        document = json.loads(output.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("unsupported branch energy pool ledger schema")
    key = f"{identity}::{pool['version']}"
    document.setdefault("pools", {})[key] = deepcopy(pool)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    return key
