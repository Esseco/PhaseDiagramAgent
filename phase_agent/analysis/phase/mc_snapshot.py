"""Select the model-specific hull used for MC energy comparisons."""


def is_current_model_only_hull(state, diagram, version):
    refresh = state.get("model_refresh") or {}
    return (
        refresh.get("hull_source_policy") == "current_model_only_v1"
        and refresh.get("new_model_version") == version
        and diagram.get("status") == "completed"
        and diagram.get("model_version") == version
        and diagram.get("energy_policy") == "current_model_only_v1"
        and diagram.get("mixed_model_versions") is False
        and set(diagram.get("source_model_versions") or []) == {version}
        and bool(diagram.get("entries"))
        and all((e.get("source_version") or e.get("model_version")) == version
                for e in diagram.get("entries") or [])
    )


def mc_model_snapshot(state, version):
    overall = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if overall.get("status") == "invalidated" and overall.get("model_version") == version:
        raise ValueError("current_model_hull_invalidated")
    if state.get("mlip_energy_policy") == "latest_valid_per_structure_v1":
        if is_current_model_only_hull(state, overall, version):
            rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
            if (
                type(rounds.get(version)) is not int
                or rounds[version] < 1
                or len(set(rounds.values())) != len(rounds)
            ):
                raise ValueError("current_model_hull_model_rounds_invalid")
            return overall
        if (
            overall.get("status") != "completed"
            or overall.get("model_version") != version
            or overall.get("energy_policy") != "latest_valid_per_structure_v1"
        ):
            raise ValueError("latest_structure_hull_required")
        rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
        sources = set(overall.get("source_model_versions") or [])
        if (
            not sources
            or len(set(rounds.values())) != len(rounds)
            or any(type(rounds.get(source)) is not int or rounds[source] < 1 for source in sources)
        ):
            raise ValueError("latest_structure_hull_model_rounds_missing")
        if any(
            (entry.get("source_version") or entry.get("model_version")) not in sources
            for entry in overall.get("entries") or []
        ):
            raise ValueError("latest_structure_hull_source_mismatch")
        return overall
    diagram = (state.get("phase_diagrams_by_model") or {}).get(version) or overall
    if not diagram:
        return {}
    sources = set(diagram.get("source_model_versions") or [])
    if (
        diagram.get("model_version") != version
        or diagram.get("mixed_model_versions") is True
        or sources - {version}
        or any(
            (e.get("source_version") or e.get("model_version")) not in (None, version)
            for e in diagram.get("entries") or []
        )
    ):
        raise ValueError("current_model_independent_hull_required")
    return diagram
