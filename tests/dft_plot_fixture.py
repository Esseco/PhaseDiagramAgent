"""Small same-frame hull evidence for tests that exercise DFT review."""

from hashlib import sha256


def add_dft_plot_evidence(state, root):
    model = state.get("active_model_version") or "m1"
    state["active_model_version"] = model
    state["analysis_output_directory"] = str(root / "analysis_outputs")
    state["upload_layout"] = {"model_rounds": {model: 1}}
    entries = [
        {"composition": {"Fe": 1, "O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": False},
        {"composition": {"Na": 2, "Fe": 1, "O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": False},
    ]
    for index, candidate in enumerate(state.get("qbc_candidates") or []):
        path = root / f"dft-plot-frame-{index}.vasp"
        path.write_text(f"synthetic frame {index}", encoding="utf-8")
        digest = sha256(path.read_bytes()).hexdigest()
        candidate.update(structure_path=str(path), structure_sha256=digest)
        if isinstance(candidate.get("dft_sampling_evidence"), dict):
            candidate["dft_sampling_evidence"]["geometry_key"] = digest
        entries.append({
            "composition": {"Na": 1, "Fe": 1, "O": 2},
            "eform_per_O2": -0.1 - index * 0.01,
            "phase": candidate.get("phase") or "O3",
            "is_stable": False,
            "structure_sha256": digest,
        })
    state["phase_diagrams"] = {"mlip": {
        "status": "completed", "model_version": model, "version": "synthetic-hull",
        "entries": entries,
    }}
