"""Single preview boundary for both initial and human-revised DFT actions."""
from copy import deepcopy
from execution_layer.workflows.preview_dft_inputs import preview_dft_inputs


def attach_dft_preview(action, state, config):
    if action.get("tool") != "select_dft_candidates":
        return deepcopy(action), None
    clean = deepcopy(action)
    clean.setdefault("parameters", {}).pop("dft_input_preview", None)
    preview, error = preview_dft_inputs(clean, state, config)
    if error:
        return None, error
    clean["parameters"]["dft_input_preview"] = preview
    clean["budget"] = preview["relative_cost"]
    return clean, None
