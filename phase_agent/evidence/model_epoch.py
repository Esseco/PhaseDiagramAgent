"""Human-readable model generations, independent of calculation sub-rounds."""


def model_epoch(state, version):
    layout = state.get("upload_layout") or {}
    scientific_epochs = layout.get("scientific_epochs") or {}
    if version in scientific_epochs:
        scientific = scientific_epochs[version]
        return f"epoch{scientific}" if type(scientific) is int and scientific >= 0 else None
    number = (layout.get("model_rounds") or {}).get(version)
    if isinstance(number, int) and not isinstance(number, bool) and number >= 1:
        return f"epoch{number - 1}"
    return None


def model_epoch_label(state, version):
    epoch = model_epoch(state, version)
    external_round = ((state.get("upload_layout") or {}).get("extrapolation_rounds") or {}).get(version)
    if epoch and type(external_round) is int and external_round > 0:
        return f"{epoch} 外推轮次{external_round}（{version}）"
    return f"{epoch}（{version}）" if epoch else f"模型轮次未记录（{version}）"
