"""Human-readable model generations, independent of calculation sub-rounds."""


def model_epoch(state, version):
    number = ((state.get("upload_layout") or {}).get("model_rounds") or {}).get(version)
    if isinstance(number, int) and not isinstance(number, bool) and number >= 1:
        return f"epoch{number - 1}"
    return None


def model_epoch_label(state, version):
    epoch = model_epoch(state, version)
    return f"{epoch}（{version}）" if epoch else f"模型轮次未记录（{version}）"
