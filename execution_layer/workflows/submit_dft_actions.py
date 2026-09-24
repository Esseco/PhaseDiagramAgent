"""Submit reserved DFT tasks with confirmed parameters supplied by the program."""


def submit_dft_actions(decisions: list[dict], *, submitter=None, dft_parameters: dict, context=None) -> list[dict]:
    results = []
    for item in decisions:
        if item["action"] not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
            results.append({**item, "status": item["action"].lower()}); continue
        if submitter is None:
            results.append({**item, "status": "not_configured", "error": "DFT submitter not configured"}); continue
        try:
            result = submitter(decision=item, dft_parameters=dict(dft_parameters), context=context or {})
            results.append({**item, **result})
        except Exception as error:
            results.append({**item, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    return results
