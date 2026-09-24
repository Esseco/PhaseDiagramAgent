"""Run enabled feature functions without silently replacing failures."""


def extract_registered_features(structure, *, registry: dict, enabled: list[str], context=None) -> dict:
    values, definitions, failures = {}, {}, {}
    for name in enabled:
        item = registry.get(name)
        if item is None:
            failures[name] = "not_registered"
            continue
        try:
            result = item["function"](structure, context or {})
            if not isinstance(result, dict) or not result:
                raise ValueError("feature function must return a non-empty dict")
            absent = [key for key, value in result.items() if value is None]
            if absent:
                raise ValueError(f"missing values: {absent}")
            values.update(result)
            definitions[name] = {key: value for key, value in item.items() if key != "function"}
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
    return {"status": "completed" if not failures else "incomplete", "values": values, "definitions": definitions, "failures": failures}
