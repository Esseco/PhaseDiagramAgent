"""Select one configured cost basis without mixing measured and proxy units."""


def select_comparable_cost(record: dict | None, *, basis="proxy_relative", measured_unit=None):
    record = record or {}
    if basis == "proxy_relative":
        proxy = record.get("proxy") or {}
        return proxy.get("value") if proxy.get("status") == "estimated" else None
    if basis == "measured":
        measured = record.get("measured") or {}
        if measured.get("status") != "measured":
            return None
        if measured_unit is not None and measured.get("unit") != measured_unit:
            return None
        return measured.get("value")
    raise ValueError(f"未知成本口径：{basis}")
