"""计算阶段定义与执行器注册表。"""


class CalculationStageRegistry:
    def __init__(self):
        self._stages = {}

    def register(self, name, *, handler=None, label=None, replace=False, defaults=None):
        if name in self._stages and not replace:
            raise ValueError(f"计算阶段已存在：{name}")
        self._stages[name] = {
            "name": name,
            "label": label or name,
            "handler": handler,
            "defaults": dict(defaults or {}),
        }

    def get(self, name):
        if name not in self._stages:
            raise KeyError(f"未知计算阶段：{name}")
        return self._stages[name]

    def names(self):
        return tuple(self._stages)
