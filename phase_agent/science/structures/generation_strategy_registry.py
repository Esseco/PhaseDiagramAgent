"""可扩展的 branch 生成策略注册表。"""


class GenerationStrategyRegistry:
    def __init__(self):
        self._strategies = {}

    def register(self, name, function, *, replace=False):
        if name in self._strategies and not replace:
            raise ValueError(f"生成策略已存在：{name}")
        self._strategies[name] = function

    def get(self, name):
        if name not in self._strategies:
            raise KeyError(f"未知生成策略：{name}")
        return self._strategies[name]

    def names(self):
        return tuple(self._strategies)
