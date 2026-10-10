# Surrogate models

管理、筛选和 RF 训练在 `py1` 中运行；真实 MACE Relax 通过 `create_py_mace_relax_backend()` 调用 `py-mace`。本目录不会把深搜结构作为输入。

新增手工特征时，新建一个 `feature_xxx.py`，返回：

```python
def feature_xxx(structure, context=None):
    return {"xxx": float(...)}

FEATURE_DEFINITION = {"name": "xxx", "version": "1", "function": feature_xxx}
```

再把定义传入 `create_feature_registry(extra_features=[FEATURE_DEFINITION])`，并将名称加入 `enabled_features`。版本或定义变化时应更新 `version`。

流程为：固定数据划分 → 初始结构 → 缓存检查 → MLIP Relax → 手工特征/可选 embedding → 只在训练集拟合筛选和预处理 → 验证集比较新增特征 → 最后一次使用独立测试集。MatterTune 和 embedding 当前均为显式适配入口，未配置时返回 `not_configured/not_implemented`。
