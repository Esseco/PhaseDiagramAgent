# Branch surrogate 数据基础

本目录只整理历史数据，不运行 MC 或 DFT。默认任务是在固定 MLIP、固定 MC 预算和固定参考凸包下，预测 branch 已发现最低能量与参考凸包的差。差值可为负，不截断。

```python
from experiments.branch_surrogate.default_branch_surrogate_config import default_branch_surrogate_config
from experiments.branch_surrogate.run_branch_dataset_preparation import run_branch_dataset_preparation

config = default_branch_surrogate_config()
config.update({
    "target_mc_budget": 10000,
    "mlip_version": "mh-1-v1",
    "hull_reference_version": "mlip-hull-v1",
})
hull = {
    "version": "mlip-hull-v1",
    "groups": {"{\"x\":\"1/2\"}": -6.25},
}
result = run_branch_dataset_preparation(
    "outputs/phase_data.json",
    config=config,
    hull_reference=hull,
    output_directory="outputs/branch_surrogate_v1",
)
print(result["report"])
```

`branch_dataset.json` 保留原始能量、预算、模型版本、成本和缺失原因；`split_manifest.json` 固定框架分组划分；`data_check_report.json` 汇总缺失和泄漏检查。若要预测 DFT，应建立新的 task、标签构建器和输出目录，不能把 DFT 能量混入本任务。
