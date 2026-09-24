# 代理模型比较

`run_surrogate_comparison()` 使用已有固定划分和同一个 branch 标签。未配置 embedding 或 MatterTune 时相应方法为 `skipped`，不会生成性能数值。

离线指标包括 MAE、RMSE、Spearman 排序、低能候选召回、重要候选漏选和 RF 不确定性与绝对误差的相关性。序贯回放每轮只使用此前已揭示的标签；需要历史标签和真实成本数量大于 `initial_revealed`。凸包改善必须传入 `hull_metric(revealed_rows)`，否则报告为 `not_configured`。

推荐文件只代表离线候选，`production_policy_change` 永远为 `false`。BOHB 效果不在本报告中评价，后续可将相同回放接口用于 BO/BOHB 对照。

默认成本口径为 `proxy_relative = evaluation_count × (atom_count/reference_atoms)^atom_exponent × scale`。扩胞由实际原子数进入成本，并通过 `max_atoms`、`max_det_H` 和 `max_proxy_cost_per_task` 封闭。实测成本与代理成本分别保存；若选择实测口径，单位不一致或缺失的记录不会进入等成本回放。
