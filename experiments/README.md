# experiments

独立方法比较和离线评估。`bohb/` 比较固定预算、随机 Hyperband 与 BOHB；`branch_surrogate/` 比较代理模型和特征方案；`compare_rule_and_agent.py` 比较规则与 Agent。

实验应复用正式层接口，不复制业务流程，也不得绕过 execution_layer 写入正式搜索状态。
