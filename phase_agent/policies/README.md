# 执行约束

execution_policy.py决定已形成方案是否获准进入执行校验；native_approval.py和file_approval.py绑定批准身份；validate_tool_action.py检查动作、版本、参数与预算约束。

策略模块不代替专家选择科学路线，不自行启动计算。预算事实和预留仍由tools/budget管理，数值科学规则由相应科学模块实施。
