# 科研生命周期与模型 epoch

一个已启用的 MLIP 版本对应一个 epoch；同一模型下的 Branch 批次、MC 分段和 DFT 批次都属于该 epoch，不因一次工具调用或 DFT 回收自动递增。

## 新材料：epoch0

1. 在独立项目确认组成、相/组分/超胞边界、母结构、模型、计算资源与预算。已有参考 DFT 时先确认使用初始模型还是先微调。
2. Strategic 选择本次需咨询的专家。没有搜索结果时通常以 Exploration 为主；当前实现没有固定“每个 epoch 第一次只能 Exploration”的路由。Reliability 有证据时可参与，无证据时明确未知。
3. Exploration 提出探索建议，Strategic 形成当前 Action。Simulation 生成执行计划，经授权后调用已有工具生成 Branch、准备并执行 Relax/MC；分步还是已批准封装范围，遵循原方案与配置。
4. 回收结果、核对相和能量，更新当前模型凸包与证据。两位专家按需分析搜索充分性及可靠性；Strategic 比较旧 Branch 续搜、新 Branch、补 DFT、微调与暂停。
5. DFT 回收后比较同帧能量/力误差、相竞争风险、覆盖、收益和成本，再选择下一动作。现有 MC/DFT 阶段门控继续生效，不是无限自由循环；每次最多执行一个已批准动作。

## 更新模型并进入 epoch1

决定微调仅代表提出训练动作。累计合格 DFT → 训练输入与远端计算 → 回收报告/模型清单 → 独立验证 → 明确激活候选模型，才发生模型切换。训练完成或交叉验证通过都不自动激活。

切换至 M1 后保留 M0 相图和审计，先审批已有结构刷新，用 M1 的结果重建凸包；不同模型能量不直接混合。刷新完成后，若当前模型没有有效新增 Branch，先单独审批生成，再单独审批 Relax/MC。旧模型的搜索完成不代表新模型已搜过。

随后重复“更新事实 → 专家建议 → Strategic 取舍 → 必要人工介入 → Simulation 执行”。epoch1 可以结束、暂停，或再次训练验证并激活 M2，进入 epoch2。两个 epoch 不等于两次 MC；每个 epoch 可以包含多次决策。

## 运行与授权

普通 Agent 使用已配置客户端；Codex 接管使用原生专家请求和相同科学执行边界，不触发项目配置模型。重大决策、高成本、严重冲突与敏感动作请求人工。普通动作只有已确认配置明确授权时可自动通过；路线未来步骤没有执行权。

对话回复后等待新消息或恢复事件，不建立后台自动科学循环。提交后等待结果；继续用于核对回传，不等于批准、提交或训练。Simulation 的准备成功不代表计算完成。短期事实及已审核长期记忆参与后续决策，但记忆不能授权执行。

入口见 [启动](../guides/start.md)、[Codex](../guides/codex-team.md)；实现见 [project/graph.py](../../phase_agent/graphs/project/graph.py)、[decision_team/graph.py](../../phase_agent/graphs/decision_team/graph.py)、[model_generation.py](../../phase_agent/policies/model_generation.py)、[activate_validated_model.py](../../phase_agent/persistence/models/activate_validated_model.py)。
