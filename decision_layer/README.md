# decision_layer

Agent action、轮次策略、DFT 决策和人工修订均只接收 StateSnapshot。Agent 选择本轮 Branch 批次，Hyperband 负责 fidelity 晋级。

子目录：`agent/` LLM 客户端与 proposal，`strategy/` 生成/轮次策略，`calculation/` 计算阶段建议，`scoring/` 可解释评分，`qbc_selection/` DFT/QBC 决策与实验基线。

主要入口：`agent/propose_tool_action.py`、`qbc_selection/decide_dft_actions.py`、`strategy/propose_round_strategy.py`。本层不执行计算、不写科学结果、不直接修改台账。
