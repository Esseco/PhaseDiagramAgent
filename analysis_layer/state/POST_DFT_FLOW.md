# DFT 回收后的决策接口

回收完整，或人工对该轮剩余任务选择不再等待后，按 model_version、search_group_index、parent_relax_round、upload_operation_id 隔离评估。未回传任务保留原状态，不取消作业。

`post_dft_assessment` 复用保存的最终帧预测计算能量和受力 MAE/RMSE。缺失原轮次模型或预测时明确报告未评估；继续时只重试缺失预测，不重新识相、回收、增加训练标签或记账。不同模型不得混合比较。

评估完整后，LLM 根据评估、现有相图、覆盖及记忆提出下一步，仍需正常审批。旧 MC→DFT 建议失效，不能复用其批准。生成新 branch 成功才消费本轮决策；后续新 MC 任务仍按正常阶段推进。

`update_mlip` 正式工具复用 `create_model_update_handler`。运行时可提供 `model_update_handler`，或提供 `mlip_trainer`、`mlip_validation_evaluator`、`mlip_reevaluation_predictor` 及对应数据 providers。未配置训练器时明确报告，不假装完成。

微调须先启用已确认的 `mlip_finetune.enabled` 并达到新增合格数据门槛。数量门槛不是科学误差触发阈值；LLM 决策须说明误差、覆盖与记忆依据。不自动填写误差容忍值；既有独立验证和单独激活审批保持不变。

验证：`tests/test_post_dft_assessment.py` 覆盖轮次/模型隔离、部分回收关闭、缺失误差、重试缓存、训练门禁、正式 LLM 阶段路由。完整回归不包含暂不启用的 BOHB 集成测试。
