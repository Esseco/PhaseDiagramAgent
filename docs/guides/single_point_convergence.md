# 单点验证与竞争相检查

默认连续稳定模型更新轮数为1；MLIP弛豫fmax保持项目原配置（本次未更改）。

收敛报告的 single_point_validation 包含当前模型在同帧DFT单点结构上的能量（eV/atom）与力分量（eV/Å）MAE/RMSE、DFT最大原子残余力。能量按结构平均，力按笛卡尔分量汇总。复用原有质量、单位、模型版本与同帧校验，不把DFT弛豫后几何作为原候选单点验证。旧模型样本不混入当前模型统计。

同一成分下取各相的MLIP最低能代表，与最低能相比较。能差严格小于0.003 eV/atom时，要求两者有合格的同帧DFT单点记录，且DFT最低能相一致。相同相不要求内部排序一致；能差大于或等于阈值无需额外竞争相检查。缺少或过时记录时报告 required_dft_structure_ids，只列需要补充的结构，不提交计算。近简并无法确定唯一相时不计为一致。

配置字段：
- convergence.stable_model_update_epochs = 1
- convergence.phase_competition_gap_ev_per_atom = 0.003
- convergence.final_energy_mae_tolerance：沿用已确认的阈值；新配置默认0.003 eV/atom。
- convergence.final_energy_rmse_tolerance、final_force_mae_tolerance、final_force_rmse_tolerance：默认null，仅报告；显式设置后加入硬门槛。

已有确认配置与快照不会被新默认值覆盖。要启用竞争相检查，在项目配置中加入该能差字段，并保留稳定轮数1；按既有配置修订流程生效。真实服务由用户手动启动/停止，源码更新不重启服务。

解释范围：给定候选几何与当前模型的单点排序，不证明DFT充分弛豫后的相稳定性。按用户指定政策，阈值外竞争相免检，不能据此保证阈值外没有模型误判。
