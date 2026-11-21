# 唯一微扰恢复维护记录

本次修改新提案共用策略及提案封装，普通 Agent 与 Codex 均不得创建 sample_direct_extrapolation / 旧本地先微扰方案。旧 frozen action 执行函数保留 legacy=True 策略核验，不迁移历史数据、输入或审批。现有恢复完成条件、50有效标签、5次限制、训练/激活/刷新边界保持。

完整刷新证据 <10% 正常，>=10% 仅微扰；模型更新次数不影响新路线；缺证据未知。历史末态失败下界只按旧执行契约使用，不能替代新路线的完整证据。现有 Relax–MC 不单独触发恢复的范围约定未改，监测继续供普通DFT选点。

实际后端沿用先全部母帧弛豫，再生成微扰并由 Process_MLIP 几何检查，最后特征差和DIRECT；未增加微扰后MLIP弛豫。父帧失败或特征计算异常保守中止；不自动重试。几何/特征排除审计保存于父帧检查点 candidate_exclusions.json，成功结果亦携带审计；不删除原始输入。后续恢复DFT复用原远端采样→静态DFT分发→回收→恢复标签/训练链，不等待常规DFT。

科学配置来源：evidence/perturbation_recovery.py 限制 1<=det(H)<=确认的max_det_H<=6；parent_first_sampling.py 沿用原始 MACE-MH-1，omat_pbe/float64，LBFGS、可变胞、fmax=0.05、500步；parent_first_handoff.py 冻结已批准参数（默认36帧、seed42、最小距离1 Å、体积比0.75–1.25）。Process_MLIP/nepkit_sampling.py 复用胞缩放a/b±1%、c±2%，Na/Li位移0.12 Å、TM0.05 Å、O0.08 Å。M3GNet特征差为候选与其成功弛豫母帧特征向量的欧氏范数，<=1合格；DFT成功参考供DIRECT覆盖使用，不作为该差值基准。

py1 合成回归：15个相关测试模块合计110通过；随后几何排除直接验证加入后，后端12通过。覆盖阈值/未知、更新次数、几何/过大/缺失/非有限特征、母帧失败、远端模拟DFT接力、标签训练、历史审批与防重放。ruff check phase_agent通过；全仓format --check报告145文件待格式化，未批量改写其他任务文件。测试TemporaryDirectory由conftest自动清理。

限制：没有运行真实MACE/M3GNet/VASP、HPC或真实服务；远端Process_MLIP依赖及计算节点环境仍需另行验证。没有部署、上传、提交、重启或修改真实科研状态。
