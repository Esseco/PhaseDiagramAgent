# DFT分析自动绘图

Agent导出本轮DFT分析产品时，使用已经通过同帧、模型版本及磁矩校验的配对表，
保存到对应模型/Search-group/DFT-round/comparisons/plots。
energy_parity为每原子能量；force_parity为每个原子的x/y/z分量，每N原子结构有3N点。
输出PNG/PDF/SVG及parity_metrics.json，坐标轴为能量eV/atom、力eV/Å，四边闭合。
图内MAE/RMSE为能量meV/atom、力meV/Å，显示一位小数；JSON保留完整精度。原有CSV及决策指标仍以eV为单位，不更改科学阈值。
指纹一致且图存在时直接复用，不重复绘制；无配对或缺matplotlib时显式报告。
受力RMS尺度比低于0.1只提示核查，不证明接口有误、不排除数据、不直接决定微调。
该提醒及图表信息进入本轮LLM分析上下文。历史已生成图保留，不更改原始数据。
