# 最小结果回传

默认原则：只传本地Agent分析、训练数据准备和回收校验必须读取的内容；远端保留的大产物用清单登记位置，不删除原文件。

|阶段|必须回传|留在超算|
|---|---|---|
|Relax/MC|result.json、task.finished.json、结果引用的最终结构；当前重启协议需要的checkpoint；批次runtime统计|原始日志、trace/status/settings/pool诊断文件，位置记入remote_artifacts.json|
|DFT|result.json、task.finished.json、training.json（全部训练帧）、mlip_result.json（同帧原模型预测）；旧结果引用的结构文件|OUTCAR、vasprun.xml、WAVECAR、CHGCAR及原始日志|
|微调|kfold_metrics.csv、energy_comparison.csv、force_comparison.csv、models.json、training.finished.json|正式committee/主模型、K折模型、训练检查点和大日志|

DFT结构、能量、受力、应力和磁矩不能只传路径：本地分析与构建训练集仍要实际读取。模型可只登记远端路径，因为训练和后续科学预测在超算执行。路径和SHA256仅标识模型，不替代独立验证，更不构成激活授权。

新收集脚本不再把模型复制到results/models。models.json包含storage=remote、绝对remote_model_path、model_path、大小、SHA256及主模型/committee身份。保持超算训练目录稳定，否则清单路径失效。

已完成训练如需只刷新模型清单，将项目新版execution_layer/remote/collect_training_results.py替换到超算训练inputs目录（旧布局在训练轮根），然后在该目录运行 `python collect_training_results.py --manifest-only`。不训练、不重算K折，不自动删除旧results/models；下载时只选择表中必需文件，旧模型副本无需下载。

更改仅对使用新版生成/导出脚本的作业生效，已提交旧脚本不会自动更新。当前远端训练结果自动登记到候选模型的完整入口仍未实现；不能据此声称回传清单后会自动验证/激活。现有本地分析输出不参与下载，按需由Agent再生成。

检查点仍保留，因为现有续跑协议会读取它；未经远端续跑适配验证，不把它贸然改为纯路径。原始JSON/CSV压缩传输和历史results清理未在此变更中实施。
