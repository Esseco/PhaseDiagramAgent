# 提交与回传

按 `epoch0/`、`epoch1/` 分轮。只上传本次交付的完整 `_input` 文件夹，执行 `sbatch submit_all.sh`；远端配对 `_output` 文件夹回传原登记位置后告诉 Agent“已回传”。

| 目录 | 内容与用途 |
| --- | --- |
| transfers/ | 普通计算和训练的配对上传 `_input`、回传 `_output` |
| relax_mc/ | Relax/MC 批次，配对 `_input` 上传、`_output` 回传 |
| verified_results/ | 已校验读取的原始结果、结构和凭据；无需再次上传 |
| recovery/ | 仅回收、错误证据和恢复绑定；无需重新计算 |
| storage_archives/ | 完整归档包，程序需要时读取与恢复 |
| G0001_*/ | 按搜索组及计算类型登记的工作目录，inputs 是输入，results 是结果 |
| MLIP-finetune-*/ | 原模型轮次下的训练工作目录 |

完整模型版本在各轮 README 和机器记录中。历史已提交任务不重投；运行记录直接使用新路径，不保留目录链接。缺少 epoch 的历史资料放 unassigned，不能据名称猜测完成状态。
