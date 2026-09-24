# Local Agent / remote compute

本目录实现本地主状态模式。Agent、配置确认、结构生成、分析、预算结算和主台账只在本地运行；远端只接收不可变批次快照、保存作业状态并在离线计算节点写结果。

公开接口在 `api.py`：`sync_results → prepare_local → sync_tasks → submit_remote → query_remote → resume`，另有 `cancel_remote`。实际 SSH/SFTP/rsync/scp 和调度器命令通过 `CommandTransferAdapter`、`RemoteSchedulerAdapter` 的调用函数注入，不假设 Slurm 或具体命令。VS Code Remote SSH 不参与自动流程。

`RemoteBatchRunner` 默认将 MLIP-Relax 按 100 个、MLIP+MC 按 20 个、DFT 按 1 个任务分批。每个任务和批次携带 config/model/batch/task 标识及 SHA-256；worker 先原子写结果，再写完成标记。本地只有通过版本和校验验证的结果才会交给现有回收/预算结算。

`VerifiedLocalMirrorTransport` 与 `FileBackedMockRemoteScheduler` 是 dry-run 示例：只回传结果摘要、完成标记、最终结构、checkpoint 和日志索引，不下载大轨迹。真实传输适配器应保持相同白名单。API key 只从本地环境提供；同步前会拒绝任务中的常见密钥字段。

离线计算职责严格分离：本地主流程是唯一可写主状态和台账；登录节点只传输以及提交、
查询、取消；计算节点只消费不可变 manifest，并以“先完整 result、后完成标记”的顺序发布。
候选结构可先进入独立 `offline_check_dedup` 批次，只有本地校验去重结果后，正式 Relax/MC
任务才可越过去重门。不同 `periodic_search_space_id` 的结构禁止自动合并。

第一版可人工上传下载；`ManualUploadBatchRunner` 写出 manifest、快照、SHA256SUMS 和操作
说明。自动同步继续复用同一协议。自动模式只通过有有效期、总预算、单批成本、并发限制和
允许动作集合的确认快照执行一个有界步骤；DFT Relax、模型激活和硬约束变更永远退出到
单独人工确认。操作异常或提交结果不确定时暂停，必须先查询，不能盲目重提。调用
`disable_automatic_mode` 可随时退回调试模式。
