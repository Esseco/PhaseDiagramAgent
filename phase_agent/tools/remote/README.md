# 远端任务协议

本地保有主状态、账本与审批；远端消费冻结任务/manifest，保存结果和完成标记。`RemoteBatchRunner` 物化批次而不直接提交；传输和调度通过显式适配器注入。人工上传使用 `ManualUploadBatchRunner`。

当前默认兼容批次上限：Relax 100、MC 10、DFT 1；实际还受确认配置、兼容性及预算限制。Relax/MC 提交批次根目录 GPU.sh，DFT 提交单任务目录脚本。仅生成输入不能标记为已提交。

新目录见 [每轮传输](../../../docs/guides/workspace.md#doc-round_transfer_layout)。版本、task/batch身份及 SHA256 核验后才回收；结果先原子发布，完成标记随后写入。最终结构及分析必需摘要回传，大模型和轨迹按项目协议留在远端。旧登记路径继续支持读取。

## 阅读入口

- [batch_runner.py](batch_runner.py)：冻结任务、manifest、分批和结果核验。
- [manual_upload_runner.py](manual_upload_runner.py)：人工上传材料与提交说明。
- [integrity.py](integrity.py)：校验与结果身份。
- [api.py](api.py)：显式传输/调度适配接口。

实际环境与提交策略以确认配置为准；本包不推断远端环境，不在登录节点替代计算节点执行科学任务。

## 并发结果打包

`create_bundle` 为每个写入者独占分配同目录的临时 `.part` 文件，完整写入后原子替换目标；失败只清理自己拥有的临时文件并保留上一完整包。`consolidate_results` 合并共享总包时，在 POSIX 上使用 `fcntl.lockf` 字节范围锁，在 Windows 上使用原有 `msvcrt` 锁，并以进程内线程锁保护文件描述符的完整生命周期。不要改回固定 `.part` 文件或只依赖 BSD `flock` 的跨节点行为。

Relax/MC、GPU 用量记录及微调交付中的 portable 打包脚本均从这份模块生成；后续新任务包自动带上修复。已准备、上传或运行的冻结任务包不随源码更新自动改变，不能直接覆盖远端副本或以打包错误为由重跑已完成计算。测试覆盖交错发布、失败清理、同进程线程及多进程竞争；具体集群的跨节点锁一致性仍需在批准的测试任务中验证。
