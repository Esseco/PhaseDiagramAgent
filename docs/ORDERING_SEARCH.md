# 独立 ordering 搜索

独立搜索优先通过 `Process_AL_MC` 本地库调用，科学逻辑见库内 `ORDERING_SEARCH.md`。Agent 操作 `prepare_ordering_search` 和 `recover_ordering_search` 均走已有 `prepare_local_batch_files` 工具与单动作审批，取消另建运行模式的需要。

输入操作参数：`mode=ordering_search_inputs`、`phase`、`tm_ratios`、`na_content`、`max_det_H` 或 `h_matrices`、可选 `reference_path`、`oxidation_states`、`options`。省略母结构路径时取已配置的相参考；取当前主模型，不自动训练新模型。

审批预览冻结扩胞、初始占据、氧化态、模型 SHA256 与搜索预算。准备生成独立目录、库运行时、`run_search.py` 和 ZIP；配置完整时附站点提交脚本，但不提交任务。部分 Na 联合搜索，以 TM 为分组目标，保留观测到最低能的 Na 伴随结构。

用户手动在计算环境运行并将完整结果目录取回。回收参数 `mode=ordering_search_results`、`plan_id`、可选 `results_directory`。校验文件和计划后登记独立结果；不自动并入主凸包、不自动生成 DFT。Agent 可据排名提出后续选择，由既有审批机制处理。

准备只记实际成本为 0，计算工作量显示为上限估计；结果不伪造实测 GPU 成本。每个工作线程加载一个模型，批准前应据 GPU 内存确定 workers。源模型、母结构或请求变化使旧批准失效。

py1 下已用假评估器验证审批、输入打包、变化失效和回收；真实模型/HPC 未执行。无需自动启动或重启服务，用户按现有方式手动管理。
