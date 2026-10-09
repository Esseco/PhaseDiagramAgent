# 两端 Python 环境

初始化配置 python_environments：local_python（本地管理/提取，默认py1）、local_mlip（本地科学后端，默认py-mace）、remote_python（超算DFT/提取环境）、remote_mlip（超算MLIP环境）。两项远端默认空，必须人工确认真实名称；不能从本地推断。current 表示入口已在正确环境运行。

DFT 输入 comparison_model.json 显式写入 remote_mlip，远端预测不再默认本机py-mace。补充预测命令 --environment 必填。本地显式比较使用 local_mlip。

remote_python 记录并在初始化校验；现有用户提交脚本/集群 shell_preamble 仍须按该环境激活。项目不自动改写用户的集群 module/source 设置。Relax/MC同样须在remote_mlip环境提交。旧的已生成GPU.sh和comparison_model.json不会被配置修改自动覆盖，应按正常重生成确认流程刷新。

旧配置缺字段会提示补齐，不修改历史模型、结果或科学参数。先在超算 conda env list 确认真实环境，不把候选默认当成存在性验证。未连接超算时，初始化只能校验填写格式，不能声称环境已可用。
