# 命名输入/输出文件夹与清单修复（2026-10-14）

根因：旧 run_upload.py 校验并提取 manifest.files，却遗漏 transfer_manifest.json 自身；协调器在任何子任务提交前读取清单时失败。新的独立计算节点驱动会把清单原字节一起写出；同一压缩包已解压且保留指纹时，可补齐缺失清单，不覆盖已有进度。

上传完整 epoch0_RelaxMC_Batch01_input，里面严格只有 submit_all.sh、run_upload.py、inputs.zip。sbatch 分配计算节点后运行 Python；结果自动发布至同级 epoch0_RelaxMC_Batch01_output，内含 results.zip（若协调器生成）、run_status.json、run.log。失败也发布状态与日志，不伪造科学结果。

本地收到 output 文件夹后自动发现 results.zip；保留旧 returned/results.zip 读取兼容。只有失败诊断文件时显示回传错误，不静默继续等待。

Co、Cr 当前命名输入包已生成，科学 state.json 与原输入压缩包 SHA256 不变，未修改任务、模型、预算、审批、计算状态。源码基线和备份在工作区 build/named_handoff/baseline。没有操作真实服务或提交 HPC 作业。
验证：生产回归 15 passed、1 deselected（此前已确认的基线提示文字断言）；覆盖真实协调器输入校验、假计算、完整 output 生成和本地自动导入。按项目 ruff.toml 规范格式化前后 AST 一致。未验证真实 HPC 执行。
