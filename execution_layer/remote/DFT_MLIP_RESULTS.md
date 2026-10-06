# DFT / MLIP 双文件回传

新生成 DFT 输入包含 comparison_model.json，固定任务所属轮次模型版本、超算模型路径、head 和环境。VASP 提取后自动对同一最终帧做一次原模型预测，不额外弛豫。

results 每个任务目录：result.json（DFT 结构、能量、受力、应力、磁矩及副文件校验）、mlip_result.json（同帧 MLIP 能量和受力、任务身份、模型与结构 SHA256）、training.json（训练帧）、task.finished.json（结果完整性标记）。最后两项仍须保留，并非只下载两个文件。

本地默认只读远端预测，不运行 MLIP。校验任务、模型版本/已知模型指纹、最后一帧结构指纹、单位、数组形状和副文件校验和；合格 DFT 才参与 MAE/RMSE。首次未知模型指纹从有效回传记录绑定，后续同版本不同权重被拒绝。严谨核验可预先在模型注册表配置原模型指纹。

已完成批次：在超算项目根目录运行下列命令，指定原轮次模型；只补预测，不重跑 VASP。请先备份 results，因为需要更新 result.json 的副文件引用和完成标记校验和。

```bash
python -m execution_layer.remote.predict_dft_final_frame \
  --results /data/home/lichaoyue/26-10-PhaseDiagramAgent/upload_batches/MLIP-round-0001_mace-mh-1/Search-group-0001/DFT-round-0001_43c800068af6/DFT-single-point/results \
  --model-path /data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model \
  --model-version mace-mh-1 --head omat_pbe --environment py-mace
```

使用已更新项目代码，入口环境须能导入 pymatgen/numpy；MACE 在指定 conda 环境内执行。成功预测缓存复用，失败可重试。完整回传该 results 文件夹后说“继续”，已回收任务可补充预测，不增加 DFT 成本或训练标签。

本地参考模型已确认存在：E:\1-guihub库\MLIP_model\mace-mh-1.model。未修改生产配置或运行评估；超算预测使用超算路径，不使用 Windows 路径。显式 dft_comparison_location=local 才允许旧本地预测方式。
