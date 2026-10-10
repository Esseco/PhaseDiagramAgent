# 相图与分析产品

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [DFT 校正综合相图](#doc-combined_phase_diagram)
- [Phase diagram outputs](#doc-phase_diagram_files)
- [DFT分析自动绘图](#doc-dft_parity_plots)
- [分析结果布局](#doc-output_layout)

<a id="doc-combined_phase_diagram"></a>

## DFT 校正综合相图

保留原始MLIP、纯DFT两张相图，新增 outputs/<模型版本>/combined/phase_diagrams/phase_diagram.csv，history保存输入版本快照。由Agent回收分析时生成，无新数据不重写；开发检查不分析生产数据。

使用项目已经校验的同帧、同模型能量配对及合格自旋数据。按相与TM/O2组分分组，计算ΔE=(E_DFT-E_MLIP)/(O数/2)。同Na锚点取均值；至少两个不同Na锚点之间线性插值，不外推、不跨相借用。校正是经验估计，不代表DFT精度证明，CSV必须区分来源。

合格且已识别的配对DFT结构直接用DFT能量；仅相同结构哈希的MLIP结构被替换，不能按structure_id替换不同几何。其余MLIP结构有覆盖时使用校正能量。无法校正的结构仍列在CSV，Eform/Ehull留空且不入凸包，状态标为partial；端点不足时标unknown，不伪造凸包。TM组分不一致明确失败。

复用原有端点最低能量、eV/O2形成能与Na凸包计算，Ehull同时保留eV/O2与eV/atom。新增mlip_energy_eV、dft_energy_eV、corrected_energy_eV、energy_source、correction_status、correction_eV_per_O2。当前MC筛选仍使用原MLIP相图，综合相图作为有来源标注的分析参考，不静默切换科学标准。

### 实现入口

[phase_agent/analysis/phase/update_phase_diagram.py](../../phase_agent/analysis/phase/update_phase_diagram.py)。

<a id="doc-phase_diagram_files"></a>

## Phase diagram outputs

New snapshots use the model version as the folder name:

    current/phase_diagrams/
      mace-mh-1/
        phase_diagram.csv          current MLIP dataset
        history/
          phase_diagram_mlip_<hull-version>.csv
          phase_diagram_mlip_<hull-version>.json
      dft/
        phase_diagram.csv          current DFT dataset, only when data exists
        history/
          phase_diagram_dft_<hull-version>.csv
          phase_diagram_dft_<hull-version>.json

Unchanged input reuses saved diagrams and CSV files. Input record ordering is not a
new dataset. CSV requests return an existing file without running the workflow;
missing files are restored from saved snapshots, without rerunning calculations or
phase identification. New energy/structure/phase evidence changes the input checksum
and updates the current CSV while retaining immutable historical snapshots.

Legacy paths remain supported and are not moved or deleted automatically. With no
new data, the existing CSV remains the returned file. The new layout is used for
new snapshots or restoration of missing files. Existing phase-identification caches
and MLIP/DFT separation remain unchanged.

[phase_agent/analysis/phase/export_phase_diagram_csv.py](../../phase_agent/analysis/phase/export_phase_diagram_csv.py)。

<a id="doc-dft_parity_plots"></a>

## DFT分析自动绘图

Agent导出本轮DFT分析产品时，使用已经通过同帧、模型版本及磁矩校验的配对表，
保存到对应模型/Search-group/DFT-round/comparisons/plots。
energy_parity为每原子能量；force_parity为每个原子的x/y/z分量，每N原子结构有3N点。
输出PNG/PDF/SVG及parity_metrics.json，坐标轴为能量eV/atom、力eV/Å，四边闭合。
图内MAE/RMSE为能量meV/atom、力meV/Å，显示一位小数；JSON保留完整精度。原有CSV及决策指标仍以eV为单位，不更改科学阈值。
指纹一致且图存在时直接复用，不重复绘制；无配对或缺matplotlib时显式报告。
受力RMS尺度比低于0.1只提示核查，不证明接口有误、不排除数据、不直接决定微调。
该提醒及图表信息进入本轮LLM分析上下文。历史已生成图保留，不更改原始数据。

[phase_agent/analysis/state/post_dft_assessment.py](../../phase_agent/analysis/state/post_dft_assessment.py)。

<a id="doc-output_layout"></a>

## 分析结果布局

新项目以配置中的 analysis_outputs 保存本地处理产品；submissions 中的 inputs/results 用于输入提交与必需结果回传。路径由配置和轮次登记解析，不从目录数量推测模型 epoch。

相图按 MLIP、DFT、combined 区分；DFT 比较、训练数据和诊断产品按科学轮次归档。索引引用实际产物路径；内容未变化时不重复导出。旧登记路径可以读取，迁移工具需要显式执行，读取文档不会移动文件。

完整目录见 [Epoch 输出](workspace.md#doc-epoch_output_layout) 和 [轮次传输](workspace.md#doc-round_transfer_layout)。

[phase_agent/analysis/feedback/output_catalog.py](../../phase_agent/analysis/feedback/output_catalog.py)。

