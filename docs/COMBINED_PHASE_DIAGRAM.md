# DFT 校正综合相图

保留原始MLIP、纯DFT两张相图，新增 outputs/<模型版本>/combined/phase_diagrams/phase_diagram.csv，history保存输入版本快照。由Agent回收分析时生成，无新数据不重写；开发检查不分析生产数据。

使用项目已经校验的同帧、同模型能量配对及合格自旋数据。按相与TM/O2组分分组，计算ΔE=(E_DFT-E_MLIP)/(O数/2)。同Na锚点取均值；至少两个不同Na锚点之间线性插值，不外推、不跨相借用。校正是经验估计，不代表DFT精度证明，CSV必须区分来源。

合格且已识别的配对DFT结构直接用DFT能量；仅相同结构哈希的MLIP结构被替换，不能按structure_id替换不同几何。其余MLIP结构有覆盖时使用校正能量。无法校正的结构仍列在CSV，Eform/Ehull留空且不入凸包，状态标为partial；端点不足时标unknown，不伪造凸包。TM组分不一致明确失败。

复用原有端点最低能量、eV/O2形成能与Na凸包计算，Ehull同时保留eV/O2与eV/atom。新增mlip_energy_eV、dft_energy_eV、corrected_energy_eV、energy_source、correction_status、correction_eV_per_O2。当前MC筛选仍使用原MLIP相图，综合相图作为有来源标注的分析参考，不静默切换科学标准。
