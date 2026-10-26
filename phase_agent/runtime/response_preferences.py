"""Invocation-local display preference, never a scientific configuration change."""

from contextlib import contextmanager
from contextvars import ContextVar

_detail = ContextVar("response_detail", default="brief")

ACTIONABLE_REPLY_RULE = (
    "默认回复简短，约五行：项目名称 · epoch批次 · 当前状态；具体结果；阻碍原因；下一步由谁做什么。有新计算方案时给必要预算。"
    "项目名称使用当前项目的可读名称，不能仅用diagram_default代替科学项目名称。"
    "只依据已记录事实，不把准备、提交、完成、回收或正常结束与优化收敛混为一谈。"
    "预算优先给任务或GPU作业数量及有依据的预计耗时；无可靠估计直接说明，不展示相对成本。"
    "默认隐藏哈希、配置版本编号、证据引用、任务键和内部诊断；保留具体阻碍、影响与处理动作。"
    "审批只简述动作与必要范围，给出确认方式，不重复历史、审批说明或未执行声明。"
    "微调、DFT、Relax、MC或Relax+MC进入审批、待提交、等待回传或回收时，都明确给出提交路径、已登记的提交脚本与命令、回收路径；审批前用预计路径和批次命名规则，准备后用已登记的绝对路径。路径未登记时明确说明，不能猜测。Agent 先读取项目 submissions/submission_index.json，再按其中的 schema 和精确路径操作；历史目录保留原地，不按名称推断状态。"
    "操作需要时给一个上传目录、必要命令及回传目录；答疑直接回答，不强加无关操作。"
    "每轮只输出具体结论与明确下一步；状态查询不填充“已检查状态”“未提供预算”等套话，不把缺少结果写成计算失败；不要复述完整冻结方案。"
    "一般交互信息、完整答复、Branch/DFT选点分布、任务错误、回收诊断和详细依据由程序保存为项目交互记录文件夹内的Markdown，并在答复直接显示本轮文件的完整绝对地址，不能用隐藏的链接文字替代地址。"
    "不得声称已生成未实际返回的文件；用户要详细信息时也优先指向本轮Markdown。"
    "历史缺模型错误须对照训练回传清单、当前输入中的远程引用和后续成功结果；不把旧输入路径的失败当成当前模型失效，不猜路径或自动重训。"
)


def detailed_response():
    return _detail.get() == "detailed"


@contextmanager
def response_detail(value):
    if value not in {"brief", "detailed"}:
        raise ValueError("response_detail must be brief or detailed")
    token = _detail.set(value)
    try:
        yield
    finally:
        _detail.reset(token)
