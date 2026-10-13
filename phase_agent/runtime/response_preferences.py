"""Invocation-local display preference, never a scientific configuration change."""

from contextlib import contextmanager
from contextvars import ContextVar

_detail = ContextVar("response_detail", default="brief")

ACTIONABLE_REPLY_RULE = (
    "默认回复简短，约五行：项目名称 · epoch批次 · 当前状态；本次发生了什么；结果如何；下一步做什么；大致预算。"
    "项目名称使用当前项目的可读名称，不能仅用diagram_default代替科学项目名称。"
    "只依据已记录事实，不把准备、提交、完成、回收或正常结束与优化收敛混为一谈。"
    "预算优先给任务或GPU作业数量及有依据的预计耗时；无可靠估计直接说明，不展示相对成本。"
    "默认隐藏哈希、配置版本编号、证据引用、任务键和内部诊断；保留具体阻碍、影响与处理动作。"
    "审批只简述动作与必要范围，给出确认方式，不重复历史、审批说明或未执行声明。"
    "操作需要时给一个上传目录、必要命令及回传目录；答疑直接回答，不强加无关操作。"
    "不要复述完整冻结方案；细节只在用户请求查看详情时展开。"
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
