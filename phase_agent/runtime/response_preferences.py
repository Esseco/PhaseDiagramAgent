"""Invocation-local display preference, never a scientific configuration change."""

from contextlib import contextmanager
from contextvars import ContextVar

_detail = ContextVar("response_detail", default="brief")

ACTIONABLE_REPLY_RULE = (
    "所有用户交互保持简洁但可操作：先说明本轮实际完成到哪一步、当前等待什么，再说明下一步。"
    "默认只显示一个本轮提交文件夹，用一句话说明用户要做的操作及结果回传；"
    "不要逐项罗列批次、压缩包、脚本和results的长路径。批次细节、命令及完整路径仅在用户问详情时展开。"
    "路径缺失就明确说未记录，不编造。上传、提交、完成、回收是不同状态，不能混淆。"
    "避免只说刷新、回传results或继续等内部简称；解释当前操作的目的。"
    "epoch表示模型更新轮次时用中文说明，不让用户误认为训练epoch。"
    "审批时说清具体批准对象和边界；答疑不强加操作步骤。"
    "只显示一个可复制的完整本地文件夹地址，不使用Markdown链接；说明目的与下一步即可，不重复当前状态或模型轮次。"
    "长证据、内部标识和重复背景放详情。"
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
