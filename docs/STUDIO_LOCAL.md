# Studio 本地聊天入口

图ID为phase_chat，定义在orchestration/studio_chat_graph.py。langgraph.json 注册 run/studio_runtime.py 的 Starlette lifespan；Agent Server 启动时创建唯一项目 handler，Studio 直接调用它，8765 控制页共用该实例，不经过 HTTP 聊天网关。启动器通过 PHASE_AGENT_RUNTIME_CONFIG 绑定工作区；直接运行 langgraph dev 时也必须设置该变量。

## 启动

项目模板和当前工作区统一使用agent_runtime.json；历史备份保留原始名称与内容，不作为启动入口。

旧run/open_webui_api.py已迁移为run/agent_api.py，共用工厂为create_agent_runtime。旧langgraph_web_chat.py、agent_chat.html及其专用测试/文档已删除。旧运行配置文件名暂可继续读取，不需要移动实际数据。

根目录仅保留一份 requirements.txt 安装清单，不再提供重复 cmd 启动脚本。控制接口8765由 Agent Server lifespan 管理；Studio默认2024。不要同时启动多个服务占用相同端口。

PowerShell 使用标准 CLI：

```powershell
Set-Location 'E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram'
conda activate py1
$env:PYTHONUTF8 = '1'
$env:PHASE_AGENT_RUNTIME_CONFIG = 'E:\0-FM-PhaseDiagram\agent_runtime.json'
$env:PHASE_STUDIO_RECEIPT_PATH = 'E:\0-FM-PhaseDiagram\workflow_state\studio_gateway\gateway.json'
$env:LANGSMITH_TRACING = 'false'
langgraph dev --host 127.0.0.1 --port 2024 --no-reload
```

可选统一管理入口（自动设置上述环境、检查端口并监控子进程）：python -m run.studio_service --runtime-config "E:\0-FM-PhaseDiagram\agent_runtime.json"。需要项目选择窗口时运行 python -m run.local_project_launcher。二选一启动，不需要额外控制服务。

打开https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024，选择phase_chat的Chat模式。Studio账号/网络访问可能需要LangSmith登录；关闭tracing不等于Studio是离线网页。密钥不写入langgraph.json。开发服务器不是生产部署。

先等待终端READY提示再打开网页。浏览器可访问http://127.0.0.1:2024/info验证连通。保持终端打开；Ctrl+C停止。若配置不存在、端口占用或90秒仍未就绪，会明确报错。不要同时使用项目选择启动器与命令行启动同一服务。

## 确认与恢复

在 Studio 的助手/运行 Context 配置中，response_detail 可选 brief（默认简洁）或 detailed（详细）。它控制项目的状态与已有工作流详细展示，不改变计算策略、模型 token 预算或审批。网页其他显示详细程度开关不保证会传入该字段。普通状态始终说明实际登记的 epoch、当前阶段、下一步、待审批动作及仍需回收/不再等待数量；没有 epoch 登记时明确报告未知，不猜测。

“现在什么进度”“现在是什么进度？”等状态查询直接读取已有状态并给出简洁摘要，不要求先选择继续/新建、不调用模型、不推进流程。恢复确认仅针对继续推进；查询不会批准已有方案。

普通消息发送后直接进入 Agent，不再要求发送确认或填写 Resume JSON。真正的业务方案仍需批准；“同意”交由原审批逻辑处理，不自动批准其他 action。升级前若停在旧 send_user_message 中断，重启后不要恢复旧中断，直接在原会话发送一条新的用户消息。

独立消息收据在调用前落盘。相同thread/message ID即使检查点丢失也不重复发送；失败或返回后中断时需核对业务状态，不自动重试。不要删收据绕过保护。历史分叉到新thread或改消息ID不属于同一身份；当前不是服务端全面禁止分叉，勿用Studio历史编辑发送生产动作。

## 当前范围

标准 messages 入口已连接同进程业务 handler，保留业务审批与防旧消息重放。phase_chat 现在递归包含实际运行的科学子图：回收与校验 → 科学反馈 → 等待检查 → 本轮评估与导出 → 有界动作循环 → 收尾与输入准备。动作循环内包含模型刷新检查、提案、审批、校验、注册工具分支和入账。配置、状态查询和 CSV 请求仍由原命令路由处理，不强迫进入科学流程。Studio 展开子图即可查看这些真实阶段；具体科学函数仍属于原业务模块，不逐个建节点。

生产图通过每次独立的 Runtime context 接收回调，完整业务结果也留在 context；图流仅返回阶段、状态和选中工具。state/ledger 仍为事实来源。不能从历史科学节点任意续跑；请发新消息从业务状态恢复。Studio 消息级回执在子图调用前落盘，失败时阻止同消息重放。真实浏览器联调需重启服务后验收。

官方 Studio 设置要求 LangSmith API key。可在启动前通过环境变量 LANGSMITH_API_KEY 配置；不写入项目配置、不提交密钥。Python 管理入口强制 LANGSMITH_TRACING=false；直接 CLI 时按上例设置，未授权上传科学运行轨迹。官方生命周期接入方式：https://docs.langchain.com/langsmith/custom-lifespan 。

## DeepSeek 决策 API

LangGraph 负责编排，DeepSeek 负责模型决策；LangSmith 密钥不能代替 DeepSeek 密钥。

- 模型名、base_url、token 上限等配置在工作区 agent_runtime.json 的 deepseek 对象，不在 langgraph.json。
- 密钥优先读取 DEEPSEEK_API_KEY，未设置时读取当前 Windows 用户的凭据管理器。
- 启动后打开 http://127.0.0.1:8765/phase/setup ，输入自己的 DeepSeek 密钥，测试并保存。该测试会调用一次 DeepSeek API；密钥不写入项目或发往超算。
- 也可在启动终端设置 DEEPSEEK_API_KEY 后再启动；不要把密钥粘贴进聊天或提交到仓库。

langgraph.json 的 dependencies、graphs、http.app 已注册项目依赖、phase_chat 图和共用项目运行生命周期。图和 app 必须使用包模块路径（orchestration.studio_chat_graph:graph、run.studio_runtime:app），不能将持有 handler 的 app 改为文件路径，否则 Agent Server 的别名加载会与聊天导入产生两份运行模块。子图由 phase_chat 引用，无需另注册可绕过聊天审批的科学运行入口。
# 科学流程面板

## Studio真实图的科学动作分组

生产 `approved_scientific_action` 图不再横向铺开13个工具叶节点，而是7个真实执行子图：branch_search、structure_and_batch_inputs、mc_search、dft_inputs、mlip_finetune、model_refresh、convergence_and_control。展开对应子图才显示具体工具。每个子图根据已校验的 selected_tool 选择一个原派发回调，审批和审计仍在父图；不是固定按branch→MC→DFT顺序执行。

structure_and_batch_inputs 是通用结构筛选/批次准备入口，可覆盖Relax/MC/DFT，不能把它全部理解成Relax。收敛与维护组包含暂停、策略调整和失败重试，不代表所有动作都是科学停止。

更新后重启服务并重新加载Studio图。生命周期和动作循环仍保留嵌套；本次仅收拢最宽的工具分支，没有完成一张平面的科学生命周期主图。自动布局的具体线长仍由Studio控制。

服务启动后可打开 `http://127.0.0.1:2024/phase/flow`（使用默认2024端口时）。这是本地只读面板：当前epoch与下一步、各科学阶段累计任务、已保存待审批方案，以及可折叠的真实技术连线。只在刷新时读取状态，不调用LLM、不回收或分析数据、不批准任务。

Studio画布仍保留真实执行图，布局不被替换。看科学进度用流程面板，调试某个节点用Studio。代码更新后需要重新启动服务。
