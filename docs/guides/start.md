# 启动、对话与调试

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [相图搜索项目 UI](#doc-project_ui)
- [Studio 本地聊天入口](#doc-studio_local)
- [项目运行模式](#doc-initial_project_mode)
- [Studio暂停与删除](#doc-studio_pause_delete)

<a id="doc-project_ui"></a>

## 相图搜索项目 UI

从正式项目根目录，在 py1 中运行 `python -m phase_agent.runtime.local_project_launcher`。
也可使用 py1 的 pythonw.exe 打开 `phase_agent/runtime/start_projects.pyw`。

1. 点击“选择文件夹”。项目名称就是文件夹名；已有 agent_runtime.json 会自动加载，新目录会建立独立配置草稿。
2. 点击“启动服务”。服务会准备好该文件夹独立的 Studio 配置和端口，在下方显示 Studio 网址，不自动打开浏览器，也不发送计算指令。
3. 复制 Studio 网址到浏览器，在 Studio 中完成配置、开始或继续搜索、处理审批。仅关闭网页不会停止服务。
4. 点击“查看状态”查看所选项目摘要和 Studio 网址；点击“停止服务”停止本窗口启动的项目服务。可同时启动多个文件夹，各项目独立运行。

主界面只有“选择文件夹”“启动服务”“查看状态”“停止服务”，项目列表每 3 秒自动刷新。“运行中”表示本地服务在线，计算进度以项目摘要和 Studio 为准。

每个项目独立保存：

```text
agent_runtime.json                    本地运行设置
parameters/                           科学配置草稿及会话
workflow_state/                       搜索状态、台账、审批、执行记录
workflow_state/studio/.langgraph_api/  Studio 对话和 checkpoint
workflow_state/studio/service.json     本地服务身份和端口
logs/agent_server.log                  服务日志
structures/                           候选结构
analysis_outputs/                     相图等分析结果
submissions/                          计算输入批次
```

项目写入路径必须在其文件夹内。导入旧项目如果提示路径冲突，先核对目录；程序不会迁移或覆盖旧结果。
代码共用，但每个项目拥有独立进程、端口和状态。操作系统锁阻止同一项目重复启动。
本地服务与 Studio 页面分开：关页面不会停止服务；“停止服务”或退出管理窗口会停止本窗口启动的本地服务，超算作业不会自动取消。

文件隔离不分配 GPU。多个计算项目仍需在各自配置中指定合适的 GPU/超算资源；现有预算与调度限制继续生效。

### 实现入口

[phase_agent/runtime/project_dashboard.py](../../phase_agent/runtime/project_dashboard.py)。

<a id="doc-studio_local"></a>

## Studio 本地聊天入口

图ID为phase_chat，定义在phase_agent/graphs/studio_chat_graph.py。langgraph.json 注册 phase_agent/runtime/studio_runtime.py 的 Starlette lifespan；Agent Server 启动时创建唯一项目 handler，Studio 直接调用它，8765 控制页共用该实例，不经过 HTTP 聊天网关。启动器通过 PHASE_AGENT_RUNTIME_CONFIG 绑定工作区；直接运行 langgraph dev 时也必须设置该变量。

### 启动

项目模板和当前工作区统一使用agent_runtime.json；历史备份保留原始名称与内容，不作为启动入口。


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

可选统一管理入口（自动设置上述环境、检查端口并监控子进程）：python -m phase_agent.runtime.studio_service --runtime-config "E:\0-FM-PhaseDiagram\agent_runtime.json"。需要项目选择窗口时运行 python -m phase_agent.runtime.local_project_launcher。二选一启动，不需要额外控制服务。

打开https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024，选择phase_chat的Chat模式。Studio账号/网络访问可能需要LangSmith登录；关闭tracing不等于Studio是离线网页。密钥不写入langgraph.json。开发服务器不是生产部署。

先等待终端READY提示再打开网页。浏览器可访问http://127.0.0.1:2024/info验证连通。保持终端打开；Ctrl+C停止。若配置不存在、端口占用或90秒仍未就绪，会明确报错。不要同时使用项目选择启动器与命令行启动同一服务。

### 确认与恢复

在 Studio 的助手/运行 Context 配置中，response_detail 可选 brief（默认简洁）或 detailed（详细）。它控制项目的状态与已有工作流详细展示，不改变计算策略、模型 token 预算或审批。网页其他显示详细程度开关不保证会传入该字段。普通状态始终说明实际登记的 epoch、当前阶段、下一步、待审批动作及仍需回收/不再等待数量；没有 epoch 登记时明确报告未知，不猜测。

“现在什么进度”“现在是什么进度？”等状态查询直接读取已有状态并给出简洁摘要，不要求先选择继续/新建、不调用模型、不推进流程。恢复确认仅针对继续推进；查询不会批准已有方案。

普通消息发送后直接进入 Agent，不再要求发送确认或填写 Resume JSON。真正的业务方案仍需批准；“同意”交由原审批逻辑处理，不自动批准其他 action。升级前若停在旧 send_user_message 中断，重启后不要恢复旧中断，直接在原会话发送一条新的用户消息。

独立消息收据在调用前落盘。相同thread/message ID即使检查点丢失也不重复发送；失败或返回后中断时需核对业务状态，不自动重试。不要删收据绕过保护。历史分叉到新thread或改消息ID不属于同一身份；当前不是服务端全面禁止分叉，勿用Studio历史编辑发送生产动作。

### 当前范围

标准 messages 入口已连接同进程业务 handler，保留业务审批与防旧消息重放。phase_chat 现在递归包含实际运行的科学子图：回收与校验 → 科学反馈 → 等待检查 → 本轮评估与导出 → 有界动作循环 → 收尾与输入准备。动作循环内包含模型刷新检查、提案、审批、校验、注册工具分支和入账。所有消息进入 dialogue 图；显式审批和只读控制使用具名节点，普通语义交给 LLM，配置编辑调用配置服务。Studio 展开子图即可查看这些真实阶段；具体科学函数仍属于原业务模块，不逐个建节点。

生产图通过每次独立的 Runtime context 接收回调，完整业务结果也留在 context；图流仅返回阶段、状态和选中工具。state/ledger 仍为事实来源。不能从历史科学节点任意续跑；请发新消息从业务状态恢复。Studio 消息级回执在子图调用前落盘，失败时阻止同消息重放。真实浏览器联调需重启服务后验收。

官方 Studio 设置要求 LangSmith API key。可在启动前通过环境变量 LANGSMITH_API_KEY 配置；不写入项目配置、不提交密钥。Python 管理入口强制 LANGSMITH_TRACING=false；直接 CLI 时按上例设置，未授权上传科学运行轨迹。官方生命周期接入方式：https://docs.langchain.com/langsmith/custom-lifespan 。

### DeepSeek 决策 API

LangGraph 负责编排，DeepSeek 负责模型决策；LangSmith 密钥不能代替 DeepSeek 密钥。

- 模型名、base_url、token 上限等配置在工作区 agent_runtime.json 的 deepseek 对象，不在 langgraph.json。
- 密钥优先读取 DEEPSEEK_API_KEY，未设置时读取当前 Windows 用户的凭据管理器。
- 启动后打开 http://127.0.0.1:8765/phase/setup ，输入自己的 DeepSeek 密钥，测试并保存。该测试会调用一次 DeepSeek API；密钥不写入项目或发往超算。
- 也可在启动终端设置 DEEPSEEK_API_KEY 后再启动；不要把密钥粘贴进聊天或提交到仓库。

langgraph.json 的 dependencies、graphs、http.app 已注册项目依赖、phase_chat 图和共用项目运行生命周期。图和 app 必须使用包模块路径（phase_agent.graphs.studio_chat_graph:graph、phase_agent.runtime.studio_runtime:app），不能将持有 handler 的 app 改为文件路径，否则 Agent Server 的别名加载会与聊天导入产生两份运行模块。子图由 phase_chat 引用，无需另注册可绕过聊天审批的科学运行入口。
## 科学流程面板

### Studio真实图的科学动作分组

生产 `approved_scientific_action` 图不再横向铺开13个工具叶节点，而是7个真实执行子图：branch_search、structure_and_batch_inputs、mc_search、dft_inputs、mlip_finetune、model_refresh、convergence_and_control。展开对应子图才显示具体工具。每个子图根据已校验的 selected_tool 选择一个原派发回调，审批和审计仍在父图；不是固定按branch→MC→DFT顺序执行。

structure_and_batch_inputs 是通用结构筛选/批次准备入口，可覆盖Relax/MC/DFT，不能把它全部理解成Relax。收敛与维护组包含暂停、策略调整和失败重试，不代表所有动作都是科学停止。

更新后重启服务并重新加载Studio图。生命周期和动作循环仍保留嵌套；本次仅收拢最宽的工具分支，没有完成一张平面的科学生命周期主图。自动布局的具体线长仍由Studio控制。

服务启动后可打开 `http://127.0.0.1:2024/phase/flow`（使用默认2024端口时）。这是本地只读面板：当前epoch与下一步、各科学阶段累计任务、已保存待审批方案，以及可折叠的真实技术连线。只在刷新时读取状态，不调用LLM、不回收或分析数据、不批准任务。

Studio画布仍保留真实执行图，布局不被替换。看科学进度用流程面板，调试某个节点用Studio。代码更新后需要重新启动服务。

[phase_agent/runtime/studio_runtime.py](../../phase_agent/runtime/studio_runtime.py)。

<a id="doc-initial_project_mode"></a>

## 项目运行模式

新建项目时选择运行模式；取消不会创建项目。默认审批模式：模型提出方案，用户批准后执行。自动模式是已有明确配置的运行选项，按预算、最大步数和等待条件停止；它不改变科学参数校验。

模式保存在项目 agent_runtime.json。加载已有项目保留该项目的选择。交互对话每轮最多一个科学动作；“继续”表示重新分析当前状态，不表示批准。

[phase_agent/runtime/local_project_launcher.py](../../phase_agent/runtime/local_project_launcher.py)。

<a id="doc-studio_pause_delete"></a>

## Studio暂停与删除

重新启动项目服务后生效。

- 运行中点击Studio的停止/取消，会向实际科学工作线程传递取消信号。下一节点、下一模型请求和下一执行动作不再启动。保留项目检查点，可发送新的“继续”恢复。
- 取消结束后可以删除该运行或对话。删除Studio记录不会删除state/ledger、结果文件或已提交的超算作业。
- 当前正在等待的同步模型HTTP请求无法立即打断；会等待返回或配置timeout，然后线程退出。正在写入或已提交的动作不回滚，恢复前仍走原有收据与审批检查。
- 请求退出前不提前释放占用，避免界面显示停止但后台继续提交任务。

实现：Studio节点提供同步/异步入口，异步取消通过ContextVar/Event传递到工作线程；节点边界、模型请求前后、动作派发前检查取消。取消不被普通Exception降级成LLM失败或默认策略。已完成节点保存检查点，未执行动作不创建开始收据。

验证：完整回归1011项通过，5项子测试通过；最后公开节点适配器17项针对性回归通过，未创建收据测试通过。隔离Studio中实际慢调用进入工作线程后，取消使线程由busy变为interrupted；运行删除、新请求执行、线程删除均成功。详见outputs/studio_cancel_verification.json。真实科学计算/远程作业取消未测试，也不由Studio删除隐式触发。


### 可见按钮入口

重新打开项目UI并重启服务，点击“管理对话”。也可以在当前Studio API地址后添加 /phase/conversations。Studio状态中的conversation_management_url提供当前项目完整链接。科学流程页和本轮过程页也包含“暂停／删除Studio对话”链接。

页面直接调用本服务的LangGraph API，每条对话显示“暂停”“删除对话”。暂停后等待请求退出；删除需要确认，且有running/pending请求时拒绝删除。busy但无活跃请求的旧残留线程允许删除。删除不会清除科学项目文件。

[phase_agent/runtime/studio_conversation_panel.py](../../phase_agent/runtime/studio_conversation_panel.py)。



## 新项目运行与验收

新建项目只创建独立配置草稿和说明，不确认科学参数、不枚举母结构、不创建科学任务。可以预先放入 `structures/reference_structures`；已有状态、记忆、提交或分析结果的目录应通过“添加已有项目”打开。

清晰运行的顺序：配置草稿 → 核对母结构与必填参数 → 用户确认配置 → LLM 分析并提出一个动作 → 校验参数与预算 → 展示方案并等待审批 → 执行批准的原方案 → 回收结果并继续分析。普通“继续”不会代替审批。

上线前分别验证三层：本地假模型/假工具回归；真实 API 的答疑与方案生成；经批准的一个超算小任务。第一层通过不能证明远端模型、环境或调度器已经可用。失败时先查看本轮过程记录中的模型输出、校验错误和执行回执，不反复修改图结构。
