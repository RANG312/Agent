# 《A Survey on Evaluation of LLM-based Agents》中文阅读笔记

> Asaf Yehudai 等，*Findings of ACL 2026*，论文页码 26690–26714。本文笔记以用户上传的 25 页 PDF 为准；括号中的“PDF 第 n 页”指文件页序。以下是按原文章节组织的中文释义与梳理，并非逐句全文翻译。论文中的 benchmark 名称、术语与数字保留原写法。

## 摘要（Abstract）

作者认为，LLM-based agents 已从只生成文本的静态模型，发展为能在动态环境中规划、推理、调用工具并连续行动的系统。因此，评估也要从“回答是否正确”扩展到“能否通过一连串决策和交互完成用户任务”。综述从五个视角组织文献：**基础能力、特定应用、通用型智能体、benchmark 的共同设计维度、开发者使用的评估框架**。作者观察到评测正转向更真实、更困难且持续更新的设置，同时指出成本效率、安全与稳健性，以及细粒度、可扩展的评估仍是缺口。（PDF 第 1 页）

## 1 Introduction｜引言

LLM 的知识和交互方式本身相对固定；agent 则把 backbone LLM 放进多步骤工作流，配合外部工具，获得计算、检索新信息和操作环境的能力。由此，单看最终文本会漏掉规划是否合理、工具是否选对、环境状态是否被正确改变，以及连续动作中错误如何累积。作者主张 benchmark 应随 agent 能力共同演化，纳入新任务和新领域。（PDF 第 1 页）

**Figure 1 是全文阅读地图**（PDF 第 2 页）：蓝色分支为第 2 章四项基础能力；橙色分支为第 3 章四类应用；绿色分支为第 4 章通用型 agent；粉色分支为第 6 章的开发评估框架与 gym-like 环境。各分支右侧列的是代表性 benchmark 或工具，例如规划的 *PlanBench / FlowBench / Natural Plan*，网页任务的 *Mind2Web / WebArena*，软件工程的 *SWE-bench*，通用任务的 *Gaia2 / OSWorld / AppWorld*，框架的 *LangSmith / Langfuse / MLGym*。图中没有单列第 5 章，因为第 5 章从横向维度比较前述 benchmark；第 7–8 章则综合趋势与结论。图的用途是定位研究类别，不能把同一行的项目理解为可直接横向排名。

## 2 Agent Capabilities Evaluation｜智能体基础能力评估

作者把 agent 视为 **backbone LLM + agent harness**（承载工作流、工具与执行逻辑的系统）。基础能力既可单独测试，也可放进完整 agent 流程测试；两种测试回答的问题不同。正文概述四种能力，附录 B 展开具体 benchmark。（PDF 第 1–3、20–24 页）

| 原文能力 | 核心定义与评估重点 | benchmark 演进及作者判断 |
| --- | --- | --- |
| **Planning and Multi-Step Reasoning**（规划与多步推理） | 把复杂任务分解成子任务，并形成通向目标的执行路径；在交互中还要跟踪状态、预判行动结果、发现和恢复错误。 | *HotpotQA、Game of 24* 等推理任务曾用于 ReAct 一类方法；*PlanBench* 测更明确的规划能力；*FlowBench* 测结构化、专业化工作流；*Natural Plan* 测自然语言表达的现实规划问题。附录还讨论 *MINT、ToolEmu、AutoPlanBench、ACPBench*。作者概括：即使先进模型在长时程规划上也有困难。 |
| **Function Calling & Tool Use**（函数调用与工具使用） | 不只是能否输出某个函数名，还包括判断何时需要工具、选工具、把对话信息映射为参数、执行调用，并利用返回结果生成后续行动或回答。 | 早期 *ToolAlpaca、ToolBench、BFCL v1* 偏向单步、显式参数和结构匹配；*BFCL v2/v3* 加入多轮、多步与状态管理；*NESTFUL* 要求后续调用依赖前次输出，*ComplexFuncBench* 涵盖隐含参数、用户约束和长上下文。*MCP Atlas、Tool-Decathlon* 进一步面向真实 MCP 工具、多领域和长交互。 |
| **Self-Reflection**（自我反思） | agent 根据反馈调整信念、推理或行动，使后续决策得到修正。 | 早期常把现有任务改造成反馈循环，以最终答案是否改正衡量，难以分离提示方式的影响；*LLF-Bench* 测多样决策任务中的反馈使用，*LLM-Evolve* 重用过去的“问题—反馈”作为上下文示例；附录中的 *Reflection-Bench* 细分新信息感知、记忆、信念更新、反事实推理等。作者仍认为缺少公认的标准评测方法。 |
| **Memory**（记忆） | 在长交互中保留、检索和利用信息；正文区分 **episodic**（过往经历）、**semantic**（事实知识）、**procedural**（操作知识）记忆。 | 早期借长上下文阅读/问答任务测试记忆系统；后来 *StreamBench* 等测跨轮次复用过去交互及反馈，*MemBench、MemoryAgentBench* 等关注检索、长程理解与动态记忆。附录还讨论 *ReadAgent、MemGPT、A-MEM、LTM-benchmark*。作者指出维持长程一致性、处理动态记忆仍有限制。 |

阅读时应区分“模型能在静态题上多步推理”与“agent 能在环境中持续规划”：后者会受到工具返回值、状态变化和此前错误影响。对工具使用也是如此，结构上合法的单次调用不等同于多轮任务成功。附录 B 的细节正是用来说明评估对象如何从孤立步骤扩展到有依赖关系的轨迹。（PDF 第 2–3、20–24 页）

## 3 Application-Specific Agents Evaluation｜特定应用智能体评估

作者选择网页、软件工程、科学研究、对话四类代表性应用。不同领域的任务与可验证结果不同，因此评估环境和指标也各异；第 5 章再把这些差异归纳为共同维度。（PDF 第 3–5 页）

| 原文章节 | 主要发展脉络与关键 benchmark | 需要记住的评估问题 |
| --- | --- | --- |
| **3.1 Web Agents** | *WebShop* 是简化的购物模拟；*Mind2Web* 用真实网站的离线数据比较预测操作与标注动作；*WebArena* 提供可交互的沙盒网站和功能正确性测试。后续 benchmark 扩展多轮对话、知识工作者流程、多站点长任务与多模态 GUI；*ST-WebAgentBench* 明确测试政策遵守和风险缓解。*WebVoyager* 提供真实感较强的多模态在线评估，但论文引述研究认为其成绩可能过于乐观，并提到更严格的 *Online-Mind2Web*。 | 离线动作匹配能观察中间步骤，却看不到错误改变环境后的连锁后果；动态网站更适合测完整任务。网页评测还需考虑视觉信息与文本信息共同定位元素。 |
| **3.2 Software Engineering Agents** | 早期 *HumanEval* 一类题目偏短且自包含；*SWE-bench* 改用真实 GitHub issue、完整仓库、可执行环境和测试，评估补丁能否解决问题。后续版本修复题目描述、测试或环境质量问题；*SWE-bench Verified* 经人工筛选验证，形成 500 题高质量子集和容器化执行，作者称其为实际通用标准。*Terminal-Bench* 测交互式命令行；*SWE-Lancer* 包含 1,400 个自由职业任务及技术和管理决策；*SWE-bench Pro* 有 41 个仓库中的 1,865 个经人工验证的任务。 | 单段代码正确率不足以代表仓库级软件工程能力。题目、单测、环境配置都会影响可解释性。正文报告 *SWE-bench Pro* 的模型 **Pass@1 低于 25%**，借此强调长时程、多文件修改的难度。（PDF 第 4 页） |
| **3.3 Scientific Agents** | 从知识回忆、科学推理和文献理解，扩展到研究流程：**scientific ideation**（新颖、相关且可行的想法）、**experiment design**（假设与方法设计；如 *AAAR-1.0*）、**code generation**（可执行的科学代码；如 *SciCode、ScienceAgentBench、CORE-Bench、PaperBench*）和 **peer review**（有实质内容的评审）。附录 C 补充 *MLGym*、*DiscoveryWorld*、*LAB-Bench* 及 Deep Research 的检索质量、知识综合与可验证性。 | 科研 agent 评估正从单点能力移向连续研究流程，最终目标是评价具有创新性的科学发现；论文将其表述为发展方向，并未给出统一、成熟的发现质量指标。 |
| **3.4 Conversational Agents** | 目标导向多轮对话从纯文本 TODS 扩展到调用工具并改变环境。*τ-Bench* 用模拟用户、API 工具及领域政策测试客服任务；其局限包括规模、用户模拟设置，以及主要依赖粗粒度终局指标。*τ²-Bench* 加入电信领域、用户也可使用工具的共享动态环境，以及程序化组合任务生成器。*IntellAgent* 根据数据库 schema 和政策文档自动生成场景；*ALMITA* 使用自动生成与人工过滤结合的方法。 | 完成用户请求并不足以概括质量：是否违反政策、对话流程是否出错、工具与环境状态是否正确，都可能被单一终局成功率掩盖。 |

## 4 Generalist Agent Evaluation｜通用型智能体评估

通用型任务把规划、推理、工具、网页、文件处理和代码执行等能力组合起来。作者区分两条互补路径：（1）设计本身就要求多能力的综合任务；（2）把若干领域 benchmark 统一到同一个评测平台。（PDF 第 5 页）

| 路径 | 原文例子 | 读法 |
| --- | --- | --- |
| 综合任务 / 完整计算机环境 | *GAIA* 的现实问题要求推理、网页浏览、多模态与工具使用；简单题逐渐饱和后，*Gaia2* 转向含邮件、消息和日历应用的移动环境，并加入歧义、噪声、时间约束与多 agent 协作。*OSWorld* 通过 UI 操作，*AppWorld* 与 *Gaia2* 主要通过代码和 API 与环境交互。 | “通用”并不意味着同一种接口。比较成绩前要看任务集合与交互方式。 |
| 汇集多个专门 benchmark | *AgentBench* 涵盖操作系统、数据库、游戏和家居任务；*HAL（Holistic Agent Leaderboard）* 汇集编码和网页等 benchmark。作者指出，仅把 benchmark 放在一起，尚不足以在各种环境中评估同一 agent harness；*Harbor* 与 *Exgentic* 尝试用统一协议解决。 | 核心问题是跨环境的标准化和可比性，而不仅是增加测试题数量。 |

## 5 Core Benchmark Dimensions｜Benchmark 的核心维度

本章把前文按应用分类的材料，改从 **data curation、environment、interaction interface、metric、safety** 五个相对正交的设计维度比较。（PDF 第 5–6 页）

| 维度 | 作者的定义、例子与主要取舍 |
| --- | --- |
| **Data Curation（数据构建）** | 多数高质量 benchmark 采用混合方式：例如 *SWE-bench Verified* 在真实 issue 基础上人工核验；*Mind2Web* 清洗和标注真实交互；*AppWorld* 的合成世界用程序化检查验证。*GAIA* 则由人编写并验证问题。人工核验有助于有效性，自动化有助于规模和持续更新，两者需要兼顾。 |
| **Environment（环境）** | **Static** 环境用离线轨迹或缓存页面，让 agent 预测下一步而不改变状态；便于扩展，却无法体现错误的下游影响。**Dynamic** 环境允许 agent 在浏览器沙盒、容器等场景中行动并改变后续观察，因此更能诊断长任务的连锁失败。 |
| **Interaction Interface（交互接口）** | **Code and Terminal**：生成 Python、Bash、SQL 等可执行命令；**Tools**：按函数 schema 调用预定义工具；**GUI**：通过 DOM/无障碍树或视觉界面模拟人的操作。接口决定行动空间、观察空间和可测能力。 |
| **Metric（指标）** | 最常见的是任务完成度，但判定方式取决于任务：SWE 常用执行单元测试，*τ-Bench* 等检查最终环境状态，*GAIA* 对照标准短答案。只看二元终局结果无法解释 agent 在途中推进了多少、在哪里失败。 |
| **Safety and Robustness（安全与稳健性）** | 原文用 **pass^k** 表示同一任务在 *k* 次独立运行中都成功的任务占比。多数 benchmark 优先测能力，较少直接测隐私、访问控制和政策合规；作者主张增加 guardrail 指标，对通过违规行动取得的成功予以惩罚。 |

**Table 1 原文对照**（PDF 第 6 页；`Mix` 表示混合接口或混合指标，`Safety` 表示是否明确加入安全约束）：

| Benchmark | Data | Env. | Interface | Metric | Safety |
| --- | --- | --- | --- | --- | --- |
| SWE-bench Ver. | Hybrid | Dynamic | Code | Unit Tests | No |
| SWE-Lancer | Hybrid | Dynamic | Code | End-to-end | No |
| Mind2Web | Hybrid | Static | GUI | Action Match | No |
| WebArena | Hybrid | Dynamic | GUI | Mix | No |
| PaperBench | Hybrid | Dynamic | Code | End-to-end | No |
| TAU-Bench（正文写作 τ-Bench） | Hybrid | Dynamic | Tools | State Match | Yes |
| AppWorld | Hybrid | Dynamic | Tools | State Match | No |
| GAIA | Human | Dynamic | Mix | Answer Match | No |

这张表尤其说明两点：其一，*Mind2Web* 与 *WebArena* 都是网页评测，但静态/动态设置不同，不能把“网页任务”当成单一评估条件；其二，表中只有 *TAU-Bench* 被标为明确包含安全约束，这支撑作者关于安全维度覆盖不足的判断。`Safety: No` 只表示论文这张表的归类，不能推出该 benchmark 对任何安全问题都毫无涉及。

## 6 Frameworks for Agent Evaluation｜智能体评估框架

前几章的 benchmark 主要用固定任务和测试集评价成品系统；本章的框架则接入开发、部署过程，支持自定义场景、持续监控和故障分析。作者列举 *LangSmith、Langfuse、Google Vertex AI、Arize AI、Galileo、Patronus AI、AgentEvals、Mosaic AI* 等。大多数框架记录执行轨迹、任务完成、延迟等指标，但“能看见轨迹”不等于“能解释轨迹为何成败”。（PDF 第 7–8 页）

| 评估层次或方法 | 中文理解与局限 |
| --- | --- |
| **Final Response Evaluation** | 按正确性、相关性、忠实性、礼貌等标准检查最终答复，常用 LLM judge；便于大规模监控和回归测试，但无法判断中间决策、执行效率和失败原因。 |
| **Stepwise Evaluation** | 分别检查生成、工具调用、路由等步骤，帮助定位错误；也可检查工具选择、参数 schema 和工具结果是否可用。*Arize Phoenix* 提供规划、检索、反思等阶段模板；*Galileo* 的 action advancement 指标尝试衡量每一步是否推进用户目标。单独给每一步打分仍可能漏掉步骤之间的依赖。 |
| **Trajectory-Based Assessment** | **Reference-based** 把实际行动轨迹与预设路径比较，可用精确、部分、无序或子集匹配；*AgentEvals* 还可按图节点与转移比较。其难点是同一目标可有多条正确路径，人工写标准路径也昂贵。**Reference-free** 借 LLM judge 从连贯性、效率、目标导向等角度直接评估观察到的轨迹，更灵活但可靠性较弱。 |
| **Supporting Capabilities** | 人工参与标注（human in the loop）、从生产日志抽取评测集、合成测试数据，以及不同运行或实验设置的 A/B 对照。 |

**Table 2 原文能力矩阵**（PDF 第 8 页；✓ 为表中支持，× 为表中未支持；原文提醒部分能力仍在早期开发阶段）：

| Framework | Stepwise | Monitoring | Trajectory | Human in the Loop | Synthetic Data | A/B |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| LangSmith | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| Langfuse | ✓ | ✓ | × | ✓ | × | ✓ |
| Google Vertex AI evaluation | ✓ | ✓ | ✓ | × | × | ✓ |
| Arize AI’s Evaluation | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| Galileo Agentic Evaluation | ✓ | ✓ | × | ✓ | × | ✓ |
| Patronus AI | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| AgentEvals（表中作 AgentsEval） | × | × | ✓ | × | × | ✓ |
| Mosaic AI | ✓ | ✓ | × | ✓ | ✓ | ✓ |

作者指出的方法取舍是：有参考路径的评估更精确、可复现，却依赖预设的“正确行为”；无参考路径的评估更灵活，却要面对 judge 的可靠性。通用 judge 覆盖广，专用 judge 在目标标准上更精准但范围窄。框架当前还难以从大量运行中归纳根因或把 A/B 结果差异归因到具体步骤，较少计入评估过程本身的 LLM judge 成本，也普遍缺少内置安全/政策评测。

本章末尾另介绍 **Gym-like Environments**：受 OpenAI Gym 启发，以受控、可交互的动态模拟环境支持 agent 训练和评估，并在不同 benchmark 间尽量标准化。文中举 *BrowserGym*（网页）、*MLGym*（AI 研究）和 *SWE-Gym*（软件工程）为例。（PDF 第 8 页）

## 7 Discussion｜讨论

| 原文部分 | 作者的归纳 |
| --- | --- |
| **7.1 Current Trends：更真实、更难的评估** | 从简化的静态环境走向可交互环境、真实 GitHub issue 以及需要专业技能的长任务。这样才能检验实际工作价值并暴露 agent 的边界。 |
| **7.1 Current Trends：Live Benchmarks** | 固定题库会过时或趋于饱和，因此出现持续更新的版本系列，如 *BFCL* 与 *SWE-bench Verified / Pro*。更新可适配更强模型、修复旧版题目和成功指标，并回应 MCP 等新的工具生态。这里的“live”主要强调持续维护与适应，不等同于所有任务都必须连到实时公网。 |
| **7.2 Future Directions：细粒度评估** | 从粗粒度 end-to-end success 扩展到对执行轨迹、工具选择和推理步骤的标准化诊断，给出能指导修改的反馈。 |
| **7.2 Future Directions：成本和效率** | 把 token 使用量、API 费用、推理时间和资源消耗纳入核心指标，避免只因成功率高就忽略部署代价。 |
| **7.2 Future Directions：扩展与自动化** | 减少对静态人工标注的单一依赖，探索合成数据和 LLM/agent-as-a-judge，同时保留对评估质量的关注。 |
| **7.2 Future Directions：安全与合规** | 增加针对对抗输入、偏见、组织政策和社会规范的测试，尤其注意多 agent 场景中的新风险。 |
| **7.2 Future Directions：解耦模型与 harness** | 现有 benchmark 往往混合衡量 backbone LLM 和 agent harness。作者主张通过受控实验分别改变模型、harness 与记忆/规划模块，以归因性能改进；*Harbor* 和 *Exgentic* 是统一评估协议的早期尝试。 |

这些是作者从文献中提出的趋势与研究方向，不是论文已经验证完成的解决方案。（PDF 第 8–9 页）

## 8 Conclusion｜结论

作者把领域变化概括为：从在简化设置里测试单项能力，逐步转向在真实、动态、复杂环境中评估完整 agent。论文最后重申，未来需要超越总体成功率，建立更细、可扩展的方法，以及成本效率、安全和稳健性的标准化指标；实践中的 benchmark 选择见附录 E。（PDF 第 9 页）

## Limitations｜论文自述局限

综述反映的是写作时点的研究快照；新 benchmark、框架和架构不断出现。为保持主线，作者选择具有代表性的工作，未穷尽小众方法；覆盖范围广也限制了逐项深入程度。第 7 章对未来方向的判断包含基于现有趋势的解释，实际发展路径仍可能改变。作者提到另有持续维护的 GitHub 清单来缓解时效问题。（PDF 第 10 页）

## 附录 A–E｜补充内容与复习索引

| 原文章节 | 对正文的补充 |
| --- | --- |
| **A Literature Review Methodology** | 通过 Google Scholar、ACL Anthology、HuggingFace Papers、arXiv 的关键词检索，前向/后向引文追踪，结构化纳入与排除标准，以及领域专家咨询选文。主要纳入提出新 benchmark、评估框架或重要方法贡献的论文；单纯提出 agent 架构或仅做传统静态 LLM 评测的工作不在重点范围。 |
| **B Agent Capabilities Evaluation** | 细述第 2 章四项能力的 benchmark 和相互关系；其中规划强调任务分解、状态/信念维护、自纠错、因果理解与元规划；工具使用补充参数映射、执行与结果整合；自我反思说明只测最终答案修正过于粗糙；记忆扩展到长期、多任务交错、外部反馈和效率。附录的 Figure 2 是比 Figure 1 更长的代表工作索引。 |
| **C Application-Specific Agents Evaluation** | 在科学 agent 部分补充端到端科研工作流、虚拟发现环境、生物领域任务，以及 Deep Research 评估中的检索质量、知识综合与可验证性。 |
| **D Generalist Agent Evaluation** | 补充真实工作场所场景：*TheAgentCompany* 模拟软件公司任务，*CRMArena* 模拟需要 UI/API 操作、政策遵守和多源数据整合的客户关系管理任务。 |
| **E Benchmark Recommendations** | 作者按应用给出实践建议：网页任务按动态性和模态选 *WebArena / Mind2Web / WebVoyager* 等；SWE 常用 *SWE-bench Verified*，更难任务参考 *SWE-bench Pro*，终端任务用 *Terminal-Bench*；对话任务常用 *τ-Bench*；通用任务按推理、工具、GUI、跨 benchmark 比较分别考虑 *GAIA、AppWorld、OSWorld、HAL*。科学任务选择高度依赖具体研究环节。 |

**数字口径提醒：**附录 E 提到 *SWE-bench Pro* 的当时 SOTA 约 **46%**，而正文第 3.2 节写的是模型 **Pass@1 低于 25%**；原文没有在这两处交代可直接比较的同一评测设置，因此复习时应分别按所在段落理解。附录 E 的网页和 SWE 成绩也属于作者写作时记录的时点数据，不是本文笔记对当前排行榜的核查。（PDF 第 4、24–25 页）

**复习主线：**先问“测哪一类 agent 或能力”（第 2–4 章），再问“任务数据、环境、接口、指标和安全约束如何设定”（第 5 章），最后看“开发过程中怎样观察步骤、轨迹与结果”（第 6 章）。第 7 章解释为什么这些设置需要更真实、可持续更新，并指出仍缺什么。
