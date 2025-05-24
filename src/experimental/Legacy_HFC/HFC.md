## **HFC**

### **1\. 引言**

HFC (Heart Flow Chat) 是一个设计用于 QQ 群聊的机器人系统。其核心目标是模拟真人在群聊中的行为，能够根据群聊上下文、自身状态和预设的个性进行观察、思考、决策并作出回应。本文档将重点解析 HFC 系统在处理群聊消息时的核心逻辑流程和关键组件。

### **2\. 核心组件**

HFC 系统由多个协同工作的核心组件构成：

* **HeartFCProcessor (消息处理器):**  
  * 作为消息的入口，负责接收原始消息数据。  
  * 对消息进行初步解析、缓冲处理和过滤（如屏蔽词检测）。  
  * 计算消息的初始兴趣度。  
  * 将处理后的消息传递给主心流 (Heartflow) 进行后续处理。  
* **Heartflow (主心流协调器):**  
  * 系统的中央协调单元。  
  * 管理机器人的整体状态 (MaiStateInfo)。  
  * 管理所有子心流 (SubHeartflow) 的生命周期和交互，通过 SubHeartflowManager 实现。  
  * 执行主心流思考 (Mind)，形成对整体环境的宏观认知。  
  * 启动和管理后台任务 (BackgroundTaskManager)，如状态更新、日志记录等。  
* **MaiStateManager 与 MaiStateInfo (机器人状态管理器):**  
  * MaiStateInfo: 存储机器人当前的全局状态，如 OFFLINE (不在线), PEEKING (窥屏), NORMAL\_CHAT (正常聊天), FOCUSED\_CHAT (专注聊天)。  
  * MaiStateManager: 根据预设规则和当前状态信息，决策机器人全局状态的转换。全局状态会影响机器人能同时处理的普通聊天和专注聊天的数量上限。  
* **SubHeartflowManager (子心流管理器):**  
  * 负责创建、获取、管理和销毁所有的 SubHeartflow 实例。  
  * 根据机器人全局状态 (MaiState) 和各子心流的活跃度，强制执行子心流数量限制。  
  * 处理子心流状态转换的逻辑，例如：  
    * sbhf\_absent\_into\_chat: 评估处于 ABSENT 状态的群聊是否应转为 CHAT 状态。  
    * sbhf\_absent\_into\_focus: 评估处于 ABSENT 或 CHAT 状态的群聊是否应转为 FOCUSED 状态。  
    * sbhf\_chat\_into\_absent: 将长时间不活跃的 CHAT 状态群聊转为 ABSENT。  
    * sbhf\_focus\_into\_absent\_or\_chat: 处理从 FOCUSED 状态退出的逻辑，对于群聊可能转为 ABSENT 或 CHAT。  
* **SubHeartflow (子心流):**  
  * 代表机器人与特定聊天对象（在此特指某个 QQ 群）的互动会话。  
  * 每个群聊对应一个 SubHeartflow 实例。  
  * 管理自身在特定群聊中的状态 (ChatStateInfo)，包括 ABSENT (不在看群), CHAT (随便水群), FOCUSED (认真水群)。  
  * 包含观察模块 (Observation)、子思维模块 (SubMind) 和兴趣聊天模块 (InterestChatting)。  
  * 根据自身状态，激活并管理 NormalChat (普通聊天) 或 HeartFChatting (专注聊天) 实例。  
* **ChatStateInfo (聊天状态信息):**  
  * 存储 SubHeartflow 在特定群聊中的当前状态 (ABSENT, CHAT, FOCUSED) 和状态持续时间等信息。  
* **Observation / ChattingObservation (观察模块):**  
  * 负责观察特定聊天流（群聊）中的新消息。  
  * 维护最近的聊天记录，并能在消息过多时进行压缩总结（通过 LLM）。  
  * 提供当前聊天上下文给思考和决策模块。  
* **InterestChatting (兴趣聊天模块):**  
  * 追踪机器人在特定群聊中的“兴趣等级”。  
  * 兴趣等级会随新消息的刺激而增加，随时间自然衰减。  
  * 当兴趣等级达到一定阈值时，会增加机器人进入专注聊天模式的概率。  
* **SubMind (子思维):**  
  * 负责特定 SubHeartflow 的“思考”过程。  
  * 结合观察到的聊天内容、自身历史想法、机器人全局状态、当前日程、情绪、知识库信息等，生成当前子心流的“内心想法”。  
  * 其输出的“内心想法”会作为 HeartFChatting 中规划器 (Planner) 的重要输入。  
  * 可以调用工具 (ToolUser) 获取额外信息。  
* **NormalChat (普通聊天控制器):**  
  * 当 SubHeartflow 处于 CHAT 状态时激活。  
  * 以较低的频率和参与度处理群聊消息。  
  * 根据消息的兴趣度、机器人意愿等因素决定是否回复。  
  * 回复生成也依赖 LLM (NormalChatGenerator)。  
* **HeartFChatting (专注聊天控制器):**  
  * 当 SubHeartflow 处于 FOCUSED 状态时激活。  
  * 实现一个核心的“观察-思考-规划-行动”循环，以较高频率和深度参与群聊。  
  * 包含 ActionManager 来管理当前可执行的动作。  
* **ActionManager (动作管理器):**  
  * 定义了机器人（在 HeartFChatting 模式下）可以执行的动作集合，如：不回复、文本回复、表情回复、结束专注模式。  
  * 可以根据上下文动态调整可用动作。  
* **HeartFCGenerator / NormalChatGenerator (回复生成器):**  
  * 封装了与大语言模型 (LLM) 的交互，用于生成实际的聊天回复文本。  
  * HeartFCGenerator (逻辑内化于 HeartFChatting.\_replier\_work) 用于专注聊天。  
  * NormalChatGenerator 用于普通聊天。  
* **HeartFCSender (消息发送器):**  
  * 负责将机器人生成的回复消息实际发送到 QQ 群。  
  * 管理消息的发送状态（如“正在思考”）。  
* **PromptBuilder (提示词构建器):**  
  * 根据当前上下文、机器人状态、个性设定等动态构建适用于不同场景（如规划、回复生成、总结等）的 LLM 提示词。

### **3\. 群聊消息核心处理流程**

#### **3.1. 消息接收与初步处理**

1. **接收消息:** HeartFCProcessor 接收到来自 QQ 群的消息。  
2. **预处理:** 进行消息解析、格式化，并检查是否包含屏蔽词或触发过滤规则。  
3. **缓冲与流管理:** message\_buffer 处理消息缓冲，chat\_manager 获取或创建对应的 ChatStream 对象。  
4. **兴趣计算:** HeartFCProcessor 调用海马体管理器 (HippocampusManager) 计算消息的初步兴趣度。如果消息中提及机器人，则兴趣度会增加。  
5. **子心流获取/创建:** HeartFCProcessor 将消息和兴趣度信息传递给 Heartflow。Heartflow 通过 SubHeartflowManager 获取或创建对应群聊的 SubHeartflow 实例。新创建的 SubHeartflow 会添加 ChattingObservation。  
6. **兴趣更新:** SubHeartflow 内的 InterestChatting 模块根据传入的兴趣值更新当前群聊的兴趣等级，并将消息存入兴趣字典。

#### **3.2. 机器人全局状态与子心流状态管理**

* **全局状态 (MaiState):**  
  * 由 MaiStateManager 根据预设规则（如持续时间、随机概率）和当前活动情况（如专注聊天数量）进行更新。  
  * 全局状态会影响 SubHeartflowManager 对子心流数量的限制（例如，NORMAL\_CHAT 状态下允许的普通聊天路数和专注聊天路数比 PEEKING 状态多）。  
* **子心流状态 (ChatState):**  
  * 每个 SubHeartflow 独立管理其在对应群聊中的状态：ABSENT, CHAT, FOCUSED。  
  * **状态转换驱动因素:**  
    * **兴趣驱动 (ABSENT/CHAT \-\> FOCUSED):** SubHeartflowManager.sbhf\_absent\_into\_focus 周期性检查。如果某群聊的 InterestChatting 模块计算出的“进入专注模式概率” (start\_hfc\_probability) 达到阈值，并且通过了 LLM 的进一步判断（如果启用了LLM判断），且当前专注聊天数量未达上限，则该群聊的 SubHeartflow 状态会提升为 FOCUSED。  
    * **LLM决策驱动 (ABSENT \-\> CHAT):** SubHeartflowManager.sbhf\_absent\_into\_chat 周期性随机选取一个 ABSENT 状态的群聊，使用 LLM 评估是否应该开始在该群“随便水水”(CHAT)，若决策为是且普通聊天数量未达上限，则转换为 CHAT。  
    * **不活跃超时 (CHAT \-\> ABSENT):** SubHeartflowManager.sbhf\_chat\_into\_absent 周期性检查 CHAT 状态的群聊，若长时间（如超过30-60分钟）机器人未在其中发言，则状态降为 ABSENT。  
    * **专注聊天退出 (FOCUSED \-\> ABSENT/CHAT):** 当 HeartFChatting 内部决策退出专注模式，或连续不回复达到阈值时，会调用 SubHeartflowManager.\_handle\_hfc\_no\_reply，进而触发 SubHeartflowManager.sbhf\_focus\_into\_absent\_or\_chat。对于群聊，会根据随机概率和普通聊天限额决定是转为 ABSENT 还是 CHAT。

#### **3.3. 普通聊天模式 (NormalChat \- CHAT 状态)**

1. **激活:** 当 SubHeartflow 状态变为 CHAT 时，会创建并启动 NormalChat 实例。  
2. **消息处理:** NormalChat 实例会处理其 interest\_dict 中的消息（这些消息是在 HeartFCProcessor 阶段因兴趣度而被加入的）。  
3. **回复决策:**  
   * willing\_manager (意愿管理器) 评估回复意愿和概率。如果消息中提及机器人，回复概率较高。  
   * 若未被提及，则综合考虑兴趣度、机器人当前状态等多种因素计算回复概率。  
4. **回复生成:** 若决定回复，NormalChatGenerator 会构建提示词并调用 LLM 生成回复内容。  
5. **发送与后续处理:**  
   * 通过 HeartFCSender (间接通过 message\_manager) 发送消息。  
   * 可能附带表情 (\_handle\_emoji)。  
   * 更新与发言者的关系 (\_update\_relationship)。  
   * 触发绰号分析 (sobriquet\_manager)。

#### **3.4. 专注聊天模式 (HeartFChatting \- FOCUSED 状态)**

1. **激活:** 当 SubHeartflow 状态变为 FOCUSED 时，会创建并启动 HeartFChatting 实例。  
2. **核心循环 (\_hfc\_loop):** HeartFChatting 以固定周期执行核心的“观察-思考-规划-行动”循环。  
   * **观察与思考 (\_get\_submind\_thinking):**  
     * ChattingObservation.observe(): 更新当前群聊的最新消息和上下文。  
     * SubMind.do\_thinking\_before\_reply(): 结合观察到的聊天内容、历史想法、机器人全局信息、工具调用结果（如果有）等，生成当前周期的“内心想法” (current\_mind)。此过程可能涉及 LLM 调用。  
   * **动态重规划检查:** 在首次思考后、Planner 决策前，系统会检查在思考期间是否有新消息。若有，且满足一定条件（如新消息数量和概率），则会重新进行观察和思考。  
   * **规划 (\_planner):**  
     * PromptBuilder.build\_planner\_prompt(): 构建规划阶段的提示词，包含当前聊天记录、SubMind 的想法、历史动作、可用动作列表等。  
     * LLM 调用: LLM 根据提示词，以 JSON 格式返回决策结果，包括：  
       * action: 选定的动作 (如 text\_reply, no\_reply, emoji\_reply, exit\_focus\_mode)。  
       * reasoning: 做出该决策的理由。  
       * emoji\_query (可选): 若要发送表情，提供表情的主题。  
       * at\_user (可选): 若要 @某人，提供用户 ID。  
       * poke\_user (可选): 若要戳一戳某人，提供用户 ID。  
     * ActionManager 会根据历史动作（如连续文本回复次数）临时移除某些可选动作，以避免行为单调。  
   * **行动 (\_handle\_action):** 根据 Planner 返回的 action 执行相应操作。  
     * **text\_reply (文本回复):**  
       * \_create\_thinking\_message(): 发送“正在思考”状态。  
       * \_replier\_work():  
         * PromptBuilder.build\_prompt(build\_mode="focus", ...): 构建回复生成阶段的提示词，包含 Planner 的决策理由、SubMind 的想法、聊天上下文等。  
         * LLM 调用: 生成回复文本。  
       * \_sender(): 通过 HeartFCSender 发送生成的文本回复，可能附带 Planner 指定的表情、@或戳一戳。  
       * 触发绰号分析 (sobriquet\_manager)。  
       * 重置“连续不回复计数器”。  
     * **emoji\_reply (表情回复):**  
       * \_handle\_emoji(): 根据 emoji\_query 查找并发送表情。  
     * **no\_reply (不回复):**  
       * 记录不回复的原因。  
       * \_wait\_for\_new\_message(): 等待一段时间或新消息的到来。  
       * 更新“连续不回复计数器” (\_lian\_xu\_bu\_hui\_fu\_ci\_shu) 和“累计等待时间” (\_lian\_xu\_deng\_dai\_shi\_jian)。  
       * 如果连续不回复次数和累计等待时间均达到阈值，则调用 on\_consecutive\_no\_reply\_callback (即 SubHeartflowManager.\_handle\_hfc\_no\_reply)，请求退出 FOCUSED 状态。  
     * **exit\_focus\_mode (结束专注模式):**  
       * Planner 主动决策结束专注。  
       * 调用 on\_consecutive\_no\_reply\_callback，请求退出 FOCUSED 状态。  
   * **循环延迟:** 每个循环结束后会有短暂延迟，避免过快消耗资源。  
   * **循环信息记录:** 每个循环的耗时、动作、思考ID等信息会记录在 CycleInfo 对象中，并存入历史。

#### **3.5. 消息发送**

* HeartFCSender 负责实际的消息发送逻辑。  
* 在发送文本消息前，会先注册一个 MessageThinking 对象，表示机器人正在输入/思考。  
* 发送时，将 MessageThinking 对象替换为 MessageSending 对象。  
* 支持发送文本、表情、@消息、戳一戳等。  
* 发送的消息会被存储到 MessageStorage。

### **4\. 关键决策点**

* **是否从 ABSENT 进入 CHAT/FOCUSED:** 由 SubHeartflowManager 中的 LLM 评估 (ABSENT-\>CHAT) 或兴趣概率+LLM评估 (ABSENT/CHAT-\>FOCUSED) 决定。  
* **是否从 CHAT/FOCUSED 返回 ABSENT:** 因不活跃超时 (CHAT-\>ABSENT) 或专注聊天内部决策/不回复超时 (FOCUSED-\>ABSENT/CHAT)。  
* **专注聊天中是否回复 (HeartFChatting.\_planner):** 基于 SubMind 的想法、聊天上下文、可用动作等，由 LLM 决定。  
* **普通聊天中是否回复 (NormalChat.normal\_response):** 基于提及、兴趣度、意愿模型等计算出的概率决定。  
* **回复类型和内容:** 由 Planner (专注聊天) 或直接的回复生成 LLM (普通聊天) 决定。

### **5\. 总结**

HFC 系统通过多层次的状态管理（全局机器人状态和各群聊的子心流状态）以及基于 LLM 的复杂决策机制，实现了在 QQ 群聊中进行拟人化交互的能力。系统能够根据上下文动态调整其参与度，从简单的“窥屏”到深度的“专注聊天”，并通过内部的思考和规划过程生成恰当的回复。兴趣机制和动作管理进一步丰富了其行为模式，使其表现更接近真实用户。