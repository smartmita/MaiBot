## **HFC 机器人日程系统 (Schedule) 逻辑分析**

### **1\. 引言**

HFC 系统的日程模块 (schedule) 旨在为机器人模拟一个每日的活动计划和当前的行动状态。这个模块使得机器人不仅仅是对聊天内容做出反应，还能表现出拥有“自己生活”的特征，例如在特定时间“正在做”某事。这有助于增强机器人的真实感和角色扮演的深度。

核心功能包括：

* **每日日程生成与优化:** 每天为机器人生成一份详细的日程安排，并结合知识库信息进行优化。  
* **当前活动追踪:** 根据生成的日程和当前时间，动态更新机器人“正在做”的事情。  
* **持久化存储:** 将生成的日程和已完成的活动记录到数据库中。

### **2\. 核心组件与流程**

#### **2.1. ScheduleGenerator 类**

这是日程系统的核心类，负责所有与日程相关的操作。

* **初始化 (\_\_init\_\_, initialize):**  
  * 加载多个专门用于日程生成的 LLM 模型实例：  
    * llm\_schedule\_initial\_generator: 用于生成每日日程的初始版本。  
    * llm\_keyword\_extractor: 用于从初始日程中提取关键词。  
    * llm\_schedule\_refiner: 用于结合知识库信息优化日程。  
    * llm\_schedule\_current\_activity: 用于根据当前时间和日程推断机器人正在进行的活动。  
  * 存储机器人的基本信息，如名称 (name)、行为习惯 (behavior) 和个性化实例 (individuality)。  
  * 初始化用于存储当日日程文本 (today\_schedule\_text) 和已完成活动列表 (today\_done\_list) 的变量。  
  * 记录昨日日程 (yesterday\_schedule\_text) 和昨日已完成活动 (yesterday\_done\_list)，作为今日日程生成的参考。  
* **主循环 (mai\_schedule\_start):**  
  * 这是一个异步的后台任务，在日程系统启用时 (global\_config.schedule.enable \== True) 运行。  
  * **每日检查与生成:** 在每天开始或系统启动时，调用 check\_and\_create\_today\_schedule() 来确保当天的日程存在。如果日期发生变化，会重新生成日程。  
  * **周期性更新当前活动:** 以设定的时间间隔（schedule\_doing\_update\_interval，默认为300秒，即5分钟）调用 move\_doing() 来更新机器人当前正在进行的活动。

#### **2.2. 每日日程生成 (check\_and\_create\_today\_schedule)**

此函数负责确保机器人每天都有一份日程安排，其主要步骤如下：

1. **加载历史日程:**  
   * 从数据库加载昨天的日程 (yesterday\_schedule\_text) 和已完成活动列表 (yesterday\_done\_list)。  
   * 尝试从数据库加载今天的日程 (raw\_today\_schedule\_text\_from\_db) 和已完成活动列表 (loaded\_today\_done\_list)。  
2. **生成新日程 (如果今日日程不存在):**  
   * **步骤 1: 生成初版日程 (generate\_daily\_schedule\_initial):**  
     * 调用 construct\_daytime\_prompt() 构建提示词，该提示词包含机器人的名字、个性、行为习惯、昨日日程以及对今日日程的具体要求（如详细到每半小时，1500字以上）。  
     * 使用 llm\_schedule\_initial\_generator 模型根据上述提示词生成一份初步的日程文本。  
   * **步骤 2: 提取关键词并检索知识 (extract\_keywords\_from\_schedule, retrieve\_knowledge\_for\_keywords):**  
     * 调用 construct\_keyword\_extraction\_prompt() 构建提示词，要求 LLM 从初版日程中提取所有名词（人名、地名、事件名等）。  
     * 使用 llm\_keyword\_extractor 模型提取关键词列表。  
     * 遍历提取到的关键词，针对每个关键词调用 qa\_manager.get\_knowledge() 从知识库中检索相关信息。  
     * retrieve\_knowledge\_for\_keywords 内部使用 extract\_individual\_knowledge\_pieces 和 more\_robust\_normalize 对检索到的知识进行拆分、规范化和去重，并根据相关性 (parse\_knowledge\_and\_get\_max\_relevance) 筛选，最终选取最相关的几条知识。  
   * **步骤 3: 结合知识优化日程 (refine\_schedule\_with\_knowledge):**  
     * 如果检索到了相关的知识库信息，则调用 construct\_schedule\_refinement\_prompt() 构建提示词。该提示词包含初版日程、检索到的知识库信息，并要求 LLM 根据这些信息修改和完善日程，使其更符合角色设定和背景故事。  
     * 使用 llm\_schedule\_refiner 模型生成优化后的日程文本。  
     * 如果未能提取关键词或未检索到相关知识，或者优化步骤失败，则会使用初版日程作为最终日程。  
   * **步骤 4: 保存最终日程:** 将最终确定的日程文本 (self.today\_schedule\_text) 和空的已完成活动列表 (self.today\_done\_list) 保存到数据库 (save\_today\_schedule\_to\_db)。  
3. **使用现有日程 (如果今日日程已存在于数据库):**  
   * 直接使用从数据库加载的 raw\_today\_schedule\_text\_from\_db 作为当日日程。  
   * self.today\_done\_list 也已从数据库加载。

#### **2.3. 当前活动更新 (move\_doing)**

此函数周期性地更新机器人“当前正在做的事情”。

1. **构建提示词 (construct\_doing\_prompt):**  
   * 提示词包含当前时间、机器人名称、个性、行为习惯、今日完整日程 (today\_schedule\_text)。  
   * 还可能包含最近几次已完成的活动 (get\_current\_num\_task) 和可选的机器人当前“内心想法” (mind\_thinking)。  
   * 要求 LLM 结合这些信息，推断出机器人当前正在进行的具体活动。  
2. **LLM 推断活动:**  
   * 使用 llm\_schedule\_current\_activity 模型根据上述提示词生成当前活动的描述。  
3. **记录与更新:**  
   * 将 LLM 返回的活动描述和当前时间作为一个元组 (current\_time, doing\_response) 添加到 self.today\_done\_list 中。  
   * 调用 update\_today\_done\_list() 将更新后的 today\_done\_list 写回数据库。

#### **2.4. 辅助功能**

* **提示词构建:**  
  * construct\_daytime\_prompt: 为生成每日初始日程构建提示词。  
  * construct\_doing\_prompt: 为推断当前活动构建提示词。  
  * construct\_keyword\_extraction\_prompt: 为从日程中提取关键词构建提示词。  
  * construct\_schedule\_refinement\_prompt: 为结合知识优化日程构建提示词。  
* **知识处理:**  
  * parse\_knowledge\_and\_get\_max\_relevance: 解析从 qa\_manager 返回的知识字符串，提取内容和最高相关性分数。  
  * extract\_individual\_knowledge\_pieces: 将包含多条知识的字符串块拆分为独立的知识片段及其相关性。  
  * more\_robust\_normalize: 对文本进行规范化处理（如去除非中英文字符、转小写、去多余空格）以方便比较和去重。  
* **数据库交互:**  
  * save\_today\_schedule\_to\_db: 将当日日程和已完成活动列表保存或更新到 MongoDB 的 schedule 集合中（按日期索引）。  
  * load\_schedule\_from\_db: 从数据库加载指定日期的日程和已完成活动列表。  
* **日程查询:**  
  * get\_current\_num\_task: 获取最近 N 条已完成的活动。  
  * get\_task\_from\_time\_to\_time: 获取指定时间范围内的已完成活动列表（此功能在代码中实现，但未被主要逻辑直接调用）。

### **3\. 数据存储**

日程相关数据存储在 MongoDB 数据库的 schedule 集合中。每条记录（文档）通常包含以下字段：

* date: 日期字符串 (YYYY-MM-DD格式)。  
* schedule: 当日的完整日程安排文本。  
* today\_done\_list: 一个列表，包含当天已完成的活动，每个活动是一个元组 (timestamp, activity\_description)。

### **4\. LLM 的作用**

日程模块深度依赖 LLM 完成以下任务：

* **生成初始日程:** 根据角色设定和昨日情况，创作全新的每日日程。  
* **提取关键词:** 从文本中识别重要的名词性关键词。  
* **优化日程:** 结合外部知识（从知识库检索得到），对初始日程进行符合逻辑和角色背景的修改与丰富。  
* **推断当前活动:** 基于完整日程和当前时间点，判断机器人“正在做什么”。

### **5\. 总结**

HFC 的日程系统通过一系列精心设计的 LLM调用和知识库交互，为机器人构建了一个动态的、具有一定真实感的每日活动计划。它不仅生成静态的日程表，还能根据时间推移更新机器人当前所处的状态，这些信息可以被其他模块（如思考模块 Mind 或提示词构建模块 PromptBuilder）利用，以生成更情境化、更符合角色当前状态的聊天回复。该系统通过每日的重新生成和优化，以及周期性的当前活动更新，使得机器人的“生活”能够持续进行。