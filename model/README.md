# 智能体记忆系统

基于 Atkinson-Shiffrin 记忆模型构建的智能体认知架构，并融合了斯坦福 Generative Agents 等现代认知架构的扩展。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from memory import MemoryManager, MemoryManagerConfig
from memory import BaseMemoryEncoder, BaseMemoryRetriever

# 1. 实现你的编码器
class MyEncoder(BaseMemoryEncoder):
    def encode(self, input_data, **kwargs):
        # 在此实现你的编码逻辑
        pass
    
    def get_target_memory_type(self, input_data):
        return MemoryType.SHORT_TERM
    
    def estimate_importance(self, input_data):
        return MemoryPriority.MEDIUM

# 2. 实现你的检索器
class MyRetriever(BaseMemoryRetriever):
    def retrieve(self, query):
        # 在此实现你的检索逻辑
        pass
    
    def compute_relevance(self, query, memory):
        return memory.get_strength()

# 3. 初始化并使用
manager = MemoryManager(config=MemoryManagerConfig())
manager.set_encoder(MyEncoder())
manager.set_retriever(MyRetriever(manager.get_stores()))

# 存储记忆
manager.encode_and_store("用户说你好")

# 检索记忆
results = manager.retrieve_by_text("用户说了什么？")

# 运行维护操作
manager.run_maintenance()
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      记忆管理器                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   感觉记忆   │───▶│   短期记忆   │───▶│   长期记忆   │  │
│   │  (Sensory)   │    │    (STM)     │    │    (LTM)     │  │
│   │  (< 1 秒)    │    │  (15-30 秒)  │    │   (永久)     │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                    │                   │         │
│          │            ┌──────────────┐            │         │
│          │            │   工作记忆   │            │         │
│          │            │  (Working)   │            │         │
│          │            └──────────────┘            │         │
│          ▼                                        ▼         │
│   ┌──────────────────────────────────────────────────────┐  │
│   │       巩固 (Consolidation) • 遗忘 (Forgetting)        │  │
│   │       反思 (Reflection)                               │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 记忆类型

- **感觉记忆 (Sensory Memory)**: 感官信息的短暂存储 (< 1 秒)，高容量、短持续时间
- **短期记忆 (Short-Term Memory)**: 临时存储，容量有限 (7±2 项)，持续 15-30 秒
- **工作记忆 (Working Memory)**: 信息的主动处理和操作
- **长期记忆 (Long-Term Memory)**: 永久存储，包含：
  - 情景记忆 (Episodic): 个人经历和事件
  - 语义记忆 (Semantic): 事实和一般知识
  - 程序性记忆 (Procedural): 技能和操作

### 记忆条目

- `MemoryItem`: 所有记忆条目的基类
- `SensoryMemoryItem`: 带有模态信息的感觉印象
- `EpisodicMemoryItem`: 带有时间和空间上下文的个人经历
- `SemanticMemoryItem`: 事实和一般知识

### 记忆状态

- `ENCODED`: 刚编码完成
- `STORED`: 已存储
- `CONSOLIDATED`: 已巩固
- `RETRIEVED`: 已检索
- `REHEARSED`: 已复述
- `FORGOTTEN`: 已遗忘
- `DECAYED`: 已衰减

### 记忆优先级

- `CRITICAL`: 关键 (4)
- `HIGH`: 高 (3)
- `MEDIUM`: 中 (2)
- `LOW`: 低 (1)
- `TRIVIAL`: 微不足道 (0)

## 工具模块

### 记忆巩固 (Consolidation)

将记忆从短期记忆转移到长期记忆，并加强现有长期记忆。

```python
# 运行巩固
result = manager.run_consolidation()
print(f"巩固了 {len(result.consolidated_ids)} 条记忆")
```

### 记忆遗忘 (Forgetting)

实现自然的记忆衰减和干扰机制。

```python
# 运行遗忘周期
result = manager.run_forgetting_cycle()
print(f"遗忘了 {len(result.forgotten_ids)} 条记忆")
```

### 记忆反思 (Reflection)

从经验中提取高级洞察，参考斯坦福 Generative Agents 的反思机制。

```python
# 运行反思
result = manager.run_reflection(focus="用户偏好")
for insight in result.insights:
    print(f"洞察: {insight}")
```

## 需要实现的部分

你需要实现以下核心接口：

### 1. 编码器 (`BaseMemoryEncoder`)

```python
class MyEncoder(BaseMemoryEncoder):
    def encode(self, input_data, **kwargs):
        """
        将原始输入编码为记忆条目
        
        实现建议：
        - 使用句子转换器生成文本嵌入
        - 使用视觉模型处理图像
        - 使用多模态模型处理混合输入
        """
        pass
    
    def get_target_memory_type(self, input_data):
        """
        确定输入应存储为哪种记忆类型
        
        考虑因素：
        - 输入类型 (文本、图像、事件等)
        - 时间上下文
        - 情感显著性
        """
        pass
    
    def estimate_importance(self, input_data):
        """
        评估输入的重要性
        
        考虑因素：
        - 情感意义
        - 任务相关性
        - 新颖性
        - 来源可信度
        """
        pass
    
    def generate_embedding(self, content):
        """
        生成内容的向量嵌入
        
        实现建议：
        - 文本: 使用 sentence-transformers
        - 图像: 使用 CLIP 或类似模型
        - 多模态: 使用多模态嵌入模型
        """
        pass
```

### 2. 检索器 (`BaseMemoryRetriever`)

```python
class MyRetriever(BaseMemoryRetriever):
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        根据查询检索记忆
        
        实现建议：
        - 向量相似度搜索 (语义检索)
        - 关键词匹配 (词汇检索)
        - 混合策略
        """
        pass
    
    def compute_relevance(self, query, memory):
        """
        计算查询与记忆之间的相关性分数
        
        考虑因素：
        - 语义相似度 (嵌入余弦相似度)
        - 关键词重叠
        - 时间邻近性
        - 记忆强度/重要性
        """
        pass
```

### 3. 可选自定义

- **自定义巩固逻辑**: 继承 `MemoryConsolidator`
- **自定义遗忘机制**: 继承 `MemoryForgetting`
- **自定义反思生成**: 继承 `MemoryReflection`

## 配置选项

```python
config = MemoryManagerConfig(
    # 存储容量
    sensory_capacity=1000,      # 感觉记忆容量
    stm_capacity=7,             # 短期记忆容量 (Miller's Law: 7±2)
    ltm_capacity=None,          # 长期记忆容量 (None 表示无限)
    working_capacity=4,         # 工作记忆容量
    
    # 衰减设置
    sensory_decay_seconds=1.0,  # 感觉记忆衰减时间
    stm_decay_seconds=30.0,     # 短期记忆衰减时间
    forgetting_decay_rate=0.1,  # 遗忘衰减率
    
    # 巩固设置
    consolidation_threshold=0.5,  # 巩固阈值
    association_threshold=0.7,    # 关联创建阈值
    
    # 反思设置
    reflection_threshold=10,        # 触发反思的记忆数量
    min_importance_for_reflection=0.5,  # 反思最小重要性
    
    # 自动处理
    auto_consolidate=True,       # 自动巩固
    auto_forget=True,            # 自动遗忘
    auto_cleanup_sensory=True,   # 自动清理感觉记忆
    
    # 持久化
    persist_file="memory_state.json",  # 状态保存文件
)
```

## 目录结构

```
memory/
├── __init__.py              # 模块入口
├── memory_manager.py        # 记忆管理器
├── README.md                # 使用文档
│
├── core/                    # 核心数据结构
│   ├── __init__.py
│   ├── memory_types.py      # 记忆类型、优先级、状态
│   ├── memory_item.py       # 记忆条目类
│   ├── base_store.py        # 存储抽象基类
│   ├── base_encoder.py      # 编码器抽象基类
│   └── base_retriever.py    # 检索器抽象基类
│
├── stores/                  # 存储实现
│   ├── __init__.py
│   ├── in_memory_store.py   # 基础内存存储
│   ├── sensory_store.py     # 感觉记忆存储
│   ├── short_term_store.py  # 短期记忆存储
│   ├── long_term_store.py   # 长期记忆存储
│   └── working_memory.py    # 工作记忆存储
│
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── memory_consolidation.py  # 记忆巩固
│   ├── memory_forgetting.py     # 记忆遗忘
│   └── memory_reflection.py     # 记忆反思
│
├── encoders/                # 编码器模块 (需自行实现)
│   └── __init__.py
│
├── retrievers/              # 检索器模块 (需自行实现)
│   └── __init__.py
│
└── examples/                # 示例代码
    └── basic_usage.py
```

## 理论基础

### Atkinson-Shiffrin 记忆模型

本系统基于 Atkinson-Shiffrin (1968) 提出的多重存储模型：

1. **感觉记忆**: 信息首先进入感觉记忆，持续时间极短
2. **注意**: 通过注意机制，信息进入短期记忆
3. **复述**: 通过复述维持短期记忆
4. **编码**: 通过编码过程进入长期记忆

### Baddeley 工作记忆模型

工作记忆实现参考 Baddeley (2000) 的模型：

- 中央执行系统 (Central Executive)
- 语音回路 (Phonological Loop)
- 视空间画板 (Visuospatial Sketchpad)
- 情景缓冲器 (Episodic Buffer)

### Generative Agents 反思机制

反思模块参考斯坦福 Park et al. (2023) 的 Generative Agents：

1. 积累足够重要的记忆后触发反思
2. 从近期经验中综合出洞察
3. 将洞察存储为高级记忆
4. 反思影响未来的检索和规划

## 参考文献

- Atkinson, R. C., & Shiffrin, R. M. (1968). Human memory: A proposed system and its control processes. *Psychology of Learning and Motivation*.
- Baddeley, A. D. (2000). The episodic buffer: a new component of working memory? *Trends in Cognitive Sciences*.
- Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST*.

## 许可证

MIT License