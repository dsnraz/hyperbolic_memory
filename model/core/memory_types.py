"""
记忆类型和枚举定义。

本模块定义记忆系统的基本类型和状态。
基于认知心理学的人类记忆模型。
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class MemoryType(Enum):
    """
    智能体记忆架构中的记忆类型。
    
    基于 Atkinson-Shiffrin 记忆模型和现代认知架构：
    - 感觉记忆: 感官信息的短暂存储 (< 1 秒)
    - 短期记忆: 容量有限的临时存储 (15-30 秒)
    - 工作记忆: 信息的主动处理
    - 长期记忆: 永久存储
    
    长期记忆子类型：
    - 情景记忆: 个人经历和事件
    - 语义记忆: 事实和一般知识
    - 程序性记忆: 技能和操作知识
    """
    
    # 主要记忆类型
    SENSORY = auto()          # 感觉记忆
    SHORT_TERM = auto()       # 短期记忆（临时缓冲区）
    WORKING = auto()          # 工作记忆（主动处理）
    LONG_TERM = auto()        # 长期记忆（永久存储）
    
    # 长期记忆子类型
    EPISODIC = auto()         # 情景记忆（个人经历）
    SEMANTIC = auto()         # 语义记忆（事实、概念）
    PROCEDURAL = auto()       # 程序性记忆（技能、程序）


class MemoryPriority(Enum):
    """
    记忆条目的优先级。
    用于检索和巩固决策。
    """
    
    CRITICAL = 4      # 关键：对智能体目标至关重要
    HIGH = 3          # 高：对当前任务重要
    MEDIUM = 2        # 中：可能有用
    LOW = 1           # 低：背景信息
    TRIVIAL = 0       # 微不足道：相关性极低


class MemoryState(Enum):
    """
    记忆生命周期中的状态。
    
    记忆条目在这些状态之间转换：
    ENCODED -> STORED -> CONSOLIDATED -> RETRIEVED -> FORGOTTEN/DECAYED
    """
    
    ENCODED = auto()        # 刚编码完成
    STORED = auto()         # 已存储在记忆系统中
    CONSOLIDATED = auto()   # 通过复述/重要性加强
    RETRIEVED = auto()      # 最近被访问/检索
    REHEARSED = auto()      # 在工作记忆中主动复述
    FORGOTTEN = auto()      # 标记为待删除
    DECAYED = auto()        # 自然衰减


@dataclass
class MemoryMetadata:
    """
    记忆条目的元数据。
    
    跟踪访问模式、重要性和生命周期信息。
    """
    
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    last_modified: Optional[datetime] = None
    importance_score: float = 0.0
    decay_rate: float = 0.1  # 记忆强度衰减速率
    consolidation_count: int = 0  # 巩固次数
    
    def update_access(self) -> None:
        """更新访问元数据（当记忆被检索时调用）。"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """获取记忆的年龄（秒）。"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_recency_score(self) -> float:
        """
        计算新近度分数。
        最近访问的记忆得分更高。
        """
        seconds_since_access = (datetime.now() - self.last_accessed).total_seconds()
        # 基于新近度的指数衰减（归一化到1天）
        return max(0.0, 1.0 - (seconds_since_access / 86400.0))