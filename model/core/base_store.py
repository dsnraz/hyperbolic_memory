"""
记忆存储抽象基类。

本模块定义记忆存储系统的抽象接口。
不同的记忆存储实现此接口以提供特定的存储机制。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from .memory_item import MemoryItem
from .memory_types import MemoryType, MemoryState


class BaseMemoryStore(ABC):
    """
    记忆存储系统的抽象基类。
    
    此类定义所有记忆存储必须实现的接口。
    记忆存储负责记忆条目的持久化和检索。
    
    不同的实现可以使用各种后端：
    - 内存存储（快速访问）
    - 向量数据库（语义搜索）
    - 图数据库（关系存储）
    - 文件存储（持久化）
    - 分布式存储（可扩展性）
    """
    
    def __init__(
        self,
        memory_type: MemoryType,
        max_capacity: Optional[int] = None,
        **kwargs
    ):
        """
        初始化记忆存储。
        
        参数:
            memory_type: 此存储管理的记忆类型
            max_capacity: 最大存储条目数（None 表示无限制）
            **kwargs: 额外的配置参数
        """
        self.memory_type = memory_type
        self.max_capacity = max_capacity
        self.config = kwargs
    
    @abstractmethod
    def store(self, item: MemoryItem) -> bool:
        """
        存储记忆条目。
        
        参数:
            item: 要存储的记忆条目
            
        返回:
            成功返回 True，失败返回 False
            
        异常:
            StorageFullError: 存储已满
            InvalidMemoryItemError: 条目无效
        """
        pass
    
    @abstractmethod
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """
        通过 ID 检索记忆条目。
        
        参数:
            item_id: 记忆条目的唯一标识符
            
        返回:
            找到则返回记忆条目，否则返回 None
        """
        pass
    
    @abstractmethod
    def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新记忆条目。
        
        参数:
            item_id: 记忆条目的唯一标识符
            updates: 要更新的字段字典
            
        返回:
            成功返回 True，失败返回 False
        """
        pass
    
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """
        删除记忆条目。
        
        参数:
            item_id: 记忆条目的唯一标识符
            
        返回:
            成功返回 True，失败返回 False
        """
        pass
    
    @abstractmethod
    def exists(self, item_id: str) -> bool:
        """
        检查记忆条目是否存在。
        
        参数:
            item_id: 记忆条目的唯一标识符
            
        返回:
            存在返回 True，否则返回 False
        """
        pass
    
    @abstractmethod
    def get_all(self) -> List[MemoryItem]:
        """
        获取存储中的所有记忆条目。
        
        返回:
            所有记忆条目的列表
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        获取存储中的条目数量。
        
        返回:
            记忆条目数量
        """
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """
        清除存储中的所有条目。
        
        返回:
            被清除的条目数量
        """
        pass
    
    # 带默认实现的可选方法
    
    def get_by_state(self, state: MemoryState) -> List[MemoryItem]:
        """
        获取特定状态的所有记忆条目。
        
        参数:
            state: 要筛选的状态
            
        返回:
            具有指定状态的记忆条目列表
        """
        return [item for item in self.get_all() if item.state == state]
    
    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[MemoryItem]:
        """
        通过标签获取记忆条目。
        
        参数:
            tags: 要筛选的标签列表
            match_all: 若为 True，所有标签都必须匹配；若为 False，任一标签匹配即可
            
        返回:
            匹配的记忆条目列表
        """
        if match_all:
            return [item for item in self.get_all() 
                   if all(tag in item.tags for tag in tags)]
        else:
            return [item for item in self.get_all() 
                   if any(tag in item.tags for tag in tags)]
    
    def get_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MemoryItem]:
        """
        获取在时间范围内创建的记忆条目。
        
        参数:
            start_time: 时间范围开始
            end_time: 时间范围结束
            
        返回:
            时间范围内的记忆条目列表
        """
        return [
            item for item in self.get_all()
            if start_time <= item.metadata.created_at <= end_time
        ]
    
    def get_weakest(self, n: int = 1) -> List[MemoryItem]:
        """
        获取 n 个最弱的记忆条目（强度最低）。
        
        参数:
            n: 要返回的条目数
            
        返回:
            最弱的记忆条目列表
        """
        items = self.get_all()
        sorted_items = sorted(items, key=lambda x: x.get_strength())
        return sorted_items[:n]
    
    def get_strongest(self, n: int = 1) -> List[MemoryItem]:
        """
        获取 n 个最强的记忆条目（强度最高）。
        
        参数:
            n: 要返回的条目数
            
        返回:
            最强的记忆条目列表
        """
        items = self.get_all()
        sorted_items = sorted(items, key=lambda x: x.get_strength(), reverse=True)
        return sorted_items[:n]
    
    def is_full(self) -> bool:
        """
        检查存储是否已满。
        
        返回:
            已满返回 True，否则返回 False
        """
        if self.max_capacity is None:
            return False
        return self.count() >= self.max_capacity
    
    def get_capacity_info(self) -> Dict[str, Any]:
        """
        获取此存储的容量信息。
        
        返回:
            包含容量信息的字典
        """
        return {
            "memory_type": self.memory_type.name,
            "current_count": self.count(),
            "max_capacity": self.max_capacity,
            "is_full": self.is_full(),
            "available_slots": (
                self.max_capacity - self.count() 
                if self.max_capacity is not None 
                else float('inf')
            )
        }
    
    def __len__(self) -> int:
        return self.count()
    
    def __contains__(self, item_id: str) -> bool:
        return self.exists(item_id)


class StorageFullError(Exception):
    """存储已满时抛出。"""
    pass


class InvalidMemoryItemError(Exception):
    """记忆条目无效时抛出。"""
    pass