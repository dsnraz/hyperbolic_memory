"""
内存存储实现。

使用 Python 字典实现的简单内存存储。
适用于测试、原型开发和小规模应用。
"""

from typing import Any, Dict, List, Optional
import threading
from datetime import datetime

from ..core.memory_item import MemoryItem
from ..core.memory_types import MemoryType
from ..core.base_store import BaseMemoryStore, StorageFullError, InvalidMemoryItemError


class InMemoryStore(BaseMemoryStore):
    """
    内存存储实现。
    
    使用字典进行存储，可选线程安全。
    适用于：
    - 测试和原型开发
    - 小规模应用
    - 临时/短暂的记忆存储
    
    注意：此存储不持久化数据。当存储被销毁或程序终止时，
    所有数据都会丢失。
    """
    
    def __init__(
        self,
        memory_type: MemoryType,
        max_capacity: Optional[int] = None,
        thread_safe: bool = True,
        **kwargs
    ):
        """
        初始化内存存储。
        
        参数:
            memory_type: 此存储管理的记忆类型
            max_capacity: 最大存储条目数
            thread_safe: 是否使用锁实现线程安全
            **kwargs: 额外的配置参数
        """
        super().__init__(memory_type, max_capacity, **kwargs)
        self._storage: Dict[str, MemoryItem] = {}
        self._lock = threading.Lock() if thread_safe else None
        self._thread_safe = thread_safe
    
    def _acquire_lock(self):
        """如果线程安全则获取锁。"""
        if self._lock:
            self._lock.acquire()
    
    def _release_lock(self):
        """如果线程安全则释放锁。"""
        if self._lock:
            self._lock.release()
    
    def store(self, item: MemoryItem) -> bool:
        """存储记忆条目。"""
        self._acquire_lock()
        try:
            if self.is_full() and item.id not in self._storage:
                raise StorageFullError(
                    f"{self.memory_type.name} 的记忆存储已满 "
                    f"（容量 {self.max_capacity} 条）"
                )
            
            if item.id in self._storage:
                # 更新现有条目
                self._storage[item.id] = item
            else:
                # 添加新条目
                self._storage[item.id] = item
            
            return True
        finally:
            self._release_lock()
    
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """通过 ID 检索记忆条目。"""
        self._acquire_lock()
        try:
            item = self._storage.get(item_id)
            if item:
                item.update_access()
            return item
        finally:
            self._release_lock()
    
    def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆条目。"""
        self._acquire_lock()
        try:
            if item_id not in self._storage:
                return False
            
            item = self._storage[item_id]
            
            # 应用更新
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
                elif hasattr(item.metadata, key):
                    setattr(item.metadata, key, value)
            
            item.metadata.last_modified = datetime.now()
            return True
        finally:
            self._release_lock()
    
    def delete(self, item_id: str) -> bool:
        """删除记忆条目。"""
        self._acquire_lock()
        try:
            if item_id in self._storage:
                del self._storage[item_id]
                return True
            return False
        finally:
            self._release_lock()
    
    def exists(self, item_id: str) -> bool:
        """检查记忆条目是否存在。"""
        self._acquire_lock()
        try:
            return item_id in self._storage
        finally:
            self._release_lock()
    
    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆条目。"""
        self._acquire_lock()
        try:
            return list(self._storage.values())
        finally:
            self._release_lock()
    
    def count(self) -> int:
        """获取条目数量。"""
        self._acquire_lock()
        try:
            return len(self._storage)
        finally:
            self._release_lock()
    
    def clear(self) -> int:
        """清除所有条目。"""
        self._acquire_lock()
        try:
            count = len(self._storage)
            self._storage.clear()
            return count
        finally:
            self._release_lock()
    
    def get_ids(self) -> List[str]:
        """获取所有记忆 ID。"""
        self._acquire_lock()
        try:
            return list(self._storage.keys())
        finally:
            self._release_lock()
    
    def batch_store(self, items: List[MemoryItem]) -> Dict[str, bool]:
        """
        批量存储多个条目。
        
        参数:
            items: 要存储的记忆条目列表
            
        返回:
            记忆 ID 到成功状态的映射字典
        """
        results = {}
        self._acquire_lock()
        try:
            for item in items:
                try:
                    if self.is_full() and item.id not in self._storage:
                        results[item.id] = False
                        continue
                    self._storage[item.id] = item
                    results[item.id] = True
                except Exception:
                    results[item.id] = False
            return results
        finally:
            self._release_lock()
    
    def batch_delete(self, item_ids: List[str]) -> Dict[str, bool]:
        """
        批量删除多个条目。
        
        参数:
            item_ids: 要删除的条目 ID 列表
            
        返回:
            条目 ID 到成功状态的映射字典
        """
        results = {}
        self._acquire_lock()
        try:
            for item_id in item_ids:
                results[item_id] = item_id in self._storage
                if results[item_id]:
                    del self._storage[item_id]
            return results
        finally:
            self._release_lock()


class SerializableInMemoryStore(InMemoryStore):
    """
    支持序列化的内存存储。
    
    添加将整个存储序列化/反序列化为字典或 JSON 兼容格式的方法。
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将整个存储序列化为字典。
        
        返回:
            包含所有记忆条目的字典
        """
        self._acquire_lock()
        try:
            return {
                "memory_type": self.memory_type.name,
                "max_capacity": self.max_capacity,
                "items": {item_id: item.to_dict() for item_id, item in self._storage.items()},
                "count": len(self._storage)
            }
        finally:
            self._release_lock()
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典加载存储。
        
        参数:
            data: 包含存储数据的字典
        """
        self._acquire_lock()
        try:
            self._storage.clear()
            
            # 根据记忆类型导入适当的条目类
            from ..core.memory_item import (
                MemoryItem, SensoryMemoryItem, EpisodicMemoryItem, SemanticMemoryItem
            )
            
            item_classes = {
                MemoryType.SENSORY: SensoryMemoryItem,
                MemoryType.EPISODIC: EpisodicMemoryItem,
                MemoryType.SEMANTIC: SemanticMemoryItem,
            }
            
            default_class = EpisodicMemoryItem
            
            for item_id, item_data in data.get("items", {}).items():
                memory_type_name = item_data.get("memory_type", "EPISODIC")
                item_class = item_classes.get(
                    MemoryType[memory_type_name], 
                    default_class
                )
                self._storage[item_id] = item_class.from_dict(item_data)
        finally:
            self._release_lock()