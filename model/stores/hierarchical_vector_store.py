"""
分层向量存储模块。

使用 ChromaDB 存储分层记忆节点，支持向量检索和层级关系查询。
支持批量写入模式以提升大数据量处理的性能。
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import json
import numpy as np

from ..hierarchical.hierarchy_types import (
    HierarchicalNode,
    HierarchyLevel,
    HierarchicalMemoryStats,
)


class HierarchicalVectorStore:
    """
    分层向量存储。
    
    使用 ChromaDB 按层级存储节点，每个层级一个独立的 collection。
    
    存储结构:
        - domain_collection: 领域层节点
        - category_collection: 类别层节点
        - keyword_collection: 关键词层节点
        - dialogue_collection: 对话层节点
    
    延迟写入模式:
        - 启用 delayed_write 后，节点先存入内存缓存
        - 手动调用 flush() 强制写入所有缓存
    """
    
    COLLECTION_NAMES = {
        HierarchyLevel.DOMAIN: "domain",
        HierarchyLevel.CATEGORY: "category",
        HierarchyLevel.KEYWORD: "keyword",
        HierarchyLevel.DIALOGUE: "dialogue",
    }
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        delayed_write: bool = True,
    ):
        """
        初始化向量存储。
        
        参数:
            persist_directory: 持久化目录（None 则使用纯内存模式）
            embedding_function: 嵌入函数
            delayed_write: 是否启用延迟写入模式（推荐开启，大幅提升性能）
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.delayed_write = delayed_write
        
        self._client = None
        
        # 存储与缓存
        self._collections: Dict[HierarchyLevel, Any] = {}
        self._memory_cache: Dict[HierarchyLevel, Dict[str, HierarchicalNode]] = {
            level: {} for level in HierarchyLevel
        }
        self._content_cache: Dict[HierarchyLevel, Dict[str, HierarchicalNode]] = {
            level: {} for level in HierarchyLevel
        }
        self._dirty_nodes: Dict[HierarchyLevel, Dict[str, HierarchicalNode]] = {
            level: {} for level in HierarchyLevel
        }
        
        self._level_counters: Dict[HierarchyLevel, int] = {
            level: 0 for level in HierarchyLevel
        }
        # 待写入节点缓存（按层级分组）
        self._pending_nodes: Dict[HierarchyLevel, List[HierarchicalNode]] = {
            level: [] for level in HierarchyLevel
        }
        # 待写入节点总数
        self._pending_count = 0
        
        # 从数据库恢复计数器（解决重启后 ID 冲突问题）
        if persist_directory:
            self._restore_counters_from_db()

    def _cache_node(self, node: HierarchicalNode) -> None:
        """将节点写入运行期缓存与内容索引。"""
        previous_node = self._memory_cache[node.level].get(node.id)
        if previous_node is not None and previous_node.content and previous_node.content != node.content:
            cached_node = self._content_cache[node.level].get(previous_node.content)
            if cached_node is not None and cached_node.id == node.id:
                del self._content_cache[node.level][previous_node.content]

        self._memory_cache[node.level][node.id] = node
        if node.content:
            self._content_cache[node.level][node.content] = node

    def _make_content_key(self, content: str) -> str:
        """为内容生成稳定键，用于持久层精确查找。"""
        if not content:
            return ""
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    
    def _restore_counters_from_db(self) -> None:
        """从 ChromaDB 恢复计数器，避免重启后 ID 冲突导致数据覆盖。"""
        # 先初始化 ChromaDB 连接
        if self._client is None:
            self._init_chroma()
        
        if self._client is None:
            return
        
        for level in HierarchyLevel:
            collection = self._get_collection(level)
            if collection is not None:
                try:
                    # 获取该层级的节点数量
                    count = collection.count()
                    self._level_counters[level] = count
                    if count > 0:
                        print(f"[计数器恢复] {level.name} 层级: 已有 {count} 个节点，新节点 ID将从 {count} 开始")
                except Exception as e:
                    print(f"[计数器恢复] 获取 {level.name} 层级节点数失败: {e}")

    def _init_chroma(self) -> bool:
        """初始化 ChromaDB 客户端和 collections。"""
        try:
            import chromadb
            
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()
            
            for level, name in self.COLLECTION_NAMES.items():
                self._collections[level] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"level": level.name}
                )
            
            return True
        except ImportError:
            print("ChromaDB 未安装，使用内存模式。请运行: pip install chromadb")
            return False
        except Exception as e:
            print(f"ChromaDB 初始化失败: {e}")
            return False
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理 metadata，过滤掉 ChromaDB 不接受的值类型。
        
        ChromaDB metadata 限制:
        - 不能有空列表 []
        - 不能有 None 值
        - 列表需要转换为逗号分隔字符串
        """
        sanitized = {}
        for key, value in metadata.items():
            # 跳过 None 值
            if value is None:
                continue
            # 跳过空列表
            if isinstance(value, list) and len(value) == 0:
                continue
            # 列表转换为字符串
            if isinstance(value, list):
                sanitized[key] = ",".join(str(v) for v in value)
            # 其他类型直接保留
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                # 非基本类型转为字符串
                sanitized[key] = str(value)
        return sanitized
    
    def _build_chroma_metadata(self, node: HierarchicalNode, level: HierarchyLevel) -> Dict[str, Any]:
        """
        构建适合 ChromaDB 存储的 metadata。
        
        确保所有值都是 ChromaDB 可接受的类型。
        """
        base_metadata = {
            "parent_ids": ",".join(node.parent_ids) if node.parent_ids else "",
            "child_ids": ",".join(node.child_ids) if node.child_ids else "",
            "level_embedding": json.dumps(node.level_embedding) if node.level_embedding is not None else "",
            "content_key": self._make_content_key(node.content),
            "level": level.name,
            "created_at": node.created_at.isoformat(),
        }
        # 合并节点额外 metadata，并清理空值
        sanitized_extra = self._sanitize_metadata(node.metadata)
        return {**base_metadata, **sanitized_extra}
    
    def _get_collection(self, level: HierarchyLevel) -> Optional[Any]:
        """获取指定层级的 collection。"""
        if self._client is None:
            self._init_chroma()
        return self._collections.get(level)
    
    def generate_level_id(self, level: HierarchyLevel) -> str:
        """生成层级内的唯一 ID（每层从 0 开始独立计数）。"""
        count = self._level_counters[level]
        self._level_counters[level] += 1
        return f"{level.name.lower()}_{count}"
    
    def add_node(self, node: HierarchicalNode, use_level_id: bool = True) -> str:
        """
        添加节点到存储。
        
        延迟写入模式下，节点先存入缓存，由外层统一 flush。
        立即写入模式下，直接写入 ChromaDB。
        """
        if use_level_id:
            node.id = self.generate_level_id(node.level)
        
        level = node.level
        
        # 延迟写入模式：存入待写入缓存
        if self.delayed_write:
            self._pending_nodes[level].append(node)
            self._cache_node(node)
            self._pending_count += 1
        else:
            # 立即写入模式
            self._write_node_to_chroma(node, level)
            self._cache_node(node)
        
        return node.id
    
    def _write_node_to_chroma(self, node: HierarchicalNode, level: HierarchyLevel) -> bool:
        """将单个节点写入 ChromaDB。"""
        collection = self._get_collection(level)
        if collection is not None and node.embedding is not None:
            try:
                metadata = self._build_chroma_metadata(node, level)
                collection.add(
                    ids=[node.id],
                    embeddings=[node.embedding],
                    documents=[node.content],
                    metadatas=[metadata]
                )
                return True
            except Exception as e:
                print(f"ChromaDB 添加节点失败: {e}")
                return False
        return False

    def _update_node_in_chroma(self, node: HierarchicalNode, level: HierarchyLevel) -> bool:
        """更新 ChromaDB 中已存在的节点。"""
        collection = self._get_collection(level)
        if collection is not None and node.embedding is not None:
            try:
                metadata = self._build_chroma_metadata(node, level)
                collection.update(
                    ids=[node.id],
                    embeddings=[np.array(node.embedding)],
                    documents=[node.content],
                    metadatas=[metadata]
                )
                return True
            except Exception as e:
                print(f"ChromaDB 更新节点失败: {e}")
                return False
        return False

    def _update_nodes_in_chroma(
        self,
        nodes: List[HierarchicalNode],
        level: HierarchyLevel,
    ) -> int:
        """批量更新 ChromaDB 中已存在的节点。"""
        if not nodes:
            return 0

        collection = self._get_collection(level)
        if collection is None:
            return 0

        valid_nodes = [node for node in nodes if node.embedding is not None]
        if not valid_nodes:
            return 0

        try:
            collection.update(
                ids=[node.id for node in valid_nodes],
                embeddings=[np.array(node.embedding) for node in valid_nodes],
                documents=[node.content for node in valid_nodes],
                metadatas=[self._build_chroma_metadata(node, level) for node in valid_nodes],
            )
            return len(valid_nodes)
        except Exception as e:
            print(f"ChromaDB 批量更新失败（层级 {level.name}）: {e}")
            success_count = 0
            for node in valid_nodes:
                if self._update_node_in_chroma(node, level):
                    success_count += 1
            return success_count
    
    def _write_batch_to_chroma(
        self, 
        nodes: List[HierarchicalNode], 
        level: HierarchyLevel
    ) -> int:
        """批量写入节点到 ChromaDB，返回成功写入的数量。"""
        if not nodes:
            return 0
        
        collection = self._get_collection(level)
        if collection is None:
            return 0
        
        # 过滤有 embedding 的节点
        valid_nodes = [n for n in nodes if n.embedding is not None]
        if not valid_nodes:
            return 0
        if len(valid_nodes) < len(nodes):
            dropped = [n.id for n in nodes if n.embedding is None]
            print(f"[WARN] {level.name} 层丢弃 {len(nodes)-len(valid_nodes)} 个无 embedding 节点: {dropped}")
        
        try:
            collection.add(
                ids=[n.id for n in valid_nodes],
                embeddings=[n.embedding for n in valid_nodes],
                documents=[n.content for n in valid_nodes],
                metadatas=[self._build_chroma_metadata(n, level) for n in valid_nodes]
            )
            return len(valid_nodes)
        except Exception as e:
            print(f"ChromaDB 批量写入失败（层级 {level.name}）: {e}")
            # 回退到逐个写入
            success_count = 0
            for node in valid_nodes:
                if self._write_node_to_chroma(node, level):
                    success_count += 1
            return success_count
    
    def add_nodes_batch(self, nodes: List[HierarchicalNode]) -> List[str]:
        """
        批量添加节点。
        
        参数:
            nodes: 节点列表
            
        返回:
            节点 ID 列表
        """
        node_ids = []
        for node in nodes:
            node.id = self.generate_level_id(node.level)
            node_ids.append(node.id)
            level = node.level
            self._pending_nodes[level].append(node)
            self._cache_node(node)
            self._pending_count += 1
        
        return node_ids

    def update_node(self, node: HierarchicalNode) -> bool:
        """更新已存在节点的内容与层级关系。"""
        level = node.level
        self._cache_node(node)

        for idx, pending_node in enumerate(self._pending_nodes[level]):
            if pending_node.id == node.id:
                self._pending_nodes[level][idx] = node
                return True

        if self.delayed_write:
            self._dirty_nodes[level][node.id] = node
            return True

        return self._update_node_in_chroma(node, level)
    
    def flush(self) -> Dict[str, int]:
        """
        强制写入所有缓存的节点到 ChromaDB。
        
        返回:
            各层级写入数量统计
        """
        dirty_count = sum(len(nodes) for nodes in self._dirty_nodes.values())
        if self._pending_count == 0 and dirty_count == 0:
            return {}
        
        stats = {}
        total_flushed = 0
        
        for level in HierarchyLevel:
            pending = self._pending_nodes[level]
            if not pending:
                pending_count = 0
            else:
                pending_count = self._write_batch_to_chroma(pending, level)
                if pending_count > 0:
                    stats[level.name] = pending_count
                    total_flushed += pending_count
                self._pending_nodes[level] = []

            dirty_nodes = list(self._dirty_nodes[level].values())
            if dirty_nodes:
                updated_count = self._update_nodes_in_chroma(dirty_nodes, level)
                if updated_count > 0:
                    stats[f"{level.name}_updated"] = updated_count
                    print(f"已批量更新 {updated_count} 个节点到 ChromaDB")
                self._dirty_nodes[level] = {}
        
        self._pending_count = 0
        
        # 输出本轮统计
        if total_flushed > 0:
            print(f"已批量写入 {total_flushed} 个节点到 ChromaDB")
            
        
        return stats
    
    def get_pending_dirty_count(self) -> int:
        """获取待写入与待更新节点数量。"""
        dirty_count = sum(len(nodes) for nodes in self._dirty_nodes.values())
        return self._pending_count + dirty_count

    def get_pending_counts(self) -> Dict[str, int]:
        """分别返回待新增写入与待更新节点数量。"""
        dirty_count = sum(len(nodes) for nodes in self._dirty_nodes.values())
        return {
            "pending": self._pending_count,
            "dirty": dirty_count,
            "total": self._pending_count + dirty_count,
        }
    
    def get_node(self, node_id: str, level: HierarchyLevel) -> Optional[HierarchicalNode]:
        """获取指定节点。"""
        # 先检查内存缓存
        if node_id in self._memory_cache[level]:
            return self._memory_cache[level][node_id]
        
        # 再检查待写入缓存
        for node in self._pending_nodes[level]:
            if node.id == node_id:
                return node
        
        # 最后从 ChromaDB 查询
        collection = self._get_collection(level)
        if collection is not None:
            try:
                result = collection.get(ids=[node_id], include=["embeddings", "documents", "metadatas"])
                if result["ids"]:
                    node = self._result_to_node(result, level, 0)
                    self._cache_node(node)
                    return node
            except Exception as e:
                print(f"ChromaDB 获取节点失败: {e}")
        
        return None

    def get_nodes_by_level(self, level: HierarchyLevel) -> List[HierarchicalNode]:
        """获取指定层级的所有节点。"""
        nodes = []
        
        # 从内存缓存获取
        nodes.extend(self._memory_cache[level].values())
        
        # 从待写入缓存获取
        nodes.extend(self._pending_nodes[level])
        
        # 从 ChromaDB 获取 (加入分批拉取机制，防止 SQLite 崩溃)
        collection = self._get_collection(level)
        if collection is not None:
            batch_size = 5000  # 安全线：每次最多取 5000 条
            offset = 0
            
            while True:
                # 核心修改：加上 limit 和 offset 分页查询
                result = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "documents", "metadatas"]
                )
                
                # 如果当前批次没有数据，说明整个库已经读取完毕，退出循环
                if not result or not result.get("ids"):
                    break
                
                # 遍历当前批次的数据并转换为 Node
                for i in range(len(result["ids"])):
                    node = self._result_to_node(result, level, i)
                    # 放入缓存与结果列表
                    if node.id not in self._memory_cache[level]:
                        self._cache_node(node)
                    nodes.append(node)
                
                # 偏移量增加，准备拉取下一批 5000 条
                offset += batch_size
                
        return nodes
    
    def _result_to_node(
        self, 
        result: Dict, 
        level: HierarchyLevel, 
        index: int
    ) -> HierarchicalNode:
        """将 ChromaDB 结果转换为节点对象。"""
        metadata = result["metadatas"][index]
        
        # 安全获取 embedding
        embeddings = result.get("embeddings")
        embedding = None
        if embeddings is not None and len(embeddings) > index:
            embedding = embeddings[index]

        level_embedding = None
        if metadata.get("level_embedding"):
            try:
                level_embedding = json.loads(metadata["level_embedding"])
            except (TypeError, json.JSONDecodeError):
                level_embedding = None
        
        return HierarchicalNode(
            id=result["ids"][index],
            content=result["documents"][index],
            level=level,
            embedding=embedding,
            level_embedding=level_embedding,
            parent_ids=metadata.get("parent_ids", "").split(",") if metadata.get("parent_ids") else [],
            child_ids=metadata.get("child_ids", "").split(",") if metadata.get("child_ids") else [],
            metadata={k: v for k, v in metadata.items()
                     if k not in ["parent_ids", "child_ids", "level_embedding", "content_key", "level", "created_at"]},
            created_at=datetime.fromisoformat(metadata["created_at"])
                      if metadata.get("created_at") else datetime.now(),
        )

    def _find_node_by_content_in_collection(
        self,
        content: str,
        level: HierarchyLevel,
    ) -> Optional[HierarchicalNode]:
        """在 Chroma collection 中按 content_key 精确匹配内容。"""
        collection = self._get_collection(level)
        if collection is None:
            return None

        content_key = self._make_content_key(content)
        if not content_key:
            return None

        try:
            result = collection.get(
                where={"content_key": content_key},
                include=["embeddings", "documents", "metadatas"],
            )
        except Exception as e:
            print(f"ChromaDB 按 content_key 查询失败: {e}")
            return None

        if not result or not result.get("ids"):
            return None

        for i, document in enumerate(result["documents"]):
            if document == content:
                node = self._result_to_node(result, level, i)
                self._cache_node(node)
                return node

        return None
    
    def get_node_by_content(
        self, 
        content: str, 
        level: HierarchyLevel
    ) -> Optional[HierarchicalNode]:
        """通过内容查找节点。"""
        cached_node = self._content_cache[level].get(content)
        if cached_node is not None:
            return cached_node

        # 运行期节点在进入内存时会同步写入 _content_cache，miss 后直接查持久层。
        return self._find_node_by_content_in_collection(content, level)
    
    def get_children(
        self, 
        parent_id: str, 
        parent_level: HierarchyLevel
    ) -> List[HierarchicalNode]:
        """获取指定节点的所有子节点。"""
        child_level = parent_level.get_child_level()
        if child_level is None:
            return []

        parent_node = self.get_node(parent_id, parent_level)
        if parent_node is None or not parent_node.child_ids:
            return []

        children = []
        seen_ids = set()
        for child_id in parent_node.child_ids:
            child_node = self.get_node(child_id, child_level)
            if child_node is None or child_node.id in seen_ids:
                continue
            children.append(child_node)
            seen_ids.add(child_node.id)

        return children
    
    def get_parents(self, node: HierarchicalNode) -> List[HierarchicalNode]:
        """获取指定节点的所有父节点。"""
        if not node.parent_ids:
            return []
        
        parent_level = node.level.get_parent_level()
        if parent_level is None:
            return []

        parents = []
        for parent_id in node.parent_ids:
            parent_node = self.get_node(parent_id, parent_level)
            if parent_node is not None:
                parents.append(parent_node)

        return parents

    def get_ancestors(self, node_id: str, level: Optional[HierarchyLevel] = None) -> List[HierarchicalNode]:
        """
        获取指定节点的全部祖先节点（支持多父）。

        返回顺序按距离当前节点由近到远展开，同一祖先节点只返回一次。
        """
        target_node = None
        if level is not None:
            target_node = self.get_node(node_id, level)
        else:
            for current_level in HierarchyLevel:
                target_node = self.get_node(node_id, current_level)
                if target_node is not None:
                    break

        if target_node is None:
            return []

        ancestors: List[HierarchicalNode] = []
        visited_ids = set()
        frontier = self.get_parents(target_node)

        while frontier:
            next_frontier: List[HierarchicalNode] = []
            for parent_node in frontier:
                if parent_node.id in visited_ids:
                    continue
                visited_ids.add(parent_node.id)
                ancestors.append(parent_node)
                next_frontier.extend(self.get_parents(parent_node))
            frontier = next_frontier

        return ancestors
    
    def delete_node(self, node_id: str, level: HierarchyLevel) -> bool:
        """删除节点。"""
        # 删除前先 flush
        dirty_count = sum(len(nodes) for nodes in self._dirty_nodes.values())
        if self._pending_count > 0 or dirty_count > 0:
            self.flush()

        target_node = self.get_node(node_id, level)
        
        collection = self._get_collection(level)
        deleted_from_store = False
        if collection is not None:
            try:
                collection.delete(ids=[node_id])
                deleted_from_store = True
            except Exception as e:
                print(f"删除节点失败: {e}")

        if target_node is None and not deleted_from_store:
            return False

        if target_node is not None:
            parent_level = level.get_parent_level()
            if parent_level is not None:
                for parent_id in target_node.parent_ids:
                    parent_node = self.get_node(parent_id, parent_level)
                    if parent_node is None:
                        continue
                    parent_node.remove_child(node_id)
                    self.update_node(parent_node)

            child_level = level.get_child_level()
            if child_level is not None:
                for child_id in target_node.child_ids:
                    child_node = self.get_node(child_id, child_level)
                    if child_node is None:
                        continue
                    if node_id in child_node.parent_ids:
                        child_node.parent_ids.remove(node_id)
                        self.update_node(child_node)

            if target_node.content in self._content_cache[level]:
                cached_node = self._content_cache[level][target_node.content]
                if cached_node.id == node_id:
                    del self._content_cache[level][target_node.content]

        self._memory_cache[level].pop(node_id, None)
        self._dirty_nodes[level].pop(node_id, None)
        self._pending_nodes[level] = [
            node for node in self._pending_nodes[level]
            if node.id != node_id
        ]

        return True
    
    def get_stats(self) -> HierarchicalMemoryStats:
        """获取存储统计信息。"""
        # 统计前先 flush
        dirty_count = sum(len(nodes) for nodes in self._dirty_nodes.values())
        if self._pending_count > 0 or dirty_count > 0:
            self.flush()
        
        stats = HierarchicalMemoryStats()
        
        for level in HierarchyLevel:
            collection = self._get_collection(level)
            if collection is not None:
                count = collection.count()
            else:
                count = len(self._memory_cache[level])
            
            stats.total_nodes += count
            
            if level == HierarchyLevel.DOMAIN:
                stats.domain_count = count
            elif level == HierarchyLevel.CATEGORY:
                stats.category_count = count
            elif level == HierarchyLevel.KEYWORD:
                stats.keyword_count = count
            elif level == HierarchyLevel.DIALOGUE:
                stats.dialogue_count = count
        
        return stats
    
    def _reset_level_runtime_state(self, level: HierarchyLevel) -> None:
        """清空某层在内存/计数器/待写队列中的状态（与 Chroma 是否为空无关，需与 clear_level 成对使用）。"""
        self._memory_cache[level].clear()
        self._content_cache[level].clear()
        self._level_counters[level] = 0
        self._dirty_nodes[level].clear()
        n_dropped = len(self._pending_nodes[level])
        self._pending_nodes[level] = []
        if n_dropped:
            self._pending_count = max(0, self._pending_count - n_dropped)

    def clear_level(self, level: HierarchyLevel) -> bool:
        """清空指定层级的所有节点。"""
        # 清空前先 flush
        if self.get_pending_dirty_count() > 0:
            self.flush()
        
        collection = self._get_collection(level)
        if collection is not None:
            try:
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception as e:
                print(f"清空层级失败: {e}")

        self._reset_level_runtime_state(level)
        return True
    
    def clear_all(self) -> bool:
        """清空所有层级。"""
        for level in HierarchyLevel:
            self.clear_level(level)
        return True
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出所有节点为字典格式。"""
        # 导出前先 flush
        if self.get_pending_dirty_count() > 0:
            self.flush()
        
        data = {
            "counters": {level.name: count for level, count in self._level_counters.items()},
            "nodes": {}
        }
        
        for level in HierarchyLevel:
            level_name = level.name.lower()
            data["nodes"][level_name] = []
            
            collection = self._get_collection(level)
            if collection is not None:
                try:
                    result = collection.get(include=["embeddings", "documents", "metadatas"])
                    for i in range(len(result["ids"])):
                        node = self._result_to_node(result, level, i)
                        data["nodes"][level_name].append(node.to_dict())
                except Exception:
                    pass
            
            for node in self._memory_cache[level].values():
                data["nodes"][level_name].append(node.to_dict())
        
        return data
    
    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """从字典导入节点。"""
        try:
            for level_name, count in data.get("counters", {}).items():
                level = HierarchyLevel[level_name.upper()]
                self._level_counters[level] = count
            
            for level_name, nodes in data.get("nodes", {}).items():
                level = HierarchyLevel[level_name.upper()]
                for node_data in nodes:
                    node = HierarchicalNode.from_dict(node_data)
                    self.add_node(node, use_level_id=False)
            
            return True
        except Exception as e:
            print(f"导入失败: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """保存到 JSON 文件。"""
        import json
        
        # 保存前先 flush
        if self.get_pending_dirty_count() > 0:
            self.flush()
        
        try:
            data = self.export_to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """从 JSON 文件加载。"""
        import json
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.import_from_dict(data)
        except Exception as e:
            print(f"加载失败: {e}")
            return False


class VectorStoreFactory:
    """向量存储工厂类。"""
    
    @staticmethod
    def create_chroma_store(
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        delayed_write: bool = True,
    ) -> HierarchicalVectorStore:
        """创建 ChromaDB 存储（默认启用延迟写入）。"""
        embedding_function = None
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(embedding_model)
                embedding_function = lambda text: model.encode(text).tolist()
            except ImportError:
                print("请安装 sentence-transformers: pip install sentence-transformers")
        
        return HierarchicalVectorStore(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            delayed_write=delayed_write,
        )
    
    @staticmethod
    def create_memory_store() -> HierarchicalVectorStore:
        """创建纯内存存储。"""
        return HierarchicalVectorStore(
            persist_directory=None,
            embedding_function=None,
            delayed_write=True,
        )