import re

# 读取原文件
with open('hierarchical_vector_store.py.bak', 'r') as f:
    content = f.read()

# 找到并替换__init__ 方法中的计数器初始化部分
old_pattern = r'''        # 计数器（每层独立从 0 开始）
        self._level_counters: Dict\[HierarchyLevel, int\] = \{
            level: 0 for level in HierarchyLevel
        \}
        # 待写入节点缓存（按层级分组）
        self._pending_nodes: Dict\[HierarchyLevel, List\[HierarchicalNode\]\] = \{
            level: \[\] for level in HierarchyLevel
        \}
        # 待写入节点总数
        self._pending_count = 0'''

new_code = '''        # 计数器（每层独立从 0 开始）
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
            self._restore_counters_from_db()'''

content = re.sub(old_pattern, new_code, content)

# 在 _init_chroma 方法之前添加 _restore_counters_from_db 方法
restore_method = '''    def _restore_counters_from_db(self) -> None:
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

'''

# 在 def _init_chroma 之前插入
content = content.replace('    def _init_chroma(self) -> bool:', restore_method + '    def _init_chroma(self) -> bool:')

# 写入新文件
with open('hierarchical_vector_store.py', 'w') as f:
    f.write(content)

print("修改完成！")
