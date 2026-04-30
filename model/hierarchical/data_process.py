"""
数据处理脚本。

将原始对话数据处理成向量并存储到分层记忆系统中。
支持批量推理模式以提升 GPU 利用率（3-5倍加速）。
"""

import json
import time
import os
from typing import Optional, List, Dict, Any, Set

from tqdm import tqdm

from ..encoders import LLMEncoder, EmbeddingEncoder
from .hierarchy_types import HierarchicalNode, HierarchyLevel
from .hierarchical_manager import HierarchicalMemoryManager, create_hierarchical_manager


class DataProcessor:
    """
    数据处理器。
    
    将原始对话数据转换为分层记忆结构并存储。
    支持批量推理以提升 GPU 利用率。
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        manager: HierarchicalMemoryManager = None,
        datapath: str = None,
        llm_model_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        device: str = "auto",
        flush_interval: int = 5000,
        llm_batch_size: int = 24,
    ):
        """
        初始化数据处理器。
        
        参数:
            embedding_model: 嵌入模型名称
            manager: 预配置的协调器（可选）
            datapath: 数据文件路径
            llm_model_path: LLM 模型路径
            persist_directory: 持久化目录
            device: 设备选择
            flush_interval: ChromaDB 写入间隔
            llm_batch_size: LLM 批量推理大小（关键参数，建议 8-16）
        """
        self.datapath = datapath
        self.flush_interval = flush_interval
        self.llm_batch_size = llm_batch_size
        
        # 断点续处理索引文件
        self.processed_index_file = None
        self.failed_index_file = None
        if persist_directory:
            self.processed_index_file = os.path.join(persist_directory, "processed_indices1.json")
            self.failed_index_file = os.path.join(persist_directory, "failed_indices1.json")
        
        self.processed_indices: Set[int] = set()
        self.failed_indices: Set[int] = set()
        
        if manager:
            self.manager = manager
        else:
            self.manager = create_hierarchical_manager(
                llm_model_path=llm_model_path,
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                device=device,
            )
        print("存在 manager")
        
        self.data = None
        if datapath:
            self.data = self.load_data()
            print("数据加载成功")
        
        if self.processed_index_file:
            self._load_processed_indices()
        if self.failed_index_file:
            self._load_failed_indices()
    
    def _load_processed_indices(self) -> int:
        """从文件加载已处理的索引列表。"""
        if not self.processed_index_file or not os.path.exists(self.processed_index_file):
            print("无已处理索引记录文件，从头开始处理")
            return 0
        
        try:
            with open(self.processed_index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.processed_indices = set(data.get("processed_indices", []))
            count = len(self.processed_indices)
            
            if count > 0:
                print(f"[断点续处理] 已加载 {count} 个已处理索引，将跳过这些样本")
            
            return count
        except Exception as e:
            print(f"[断点续处理] 加载已处理索引失败: {e}")
            self.processed_indices = set()
            return 0
    
    def _load_failed_indices(self) -> int:
        """从文件加载失败的索引列表。"""
        if not self.failed_index_file or not os.path.exists(self.failed_index_file):
            print("无失败索引记录文件")
            return 0
        
        try:
            with open(self.failed_index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.failed_indices = set(data.get("failed_indices", []))
            count = len(self.failed_indices)
            
            if count > 0:
                print(f"[断点续处理] 已加载 {count} 个失败索引")
            
            return count
        except Exception as e:
            print(f"[断点续处理] 加载失败索引失败: {e}")
            self.failed_indices = set()
            return 0
    
    def _save_processed_indices(self):
        """保存已处理的索引到文件。"""
        if not self.processed_index_file:
            return
        
        try:
            os.makedirs(os.path.dirname(self.processed_index_file) or ".", exist_ok=True)
            
            data = {
                "processed_indices": sorted(list(self.processed_indices)),
                "total_count": len(self.processed_indices),
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            with open(self.processed_index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"[断点续处理] 保存已处理索引失败: {e}")
    
    def _save_failed_indices(self):
        """保存失败的索引到文件。"""
        if not self.failed_index_file:
            return
        
        try:
            os.makedirs(os.path.dirname(self.failed_index_file) or ".", exist_ok=True)
            
            data = {
                "failed_indices": sorted(list(self.failed_indices)),
                "total_count": len(self.failed_indices),
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            with open(self.failed_index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"失败索引已保存到: {self.failed_index_file}")
            
        except Exception as e:
            print(f"[断点续处理] 保存失败索引失败: {e}")

    def _flush_and_save_progress(self):
        """先写入节点，再保存成功/失败索引，避免索引领先于节点落盘。"""
        self.manager.flush()
        self._save_processed_indices()
        self._save_failed_indices()
    
    def load_data(self) -> List[Any]:
        """加载原始数据。"""
        with open(self.datapath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            for key in ["data", "items", "dialogues", "conversations"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        
        return data if isinstance(data, list) else [data]
    
    def process_file(
        self, 
        max_items: int = None,
        process_batch_size: int = 128,
        show_progress: bool = True,
        auto_flush: bool = True,
        skip_processed: bool = True,
    ):
        """
        处理数据文件（批量推理模式）。
        
        参数:
            max_items: 最大处理数量（用于测试）
            process_batch_size: 处理批次大小（每批处理多少条对话）
            show_progress: 是否显示进度
            auto_flush: 是否自动定期 flush
            skip_processed: 是否跳过已处理的样本
        """
        if self.data is None:
            raise ValueError("数据未加载，请先设置 datapath")
        
        total_data_len = len(self.data)
        
        # 筛选待处理数据
        items_to_process = []
        indices_to_process = []
        
        total_to_check = min(total_data_len, max_items) if max_items else total_data_len
        
        for idx in range(total_to_check):
            if skip_processed and idx in self.processed_indices:
                continue
            items_to_process.append(self.data[idx])
            indices_to_process.append(idx)
        
        skip_count = total_to_check - len(items_to_process)
        
        print(f"\n{'='*60}")
        print(f"批量推理模式配置:")
        print(f"  LLM 批量大小: {self.llm_batch_size}")
        print(f"  处理批次大小: {process_batch_size}")
        print(f"  数据总量: {total_data_len}")
        print(f"  已处理跳过: {skip_count}")
        print(f"  待处理数量: {len(items_to_process)}")
        print(f"  flush 间隔: {self.flush_interval}")
        print(f"{'='*60}\n")
        
        # 分批处理
        total_batches = (len(items_to_process) + process_batch_size - 1) // process_batch_size
        
        success_count = 0
        fail_count = 0
        fail_indices = []
        total_flush_seconds = 0.0
        last_batch_total_seconds = 0.0
        last_batch_flush_seconds = 0.0
        last_batch_other_seconds = 0.0
        
        start_time = time.time()
        last_flush_count = 0
        
        # 批量处理进度条
        batch_pbar = tqdm(
            range(total_batches),
            desc="批量处理",
            unit="batch",
            disable=not show_progress,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        
        for batch_idx in batch_pbar:
            batch_start = time.perf_counter()
            batch_flush_seconds = 0.0
            start_idx = batch_idx * process_batch_size
            end_idx = min(start_idx + process_batch_size, len(items_to_process))
            
            batch_items = items_to_process[start_idx:end_idx]
            batch_dialogues = [
                json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                for item in batch_items
            ]
            batch_indices = indices_to_process[start_idx:end_idx]
            
            # 批量处理对话（使用批量推理）
            nodes_list, final_is_ok_list = self.manager.batch_process_dialogues(
                batch_dialogues,
                llm_batch_size=self.llm_batch_size,
                show_progress=False,  # 外层已有进度条
            )
            
            # 统计结果
            batch_success = sum(final_is_ok_list)
            batch_fail = len(final_is_ok_list) - batch_success
            
            success_count += batch_success
            fail_count += batch_fail
            
            # 记录成功/失败的索引
            for i, (final_is_ok, orig_idx) in enumerate(zip(final_is_ok_list, batch_indices)):
                if final_is_ok:
                    self.processed_indices.add(orig_idx)
                    # 如果之前是失败的，现在成功了，从失败列表中移除
                    if orig_idx in self.failed_indices:
                        self.failed_indices.discard(orig_idx)
                else:
                    fail_indices.append(orig_idx)
                    self.failed_indices.add(orig_idx)
            
            # 定期 flush
            if auto_flush and success_count - last_flush_count >= self.flush_interval:
                batch_pbar.write(f"达到 flush 阈值 ({self.flush_interval})，正在写入...")
                flush_start = time.perf_counter()
                self._flush_and_save_progress()
                flush_elapsed = time.perf_counter() - flush_start
                batch_flush_seconds += flush_elapsed
                total_flush_seconds += flush_elapsed
                last_flush_count = success_count
            
            # 更新进度条信息
            elapsed = time.time() - start_time
            speed = success_count / elapsed if elapsed > 0 else 0
            pending_counts = self.manager.vector_store.get_pending_counts()
            last_batch_perf = self.manager.get_last_batch_perf()
            batch_total_seconds = time.perf_counter() - batch_start
            batch_other_seconds = max(
                0.0,
                batch_total_seconds
                - last_batch_perf.get("llm_seconds", 0.0)
                - last_batch_perf.get("embedding_seconds", 0.0)
                - last_batch_perf.get("node_build_seconds", 0.0)
                - batch_flush_seconds,
            )
            last_batch_total_seconds = batch_total_seconds
            last_batch_flush_seconds = batch_flush_seconds
            last_batch_other_seconds = batch_other_seconds
            batch_pbar.set_postfix({
                "成功": success_count,
                "失败": fail_count,
                "待增": pending_counts["pending"],
                "待更": pending_counts["dirty"],
                "速度": f"{speed:.1f}/s",
                "llm": f"{last_batch_perf.get('llm_seconds', 0.0):.1f}s",
                "emb": f"{last_batch_perf.get('embedding_seconds', 0.0):.1f}s",
                "build": f"{last_batch_perf.get('node_build_seconds', 0.0):.1f}s",
                "flush": f"{batch_flush_seconds:.1f}s",
                "other": f"{batch_other_seconds:.1f}s",
                "batch": f"{batch_total_seconds:.1f}s",
            })
        
        batch_pbar.close()
        
        # 最终 flush 后再保存索引，避免索引领先于节点落盘
        if self.manager.get_pending_dirty_count() > 0:
            print("\n处理完成，正在写入剩余节点...")
            flush_start = time.perf_counter()
            self._flush_and_save_progress()
            flush_elapsed = time.perf_counter() - flush_start
            total_flush_seconds += flush_elapsed
        else:
            self._save_processed_indices()
            self._save_failed_indices()
        
        elapsed_total = time.time() - start_time
        perf_stats = self.manager.get_perf_stats()
        
        print(f"\n{'='*60}")
        print(f"处理完成！")
        print(f"  本轮成功: {success_count} 条")
        print(f"  本轮跳过: {skip_count} 条（已处理）")
        print(f"  本轮失败: {fail_count} 条")
        print(f"  累计已处理: {len(self.processed_indices)} 条")
        print(f"  累计失败: {len(self.failed_indices)} 条")
        print(f"  总耗时: {elapsed_total:.2f} 秒")
        print(f"  平均速度: {success_count / elapsed_total:.2f} 条/秒")
        print(f"  LLM耗时: {perf_stats['llm_seconds']:.2f} 秒")
        print(f"  Embedding耗时: {perf_stats['embedding_seconds']:.2f} 秒")
        print(f"  构图耗时: {perf_stats['node_build_seconds']:.2f} 秒")
        print(f"  关系更新耗时: {perf_stats['relation_update_seconds']:.2f} 秒")
        print(f"  Flush耗时: {total_flush_seconds:.2f} 秒")
        print(f"  最近批次耗时: {last_batch_total_seconds:.2f} 秒")
        print(f"  最近批次Flush耗时: {last_batch_flush_seconds:.2f} 秒")
        print(f"  最近批次其他耗时: {last_batch_other_seconds:.2f} 秒")
        print(f"{'='*60}")
    
    def process_file_single(
        self,
        max_items: int = None,
        show_progress: bool = True,
        auto_flush: bool = True,
        skip_processed: bool = True,
    ):
        """
        单条处理模式（兼容旧版本，不推荐用于大规模数据）。
        """
        if self.data is None:
            raise ValueError("数据未加载，请先设置 datapath")
        
        total_data_len = len(self.data)
        
        skip_count = 0
        if skip_processed and self.processed_indices:
            max_check = min(total_data_len, max_items) if max_items else total_data_len
            skip_count = sum(1 for i in range(max_check) if i in self.processed_indices)
        
        total_to_process = min(total_data_len, max_items) if max_items else total_data_len
        remaining = total_to_process - skip_count
        
        count = 0
        fail_to_deal = []
        total_flush_seconds = 0.0
        
        start_time = time.time()
        last_flush_count = 0
        
        print(f"开始处理（单条模式）:")
        print(f"  数据总量: {total_data_len}")
        print(f"  已处理跳过: {skip_count}")
        print(f"  待处理数量: {remaining}")
        
        data_iter = enumerate(self.data[:max_items] if max_items else self.data)
        
        pbar = tqdm(
            data_iter,
            total=total_to_process,
            desc="处理对话",
            unit="条",
            disable=not show_progress,
        )
        
        for index, data_item in pbar:
            if skip_processed and index in self.processed_indices:
                pbar.set_postfix({"状态": "跳过"})
                continue
            
            if isinstance(data_item, dict):
                dialogue = json.dumps(data_item, ensure_ascii=False)
            else:
                dialogue = str(data_item)
            
            nodes, final_is_ok = self.manager.process_dialogue(dialogue)
            
            if final_is_ok:
                count += 1
                self.processed_indices.add(index)
                # 如果之前是失败的，现在成功了，从失败列表中移除
                if index in self.failed_indices:
                    self.failed_indices.discard(index)
                
                pending_counts = self.manager.vector_store.get_pending_counts()
                elapsed = time.time() - start_time
                speed = count / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "成功": count,
                    "失败": len(fail_to_deal),
                    "待增": pending_counts["pending"],
                    "待更": pending_counts["dirty"],
                    "速度": f"{speed:.1f}/s"
                })
                
                if auto_flush and count - last_flush_count >= self.flush_interval:
                    tqdm.write(f"达到 flush 阈值 ({self.flush_interval})，正在写入...")
                    flush_start = time.perf_counter()
                    self._flush_and_save_progress()
                    total_flush_seconds += time.perf_counter() - flush_start
                    last_flush_count = count
            else:
                fail_to_deal.append(index)
                self.failed_indices.add(index)
        
        pbar.close()
        
        if self.manager.get_pending_dirty_count() > 0:
            print("处理完成，正在写入剩余节点...")
            flush_start = time.perf_counter()
            self._flush_and_save_progress()
            total_flush_seconds += time.perf_counter() - flush_start
        else:
            self._save_processed_indices()
            self._save_failed_indices()
        
        elapsed_total = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"处理完成！")
        print(f"  本轮成功: {count} 条")
        print(f"  本轮跳过: {skip_count} 条")
        print(f"  本轮失败: {len(fail_to_deal)} 条")
        print(f"  累计失败: {len(self.failed_indices)} 条")
        print(f"  总耗时: {elapsed_total:.2f} 秒")
        print(f"  平均速度: {count / elapsed_total:.2f} 条/秒")
        print(f"  Flush耗时: {total_flush_seconds:.2f} 秒")
        print(f"{'='*50}")
    
    def flush(self):
        """手动触发 flush。"""
        self.manager.flush()
    
    def get_pending_dirty_count(self) -> int:
        """获取待写入与待更新节点数量。"""
        return self.manager.vector_store.get_pending_dirty_count()
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息。"""
        stats = self.manager.get_stats()
        return stats.to_dict()


def main():
    LLM_MODEL_PATH = "/share/home/leiyh5/models/Qwen2.5-7B-Instruct"
    DATA_FILE = "/share/home/leiyh5/Memory/data/locomo/extract_ratio_1_0/locomo_train_interactions.json"
    PERSIST_DIR = "/share/home/leiyh5/Memory/data/hierarchical_memory_locomo_total"
    
    processor = DataProcessor(
        llm_model_path=LLM_MODEL_PATH,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        persist_directory=PERSIST_DIR,
        device="auto",
        datapath=DATA_FILE,
        flush_interval=2048,
        llm_batch_size=64,  # 批量推理大小
    )
    print("处理器创建成功")
    
    # 使用批量推理模式
    processor.process_file(
        process_batch_size=256,  # 每批处理 128 条对话
    )
    
    stats = processor.get_stats()
    print(f"\n统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  领域数: {stats['domain_count']}")
    print(f"  类别数: {stats['category_count']}")
    print(f"  关键词数: {stats['keyword_count']}")
    print(f"  对话数: {stats['dialogue_count']}")


if __name__ == "__main__":
    main()